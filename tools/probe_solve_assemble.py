"""What sets kda_solve_assemble's time: bytes, the P round trip, or the L0 chain?

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.35 put the coupling
block kernel first on the AIC-side list: 151.0 MB per call in 0.928 ms is
163 GB/s where the cube solve's cold-path marginal rate is 839 GB/s, so ~0.75 ms
of it is structure.  This probe runs five arms of
``kernels/v1/k1_solve_assemble_probe.cpp`` - a transcription of the shipped
kernel whose arms remove one structural candidate at a time - interleaved in one
process, and replays the production launch next to them:

  mode 0  shipped structure (control)                     12 KB/chunk  151.0 MB
  mode 1  loads only (no LoadData, no Mmad, no Fixpipe)    8 KB/chunk  100.7 MB
  mode 2  loads + Fixpipe stores (no LoadData, no Mmad)   12 KB/chunk  151.0 MB
  mode 3  control minus the P round trip                   8 KB/chunk  100.7 MB
  mode 4  pass 0 only (half the chunk-passes)              6 KB/chunk   75.5 MB

What each delta means:

  mode 1            the GM traffic as it is actually issued: 6 DMA calls per
                    chunk-pass (2 KB, 1 KB, 1 KB / 2 KB, 1 KB, 1 KB) - if this
                    is already ~0.9 ms, the transfer pattern is the problem
  mode 2 - mode 1   the store side (Fixpipe, 2 KB per chunk-pass)
  mode 0 - mode 2   the L0 chain (LoadData x5, Mmad, two flag waits per pass)
  mode 0 - mode 3   the P round trip (2 KB store + 2 KB load per chunk), i.e.
                    what an L0C -> L1 -> L0B single-pass structure takes back
  2 x mode 4        vs mode 0: a per-block fixed cost (the two-pass barrier)
                    would make the half-work arm cost more than half

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble.py
"""
from __future__ import annotations

import faulthandler
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
CHUNKS = 12288
PC = api.CHUNK
M = PC // 2
MM = M * M
# Bytes per chunk per arm, in bf16 [M, M] tiles of TILE bytes.  The shipped
# structure is 6 tiles - pass 0: Lneg + Xb(2 bands) + P store, pass 1: Xb +
# P load + A16 store - and each arm removes whole tiles from that list.
TILE = M * M * 2
MODES = [
    (0, "shipped (control)", 6 * TILE),
    (1, "loads only", 4 * TILE),          # no P store (pass 0), no A16 store
    (2, "loads + stores", 6 * TILE),
    (3, "no P round trip", 4 * TILE),     # P store and P load both gone
    (4, "pass 0 only", 3 * TILE),         # Lneg + Xb + P store
]


def main() -> None:
    api._rtc("kernels/v1/k1_solve_assemble_probe.cpp", "kda_solve_assemble_probe")
    a16 = torch.zeros(CHUNKS, PC, PC, dtype=torch.bfloat16, device=DEV)
    xb = torch.zeros(CHUNKS, 2, M, M, dtype=torch.bfloat16, device=DEV)
    lneg = torch.zeros(CHUNKS, M, M, dtype=torch.bfloat16, device=DEV)
    pmid = torch.zeros(CHUNKS, M, M, dtype=torch.bfloat16, device=DEV)
    grid = (CHUNKS + api.ASM_NCHUNK - 1) // api.ASM_NCHUNK
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    argv = api._pack_ptrs([a16, xb, lneg, pmid])

    def launch(mode: int) -> None:
        api._launch("kda_solve_assemble_probe", grid,
                    argv + [api._i(CHUNKS), api._i(mode)], cur_h)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("assemble, %d chunks, grid %d, CHUNK=%d, ASM_NCHUNK=%d"
          % (CHUNKS, grid, PC, api.ASM_NCHUNK))
    for mode, _, _ in MODES:
        launch(mode)
    torch.npu.synchronize()
    best = {mode: 1e9 for mode, _, _ in MODES}
    for _ in range(5):
        for mode, _, _ in MODES:
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    # ---- the production kernel, replayed from its captured launches ---------
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
    bias = torch.randn(H, D, device=DEV) * 0.03
    seq = []
    orig = api._launch

    def spy(name, blocks, args, stream):
        seq.append((name, int(blocks), list(args)))
        return orig(name, blocks, args, stream)

    api._launch = spy
    print("warming the production pipeline (RTC) ...", flush=True)
    api.kda_bt16_fwd_ascendc(q, k, v, g, beta, A_log=a_log, bias=bias,
                             lower_bound=-1.0, output_final_state=True)
    torch.npu.synchronize()
    api._launch = orig
    s = torch_npu.npu.Stream(device=DEV)

    def play(names):
        for nm, blk, ar in seq:
            if nm in names:
                api.launch_argsarray_engine(nm, blk, s.npu_stream, ar, 0)

    prod = timeit(lambda: (s.wait_stream(cur), play({"kda_solve_assemble"}),
                           cur.wait_stream(s)), 5)

    print()
    print("  arm                  bytes/call   ms (MIN of 5)   GB/s")
    for mode, name, per_chunk in MODES:
        nbytes = per_chunk * CHUNKS
        print("  mode %d %-15s %7.1f MB   %8.3f        %6.1f"
              % (mode, name, nbytes / 1e6, best[mode],
                 nbytes / 1e6 / best[mode]), flush=True)
    print("  %-20s %7.1f MB   %8.3f        %6.1f"
          % ("production (24 launches)", 151.0, prod, 151.0 / prod), flush=True)

    print()
    print("verdict arithmetic")
    print("  (a) the GM pattern alone      mode 1          %.3f ms   %.1f%% of control"
          % (best[1], 100.0 * best[1] / best[0]))
    print("  (b) store side                mode 2 - mode 1  %+.3f ms" % (best[2] - best[1]))
    print("  (c) L0 chain                  mode 0 - mode 2  %+.3f ms" % (best[0] - best[2]))
    print("  (d) P round trip              mode 0 - mode 3  %+.3f ms" % (best[0] - best[3]))
    print("  (e) per-pass scaling          2 x mode 4 = %.3f vs mode 0 %.3f  (%+.3f)"
          % (2 * best[4], best[0], 2 * best[4] - best[0]))
    print("  (f) launched form             production - control %+.3f ms"
          % (prod - best[0]))


if __name__ == "__main__":
    main()
