"""The Cube solve's per-pass A16 re-read: are those bytes on the critical path?

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.34 split the solve
stage into AIV 2.144 / assemble 0.928 / cube 1.586 ms and left two candidates
on the AIC side.  The first one is that ``kda_solve_wu_cube_kernel`` loads A16
from GM in each of its two passes: 8 KB per chunk per pass, 16 KB per chunk in
total, 100.7 MB of the kernel's own 1006.6 MB per call at [1,8192,96,128]/C=64
(80 KB per chunk).

Whether that is worth anything depends on a property nobody has measured: is
this kernel bound by GM traffic or by waves / L0 crossings / the Mmad-Fixpipe
chain?  A bandwidth probe would only answer it by inference, so this one does
not infer - it *is* the candidate:

  mode 0  the shipped load structure (A16 re-issued in both passes), control
  mode 1  the block's NC tiles stay resident in L1 (NC x 8 KB) and only the
          L1 -> L0A crossing is re-issued: the candidate
  mode 2  no A16 load at all (garbage operand, timing only): the floor, i.e.
          the second point that says whether time is linear in bytes

All three modes run ``kernels/v1/k1_solve_wu_cube_a16_probe.cpp``, a
transcription of the shipped kernel that differs in the A16 load path and
nothing else - same RHS loads, same LoadData crossings, same Mmad and Fixpipe,
same L0/L0C slots, same InitBuffer arithmetic - so the deltas are the A16 GM
traffic.  The production kernel is replayed from a captured launch in the same
process, which is what ties mode 0 to the 1.586 ms the stage measurement
attributes to it.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_cube_a16.py
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
CHUNKS = 12288          # (T / CHUNK) x H at CHUNK = 64
CH = api.CHUNK
A16_B = 2 * CH * CH * 2          # two passes of a [CH, CH] bf16 tile
RHS_B = 2 * CH * D * 2           # rk and rv, [CH, D] bf16 each
WU_B = 2 * CH * D * 2            # W and U, the same shape
MODES = [(0, "shipped  (A16 x2)", A16_B + RHS_B + WU_B),
         (1, "resident (A16 x1)", A16_B // 2 + RHS_B + WU_B),
         (2, "floor    (A16 x0)", RHS_B + WU_B)]


def main() -> None:
    api._rtc("kernels/v1/k1_solve_wu_cube_a16_probe.cpp",
             "kda_solve_wu_cube_a16_probe")
    chunks = CHUNKS
    a16 = torch.zeros(chunks, CH, CH, dtype=torch.bfloat16, device=DEV)
    rk = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    rv = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    W = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    U = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    grid = chunks // api.WU_NCHUNK
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    argv = api._pack_ptrs([a16, rk, rv, W, U])

    def launch(mode: int) -> None:
        api._launch("kda_solve_wu_cube_a16_probe", grid,
                    argv + [api._i(chunks), api._i(mode)], cur_h)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("cube solve, %d chunks, grid %d, CHUNK=%d" % (chunks, grid, CH))
    print("  bytes per chunk    A16 %5d B (x2 passes = the re-read), "
          "RHS %5d B, W/U %5d B" % (CH * CH * 2, RHS_B, WU_B))
    for mode, _, nbytes in MODES:
        launch(mode)                     # warm
    torch.npu.synchronize()

    # Interleaved: the arms are 10% apart at most, so a drifting clock would
    # otherwise show up as the signal.
    best = {mode: 1e9 for mode, _, _ in MODES}
    for _ in range(5):
        for mode, _, _ in MODES:
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    # ---- the production kernel, replayed from a captured launch -------------
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

    prod = timeit(lambda: (s.wait_stream(cur), play({"kda_solve_wu_cube_kernel"}),
                           cur.wait_stream(s)), 5)

    print()
    print("  arm                   GM bytes/call   ms (MIN of 5)   GB/s   vs shipped")
    base = None
    for mode, name, nbytes in MODES:
        ms = best[mode]
        if base is None:
            base = ms
        print("  %-20s %10.1f MB   %8.3f        %6.1f   %+8.3f"
              % (name, nbytes * chunks / 1e6, ms, nbytes * chunks / 1e6 / ms, ms - base),
              flush=True)
    print("  %-20s %10.1f MB   %8.3f        %6.1f   %+8.3f"
          % ("production cube", (A16_B + RHS_B + WU_B) * chunks / 1e6, prod,
             (A16_B + RHS_B + WU_B) * chunks / 1e6 / prod, prod - base), flush=True)

    print()
    print("verdict arithmetic")
    d1 = best[0] - best[1]
    d2 = best[1] - best[2]
    a16_mb = A16_B // 2 * chunks / 1e6
    print("  the re-read is %.1f MB of %.1f MB (%.1f%% of this kernel's traffic)"
          % (a16_mb, (A16_B + RHS_B + WU_B) * chunks / 1e6,
             100.0 * a16_mb / ((A16_B + RHS_B + WU_B) * chunks / 1e6)))
    print("  control vs candidate (mode 0 - mode 1): %+.3f ms  (residency buys this)" % d1)
    print("  candidate vs floor   (mode 1 - mode 2): %+.3f ms  (the second A16 load's price)" % d2)
    print("  linear-in-bytes check: 2 x %.3f = %.3f vs %.3f ms"
          % (d2, 2 * d2, d1))
    if d1 > 0.05:
        print("  => the A16 traffic is on the critical path; ship the resident form")
    else:
        print("  => the A16 traffic is NOT what sets this kernel's time; "
              "the resident form is not worth a kernel change")


if __name__ == "__main__":
    main()
