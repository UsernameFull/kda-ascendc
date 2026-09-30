"""What does a cross-launch byte cost the Cube solve? (plan 11.28's missing rate)

The ledger has two prices for GM traffic and never reconciled them: section
11.12 measured 0.2 ms/GB for bytes a *later block of the same launch* re-reads
(an L2 hit), while section 11.25 route C measured 2.01 GB of cross-launch
handoff costing 2.104 ms - 1.047 ms/GB, within 10% of the 1165 GB/s this device
copies at.  Section 11.28 used the cheap one: it rejected the Level 2/3 window
streaming by pricing its 0.60 GB of extra handoff at 0.12 + 0.08 ms.  At the
route-C rate the same bytes cost 0.52 + 0.34 ms, which is a different verdict,
and the same 5x ambiguity sits under every remaining byte-elimination candidate
(the paired solve cut, ndNum coalescing of the RHS loads).

So this measures the rate rather than inferring it, on the largest single block
of cross-launch reads in the pipeline: ``kda_solve_wu_cube_kernel`` pulls rk and
rv - written by earlier launches, 25x the L2 between them - at 402.65 MB per
call at [1,8192,96,128]/C=64.  ``kernels/v1/k1_solve_cube_rhs_probe.cpp`` is a
transcription of that kernel whose RHS *address* is the only variable:

  arm 0  control - the shipped address pattern, ``rhs[c0 + ch]``, cold
  arm 1  L2-hot  - ``rhs[(c0 % hot) + ch]``, same call count and shapes, so the
         whole grid reads one small window that stays resident
  arm 2  floor   - the RHS DataCopy calls deleted, queue protocol kept
  arm 3  half    - pass 0 cold, pass 1 hot (the linearity point)

  (0 - 1) / 0.40265 GB  is the marginal price of a cross-launch byte here.
  (1 - 2)               is the price of the 16 calls per block with no bytes.
  (0 - 3) vs (3 - 1)    says whether that price is per-byte or per-call.

Arms 1/2/3 compute garbage on purpose, so there is no bit-identity gate; arm 0
is tied to reality by replaying the shipped kernel's captured launches in the
same process and printing it in the same table.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_cube_rhs.py
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
CHUNKS = 12288                 # (T / CHUNK) x H at CHUNK = 64
CH = api.CHUNK
# The two prices the ledger carries, and the copy rate this device measures at.
L2_MS_PER_GB = 0.2             # section 11.12
HBM_GB_PER_MS = 1.165          # the roofline row (/tmp/roof.py)
ROUTEC_MS_PER_GB = 1.047       # section 11.25 route C: 2.01 GB -> 2.104 ms


def main() -> None:
    api._rtc("kernels/v1/k1_solve_cube_rhs_probe.cpp", "kda_solve_cube_rhs_probe")
    chunks = CHUNKS
    grid = (chunks + api.WU_NCHUNK - 1) // api.WU_NCHUNK
    rhs_gb = 2 * chunks * CH * D * 2 / 1e9       # rk + rv, each read once
    torch.manual_seed(20260928)
    # Not zeros: an all-same payload would let the DRAM access pattern flatter
    # the hot arm, and the point of arm 1 is that only the address changes.
    a16 = torch.randn(chunks, CH, CH, device=DEV).to(torch.bfloat16)
    rk = torch.randn(chunks, CH, D, device=DEV).to(torch.bfloat16)
    rv = torch.randn(chunks, CH, D, device=DEV).to(torch.bfloat16)
    W = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    U = torch.zeros(chunks, CH, D, dtype=torch.bfloat16, device=DEV)
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    argv = api._pack_ptrs([a16, rk, rv, W, U])

    def launch(mode: int, hot: int) -> None:
        api._launch("kda_solve_cube_rhs_probe", grid,
                    argv + [api._i(chunks), api._i(mode), api._i(hot)], cur_h)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    # (mode, hot, label, cold GB per call).  hot is swept on arm 1 so the price
    # can be shown to be a property of the bytes and not of the window size.
    arms = [(0, 32, "control: RHS cold (shipped)", rhs_gb),
            (1, 32, "RHS hot, window 32 chunks", 0.0),
            (1, 128, "RHS hot, window 128 chunks", 0.0),
            (1, 512, "RHS hot, window 512 chunks", 0.0),
            (3, 32, "half cold (pass 0 only)", rhs_gb / 2),
            (2, 32, "floor: no RHS calls at all", 0.0)]

    print("cube solve RHS price probe, %d chunks, grid %d, CHUNK=%d, NC=%d"
          % (chunks, grid, CH, api.WU_NCHUNK), flush=True)
    print("  RHS read per call: %.3f GB in %d Nd2Nz calls per block"
          % (rhs_gb, 2 * api.WU_NCHUNK * (CH // 16)), flush=True)
    for mode, hot, _, _ in arms:
        launch(mode, hot)                       # warm
    torch.npu.synchronize()

    # Interleaved: the arms are within 30% of each other, so a drifting clock
    # would otherwise be the signal (the round-robin section 11.41 asks for).
    best = {(m, h): 1e9 for m, h, _, _ in arms}
    for _ in range(5):
        for mode, hot, _, _ in arms:
            best[(mode, hot)] = min(best[(mode, hot)],
                                    timeit(lambda m=mode, h=hot: launch(m, h), 1))

    # ---- the shipped kernel, replayed from a captured launch ----------------
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

    def spy(name, blocks, args, stream, _seq=seq):
        _seq.append((name, int(blocks), list(args)))
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

    ncap = sum(1 for nm, _, _ in seq if nm == "kda_solve_wu_cube_kernel")
    prod = timeit(lambda: (s.wait_stream(cur), play({"kda_solve_wu_cube_kernel"}),
                           cur.wait_stream(s)), 5)

    print()
    print("  arm                          cold GB/call   ms (MIN of 5)  ms/GB(cold)  vs control")
    base = best[(0, 32)]
    for mode, hot, label, cold in arms:
        ms = best[(mode, hot)]
        print("  %-28s %8.3f       %8.3f     %8s   %+8.3f"
              % (label, cold, ms,
                 ("%.3f" % ((ms - best[(1, 32)]) / cold)) if cold > 0 else "-",
                 ms - base), flush=True)
    print("  %-28s %8.3f       %8.3f     %8.3f   %+8.3f   (%d captured launches)"
          % ("production cube (replayed)", rhs_gb, prod,
             (prod - best[(1, 32)]) / rhs_gb, prod - base, ncap), flush=True)

    d_all = best[(0, 32)] - best[(1, 32)]
    d_half_a = best[(0, 32)] - best[(3, 32)]
    d_half_b = best[(3, 32)] - best[(1, 32)]
    d_calls = best[(1, 32)] - best[(2, 32)]
    price = d_all / rhs_gb
    print()
    print("verdict arithmetic")
    print("  the price of %.3f GB of cross-launch RHS bytes: %+.3f ms" % (rhs_gb, d_all))
    print("     => %.3f ms/GB   (%.3f GB/ms)"
          % (price, rhs_gb / d_all if d_all > 0 else float("inf")))
    print("  ledger prices, same bytes:")
    print("     11.12 L2-local  %.2f ms/GB predicts %+.3f ms" % (L2_MS_PER_GB, L2_MS_PER_GB * rhs_gb))
    print("     11.25 route C   %.2f ms/GB predicts %+.3f ms" % (ROUTEC_MS_PER_GB, ROUTEC_MS_PER_GB * rhs_gb))
    print("     HBM copy rate   %.2f ms/GB predicts %+.3f ms" % (1.0 / HBM_GB_PER_MS, rhs_gb / HBM_GB_PER_MS))
    print("  window sweep on the hot arm (bytes or window?): 32 %.3f  128 %.3f  512 %.3f ms"
          % (best[(1, 32)], best[(1, 128)], best[(1, 512)]))
    print("  linearity: cold half %+.3f ms, hot half %+.3f ms (equal => per-byte, not per-call)"
          % (d_half_a, d_half_b))
    print("  the %d RHS calls per block with no bytes behind them: %+.3f ms"
          % (2 * api.WU_NCHUNK * (CH // 16), d_calls))
    print("  control vs production replay: %+.3f ms (transcription overhead)" % (prod - base))
    if price > 0.5 * (L2_MS_PER_GB + ROUTEC_MS_PER_GB):
        print("  => cross-launch bytes are priced near the HBM copy rate.  Section 11.28's")
        print("     0.20 ms for the Level 2/3 window streaming reprices to %.2f ms, and every"
              % (price * 0.60))
        print("     byte-elimination candidate is worth %.1fx what the L2 ledger says."
              % (price / L2_MS_PER_GB))
    else:
        print("  => cross-launch bytes are priced near the L2 rate; the 11.28 verdict stands")
        print("     and byte elimination on this kernel is worth ~%.3f ms at most." % d_all)


if __name__ == "__main__":
    main()
