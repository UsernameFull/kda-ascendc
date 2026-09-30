"""Does pre_gram's block wall follow the AIV's vector stream, add and delete?

On-board PipeUtilization of the shipped kernel (archived under
``/data/models/Qwen3-4B/kda_msprof_20260930_pregram``) says the AIV vector
pipe is busy 647.2 of 820.9 us per subcore (78.8%), the AIC wall is 819.4 us
- the two halves of the MIX block are balanced - and the AIV's scalar unit
spends 612.3 us stalled on that vector queue, with zero MTE2/MTE3 stalls.
That reads as "AIV vector-issue bound, cut instructions and collect".  But
section 11.20 measured that *deleting* ~128 vector instructions per chunk
buys only 1.2%, which cannot both be true under a 1:1 rule.  A deletion can
hide behind an unchanged arrival (11.20's post_gram sits on the FL_DONE
wait); the two clean measurements are to *add* a known amount of vector work
and to *delete* one block at a time, in the same binary, on the same launch.

``k1_pg_addwork_probe.cpp`` is a clone of ``k1_pre_gram_mix`` with two extra
trailing int32s:

  addWork  runs that many extra NG-wide ``Muls(zz, zz, 1.0f)`` instructions
           per pass, right before the Gram block that reads ``zz``.  The
           chain is live (no dead-store elimination) and multiplying by
           exactly 1.0f is an IEEE identity, so both arm outputs stay
           bit-for-bit equal to the shipped kernel and only the vector-issue
           load changes.
  ablate   a timing-only bitmask (1 = gate cumsum, 2 = sigmoid pass loop,
           4 = both post_gram calls).  An ablated arm is wrong by
           construction - the established delete-the-work-keep-the-clock
           method - and only its launch time is read.

Every arm replays the *captured production launch* of ``kda_pre_gram_mix``
(same pointers, grid, flags), so all arms see the identical work; the replay
happens right after the capture, before anything else churns the caching
allocator.  Reported: MIN-of-N launch times, deltas against the control, the
price per added instruction, and bit-identity of the two add-work arms
against the shipped pipeline.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pre_gram_addwork.py
"""
from __future__ import annotations

import faulthandler
import os
import struct
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
# (addWork, ablate) arms; labels are printed with the table.
ARMS = [(0, 0), (32, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 8)]
LABEL = {(0, 0): "control", (32, 0): "+32 instr/pass", (0, 1): "-gate cumsum",
         (0, 2): "-sigmoid", (0, 3): "-cumsum,sigmoid", (0, 4): "-post_gram",
         (0, 8): "cumsum->blk scan"}
ROUNDS = int(os.environ.get("KDA_ADDWORK_ROUNDS", "4"))
PROBE = "kda_pg_addwork_probe"
# Passes per chunk, chunks per subcore and launch waves in this geometry, for
# the per-instruction bookkeeping only (M=64, MT=16 -> NP=4, unroll=64; 96
# blocks at ~819 us against the ~4112 us task duration -> 5 waves).
NP, CHUNKS_PER_SUBCORE, WAVES = 4, 64, 5


def capture(q, k, v, g, beta, kw):
    seq = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == "kda_pre_gram_mix":
            seq.append((int(blocks), list(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return seq[-1], out, st


def run_pipeline(q, k, v, g, beta, kw, add, abl):
    """Whole pipeline with the pre_gram launch replaced by the probe arm."""
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == "kda_pre_gram_mix":
            api.launch_argsarray_engine(PROBE, int(blocks), stream,
                                        list(args) + [api._i(add), api._i(abl)], 0)
            return
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return out, st


def n_diff(a, b):
    return int((a != b).sum().item())


def main() -> None:
    api._rtc("kernels/v1/k1_pg_addwork_probe.cpp", PROBE)

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    (grid, args), out_p, st_p = capture(q, k, v, g, beta, kw)
    last = struct.unpack("<i", args[-1])[0] if len(args[-1]) == 4 else None
    print("captured kda_pre_gram_mix: grid %d, %d args, trailing int %s"
          % (grid, len(args), last), flush=True)

    stream = torch_npu.npu.current_stream().npu_stream
    best = {}
    for r in range(ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for (aw, ab) in order:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            api.launch_argsarray_engine(PROBE, grid, stream,
                                        args + [api._i(aw), api._i(ab)], 0)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            best[(aw, ab)] = dt if (aw, ab) not in best else min(best[(aw, ab)], dt)

    base = best[(0, 0)]
    print("replay launch times (MIN of %d, ms), control = %.1f:" % (ROUNDS, base),
          flush=True)
    for arm in ARMS:
        print("  %-16s %8.1f   delta %+8.1f" % (LABEL[arm], best[arm], best[arm] - base),
              flush=True)

    ops_total = 32 * NP * CHUNKS_PER_SUBCORE * WAVES
    marg_ns = (best[(32, 0)] - base) * 1e6 / ops_total
    print("added-instruction price: %+8.1f ms over %d instructions -> %.2f ns each "
          "(%.1f cycles at 1.8 GHz)"
          % (best[(32, 0)] - base, ops_total, marg_ns, marg_ns * 1.8), flush=True)
    d1 = best[(0, 1)] - base
    d2 = best[(0, 2)] - base
    d3 = best[(0, 3)] - base
    print("additivity: cumsum %+.1f + sigmoid %+.1f = %+.1f vs both %+.1f ms"
          % (d1, d2, d1 + d2, d3), flush=True)

    out0, st0 = run_pipeline(q, k, v, g, beta, kw, 0, 0)
    out32, st32 = run_pipeline(q, k, v, g, beta, kw, 32, 0)
    print("bit-identity vs shipped: probe@0 out=%d st=%d | probe@32 out=%d st=%d"
          % (n_diff(out0, out_p), n_diff(st0, st_p),
             n_diff(out32, out_p), n_diff(st32, st_p)), flush=True)
    outS, stS = run_pipeline(q, k, v, g, beta, kw, 0, 8)
    mx_o = float((outS - out0).abs().max().item())
    mx_s = float((stS - st0).abs().max().item())
    print("cumsum->blocked-scan vs serial: out differ=%d/%d max|d|=%.3e | "
          "state differ=%d max|d|=%.3e"
          % (n_diff(outS, out0), out0.numel(), mx_o, n_diff(stS, st0), mx_s),
          flush=True)


if __name__ == "__main__":
    main()
