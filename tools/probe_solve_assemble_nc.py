"""The assemble's block size (KDA_ASM_NCHUNK) re-tested after 11.37.

`docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md` section 11.34 measured
``KDA_ASM_NCHUNK = 6`` and ``8`` hanging the block on the first call (AICore
spinning at 100%) and pinned the assemble's NC at 4.  Section 11.37 then
replaced the shipped per-chunk queue with an explicit B1 buffer for the
production load path and noted that the depth constraint *belongs to the
queue*: "re-testing a fatter block is a follow-up, not a claim"
(ASCENDC_V1_KERNELS.md, load mode 1/2).

This probe is that follow-up, and the answer is two-sided:

* the hang does not come back.  The queue form still deadlocks at NC = 6/8
  (reproduced here as the negative control), while the production path (load
  mode 2, explicit buffer) runs every arm to 16 and is bit-identical to NC = 4;
* and it buys nothing.  e2e is flat inside the noise floor - NC = 4 / 6 / 8
  measured 10.389 / 10.390 / 10.392 ms, median of 3, in one process - so the
  kernel is not wave-limited at NC = 4 once the batched loads and P-on-chip
  are in, and NC stays 4.

NC is a compile-time define, so the arms are the same kernel source compiled at
NC and registered under their own function names (the source is renamed, the
defines prefix is ``api._defines()`` with that arm's NC), which is what lets
one process interleave them.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble_nc.py
  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble_nc.py --e2e

Sections, in the order the argument goes:

1. budgets + liveness + bit-identity per arm.  A hang in an arm is the 11.34
   failure mode and the process watchdog is the only exit; identity is A16's
   lower-left block - the only thing this kernel writes - against NC = 4,
   element for element (fatter blocks move work between blocks, never within a
   chunk).
2. the isolated replay.  **This column ranks the wrong thing and is kept only
   to show that**: the same kernel at the same NC measured 0.15 vs 0.46 ms when
   only the operand tensors' addresses changed, so an isolated number here is a
   property of the allocation, not of the kernel.  Use it for liveness and
   identity, never for a verdict.
3. ``--e2e``: the production口径, one full pipeline per arm (the arm's own NC
   through the module attribute + a recompile under the production kernel
   name), do_bench medians round-robin, then the stage profile.  This is what
   decides, and this is what says NC is not a knob.

``--load-mode 0`` reproduces the shipped queue form (expected: a hang at
NC >= 6, the negative control for the mechanism, *not* a regression of this
probe) and ``--load-mode 1`` keeps P in GM.
"""
from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

faulthandler.dump_traceback_later(1800, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402
from kda_ascendc_v1_launcher import rtc_compile  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
SRC = "kernels/v1/k1_solve_assemble.cpp"
DEFAULT_NCS = [4, 6, 8, 12, 16]


def arm_name(nc: int) -> str:
    return "kda_solve_assemble_nc%d" % nc


def compile_arm(nc: int) -> str:
    """The shipped source at NC, registered under its own kernel name.

    ``KDA_ASM_NCHUNK`` rides in front of the source like every other RTC
    compile (api._defines() is the one place that decides the prefix, so the
    arm's NC is written through the module attribute rather than by hand).
    """
    name = arm_name(nc)
    old = api.ASM_NCHUNK
    api.ASM_NCHUNK = nc
    try:
        defines = api._defines()
    finally:
        api.ASM_NCHUNK = old
    src = (ROOT / SRC).read_text(encoding="utf-8-sig")
    assert src.count("void kda_solve_assemble(") == 1, "kernel entry point moved"
    src = src.replace("void kda_solve_assemble(", "void %s(" % name)
    rtc_compile(defines + src, name, "")
    return name


def tiles_per_chunk(mode: int) -> int:
    """bf16 [M, M] tiles moved per chunk: pass 0 Lneg + Xb + (P store),
    pass 1 Xb + (P load) + A16 store."""
    return 4 if mode >= 2 else 6


def budgets(nc: int) -> str:
    pc = api.CHUNK
    m = pc // 2
    mm = m * m
    lbsz = mm * 2
    l0a = 2 * nc * mm * 2          # a8: SLOTS * MM bf16 per L0A/L0B
    l0c = 2 * nc * mm * 4          # cfall: SLOTS * MM fp32
    l1 = 5 * nc * lbsz             # qa + qb (declared always) + the 3 mode buffers
    return "NC=%2d  L0A/L0B %3d/%3d KB  L0C %3d/%3d KB  L1 %3d KB" % (
        nc, l0a // 1024, 64, l0c // 1024, 128, l1 // 1024)


def schedule(nc: int) -> list[tuple[int, int, int]]:
    """The production slice partition as (chunk_offset, chunk_count, grid).

    Mirrors _launch_solve_two_level: every solve slice is a whole ``unit``
    (lcm of the wide / assemble / cube block sizes) and the number of slices is
    ``min(SOLVE_OVERLAP, ngrp // 8)``.  The offset matters - production hands
    each launch ``a16[lo:]`` etc., so a replay that does not offset the views
    re-runs the first slice and leaves the tail untouched (the identity check
    then compares two equally wrong outputs).
    """
    c_solve = B * H * (T // api.CHUNK)
    nch = api.SOLVE_WIDE_NCHUNK // api.SOLVE_WIDE_SUBB
    unit = math.lcm(nch, nc, api.WU_NCHUNK)
    ngrp = c_solve // unit
    slices = min(api.SOLVE_OVERLAP, max(1, ngrp // 8)) if api.SOLVE_OVERLAP > 0 else 0
    if slices < 1:
        return [(0, c_solve, (c_solve + nc - 1) // nc)]
    pairs = [(i * ngrp // slices, (i + 1) * ngrp // slices) for i in range(slices)]
    return [(lo * unit, (hi - lo) * unit, ((hi - lo) * unit + nc - 1) // nc)
            for lo, hi in pairs]


def run_e2e(ncs: list[int], mode: int, reps: int) -> None:
    """One full pipeline per arm, in this process - the arm's own NC.

    This is the口径 that decides: the replay above is address-sensitive, while
    e2e/stage carry the production allocation and the slice overlap.

    ``api.ASM_NCHUNK`` is read at import, not per call, so an arm is turned by
    setting the module attribute and re-compiling ``kda_solve_assemble`` under
    its production name with that arm's defines prefix (the launch geometry
    then follows the attribute, the kernel follows the recompile).  The RTC
    compile per arm is what dominates this section's wall time.
    """
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)
    saved = api.ASM_NCHUNK

    def call(nc: int):
        api.ASM_NCHUNK = nc
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)

    print()
    print("=== e2e, one arm per NC (3 rounds round-robin, do_bench median) ===")
    ref = None
    for nc in ncs:
        api.ASM_NCHUNK = nc
        t0 = time.perf_counter()
        api._rtc(SRC, "kda_solve_assemble")
        print("   (recompiled kda_solve_assemble at NC=%d in %.0f s)"
              % (nc, time.perf_counter() - t0), flush=True)
        out, st = call(nc)
        torch.npu.synchronize()
        if ref is None:
            ref = out
            nd = 0
        else:
            nd = int((out != ref).sum())
        print("arm NC=%-2d warm done (out mean|.| %.4f, %d/%d elements differ)"
              % (nc, float(out.float().abs().mean()), nd, out.numel()), flush=True)
    rounds: dict[int, list[float]] = {nc: [] for nc in ncs}
    order = list(reversed(ncs))
    for rnd in range(3):
        for nc in (order if rnd % 2 else ncs):
            rounds[nc].append(do_bench(lambda nc=nc: call(nc), warmup=50, rep=400))
    best = {nc: sorted(v)[1] for nc, v in rounds.items()}   # median of 3
    print()
    print("=== KDA_PROFILE=1 per arm: the real stage (marks carry their sync) ===")
    os.environ["KDA_PROFILE"] = "1"
    for nc in ncs:
        call(nc)
        torch.npu.synchronize()
        prof = dict(api.get_last_profile())
        print("  NC=%-2d  e2e %7.3f ms (rounds %s)  solve %6.3f  k2 %6.3f"
              % (nc, best[nc], " ".join("%.3f" % x for x in sorted(rounds[nc])),
                 prof.get("solve_ms", -1), prof.get("k2_ms", -1)), flush=True)
    os.environ.pop("KDA_PROFILE", None)
    api.ASM_NCHUNK = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncs", default=",".join(str(x) for x in DEFAULT_NCS))
    ap.add_argument("--load-mode", type=int, default=2,
                    help="2 production / 1 batched P in GM / 0 shipped queue")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--e2e", action="store_true",
                    help="also run the production-口径 arms (one full pipeline "
                         "per NC in this process; that section decides)")
    args = ap.parse_args()
    ncs = [int(x) for x in args.ncs.split(",")]
    mode = args.load_mode
    pc, m, mm = api.CHUNK, api.CHUNK // 2, (api.CHUNK // 2) ** 2
    chunks = B * H * (T // pc)
    print("assemble NC re-test: CHUNK=%d load_mode=%d chunks=%d overlap=%d"
          % (pc, mode, chunks, api.SOLVE_OVERLAP), flush=True)
    for nc in ncs:
        print("   " + budgets(nc), flush=True)

    names = {}
    for nc in ncs:
        t0 = time.perf_counter()
        names[nc] = compile_arm(nc)
        print("compiled %s in %.0f s" % (names[nc], time.perf_counter() - t0),
              flush=True)

    torch.manual_seed(1312)
    a16 = (torch.randn(chunks, pc, pc, device=DEV) * 0.1).to(torch.bfloat16)
    xb = (torch.randn(chunks, 2, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    lneg = (torch.randn(chunks, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    pmid = torch.zeros(chunks, m, m, dtype=torch.bfloat16, device=DEV)
    stream = torch_npu.npu.current_stream().npu_stream

    def launch(nc: int, off: int, n: int, grid: int) -> None:
        args = api._pack_ptrs([a16[off:], xb[off:], lneg[off:],
                               None if mode >= 2 else pmid[off:]])
        api._launch(names[nc], grid, args + [api._i(n), api._i(mode)], stream)

    def timeit(fn, reps: int):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    ref = None
    best = {}
    for nc in ncs:
        launches = schedule(nc)
        t0 = time.perf_counter()
        a16.zero_()
        # The zero_() is a torch op and the launches go straight to
        # aclrtLaunchKernel, so the ordering is not guaranteed (api.py records
        # this) - drain before the arm's stores land.
        torch.npu.synchronize()
        for off, n, grid in launches:
            launch(nc, off, n, grid)
        torch.npu.synchronize()
        print("arm NC=%-2d live: %d launches, first pass %.1f ms"
              % (nc, len(launches), (time.perf_counter() - t0) * 1e3), flush=True)
        if ref is None:
            ref = a16.clone()
        else:
            nd = int((a16 != ref).sum())
            print("arm NC=%-2d identity: %d/%d elements differ%s"
                  % (nc, nd, a16.numel(),
                     "" if nd == 0 else "  <-- NOT bit-identical"), flush=True)
        best[nc] = timeit(
            lambda nc=nc, ls=launches: [launch(nc, o, n, g) for o, n, g in ls],
            args.reps)
        nbytes = chunks * tiles_per_chunk(mode) * mm * 2 / 1e6
        print("arm NC=%-2d asm replay %.3f ms  (%.0f MB/call, %.0f GB/s)"
              % (nc, best[nc], nbytes, nbytes / best[nc]), flush=True)

    print()
    print("  arm    calls/chunk   waves/blk   asm replay   Δ vs NC=%d" % ncs[0])
    base = best[ncs[0]]
    for nc in ncs:
        launches = schedule(nc)
        waves = sum(g for _, _, g in launches) / len(launches) / 24.0
        calls = 1.0 + 3.0 / nc           # mode 2: 1 + NC calls (pass 0), 2 (pass 1)
        print("  NC=%-2d     %6.2f       %6.2f    %7.3f ms   %+7.3f ms"
              % (nc, calls, waves, best[nc], best[nc] - base), flush=True)
    print()
    print("(bytes/call is identical across arms: NC moves work between blocks,")
    print(" never the per-chunk traffic; a flat curve means the block size is)")
    print("(not a structural knob for this kernel once the queue is gone.)")
    print()
    print("NOTE on the replay column: the isolated reading depends on where the")
    print("operands live (the same kernel on the same NC measured 0.15 vs 0.46 ms")
    print("when only the tensors' addresses changed), so the replay above ranks")
    print("something other than the kernel - keep it for liveness and identity.")
    print("The decision口径 is the full pipeline: rerun with --e2e.")

    if args.e2e:
        run_e2e(ncs, mode, args.reps)


if __name__ == "__main__":
    main()
