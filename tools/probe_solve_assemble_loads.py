"""The assemble's batched load path (plan 11.37), measured in production.

The coalescing probe (kernels/v1/k1_solve_assemble_coalesce_probe.cpp) held the
bytes fixed at 100.7 MB and varied only the call count: 6 calls per chunk
(0.313 ms of traffic alone), 4 with the two B bands merged (0.252), 2.5 with the
A operand batched over the block (0.217), and 0.157 for the re-laid-out ceiling.
It also separated the two L1 structures, and the pairing that wins is the
batched calls *with* an explicit buffer rather than the shipped queue: 0.531 vs
0.606 ms for the same calls, against 0.706 ms for the shipped structure overall
- and P/A16 were bit-identical in all of them.

This probe is that candidate in the production kernel, which is what decides it:
`kda_solve_assemble` now takes a second argument (api.asm_load_mode(),
KDA_ASM_LOADS, read per call so one process can flip it) and the arms are

  mode 0  shipped: per-chunk queue, six ND2NZ calls per chunk
  mode 1  batched: explicit B1 buffers, 1 + NC calls for pass 0 and 2 for pass 1

Both are the same arithmetic and land byte-identical operands in L1, so the
probe first checks the *outputs* are identical (not just close), then measures
e2e round-robin, the isolated solve replays, and the profiled stage.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble_loads.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
MODES = [(0, "shipped calls (queue)"), (1, "batched calls (explicit B1)")]
ENV = "KDA_ASM_LOADS"


def inputs():
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
    bias = torch.randn(H, D, device=DEV) * 0.03
    return q, k, v, g, beta, a_log, bias


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-only", action="store_true",
                    help="skip the e2e sweep and the profiled section (the "
                         "replay arms carry the attribution); used when the run "
                         "is about a knob the e2e arm cannot resolve")
    ap.add_argument("--rev", action="store_true",
                    help="round-robin the e2e arms in reverse order, so a "
                         "systematic drift between the three positions shows up "
                         "as a sign flip")
    args = ap.parse_args()
    q, k, v, g, beta, a_log, bias = inputs()
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

    def call(mode: int):
        os.environ[ENV] = str(mode)
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)

    print("warming (RTC at C=%d) + identity check ..." % api.CHUNK, flush=True)
    ref = call(0)
    torch.npu.synchronize()
    again = call(0)
    torch.npu.synchronize()
    same = (torch.equal(ref[0], again[0]) and torch.equal(ref[1], again[1]))
    print("   mode 0 is deterministic: %s" % same, flush=True)
    for mode, name in MODES[1:]:
        out, st = call(mode)
        torch.npu.synchronize()
        diff = float((out.float() - ref[0].float()).abs().max())
        nd = int((out != ref[0]).sum())
        print("   mode %d (%s): %d/%d elements differ, max|d| %.3e  (states: %s)"
              % (mode, name, nd, out.numel(), diff,
                 "identical" if torch.equal(st, ref[1]) else "differ"), flush=True)

    if args.arms_only:
        print()
        print("(--arms-only: e2e and profile sections skipped)", flush=True)
        os.environ.pop(ENV, None)
        return

    print()
    print("=== full pipeline, mode round-robin, do_bench median of 3 ===")
    med = {}
    order = list(reversed(MODES)) if args.rev else list(MODES)
    for rnd in range(3):
        for mode, name in order:
            m = do_bench(lambda md=mode: call(md), warmup=50, rep=400)
            med.setdefault(mode, []).append(m)
    base = None
    for mode, name in MODES:
        vals = sorted(med[mode])
        mid = vals[len(vals) // 2]
        if base is None:
            base = mid
        print("   mode %d %-22s %.3f ms (rounds %s)  Δ %+.3f ms"
              % (mode, name, mid, " ".join("%.3f" % x for x in sorted(med[mode])),
                 mid - base), flush=True)

    # ---- isolated solve arms, per mode, from that mode's own captured args ---
    print()
    print("=== isolated solve (replay of each mode's captured launches, MIN of 5) ===")
    cur = torch_npu.npu.current_stream()
    sa = torch_npu.npu.Stream(device=DEV)
    sb = torch_npu.npu.Stream(device=DEV)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("   mode  AIV      ASM      CUBE     AIC      overlapped  sliced")
    for mode, name in MODES:
        seq = []
        orig = api._launch

        def spy(kn, blocks, args, stream, _seq=seq):
            _seq.append((kn, int(blocks), list(args)))
            return orig(kn, blocks, args, stream)

        api._launch = spy
        call(mode)
        torch.npu.synchronize()
        api._launch = orig

        def play(names, stream):
            for kn, blk, ar in seq:
                if kn in names:
                    api.launch_argsarray_engine(kn, blk, stream.npu_stream, ar, 0)

        def replay_sliced(units):
            """The production schedule, replayed: wide slice i on sa, an event,
            then that slice's assemble/cube on sb.  The event is what the plain
            "both" arm below leaves out, so this arm is the stage's real floor.
            ``units`` is the capture filtered to the solve's three kernels (the
            capture also holds pre_gram and K2, and the first launch of a call
            is never a wide one)."""
            sa.wait_stream(cur)
            sb.wait_stream(cur)
            i = 0
            while i < len(units):
                kn, blk, ar = units[i]
                if kn != "kda_solve_wu_wide":
                    i += 1
                    continue
                api.launch_argsarray_engine(kn, blk, sa.npu_stream, ar, 0)
                ev = torch_npu.npu.Event()
                ev.record(sa)
                sb.wait_event(ev)
                j = i + 1
                while j < len(units) and units[j][0] != "kda_solve_wu_wide":
                    k2, b2, a2 = units[j]
                    api.launch_argsarray_engine(k2, b2, sb.npu_stream, a2, 0)
                    j += 1
                i = j
            cur.wait_stream(sa)
            cur.wait_stream(sb)

        asm_set = {"kda_solve_assemble"}
        cube_set = {"kda_solve_wu_cube_kernel"}
        rest = asm_set | cube_set
        wide = {"kda_solve_wu_wide"}
        aiv = timeit(lambda: (sa.wait_stream(cur), play(wide, sa), cur.wait_stream(sa)))
        asm = timeit(lambda: (sb.wait_stream(cur), play(asm_set, sb), cur.wait_stream(sb)))
        cub = timeit(lambda: (sb.wait_stream(cur), play(cube_set, sb), cur.wait_stream(sb)))
        aic = timeit(lambda: (sb.wait_stream(cur), play(rest, sb), cur.wait_stream(sb)))
        both = timeit(lambda: (sa.wait_stream(cur), sb.wait_stream(cur),
                               play(wide, sa), play(rest, sb),
                               cur.wait_stream(sa), cur.wait_stream(sb)))
        units = [x for x in seq if x[0] in (wide | rest)]
        sliced = timeit(lambda: replay_sliced(units))
        print("   %-5d %7.3f  %7.3f  %7.3f  %7.3f  %7.3f   %7.3f  "
              "(launches: %d wide / %d asm / %d cube)"
              % (mode, aiv, asm, cub, aic, both, sliced,
                 sum(1 for x in seq if x[0] in wide),
                 sum(1 for x in seq if x[0] in asm_set),
                 sum(1 for x in seq if x[0] in cube_set)), flush=True)

    # ---- the real stage, profiled (the decision number is solve_ms here, not
    # the replay above: the replay has no per-slice events between the streams,
    # so its "overlapped" column is a floor rather than the stage).
    print()
    print("=== KDA_PROFILE=1 per mode, the real call (marks include their sync) ===")
    os.environ["KDA_PROFILE"] = "1"
    for mode, name in MODES:
        call(mode)
        torch.npu.synchronize()
        prof = dict(api.get_last_profile())
        print("   mode %d %-22s pre_gram %6.3f  solve %6.3f  k2 %6.3f  total %6.3f ms"
              % (mode, name, prof.get("pre_gram_ms", -1), prof.get("solve_ms", -1),
                 prof.get("k2_ms", -1), prof.get("total_ms", -1)), flush=True)
    os.environ.pop("KDA_PROFILE", None)
    os.environ.pop(ENV, None)


if __name__ == "__main__":
    main()
