#!/usr/bin/env python3
"""KDA forward latency at one (B, T, H, D) shape: ours vs FLA, on one NPU.

Measured in a single process on a single device:

  fla-bench  FLA ``chunk_kda`` exactly as FLA's own harness calls it
             (``benchmarks/ops/run.py`` -> ``benchmarks/ops/registry.py``:
             ``g`` = logsigmoid clamp_min(-5), beta = sigmoid, no A_log /
             dt_bias, ``use_qk_l2norm_in_kernel=True``, ``safe_gate=True``,
             ``lower_bound=-5``).  This is the config behind FLA's published
             H100/H200 CI numbers.
  fla-alog   the same op with the A_log + dt_bias gate parameterisation our
             kernels implement (raw ``g``, raw ``beta``, both read in-kernel).
  proto      our two-stage Triton KDA (``kda_bt16_fwd``), A_log config.
  ascendc    our AscendC path (``kda_bt16_fwd_ascendc``), A_log config,
             ``--ascendc-modes`` (default ``persistent_loop``).

Timing follows FLA's harness: ``triton.testing.do_bench`` with
``quantiles=[0.5, 0.2, 0.8]`` and warmup/rep given in milliseconds; the inputs
are built before timing; every implementation gets untimed warm-up calls.

The production numbers are pinned in a committed golden
(``benchmarks/golden/``): ``--update-golden`` records one, ``--gate`` checks a
fresh run against it (geometry, median, p80, stage breakdown, and the numeric
distance to FLA) and exits non-zero on a regression.  ``tools/run_bench_gate.sh``
is the one-liner for the production shape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "python"))
FLA_ROOT = os.environ.get("FLA_ROOT")
if not FLA_ROOT:
    for cand in (ROOT.parent / "fla", Path("/tmp/fla-src2")):
        if cand.is_dir():
            FLA_ROOT = str(cand)
            break
if FLA_ROOT and FLA_ROOT not in sys.path:
    sys.path.insert(0, FLA_ROOT)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from triton.testing import do_bench  # noqa: E402

QUANTILES = [0.5, 0.2, 0.8]
LOWER_BOUND = -5.0


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def make_inputs(b, t, h, d, config, device, seed=1312):
    """[B, T, H, D] inputs; ``config`` is ``fla-bench`` or ``kda-model``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q = randn(b, t, h, d, dtype=torch.bfloat16)
    k = randn(b, t, h, d, dtype=torch.bfloat16)
    v = randn(b, t, h, d, dtype=torch.bfloat16)
    beta_raw = randn(b, t, h)
    if config == "fla-bench":
        g = torch.nn.functional.logsigmoid(randn(b, t, h, d)).clamp_min(LOWER_BOUND).contiguous()
        return dict(q=q, k=k, v=v, g=g, beta=beta_raw.sigmoid(),
                    A_log=None, dt_bias=None)
    # The raw gate stays raw (the A_log + dt_bias transform happens in-kernel),
    # but it must be a *model* gate: FLA's harness and every trained model feed
    # a non-positive one.  A N(0,1) gate is not one - it saturates half the
    # tokens at the -5 floor, and the C=64 solve then overflows to inf
    # (tests/test_c64_gate_overflow.py is the minimal repro of that bug; it is
    # not what the gate should be measuring).
    g = torch.nn.functional.logsigmoid(randn(b, t, h, d)).clamp_min(LOWER_BOUND)
    a_log = randn(h)
    dt_bias = randn(h, d) * 0.1
    return dict(q=q, k=k, v=v, g=g, beta=beta_raw,
                A_log=a_log, dt_bias=dt_bias)


# --------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------

def fla_call(inp, chunk_size, use_beta_sigmoid):
    from fla.ops.kda import chunk_kda

    kw = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              safe_gate=True, lower_bound=LOWER_BOUND, chunk_size=chunk_size,
              output_final_state=True, state_v_first=True)
    dt_bias = inp["dt_bias"]
    kw["dt_bias"] = None if dt_bias is None else dt_bias.reshape(-1)
    kw["A_log"] = inp["A_log"]
    if use_beta_sigmoid:
        kw["use_beta_sigmoid_in_kernel"] = True
    return chunk_kda(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], **kw)


def proto_call(inp):
    from kda_bt16 import kda_bt16_fwd

    return kda_bt16_fwd(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=LOWER_BOUND,
        A_log=inp["A_log"], dt_bias=inp["dt_bias"].reshape(-1),
        state_v_first=True, output_final_state=True,
    )


def ascendc_call(inp, mode, lower_bound=LOWER_BOUND):
    from kda_ascendc_v1.api import C16_ONLY_K2_MODES, kda_bt16_fwd_ascendc

    # The public entry point serves the chunk-generic loop; the S12-S15 modes
    # are C=16 kernels and only run under a C=16 build, so they go through the
    # experimental entry point (which refuses them at any other KDA_CHUNK).
    if mode in C16_ONLY_K2_MODES:
        from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental
        run = kda_bt16_fwd_ascendc_experimental
    else:
        run = kda_bt16_fwd_ascendc

    b, t, h, d = inp["q"].shape
    return run(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
        A_log=inp["A_log"], bias=inp["dt_bias"], lower_bound=lower_bound,
        output_final_state=True, k2_mode=mode,
    )


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def timeit(fn, warmup_ms, rep_ms):
    torch.npu.synchronize()
    med, p20, p80 = do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=QUANTILES)
    return {"median_ms": med, "p20_ms": p20, "p80_ms": p80}


def max_abs_diff(a, b):
    if a is None or b is None:
        return None
    return float((a.float() - b.float()).abs().max().cpu())


# --------------------------------------------------------------------------
# production gate
# --------------------------------------------------------------------------
#
# A timing without a geometry, a methodology and a reference is not a
# measurement, and a measurement that is not re-checked is not a gate: this
# shape went from 14.8 to 8.28 ms over the refactor, and nothing in the tree
# would have failed if a later change had put 20% of that back.  So the
# production config gets a committed golden next to its timings, and
# ``--gate`` re-measures and compares.
#
# Two properties of this host shape the design:
#
# * device 0 is shared, and the same build measures ~8.28 ms on a quiet device
#   and up to ~9.0 ms when the other tenants are busy (a +/-0.7 ms swing on the
#   3.25 ms pre_gram alone).  The golden therefore carries a *ratio* tolerance
#   against its own median and a separate absolute ceiling, and the report
#   names which one fired: a single failing number on a busy device is not
#   proof of a regression.
# * a *faster* wrong kernel is the failure mode this repo has already shipped
#   once (the C=16 kernel answering a C=64 call, 1.45 ms).  Geometry equality
#   is therefore a hard failure, and so is a numeric distance to FLA that
#   grows.

GOLDEN_DIR = ROOT / "benchmarks" / "golden"
GOLDEN_SCHEMA = 1
PRODUCTION_CEILING_MS = 8.5
PRODUCTION_SHAPE = "1,8192,96,128"
# FLA's own CI row for this shape (see docs/FLA_COMPARE_B1_T8192_H96_D128.md:
# PR #858, job NVIDIA-H100-PT2-7, chunk_kda fwd 1/8192/96/128 = 2.722 ms).
FLA_PUBLISHED_MS = {PRODUCTION_SHAPE: 2.722}
STAGE_ITERS = 4
STAGE_KEYS = ("pre_gram_ms", "solve_ms", "k2_ms", "init_ms", "kg_transpose_ms", "total_ms")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def golden_file(b, t, h, d) -> Path:
    return GOLDEN_DIR / f"fla_compare_{b}_{t}_{h}_{d}.json"


def first_ok(runs, prefix):
    for tag, row in runs.items():
        if tag.startswith(prefix) and "median_ms" in row:
            return tag
    return None


def collect_stages(fn, iters=STAGE_ITERS):
    """Per-stage ms, profiled: MIN of ``iters`` calls with ``KDA_PROFILE=1``.

    The profile syncs the device at every stage boundary, so it is not the
    number to quote as e2e - it is the number that says *which* stage moved.
    """
    from kda_ascendc_v1.api import get_last_profile

    prev = os.environ.get("KDA_PROFILE")
    os.environ["KDA_PROFILE"] = "1"
    best = None
    try:
        for _ in range(iters):
            fn()
            torch.npu.synchronize()
            prof = get_last_profile()
            if "total_ms" not in prof:
                continue
            row = {k: float(prof[k]) for k in STAGE_KEYS if k in prof}
            if best is None or row["total_ms"] < best["total_ms"]:
                best = row
    finally:
        if prev is None:
            os.environ.pop("KDA_PROFILE", None)
        else:
            os.environ["KDA_PROFILE"] = prev
    return best


def build_golden(results, subject, mode, *, median_tol, p80_tol, ceiling_ms):
    row = results["runs"][subject]
    gate = {
        "subject": subject,
        "mode": mode,
        "median_ms": row["median_ms"],
        "median_tol": median_tol,
        "p80_ms": row.get("p80_ms"),
        "p80_tol": p80_tol,
        "median_ms_max": ceiling_ms,
        "stage_tol": 1.10,
        "diff_tol": 1.5,
    }
    for src, dst in (("o_max_abs_diff_vs_fla", "o_max_abs_diff"),
                     ("state_max_abs_diff_vs_fla", "state_max_abs_diff")):
        if row.get(src) is not None:
            gate[dst] = row[src]
    shape_key = ",".join(str(x) for x in results["shape"])
    fla = {tag: {"median_ms": r["median_ms"]}
           for tag, r in sorted(results["runs"].items())
           if tag.startswith("fla-") and "median_ms" in r}
    if shape_key in FLA_PUBLISHED_MS:
        fla["published_h100_ms"] = FLA_PUBLISHED_MS[shape_key]
    return {
        "schema": GOLDEN_SCHEMA,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "shape": results["shape"],
        "device": results["device"],
        # the input construction rides along too: two different distributions
        # timed 0.3% apart (they do not drive the kernels), but a reader has to
        # know what was timed
        "inputs": {"config": "kda-model", "seed": 1312,
                   "note": "q/k/v ~ N(0,1) bf16, g = logsigmoid(N(0,1)).clamp_min(-5) fp32, "
                           "beta raw ~ N(0,1), A_log ~ N(0,1), dt_bias ~ N(0,0.1); "
                           "the values do not drive the kernels (0.3% measured), but a raw "
                           "N(0,1) gate is not a model input and breaks C=64 - see "
                           "tests/test_c64_gate_overflow.py"},
        "config": {"KDA_CHUNK": results["compile"]["KDA_CHUNK"],
                   "ascendc_mode": mode,
                   "warmup_ms": results["warmup_ms"],
                   "rep_ms": results["rep_ms"],
                   "quantiles": QUANTILES},
        "compile": results["compile"],
        "gate": gate,
        "timings": row,
        "stages": results.get("stages"),
        "fla": fla,
        "note": ("ratio vs this golden, plus the absolute ceiling; device 0 is shared, "
                 "so a failing number on a busy device is not proof of a regression"),
    }


def gate_failures(row, stages, compile_now, golden, *, device=None,
                  median_tol=None, p80_tol=None, ceiling_ms=None):
    """Every reason this run fails the golden; an empty list means it passes."""
    if not row or "median_ms" not in row:
        return [f"subject {golden.get('gate', {}).get('subject')!r} did not run: "
                f"{(row or {}).get('error', 'no row in this run')[:200]}"]
    g = golden.get("gate", {})
    notes = []
    fails = []

    for key, want in sorted((golden.get("compile") or {}).items()):
        got = (compile_now or {}).get(key)
        if got != want:
            fails.append(f"geometry {key}: golden {want} != this build {got} "
                         f"(timings from two geometries are not comparable; re-record with --update-golden)")

    med = row["median_ms"]
    tol = median_tol if median_tol is not None else g.get("median_tol", 1.05)
    if g.get("median_ms") is not None:
        limit = g["median_ms"] * tol
        if med > limit:
            fails.append(f"median {med:.3f} ms > golden {g['median_ms']:.3f} x {tol:.2f} = {limit:.3f} ms")
        elif med > g["median_ms"]:
            notes.append(f"median {med:.3f} ms is {(med / g['median_ms'] - 1) * 100:+.1f}% off golden "
                         f"{g['median_ms']:.3f} ms (inside the {tol:.2f}x tolerance)")
    ceiling = ceiling_ms if ceiling_ms is not None else g.get("median_ms_max")
    if ceiling is not None and med > ceiling:
        fails.append(f"median {med:.3f} ms > ceiling {ceiling:.3f} ms")

    if row.get("p80_ms") is not None and g.get("p80_ms") is not None:
        ptol = p80_tol if p80_tol is not None else g.get("p80_tol", 1.10)
        plimit = g["p80_ms"] * ptol
        if row["p80_ms"] > plimit:
            fails.append(f"p80 {row['p80_ms']:.3f} ms > golden {g['p80_ms']:.3f} x {ptol:.2f} = {plimit:.3f} ms")

    dtol = g.get("diff_tol", 1.5)
    for src, dst, label in (("o_max_abs_diff_vs_fla", "o_max_abs_diff", "out"),
                            ("state_max_abs_diff_vs_fla", "state_max_abs_diff", "state")):
        if g.get(dst) is None:
            continue
        if not math.isfinite(g[dst]):
            fails.append(f"the golden's own {label} distance to FLA is {g[dst]}: it was recorded on a "
                         f"run that was not finite; re-record with --update-golden")
            continue
        if row.get(src) is None:
            notes.append(f"numeric check vs FLA skipped: this run has no {label} reference")
            continue
        if not math.isfinite(row[src]):
            # a NaN distance is also what a NaN output looks like, and every
            # comparison against NaN is False - this has to be a hard failure
            fails.append(f"{label} max-abs vs FLA is {row[src]}: the output is not finite")
            continue
        limit = max(g[dst] * dtol, g[dst] + 1e-4)
        if row[src] > limit:
            fails.append(f"{label} max-abs vs FLA {row[src]:.3e} > golden {g[dst]:.3e} x {dtol:.2f} = {limit:.3e}")

    if stages and g.get("stage_tol") and golden.get("stages"):
        stol = g["stage_tol"]
        for key in STAGE_KEYS:
            if key == "total_ms" or key not in stages or key not in golden["stages"]:
                continue
            limit = golden["stages"][key] * stol
            if stages[key] > limit:
                fails.append(f"{key} {stages[key]:.3f} > golden {golden['stages'][key]:.3f} x {stol:.2f} = {limit:.3f}")
            elif stages[key] > golden["stages"][key]:
                notes.append(f"{key} {stages[key]:.3f} vs golden {golden['stages'][key]:.3f} "
                             f"({(stages[key] / golden['stages'][key] - 1) * 100:+.1f}%)")

    if device is not None and golden.get("device") and device != golden["device"]:
        notes.append(f"device {device} != golden's {golden['device']} (timings are device-specific)")
    return fails + [f"note: {n}" for n in notes]


def print_gate_report(golden, row, stages, failures, device, ceiling_ms=None):
    g = golden.get("gate", {})
    row = row or {}
    hard = [f for f in failures if not f.startswith("note: ")]
    notes = [f[len("note: "):] for f in failures if f.startswith("note: ")]
    print(f"gate {golden['shape']} C={golden['compile'].get('KDA_CHUNK')} "
          f"{g.get('subject')} vs golden @{golden.get('git_commit')} ({golden.get('recorded_utc')}):", flush=True)
    if row.get("median_ms") is not None:
        if g.get("median_ms") is not None:
            limit = g["median_ms"] * g.get("median_tol", 1.05)
            ceiling = ceiling_ms if ceiling_ms is not None else g.get("median_ms_max")
            print(f"  median  {row['median_ms']:8.3f} ms   golden {g['median_ms']:.3f} x "
                  f"{g.get('median_tol')} = {limit:.3f}, ceiling {ceiling}", flush=True)
        else:
            print(f"  median  {row['median_ms']:8.3f} ms   (golden has no median)", flush=True)
        if row.get("p80_ms") is not None and g.get("p80_ms") is not None:
            print(f"  p80     {row['p80_ms']:8.3f} ms   golden {g['p80_ms']:.3f} x {g.get('p80_tol')} = "
                  f"{g['p80_ms'] * g.get('p80_tol', 1.10):.3f}", flush=True)
    if stages and golden.get("stages"):
        print("  stages  " + "  ".join(f"{k[:-3]} {stages.get(k, float('nan')):.3f}/{golden['stages'].get(k, float('nan')):.3f}"
                                       for k in ("pre_gram_ms", "solve_ms", "k2_ms")), flush=True)
    for src, dst, label in (("o_max_abs_diff_vs_fla", "o_max_abs_diff", "out"),
                            ("state_max_abs_diff_vs_fla", "state_max_abs_diff", "state")):
        if g.get(dst) is not None and row.get(src) is not None:
            print(f"  {label:<6}  {row[src]:.3e} <= golden {g[dst]:.3e} x {g.get('diff_tol')} = "
                  f"{max(g[dst] * g.get('diff_tol', 1.5), g[dst] + 1e-4):.3e}", flush=True)
    for tag, fla_row in sorted((golden.get("fla") or {}).items()):
        if not isinstance(fla_row, dict) or row.get("median_ms") is None:
            continue
        print(f"  vs FLA  {tag} {fla_row['median_ms']:.3f} ms -> {fla_row['median_ms'] / row['median_ms']:.2f}x "
              f"faster than ours", flush=True)
    pub = (golden.get("fla") or {}).get("published_h100_ms")
    if pub and row.get("median_ms") is not None:
        print(f"  vs H100 published {pub:.3f} ms -> {row['median_ms'] / pub:.2f}x slower", flush=True)
    for line in notes:
        print(f"  note: {line}", flush=True)
    for line in hard:
        print(f"  FAIL: {line}", flush=True)
    print(f"gate: {'FAIL' if hard else 'PASS'}", flush=True)
    return not hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,8192,96,128", help="B,T,H,D")
    ap.add_argument("--impl", default="fla-bench,fla-alog,proto,ascendc",
                    help="comma list of fla-bench,fla-alog,proto,ascendc")
    ap.add_argument("--ascendc-modes", default="persistent_loop")
    ap.add_argument("--chunk-sizes", default="64,32")
    ap.add_argument("--warmup-ms", type=int, default=int(os.environ.get("FLA_BENCH_WARMUP_MS", "100")))
    ap.add_argument("--rep-ms", type=int, default=int(os.environ.get("FLA_BENCH_REP_MS", "1000")))
    ap.add_argument("--device", default=0, type=int)
    ap.add_argument("--kda-chunk", type=int, default=None,
                    help="build the AscendC kernels at this KDA_CHUNK (required to record a golden; "
                         "in --gate it defaults to the golden's)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--check", action="store_true", help="cross-check our outputs against FLA chunk64")
    ap.add_argument("--golden", default=None,
                    help=f"golden path (default benchmarks/golden/fla_compare_<shape>.json)")
    ap.add_argument("--update-golden", action="store_true",
                    help="record this run as the golden (production config; writes into the repo)")
    ap.add_argument("--gate", action="store_true",
                    help="check this run against the golden and exit non-zero on a regression")
    ap.add_argument("--gate-median-tol", type=float, default=1.05,
                    help="median tolerance vs the golden (default 1.05)")
    ap.add_argument("--gate-p80-tol", type=float, default=1.10)
    ap.add_argument("--gate-ceiling-ms", type=float, default=None,
                    help="absolute median ceiling (default: the golden's, recorded as 8.5 ms)")
    ap.add_argument("--stages", dest="stages", action="store_true", default=None,
                    help="record the per-stage breakdown (default: on in --gate/--update-golden)")
    ap.add_argument("--no-stages", dest="stages", action="store_false")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    impls = [s.strip() for s in args.impl.split(",") if s.strip()]
    modes = [s.strip() for s in args.ascendc_modes.split(",") if s.strip()]
    chunks = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]
    want_stages = args.stages if args.stages is not None else (args.gate or args.update_golden)

    # KDA_CHUNK is a compile-time constant of the RTC kernels and every
    # geometry gets its own timings - the 8.28 ms baseline is the C=64 build,
    # and the same call at C=16 is 11.5 ms because K2 runs four times as many
    # steps.  A recording that silently took the build default would pin the
    # wrong number under the production name, so it has to be explicit.
    if args.kda_chunk is not None:
        os.environ["KDA_CHUNK"] = str(args.kda_chunk)
    elif args.update_golden and not os.environ.get("KDA_CHUNK"):
        print("update-golden: REFUSED, pass --kda-chunk (or set KDA_CHUNK): the geometry is part "
              "of the golden and the build default is not a production setting", flush=True)
        sys.exit(1)

    # The golden fixes the geometry: read it before anything imports the api
    # (KDA_CHUNK is read at import time) and refuse a mismatched build before
    # spending minutes of device time on a run that cannot be compared.
    golden_path = Path(args.golden) if args.golden else golden_file(b, t, h, d)
    golden = None
    if (args.gate or args.update_golden) and golden_path.exists():
        golden = json.loads(golden_path.read_text())
        gold_chunk = (golden.get("compile") or {}).get("KDA_CHUNK")
        if gold_chunk is not None and not os.environ.get("KDA_CHUNK"):
            os.environ["KDA_CHUNK"] = str(gold_chunk)
            print(f"KDA_CHUNK={gold_chunk} taken from {golden_path.name}", flush=True)
        if args.gate:
            from kda_ascendc_v1.api import compile_config
            now = compile_config()
            bad = [f"{k}: golden {v} != this build {now.get(k)}"
                   for k, v in sorted((golden.get("compile") or {}).items()) if now.get(k) != v]
            if bad:
                print(f"gate: FAIL (geometry, checked before the run) vs golden @{golden.get('git_commit')}:",
                      flush=True)
                for line in bad:
                    print(f"  FAIL: {line}", flush=True)
                sys.exit(1)
    elif args.gate and not args.update_golden:
        print(f"gate: FAIL no golden at {golden_path} (record one with --update-golden)", flush=True)
        sys.exit(1)

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    torch.empty(1, device=torch.device("npu", args.device))
    torch.npu.synchronize()
    print(f"shape B={b} T={t} H={h} D={d}  ({torch.npu.get_device_name(args.device)})", flush=True)

    bench_inputs = make_inputs(b, t, h, d, "fla-bench", "npu")
    model_inputs = make_inputs(b, t, h, d, "kda-model", "npu")
    torch.npu.synchronize()

    results = {"shape": [b, t, h, d], "device": torch.npu.get_device_name(args.device),
               "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms, "runs": {}}

    def record(tag, fn, ref=None, extra=None):
        t0 = time.time()
        out = fn()
        torch.npu.synchronize()
        warm_s = time.time() - t0
        o, ht = (out[0], out[1] if len(out) > 1 else None)
        stat = timeit(fn, args.warmup_ms, args.rep_ms)
        stat["first_call_s"] = round(warm_s, 2)
        if ref is not None:
            stat["o_max_abs_diff_vs_fla"] = max_abs_diff(o, ref[0])
            stat["state_max_abs_diff_vs_fla"] = max_abs_diff(ht, ref[1])
        if extra:
            stat.update(extra)
        results["runs"][tag] = stat
        print(f"  {tag:<28} {stat['median_ms']:9.3f} ms  "
              f"(p20 {stat['p20_ms']:.3f} / p80 {stat['p80_ms']:.3f})  "
              f"first-call {stat['first_call_s']:.1f}s"
              + (f"  o-diff {stat['o_max_abs_diff_vs_fla']:.2e}"
                 if 'o_max_abs_diff_vs_fla' in stat else ""), flush=True)
        return out

    refs = {}
    if "fla-bench" in impls:
        print("fla-bench (FLA harness config, matches the published H100/H200 numbers):", flush=True)
        for cs in chunks:
            refs[f"fla-bench/{cs}"] = record(f"fla-bench chunk{cs}", lambda cs=cs: fla_call(bench_inputs, cs, False))

    if "fla-alog" in impls:
        print("fla-alog (A_log + dt_bias, same math as our kernels):", flush=True)
        for cs in chunks:
            out = record(f"fla-alog chunk{cs}", lambda cs=cs: fla_call(model_inputs, cs, True))
            refs.setdefault(f"fla-alog/{cs}", out)

    if "proto" in impls:
        print("proto (our two-stage Triton, A_log config):", flush=True)
        ref = refs.get(f"fla-alog/{chunks[0]}")
        out = record("proto kda_bt16_fwd", lambda: proto_call(model_inputs), ref=ref)

    # The gate's numeric arm needs a reference even when FLA is not on the
    # timing list, so the reference call is made on demand (untimed).
    ref_tag = f"fla-alog/{chunks[0]}"
    if (args.gate or args.update_golden) and refs.get(ref_tag) is None:
        print(f"{ref_tag} (reference only, untimed):", flush=True)
        try:
            refs[ref_tag] = fla_call(model_inputs, chunks[0], True)
        except Exception as exc:
            print(f"  reference unavailable ({str(exc)[:120]}): the gate runs without its numeric arm",
                  flush=True)

    if "ascendc" in impls:
        print("ascendc (our AscendC path, A_log config):", flush=True)
        from kda_ascendc_v1.api import get_last_profile

        for mode in modes:
            try:
                out = record(f"ascendc {mode}", lambda mode=mode: ascendc_call(model_inputs, mode),
                             ref=refs.get(ref_tag))
                prof = get_last_profile()
                row = results["runs"][f"ascendc {mode}"]
                row["launch_counts"] = prof.get("launch_counts")
                row["launch_total"] = prof.get("launch_total")
                row["compile"] = prof.get("compile")
                results.setdefault("compile", prof.get("compile"))
            except Exception as exc:  # keep going: a broken mode must not kill the run
                print(f"  ascendc {mode:<18} FAILED: {str(exc)[:160]}", flush=True)
                results["runs"][f"ascendc {mode}"] = {"error": str(exc)[:400]}

        subject = first_ok(results["runs"], "ascendc ")
        if want_stages and subject is not None:
            mode = subject[len("ascendc "):]
            stages = collect_stages(lambda: ascendc_call(model_inputs, mode))
            if stages:
                results["stages"] = stages
                print("  stages (MIN of %d, KDA_PROFILE=1): %s" % (
                    STAGE_ITERS, "  ".join(f"{k} {stages[k]:.3f}" for k in
                                           ("pre_gram_ms", "solve_ms", "k2_ms", "total_ms") if k in stages)),
                      flush=True)

    if results.get("compile") is None:
        # a run without the AscendC path still deserves a self-describing json
        try:
            from kda_ascendc_v1.api import compile_config
            results["compile"] = compile_config()
        except Exception:
            pass

    out_path = args.json or str(ROOT / "results" / "bench" / f"fla_compare_{b}_{t}_{h}_{d}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"wrote {out_path}", flush=True)

    if not (args.gate or args.update_golden):
        return

    subject = golden.get("gate", {}).get("subject") if golden else None
    if args.update_golden and not args.gate:
        subject = first_ok(results["runs"], "ascendc ") or subject
    if subject is None:
        subject = first_ok(results["runs"], "ascendc ") or "ascendc " + modes[0]
    row = results["runs"].get(subject)
    stages = results.get("stages")

    if args.update_golden:
        if row is None or row.get("median_ms") is None:
            print(f"update-golden: REFUSED, {subject!r} did not produce a timing", flush=True)
            sys.exit(1)
        if results.get("stages") is None and want_stages:
            print("update-golden: REFUSED, the stage breakdown is missing", flush=True)
            sys.exit(1)
        if golden is not None and (golden.get("compile") or {}) != (results.get("compile") or {}):
            print("update-golden: WARNING the geometry changed vs the golden being replaced: "
                  f"{golden.get('compile')} -> {results.get('compile')}", flush=True)
        ceiling = args.gate_ceiling_ms if args.gate_ceiling_ms is not None else PRODUCTION_CEILING_MS
        golden = build_golden(results, subject, mode=subject[len("ascendc "):],
                              median_tol=args.gate_median_tol, p80_tol=args.gate_p80_tol,
                              ceiling_ms=ceiling)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with open(golden_path, "w") as fh:
            json.dump(golden, fh, indent=2, sort_keys=True)
        print(f"wrote golden {golden_path} (median {golden['gate']['median_ms']:.3f} ms, "
              f"C={golden['config']['KDA_CHUNK']}, ceilings {golden['gate']['median_ms_max']} ms)", flush=True)
        if not args.gate:
            return

    failures = gate_failures(row, stages, results.get("compile"), golden,
                            device=results.get("device"),
                            median_tol=args.gate_median_tol, p80_tol=args.gate_p80_tol,
                            ceiling_ms=args.gate_ceiling_ms)
    ok = print_gate_report(golden, row, stages, failures, results.get("device"),
                           ceiling_ms=args.gate_ceiling_ms)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
