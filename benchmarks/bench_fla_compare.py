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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    g = randn(b, t, h, d)
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
    from kda_ascendc_v1.api import kda_bt16_fwd_ascendc

    b, t, h, d = inp["q"].shape
    return kda_bt16_fwd_ascendc(
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
    ap.add_argument("--json", default=None)
    ap.add_argument("--check", action="store_true", help="cross-check our outputs against FLA chunk64")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    impls = [s.strip() for s in args.impl.split(",") if s.strip()]
    modes = [s.strip() for s in args.ascendc_modes.split(",") if s.strip()]
    chunks = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]

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

    if "ascendc" in impls:
        print("ascendc (our AscendC path, A_log config):", flush=True)
        from kda_ascendc_v1.api import get_last_profile

        for mode in modes:
            try:
                out = record(f"ascendc {mode}", lambda mode=mode: ascendc_call(model_inputs, mode),
                             ref=refs.get(f"fla-alog/{chunks[0]}"))
                prof = get_last_profile()
                results["runs"][f"ascendc {mode}"]["launch_counts"] = prof.get("launch_counts")
                results["runs"][f"ascendc {mode}"]["launch_total"] = prof.get("launch_total")
            except Exception as exc:  # keep going: a broken mode must not kill the run
                print(f"  ascendc {mode:<18} FAILED: {str(exc)[:160]}", flush=True)
                results["runs"][f"ascendc {mode}"] = {"error": str(exc)[:400]}

    out_path = args.json or str(ROOT / "results" / "bench" / f"fla_compare_{b}_{t}_{h}_{d}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
