"""Baseline timing without the FLA arm: do_bench + per-stage, one process.

``bench_fla_compare.py`` is the production harness, but its FLA arm runs
``fla.ops.kda.chunk_kda``, which autotunes Triton on this machine and has been
observed to leave the device in an aicore-timeout state (507014) partway
through the same process - after which the *AscendC* arm reports "did not run"
and the gate fails for a reason that has nothing to do with the code under
test.  For optimization rounds the FLA reference is not needed: the numeric
side is covered by the pytest matrix against the fp32 reference, and the timing
side only needs a reproducible AscendC number.

This script therefore measures exactly what the harness would measure for the
AscendC subject, in the same order and with the same conventions:

  * ``triton.testing.do_bench`` (warmup/rep in ms, quantiles 0.5/0.2/0.8), and
  * the per-stage profile (``KDA_PROFILE=1``, MIN of ``STAGE_ITERS``), which is
    the attribution number, not the e2e one.

It also reports the host-side attribution (alloc/pack/args/launch) that
``tools/probe_host_cost.py`` prices, so a single run gives both the e2e number
and the breakdown a candidate has to move.

  KDA_CHUNK=64 python3 -u tools/bench_baseline.py --tag stock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

D = 128
LOWER_BOUND = -5.0
QUANTILES = [0.5, 0.2, 0.8]
STAGE_ITERS = 4
STAGE_KEYS = ("pre_gram_ms", "solve_ms", "k2_ms", "init_ms",
              "kg_transpose_ms", "total_ms")


def make_inputs(b, t, h, d, device, seed=1312):
    """The harness's ``kda-model`` config, verbatim (bench_fla_compare.py)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q = randn(b, t, h, d, dtype=torch.bfloat16)
    k = randn(b, t, h, d, dtype=torch.bfloat16)
    v = randn(b, t, h, d, dtype=torch.bfloat16)
    beta_raw = randn(b, t, h)
    g = torch.nn.functional.logsigmoid(randn(b, t, h, d)).clamp_min(LOWER_BOUND).contiguous()
    a_log = randn(h)
    dt_bias = randn(h, d) * 0.1
    return dict(q=q, k=k, v=v, g=g, beta=beta_raw.sigmoid(), A_log=a_log, bias=dt_bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stock")
    ap.add_argument("--shape", default="1,8192,96,128")
    ap.add_argument("--device", default=0, type=int)
    ap.add_argument("--warmup-ms", type=int, default=int(os.environ.get("FLA_BENCH_WARMUP_MS", "100")))
    ap.add_argument("--rep-ms", type=int, default=int(os.environ.get("FLA_BENCH_REP_MS", "1000")))
    ap.add_argument("--json", default=None)
    ap.add_argument("--host-attribution", action="store_true")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    torch.empty(1, device=torch.device("npu", args.device))
    torch.npu.synchronize()

    from kda_ascendc_v1.api import (compile_config, get_last_profile,
                                    kda_bt16_fwd_ascendc)
    m = make_inputs(b, t, h, d, torch.device("npu", args.device))
    kw = dict(A_log=m["A_log"], bias=m["bias"], lower_bound=LOWER_BOUND,
              output_final_state=True)

    print("shape B=%d T=%d H=%d D=%d  (%s)  C=%d  tag=%s"
          % (b, t, h, d, torch.npu.get_device_name(args.device), compile_config()["KDA_CHUNK"], args.tag),
          flush=True)

    t0 = time.time()
    out = kda_bt16_fwd_ascendc(m["q"], m["k"], m["v"], m["g"], m["beta"], **kw)
    torch.npu.synchronize()
    print("  first call (incl. RTC) %.1fs" % (time.time() - t0), flush=True)
    finite = all(torch.isfinite(x).all().item() for x in out if x is not None)
    print("  finite: %s" % finite, flush=True)

    stat = do_bench(lambda: kda_bt16_fwd_ascendc(m["q"], m["k"], m["v"], m["g"], m["beta"], **kw),
                    warmup=args.warmup_ms, rep=args.rep_ms, quantiles=QUANTILES)
    print("  e2e do_bench  median %8.3f  p20 %8.3f  p80 %8.3f  ms"
          % (stat[0], stat[1], stat[2]), flush=True)

    prof_prev = os.environ.get("KDA_PROFILE")
    os.environ["KDA_PROFILE"] = "1"
    stages = None
    try:
        for _ in range(STAGE_ITERS):
            kda_bt16_fwd_ascendc(m["q"], m["k"], m["v"], m["g"], m["beta"], **kw)
            torch.npu.synchronize()
            prof = get_last_profile()
            if "total_ms" not in prof:
                continue
            row = {k: float(prof[k]) for k in STAGE_KEYS if k in prof}
            if stages is None or row["total_ms"] < stages["total_ms"]:
                stages = row
                launch_counts = prof.get("launch_counts")
    finally:
        if prof_prev is None:
            os.environ.pop("KDA_PROFILE", None)
        else:
            os.environ["KDA_PROFILE"] = prof_prev
    if stages:
        print("  stages (MIN of %d, KDA_PROFILE=1): %s" % (
            STAGE_ITERS, "  ".join("%s %.3f" % (k, stages[k]) for k in
                                   ("pre_gram_ms", "solve_ms", "k2_ms", "total_ms")
                                   if k in stages)), flush=True)

    result = {"tag": args.tag, "shape": [b, t, h, d], "device": torch.npu.get_device_name(args.device),
              "median_ms": stat[0], "p20_ms": stat[1], "p80_ms": stat[2],
              "stages": stages, "compile": compile_config(),
              "launch_counts": launch_counts, "finite": finite}

    if args.host_attribution:
        reps = int(os.environ.get("KDA_HOST_REPS", "12"))

        def host_min(fn):
            best = 1e9
            for _ in range(reps):
                t1 = time.perf_counter()
                fn()
                best = min(best, (time.perf_counter() - t1) * 1e3)
            return best

        e2e_host = host_min(lambda: kda_bt16_fwd_ascendc(m["q"], m["k"], m["v"], m["g"], m["beta"], **kw))
        result["host_wall_ms"] = e2e_host
        print("  host-side wall (hot, no sync in window) %8.4f ms" % e2e_host, flush=True)

    out_path = args.json or str(ROOT / "results" / "bench" / ("baseline_%s.json" % args.tag))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("  wrote %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
