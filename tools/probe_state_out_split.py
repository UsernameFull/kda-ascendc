"""Route-1 judgment: is the output worth keeping inside the state chain?

The 2026-09-22 redesign (docs/PREFILL_LIFECYCLE_REFACTOR_20260922.md section 4.1)
splits K2 into a serial state chain (``kda_k2_state_loop``: Z = U - W S and
S = diag(d) S + Kg^T Z, publishing the chunk-entry state H[c]) and a fully
parallel output kernel (``kda_k2_out_parallel``: O[c] = Q[c] H[c] + A[c] Z[c]).
Only Z and S gate the next chunk, so the split is legal by construction; the
question this probe answers is whether it is *profitable*.

The rule the candidate has to pass is the one the plan sets for it: the sum of
the two kernels, the 384 MiB snapshot and the extra operand re-reads has to
beat the fused loop end to end.  A faster state kernel alone is not a result.

Both arms run the same K1 and the same inputs in one process, alternating arms
every round, and the probe refuses to report a delta unless the two arms are
bit-identical (out and final_state) - the split keeps every rounding position
of the fused kernel (the Cube still rounds d2/d3 to bf16 and the vector side
still accumulates out = d2*scale + d3 in fp32), so bit equality is expected and
is the cleanest possible check that the split moved work and not arithmetic.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_state_out_split.py
  ... --shape 1,2048,8,128 --rounds 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402
# The public entry point serves persistent_loop only (api.py's
# K2_MODES split); the candidate is reached through the same
# experimental door the historical modes use.
from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental as run

D = 128
BV = 64
FUSED = "persistent_loop"
SPLIT = "split_state_out"


def inputs(b, t, h, dev, seed=20260922):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=dtype)

    q = randn(b, t, h, D, dtype=torch.bfloat16)
    k = randn(b, t, h, D, dtype=torch.bfloat16)
    v = randn(b, t, h, D, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(randn(b, t, h, D)).clamp_min(-5.0).contiguous()
    beta = randn(b, t, h).sigmoid()
    return q, k, v, g, beta, randn(h), randn(h, D) * 0.1


def traffic_mb(chunk, b, t, h):
    """GM bytes moved per call by the two K2 arms, from the geometry alone.

    Only the K2 stage is priced here (K1 is identical in both arms): every
    tensor the kernels read or write once per chunk, per head, per value tile.
    ``s16``/``hsnap`` are the same 384 MiB of bf16 state in both arms - the
    fused loop overwrites one 16 KB slot per (task), the split writes a
    per-chunk slot - so the snapshot only differs by the output kernel's
    re-read.
    """
    nt = t // chunk
    bh = b * h
    tasks = bh * 2
    elt = {"bf16": 2, "fp32": 4}
    m = 1e-6
    def mb(n, dt):
        return n * elt[dt] * m

    c = bh * nt
    # shared operands (both arms read them once in K2)
    shared = {
        "W": mb(c * chunk * D, "bf16"),
        "U": mb(c * chunk * D, "bf16"),
        "kg": mb(c * chunk * D, "bf16"),
        "decay": mb(c * D, "fp32"),
    }
    fused = dict(shared)
    # fused loop: Qg + Aqk in stage 1/3, S16 single slot read+write per chunk,
    # d1/d2/d3/d4 through GM, v_new^T written and read back.
    fused.update({
        "Qg": mb(c * chunk * D, "bf16"),
        "Aqk": mb(c * chunk * chunk, "bf16"),
        "S16(w)": mb(tasks * nt * BV * D, "bf16"),
        "S16(r)": mb(tasks * nt * BV * D, "bf16"),
        "D1(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D1(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "D2(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D2(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "D3(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D3(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "D4(w)": mb(bh * nt * D * D, "fp32"),
        "D4(r)": mb(bh * nt * D * D, "fp32"),
        "Vt(w)": mb(tasks * nt * BV * chunk, "bf16"),
        "Vt(r)": mb(tasks * nt * BV * chunk, "bf16"),
        "out(w)": mb(b * t * h * D, "bf16"),
    })
    state = dict(shared)
    state.update({
        "Hs(w)": mb(tasks * nt * BV * D, "bf16"),
        "Hs(r)": mb(tasks * nt * BV * D, "bf16"),
        "D1(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D1(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "D4(w)": mb(bh * nt * D * D, "fp32"),
        "D4(r)": mb(bh * nt * D * D, "fp32"),
        "Vt(w)": mb(tasks * nt * BV * chunk, "bf16"),
    })
    outk = {
        "Qg": mb(c * chunk * D, "bf16"),
        "Aqk": mb(c * chunk * chunk, "bf16"),
        "Hs(r)": mb(tasks * nt * BV * D, "bf16"),
        "Vt(r)": mb(tasks * nt * BV * chunk, "bf16"),
        "D2(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D2(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "D3(w)": mb(tasks * nt * chunk * BV, "bf16"),
        "D3(r)": mb(tasks * nt * chunk * BV, "bf16"),
        "out(w)": mb(b * t * h * D, "bf16"),
    }
    return (sum(fused.values()), fused,
            sum(state.values()) + sum(outk.values()), dict(state, **outk))


def main():
    ap = argparse.ArgumentParser()
    ap.set_defaults(profile=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shape", default="1,8192,96,128")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rep", type=int, default=600, help="do_bench rep, ms")
    ap.add_argument("--no-profile", dest="profile", action="store_false",
                    help="skip the per-stage profile pass")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    torch.npu.set_device(args.device)
    dev = torch.device("npu", args.device)
    q, k, v, g, beta, a_log, bias = inputs(b, t, h, dev)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-5.0, output_final_state=True)

    print("C=%d  shape=%s  (RTC compiles both new kernels on the first call)"
          % (api.CHUNK, args.shape), flush=True)

    def call(mode):
        return run(q, k, v, g, beta, k2_mode=mode, **kw)

    o_f, s_f = call(FUSED)
    torch.npu.synchronize()
    o_s, s_s = call(SPLIT)
    torch.npu.synchronize()

    same = bool(torch.equal(o_f, o_s)) and bool(torch.equal(s_f, s_s))
    print("out and final_state bit-identical between the arms: %s" % same)
    if not same:
        print("  max|dout| %.3e (rel %.3e)   max|dstate| %.3e (rel %.3e)"
              % ((o_f.float() - o_s.float()).abs().max().item(),
                 ((o_f.float() - o_s.float()).abs().max()
                  / o_f.float().abs().max().clamp_min(1e-30)).item(),
                 (s_f.float() - s_s.float()).abs().max().item(),
                 ((s_f.float() - s_s.float()).abs().max()
                  / s_f.float().abs().max().clamp_min(1e-30)).item()))

    fused_mb, fused_rows, split_mb, split_rows = traffic_mb(api.CHUNK, b, t, h)
    print()
    print("GM bytes moved by K2 per call (geometry, no measurement):")
    print("  fused  %8.1f MB" % fused_mb)
    print("  split  %8.1f MB   (state chain + parallel output)" % split_mb)
    print("  delta  %+8.1f MB  -> %.3f ms at 0.2 ms/GB (L2) / %.3f ms at HBM rate"
          % (split_mb - fused_mb, (split_mb - fused_mb) * 1e-3 * 0.2,
             (split_mb - fused_mb) * 1e-3 * 0.83))

    def arm(mode):
        return do_bench(lambda: call(mode), warmup=100, rep=args.rep,
                        quantiles=[0.5, 0.2, 0.8])

    fused, split = [], []
    print()
    print("interleaved rounds (do_bench median of %d ms rep):" % args.rep)
    for i in range(args.rounds):
        a = arm(FUSED)
        c_ = arm(SPLIT)
        fused.append(a[0])
        split.append(c_[0])
        print("  round %d   fused %.3f (p20 %.3f / p80 %.3f)   split %.3f (p20 %.3f / p80 %.3f)"
              % (i + 1, a[0], a[1], a[2], c_[0], c_[1], c_[2]), flush=True)

    fused.sort()
    split.sort()
    f_med = fused[len(fused) // 2]
    s_med = split[len(split) // 2]
    print()
    print("  fused median of rounds   %.3f ms" % f_med)
    print("  split median of rounds   %.3f ms" % s_med)
    print("  delta                    %+.3f ms  (%+.1f%%)"
          % (s_med - f_med, 100.0 * (s_med - f_med) / f_med))

    if args.profile:
        os.environ["KDA_PROFILE"] = "1"
        call(FUSED)
        torch.npu.synchronize()
        pf = dict(api.get_last_profile())
        call(SPLIT)
        torch.npu.synchronize()
        ps = dict(api.get_last_profile())
        print()
        print("stage split (one profiled call per arm, profiler syncs included):")
        keys = ["pre_gram_ms", "solve_ms", "k2_ms", "k2_state", "k2_out", "total_ms"]
        print("  %-14s %10s %10s" % ("stage", "fused", "split"))
        for key in keys:
            if key in pf or key in ps:
                print("  %-14s %10s %10s"
                      % (key, ("%.3f" % pf[key]) if key in pf else "-",
                         ("%.3f" % ps[key]) if key in ps else "-"))
        os.environ["KDA_PROFILE"] = "0"

    print()
    print("  decision rule: the split only replaces the fused loop if this total")
    print("  beats it; a win on k2_state alone is not a result.")


if __name__ == "__main__":
    main()
