"""C128 geometry prototype: can the chunk-generic kernels build and run at 128?

The 2026-09-23 order puts the layered C128 geometry first, with an explicit
decision rule: implement C128 Gram + solve, consume it with the existing K2
adapter, and compare solve time / GM traffic / descriptor counts against C64 -
**if C128's K1 has no gain, K2 must not be touched**.  This probe is the first
step of that: it does not rewrite anything, it asks whether the *existing*
chunk-generic kernels build and run at KDA_CHUNK=128 at all, what they cost,
and whether the answer is still the right answer.

Three things are measured, all in one process per build (the chunk size is a
compile-time constant, so C64 and C128 cannot be interleaved in one process;
run this tool twice and compare the two reports):

  full     the whole pipeline, end to end
  k1       the same run with every ``kda_k2_*`` launch dropped (the harness
           monkeypatches api._launch), i.e. pre_gram + solve alone
  stages   one KDA_PROFILE=1 call: pre_gram_ms / solve_ms / k2_ms

and with ``--ref`` it also runs the host fp32 reference (the BT=16 one the
chunk matrix uses - the chunked recurrence is chunk-size invariant, so it is
the right anchor for any build) at a small shape and prints out/state error.

C128 is refused by the public api on purpose (api.SUPPORTED_CHUNKS): the point
of this probe is to find out whether that refusal is still about *feasibility*
(the naive build overflowing UB/L0C) or only about *validated coverage*.

  KDA_CHUNK=128 KDA_ALLOW_UNSUPPORTED_CHUNK=1 ASCEND_RT_VISIBLE_DEVICES=1 \
      python3 -u tools/probe_chunk128_k1.py --ref
  KDA_CHUNK=64 ... python3 -u tools/probe_chunk128_k1.py        # the baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
import kda_ascendc_v1.api as api  # noqa: E402

D = 128


def inputs(b, t, h, dev, seed=20260923, gate_std=0.1):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=dtype)

    q = (randn(b, t, h, D) * 0.2).to(torch.bfloat16)
    k = (randn(b, t, h, D) * 0.2).to(torch.bfloat16)
    v = (randn(b, t, h, D) * 0.1).to(torch.bfloat16)
    g = randn(b, t, h, D) * gate_std
    beta = randn(b, t, h)
    a_log = torch.linspace(-1.0, 0.2, h, device=dev)
    bias = randn(h, D) * 0.03
    state = randn(b, h, D, D) * 0.01
    return q, k, v, g, beta, a_log, bias, state


def drop_k2():
    """Make every kda_k2_* launch a no-op (the K1-only arm)."""
    real = api._launch

    def launch(name, blocks, args, stream):
        if name.startswith("kda_k2"):
            api._LAUNCH_COUNTS[name] = api._LAUNCH_COUNTS.get(name, 0) + 1
            api._LAUNCH_BLOCKS[name] = api._LAUNCH_BLOCKS.get(name, 0) + int(blocks)
            return
        return real(name, blocks, args, stream)

    api._launch = launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shape", default="1,8192,96,128")
    ap.add_argument("--ref-shape", default="1,256,2,128")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rep", type=int, default=600)
    ap.add_argument("--ref", action="store_true", help="run the host fp32 anchor")
    ap.add_argument("--no-k1", dest="k1", action="store_false", default=True)
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    torch.npu.set_device(args.device)
    dev = torch.device("npu", args.device)
    chunk = api.CHUNK
    print("build KDA_CHUNK=%d  MAXH=%d  wide NCHUNK=%d SUBB=%d  shape=%s"
          % (chunk, api.PERSIST_MAXH, api.SOLVE_WIDE_NCHUNK, api.SOLVE_WIDE_SUBB,
             args.shape))
    print("nt = %d chunks, %d tokens/chunk, compile_config=%s"
          % (t // chunk, chunk, api.compile_config()))

    # --- correctness first: a build that runs but answers the wrong question is
    # not a C128 result (the matrix test's whole reason for existing).
    if args.ref:
        from test_torch_reference import torch_reference_kda_bt16
        rb, rt, rh, _ = (int(x) for x in args.ref_shape.split(","))
        q, k, v, g, beta, a_log, bias, init = inputs(rb, rt, rh, dev)
        kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  output_final_state=True, initial_state=init)
        try:
            out, state = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
            torch.npu.synchronize()
        except Exception as exc:      # noqa: BLE001 - a device fault is a datum
            print("reference run FAILED: %s: %s" % (type(exc).__name__, exc))
            traceback.print_exc()
            return
        ref_out, ref_state = torch_reference_kda_bt16(
            q.float().cpu(), k.float().cpu(), v.float().cpu(), g.cpu(), beta.cpu(),
            D ** -0.5, lower_bound=-1.0, A_log=a_log.cpu(), dt_bias=bias.cpu(),
            initial_state=init.cpu(), state_v_first=True)
        out_rel = float((out.float().cpu() - ref_out.float()).abs().max()
                        / (ref_out.abs().max() + 1e-12))
        st_rel = float((state.cpu() - ref_state.float()).abs().max()
                       / (ref_state.abs().max() + 1e-12))
        print("vs host fp32 reference (bt=16, chunk-size invariant): "
              "out_rel %.3e   state_rel %.3e   %s (both < 2e-2 pass)"
              % (out_rel, st_rel,
                 "PASS" if max(out_rel, st_rel) < 2e-2 else "FAIL"))

    q, k, v, g, beta, a_log, bias, init = inputs(b, t, h, dev)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              output_final_state=True, initial_state=init)

    print()
    print("warming (RTC at KDA_CHUNK=%d)..." % chunk, flush=True)
    try:
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
        full_ok = True
    except Exception as exc:          # noqa: BLE001
        print("full pipeline FAILED at C=%d: %s: %s" % (chunk, type(exc).__name__, exc))
        traceback.print_exc()
        full_ok = False

    def bench(fn, label):
        vals = []
        for i in range(args.rounds):
            med, p20, p80 = do_bench(fn, warmup=100, rep=args.rep,
                                     quantiles=[0.5, 0.2, 0.8])
            vals.append(med)
            print("  %-6s round %d  %.3f ms (p20 %.3f / p80 %.3f)"
                  % (label, i + 1, med, p20, p80), flush=True)
        vals.sort()
        return vals[len(vals) // 2]

    full_med = None
    if full_ok:
        full_med = bench(lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw),
                         "full")

    k1_med = None
    if args.k1:
        drop_k2()
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
        k1_med = bench(lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw),
                       "k1")

    os.environ["KDA_PROFILE"] = "1"
    prof = {}
    if full_ok:
        try:
            api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
            torch.npu.synchronize()
            prof = dict(api.get_last_profile())
        except Exception as exc:      # noqa: BLE001
            print("profile pass failed: %s" % exc)
    print()
    print("stage split (profiler marks include their sync; compare within a build):")
    for key in ("pre_gram_ms", "solve_ms", "k2_ms", "total_ms"):
        if key in prof:
            print("  %-12s %7.3f ms" % (key, prof[key]))

    print()
    print("summary for KDA_CHUNK=%d:" % chunk)
    print("  full e2e     %s" % ("%.3f ms" % full_med if full_med else "-"))
    print("  K1 only      %s" % ("%.3f ms" % k1_med if k1_med else "-"))
    if k1_med:
        print("  K1 per token %.6f ms  (x1000 tokens: %.3f ms)"
              % (k1_med / (b * t), 1000.0 * k1_med / (b * t)))


if __name__ == "__main__":
    main()
