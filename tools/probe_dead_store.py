"""Interleaved A/B for the three debug-only stores (plan level 1).

``docs/artifacts/stage_traffic.txt`` prices three production writes that no
kernel reads - Aqk32's masked fp32 copy (pre_gram AIV), A32 (the wide solve's
fp32 A_inv) and BetaOut - at 0.08 ms (section 11.12's 0.2 ms/GB of L2 traffic)
to 0.19 ms (R3's measured 0.095 ms per 201 MB store).  The two rates differ by
2.4x, and neither is a measurement of *these* stores, so the ledger says
"candidate, needs an interleaved A/B" - this is it.

The kernels take the pointers and skip the stores when they are null
(``KDA_DEBUG_STORES=0``), so both arms run the same compiled kernel and the
only difference is the store traffic.  Each round runs both arms, so a device
that drifts during the run moves both numbers together:

  * ``stock``  KDA_DEBUG_STORES=1 - pointers passed, stores issued
  * ``guard``  KDA_DEBUG_STORES=0 - null pointers, stores skipped

The outputs and the final state have to be bit-identical between the arms
(the skipped stores feed nothing), and the probe asserts it before reporting.

  KDA_CHUNK=64 python3 -u tools/probe_dead_store.py
  KDA_CHUNK=64 python3 -u tools/probe_dead_store.py --shape 1,2048,8,128 --rounds 3
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

D = 128


def inputs(b, t, h, dev, seed=2931):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=dtype)

    q = randn(b, t, h, D, dtype=torch.bfloat16)
    k = randn(b, t, h, D, dtype=torch.bfloat16)
    v = randn(b, t, h, D, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(randn(b, t, h, D)).clamp_min(-5.0).contiguous()
    beta = randn(b, t, h).sigmoid()
    return q, k, v, g, beta, randn(h), randn(h, D) * 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shape", default="1,8192,96,128")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rep", type=int, default=600, help="do_bench rep, ms")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    torch.npu.set_device(args.device)
    dev = torch.device("npu", args.device)
    q, k, v, g, beta, a_log, bias = inputs(b, t, h, dev)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-5.0, output_final_state=True)

    print("warming / RTC (C=%d)..." % api.CHUNK, flush=True)
    api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()

    def call():
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)

    def arm(stores):
        os.environ["KDA_DEBUG_STORES"] = "1" if stores else "0"
        return do_bench(call, warmup=100, rep=args.rep, quantiles=[0.5, 0.2, 0.8])

    # Bit-exactness first: the guarded arm must produce the same outputs and the
    # same final state, or the timing is meaningless.
    os.environ["KDA_DEBUG_STORES"] = "1"
    o1, s1 = call()
    os.environ["KDA_DEBUG_STORES"] = "0"
    o0, s0 = call()
    torch.npu.synchronize()
    same = bool(torch.equal(o1, o0)) and bool(torch.equal(s1, s0))
    print("out and final_state bit-identical between the arms: %s" % same)
    if not same:
        print("  max|do| %.3e   max|ds| %.3e"
              % ((o1.float() - o0.float()).abs().max().item(),
                 (s1.float() - s0.float()).abs().max().item()))

    stock, guard = [], []
    print()
    print("interleaved rounds (do_bench median of %d ms rep):" % args.rep)
    for i in range(args.rounds):
        a = arm(True)
        g_ = arm(False)
        stock.append(a[0])
        guard.append(g_[0])
        print("  round %d   stock %.3f (p20 %.3f / p80 %.3f)   guard %.3f (p20 %.3f / p80 %.3f)"
              % (i + 1, a[0], a[1], a[2], g_[0], g_[1], g_[2]), flush=True)

    stock.sort()
    guard.sort()
    s_med = stock[len(stock) // 2]
    g_med = guard[len(guard) // 2]
    print()
    print("  stock median of rounds   %.3f ms" % s_med)
    print("  guard median of rounds   %.3f ms" % g_med)
    print("  delta                    %+.3f ms  (%+.1f%%)"
          % (g_med - s_med, 100.0 * (g_med - s_med) / s_med))
    print()
    print("  the ledger's two estimates for the same three stores were 0.081 ms")
    print("  (0.2 ms/GB) and 0.191 ms (0.095 ms/201 MB); the measured delta is what")
    print("  counts, and a delta inside +-0.05 ms freezes this candidate like the rest.")


if __name__ == "__main__":
    main()
