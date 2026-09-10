# Benchmark + precision comparison: BT=16 two-stage kernels vs FLA chunk_kda.
# Usage: ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/bench_bt16.py [--device npu|cuda] [--quick]

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FLA_ROOT = os.environ.get("FLA_ROOT", str(REPO_ROOT.parent / "fla"))
if os.path.isdir(FLA_ROOT):
    sys.path.insert(0, FLA_ROOT)

import numpy as np
import torch

from kda_bt16 import kda_bt16_fwd, kda_bt16_kernel_k1, kda_bt16_kernel_k2

LOWER_BOUND = -5.0
SCALE = 0.08838834764831845  # 128 ** -0.5


def make_inputs(B, T, H, HV, D, dev):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    v = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    g = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV * D, dtype=torch.float32, device=dev) * 0.1
    return q, k, v, g, beta, A_log, dt_bias


def bench(fn, dev, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize() if dev == "npu" else torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize() if dev == "npu" else torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.array(ts)
    return ts.min(), np.median(ts), ts.mean()


def proto_fn(q, k, v, g, beta, A_log, dt_bias):
    # Convert to bf16 for kernel (fp32 gold comparison requires fp32 input first)
    q_bf16, k_bf16, v_bf16 = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    return lambda: kda_bt16_fwd(
        q_bf16, k_bf16, v_bf16, g, beta,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=True,
        reuse_workspace=os.environ.get("KDA_REUSE_WORKSPACE", "0") == "1",
    )


def chunk_fn(q, k, v, g, beta, A_log, dt_bias, cs):
    # Convert to bf16 for fair comparison with prototype
    q_bf16, k_bf16, v_bf16 = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    return lambda: chunk_kda(
        q_bf16, k_bf16, v_bf16, g, beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        chunk_size=cs,
    )


def proto_breakdown(B, q, k, v, g, beta, A_log, dt_bias, T, H, HV, D, dev, BT=16):
    # Convert to bf16 for kernels
    q_bf16, k_bf16, v_bf16 = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    NT = -(-T // BT)
    BH = B * HV
    # Match kda_bt16_fwd: Ascend needs V-splitting to stay within the 192 KiB UB.
    BV = 64 if dev == "npu" and D == 128 else D
    NV = -(-D // BV)
    Aqk = torch.empty(B, T, HV, BT, dtype=torch.bfloat16, device=dev)
    # A_inv not allocated for production (STORE_A_INV=False)
    Akk_scratch = torch.empty(B, T, HV, BT, dtype=torch.float32, device=dev)
    w = torch.empty(B, T, HV, D, dtype=torch.bfloat16, device=dev)
    u = torch.empty(B, T, HV, D, dtype=torch.bfloat16, device=dev)
    g_cumsum = torch.empty(B, T, HV, D, dtype=torch.float32, device=dev)
    o = torch.empty(B, T, HV, D, dtype=torch.bfloat16, device=dev)

    def k1():
        kda_bt16_kernel_k1[(NT * BH,)](
            q_bf16, k_bf16, v_bf16, g, beta, A_log, dt_bias, Aqk, None, Akk_scratch, w, u, g_cumsum,
            SCALE, LOWER_BOUND, T,
            H=H, HV=HV, K=D, V=D, BT=BT, BK=D, NT=NT,
            USE_GATE=True, HAS_BIAS=True, USE_QK_L2NORM=True, USE_BETA_SIGMOID=True,
            STORE_A_INV=False,  # Production path: no intermediate write
        )

    def k2():
        kda_bt16_kernel_k2[(B * HV,)](
            q_bf16, k_bf16, w, u, g_cumsum, Aqk, o, None, None, SCALE, T,
            H=H, HV=HV, K=D, V=D, BT=BT, BV=BV, NT=NT, NV=NV,
            USE_INITIAL_STATE=False, STORE_FINAL_STATE=False, STATE_V_FIRST=True,
            USE_QK_L2NORM=True, NUM_STAGES=1,
        )

    return k1, k2


def err(actual, gold):
    """Compute max absolute error and relative error.
    
    Args:
        actual: Output from implementation being tested
        gold: Ground truth reference output
    """
    d = (actual - gold).abs().max().item()
    denom = gold.abs().max().item()
    r = d / max(denom, 1e-12)
    return d, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu", choices=["npu", "cuda"])
    ap.add_argument("--quick", action="store_true", help="skip precision section")
    args = ap.parse_args()
    dev = args.device

    D = 128
    cases = [
        (2, 1024, 4, 4),
        (2, 4096, 8, 8),
        (1, 8192, 32, 32),
    ]

    print(f"{'shape':<22}{'impl':<12}{'min(us)':>10}{'med(us)':>10}")
    for B, T, H, HV in cases:
        q, k, v, g, beta, A_log, dt_bias = make_inputs(B, T, H, HV, D, dev)
        print(f"B={B} T={T} H=HV={H}")

        for name, fn, wu, reps in [
            ("proto", proto_fn(q, k, v, g, beta, A_log, dt_bias), 5, 20),
            ("chunk64", chunk_fn(q, k, v, g, beta, A_log, dt_bias, 64), 5, 20),
            ("chunk32", chunk_fn(q, k, v, g, beta, A_log, dt_bias, 32), 5, 20),
        ]:
            try:
                mn, md, mean = bench(fn, dev, warmup=wu, reps=reps)
                print(f"{'':<22}{name:<12}{mn:>10.1f}{md:>10.1f}")
            except Exception as e:
                print(f"{'':<22}{name:<12}  FAILED: {str(e)[:80]}")

        try:
            k1, k2 = proto_breakdown(B, q, k, v, g, beta, A_log, dt_bias, T, H, HV, D, dev)
            mn1, _, _ = bench(k1, dev, warmup=5, reps=20)
            mn2, _, _ = bench(k2, dev, warmup=5, reps=20)
            print(f"{'':<22}{'proto K1':<12}{mn1:>10.1f}{'':>10}")
            print(f"{'':<22}{'proto K2':<12}{mn2:>10.1f}{'':>10}")
        except Exception as e:
            print(f"breakdown FAILED: {str(e)[:80]}")

    if args.quick:
        return

    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    # precision vs fp32 recurrent gold, plus cross-diff
    print("\nprecision (o / ht max-abs-diff vs fp32 recurrent gold):")
    for B, T, H, HV in cases:
        q, k, v, g, beta, A_log, dt_bias = make_inputs(B, T, H, HV, D, dev)
        # Gold reference: fp32 input
        o_g, ht_g = fused_recurrent_kda(
            q, k, v, g, beta,
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=LOWER_BOUND,
            output_final_state=True,
            state_v_first=True,
        )
        # Prototype: bf16 input (convert after gold reference)
        q_bf16, k_bf16, v_bf16 = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        o_p, ht_p = kda_bt16_fwd(
            q_bf16, k_bf16, v_bf16, g, beta,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, lower_bound=LOWER_BOUND,
            A_log=A_log, dt_bias=dt_bias, state_v_first=True,
            output_final_state=True,
        )
        # FLA chunk implementations: also bf16 for fair precision comparison
        o_c, ht_c = chunk_kda(
            q_bf16, k_bf16, v_bf16, g, beta,
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True,
            lower_bound=LOWER_BOUND, state_v_first=True, chunk_size=64,
            output_final_state=True,
        )
        o_c2, ht_c2 = chunk_kda(
            q_bf16, k_bf16, v_bf16, g, beta,
            A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True,
            lower_bound=LOWER_BOUND, state_v_first=True, chunk_size=32,
            output_final_state=True,
        )
        for tag, a, b in [("proto", o_p.float(), o_g), ("chunk64", o_c.float(), o_g),
                          ("chunk32", o_c2.float(), o_g)]:
            d, r = err(a, b)
            print(f"B={B} T={T} H={H}: {tag:<8} o diff={d:.3e} ratio={r:.2e}")
        for tag, a, b in [("proto", ht_p, ht_g), ("chunk64", ht_c, ht_g), ("chunk32", ht_c2, ht_g)]:
            d, r = err(a, b)
            print(f"{'':>30} {tag:<8} ht diff={d:.3e} ratio={r:.2e}")
        d, r = err(o_p.float(), o_c.float())
        print(f"{'':>30} proto vs chunk64: o diff={d:.3e} ratio={r:.2e}")


if __name__ == "__main__":
    main()
