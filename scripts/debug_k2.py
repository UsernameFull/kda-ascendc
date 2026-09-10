# Stage-2 validation: compare the K2 output/state recurrence against a torch
# fp64 per-chunk reference driven by K1's OWN intermediates.
# Usage: ASCEND_RT_VISIBLE_DEVICES=0 python scripts/debug_k2.py

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FLA_ROOT = os.environ.get("FLA_ROOT", str(REPO_ROOT.parent / "fla"))
if os.path.isdir(FLA_ROOT):
    sys.path.insert(0, FLA_ROOT)

import torch

from fla.modules.l2norm import l2norm_fwd
from kda_bt16 import kda_bt16_debug

torch.manual_seed(42)
dev = "npu"
B, T, H, HV, D = 2, 64, 4, 4, 128
scale = 0.1
BT = 16

q = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
k = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
v = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
g = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
beta = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
A_log = torch.randn(HV, dtype=torch.float32, device=dev)
dt_bias = torch.randn(HV * D, dtype=torch.float32, device=dev) * 0.1

o_p, ht_p, dbg = kda_bt16_debug(
    q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
    g, beta,
    scale=scale,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    lower_bound=-5.0,
    A_log=A_log,
    dt_bias=dt_bias,
    state_v_first=True,
)

# torch fp64 reference of the K2 recurrence using K1's OWN intermediates
q_n, _ = l2norm_fwd(q.to(torch.bfloat16))
k_n, _ = l2norm_fwd(k.to(torch.bfloat16))

W = dbg["w"].cpu().float()
U = dbg["u"].cpu().float()
Aqk = dbg["Aqk"].cpu().float()
G = dbg["g"].cpu().float()
Q = q_n.cpu().float()
K = k_n.cpu().float()

h = torch.zeros(HV, D, D)
for b in range(B):
    for hv in range(HV):
        h[hv] = torch.zeros(D, D)
        for i_t in range(T // BT):
            sl = slice(i_t * BT, (i_t + 1) * BT)
            g_t = G[b, sl, hv]
            last_idx = min((i_t + 1) * BT, T) - 1
            v_new = U[b, sl, hv] - W[b, sl, hv] @ h[hv]
            o_ref = scale * (Q[b, sl, hv] * torch.exp2(g_t.double())).double() @ h[hv].double() + Aqk[b, sl, hv].double() @ v_new.double()
            o_ref = o_ref.float()
            g_last = G[b, last_idx, hv]
            h[hv] = h[hv] * torch.exp2(g_last.float()) + (K[b, sl, hv] * torch.exp2(g_last - g_t.float())).T @ v_new
            o_p_sl = o_p[b, sl, hv].cpu().float()
            d = (o_p_sl - o_ref).abs().max().item()
            print(f"b{b} hv{hv} t{i_t}: o maxdiff={d:.3e}")

ht_ref = h
print("ht maxdiff:", (ht_p.cpu().float() - ht_ref).abs().max().item())