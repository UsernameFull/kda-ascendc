# Stage-1 validation: compare K1 intermediates (Aqk/A_inv/w/u/g) against a torch
# fp64 per-chunk reference (BT=16) computed on CPU.
# Usage: ASCEND_RT_VISIBLE_DEVICES=0 python scripts/debug_k1.py

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
from fla.ops.utils.constant import RCP_LN2
from kda_bt16 import kda_bt16_debug

torch.manual_seed(42)
dev = "npu"
B, T, H, HV, D = 2, 64, 4, 4, 128
scale = 0.1

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

# ---- torch fp64 per-chunk reference (BT=16), computed on CPU ----
BT = 16
q_n, _ = l2norm_fwd(q.to(torch.bfloat16))
k_n, _ = l2norm_fwd(k.to(torch.bfloat16))
beta_s = torch.sigmoid(beta)

qc = q_n.cpu().double()
kc = k_n.cpu().double()
vc = v.cpu().double()
gc = g.cpu().double()
bc = beta_s.cpu().double()
Al = A_log.cpu().double()
db = dt_bias.cpu().double().view(HV, D)

g_ref = torch.zeros(B, T, HV, D, dtype=torch.float64)
gate = (-5.0 * torch.sigmoid(torch.exp(Al)[None, None, :, None] * (gc + db[None, None])) * RCP_LN2)
for i_t in range(-(-T // BT)):
    sl = slice(i_t * BT, min((i_t + 1) * BT, T))
    g_ref[:, sl] = torch.cumsum(gate[:, sl], dim=1)

print("g maxdiff:", (g_ref.float() - dbg["g"].float().cpu()).abs().max().item())

# per-chunk comparison of Aqk/A_inv/w/u
for (b, t, hv) in [(0, 0, 0), (0, 1, 1), (1, 3, 2)]:
    i_ti = t * BT
    sl = slice(i_ti, i_ti + BT)
    qn = qc[b, sl, hv]
    kn = kc[b, sl, hv]
    vn = vc[b, sl, hv]
    gn = g_ref[b, sl, hv]
    bn = bc[b, sl, hv]
    mid = min(BT // 2, T - i_ti - 1)
    gm = gn - gn[mid][None]
    gq, gk = torch.exp2(gm), torch.exp2(-gm)
    Aqk_ref = torch.tril((qn * gq) @ (kn * gk).T * scale)
    Akk_ref = torch.tril((kn * gq) @ (kn * gk).T * bn[:, None], diagonal=-1)
    A_inv_ref = torch.linalg.inv(torch.eye(BT, dtype=torch.float64) + Akk_ref)
    w_ref = A_inv_ref @ (kn * bn[:, None] * torch.exp2(gn))
    u_ref = A_inv_ref @ (vn * bn[:, None])

    for name, a, ref in [
        ("Aqk  ", dbg["Aqk"][b, sl, hv].float().cpu(), Aqk_ref),
        ("A_inv", dbg["A_inv"][b, sl, hv].float().cpu(), A_inv_ref),
        ("w    ", dbg["w"][b, sl, hv].float().cpu(), w_ref),
        ("u    ", dbg["u"][b, sl, hv].float().cpu(), u_ref),
    ]:
        d = (a.double() - ref).abs().max().item()
        print(f"b{b} t{t} hv{hv} {name}: maxdiff={d:.3e} proto_nan={torch.isnan(a).any().item()}")

# dump matrices for (1,3,2)
b, t, hv = 1, 3, 2
i_ti = t * BT
sl = slice(i_ti, i_ti + BT)
qn = qc[b, sl, hv]
kn = kc[b, sl, hv]
gn = g_ref[b, sl, hv]
bn = bc[b, sl, hv]
gm = gn - gn[min(BT // 2, T - i_ti - 1)][None]
gq, gk = torch.exp2(gm), torch.exp2(-gm)
Akk_ref = torch.tril((kn * gq) @ (kn * gk).T * bn[:, None], diagonal=-1)
A_inv_ref = torch.linalg.inv(torch.eye(BT, dtype=torch.float64) + Akk_ref)
A_inv_proto = dbg["A_inv"][b, sl, hv].float().cpu().double()
print("(1,3,2) A_inv ref   :", A_inv_ref[0, :4].tolist())
print("(1,3,2) A_inv proto :", A_inv_proto[0, :4].tolist())
print("(1,3,2) diff row0   :", (A_inv_proto - A_inv_ref)[0, :4].tolist())
print("(1,3,2) Aqk proto   :", dbg["Aqk"][b, sl, hv].float().cpu()[0, :4].tolist())
q_proto = qn * gq
k_proto = kn * gk
Aqk_ref = torch.tril(q_proto @ k_proto.T * scale)
print("(1,3,2) Aqk ref     :", Aqk_ref[0, :4].tolist())
print("(1,3,2) g row0      :", gn[0, :4].tolist())
print("(1,3,2) g proto     :", dbg["g"][b, sl, hv].float().cpu()[0, :4].tolist())

Aqk_p = dbg["Aqk"][b, sl, hv].float().cpu()
Aqk_r = Aqk_ref
print("(1,3,2) Aqk proto rows 0-3:")
print(Aqk_p[:4].numpy())
print("(1,3,2) Aqk ref   rows 0-3:")
print(Aqk_r[:4].numpy())
print("(1,3,2) A_inv proto rows 0-3:")
print(dbg["A_inv"][b, sl, hv].float().cpu()[:4].numpy())