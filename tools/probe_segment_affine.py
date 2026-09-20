"""Segment-scan (route A) step 1: prove the affine form on the host.  CPU only.

The whole route rests on the chunk being an affine map of the incoming state,

    S_out = M @ S_in + Q

with, in this repo's [V, K] (state_v_first) convention,

    M = diag(d) - w^T @ kg        Q = u^T @ kg        d = exp2(g_last)

(w, u, kg are the chunk's K1 outputs for that chunk, as in
kernels/v1/k2_persistent_loop.cpp).  This probe verifies, in fp32 against the
serial recurrence of tests/test_torch_reference.py:

  1. one chunk is affine, for both the state and the output
  2. a segment composes: M_seg = M_1 M_2 ... M_s, Q_seg = sum_j Q_j M_{j+1..s}
  3. the h = I trick: carrying h = I through the chunk recurrence and
     subtracting the h = 0 carry returns M_seg exactly, so pass 1 can build
     M_seg / Q_seg with the *existing* chunk math and no explicit product

It also shows that the naive pass-3 form (o_i = Aqk u + R_i Q_i^T evaluated
against the *segment* input state) is wrong by O(1), and what the corrected
modifiers are:

    o_i = [Aqk_i u_i + R_i Q_{j->i-1}^T] + [R_i M_{j->i-1}^T] S_in[j]^T

with Q_{j->i-1} = S0 (h=0 carry at chunk i) and M_{j->i-1} = S1 - S0, both
produced by pass 1 for free.  Run:

    python3 tools/probe_segment_affine.py
"""
import torch

D, BT, LB, RCP_LN2, EPS = 128, 16, -1.0, 1.4426950216, 1e-6
H, B, NT = 1, 1, 32
T = BT * NT
scale = D ** -0.5
torch.manual_seed(0)
q = torch.randn(B, T, H, D) * 0.2
k = torch.randn(B, T, H, D) * 0.2
v = torch.randn(B, T, H, D) * 0.1
g_raw = torch.randn(B, T, H, D) * 0.1
beta_raw = torch.randn(B, T, H)
a_log = torch.linspace(-1.0, 0.2, H)
bias = torch.randn(H, D) * 0.03
A_exp = torch.exp(a_log)
gate = LB * torch.sigmoid(A_exp[None, None, :, None] * (g_raw + bias[None, None, :, :])) * RCP_LN2
gc0 = torch.zeros_like(gate)
for i in range(NT):
    s, e = i * BT, (i + 1) * BT
    gc0[:, s:e] = torch.cumsum(gate[:, s:e], dim=1)
bs = beta_raw.sigmoid()
QN = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + EPS))[0, :, 0]
KN = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + EPS))[0, :, 0]
VC, GC, BETA = v[0, :, 0], gc0[0, :, 0], bs[0, :, 0]


def terms(i, dtype=torch.float32):
    s = i * BT
    qc, kc, vc = (x[s:s + BT].to(dtype) for x in (QN, KN, VC))
    gc, bc = GC[s:s + BT].to(dtype), BETA[s:s + BT].to(dtype)
    gr = gc - gc[min(BT // 2, BT - 1)][None, :]
    Aqk = torch.tril((qc * torch.exp2(gr)) @ (kc * torch.exp2(-gr)).T) * scale
    Akk = torch.tril(((kc * torch.exp2(gr)) @ (kc * torch.exp2(-gr)).T) * bc[:, None], diagonal=-1)
    Ai = torch.eye(BT, dtype=dtype)
    for r in range(1, BT):
        for j in range(r):
            Ai[r] -= Akk[r, j] * Ai[j]
    w = Ai @ (kc * bc[:, None] * torch.exp2(gc))
    u = Ai @ (vc * bc[:, None])
    kg = kc * torch.exp2(gc[-1][None, :] - gc)
    qg = qc * torch.exp2(gc)
    return dict(w=w, u=u, kg=kg, qg=qg, Aqk=Aqk, d=torch.exp2(gc[-1]))


TH = [terms(i) for i in range(NT)]


def direct():
    S = torch.zeros(D, D)
    out = []
    for i in range(NT):
        t = TH[i]
        vn = t["u"] - t["w"] @ S.T
        out.append(scale * (t["qg"] @ S.T) + t["Aqk"] @ vn)
        S = S * t["d"][None, :] + vn.T @ t["kg"]
    return torch.cat(out), S


# ---- 1/2: affine form and segment composition (fp64 reference) -------------
TH64 = [terms(i, torch.float64) for i in range(NT)]
S_in = torch.randn(D, D, dtype=torch.float64) * 0.01
t64 = TH64[0]
S_out = S_in @ (torch.diag(t64["d"]) - t64["w"].T @ t64["kg"]) + t64["u"].T @ t64["kg"]
vn = t64["u"] - t64["w"] @ S_in.T
S_ref = S_in * t64["d"][None, :] + vn.T @ t64["kg"]
print("1) per-chunk affine   |S_out - (S M + Q)| max = %.3e"
      % float((S_out - S_ref).abs().max()))

S, Ms, Qs = torch.zeros(D, D, dtype=torch.float64), [], []
for i in range(NT):
    t = TH64[i]
    S = S * t["d"][None, :] + (t["u"] - t["w"] @ S.T).T @ t["kg"]
    Ms.append(torch.diag(t["d"]) - t["w"].T @ t["kg"])
    Qs.append(t["u"].T @ t["kg"])
M_seg = torch.eye(D, dtype=torch.float64)
for Mi in Ms:
    M_seg = M_seg @ Mi
Q_seg = torch.zeros(D, D, dtype=torch.float64)
for j in range(NT):
    term = Qs[j]
    for Mi in Ms[j + 1:]:
        term = term @ Mi
    Q_seg = Q_seg + term
print("2) segment composition (NT=%d)      |S_direct - (S M_seg + Q_seg)| max = %.3e"
      % (NT, float((S - (S_in @ M_seg + Q_seg)).abs().max())))

# ---- 3: the h = I trick ----------------------------------------------------
S0, S1 = torch.zeros(D, D), torch.eye(D)
for i in range(NT):
    t = TH[i]
    S0 = S0 * t["d"][None, :] + t["u"].T @ t["kg"]
    S1 = S1 * t["d"][None, :] + (t["u"] - t["w"] @ S1.T).T @ t["kg"]
print("3) h=I trick          |Q_seg - S0| max = %.3e   |M_seg - (S1 - S0)| max = %.3e"
      % (float((Q_seg.float() - S0).abs().max()), float((M_seg.float() - (S1 - S0)).abs().max())))

# ---- pass 3: naive form is wrong, corrected form is not --------------------
ref_o, ref_s = direct()



def three_pass(seg):
    segs = []
    for j in range(0, NT, seg):
        S0 = torch.zeros(D, D)
        S1 = torch.eye(D)
        mods = []
        for i in range(j, j + seg):
            t = TH[i]
            R = scale * t["qg"] - t["Aqk"] @ t["w"]
            mods.append((t["Aqk"] @ t["u"] + R @ S0.T, R @ (S1 - S0).T))
            S0 = S0 * t["d"][None, :] + t["u"].T @ t["kg"]
            S1 = S1 * t["d"][None, :] + (t["u"] - t["w"] @ S1.T).T @ t["kg"]
        segs.append((S1 - S0, S0.clone(), mods))
    S = torch.zeros(D, D)
    S_in_j = []
    for Mj, Qj, _ in segs:
        S_in_j.append(S)
        S = S @ Mj + Qj
    out, out_naive = [], []
    for j, (Mj, Qj, mods) in enumerate(segs):
        for i in range(seg):
            t = TH[j * seg + i]
            R = scale * t["qg"] - t["Aqk"] @ t["w"]
            out_naive.append(t["Aqk"] @ t["u"] + R @ S_in_j[j].T)
            o_p, R_p = mods[i]
            out.append(o_p + R_p @ S_in_j[j].T)
    return torch.cat(out), torch.cat(out_naive), S


print()
print("seg  pass3_naive_o_rel  pass3_corrected_o_rel  corrected_o_abs  state_rel")
for seg in (4, 8, 16):
    o, o_naive, S = three_pass(seg)
    print("%-4d %.3e           %.3e                %.3e          %.3e" % (
        seg,
        float((o_naive - ref_o).abs().max()) / float(ref_o.abs().max()),
        float((o - ref_o).abs().max()) / float(ref_o.abs().max()),
        float((o - ref_o).abs().max()),
        float((S - ref_s).abs().max()) / float(ref_s.abs().max())))
