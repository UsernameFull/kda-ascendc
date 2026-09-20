"""Route A arithmetic floor: batch every (head, segment) pair, seg-sequential.

The per-launch version of this probe (probe_segment_cost.py) measures host
launch overhead, not device cost: its carries are 128 sequential tiny torch
ops per head.  The segments of a head are *independent*, so all of them - and
all heads - can be in flight at once, which is exactly how a kernel would
schedule pass 1.  This probe measures that floor:

  state  [NS_total, D, D] fp32   (NS_total = heads * segments)
  8 sequential steps (seg=8), each step batched over every (head, segment):
      S0 = S0 * d_k + U_k^T @ Kg_k
      S1 = S1 * d_k + (U_k - W_k @ S1^T)^T @ Kg_k
  then M_seg = S1 - S0, Q_seg = S0      (writes, 128x128 fp32 = 64 KB each)

Pass 3 floor: bmm over every chunk:  R'_i @ S_in[j]^T   [C, M, D] x [C, D, D]

Reported against the measured K2 stage (4.20 ms at this shape) and its
1-head chain floor (1.03 ms / 128 steps = 8.1 us).

Usage: ASCEND_RT_VISIBLE_DEVICES=1 KDA_CHUNK=64 python3 tools/probe_segment_floor.py
"""
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from kda_ascendc_v1.api import CHUNK, kda_bt16_fwd_ascendc, get_last_profile  # noqa: E402

D, DEV, LB = 128, torch.device("npu:0"), -1.0
B, T, H = 1, 8192, 96
k2_measured_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 4.20


def sync():
    torch.npu.synchronize()


def timeit(fn, warm=2, reps=3):
    for _ in range(warm):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return min(out)


torch.manual_seed(1312)
q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
g = torch.randn(B, T, H, D, device=DEV) * 0.1
beta = torch.randn(B, T, H, device=DEV)
a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
bias = torch.randn(H, D, device=DEV) * 0.03
kw = dict(A_log=a_log, bias=bias, lower_bound=LB, output_final_state=True)

kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
sync()
print("e2e wall: %.3f ms" % timeit(lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)))

out, state, dbg = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw, return_intermediates=True)
sync()
NT = T // CHUNK
BH = B * H
C = BH * NT
W = dbg["W"].view(BH, NT, CHUNK, D).to(torch.bfloat16)
U = dbg["U"].view(BH, NT, CHUNK, D).to(torch.bfloat16)
Kg = dbg["Kg"].view(BH, NT, CHUNK, D).to(torch.bfloat16)
Aqk = dbg["Aqk"].view(BH, NT, CHUNK, CHUNK).to(torch.bfloat16)
Dec = dbg["Decay"].view(BH, NT, D).float()
print("shape: heads %d, NT %d, chunks %d" % (BH, NT, C))


def pack(seg):
    """[BH, NT, ...] -> [BH * NT/seg, seg, ...] so a segment is one batch row."""
    NS = NT // seg
    def r(t):
        return t.view(BH, NS, seg, *t.shape[2:])
    return NS, r(W), r(U), r(Kg), r(Aqk), Dec.view(BH, NS, seg, D)


def pass1(seg):
    NS, Wr, Ur, Kgr, Aqkr, Dr = pack(seg)
    S0 = torch.zeros(BH * NS, D, D, device=DEV)
    S1 = torch.zeros(BH * NS, D, D, device=DEV)
    S1[:, torch.arange(D, device=DEV), torch.arange(D, device=DEV)] = 1.0
    for kk in range(seg):
        d = Dr[:, :, kk].reshape(BH * NS, D)
        uk = Ur[:, :, kk].reshape(BH * NS, CHUNK, D)
        wk = Wr[:, :, kk].reshape(BH * NS, CHUNK, D)
        kgk = Kgr[:, :, kk].reshape(BH * NS, CHUNK, D)
        S0 = S0 * d[:, None, :] + torch.bmm(uk.float().transpose(1, 2), kgk.float())
        vn = uk.float() - torch.bmm(wk.float(), S1.transpose(1, 2))
        S1 = S1 * d[:, None, :] + torch.bmm(vn.transpose(1, 2), kgk.float())
    M = (S1 - S0).to(torch.bfloat16)
    return M, S0.to(torch.bfloat16)


def pass3(seg):
    NS, Wr, Ur, Kgr, Aqkr, Dr = pack(seg)
    M = torch.zeros(BH * NS, D, D, device=DEV, dtype=torch.bfloat16)
    S_in = torch.zeros(BH * NS, D, D, device=DEV, dtype=torch.bfloat16)
    S_in[:, torch.arange(D, device=DEV), torch.arange(D, device=DEV)] = 1.0
    # per-chunk R' = -Aqk @ W   [C, M, D]  (scale*qg dropped: same shape/kind)
    R = -torch.bmm(Aqkr.float().reshape(BH * NS * seg, CHUNK, CHUNK),
                   Wr.float().reshape(BH * NS * seg, CHUNK, D))
    # out += R' @ S_in^T per chunk: [C, M, D] x [C, D, D] -> [C, M, D]
    S_in_c = S_in.float().repeat_interleave(seg, dim=0)
    acc = torch.bmm(R, S_in_c.transpose(1, 2))
    return acc, M


print()
print("seg  pass1_ms  (x1)   per-head-ms   pass3_ms  total_ms   vs k2=%.2f ms" % k2_measured_ms)
for seg in (4, 8, 16):
    t1 = timeit(lambda: pass1(seg))
    t3 = timeit(lambda: pass3(seg))
    tot = t1 + t3
    print("%-4d %-11.3f %-13.3f %-9.3f %-10.3f %s" % (
        seg, t1, t1 / (BH * (NT // seg)) * (BH * (NT // seg)),
        t3, tot, ("%.2fx" % (tot / k2_measured_ms))))
print()
print("read: pass1 here is the FULL shape (every head x every segment batched),")
print("      8 sequential steps - the floor a pass-1 kernel could reach.")
