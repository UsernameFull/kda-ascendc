"""Host-side cost attribution for the production call.

Plan section 11.26 measured the *exposed* host share of a production call at
0.873 ms and attributed it to per-call workspace allocation (~0.67 GB) and
argument packing.  That number is what the workspace-pool route (S1) has to
beat, so before building a pool this probe breaks the host side into the steps
the api actually runs and prices each one on its own:

  alloc      the ``torch.empty`` calls that make the per-call workspace
  pack       ``permute().contiguous()`` / ``view`` / ``zero_()`` input packing
  args       ``_pack_ptrs`` + ``_i``/``_f`` blob building for the launches
  launch     the launcher calls themselves (all of them, one production call)

Method: each step is timed *standalone* (no device work in between, so the
number is the host cost of that step and not a synchronization effect), and
then the whole call is timed hot against the sum.  Report MIN of N; the
values are host-side and independent of the concurrent-arm convention used by
probe_stage_overlap.py.

The point of the breakdown is to decide whether a fixed-shape workspace pool
can pay for itself: a pool removes the ``torch.empty`` calls but *not* the
packing, so the pool's ceiling is the ``alloc`` row.

  KDA_CHUNK=64 python3 -u tools/probe_host_cost.py
"""
import os, sys, time
from pathlib import Path

import torch, torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api

D, BV, NV = 128, 64, 2
DEV = torch.device("npu:0")
B, T, H = 1, 8192, 96
CHUNK = api.CHUNK
NT = T // CHUNK
BH = B * H
C = BH * NT
TASKS = BH * NV
REPS = int(os.environ.get("KDA_HOST_REPS", "12"))

torch.manual_seed(1312)
q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
g = torch.randn(B, T, H, D, device=DEV) * 0.1
beta = torch.randn(B, T, H, device=DEV)
a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
bias = torch.randn(H, D, device=DEV) * 0.03
kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

print("warming / RTC (C=%d, H=%d)..." % (CHUNK, H), flush=True)
out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
torch.npu.synchronize()
print("   e2e warm call done", flush=True)


def host_min(fn, reps=REPS):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def timed(fn, reps=REPS):
    """Host-only MIN of ``reps`` runs with no device sync inside the window."""
    return host_min(fn, reps)


def alloc_step():
    """Every ``torch.empty`` a production call makes, in the api's order."""
    qn = torch.empty((C, CHUNK, D), dtype=torch.bfloat16, device=DEV)
    kn = torch.empty_like(qn)
    gate = torch.empty((C, CHUNK, D), dtype=torch.float32, device=DEV)
    gc = torch.empty_like(gate)
    beta_out = torch.empty((C, CHUNK), dtype=torch.float32, device=DEV)
    decay = torch.empty((C, D), dtype=torch.float32, device=DEV)
    rk = torch.empty_like(qn); rv = torch.empty_like(qn)
    qg = torch.empty_like(qn); kg = torch.empty_like(qn)
    aqk32 = torch.empty((C, CHUNK, CHUNK), dtype=torch.float32, device=DEV)
    aqk16 = torch.empty((C, CHUNK, CHUNK), dtype=torch.bfloat16, device=DEV)
    L = torch.empty((C, CHUNK, CHUNK), dtype=torch.float32, device=DEV)
    a32 = torch.empty((C, CHUNK, CHUNK), dtype=torch.float32, device=DEV)
    a16 = torch.empty((C, CHUNK, CHUNK), dtype=torch.bfloat16, device=DEV)
    W = torch.empty_like(qn); U = torch.empty_like(qn)
    sub = CHUNK // api.SOLVE_WIDE_SUBB
    xb = torch.empty((C, api.SOLVE_WIDE_SUBB, sub, sub), dtype=torch.bfloat16, device=DEV)
    lneg = torch.empty((C, sub, sub), dtype=torch.bfloat16, device=DEV)
    pmid = torch.empty((C, sub, sub), dtype=torch.bfloat16, device=DEV)
    s32 = torch.empty((TASKS, BV, D), dtype=torch.float32, device=DEV)
    s16 = torch.empty((TASKS, BV, D), dtype=torch.bfloat16, device=DEV)
    d1 = torch.empty((TASKS, NT, CHUNK, BV), dtype=torch.bfloat16, device=DEV)
    d2 = torch.empty_like(d1); d3 = torch.empty_like(d1)
    d4f = torch.empty((BH, D, D), dtype=torch.float32, device=DEV)
    out_public = torch.empty((B, T, H, D), dtype=torch.bfloat16, device=DEV)
    vnew_t = torch.empty((TASKS, NT, BV, CHUNK), dtype=torch.bfloat16, device=DEV)
    return (qn, kn, gate, gc, beta_out, decay, rk, rv, qg, kg, aqk32, aqk16,
            L, a32, a16, W, U, xb, lneg, pmid, s32, s16, d1, d2, d3, d4f,
            out_public, vnew_t)


def pack_step():
    q_, k_, v_, g_, beta_ = [x.contiguous() for x in (q, k, v, g, beta)]
    qk_row_bytes = H * D * 2 - D * 2
    g_row_bytes = H * D * 4 - D * 4
    beta_pack = beta.view(B, NT, CHUNK, H).permute(0, 3, 1, 2).contiguous().view(C, CHUNK)
    return qk_row_bytes, g_row_bytes, beta_pack


def args_step():
    blobs = api._pack_ptrs([None] * 16) + [api._i(1)] * 6 + [api._f(0.5)] * 2
    return blobs


rows = [
    ("alloc  (%d tensors, %.2f GB)" % (28, 4152e6 / 1e9),
     timed(alloc_step)),
    ("pack   (contiguous + beta permute)", timed(pack_step)),
    ("args   (24 blobs)", timed(args_step)),
]
for name, ms in rows:
    print("   %-38s %8.4f ms" % (name, ms))

# The launches themselves: capture the call, then replay the launcher entries.
seq = []
_orig = api._launch


def _spy(name, blocks, args, stream):
    seq.append((name, int(blocks), list(args)))
    return _orig(name, blocks, args, stream)


api._launch = _spy
api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
torch.npu.synchronize()
api._launch = _orig
print("   captured %d launches" % len(seq), flush=True)


def launch_step():
    for nm, blk, ar in seq:
        api.launch_argsarray_engine(nm, blk, api.torch_npu.npu.current_stream().npu_stream, ar, 0)


launch_ms = timed(launch_step)
print("   %-38s %8.4f ms  (%d launches)" % ("launch (replay, no device work)", launch_ms, len(seq)))

e2e = timed(lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw))
torch.npu.synchronize()
print()
print("   %-38s %8.4f ms" % ("e2e host-side wall (hot, no sync in window)", e2e))
print("   %-38s %8.4f ms" % ("sum of host steps", sum(ms for _, ms in rows) + launch_ms))
