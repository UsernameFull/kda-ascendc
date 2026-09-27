"""Is dropping the store-side pass marker numerically safe?

§11.13 says the pair exists to close "MTE3-read -> V-write" across the pass
boundary, and D2 measured -0.230 ms when it was dropped -- but D2 was
timing-only.  The repo's own rule is "先看逐位一致，再看时间", so: drop it,
run the full pipeline, and diff every intermediate plus out/state.

If it IS bit-identical, the 0.230 ms is free.  If it is not, the marker is
load-bearing and recovering its cost needs ~28 KB of double buffering.

ASCEND_RT_VISIBLE_DEVICES=1 KDA_CHUNK=64 python3 /tmp/d2num.py
"""
import faulthandler, struct, sys, time
from pathlib import Path
import torch, torch_npu
faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path("/workspace/kda-ascendc"); sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 1024, 16          # small but multi-pass (C=64 -> NP=4, XBAND on)
torch.manual_seed(1312)
q = (torch.randn(B,T,H,D,device=DEV)*0.2).to(torch.bfloat16)
k = (torch.randn(B,T,H,D,device=DEV)*0.2).to(torch.bfloat16)
v = (torch.randn(B,T,H,D,device=DEV)*0.1).to(torch.bfloat16)
g = torch.randn(B,T,H,D,device=DEV)*0.1
beta = torch.randn(B,T,H,device=DEV)
a_log = torch.linspace(-1.0,0.2,H,device=DEV); bias = torch.randn(H,D,device=DEV)*0.03
kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True,
          return_intermediates=True)

# stock compile + reference
out_ref, st_ref, dbg_ref = api.kda_bt16_fwd_ascendc(q,k,v,g,beta,**kw)
torch.npu.synchronize()
print("stock reference done", flush=True)

src = (ROOT/"kernels/v1/k1_pre_gram_mix.cpp").read_text(encoding="utf-8-sig")
MARK = "SetFlag<HardEvent::MTE3_V>(e3p); WaitFlag<HardEvent::MTE3_V>(e3p);"
assert src.count(MARK) == 1
api.rtc_compile(api._defines() + src.replace(MARK, "").replace(
    "kda_pre_gram_mix", "kda_pg_d2"), "kda_pg_d2", "")
print("compiled the marker-dropped twin", flush=True)

_real = api._launch
def run_twin():
    def patched(n, b, a, s):
        return _real("kda_pg_d2" if n == "kda_pre_gram_mix" else n, b, a, s)
    api._launch = patched
    try:
        return api.kda_bt16_fwd_ascendc(q,k,v,g,beta,**kw)
    finally:
        api._launch = _real

out_t, st_t, dbg_t = run_twin()
torch.npu.synchronize()

print()
print("=== drop the store-side marker: bit-exactness vs stock ===")
print("%-12s %-14s %s" % ("tensor", "max|diff|", "verdict"))
worst = 0.0
for name in sorted(dbg_ref):
    a, b = dbg_ref[name], dbg_t[name]
    if a is None or b is None:
        continue
    try:
        d = float((a.float() - b.float()).abs().max().cpu())
    except Exception:
        d = float("nan")
    worst = max(worst, d if d == d else 0.0)
    print("%-12s %-14.6e %s" % (name, d, "SAME" if d == 0.0 else "**DIFF**"))
for name, a, b in (("out", out_ref, out_t), ("state", st_ref, st_t)):
    d = float((a.float()-b.float()).abs().max().cpu())
    worst = max(worst, d)
    print("%-12s %-14.6e %s" % (name, d, "SAME" if d == 0.0 else "**DIFF**"))
print()
print("worst diff over all tensors: %.6e  -> %s"
      % (worst, "marker is DEAD, 0.230 ms is free" if worst == 0.0
         else "marker is LOAD-BEARING, needs double buffering"))
