"""Route (1) step A: price the dead V store and the transpose separately.

k2_persistent_loop.cpp writes TWO copies of v_new per chunk-head: `V` (row-major
[M, BV], line 591) and `Vt` (packed block order, line 606).  Grep says the kernel
never reads `V` -- only the host does, and only under return_intermediates.  So:

  V5  guard the V store behind a null pointer (the pQn/pKn pattern):
      a production-path deletion, bit-exact by construction.
  V6  V5 + delete the 16 `AscendC::Transpose` and the `sc` gather that feed Vt.
      Numerically wrong (d34's operands go stale) -- prices that work.

Interleaved, same process, MIN of 11.

ASCEND_RT_VISIBLE_DEVICES=3 KDA_CHUNK=64 python3 -u tools/probe_k2_vt.py
"""
import faulthandler, statistics, sys, time
from pathlib import Path
import torch, torch_npu
faulthandler.dump_traceback_later(2400, exit=True)
ROOT = Path("/workspace/kda-ascendc"); sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
STOCK = "kda_k2_persistent_loop"
cap = {}; _L = api._launch
def spy(n, b, a, s):
    if n == STOCK and "args" not in cap: cap.update(blocks=b, args=list(a), stream=s)
    return _L(n, b, a, s)
api._launch = spy
torch.manual_seed(1312)
q = (torch.randn(B,T,H,D,device=DEV)*0.2).to(torch.bfloat16)
k = (torch.randn(B,T,H,D,device=DEV)*0.2).to(torch.bfloat16)
v = (torch.randn(B,T,H,D,device=DEV)*0.1).to(torch.bfloat16)
g = torch.randn(B,T,H,D,device=DEV)*0.1
beta = torch.randn(B,T,H,device=DEV)
a_log = torch.linspace(-1.0,0.2,H,device=DEV); bias = torch.randn(H,D,device=DEV)*0.03
api.kda_bt16_fwd_ascendc(q,k,v,g,beta,A_log=a_log,bias=bias,lower_bound=-1.0,output_final_state=True)
torch.npu.synchronize(); api._launch = _L
src = (ROOT/"kernels/v1/k2_persistent_loop.cpp").read_text(encoding="utf-8-sig")
defs = api._defines()

def drop(t, p):
    assert t.count(p) == 1, (repr(p[:60]), t.count(p))
    return t.replace(p, "")

GATHER = """                for (int32_t j0 = 0; j0 < BV / FR; ++j0) {
                    for (int32_t m0 = 0; m0 < NB; ++m0) {
                        DataCopy(sc[(j0 * NB + m0) * FR * FR], vb[m0 * FR * BV + j0 * FR],
                                 DataCopyParams(FR, 1, BV / FR - 1, 0));
                    }
                }
"""
XPOS = """                for (int32_t bl = 0; bl < (BV / FR) * NB; ++bl) {
                    AscendC::Transpose(vt[bl * FR * FR], sc[bl * FR * FR]);
                }
"""
V_STORE = "DataCopy(V[out0], vb, DataCopyParams(M, BV / 16, 0, 0));"

v5 = drop(src, V_STORE)
v6 = drop(drop(v5, GATHER), XPOS)
variants = {"V0 stock-recompiled": src, "V5 V store dropped": v5,
            "V6 V + gather + transposes dropped": v6}
comp = {}
for lab, txt in variants.items():
    nm = "kda_k2_vt_" + str(len(comp))
    api.rtc_compile(defs + txt.replace(STOCK, nm), nm, ""); comp[lab] = nm
    print("compiled %-36s -> %s" % (lab, nm), flush=True)

REPS = 11
def t1(name):
    torch.npu.synchronize(); t0 = time.perf_counter()
    _L(name, cap["blocks"], cap["args"], cap["stream"]); torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3
ms = {lab: [] for lab in variants}; ms[STOCK] = []
t1(STOCK)
for _ in range(REPS):
    ms[STOCK].append(t1(STOCK))
    for lab, nm in comp.items(): ms[lab].append(t1(nm))
base = min(ms[STOCK])
print("\nK2 only, [1,%d,%d,%d], C=%d, %d blocks, MIN of %d interleaved"
      % (T,H,D,api.CHUNK,cap["blocks"],REPS))
print("%-36s %-9s %-9s %-9s %s" % ("variant","min_ms","median","d_vs_stock","us/step"))
for lab in [STOCK]+list(comp):
    x = ms[lab]; mn = min(x)
    print("%-36s %-9.3f %-9.3f %-+9.3f %-8.2f" % (
        lab if lab != STOCK else "stock (as shipped)", mn, statistics.median(x),
        mn-base if lab != STOCK else 0.0, 1000*mn/512))
