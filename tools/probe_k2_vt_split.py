"""Where do the v_new repack's 0.152 ms go: the gather or the transposes?

tools/probe_k2_vt.py re-measured the section 11.25 ablation at [1,8192,96,128]:
dropping the `V` store is -0.015 (already guarded in production), and dropping
the gather plus the 16 `AscendC::Transpose` calls on top of that is -0.152 ms of
K2.  A legal replacement needs to know which half carries the cost:

  V6  gather + transposes both gone (the 11.25 floor, wrong by construction)
  V5a gather kept, transposes gone (wrong: stale Vt)
  V5b gather replaced by one contiguous copy of vb, transposes kept (wrong
      layout, same bytes)
  V7  gather kept, the 16 `Transpose` calls replaced by 16 `TransDataTo5HD`
      calls (the nchwconv primitive - same 16x16 result, different instruction)

  V5a - V6  pays the gather and its MTE2_V/V_MTE3 pairs
  V5b - V5a  differences the transpose cost against the gather's

V7 was the only legal candidate and it is rejected on speed alone (+0.154 ms
of K2, worse than the 16 primitive transposes it replaces) - its numerics were
therefore never checked; the arms above V6 are ablations.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_k2_vt_split.py
"""
from __future__ import annotations

import faulthandler
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(2400, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
STOCK = "kda_k2_persistent_loop"
cap = {}
_L = api._launch


def spy(n, b, a, s):
    if n == STOCK and "args" not in cap:
        cap.update(blocks=b, args=list(a), stream=s)
    return _L(n, b, a, s)


api._launch = spy
torch.manual_seed(1312)
q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
g = torch.randn(B, T, H, D, device=DEV) * 0.1
beta = torch.randn(B, T, H, device=DEV)
a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
bias = torch.randn(H, D, device=DEV) * 0.03
api.kda_bt16_fwd_ascendc(q, k, v, g, beta, A_log=a_log, bias=bias,
                         lower_bound=-1.0, output_final_state=True)
torch.npu.synchronize()
api._launch = _L
src = (ROOT / "kernels/v1/k2_persistent_loop.cpp").read_text(encoding="utf-8-sig")
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
XPOS5HD = """                for (int32_t bl = 0; bl < (BV / FR) * NB; ++bl) {
                    uint64_t dstList[16];
                    uint64_t srcList[16];
                    for (int32_t r = 0; r < 16; ++r) {
                        dstList[r] = static_cast<uint64_t>(vt[bl * FR * FR + r * FR].GetPhyAddr());
                        srcList[r] = static_cast<uint64_t>(sc[bl * FR * FR + r * FR].GetPhyAddr());
                    }
                    TransDataTo5HDParams tp(false, false, 1, 0, 0);
                    TransDataTo5HD<uint16_t>(dstList, srcList, tp);
                }
"""
BIGCOPY = """                DataCopy(sc, vb, TILE);
"""

# The V store is already guarded in production (section 11.25), and the probe
# drops it in every arm so the arms differ only in the gather/transpose pair.
V_STORE = "if (pVnew != nullptr) DataCopy(V[out0], vb, DataCopyParams(M, BV / 16, 0, 0));"
assert src.count(V_STORE) == 1
base_src = drop(src, V_STORE)
v6 = drop(drop(base_src, GATHER), XPOS)          # the 11.25 floor
v5a = drop(base_src, XPOS)                       # gather kept, transposes gone
v5b = base_src.replace(GATHER, BIGCOPY)          # big copy + transposes
v7 = base_src.replace(XPOS, XPOS5HD)             # the only legal candidate

variants = [("V0 stock", src), ("V6 floor (no gather, no xpose)", v6),
            ("V5a gather only", v5a), ("V5b bigcopy + xpose", v5b),
            ("V7 gather + TransDataTo5HD", v7)]
comp = {}
for lab, txt in variants:
    nm = "kda_k2_vts_%d" % len(comp)
    api.rtc_compile(defs + txt.replace(STOCK, nm), nm, "")
    comp[lab] = nm
    print("compiled %-30s -> %s" % (lab, nm), flush=True)

REPS = 11


def t1(name):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    _L(name, cap["blocks"], cap["args"], cap["stream"])
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3


ms = {lab: [] for lab, _ in variants}
ms[STOCK] = []
t1(STOCK)
for _ in range(REPS):
    ms[STOCK].append(t1(STOCK))
    for lab, nm in comp.items():
        ms[lab].append(t1(nm))
base = min(ms[STOCK])
print("\nK2 only, [1,%d,%d,%d], C=%d, %d blocks, MIN of %d interleaved"
      % (T, H, D, api.CHUNK, cap["blocks"], REPS))
print("%-30s %-9s %-9s %s" % ("variant", "min_ms", "median", "d_vs_stock"))
for lab, _ in variants:
    x = ms[lab]
    print("%-30s %-9.3f %-9.3f %+.3f" % (lab, min(x), statistics.median(x), min(x) - base))
