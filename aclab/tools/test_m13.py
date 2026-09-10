"""M1.3: complete K2 single-step math closure via SEPARATE verified kernels.
Each dot uses its own verified kernel (no shared-buffer pollution):
  d1+d2: k2_m1  (AIC, w@h^T + qg@h^T -> fp32 GM)
  d3:    kda_bt16_smoke P3 (Aqk@v_new)
  d4:    kda_k2_m128 (v_new^T @ kg)
Glue (v_new, qg, kg, output, state) in numpy (bf16-exact ref).
Compare final output/state vs the SAME math done purely in torch bf16."""
import sys, struct, os
from pathlib import Path

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
ACLAB_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = Path(os.environ.get(
    "KDA_BT16_LAUNCHER_DIR",
    Path.home() / ".cache" / "torch_extensions"
    / f"py{sys.version_info.major}{sys.version_info.minor}_cpu" / "kda_bt16_launcher",
))
sys.path.insert(0, str(LAUNCHER_DIR))
import numpy as np
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine

DEV = torch.device("npu:0")
def nd(x):
    return torch_npu.npu_format_cast(x.contiguous(), 2).contiguous()
rtc_compile(open(ACLAB_ROOT / "k2_m1.cpp").read(), "k2_m1", "")
torch.npu.synchronize()
print("k2_m1 compiled")
rtc_compile(open(ACLAB_ROOT / "kda_bt16_smoke.cpp").read(), "kda_bt16_smoke_kernel", "")
torch.npu.synchronize()
print("smoke compiled")
rtc_compile(open(ACLAB_ROOT / "kda_k2_m128.cpp").read(), "kda_k2_m128_kernel", "")
torch.npu.synchronize()
print("k2_m128 compiled")

torch.manual_seed(7)
bf16 = torch.bfloat16
w = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
u = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
aqk = torch.randn(16,16, dtype=bf16, device=DEV)*0.3
q = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
k = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
g = torch.randn(16,128, dtype=torch.float32, device=DEV)*0.5  # g_cum log2 domain
h = torch.zeros(128,128, dtype=torch.float32, device=DEV)     # zero initial state
scale = 128**-0.5

# ---- torch bf16 reference (exact K2 math) ----
qn = q.float(); kn = k.float()
qsum = torch.sqrt((qn*qn).sum(1, keepdim=True)+1e-6)
ksum = torch.sqrt((kn*kn).sum(1, keepdim=True)+1e-6)
qn2 = qn/qsum; kn2 = kn/ksum
qg_c = nd((qn2*torch.exp2(g)).to(bf16))
hc = h.to(bf16)
d1_ref = w.float() @ hc.float().t()
v_new = u.float() - d1_ref
vb = v_new.to(bf16)
d2_ref = qg_c.float() @ hc.float().t()
d3_ref = aqk.float() @ vb.float()
o_ref = scale*d2_ref + d3_ref
g_last = g[-1]
kg = nd((kn2*torch.exp2(g_last[None,:]-g)).to(bf16))
d4_ref = vb.float().t() @ kg.float()
h_ref = h*torch.exp2(g_last)[None,:] + d4_ref

# ---- Cube-side computation ----
# d1+d2 via k2_m1
c1 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
c2 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
hb = nd(h.to(bf16))
wnd = nd(w); qgnd = nd(qg_c)
args = [struct.pack("<q",p) for p in [wnd.data_ptr(), qgnd.data_ptr(), hb.data_ptr(), c1.data_ptr(), c2.data_ptr(), 0]]+[struct.pack("<i",5)]
launch_argsarray_engine("k2_m1", 1, torch_npu.npu.current_stream().npu_stream, args, 0)  # warmup
torch.npu.synchronize()
launch_argsarray_engine("k2_m1", 1, torch_npu.npu.current_stream().npu_stream, args, 0)
torch.npu.synchronize()

# glue: v_new = u - c1
vb_t = nd((u.float()-c1).to(bf16))
vT = nd(vb_t.t().contiguous())

# d3 via smoke P3 (Aqk @ v_new -> C3)
c3 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
aqknd = nd(aqk); vbnd = nd(vb_t)
ptrs3 = [aqknd.data_ptr(), vbnd.data_ptr(), c3.data_ptr(), 0,0,0,0,0,0]
args3 = [struct.pack("<q",p) for p in ptrs3]+[struct.pack("<i",3)]
launch_argsarray_engine("kda_bt16_smoke_kernel", 1, torch_npu.npu.current_stream().npu_stream, args3, 0)  # warmup
torch.npu.synchronize()
launch_argsarray_engine("kda_bt16_smoke_kernel", 1, torch_npu.npu.current_stream().npu_stream, args3, 0)
torch.npu.synchronize()

# d4 via kda_k2_m128 (v_new^T @ kg -> C4)
c4 = torch.zeros(128,128, dtype=torch.float32, device=DEV)
kgnd = nd(kg)
args4 = [struct.pack("<q",p) for p in [vT.data_ptr(), kgnd.data_ptr(), c4.data_ptr(), 0]]+[struct.pack("<i",0)]
launch_argsarray_engine("kda_k2_m128_kernel", 1, torch_npu.npu.current_stream().npu_stream, args4, 0)  # warmup
torch.npu.synchronize()
launch_argsarray_engine("kda_k2_m128_kernel", 1, torch_npu.npu.current_stream().npu_stream, args4, 0)
torch.npu.synchronize()

# glue: output + state
o_cube = scale*c2 + c3
h_cube = h*torch.exp2(g_last)[None,:] + c4

print(f"d1 err={(c1-d1_ref).abs().max().item():.2e}")
print(f"d2 err={(c2-d2_ref).abs().max().item():.2e}")
print(f"d3 err={(c3-d3_ref).abs().max().item():.2e}")
print(f"d4 err={(c4-d4_ref).abs().max().item():.2e}")
print(f"OUTPUT err={(o_cube-o_ref).abs().max().item():.2e}")
print(f"STATE  err={(h_cube-h_ref).abs().max().item():.2e}")
