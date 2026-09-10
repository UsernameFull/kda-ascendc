"""Stage B: d3 (smoke P3) + d4 (k2_m128) in independent process.
Reads stageA outputs, computes vb = u - c1, launches d3/d4, saves results."""
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
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine

DEV = torch.device("npu:0")
def nd(x):
    return torch_npu.npu_format_cast(x.contiguous(), 2).contiguous()
rtc_compile(open(ACLAB_ROOT / "kda_bt16_smoke.cpp").read(), "kda_bt16_smoke_kernel", "")
torch.npu.synchronize()

torch.manual_seed(7)
bf16 = torch.bfloat16
w = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
u = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
aqk = torch.randn(16,16, dtype=bf16, device=DEV)*0.3
q = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
k = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
g = torch.randn(16,128, dtype=torch.float32, device=DEV)*0.5
h = torch.zeros(128,128, dtype=torch.float32, device=DEV)
scale = 128**-0.5
qn = q.float(); kn = k.float()
qsum = torch.sqrt((qn*qn).sum(1, keepdim=True)+1e-6); ksum = torch.sqrt((kn*kn).sum(1, keepdim=True)+1e-6)
qn2 = qn/qsum; kn2 = kn/ksum
qg_c = nd((qn2*torch.exp2(g)).to(bf16))
g_last = g[-1]
kg = nd((kn2*torch.exp2(g_last[None,:]-g)).to(bf16))
# c1 = w@h^T with h=0 -> 0; c2 = qg@h^T -> 0 (h is zero state)
c1 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
c2 = torch.zeros(16,128, dtype=torch.float32, device=DEV)

# glue: v_new = u - c1
vb_t = nd((u.float()-c1).to(torch.bfloat16))
vT = nd(vb_t.t().contiguous())
d3_ref = aqk.float() @ vb_t.float()
d4_ref = vb_t.float().t() @ kg.float()

# d3 via smoke P3 (must warmup with P1 mode first, per test_smoke pattern)
c3 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
aqknd = nd(aqk); vbnd = nd(vb_t)
A16 = torch.randn(16,128, dtype=torch.bfloat16, device=DEV)
C1w = torch.zeros(16,16, dtype=torch.float32, device=DEV)
warm = [A16.data_ptr(), A16.data_ptr(), A16.data_ptr(), A16.data_ptr(), aqknd.data_ptr(), C1w.data_ptr(), c3.data_ptr(), C1w.data_ptr(), C1w.data_ptr()]
wargs = [struct.pack("<q",p) for p in warm]+[struct.pack("<i",1)]  # P1 mode warmup
launch_argsarray_engine("kda_bt16_smoke_kernel", 1, torch_npu.npu.current_stream().npu_stream, wargs, 0)
torch.npu.synchronize()
ptrs3 = [aqknd.data_ptr(), vbnd.data_ptr(), c3.data_ptr(), 0,0,0,0,0,0]
args3 = [struct.pack("<q",p) for p in ptrs3]+[struct.pack("<i",3)]
launch_argsarray_engine("kda_bt16_smoke_kernel", 1, torch_npu.npu.current_stream().npu_stream, args3, 0)
torch.npu.synchronize()
print(f"d3 err={(c3-d3_ref).abs().max().item():.2e}")

# d4 via kda_k2_m128
rtc_compile(open(ACLAB_ROOT / "kda_k2_m128.cpp").read(), "kda_k2_m128_kernel", "")
torch.npu.synchronize()
c4 = torch.zeros(128,128, dtype=torch.float32, device=DEV)
kgnd = nd(kg)
args4 = [struct.pack("<q",p) for p in [vT.data_ptr(), kgnd.data_ptr(), c4.data_ptr(), 0]]+[struct.pack("<i",0)]
launch_argsarray_engine("kda_k2_m128_kernel", 1, torch_npu.npu.current_stream().npu_stream, args4, 0)  # warmup
torch.npu.synchronize()
launch_argsarray_engine("kda_k2_m128_kernel", 1, torch_npu.npu.current_stream().npu_stream, args4, 0)
torch.npu.synchronize()
print(f"d4 err={(c4-d4_ref).abs().max().item():.2e}")

# compose output + state, save
g_last = g[-1]
o_cube = scale*c2 + c3
h_cube = h*torch.exp2(g_last)[None,:] + c4
torch.save({"c3": c3.cpu(), "c4": c4.cpu(), "o_cube": o_cube.cpu(), "h_cube": h_cube.cpu()},
           "/tmp/opencode/k2step_stageB.pt")
print("saved stageB")
