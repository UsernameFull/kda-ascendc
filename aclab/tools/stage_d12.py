"""Stage A: d1+d2 via k2_m1 (AIC). Saves c1,c2 + inputs needed downstream to .pt."""
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
rtc_compile(open(ACLAB_ROOT / "k2_m1.cpp").read(), "k2_m1", "")
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

hb = nd(h.to(bf16)); wnd = nd(w)
c1 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
c2 = torch.zeros(16,128, dtype=torch.float32, device=DEV)
args = [struct.pack("<q",p) for p in [wnd.data_ptr(), qg_c.data_ptr(), hb.data_ptr(), c1.data_ptr(), c2.data_ptr(), 0]]+[struct.pack("<i",5)]
launch_argsarray_engine("k2_m1", 1, torch_npu.npu.current_stream().npu_stream, args, 0)  # warmup
torch.npu.synchronize()
launch_argsarray_engine("k2_m1", 1, torch_npu.npu.current_stream().npu_stream, args, 0)
torch.npu.synchronize()

d1_ref = w.float() @ h.float().t()
d2_ref = qg_c.float() @ h.float().t()
print(f"d1 err={(c1-d1_ref).abs().max().item():.2e}")
print(f"d2 err={(c2-d2_ref).abs().max().item():.2e}")

# save everything downstream needs
torch.save({
    "w": w.cpu(), "u": u.cpu(), "aqk": aqk.cpu(), "q": q.cpu(), "k": k.cpu(),
    "g": g.cpu(), "h": h.cpu(), "kg": kg.cpu(), "c1": c1.cpu(), "c2": c2.cpu(),
    "scale": scale,
}, "/tmp/opencode/k2step_stageA.pt")
print("saved stageA")
