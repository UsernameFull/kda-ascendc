"""AIV glue benchmark: correctness + latency of k2_glue_min_kernel."""
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
import time
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine

DEV = torch.device("npu:0")
def nd(x):
    return torch_npu.npu_format_cast(x.contiguous(), 2).contiguous()
rtc_compile((ACLAB_ROOT / "k2_glue_min.cpp").read_text(), "k2_glue_min_kernel", "")
torch.npu.synchronize()

torch.manual_seed(11)
bf16 = torch.bfloat16
u = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
g_last = torch.randn(128, dtype=torch.float32, device=DEV)*0.5
h = torch.randn(128,128, dtype=torch.float32, device=DEV)*0.3
d1 = torch.randn(16,128, dtype=torch.float32, device=DEV)*0.3
d2 = torch.randn(16,128, dtype=torch.float32, device=DEV)*0.3
d3 = torch.randn(16,128, dtype=torch.float32, device=DEV)*0.3
d4 = torch.randn(128,128, dtype=torch.float32, device=DEV)*0.3
scale = 128**-0.5

# torch reference
v_ref = u.float() - d1
o_ref = scale*d2 + d3
h_ref = h*torch.exp2(g_last)[None,:] + d4

vnew = torch.zeros(16,128, dtype=bf16, device=DEV)
out = torch.zeros(16,128, dtype=bf16, device=DEV)
hnew = torch.zeros(128,128, dtype=torch.float32, device=DEV)

und = nd(u)
ptrs = [und.data_ptr(), g_last.data_ptr(), h.data_ptr(), d1.data_ptr(), d2.data_ptr(), d3.data_ptr(), d4.data_ptr(),
        vnew.data_ptr(), out.data_ptr(), hnew.data_ptr(), 0]
args = [struct.pack("<q",p) for p in ptrs]+[struct.pack("<i",0)]

# warmup x3
for _ in range(3):
    launch_argsarray_engine("k2_glue_min_kernel", 1, torch_npu.npu.current_stream().npu_stream, args, 0)
    torch.npu.synchronize()

# correctness
ev = vnew.float() - v_ref
print(f"v_new err={(ev).abs().max().item():.2e}")
eo = out.float() - o_ref
print(f"out   err={(eo).abs().max().item():.2e}")
eh = hnew - h_ref
print(f"state err={(eh).abs().max().item():.2e}")

# latency: N launches back-to-back, wall-clock / N
N = 200
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    launch_argsarray_engine("k2_glue_min_kernel", 1, torch_npu.npu.current_stream().npu_stream, args, 0)
torch.npu.synchronize()
t1 = time.perf_counter()
us = (t1-t0)/N*1e6
print(f"AIV glue: {us:.2f} us/step (wall, incl. launch overhead)")
