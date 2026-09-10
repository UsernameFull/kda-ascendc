"""Stage B: d3+d4 via exact test_smoke.py structure (P1 warmup, then P3/P4-style)."""
import sys, struct
from pathlib import Path
import os

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
KERNEL_SRC = open(ACLAB_ROOT / "kda_bt16_smoke.cpp").read()
rtc_compile(KERNEL_SRC, "kda_bt16_smoke_kernel", "")

def stream_handle():
    return torch_npu.npu.current_stream().npu_stream

torch.manual_seed(7)
bf16 = torch.bfloat16
aqk = torch.randn(16,16, dtype=bf16, device=DEV)*0.3
vb = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
A = torch.randn(16,128, dtype=bf16, device=DEV)
B16h = vb  # d3 uses v_new as B

C1 = torch.zeros(16,16, dtype=torch.float32, device=DEV)
C3 = torch.zeros(16,128, dtype=torch.float32, device=DEV)

def run(nprobe, B16h_in, C3_out):
    ptrs = [A.data_ptr(), B16h_in.data_ptr(), A.data_ptr(), A.data_ptr(), aqk.data_ptr(),
            C1.data_ptr(), C3_out.data_ptr(), C1.data_ptr(), C1.data_ptr()]
    args = [struct.pack("<q", p) for p in ptrs] + [struct.pack("<i", nprobe)]
    launch_argsarray_engine("kda_bt16_smoke_kernel", 1, stream_handle(), args, 0)
    torch.npu.synchronize()

# P1 warmup (exactly like test_smoke)
run(1, B16h, C3)
run(1, B16h, C3)  # discard
# P3
run(3, B16h, C3)
ref3 = aqk.float() @ vb.float()
print(f"d3 err={(C3-ref3).abs().max().item():.2e}")

# ---- d4 via kda_k2_m128 (v_new^T @ kg -> C4) ----
rtc_compile(open(ACLAB_ROOT / "kda_k2_m128.cpp").read(), "kda_k2_m128_kernel", "")
torch.npu.synchronize()
kg = torch.randn(16,128, dtype=bf16, device=DEV)*0.3
vT = vb.t().contiguous()
C4 = torch.zeros(128,128, dtype=torch.float32, device=DEV)
def run4(A4, B4, C4_out):
    args = [struct.pack("<q",p) for p in [A4.data_ptr(), B4.data_ptr(), C4_out.data_ptr(), 0]]+[struct.pack("<i",0)]
    launch_argsarray_engine("kda_k2_m128_kernel", 1, stream_handle(), args, 0)
    torch.npu.synchronize()
run4(vT, kg, C4)   # warmup
run4(vT, kg, C4)   # discard
run4(vT, kg, C4)   # record
ref4 = vb.float().t() @ kg.float()
print(f"d4 err={(C4-ref4).abs().max().item():.2e}")
torch.save({"d3": C3.cpu(), "d4": C4.cpu(), "vb": vb.cpu(), "kg": kg.cpu(), "aqk": aqk.cpu()},
           "/tmp/opencode/k2step_stageB.pt")
