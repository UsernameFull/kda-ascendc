"""GM-boundary golden K2 single step: compose all verified kernels.
Phases (each in an independent subprocess):
  P1 k2_m1        (AIC): d1=w@h^T, d2=qg@h^T
  P2 k2_glue_v    (AIV): v_new = u - d1
  P3 k2_d3 (AIC): d3 = aqk@v_new ; k2_m128 (AIC): d4 = v_new^T@kg
  P4 k2_glue_final(AIV): out = scale*d2+d3, h_new = h*exp2(g_last)+d4
Then compare vs pure-torch reference (exact same inputs)."""
import subprocess, sys, os
from pathlib import Path

ACLAB_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = Path(os.environ.get(
    "KDA_BT16_LAUNCHER_DIR",
    Path.home() / ".cache" / "torch_extensions"
    / f"py{sys.version_info.major}{sys.version_info.minor}_cpu" / "kda_bt16_launcher",
))
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
ROOT = str(ACLAB_ROOT)

# ---- shared input generator (exact seed order, single source of truth) ----
import torch
torch.manual_seed(42)
bf16 = torch.bfloat16
w = (torch.randn(16,128)*0.3).to(bf16); u = (torch.randn(16,128)*0.3).to(bf16)
q = (torch.randn(16,128)*0.3).to(bf16); k = (torch.randn(16,128)*0.3).to(bf16)
g = torch.randn(16,128)*0.5; aqk = (torch.randn(16,16)*0.3).to(bf16)
h = torch.zeros(128,128); scale = 128**-0.5
qn = q.float(); kn = k.float()
qsum = torch.sqrt((qn*qn).sum(1,keepdim=True)+1e-6); ksum = torch.sqrt((kn*kn).sum(1,keepdim=True)+1e-6)
qn2 = qn/qsum; kn2 = kn/ksum
g_last = g[-1]
qg = (qn2*torch.exp2(g)).to(bf16)
kg = (kn2*torch.exp2(g_last[None,:]-g)).to(bf16)
# torch reference of full step (w/u already bf16)
hc = h.to(bf16)
d1r = w.float() @ hc.float().t()
v_ref = u.to(bf16).float() - d1r
vb = v_ref.to(bf16)
d2r = qg.float() @ hc.float().t()
d3r = aqk.to(bf16).float() @ vb.float()
o_ref = scale*d2r + d3r
d4r = vb.float().t() @ kg.float()
h_ref = h*torch.exp2(g_last)[None,:] + d4r
torch.save({"w":w,"u":u,"q":q,"k":k,"g":g,"aqk":aqk,"h":h,"scale":scale,
            "qg":qg,"kg":kg,"g_last":g_last,
            "v_ref":v_ref,"o_ref":o_ref,"h_ref":h_ref,"d1r":d1r,"d2r":d2r,"d3r":d3r,"d4r":d4r},
           "/tmp/opencode/golden_inputs.pt")
print("inputs saved")

# ---- P1: k2_m1 (d1,d2) ----
p1 = f'''
import sys, struct, os
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="1"
sys.path.insert(0,"{LAUNCHER_DIR}")
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine
DEV=torch.device("npu:0")
def nd(x): return torch_npu.npu_format_cast(x.contiguous(),2).contiguous()
d=torch.load("/tmp/opencode/golden_inputs.pt",weights_only=False)
w=d["w"].to(DEV); qg=nd(d["qg"].to(DEV)); h=nd(d["h"].to(torch.bfloat16).to(DEV))
c1=torch.zeros(16,128,dtype=torch.float32,device=DEV); c2=torch.zeros(16,128,dtype=torch.float32,device=DEV)
rtc_compile(open("{ROOT}/k2_m1.cpp").read(),"k2_m1","")
torch.npu.synchronize()
args=[struct.pack("<q",p) for p in [nd(w).data_ptr(),qg.data_ptr(),h.data_ptr(),c1.data_ptr(),c2.data_ptr(),0]]+[struct.pack("<i",5)]
for _ in range(3):
    launch_argsarray_engine("k2_m1",1,torch_npu.npu.current_stream().npu_stream,args,0); torch.npu.synchronize()
d1r=d["d1r"]; d2r=d["d2r"]
print(f"P1 d1 err={{ (c1.cpu()-d1r).abs().max().item():.2e }} d2 err={{ (c2.cpu()-d2r).abs().max().item():.2e }}")
torch.save({{"d1":c1.cpu(),"d2":c2.cpu()}},"/tmp/opencode/p1.pt")
'''
r1 = subprocess.run([sys.executable,"-c",p1],capture_output=True,text=True,timeout=240)
print(r1.stdout.strip().splitlines()[-1] if r1.stdout else "P1 no output")

# ---- P2: k2_glue_v (v_new = u - d1) ----
p2 = f'''
import sys, struct, os
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="1"
sys.path.insert(0,"{LAUNCHER_DIR}")
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine
DEV=torch.device("npu:0")
def nd(x): return torch_npu.npu_format_cast(x.contiguous(),2).contiguous()
d=torch.load("/tmp/opencode/golden_inputs.pt",weights_only=False); p1=torch.load("/tmp/opencode/p1.pt",weights_only=False)
u=nd(d["u"].to(DEV)); d1=nd(p1["d1"].to(DEV)); vb_ref=d["v_ref"].to(torch.bfloat16)
vnew=torch.zeros(16,128,dtype=torch.bfloat16,device=DEV)
rtc_compile(open("{ROOT}/k2_glue_v.cpp").read(),"k2_glue_v_kernel","")
torch.npu.synchronize()
args=[struct.pack("<q",p) for p in [u.data_ptr(),d1.data_ptr(),vnew.data_ptr(),0]]+[struct.pack("<i",0)]
for _ in range(3):
    launch_argsarray_engine("k2_glue_v_kernel",1,torch_npu.npu.current_stream().npu_stream,args,0); torch.npu.synchronize()
print(f"P2 v_new err={{ (vnew.float()-d['v_ref']).abs().max().item():.2e }}")
torch.save({{"vnew":vnew.cpu()}},"/tmp/opencode/p2.pt")
'''
r2 = subprocess.run([sys.executable,"-c",p2],capture_output=True,text=True,timeout=240)
print(r2.stdout.strip().splitlines()[-1] if r2.stdout else "P2 no output")

# ---- P3: k2_d3 (d3) + k2_m128 (d4) ----
p3 = f'''
import sys, struct, os
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="1"
sys.path.insert(0,"{LAUNCHER_DIR}")
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine
DEV=torch.device("npu:0")
def nd(x): return torch_npu.npu_format_cast(x.contiguous(),2).contiguous()
d=torch.load("/tmp/opencode/golden_inputs.pt",weights_only=False); p2=torch.load("/tmp/opencode/p2.pt",weights_only=False)
aqk=nd(d["aqk"].to(DEV)); vnew=nd(p2["vnew"].to(DEV)); kg=nd(d["kg"].to(DEV))
vT=nd(vnew.t().contiguous())
c3=torch.zeros(16,128,dtype=torch.float32,device=DEV); c4=torch.zeros(128,128,dtype=torch.float32,device=DEV)
rtc_compile(open("{ROOT}/k2_d3.cpp").read(),"k2_d3_kernel","")
torch.npu.synchronize()
args3=[struct.pack("<q",p) for p in [aqk.data_ptr(),vnew.data_ptr(),c3.data_ptr(),0]]+[struct.pack("<i",0)]
for _ in range(3):
    launch_argsarray_engine("k2_d3_kernel",1,torch_npu.npu.current_stream().npu_stream,args3,0); torch.npu.synchronize()
rtc_compile(open("{ROOT}/kda_k2_m128.cpp").read(),"kda_k2_m128_kernel","")
torch.npu.synchronize()
args4=[struct.pack("<q",p) for p in [vT.data_ptr(),kg.data_ptr(),c4.data_ptr(),0]]+[struct.pack("<i",0)]
for _ in range(3):
    launch_argsarray_engine("kda_k2_m128_kernel",1,torch_npu.npu.current_stream().npu_stream,args4,0); torch.npu.synchronize()
print(f"P3 d3 err={{ (c3.cpu()-d['d3r']).abs().max().item():.2e }} d4 err={{ (c4.cpu()-d['d4r']).abs().max().item():.2e }}")
torch.save({{"d3":c3.cpu(),"d4":c4.cpu()}},"/tmp/opencode/p3.pt")
'''
r3 = subprocess.run([sys.executable,"-c",p3],capture_output=True,text=True,timeout=260)
print(r3.stdout.strip().splitlines()[-1] if r3.stdout else "P3 no output")

# ---- P4: k2_glue_final (out, h_new) ----
p4 = f'''
import sys, struct, os
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="1"
sys.path.insert(0,"{LAUNCHER_DIR}")
import torch, torch_npu
from kda_bt16_launcher import rtc_compile, launch_argsarray_engine
DEV=torch.device("npu:0")
def nd(x): return torch_npu.npu_format_cast(x.contiguous(),2).contiguous()
d=torch.load("/tmp/opencode/golden_inputs.pt",weights_only=False)
p1=torch.load("/tmp/opencode/p1.pt",weights_only=False); p3=torch.load("/tmp/opencode/p3.pt",weights_only=False)
gl=nd(d["g_last"].to(DEV)); h=nd(d["h"].to(DEV))
d2=nd(p1["d2"].to(DEV)); d3=nd(p3["d3"].to(DEV)); d4=nd(p3["d4"].to(DEV))
out=torch.zeros(16,128,dtype=torch.bfloat16,device=DEV); hnew=torch.zeros(128,128,dtype=torch.float32,device=DEV)
rtc_compile(open("{ROOT}/k2_glue_final.cpp").read(),"k2_glue_final_kernel","")
torch.npu.synchronize()
args=[struct.pack("<q",p) for p in [gl.data_ptr(),h.data_ptr(),d2.data_ptr(),d3.data_ptr(),d4.data_ptr(),out.data_ptr(),hnew.data_ptr(),0]]+[struct.pack("<i",0)]
for _ in range(3):
    launch_argsarray_engine("k2_glue_final_kernel",1,torch_npu.npu.current_stream().npu_stream,args,0); torch.npu.synchronize()
print(f"P4 out err={{ (out.float()-d['o_ref']).abs().max().item():.2e }} state err={{ (hnew-d['h_ref']).abs().max().item():.2e }}")
'''
r4 = subprocess.run([sys.executable,"-c",p4],capture_output=True,text=True,timeout=240)
print(r4.stdout.strip().splitlines()[-1] if r4.stdout else "P4 no output")
