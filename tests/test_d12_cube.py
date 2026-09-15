from __future__ import annotations
import json, os, struct, sys
from pathlib import Path
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT=Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
EXT=ROOT/'build/S02_clean/torch_extensions/kda_ascendc_v1_launcher'
if not EXT.exists():
    pytest.skip("AscendC launcher extension is not built", allow_module_level=True)
sys.path.insert(0,str(EXT))
from kda_ascendc_v1_launcher import launch_argsarray_engine
sys.path.insert(0,str(ROOT/'python'))
from kda_ascendc_v1.api import _rtc
DEV=torch.device('npu:0'); M=16; BV=64; D=128; NV=2
def ptr(x): return struct.pack('<Q',int(x.data_ptr()))
def si(x): return struct.pack('<i',int(x))
def main():
 torch.npu.set_device(DEV); torch.manual_seed(121); torch.empty(1,device=DEV); torch.npu.synchronize()
 BH,NT=int(os.environ.get("D12_BH","1")),1; tasks=BH*NV; C=BH*NT
 W=(torch.randn(C,M,D)*.05).to(torch.bfloat16).to(DEV)
 Q=(torch.randn(C,M,D)*.05).to(torch.bfloat16).to(DEV)
 S=(torch.randn(tasks,BV,D)*.05).to(torch.bfloat16).to(DEV)
 d1=torch.full((tasks,NT,M,BV),-777.,dtype=torch.float32,device=DEV)
 d2=torch.full_like(d1,-777.)
 _rtc('kernels/v1/k2_d12_cube.cpp','kda_k2_d12_cube_kernel')
 args=[ptr(W),ptr(Q),ptr(S),ptr(d1),ptr(d2),si(BH),si(NT),si(NV),si(0)]
 launch_argsarray_engine('kda_k2_d12_cube_kernel',tasks,torch_npu.npu.current_stream().npu_stream,args,0)
 torch.npu.synchronize()
 Wc,Qc,Sc=W.cpu().float(),Q.cpu().float(),S.cpu().float()
 r1=torch.empty(tasks,M,BV); r2=torch.empty_like(r1)
 for task in range(tasks):
  bh=task//NV; c=bh*NT
  st=Sc[task]
  r1[task]=Wc[c]@st.T
  r2[task]=Qc[c]@st.T
 got1,got2=d1.cpu()[:,0],d2.cpu()[:,0]
 e1=(got1-r1).abs(); e2=(got2-r2).abs()
 canary=bool(torch.all(d1.cpu()!=-777.)) and bool(torch.all(d2.cpu()!=-777.))
 out={'suite':'d12_cube_microkernel','status':'passed' if float(e1.max())<0.03 and float(e2.max())<0.03 and canary else 'failed','max_abs_d1':float(e1.max()),'max_abs_d2':float(e2.max()),'canary':canary,'finite':bool(torch.isfinite(got1).all() and torch.isfinite(got2).all())}
 (ROOT/'results/S11').mkdir(exist_ok=True); (ROOT/'results/S11/d12_cube_check.json').write_text(json.dumps(out,indent=2)+chr(10)); print(json.dumps(out,sort_keys=True)); assert out['status']=='passed'
if __name__=='__main__': main()
