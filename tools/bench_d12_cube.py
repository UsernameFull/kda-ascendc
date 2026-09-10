from __future__ import annotations
import json,struct,sys,time,os
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parents[1]; EXT=ROOT/'build/S02_clean/torch_extensions/kda_ascendc_v1_launcher'; sys.path.insert(0,str(EXT))
from kda_ascendc_v1_launcher import rtc_compile, launch_argsarray_engine
DEV=torch.device('npu:0'); M=16; BV=64; D=128; NV=2
def ptr(x): return struct.pack('<Q',int(x.data_ptr()))
def si(x): return struct.pack('<i',int(x))
def launch(name,blocks,args):
 launch_argsarray_engine(name,blocks,torch_npu.npu.current_stream().npu_stream,args,0)
def timed(name,blocks,args,warm=3,reps=10):
 for _ in range(warm): launch(name,blocks,args); torch.npu.synchronize()
 vals=[]
 for _ in range(reps):
  torch.npu.synchronize(); t=time.perf_counter(); launch(name,blocks,args); torch.npu.synchronize(); vals.append((time.perf_counter()-t)*1e3)
 vals.sort(); return {'samples_ms':vals,'median_ms':vals[len(vals)//2]}
def main():
 torch.npu.set_device(DEV); torch.manual_seed(122); torch.empty(1,device=DEV); torch.npu.synchronize()
 BH,NT=8,1; tasks=BH*NV; C=BH*NT
 W=(torch.randn(C,M,D)*.05).to(torch.bfloat16).to(DEV); Q=(torch.randn_like(W)*.05).to(torch.bfloat16).to(DEV); S=(torch.randn(tasks,BV,D)*.05).to(torch.bfloat16).to(DEV)
 d1=torch.empty(tasks,NT,M,BV,dtype=torch.float32,device=DEV); d2=torch.empty_like(d1)
 rtc_compile((ROOT/'kernels/v1/k2_d12.cpp').read_text(),'kda_k2_d12_kernel',''); rtc_compile((ROOT/'kernels/v1/k2_d12_cube.cpp').read_text(),'kda_k2_d12_cube_kernel','')
 common=[si(BH),si(NT),si(NV),si(0)]
 a=[ptr(W),ptr(Q),ptr(S),ptr(d1),ptr(d2)]+common
 b=[ptr(W),ptr(Q),ptr(S),ptr(d1),ptr(d2)]+common
 old=timed('kda_k2_d12_kernel',tasks,a); cube=timed('kda_k2_d12_cube_kernel',tasks,b)
 out={'suite':'d12_cube_bench','shape':{'BH':BH,'NT':NT,'NV':NV,'BV':BV,'D':D},'aiv_scalar':old,'cube':cube,'speedup':old['median_ms']/cube['median_ms']}
 (ROOT/'results/S11/d12_cube_bench.json').write_text(json.dumps(out,indent=2)+chr(10)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
