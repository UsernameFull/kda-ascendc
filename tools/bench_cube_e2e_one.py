from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch, torch_npu
ROOT=Path('/workspace/kda_ascendc_luna_20260905')
sys.path.insert(0,str(ROOT/'python'))
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc
DEV=torch.device('npu:0'); D=128

def sync(): torch.npu.synchronize()
def timed(fn,warm=None,reps=None):
    warm = int(os.environ.get('CUBE_WARM','2')) if warm is None else warm
    reps = int(os.environ.get('CUBE_REPS','5')) if reps is None else reps
    for _ in range(warm): fn(); sync()
    vals=[]
    for _ in range(reps):
        sync(); t=time.perf_counter(); fn(); sync(); vals.append((time.perf_counter()-t)*1e3)
    vals.sort(); return {'samples_ms':vals,'median_ms':vals[len(vals)//2]}

def main():
    torch.npu.set_device(DEV); torch.manual_seed(1309)
    torch.empty(1,device=DEV); sync()
    b=int(os.environ.get('CUBE_B','1')); t=int(os.environ.get('CUBE_T','32')); h=int(os.environ.get('CUBE_H','2'))
    q=(torch.randn(b,t,h,D)*.2).to(torch.bfloat16).to(DEV)
    k=(torch.randn_like(q)*.2).to(torch.bfloat16).to(DEV)
    v=(torch.randn_like(q)*.1).to(torch.bfloat16).to(DEV)
    g=torch.randn(b,t,h,D,device=DEV)*.1; beta=torch.randn(b,t,h,device=DEV)
    alog=torch.linspace(-1,.2,h,device=DEV); bias=torch.randn(h,D,device=DEV)*.03; h0=torch.zeros(b,h,D,D,device=DEV)
    common=dict(A_log=alog,bias=bias,lower_bound=-1.,initial_state=h0,output_final_state=True)
    fs=lambda mode: (lambda: kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode=mode,**common))
    f_sep=fs('separated'); f_cube=fs('cube_separated'); f_pers=fs('persistent')
    f_sep(); sync(); f_cube(); sync(); f_pers(); sync()
    sp=timed(f_sep); cp=timed(f_cube); pp=timed(f_pers)
    oo,ss=f_sep(); co,cs=f_cube(); po,ps=f_pers(); sync()
    row={'shape':[b,t,h,D],'separated':sp,'cube_separated':cp,'persistent':pp,
         'cube_speedup_vs_separated':sp['median_ms']/cp['median_ms'],
         'persistent_speedup_vs_separated':sp['median_ms']/pp['median_ms'],
         'cube_output_max_abs_vs_separated':float((co-oo).abs().max()),
         'cube_state_max_abs_vs_separated':float((cs-ss).abs().max()),
         'persistent_output_max_abs_vs_separated':float((po-oo).abs().max()),
         'persistent_state_max_abs_vs_separated':float((ps-ss).abs().max()),
         'finite':bool(torch.isfinite(co).all() and torch.isfinite(cs).all())}
    outp=ROOT/'results/S12'; outp.mkdir(exist_ok=True)
    fn=outp/f'cube_e2e_{b}_{t}_{h}.json'; fn.write_text(json.dumps(row,indent=2)+os.linesep)
    print(json.dumps(row,indent=2),flush=True)
if __name__=='__main__': main()
