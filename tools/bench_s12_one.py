from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch, torch_npu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python')); sys.path.insert(0,str(ROOT/'src'))
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc
from kda_bt16 import kda_bt16_fwd
DEV=torch.device('npu:0'); D=128

def sync(): torch.npu.synchronize()
def timed(fn,warm=1,reps=3):
    for _ in range(warm): fn(); sync()
    vals=[]
    for _ in range(reps):
        sync(); t=time.perf_counter(); fn(); sync(); vals.append((time.perf_counter()-t)*1e3)
    vals.sort(); return {'samples_ms':vals,'median_ms':vals[len(vals)//2]}
def main():
    torch.npu.set_device(DEV); torch.manual_seed(1312); torch.empty(1,device=DEV); sync()
    b=int(os.environ.get('CUBE_B','1')); t=int(os.environ.get('CUBE_T','32')); h=int(os.environ.get('CUBE_H','2'))
    q=(torch.randn(b,t,h,D)*.2).to(torch.bfloat16).to(DEV); k=(torch.randn_like(q)*.2).to(torch.bfloat16).to(DEV); v=(torch.randn_like(q)*.1).to(torch.bfloat16).to(DEV)
    g=torch.randn(b,t,h,D,device=DEV)*.1; beta=torch.randn(b,t,h,device=DEV); alog=torch.linspace(-1,.2,h,device=DEV); bias=torch.randn(h,D,device=DEV)*.03; h0=torch.zeros(b,h,D,D,device=DEV)
    kw=dict(A_log=alog,bias=bias,lower_bound=-1.,initial_state=h0,output_final_state=True)
    sep=lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='separated',**kw)
    cube=lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='cube_separated',**kw)
    pers=lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='persistent',**kw)
    tri=lambda:kda_bt16_fwd(q,k,v,g,beta,initial_state=h0,output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,safe_gate=True,lower_bound=-1.,A_log=alog,dt_bias=bias.reshape(-1))
    sep(); sync(); cube(); sync(); pers(); sync(); tri(); sync()
    ts=timed(sep); tc=timed(cube); tp=timed(pers); tt=timed(tri)
    so,ss=sep(); co,cs=cube(); po,ps=pers(); to,st=tri(); sync()
    row={'shape':[b,t,h,D],'separated':ts,'cube_separated':tc,'persistent':tp,'triton':tt,
      'cube_speedup_vs_separated':ts['median_ms']/tc['median_ms'],'cube_speedup_vs_triton':tt['median_ms']/tc['median_ms'],
      'triton_speedup_vs_cube':tc['median_ms']/tt['median_ms'],'cube_output_max_abs_vs_triton':float((co-to).abs().max()),
      'cube_state_max_abs_vs_triton':float((cs-st).abs().max()),'finite':bool(torch.isfinite(co).all() and torch.isfinite(cs).all() and torch.isfinite(to).all() and torch.isfinite(st).all())}
    outp=ROOT/'results/S12'; outp.mkdir(exist_ok=True); (outp/f'bench_{b}_{t}_{h}.json').write_text(json.dumps(row,indent=2)+os.linesep); print(json.dumps(row,indent=2),flush=True)
if __name__=='__main__': main()
