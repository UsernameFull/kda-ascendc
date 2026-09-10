from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch, torch_npu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT/'src'))
from kda_ascendc_v1.api import get_last_profile, kda_bt16_fwd_ascendc
from kda_bt16 import kda_bt16_fwd
DEV=torch.device('npu:0'); D=128
def sync(): torch.npu.synchronize()
def timed(fn,warm=2,reps=3):
    for _ in range(warm): fn(); sync()
    vals=[]
    for _ in range(reps):
        sync(); t=time.perf_counter(); fn(); sync(); vals.append((time.perf_counter()-t)*1e3)
    vals.sort(); return {'samples_ms':vals,'median_ms':vals[len(vals)//2]}
def main():
    torch.npu.set_device(DEV); torch.manual_seed(1313); torch.empty(1,device=DEV); sync()
    b=int(os.environ.get('CUBE_B','1')); t=int(os.environ.get('CUBE_T','32')); h=int(os.environ.get('CUBE_H','2'))
    q=(torch.randn(b,t,h,D)*.2).to(torch.bfloat16).to(DEV)
    k=(torch.randn_like(q)*.2).to(torch.bfloat16)
    v=(torch.randn_like(q)*.1).to(torch.bfloat16).to(DEV)
    k=k.to(DEV)
    g=torch.randn(b,t,h,D,device=DEV)*.1; beta=torch.randn(b,t,h,device=DEV)
    alog=torch.linspace(-1,.2,h,device=DEV); bias=torch.randn(h,D,device=DEV)*.03; h0=torch.zeros(b,h,D,D,device=DEV)
    kw=dict(A_log=alog,bias=bias,lower_bound=-1.,initial_state=h0,output_final_state=True)
    funcs={
      'cube_separated':lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='cube_separated',**kw),
      'cube_full_d4':lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='cube_full_d4',**kw),
      'mix_aic_1_2':lambda:kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='mix_aic_1_2',**kw),
      'triton':lambda:kda_bt16_fwd(q,k,v,g,beta,initial_state=h0,output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,safe_gate=True,lower_bound=-1.,A_log=alog,dt_bias=bias.reshape(-1)),
    }
    for fn in funcs.values(): fn(); sync()
    timing={}
    profiles={}
    for name, fn in funcs.items():
        timing[name]=timed(fn)
        if name != 'triton' and os.environ.get('KDA_PROFILE','0') == '1':
            fn(); sync()
            profiles[name]=get_last_profile()
    vals={name:fn() for name,fn in funcs.items()}; sync()
    outm,stm=vals['mix_aic_1_2']; outfull,sfull=vals['cube_full_d4']; outcube,scube=vals['cube_separated']; outtri,stri=vals['triton']
    row={'shape':[b,t,h,D],**timing,'profiles':profiles,
      'mix_speedup_vs_cube_full':timing['cube_full_d4']['median_ms']/timing['mix_aic_1_2']['median_ms'],
      'mix_speedup_vs_cube_separated':timing['cube_separated']['median_ms']/timing['mix_aic_1_2']['median_ms'],
      'mix_speedup_vs_triton':timing['triton']['median_ms']/timing['mix_aic_1_2']['median_ms'],
      'mix_output_max_abs_vs_full':float((outm-outfull).abs().max()),
      'mix_state_max_abs_vs_full':float((stm-sfull).abs().max()),
      'mix_output_max_abs_vs_triton':float((outm-outtri).abs().max()),
      'mix_state_max_abs_vs_triton':float((stm-stri).abs().max()),
      'finite':bool(torch.isfinite(outm).all() and torch.isfinite(stm).all())}
    outp=ROOT/'results/S14'; outp.mkdir(exist_ok=True)
    fn=outp/f'bench_mix_{b}_{t}_{h}.json'; fn.write_text(json.dumps(row,indent=2)+os.linesep)
    print(json.dumps(row,indent=2),flush=True)
if __name__=='__main__': main()
