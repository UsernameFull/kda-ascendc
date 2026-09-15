import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from kda_ascendc_v1.experimental import get_last_profile, kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc
torch.npu.set_device(0); dev=torch.device('npu:0'); D=128
for b,t,h in [(1,16,2),(1,64,2)]:
    torch.manual_seed(9000+b+t+h)
    q=(torch.randn(b,t,h,D,device=dev)*.2).to(torch.bfloat16)
    k=(torch.randn(b,t,h,D,device=dev)*.2).to(torch.bfloat16)
    v=(torch.randn(b,t,h,D,device=dev)*.1).to(torch.bfloat16)
    g=torch.randn(b,t,h,D,device=dev)*.1; beta=torch.randn(b,t,h,device=dev)
    al=torch.linspace(-1,.2,h,device=dev); bias=torch.randn(h,D,device=dev)*.03
    init=torch.zeros(b,h,D,D,device=dev)
    kw=dict(A_log=al,bias=bias,lower_bound=-1.,initial_state=init,output_final_state=True)
    ref=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='persistent',**kw); torch.npu.synchronize()
    got=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='persistent_scan_cube',**kw); torch.npu.synchronize()
    oe=float((got[0]-ref[0]).abs().max().cpu()); se=float((got[1]-ref[1]).abs().max().cpu())
    print('shape',b,t,h,'out_error',oe,'state_error',se,'profile',get_last_profile())
    assert oe < 1e-2 and se < 1e-2
