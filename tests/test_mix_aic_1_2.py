import os
import sys
from pathlib import Path
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT=Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'python'))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
torch.npu.set_device(0); dev=torch.device('npu:0')
def check(b,t,h):
    torch.manual_seed(7000+b+t+h)
    q=(torch.randn(b,t,h,128,device=dev)*.2).to(torch.bfloat16)
    k=(torch.randn_like(q)*.2).to(torch.bfloat16)
    v=(torch.randn_like(q)*.1).to(torch.bfloat16)
    g=torch.randn(b,t,h,128,device=dev)*.1
    beta=torch.randn(b,t,h,device=dev)
    alog=torch.linspace(-1,.2,h,device=dev)
    bias=torch.randn(h,128,device=dev)*.03
    h0=torch.zeros(b,h,128,128,device=dev)
    kw=dict(A_log=alog,bias=bias,lower_bound=-1.,initial_state=h0,output_final_state=True)
    from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc
    out,st,dbg=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='mix_aic_1_2',return_intermediates=True,**kw)
    torch.npu.synchronize()
    ref,rs=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='cube_full_d4',**kw)
    torch.npu.synchronize()
    assert dbg['d4_full'].shape == (b*h,128,128)
    assert bool(torch.isfinite(dbg['d4_full']).all())
    assert bool(torch.isfinite(out).all() and torch.isfinite(st).all())
    e1=float((out-ref).abs().max().cpu()); e2=float((st-rs).abs().max().cpu())
    assert e1 <= 1e-6 and e2 <= 1e-7, (e1,e2)
    return e1,e2
print({'t16_h2':check(1,16,2),'t64_h2':check(1,64,2),'t16_h32':check(1,16,32),'status':'passed'})
