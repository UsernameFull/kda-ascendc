from __future__ import annotations
import json, os, sys
from pathlib import Path
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT=Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'python'))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc
DEV=torch.device('npu:0'); D=128

def main():
    torch.npu.set_device(DEV); torch.manual_seed(1212); torch.empty(1,device=DEV); torch.npu.synchronize()
    cases=[(1,32,2),(2,1024,4),(2,4096,8),(1,8192,32)]
    rows=[]
    for b,t,h in cases:
        q=(torch.randn(b,t,h,D)*.2).to(torch.bfloat16).to(DEV)
        k=(torch.randn_like(q)*.2).to(torch.bfloat16).to(DEV)
        v=(torch.randn_like(q)*.1).to(torch.bfloat16).to(DEV)
        g=torch.randn(b,t,h,D,device=DEV)*.1; beta=torch.randn(b,t,h,device=DEV)
        alog=torch.linspace(-1,.2,h,device=DEV); bias=torch.randn(h,D,device=DEV)*.03
        h0=torch.randn(b,h,D,D,device=DEV)*.01
        kw=dict(A_log=alog,bias=bias,lower_bound=-1.,initial_state=h0,output_final_state=True)
        so,ss=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='separated',**kw)
        co,cs=kda_bt16_fwd_ascendc(q,k,v,g,beta,k2_mode='cube_separated',**kw)
        torch.npu.synchronize()
        oe=float((co-so).abs().max()); se=float((cs-ss).abs().max())
        rows.append({'shape':[b,t,h,D],'output_max_abs':oe,'state_max_abs':se,'finite':bool(torch.isfinite(co).all() and torch.isfinite(cs).all())})
        print(rows[-1],flush=True)
    out={'suite':'cube_separated_regression','status':'passed' if all(r['finite'] and r['output_max_abs']<1e-3 and r['state_max_abs']<1e-4 for r in rows) else 'failed','rows':rows}
    (ROOT/'results/S12').mkdir(exist_ok=True); (ROOT/'results/S12/cube_separated_check.json').write_text(json.dumps(out,indent=2)+os.linesep)
    print(json.dumps(out,indent=2)); assert out['status']=='passed'
if __name__=='__main__': main()
