"""Drive one KDA forward so `msprof op` can capture a named kernel.

`msprof op` profiles an *application* (an executable), not a library call, so
instruction-level collection needs a script that runs one forward pass and
exits.  This is that script; the collection itself is a separate step:

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 \\
    msprof op --application="python3 tools/msprof_op_kda.py" \\
              --aic-metrics=PipeUtilization --kernel-name=kda_solve_assemble \\
              --launch-count=1 --output=/tmp/msprof_out

The shapes match every other probe in this repo ([1, 8192, 96, 128], C=64) so
the numbers are comparable with the docs.  ``--kernels`` only exists to warm
the pipeline the way a real call does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96


def main() -> None:
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)
    warm = int(os.environ.get("KDA_MSOPP_WARM", "0"))
    for _ in range(warm):
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
    out, state = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    print("done: out mean|.| %.5f, state mean|.| %.5f"
          % (float(out.float().abs().mean()), float(state.float().abs().mean())),
          flush=True)


if __name__ == "__main__":
    main()
