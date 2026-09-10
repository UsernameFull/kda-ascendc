import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
torch.npu.set_device(0)
from kda_ascendc_v1.api import rtc_compile

rtc_compile((ROOT / 'kernels/v1/k2_triton_cube_aic_probe.cpp').read_text(),
            'kda_k2_triton_cube_aic_probe', '')
print('RTC_OK', flush=True)
