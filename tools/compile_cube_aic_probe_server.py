import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
torch.npu.set_device(0)
from kda_ascendc_v1.api import _rtc

_rtc('kernels/v1/k2_triton_cube_aic_probe.cpp', 'kda_k2_triton_cube_aic_probe')
print('RTC_OK', flush=True)
