import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

torch.npu.set_device(0)

from kda_ascendc_v1.api import _compile_persistent_scan_cube

_compile_persistent_scan_cube()
print('RTC_OK')
