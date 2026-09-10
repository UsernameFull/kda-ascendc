import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
torch.npu.set_device(0)
import kda_ascendc_v1.api as api
api._compile_all()
print('RTC_ALL_OK')
