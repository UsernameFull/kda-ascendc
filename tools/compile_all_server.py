import sys
import torch
import torch_npu
sys.path.insert(0, '/workspace/kda_ascendc_luna_20260905/python')
torch.npu.set_device(0)
import kda_ascendc_v1.api as api
api._compile_all()
print('RTC_ALL_OK')
