#!/bin/bash
set -e

# Build launcher.so for AscendC RTC kernel compilation and execution
# Usage: ./build_launcher.sh

cd "$(dirname "$0")"

python -c "
import torch
from torch.utils.cpp_extension import load

launcher = load(
    name='kda_bt16_launcher',
    sources=['launcher.cpp'],
    extra_cflags=['-O3', '-std=c++17', '-I/usr/local/Ascend/cann-9.1.0/aarch64-linux/include'],
    extra_ldflags=['-L/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64', '-lascendcl', '-lacl_rtc'],
    verbose=True
)

print('Build successful. Module exports:')
print(dir(launcher))
"

echo "Done. Launcher built and cached in torch_extensions."
