"""Reusable raw-launch harness for AscendC kernels built via ascendc cmake."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/usr/local/Ascend/cann-9.1.0/python/site-packages")
import numpy as np
import acl
from mskl.launcher import get_kernel_from_binary
from mskl.launcher.context import context

ACLAB_ROOT = Path(__file__).resolve().parent
# `ascendc_library(k2dev STATIC hello_kernel.cpp)` emits the raw device ELF here.
DEFAULT_OBJ = ACLAB_ROOT / "build" / "k2dev_aiv_device_dir" / "device_aiv.o"


class _TO:
    blockdim = 1
    workspace_size = 0
    tiling_key = 0
    workspace = np.zeros(75 * 1024 * 1024, dtype=np.uint8)
    tiling_data = np.array([], dtype=np.uint8)


context.tiling_output = _TO()
context.op_type = "manual"

H2D, D2H = 1, 2


class RawKernel:
    def __init__(self, o_path, kernel_type="vec"):
        self.k = get_kernel_from_binary(o_path, kernel_type=kernel_type, tiling_key=0)

    def __call__(self, *int_addrs, blockdim=1):
        self.k.launch(*[int(a) for a in int_addrs], blockdim=blockdim)


def dmalloc(nbytes):
    p, r = acl.rt.malloc(nbytes, 2)
    assert r == 0, f"malloc fail {r}"
    return int(p)


def h2d(dev_ptr, arr):
    assert acl.rt.memcpy(dev_ptr, arr.nbytes, arr.ctypes.data, arr.nbytes, H2D) == 0


def d2h(arr, dev_ptr):
    assert acl.rt.memcpy(arr.ctypes.data, arr.nbytes, dev_ptr, arr.nbytes, D2H) == 0


acl.init()
acl.rt.set_device(0)
ctx = acl.rt.create_context(0)

if __name__ == "__main__":
    obj = os.environ.get("KERNEL_OBJ", str(DEFAULT_OBJ))
    if not os.path.isfile(obj):
        raise SystemExit(
            f"device object not found: {obj}\n"
            f"build it first:  cmake -B build -S . && cmake --build build\n"
            f"or point KERNEL_OBJ at an existing device_aiv.o"
        )
    k = RawKernel(obj)
    N = 1024
    x = np.arange(N, dtype=np.float32)
    x_dev, y_dev = dmalloc(N * 4), dmalloc(N * 4)
    h2d(x_dev, x)
    k(x_dev, y_dev, 0, 0, blockdim=1)
    y = np.zeros(N, dtype=np.float32)
    d2h(y, y_dev)
    print("y[:8] =", y[:8], " expect", (2 * x[:8]))
    print("MATCH:", np.allclose(y, 2 * x))
