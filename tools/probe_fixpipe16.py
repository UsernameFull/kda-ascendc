"""Which fixpipe shape faults: the bisect for k1_solve_assemble4's stores.

One launch per arm on a small grid; the driver syncs and prints after each, so
the first sync that throws names the failing shape.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_fixpipe16.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

DEV = torch.device("npu:0")
MODES = [
    (0, "16x16 RM at 1 KB base"),
    (1, "16x16 RM at base + 1040 el (2080 B)"),
    (2, "NZ -> L1"),
    (3, "trio ndNum=3 srcNd=1 dstNd=1040"),
    (4, "control 32x32 RM at 4 KB base"),
    (5, "16x16 RM at base + 2064 el (4128 B)"),
    (6, "pair ndNum=2 at 2048 el"),
]


def main() -> None:
    api._rtc("kernels/v1/k1_fixpipe16_probe.cpp", "kda_fixpipe16_probe")
    out = torch.zeros(8 * 4096, dtype=torch.bfloat16, device=DEV)
    stream = torch_npu.npu.current_stream()
    for mode, label in MODES:
        args = api._pack_ptrs([out]) + [api._i(mode)]
        try:
            api.launch_argsarray_engine("kda_fixpipe16_probe", 8,
                                        stream.npu_stream, args, 0)
            torch.npu.synchronize()
            print("mode %d %-38s OK" % (mode, label), flush=True)
        except Exception as e:
            print("mode %d %-38s FAULT: %s" % (mode, label, str(e)[:160]),
                  flush=True)
            raise
        time.sleep(0.2)


if __name__ == "__main__":
    main()
