"""How much of one e2e call is host work?  A launch-suppressed call tells it.

The blocked-scan e2e runs (tools/probe_pg_cumsum_scan_e2e.py, 4 arms) showed
the wall tracking the pre_gram device span 1:1 in one process (scan -0.115 ->
wall -0.10) and not at all in another (everything pinned at ~10.96 ms while the
device span moved 0.54 ms), which reads as wall = max(host floor, device).  This
tool measures the floor: it times the same call twice, once normally and once
with every ``api._launch`` swallowed by a no-op, so the second number is the
host-side work (allocations, packing, torch ops, argument marshalling) with no
kernel execution.  Launch counts come from the api's own ``_LAUNCH_COUNTS``.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_host_floor.py
"""
from __future__ import annotations

import faulthandler
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(1800, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
REPS = 4


def timed(call, reps):
    host, wall = [], []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out, st = call()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        wall.append((t2 - t0) * 1e3)
    return min(host), min(wall)


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
    call = lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    call()
    call()

    def nolaunch():
        orig = api._launch
        api._launch = lambda kernel, blocks, args, stream: None
        try:
            return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        finally:
            api._launch = orig

    def spy_call(mode):
        orig = api._launch

        def spy(kernel, blocks, args, stream):
            if mode == "noop":
                return None
            return orig(kernel, blocks, args, stream)

        api._launch = spy
        try:
            return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        finally:
            api._launch = orig

    h1, w1 = timed(call, REPS)
    api._LAUNCH_COUNTS.clear()
    out, st = call()
    torch.npu.synchronize()
    counts = dict(api._LAUNCH_COUNTS)
    total = sum(counts.values())
    h2, w2 = timed(nolaunch, REPS)
    h3, w3 = timed(lambda: spy_call("pass"), REPS)
    h4, w4 = timed(lambda: spy_call("noop"), REPS)
    print("normal   call: host enqueue MIN %.3f  wall MIN %.3f ms" % (h1, w1), flush=True)
    print("no-launch call: host enqueue MIN %.3f  wall MIN %.3f ms" % (h2, w2), flush=True)
    print("spy-pass call: host enqueue MIN %.3f  wall MIN %.3f ms" % (h3, w3), flush=True)
    print("spy-noop call: host enqueue MIN %.3f  wall MIN %.3f ms" % (h4, w4), flush=True)
    print("launches per call: %d (%s)" % (
        total, ", ".join("%s x%d" % (n, c) for n, c in sorted(counts.items()))),
        flush=True)


if __name__ == "__main__":
    main()
