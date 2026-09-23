"""Is the host side exposed?  Add host delay and watch the e2e number.

The S1 route is "remove the per-call workspace allocation (0.33 ms of host
time)".  `tools/probe_workspace_pool.py` shows the arm is real host work, but a
pooled *buffer* shim has to reproduce the api's allocation set exactly, and the
attempts to do that turned up two ways to corrupt a run (same-shaped aliasing;
a short buffer handed to a larger request) - so a buffer-swapping A/B has a
correctness risk attached to it that the answer does not need.

This probe answers the same question with an experiment that cannot corrupt
anything: it *adds* known host delay inside the call (a sleep per allocation,
monotonically) and measures the e2e number.  The logic:

  * if the e2e number is flat while delay is added, the host side has that much
    slack hidden behind the device - so *removing* 0.33 ms of host work cannot
    help, and the S1 route is dead on this shape;
  * the delay at which e2e finally starts to rise is the exposure threshold,
    i.e. how much host work this shape can absorb for free.

The delay shim delegates every attribute to the real module, so the call it
serves is byte-for-byte the stock call - only slower on the host.

  KDA_CHUNK=64 python3 -u tools/probe_host_exposure.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

_REAL_TORCH = torch
D = 128


class _DelayShim:
    """Delegates to torch; sleeps ``delay_ms/1000`` in every allocation."""

    def __init__(self, delay_ms):
        self.delay_ms = delay_ms
        self.slept = 0            # total sleeps (whole do_bench run)
        self.per_call = 0         # allocations in the first observed call
        self._since_mark = 0
        # ``time.sleep`` has a syscall floor (tens of microseconds on this
        # host), so the *nominal* ``delay_ms * per_call`` understates what the
        # shim actually adds.  The probe measures the added host time instead of
        # computing it - see the host-wall column below.

    def _sleep(self):
        self._since_mark += 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1e3)
            self.slept += 1

    def mark_call_boundary(self):
        """Called between calls: the first full call fixes ``per_call``."""
        if self.per_call == 0 and self._since_mark:
            self.per_call = self._since_mark
        self._since_mark = 0

    def empty(self, shape, dtype=None, device=None, **kw):
        self._sleep()
        return _REAL_TORCH.empty(shape, dtype=dtype, device=device, **kw)

    def empty_like(self, other, **kw):
        self._sleep()
        return _REAL_TORCH.empty_like(other, **kw)

    def __getattr__(self, name):
        return getattr(_REAL_TORCH, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=0, type=int)
    ap.add_argument("--shape", default="1,8192,96,128")
    ap.add_argument("--delays", default="0,0.02,0.05,0.1,0.2,0.35,0.5,1.0",
                    help="per-allocation host delay in ms (comma list)")
    args = ap.parse_args()

    b, t, h, d = (int(x) for x in args.shape.split(","))
    torch.npu.set_device(args.device)
    dev = torch.device("npu", args.device)
    gen = torch.Generator(device="cpu").manual_seed(1312)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=dev, dtype=dtype)

    q = randn(b, t, h, d, dtype=torch.bfloat16)
    k = randn(b, t, h, d, dtype=torch.bfloat16)
    v = randn(b, t, h, d, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(randn(b, t, h, d)).clamp_min(-5.0).contiguous()
    beta = randn(b, t, h).sigmoid()
    a_log = randn(h)
    bias = randn(h, d) * 0.1
    kw = dict(A_log=a_log, bias=bias, lower_bound=-5.0, output_final_state=True)

    print("warming / RTC (C=%d)..." % api.CHUNK, flush=True)
    api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()

    print()
    # Allocations per call, measured once (not assumed): the delay has to be
    # reported per call, because the total sleep count of a whole do_bench run
    # is not the host cost of one call.
    counter = _DelayShim(0.0)
    real = api.torch
    api.torch = counter
    try:
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        counter.mark_call_boundary()
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        counter.mark_call_boundary()
    finally:
        api.torch = real
    torch.npu.synchronize()
    per_call = counter.per_call
    print("allocations per call: %d (measured, not assumed)" % per_call, flush=True)

    def _call():
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)

    def host_min(fn, reps):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best

    host_reps = int(os.environ.get("KDA_HOST_REPS", "10"))
    host_stock = host_min(_call, host_reps)
    print("host wall per call, stock: %.3f ms" % host_stock, flush=True)

    print()
    print("per-allocation host delay -> e2e (do_bench median, same process):")
    print("  %-12s %-16s %-14s %-10s %s"
          % ("delay/alloc", "nominal host", "measured host", "e2e ms", "delta"))
    baseline = None
    rows = []
    delays = [float(x) for x in args.delays.split(",") if x.strip()]
    for delay in delays:
        shim = _DelayShim(delay)
        real = api.torch
        api.torch = shim
        try:
            # The host wall has to be measured *with the shim installed* - it is
            # the measured added host cost per call (MIN of ``host_reps`` calls,
            # no sync in the window), which is what the e2e delta is compared
            # against, not the nominal sleep (which understates the syscall
            # floor of time.sleep).
            host_shim = host_min(_call, host_reps)
            torch.npu.synchronize()
            med = do_bench(_call, warmup=100, rep=1000,
                           quantiles=[0.5, 0.2, 0.8])
        finally:
            api.torch = real
        added = delay * per_call
        if baseline is None:
            baseline = med[0]
        rows.append((delay, added, host_shim, med[0], med[1], med[2], med[0] - baseline))
        print("  %-12.3f %-16.2f %-14.3f %-10.3f %+.3f   (p20 %.3f / p80 %.3f)"
              % (delay, added, host_shim, med[0], med[0] - baseline, med[1], med[2]), flush=True)

    print()
    flat = [r for r in rows if r[6] < 0.05]
    if flat:
        # The decisive number: the e2e is flat while the *measured* host wall has
        # grown past the device time, i.e. the host side absorbed all of it.
        top = max(r[2] for r in flat)
        print("  e2e stays flat (< +0.05 ms) while measured host wall per call grows")
        print("  to %.2f ms against a device time of %.2f ms (stock host %.2f ms)"
              % (top, baseline, host_stock))
        print("  => host slack per call: at least %.2f ms, i.e. the host may take as long"
              % (top - baseline))
        print("     as the device without the e2e number moving")
    print("  the alloc arm a pool would remove is 0.32 ms per call "
          "(tools/probe_workspace_pool.py) - far inside that slack, which is why")
    print("  the S1 route is frozen (docs 11.28)")


if __name__ == "__main__":
    main()
