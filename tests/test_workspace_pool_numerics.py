"""The S1 route's premise, pinned: host work only counts if it is exposed.

Plan section S1 would cache the per-call workspace allocations in a pool to
remove the ~0.33 ms of host time they cost (``tools/probe_workspace_pool.py``).
A host-side saving only reaches the end-to-end number if the host side is
*exposed*, and the repo's own measurement of that (docs 11.26) is a share, not a
budget.  This file pins the budget with an experiment that cannot corrupt
anything: it *adds* a known host delay inside every allocation and watches
``do_bench``.

Measured at [1,8192,96,128], C=64, same process (``tools/probe_host_exposure.py``):

  measured host wall per call   e2e median
   5.844 ms                      10.774 ms   (stock)
   9.341 ms                      10.773 ms   (flat)
  10.903 ms                      10.768 ms   (flat - host wall now exceeds device time)
  13.626 ms                      13.859 ms   (+3.1)
  18.323 ms                      18.368 ms   (+7.6)

so the host side absorbs roughly a device-time's worth of extra work (>=5 ms)
before the e2e number moves at all, and the S1 route's 0.32 ms sits far inside
that slack.  The measured wall is the column that matters: ``time.sleep`` costs
tens of microseconds more than its nominal interval, so a nominal-delay column
understates what was added (an earlier version of this file made exactly that
mistake and concluded the slack was 2.4 ms).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "python"))

import kda_ascendc_v1.api as api  # noqa: E402

D = 128


def _skip_unless_supported():
    if api.CHUNK not in api.SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % api.CHUNK)


def _inputs(b, t, h, device, seed=240922):
    torch.manual_seed(seed)
    q = (torch.randn(b, t, h, D, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, D, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, D, device=device) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, D, device=device) * 0.1
    beta = torch.randn(b, t, h, device=device)
    a_log = torch.linspace(-1.0, 0.2, h, device=device)
    bias = torch.randn(h, D, device=device) * 0.03
    return q, k, v, g, beta, a_log, bias


class _CounterShim:
    """Counts allocation requests; ``delay_ms`` delays once per request.

    The delay is a *busy* spin, not ``time.sleep``: sleep's syscall floor
    (tens of microseconds here) is larger than the amounts this test needs to
    add, and with ~30 allocations per call that floor alone is milliseconds.
    A busy spin is exact and does not fight for the GIL, which is what the api's
    host path wants anyway.
    """

    def __init__(self, delay_ms=0.0):
        self.delay_ms = delay_ms
        self.requests = 0
        self.spun = 0

    def _bump(self):
        self.requests += 1
        if self.delay_ms:
            deadline = time.perf_counter() + self.delay_ms / 1e3
            while time.perf_counter() < deadline:
                pass
            self.spun += 1

    def empty(self, shape, dtype=None, device=None, **kw):
        self._bump()
        return torch.empty(shape, dtype=dtype, device=device, **kw)

    def empty_like(self, other, **kw):
        self._bump()
        return torch.empty_like(other, **kw)

    def __getattr__(self, name):
        return getattr(torch, name)


def _measure(delay_ms, inp, kw, device, host_reps=8, rep_ms=800):
    """Return (do_bench stat, host-wall ms, per-call allocations, out, state).

    The host wall is measured with the shim installed: the *nominal*
    ``delay_ms * per_call`` is not what the call pays (a spin has no syscall
    floor, but the api's own host work still rides along), and the slack claim
    has to be made against measured host time.
    """
    q, k, v, g, beta, a_log, bias = inp
    shim = _CounterShim(delay_ms)
    real = api.torch
    api.torch = shim

    def call():
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, A_log=a_log, bias=bias,
                                        lower_bound=-1.0, output_final_state=True)

    try:
        o, s = call()
        torch.npu.synchronize()
        shim.requests = 0
        call()
        torch.npu.synchronize()
        per_call = shim.requests
        best = 1e9
        for _ in range(host_reps):
            t0 = time.perf_counter()
            call()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        host_wall = best
        torch.npu.synchronize()
        from triton.testing import do_bench
        med = do_bench(call, warmup=100, rep=rep_ms, quantiles=[0.5, 0.2, 0.8])
    finally:
        api.torch = real
    return med, host_wall, per_call, o, s


def test_allocation_count_is_measured_not_assumed():
    """The experiment is meaningless without the per-call count."""
    _skip_unless_supported()
    device = torch.device("npu:0")
    inp = _inputs(1, 2 * api.CHUNK, 2, device)
    _, _, per_call, _, _ = _measure(0.0, inp, {}, device, rep_ms=200)
    assert per_call > 0, "no allocations were counted"


def test_host_exposure_is_max0_added_host_minus_device_time():
    """Pin the mechanism, at a shape where the host is already exposed.

    The S1 route ("remove 0.32 ms of host work") only pays if the host side is
    *hidden*, and whether it is hidden is a property of the shape, not of the
    route.  The two measurements in `tools/probe_host_exposure.py`:

      [1,8192,96,128]  host wall 5.84 ms, device 10.77 ms -> slack ~= device-host,
                       and the 0.32 ms arm is invisible (e2e flat while the
                       measured host wall grows to 10.9 ms).
      [1,1024,8,128]   host wall ~= device -> no slack, and any added host time
                       shows up in the e2e almost one-for-one.

    The second case is the cheap one, and it is what this test pins: e2e moves
    by the added host time when the host is already at the device's level.  That
    is also the interesting half of the finding - the host side is the
    bottleneck at small shapes (the ~74 launches are shape-independent), so a
    host-side saving that is invisible at the production shape is not
    universally invisible.
    """
    _skip_unless_supported()
    device = torch.device("npu:0")
    inp = _inputs(1, 16 * api.CHUNK, 8, device)
    med0, host0, per_call, o0, s0 = _measure(0.0, inp, {}, device, rep_ms=300)
    # This shape is host-bound: the stock host wall is at (or above) the e2e
    # number.  If it is not, the assertion below would be checking nothing.
    if host0 < med0[0] - 0.2:
        pytest.skip("this geometry is device-bound (host wall %.3f vs e2e %.3f), so it "
                    "cannot show the exposed case" % (host0, med0[0]))
    med1, host1, _, o1, s1 = _measure(0.4 / max(per_call, 1), inp, {}, device, rep_ms=300)
    added = host1 - host0
    moved = med1[0] - med0[0]
    assert added > 0.2, "the shim added no measurable host time: the test proves nothing"
    assert torch.equal(o0, o1), "the shim changed the output"
    assert torch.equal(s0, s1), "the shim changed the state"
    assert 0.5 * added <= moved <= 1.5 * added + 0.1, (
        "added %.3f ms of host work per call but the e2e moved %.3f ms; this shape is "
        "supposed to be host-bound with an exposed host (docs 11.28)"
        % (added, moved))
