"""The debug-only stores: wiring, bit-exactness and the debug views.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.29: three buffers a
production call allocates are written for the ``return_intermediates`` views and
read by nothing on the device - Aqk32's masked fp32 copy, A32 and BetaOut.  The
two kernels take a ``debugStores`` int and ``api.py`` passes 0 unless the caller
asked for intermediates, which measured -0.164 ms interleaved at
[1,8192,96,128]/C=64 (tools/probe_dead_store.py).

What has to stay true, and is pinned here:

  * the flag really reaches the kernels (a host-side check of the launch args -
    the device half then only has to prove the arithmetic did not move);
  * the two arms are bit-identical on the consumed outputs and the final state,
    which is the whole safety argument for skipping a store;
  * ``return_intermediates`` still fills all three tiles, so a guard cannot
    quietly turn the debug views into uninitialised memory.

The shape is small: RTC compile dominates, and the store traffic this is about
is a property of the pointer, not of the shape.
"""
from __future__ import annotations

import os
import struct
import sys
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
B, T, H = 1, 512, 4
KW = dict(lower_bound=-1.0, output_final_state=True)


def _skip_unless_supported():
    if api.CHUNK not in api.SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % api.CHUNK)


@pytest.fixture(scope="module")
def inputs():
    _skip_unless_supported()
    device = torch.device("npu:0")
    torch.manual_seed(1129)
    q = (torch.randn(B, T, H, D, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=device) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device) * 0.1
    beta = torch.randn(B, T, H, device=device)
    kw = dict(KW, A_log=torch.linspace(-1.0, 0.2, H, device=device),
              bias=torch.randn(H, D, device=device) * 0.03)
    return q, k, v, g, beta, kw


def _call(inputs, **extra):
    q, k, v, g, beta, kw = inputs
    return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **dict(kw, **extra))


class _LaunchSpy:
    """Records the raw argument blobs of every launch, delegating the launch."""

    def __init__(self):
        self.calls: list[tuple[str, list[bytes]]] = []

    def __enter__(self):
        self._real = api.launch_argsarray_engine

        def spy(name, blocks, stream, args, flag):
            self.calls.append((name, list(args)))
            return self._real(name, blocks, stream, args, flag)

        api.launch_argsarray_engine = spy
        return self

    def __exit__(self, *exc):
        api.launch_argsarray_engine = self._real
        return False

    def last_int(self, kernel):
        """The trailing int32 of the last launch of ``kernel`` (the flag)."""
        for name, args in reversed(self.calls):
            if name == kernel:
                return struct.unpack("<i", args[-1])[0]
        raise AssertionError("no %s launch was recorded" % kernel)


def test_the_flag_reaches_both_kernels(inputs):
    """Production passes 0; KDA_DEBUG_STORES=1 passes 1. Wiring, not timing."""
    for kernel in ("kda_pre_gram_mix", "kda_solve_wu_wide"):
        os.environ.pop("KDA_DEBUG_STORES", None)
        with _LaunchSpy() as spy:
            _call(inputs)
        assert spy.last_int(kernel) == 0, (kernel, "default is not guarded")
        os.environ["KDA_DEBUG_STORES"] = "1"
        try:
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_DEBUG_STORES", None)
        assert spy.last_int(kernel) == 1, (kernel, "the escape hatch is dead")
    torch.npu.synchronize()


def test_the_arms_are_bit_identical(inputs):
    """The safety argument for skipping a store: nothing consumes it."""
    os.environ.pop("KDA_DEBUG_STORES", None)
    o0, s0 = _call(inputs)
    os.environ["KDA_DEBUG_STORES"] = "1"
    try:
        o1, s1 = _call(inputs)
    finally:
        os.environ.pop("KDA_DEBUG_STORES", None)
    torch.npu.synchronize()
    assert torch.equal(o0, o1), "the guard changed the output"
    assert torch.equal(s0, s1), "the guard changed the final state"


def _written_mask(chunk, subb, rows, cols, device):
    """The part of A_inv's fp32 debug view that some kernel actually writes.

    The wide kernel writes the diagonal sub-blocks and blanks the strict upper
    ones; the lower-left coupling block (X21) is formed by the *assemble*
    kernel into Pmid and nothing copies it into A32.  So that block is
    uninitialised device memory - two runs of the same call differ there
    (measured: -2.68e-17 vs 0.0 in the first band).  Pre-existing, and not a
    property of the guard: both runs below have the stores on.  Recorded in
    docs 11.29.
    """
    sub = chunk // subb
    r = torch.arange(rows, device=device).view(-1, 1) // sub
    c = torch.arange(cols, device=device).view(1, -1) // sub
    return (c >= r).expand(rows, cols)


def test_return_intermediates_still_fills_the_debug_tiles(inputs):
    """A guard may not turn the debug views into uninitialised memory."""
    os.environ.pop("KDA_DEBUG_STORES", None)
    _, _, debug = _call(inputs, return_intermediates=True)
    os.environ["KDA_DEBUG_STORES"] = "1"
    try:
        _, _, debug1 = _call(inputs, return_intermediates=True)
    finally:
        os.environ.pop("KDA_DEBUG_STORES", None)
    torch.npu.synchronize()
    mask = _written_mask(api.CHUNK, api.SOLVE_WIDE_SUBB, api.CHUNK, api.CHUNK,
                         debug["A32"].device)
    for name in ("Aqk32", "A32", "Beta"):
        tile = debug[name]
        assert float(tile.float().abs().sum()) > 0.0, name
        if name == "A32":
            assert torch.equal(tile[:, mask], debug1[name][:, mask]), name
            continue
        assert torch.equal(tile, debug1[name]), name
