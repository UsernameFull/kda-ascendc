"""The wide solve's A16-store ablation knob: wired, positioned, and inert.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.34 priced the wide
kernel's parent-tile writes (the two diagonal sub-blocks of A_inv plus the
strict-upper blank the Cube solve reads as zeros) by ablation:
``tools/probe_solve_a16_ablation.py`` runs mode 0/1/2 in one process, and the
answer was that those stores are not on the solve stage's critical path.

The knob is a runtime int32 the kernel takes between the chunk count and the
debug flag, so the probe could flip it per call without a recompile.  What has
to stay true is the wiring, and only the wiring: a mode in the wrong slot is
read by the device as the chunk count (or as the debug flag), and both failures
are silent - the solve computes a different A_inv, it does not raise.  The
three arms' timings belong to the probe.

Mode 2 leaves ``kda_solve_wu_cube_kernel`` a partially written operand *by
construction*, so what is pinned here is that production passes 0 - never that
a non-zero mode is safe to run.
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
B, T, H = 1, 512, 16
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


def _call(inputs):
    q, k, v, g, beta, kw = inputs
    return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)


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

    def wide_trailer(self, kernel="kda_solve_wu_wide"):
        """Every launch's trailing (C, a16Mode, debugStores) int triple."""
        out = []
        for name, args in self.calls:
            if name == kernel:
                out.append(tuple(struct.unpack("<i", a)[0] for a in args[-3:]))
        if not out:
            raise AssertionError("no %s launch was recorded" % kernel)
        return out


def test_production_passes_mode_zero(inputs):
    """The default is the store-everything arm, on every slice of the stage."""
    os.environ.pop("KDA_SOLVE_A16_MODE", None)
    os.environ.pop("KDA_DEBUG_STORES", None)
    with _LaunchSpy() as spy:
        _call(inputs)
    c_real = B * H * (T // api.CHUNK)
    c_solve = -(-c_real // api.SOLVE_WIDE_NCH) * api.SOLVE_WIDE_NCH
    trailers = spy.wide_trailer()
    for _, mode, debug in trailers:
        assert mode == 0, "production is not the ablation's mode 0"
        assert debug == 0, "the debug flag moved"
    # Every chunk is solved exactly once across the slices.
    assert sum(c for c, _, _ in trailers) == c_solve, trailers
    torch.npu.synchronize()


def test_the_mode_is_read_per_call_not_frozen(inputs):
    """A probe flips this between two arms of one process: it has to move."""
    for mode in (1, 2, 0):
        os.environ["KDA_SOLVE_A16_MODE"] = str(mode)
        try:
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_SOLVE_A16_MODE", None)
        trailers = spy.wide_trailer()
        assert [t[1] for t in trailers] == [mode] * len(trailers), trailers
    torch.npu.synchronize()


def test_every_slice_carries_the_mode_and_the_flag_keeps_the_last_slot(inputs):
    """Both trailing ints have to survive the sliced two-level path.

    The wide half is launched once per slice, so a mode that only reached the
    first slice would leave the rest of the stage in production while the probe
    reports the ablation's timings.
    """
    os.environ["KDA_SOLVE_A16_MODE"] = "2"
    os.environ["KDA_DEBUG_STORES"] = "1"
    try:
        with _LaunchSpy() as spy:
            _call(inputs)
    finally:
        os.environ.pop("KDA_SOLVE_A16_MODE", None)
        os.environ.pop("KDA_DEBUG_STORES", None)
    trailers = spy.wide_trailer()
    assert len(trailers) >= 2, "the shape did not exercise more than one slice"
    for c, mode, debug in trailers:
        assert (mode, debug) == (2, 1), trailers
        assert c == trailers[0][0], "the slices disagree on their chunk count"
    torch.npu.synchronize()
