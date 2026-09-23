"""The Cube solve's A16-residency knob: wired, per slice, and read per call.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.38 keeps the block's
A16 tile in L1 for both passes of ``kda_solve_wu_cube_kernel`` instead of
re-reading it (the second read is an L2 hit, section 11.35).  The kernel takes
the mode as a runtime argument (api.cube_a16_resident(), KDA_CUBE_A16_RESIDENT)
so ``tools/probe_solve_cube_a16_resident.py`` could price both arms in one
process.  What has to stay true is the wiring: a mode in the wrong slot is read
as the chunk count, and that failure is silent - the solve computes a different
A_inv rather than raising.  The timings belong to the probe.
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
KERNEL = "kda_solve_wu_cube_kernel"


def _skip_unless_supported():
    if api.CHUNK not in api.SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % api.CHUNK)


@pytest.fixture(scope="module")
def inputs():
    _skip_unless_supported()
    device = torch.device("npu:0")
    torch.manual_seed(1133)
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

    def trailers(self, kernel=KERNEL):
        """Every launch's trailing (C, a16Mode) int pair."""
        out = []
        for name, args in self.calls:
            if name == kernel:
                out.append(tuple(struct.unpack("<i", a)[0] for a in args[-2:]))
        if not out:
            raise AssertionError("no %s launch was recorded" % kernel)
        return out


def test_production_keeps_a16_resident(inputs):
    """KDA_CUBE_A16_RESIDENT unset -> mode 1, on every slice, in the last slot."""
    os.environ.pop("KDA_CUBE_A16_RESIDENT", None)
    assert api.cube_a16_resident() == 1
    with _LaunchSpy() as spy:
        _call(inputs)
    trailers = spy.trailers()
    for n, mode in trailers:
        assert mode == 1, "production is not the resident form"
    c_solve = -(-(B * H * (T // api.CHUNK)) // api.SOLVE_WIDE_NCH) * api.SOLVE_WIDE_NCH
    assert sum(n for n, _ in trailers) == c_solve, trailers
    torch.npu.synchronize()


def test_the_mode_is_read_per_call_not_frozen(inputs):
    """A probe flips this between two arms of one process: it has to move."""
    for mode in (0, 1, 0):
        os.environ["KDA_CUBE_A16_RESIDENT"] = str(mode)
        try:
            assert api.cube_a16_resident() == mode
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_CUBE_A16_RESIDENT", None)
        trailers = spy.trailers()
        assert [t[1] for t in trailers] == [mode] * len(trailers), trailers
    torch.npu.synchronize()


def test_the_two_arms_agree_bit_for_bit(inputs):
    """The knob's whole justification: same operands, same outputs."""
    outs = []
    for mode in (0, 1):
        os.environ["KDA_CUBE_A16_RESIDENT"] = str(mode)
        try:
            out, state = _call(inputs)
        finally:
            os.environ.pop("KDA_CUBE_A16_RESIDENT", None)
        torch.npu.synchronize()
        outs.append((out.clone(), state.clone()))
    assert torch.equal(outs[0][0], outs[1][0]), "the two arms disagree"
    assert torch.equal(outs[0][1], outs[1][1]), "the final states disagree"
