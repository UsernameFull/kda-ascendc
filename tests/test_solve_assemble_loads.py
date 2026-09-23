"""The assemble's load-path knob: wired, per slice, and read per call.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.37 replaced the
coupling block's six-per-chunk ND2NZ pattern with a batched one (one call per
block for the A operand, a chunk's two B bands merged, and pass 1's whole B
operand in one call), keeping the shipped form behind a runtime argument so
``tools/probe_solve_assemble_loads.py`` could price both in production.  The
answer was bit-identical outputs and -0.178 ms on the stage; this file pins the
wiring that makes the arm real:

* ``api.asm_load_mode()`` reads ``KDA_ASM_LOADS`` per call (not frozen at
  import), so one process can round-robin the two arms, and production is the
  batched path (1) rather than whatever happens to be in the environment;
* ``kda_solve_assemble`` carries the mode in its *last* argument slot - a mode
  in the wrong slot is read as the chunk count, and that failure is silent (a
  different A_inv, no exception);
* every slice of the two-level path carries it, not just the first.

Timings belong to the probe; this is the wiring.
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
KERNEL = "kda_solve_assemble"


def _skip_unless_two_level():
    if api.CHUNK not in api.SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % api.CHUNK)
    if api.SOLVE_WIDE_SUBB <= 1:
        pytest.skip("KDA_CHUNK=%d does not run the two-level solve" % api.CHUNK)


@pytest.fixture(scope="module")
def inputs():
    _skip_unless_two_level()
    device = torch.device("npu:0")
    torch.manual_seed(1131)
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
        """Every launch's trailing (C, loadMode) int pair."""
        out = []
        for name, args in self.calls:
            if name == kernel:
                out.append(tuple(struct.unpack("<i", a)[0] for a in args[-2:]))
        if not out:
            raise AssertionError("no %s launch was recorded" % kernel)
        return out


def test_production_is_the_batched_path(inputs):
    """KDA_ASM_LOADS unset -> mode 1, on every slice, in the last slot."""
    os.environ.pop("KDA_ASM_LOADS", None)
    assert api.asm_load_mode() == 1
    with _LaunchSpy() as spy:
        _call(inputs)
    trailers = spy.trailers()
    for n, mode in trailers:
        assert mode == 1, "production is not the batched load path"
    c_solve = -(-(B * H * (T // api.CHUNK)) // api.SOLVE_WIDE_NCH) * api.SOLVE_WIDE_NCH
    assert sum(n for n, _ in trailers) == c_solve, trailers
    torch.npu.synchronize()


def test_the_mode_is_read_per_call_not_frozen(inputs):
    """A probe flips this between two arms of one process: it has to move.

    Both arms have to reach *every* slice: a mode that only made the first
    launch would time a half-converted stage.
    """
    for mode in (0, 1, 0):
        os.environ["KDA_ASM_LOADS"] = str(mode)
        try:
            assert api.asm_load_mode() == mode
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_ASM_LOADS", None)
        trailers = spy.trailers()
        assert [t[1] for t in trailers] == [mode] * len(trailers), trailers
        assert len(trailers) >= 2, "the shape did not exercise more than one slice"
    torch.npu.synchronize()


def test_the_two_arms_agree_bit_for_bit(inputs):
    """The knob's whole justification: same operands, same outputs."""
    outs = []
    for mode in (0, 1):
        os.environ["KDA_ASM_LOADS"] = str(mode)
        try:
            out, state = _call(inputs)
        finally:
            os.environ.pop("KDA_ASM_LOADS", None)
        torch.npu.synchronize()
        outs.append((out.clone(), state.clone()))
    assert torch.equal(outs[0][0], outs[1][0]), "the load paths disagree"
    assert torch.equal(outs[0][1], outs[1][1]), "the final states disagree"
