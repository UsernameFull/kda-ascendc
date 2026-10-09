"""The assemble's load-path knob: wired, per slice, and read per call.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.37 replaced the
coupling block's six-per-chunk ND2NZ pattern with a batched one (one call per
block for the A operand, a chunk's two B bands merged, and pass 1's whole B
operand in one call), keeping the shipped form behind a runtime argument so
``tools/probe_solve_assemble_loads.py`` could price both in production.  The
answer was bit-identical outputs and -0.178 ms on the stage; this file pins the
wiring that makes the arm real:

* ``api.asm_load_mode()`` reads ``KDA_ASM_LOADS`` per call (not frozen at
  import), so one process can round-robin the arms, and production is mode 5
  (window fill + batched store, section 11.61) rather than whatever happens to
  be in the environment;
* ``kda_solve_assemble`` carries the mode in its *last* argument slot - a mode
  in the wrong slot is read as the chunk count, and that failure is silent (a
  different A_inv, no exception);
* every slice of the two-level path carries it, not just the first.

The same file also pins section 11.50's arm: mode 4 is mode 2 with the
whole-window Xb fill (both passes' Xb operands in one ND2NZ call, so the block
sends two MTE2 calls instead of 1 + nch + 2); it has to stay bit-identical and
keep P on chip like mode 2.  Modes 3 (batched store) and 4 (window fill) were
separate knobs while the wide half was the wall; section 11.61 re-measured
both in the AIC-limited regime (each a win) and composed them as mode 5 =
window fill + batched store, which has to stay bit-identical too - and 5 is
now production.

Timings belong to the probe; this is the wiring.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "python"))

import kda_ascendc_v1.api as api  # noqa: E402

D = 128
B, H = 1, 16
# T is sized so the multislice test can see more than one slice at all: the
# launcher caps the slice count at ngrp // 8 (docs 11.59), so only ngrp >= 16
# yields two.  One wiring group per chunk-build unit: T = unit * CHUNK gives
# c = H * unit chunks and ngrp = H = 16.  (T reads 768 at C=64, 512 at C=16/32.)
UNIT = math.lcm(api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK)
T = UNIT * api.CHUNK
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

    def ptrs(self, index, kernel=KERNEL):
        """Every launch's argument at ``index``, read back as an integer."""
        out = []
        for name, args in self.calls:
            if name == kernel:
                out.append(int.from_bytes(args[index], "little"))
        if not out:
            raise AssertionError("no %s launch was recorded" % kernel)
        return out

    def trailers(self, kernel=KERNEL):
        """Every launch's trailing (C, loadMode) int pair."""
        out = []
        for name, args in self.calls:
            if name == kernel:
                out.append(tuple(struct.unpack("<i", a)[0] for a in args[-2:]))
        if not out:
            raise AssertionError("no %s launch was recorded" % kernel)
        return out


def test_production_is_mode_5(inputs):
    """KDA_ASM_LOADS unset -> mode 5, on every slice, in the last slot.

    Mode 5 is section 11.61's composition (window fill + batched store); the
    default moved there from 2 once the wide half stopped being the wall and
    the AIC half's two stored knobs were re-measured as wins.
    """
    os.environ.pop("KDA_ASM_LOADS", None)
    assert api.asm_load_mode() == 5
    with _LaunchSpy() as spy:
        _call(inputs)
    trailers = spy.trailers()
    for n, mode in trailers:
        assert mode == 5, "production is not mode 5 on every slice"
    c_solve = api._solve_padded_chunks(B * H * (T // api.CHUNK))
    assert sum(n for n, _ in trailers) == c_solve, trailers
    torch.npu.synchronize()


def test_the_mode_is_read_per_call_not_frozen(inputs):
    """A probe flips this between two arms of one process: it has to move.

    Both arms have to reach *every* slice: a mode that only made the first
    launch would time a half-converted stage.
    The default whole-wave slice policy (docs 11.59) reads a single slice at
    any shape this file can afford, so the multislice schedule is forced with
    KDA_SOLVE_SLICE_CHUNKS (read per call) one wiring group deep; the launcher
    then clamps to ngrp // 8 = 2 slices.  (A fixed 32-chunk size no longer
    does this: at NCHUNK = 12 the group count is not a multiple of the cap and
    the clamp collapses the schedule back to one slice - docs 11.64.)
    """
    for mode in (0, 2, 4, 5, 1, 0):
        os.environ["KDA_ASM_LOADS"] = str(mode)
        os.environ["KDA_SOLVE_SLICE_CHUNKS"] = str(UNIT)
        try:
            assert api.asm_load_mode() == mode
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_ASM_LOADS", None)
            os.environ.pop("KDA_SOLVE_SLICE_CHUNKS", None)
        trailers = spy.trailers()
        assert [t[1] for t in trailers] == [mode] * len(trailers), trailers
        assert len(trailers) >= 2, "the shape did not exercise more than one slice"
    torch.npu.synchronize()


def test_mode_3_is_the_batched_store_arm(inputs):
    """Section 11.42's ndNum store: kept as an arm, bit-exact and wired.

    The batched A16 store (one ndNum call for the block's four lower-left
    blocks, srcNdStride = 4 in 1 KB fractal units) is measured in
    tools/probe_fixpipe_shape.py and tools/probe_solve_assemble_loads.py: it
    won 0.10 ms isolated but lost on the stage/e2e while the AIV half was the
    wall (the batched store is a serial tail after pass 1); once section 11.59
    made the AIC half the floor, section 11.61 re-measured it as -0.071 wall
    and it is half of production mode 5.  This pins that the bare arm is
    reachable, that it still is mode 2's structure (P on chip -> no P tile),
    and that it lands the same numbers as mode 2.
    """
    os.environ["KDA_ASM_LOADS"] = "3"
    try:
        with _LaunchSpy() as spy:
            _call(inputs)
        ptrs = spy.ptrs(3)
        assert ptrs and all(p == 0 for p in ptrs), \
            "mode 3 is mode 2's structure and must not allocate the P tile"
    finally:
        os.environ.pop("KDA_ASM_LOADS", None)
    torch.npu.synchronize()


def test_mode_2_leaves_the_p_tile_unallocated(inputs):
    """P on chip means no GM tile: mode 2 hands the kernel a null pointer.

    The tile is ``[c_solve, M, M]`` bf16 - 25.17 MB per call at [1,8192,96,128]
    (24 MiB; the 50.3 MB in docs section 11.39 is its GM round trip) - and mode
    2 neither writes nor reads it (the fixpipe goes to L1, the load comes from
    L1), so allocating it would be dead memory.  Modes 0 and 1 do use it, which
    is why this is a property of the mode and not of the call site.
    """
    for mode, expect_null in ((2, True), (4, True), (5, True), (1, False),
                              (0, False)):
        os.environ["KDA_ASM_LOADS"] = str(mode)
        try:
            with _LaunchSpy() as spy:
                _call(inputs)
        finally:
            os.environ.pop("KDA_ASM_LOADS", None)
        ptrs = spy.ptrs(3)  # the assemble's fourth argument is the P tile
        assert ptrs, "no assemble launch was recorded"
        if expect_null:
            assert all(p == 0 for p in ptrs), \
                "mode 2 allocated a P tile it never touches"
        else:
            assert all(p != 0 for p in ptrs), \
                "mode %d needs a P tile and got a null" % mode
    torch.npu.synchronize()


def test_the_arms_agree_bit_for_bit(inputs):
    """The knob's whole justification: same operands, same outputs."""
    outs = []
    for mode in (0, 1, 2, 3, 4, 5):
        os.environ["KDA_ASM_LOADS"] = str(mode)
        try:
            out, state = _call(inputs)
        finally:
            os.environ.pop("KDA_ASM_LOADS", None)
        torch.npu.synchronize()
        outs.append((out.clone(), state.clone()))
    for i, mode in enumerate((1, 2, 3, 4, 5), start=1):
        assert torch.equal(outs[0][0], outs[i][0]), "mode %d differs" % mode
        assert torch.equal(outs[0][1], outs[i][1]), "mode %d's state differs" % mode
