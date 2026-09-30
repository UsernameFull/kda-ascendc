"""The SB = 4 coupling kernel: the host gate, the store arms and the knob slots.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.54 lands
``kernels/v1/k1_solve_assemble4.cpp`` - the six strictly-lower 16-blocks of a
64-wide chunk, solved on chip out of one Lneg bundle (the SB = 4 route of
section 11.52) - behind ``KDA_SOLVE_WIDE_SUBB = 4``.  This file pins what the
kernel owes, not how fast it is (the verdict is a loss; the probe and the docs
own the numbers):

* the six blocks against an fp64 inverse of the same L.  That is a tolerance,
  not bit equality: the SB = 4 schedule rounds its intermediates (X10, X20,
  X21, Ea, Gr, Gl) through bf16 on the fixpipe path where the SB = 2 form does
  not, so ~3e-3 is the honest ceiling - but the strict upper triangle has to be
  exactly zero, and the operand bundle has to be the negated 16-blocks in the
  layout the kernel reads (the host writer is part of the contract);
* storeMode 0 (three ndNum-grouped fixpipe stores) and storeMode 1 (six
  per-piece stores) are bit-identical: the store form is scheduling, and the
  same L0C slots leave through the same quantization either way;
* the two trailing knobs (ablation mask, level bisect) sit *after* storeMode in
  the argument list.  A launch that stops at storeMode reads the chunk count as
  the ablation mask and the store mode as the level - a silent wrong arm (level
  0 only), not an exception - so the slot order is pinned here rather than
  trusted to the probes.

With no KDA_SOLVE_WIDE_SUBB = 4 the module skips: the two-level 32-level solve
runs the *other* assemble kernel and this contract does not exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import kda_ascendc_v1.api as api  # noqa: E402

DEV = torch.device("npu:0")
KERNEL = "kda_solve_assemble4"
C = 8  # two blocks per launch at ASM_NCHUNK = 4


def _skip_unless_sb4():
    if api.CHUNK != 64:
        pytest.skip("KDA_CHUNK=64 is required (the SB = 4 kernel is 64-wide)")
    if api.SOLVE_WIDE_SUBB != 4:
        pytest.skip("KDA_SOLVE_WIDE_SUBB=4 is required")


@pytest.fixture(scope="module")
def operands():
    """L, xb, lneg and a16 in exactly the layout the kernel reads."""
    _skip_unless_sb4()
    pc, sb = api.CHUNK, api.SOLVE_WIDE_SUBB
    m = pc // sb
    mm = m * m
    torch.manual_seed(1312)
    L = (torch.randn(C, pc, pc, device=DEV) * 0.1).tril(-1).contiguous()
    xb = torch.zeros(C, sb, m, m, dtype=torch.bfloat16, device=DEV)
    for s in range(sb):
        blk = L[:, s * m:(s + 1) * m, s * m:(s + 1) * m]
        xb[:, s] = torch.linalg.inv(
            torch.eye(m, device=DEV) + blk.float()).bfloat16()
    # One bundle: [n10][n32][the 32x32 row-major rectangle rows 2M..4M, cols
    # 0..2M] - the kernel reads the rect's quadrants with a row stride of 2M.
    lneg = torch.zeros(C, 6 * mm, dtype=torch.bfloat16, device=DEV)
    lneg[:, 0 * mm:1 * mm] = L[:, 1 * m:2 * m, 0:m].reshape(C, mm).bfloat16()
    lneg[:, 1 * mm:2 * mm] = L[:, 3 * m:4 * m, 2 * m:3 * m].reshape(C, mm).bfloat16()
    lneg[:, 2 * mm:6 * mm] = L[:, 2 * m:4 * m, 0:2 * m].reshape(C, 4 * mm).bfloat16()
    lneg = (-lneg.float()).bfloat16().contiguous()
    a16 = torch.zeros(C, pc, pc, dtype=torch.bfloat16, device=DEV)
    torch.npu.synchronize()

    src = (ROOT / "kernels/v1/k1_solve_assemble4.cpp").read_text(encoding="utf-8-sig")
    api.rtc_compile(api._defines() + src, KERNEL, "")
    return L, xb, lneg, a16


def _run(operands, out, store_mode: int) -> None:
    """One full launch (no ablation, all six levels) into ``out``."""
    _, xb, lneg, _ = operands
    args = api._pack_ptrs([out, xb, lneg]) + \
        [api._i(C), api._i(store_mode), api._i(0), api._i(6)]
    nc = api.ASM_NCHUNK
    stream = torch_npu.npu.current_stream()
    api.launch_argsarray_engine(KERNEL, (C + nc - 1) // nc, stream.npu_stream, args, 0)
    torch.npu.synchronize()


def test_blocks_match_fp64(operands):
    L, xb, lneg, a16 = operands
    _run(operands, a16, store_mode=0)
    pc, sb = api.CHUNK, api.SOLVE_WIDE_SUBB
    m = pc // sb
    h = a16.cpu().float()
    ref = torch.linalg.inv(torch.eye(pc, dtype=torch.float64) + L.cpu().double()).float()
    worst = 0.0
    for s2 in range(sb):
        for s1 in range(s2):
            got = h[:, s2 * m:(s2 + 1) * m, s1 * m:(s1 + 1) * m]
            want = ref[:, s2 * m:(s2 + 1) * m, s1 * m:(s1 + 1) * m]
            worst = max(worst, float((got - want).abs().max()))
    diag = float(h.triu(diagonal=1).abs().max())
    # the diagonal tiles belong to the wide kernel in a real call, so gate the
    # host-side xb operand against fp64 as well (it is what the Cube consumes).
    for s in range(sb):
        want = ref[:, s * m:(s + 1) * m, s * m:(s + 1) * m]
        worst = max(worst, float((xb[:, s].cpu().float() - want).abs().max()))
    assert worst < 5e-3, "six-block coupling deviates from fp64: %.3e" % worst
    assert diag == 0.0, "strict upper triangle is not blank: %.3e" % diag


def test_store_modes_bit_identical(operands):
    L, xb, lneg, a16 = operands
    per_piece = torch.zeros_like(a16)
    _run(operands, a16, store_mode=0)
    _run(operands, per_piece, store_mode=1)
    assert torch.equal(a16.cpu(), per_piece.cpu()), \
        "storeMode 0/1 must differ only in how the stores are batched"


def test_knob_slots_are_trailing(operands):
    """storeMode 1 with the level knob at 0 leaves the tile unwritten."""
    L, xb, lneg, _ = operands
    a16 = torch.zeros_like(L, dtype=torch.bfloat16)
    args = api._pack_ptrs([a16, xb, lneg]) + \
        [api._i(C), api._i(1), api._i(0), api._i(0)]
    nc = api.ASM_NCHUNK
    stream = torch_npu.npu.current_stream()
    api.launch_argsarray_engine(KERNEL, (C + nc - 1) // nc, stream.npu_stream, args, 0)
    torch.npu.synchronize()
    assert float(a16.abs().max()) == 0.0, \
        "level 0 writes no store, so the ablation knob is not the trailing slot"
