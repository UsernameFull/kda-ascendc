"""A legal raw gate must keep the C=64 Gram (and everything after it) finite.

Found while pinning P1-1's benchmark golden: the benchmark's own "model"
inputs (raw ``g``, ``A_log`` ~ N(0,1), shape ``[1, 8192, 96, 128]``) made the
C=64 build return NaN while the C=16 build returned a finite answer that
matched FLA to 9.8e-4 on the very same tensors.  The minimal repro below needs
no big shape: at T=128, H=2 the C=64 solve used to blow up.

The gate takes the full lower bound (-5) per row when its sigmoid saturates, so
one chunk's cumsum can span 320 in the log domain.  K1's Gram folds the decay
into its two operands (``q*e^gc`` and ``k*e^-gc``), and with one reference for
the whole chunk those operands reach ``exp(+-160)`` - past the fp32/bf16 exp
range of ``exp(88)`` - so ``Aqk32`` came out ``inf`` and ``L/W/U/out/state``
NaN (C=16, whose chunk-span stays inside the range, was finite throughout).
The fix gives each 32-row gate *band* its own reference row and publishes band
0's k side a second time under band 1's centre, which is what the (1, 0) Gram
block needs to have both of its operands in range; see the constants note in
``kernels/v1/k1_pre_gram_mix.cpp`` and docs section 11.23.  This test is its
regression gate: with the fix the C=64 build is finite and its (1, 0) block is
genuinely populated rather than silently zeroed.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from kda_ascendc_v1.api import CHUNK, SUPPORTED_CHUNKS, kda_bt16_fwd_ascendc  # noqa: E402

D = 128


@pytest.mark.npu
def test_a_raw_gate_keeps_the_solve_finite():
    if CHUNK not in SUPPORTED_CHUNKS:
        pytest.skip(f"this build's KDA_CHUNK={CHUNK} is not supported (see api.SUPPORTED_CHUNKS)")
    b, t, h = 1, 128, 2
    torch.manual_seed(7)
    dev = "npu:0"
    q = torch.randn(b, t, h, D).to(dev).to(torch.bfloat16)
    k = torch.randn(b, t, h, D).to(dev).to(torch.bfloat16)
    v = torch.randn(b, t, h, D).to(dev).to(torch.bfloat16)
    beta = torch.randn(b, t, h).to(dev)
    # a raw gate: any finite value is legal, and this one is deliberately not
    # the saturated-but-tame ~-2.5 of the timing gates
    g = torch.randn(b, t, h, D).to(dev)
    a_log = torch.linspace(-1.0, 0.2, h).to(dev)
    bias = (torch.randn(h, D) * 0.1).to(dev)

    out, state, dbg = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, A_log=a_log, bias=bias, lower_bound=-5.0,
        output_final_state=True, return_intermediates=True, k2_mode="persistent_loop")
    torch.npu.synchronize()

    finite = {
        "Aqk32": dbg["Aqk32"].float(),
        "W": dbg["W"].float(),
        "U": dbg["U"].float(),
        "out": out.float(),
        "state": state.float(),
    }
    broken = {name: int(torch.isnan(x).sum() + torch.isinf(x).sum())
              for name, x in finite.items() if not torch.isfinite(x).all()}
    assert not broken, f"non-finite values in {broken}"
    # The (1, 0) block is the one the band split has to publish twice; a fix
    # that got its operands wrong shows up here long before it shows up in the
    # output (its entries are ~1e-2 of the tile's largest, so it also catches a
    # block that quietly became zero).
    if CHUNK > 32:
        blk = dbg["Aqk32"].float()[:, CHUNK // 2:, :CHUNK // 2]
        assert blk.abs().max() > 1e-6, "the (1, 0) Gram block is empty"
