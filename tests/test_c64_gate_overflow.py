"""[C=64 only] a legal raw gate overflows the two-level solve.

Found while pinning P1-1's benchmark golden: the benchmark's own "model"
inputs (raw ``g``, ``A_log`` ~ N(0,1), shape ``[1, 8192, 96, 128]``) made the
C=64 build return NaN while the C=16 build returned a finite answer that
matched FLA to 9.8e-4 on the very same tensors.  The minimal repro below needs
no big shape: at T=128, H=2 the C=64 solve already blows up.

Localised by dumping the K1 intermediates (``return_intermediates=True``):
``Qn/Kn/Gate/Gc/Beta/Decay/Rk/Rv/Qg/Kg`` are all finite, ``Aqk32`` (the
intra-chunk matrix the solve assembles) holds ``inf``, and everything
downstream of it (``L``, ``W``, ``U``, ``out``, ``state``) is NaN.  Bounded
inputs - ``|k|`` is l2-normalised, ``beta`` and the gate both come out of
sigmoids - so the growth is in the solve's own arithmetic, not in the data it
is fed; C=16 (one 16x16 solve) is fine and C=64 (the two-level assembly) is
not, which is why the existing C=64 numerics gates (tame gates, ~-2.5) never
saw it and this one does.

The gate transform is ``-5 * sigmoid(exp(A_log) * (g + dt_bias))``, i.e. any
finite ``g`` is legal API input, and FLA answers the same tensors with a finite
result.  Strict xfail: while the fault is open this documents it, and when
someone fixes the solve this test turns into XPASS and has to be un-marked.
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
REASON = ("C=64's two-level solve overflows on a legal raw gate: Aqk32 -> inf -> W/U/out NaN "
          "(the same input at C=16 is finite)")


@pytest.mark.npu
@pytest.mark.xfail(CHUNK == 64, strict=True, reason=REASON)
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
