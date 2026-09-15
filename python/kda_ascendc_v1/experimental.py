"""Experimental K2 modes of the AscendC KDA backend.

``kda_ascendc_v1.kda_bt16_fwd_ascendc`` serves one K2 implementation - the
chunk-generic device-side loop - and rejects everything else.  The historical
per-chunk modes (``separated``, ``cube_separated``, ``cube_d3_separated``,
``cube_full_d4``, ``mix_aic_1_2``, ``mix_d12_vnew``, ``persistent``,
``persistent_scan``, ``persistent_scan_cube``, ``triton_aiv``) live here so the
S12-S15 benchmark scripts and their correctness oracles keep working.

They are all C=16 implementations: each kernel carries ``constexpr int32_t
M = 16`` instead of the build's ``KDA_CHUNK``, so running one under a C=32/64
build reads 16 rows of every chunk and returns a fast wrong answer.  The entry
point below refuses that combination.
"""
from __future__ import annotations

import torch

from .api import (C16_ONLY_K2_MODES, K2_MODES, PERSISTENT_LOOP, _kda_fwd_impl,
                  get_last_profile, kda_bt16_fwd_ascendc)

__all__ = [
    "kda_bt16_fwd_ascendc_experimental",
    "kda_bt16_fwd_ascendc",
    "get_last_profile",
]


def kda_bt16_fwd_ascendc_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    A_log: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    return_intermediates: bool = False,
    k2_mode: str = PERSISTENT_LOOP,
):
    """Same call as the public entry point, with every historical ``k2_mode``.

    The C=16-only modes are only accepted when the process was built with
    ``KDA_CHUNK=16`` (see ``python/kda_ascendc_v1/api.py``).
    """
    if k2_mode not in K2_MODES:
        raise ValueError("unsupported k2_mode %r (choose from %s)"
                         % (k2_mode, ", ".join(sorted(K2_MODES))))
    return _kda_fwd_impl(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, A_log=A_log, bias=bias,
        lower_bound=lower_bound, return_intermediates=return_intermediates,
        k2_mode=k2_mode)
