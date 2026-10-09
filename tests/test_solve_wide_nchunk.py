"""The wide solve's chunks-per-block: the C=64 default is 12, and the padded
chunk count stays a whole wiring unit so every chunk gets launched.

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.64 re-tested
``KDA_SOLVE_WIDE_NCHUNK`` once the wide half became the solve stage's wall:
8 -> 12 is bit-identical on every output (A16 / Xb / Lneg, plus the diag-block
fp64 gate and the strict-upper blank) and drops the wide stream 1.52 -> 1.40 ms
isolated, -0.15 ms e2e, so the C=64 default flips.  The knob is a compile-time
define (the tile depth is baked into every InitBuffer), so there is no runtime
arm to switch in one process; the timings belong to the probes.

What this file pins is the geometry the flip touches.  The two-level solve
launches whole wiring-unit groups - the lcm of the three kernels'
chunks-per-block - so the padded chunk count has to be a multiple of that unit,
or the trailing group is never launched at all: with the pre-fix rounding to
SOLVE_WIDE_NCH alone, c = 4 launched *zero* solve kernels and c = 64 lost its
last 4 chunks.  The coverage test runs the real ``_launch_solve_two_level``
with the launcher spied out, so it needs no kernel execution.
"""
from __future__ import annotations

import math
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


def _skip_unless_supported():
    if api.CHUNK not in api.SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % api.CHUNK)


def test_the_default_chunks_per_block():
    """The C=64 flip, and the smaller builds' swept values, stay put."""
    _skip_unless_supported()
    if os.environ.get("KDA_SOLVE_WIDE_NCHUNK"):
        pytest.skip("explicit KDA_SOLVE_WIDE_NCHUNK override in the environment")
    if api.CHUNK <= 16:
        want = 32
    elif api.CHUNK <= 32:
        want = 16
    else:
        want = 12
    assert api.SOLVE_WIDE_NCHUNK == want, (
        "the C=%d default moved off %d; docs 11.64 and 11.52 own these values"
        % (api.CHUNK, want))
    assert api.SOLVE_WIDE_NCH == want // api.SOLVE_WIDE_SUBB


def test_the_padded_count_is_a_whole_wiring_unit():
    """Rounding to SOLVE_WIDE_NCH alone under-covers; the unit does not."""
    unit = math.lcm(api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK)
    for c in (1, 4, 6, 12, 16, 60, 64, 66, 128, 2048, 12288):
        padded = api._solve_padded_chunks(c)
        assert padded >= c, (c, padded)
        assert padded - c < unit, (c, padded)
        assert padded % unit == 0, (c, padded)
        assert padded % api.SOLVE_WIDE_NCH == 0, (c, padded)


def test_wide_launches_cover_every_padded_chunk():
    """Every chunk of [0, c_solve) is in exactly one wide slice, and the
    slice chunk counts are whole blocks for all three kernels."""
    _skip_unless_supported()
    if api.SOLVE_WIDE_SUBB != 2:
        pytest.skip("the wiring-unit rounding only bites the two-level path")
    dev = torch.device("npu:0")
    z = torch.zeros(1, dtype=torch.float32, device=dev)

    def run(chunks):
        c_solve = api._solve_padded_chunks(chunks)
        seen = []
        real = api._launch

        def spy(name, blocks, args, stream):
            if name == "kda_solve_wu_wide":
                seen.append((int(blocks), struct.unpack("<i", args[6])[0]))
            return None

        api._launch = spy
        try:
            api._launch_solve_two_level(
                c_solve, chunks, api.SOLVE_WIDE_NCH, api.ASM_NCHUNK,
                api.WU_NCHUNK, api.SOLVE_OVERLAP, z, z, z, z, z, z, None,
                z, z, z, z, torch_npu.npu.current_stream(), False)
        finally:
            api._launch = real
        return c_solve, seen

    for chunks in (4, 16, 64, 128, 12288):
        c_solve, seen = run(chunks)
        assert seen, chunks
        start = 0
        for blocks, n in seen:
            assert n % api.SOLVE_WIDE_NCH == 0, (chunks, n)
            assert blocks == n // api.SOLVE_WIDE_NCH, (chunks, n, blocks)
            start += n
        assert start == c_solve, (chunks, start, c_solve)
        assert c_solve >= chunks
