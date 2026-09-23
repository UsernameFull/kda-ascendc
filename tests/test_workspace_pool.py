"""A fixed-shape workspace pool is only correct if it hands out distinct bytes.

Plan section S1 is "cache the per-call workspace allocations in a shape-keyed
pool and save 0.2-0.8 ms".  The prototype in ``tools/probe_workspace_pool.py``
prices the arm at 0.33 ms per call and the exposure probe
(``tools/probe_host_exposure.py``, pinned in ``test_workspace_pool_numerics.py``)
shows that arm is invisible end-to-end - the host has ~2.4 ms per call of slack
behind the device.  The route is therefore frozen, and what remains worth
pinning is the *hazard* the prototype ran into, because it is the same
faster-wrong-answer failure mode this repo has already shipped once (a C=16
kernel answering a C=64 call):

  * keying the pool on ``(shape, dtype)`` is not enough.  Dozens of the api's
    buffers share a shape (``rk``/``rv``/``qg``/``kg``/``W``/``U`` are all
    ``[c, CHUNK, D]`` bf16 at C=64), so a shape-keyed lookup hands ``rv`` the
    bytes of ``rk``, two kernels write the same memory, and the call returns a
    wrong answer - measured max|do| 3.368e-01, max|ds| 2.799e-01, while the e2e
    number looked 0.606 ms *better*.
  * a pooled buffer must hold at least the requested element count.  A short
    buffer reaches the api's own ``.view()`` and raises ("shape [8, 2, 64, 128]
    is invalid for input of size 262144"), or at a larger request reads past it.

These tests are host-side and cheap; they check the container contract rather
than device numerics, and the pool is never handed to a real call here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402
from kda_ascendc_v1.api import CHUNK, SUPPORTED_CHUNKS  # noqa: E402
from probe_workspace_pool import WorkspacePool  # noqa: E402


def _skip_unless_supported():
    if CHUNK not in SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is an unsupported build" % CHUNK)


def _entry(pool, dev, b=1, t=None, h=2):
    t = t or 2 * CHUNK
    entry = pool.checkout(dev, b, t, h, CHUNK, False)
    assert entry is not None
    return entry


def test_pool_returns_one_distinct_tensor_per_request():
    """Same-shaped buffers must not be aliased: that is the whole hazard."""
    _skip_unless_supported()
    pool = WorkspacePool()
    dev = torch.device("npu:0")
    entry = _entry(pool, dev)
    buckets: dict[tuple, list[int]] = {}
    for tensor in entry.values():
        buckets.setdefault((tuple(tensor.shape), tensor.dtype), []).append(int(tensor.data_ptr()))
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    assert collisions, "expected same-shaped buffers at this geometry"
    for key, ptrs in collisions.items():
        assert len(set(ptrs)) == len(ptrs), (key, len(ptrs), len(set(ptrs)))
    pool.checkin(entry)


def test_busy_entry_is_refused_not_shared():
    """Two in-flight calls must not be handed the same bytes."""
    _skip_unless_supported()
    pool = WorkspacePool()
    dev = torch.device("npu:0")
    first = pool.checkout(dev, 1, 2 * CHUNK, 2, CHUNK, False)
    assert first is not None
    second = pool.checkout(dev, 1, 2 * CHUNK, 2, CHUNK, False)
    assert second is None, "a leased entry was handed out twice"
    assert pool.refused == 1
    pool.checkin(first)
    third = pool.checkout(dev, 1, 2 * CHUNK, 2, CHUNK, False)
    assert third is first, "the entry did not come back after checkin"
    pool.checkin(third)


def test_a_different_shape_gets_its_own_entry():
    """No growing, no re-keying: a new geometry is a new entry."""
    _skip_unless_supported()
    pool = WorkspacePool()
    dev = torch.device("npu:0")
    a = pool.checkout(dev, 1, 2 * CHUNK, 2, CHUNK, False)
    pool.checkin(a)
    b = pool.checkout(dev, 1, 2 * CHUNK, 4, CHUNK, False)
    pool.checkin(b)
    assert a is not b
    assert len(pool._entries) == 2
    assert pool.misses == 2 and pool.hits == 0


def test_pooled_buffers_are_large_enough_for_the_api_views():
    """The kernel views the state buffer as [bh, NV, BV, D]; pin the sizes.

    The api derives ``final_state`` by viewing ``s32`` as
    ``(bh, NV, BV, D)`` and ``out_public`` as ``(b, t, h, D)``.  A pooled buffer
    with fewer elements than those views need is not caught by the pool itself
    (the shapes differ), so the entry has to be checked against them here.
    """
    _skip_unless_supported()
    pool = WorkspacePool()
    dev = torch.device("npu:0")
    b, t, h = 1, 2 * CHUNK, 4
    entry = _entry(pool, dev, b=b, t=t, h=h)
    bh, nv, bv, d = b * h, 2, 64, 128
    nt = t // CHUNK
    assert entry["s32"].numel() >= bh * nv * bv * d
    assert entry["out_public"].numel() == b * t * h * d
    assert entry["d1"].numel() == bh * nv * nt * CHUNK * bv
    pool.checkin(entry)
