"""The stage ledger is only worth having if a stale row cannot pass it.

Plan sections 2.1/2.4 ask for a per-intermediate lifecycle record and a byte
budget per candidate.  ``tools/gen_stage_lifecycle.py`` writes both, and this
file is what stops the CSV from drifting away from the kernels: every row
cites the symbols it appears as and the launch site it is handed over at, and
those citations are checked against the sources.  The tests are host-side (no
device, no RTC) because the failure they guard against is a *record* problem,
not a runtime one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
if not (ROOT / "kernels" / "v1" / "k2_persistent_loop.cpp").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import gen_stage_lifecycle as sl  # noqa: E402


def _geom(chunk=64):
    return sl.geometry(chunk, 1, 512, 4)


def test_every_row_validates_against_its_source():
    for chunk in (16, 32, 64):
        errors, _ = sl.validate(sl.geometry(chunk, 1, 512, 4))
        assert errors == [], "C=%d\n%s" % (chunk, "\n".join(errors))


def test_workspace_total_matches_the_budget_generator():
    """One authority for the per-call bytes: the budget tool's own arithmetic."""
    from gen_ub_l1_budget import budget as ub_budget
    geom = _geom()
    _, total = sl.validate(geom)
    rep = ub_budget(geom["chunk"], geom["b"], geom["t"], geom["h"], geom["maxh"])
    ref = sum(v[0] for k, v in rep["workspace"].items()
              if k not in rep["workspace_conditional"])
    assert total == ref


def test_a_stale_symbol_is_caught():
    """The citations have to have teeth, or the CSV is a comment."""
    geom = _geom()
    assert sl.validate(geom)[0] == []
    row = next(r for r in sl.ROWS if r["name"] == "Aqk16")
    row["sym"] = list(row["sym"]) + ["aqk16_that_was_renamed"]
    try:
        errors, _ = sl.validate(geom)
    finally:
        row["sym"] = [s for s in row["sym"] if s != "aqk16_that_was_renamed"]
    assert any("aqk16_that_was_renamed" in e for e in errors)


def test_the_debug_only_stores_stay_out_of_the_production_traffic():
    """The guard this ledger proposed has landed (docs 11.29), so the ledger has
    to count the three stores as debug traffic and not as production traffic.

    Aqk32's masked copy, A32 and BetaOut are written only for
    ``return_intermediates``; production passes ``debugStores=0`` and the
    interleaved A/B measured -0.164 ms for the three together.  If someone
    re-enables a store in the production path, the bytes below move and this
    test fails - which is the point: the ledger is the record of what a call
    actually moves.
    """
    rows = sl.traffic(sl.geometry(64, 1, 8192, 96))
    skipped = {r["row"]["name"]: r["nbytes"] for r in rows if r["row"].get("skipped")}
    assert set(skipped) == {"Aqk32 masked/scaled", "A32", "BetaOut"}, skipped
    assert all(r["writes"] == 0 for r in rows if r["row"].get("skipped"))
    assert sum(skipped.values()) > 4e8, "the three stores should be ~406 MB at C=64"
    for r in rows:
        if r["row"]["role"] == "debug":
            assert r["writes"] == 0 and r["reads"] == 0, r["row"]["name"]


def test_slot_map_has_no_duplicate_local_slot():
    for pair in {s["pair"] for s in sl.SLOTS}:
        locals_ = [s["local_slot"] for s in sl.SLOTS if s["pair"] == pair]
        assert len(locals_) == len(set(locals_)), pair
