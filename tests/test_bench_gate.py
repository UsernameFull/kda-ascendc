"""The production benchmark gate: the golden, and how it fails.

``benchmarks/bench_fla_compare.py --gate`` compares a fresh run against
``benchmarks/golden/<shape>.json``.  The comparison function is pure, so what
the gate *means* is pinned here on synthetic numbers: a 20% median regression,
a changed geometry, a numeric drift against FLA and a stage regression each
have to fail, while a small slow-down inside the tolerance only gets a note.
The real timing of the machine is not tested here (that is the gate's job, on
a quiet device).
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
GOLDEN = ROOT / "benchmarks" / "golden" / "fla_compare_1_8192_96_128.json"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location(
        "bench_fla_compare", ROOT / "benchmarks" / "bench_fla_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_fla_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def golden(bench):
    return json.loads(GOLDEN.read_text())


def scaled(row, factor, **extra):
    row = dict(row)
    for key in ("median_ms", "p20_ms", "p80_ms"):
        if row.get(key) is not None:
            row[key] = row[key] * factor
    row.update(extra)
    return row


@pytest.mark.unit
def test_golden_is_self_describing(bench, golden):
    assert GOLDEN.exists(), "the production golden is committed with the gate"
    assert golden["schema"] == bench.GOLDEN_SCHEMA
    assert golden["shape"] == [1, 8192, 96, 128]
    # the geometry is what makes two timings comparable at all
    assert golden["compile"]["KDA_CHUNK"] == 64
    assert golden["config"]["KDA_CHUNK"] == 64
    assert golden["gate"]["subject"] == "ascendc persistent_loop"
    assert golden["gate"]["median_ms"] <= golden["gate"]["median_ms_max"]
    assert set(golden["stages"]) >= {"pre_gram_ms", "solve_ms", "k2_ms", "total_ms"}
    assert golden["git_commit"] and golden["recorded_utc"]
    # the numeric arm has to be alive: a golden recorded on a non-finite run
    # would make every later run skip the "fast but wrong" check
    for gate_key, row_key in (("o_max_abs_diff", "o_max_abs_diff_vs_fla"),
                              ("state_max_abs_diff", "state_max_abs_diff_vs_fla")):
        assert math.isfinite(golden["gate"][gate_key]), f"{gate_key} is not finite in the golden"
        assert math.isfinite(golden["timings"][row_key]), f"{row_key} is not finite in the golden"
    # the FLA comparison rides along: a measured row when FLA ran, and the
    # published H100 row for the ratio
    assert golden["fla"]["published_h100_ms"] > 0
    assert any(isinstance(v, dict) for v in golden["fla"].values())


@pytest.mark.unit
def test_gate_passes_on_the_goldens_own_numbers(bench, golden):
    failures = bench.gate_failures(golden["timings"], golden["stages"], golden["compile"],
                                   golden, device=golden["device"])
    assert failures == []
    assert bench.print_gate_report(golden, golden["timings"], golden["stages"], failures,
                                  golden["device"])


@pytest.mark.unit
def test_gate_fails_on_a_median_regression(bench, golden):
    row = scaled(golden["timings"], 1.2)
    failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden)
    assert any(f.startswith("median ") and "golden" in f for f in failures), failures
    assert not bench.print_gate_report(golden, row, golden["stages"], failures, golden["device"])


@pytest.mark.unit
def test_gate_fails_on_the_absolute_ceiling(bench, golden):
    row = dict(golden["timings"])
    row["median_ms"] = golden["gate"]["median_ms_max"] + 0.01
    # the ceiling is the absolute arm and has to fire on its own: with the
    # golden's own tolerance relaxed, only it can catch this run
    failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden, median_tol=1.5)
    assert any("ceiling" in f for f in failures), failures
    assert not any(f.startswith("median ") and " x " in f for f in failures), failures


@pytest.mark.unit
def test_gate_fails_on_a_geometry_change(bench, golden):
    other = dict(golden["compile"])
    other["KDA_CHUNK"] = 16
    failures = bench.gate_failures(golden["timings"], golden["stages"], other, golden)
    assert any("geometry KDA_CHUNK" in f for f in failures), failures


@pytest.mark.unit
def test_gate_treats_a_non_finite_output_as_a_failure(bench, golden):
    # every comparison against NaN is False, so this has to be explicit: a
    # NaN run must not read as "inside the tolerance"
    for key in ("o_max_abs_diff_vs_fla", "state_max_abs_diff_vs_fla"):
        row = dict(golden["timings"])
        row[key] = float("nan")
        failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden)
        assert any("not finite" in f for f in failures), failures
        assert not bench.print_gate_report(golden, row, golden["stages"], failures, golden["device"])


@pytest.mark.unit
def test_gate_refuses_a_golden_that_was_recorded_on_a_non_finite_run(bench, golden):
    bad = json.loads(json.dumps(golden))
    bad["gate"]["o_max_abs_diff"] = float("nan")
    bad["timings"]["o_max_abs_diff_vs_fla"] = float("nan")
    failures = bench.gate_failures(golden["timings"], golden["stages"], golden["compile"], bad)
    assert any("re-record" in f for f in failures), failures


@pytest.mark.unit
def test_gate_fails_on_numeric_drift(bench, golden):
    row = dict(golden["timings"])
    row["o_max_abs_diff_vs_fla"] = golden["gate"]["o_max_abs_diff"] * 10
    failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden)
    assert any(f.startswith("out max-abs") for f in failures), failures


@pytest.mark.unit
def test_gate_fails_when_the_subject_did_not_run(bench, golden):
    failures = bench.gate_failures({"error": "507015 aicore exception"}, None,
                                   golden["compile"], golden)
    assert len(failures) == 1 and "did not run" in failures[0], failures
    failures = bench.gate_failures(None, None, golden["compile"], golden)
    assert len(failures) == 1 and "did not run" in failures[0], failures


@pytest.mark.unit
def test_gate_fails_on_a_stage_regression(bench, golden):
    stages = dict(golden["stages"])
    stages["pre_gram_ms"] = golden["stages"]["pre_gram_ms"] * 1.5
    stages["total_ms"] = golden["stages"]["total_ms"] + golden["stages"]["pre_gram_ms"] * 0.5
    failures = bench.gate_failures(golden["timings"], stages, golden["compile"], golden)
    assert any(f.startswith("pre_gram_ms ") and ">" in f for f in failures), failures


@pytest.mark.unit
def test_gate_reports_a_small_slowdown_as_a_note(bench, golden):
    row = scaled(golden["timings"], 1.02)
    failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden)
    assert failures and all(f.startswith("note: ") for f in failures), failures
    assert bench.print_gate_report(golden, row, golden["stages"], failures, golden["device"])


@pytest.mark.unit
def test_gate_notes_a_missing_numeric_reference(bench, golden):
    row = {k: v for k, v in golden["timings"].items() if "diff_vs_fla" not in k}
    failures = bench.gate_failures(row, golden["stages"], golden["compile"], golden)
    assert failures and all(f.startswith("note: ") for f in failures), failures
    assert any("numeric check" in f for f in failures), failures
