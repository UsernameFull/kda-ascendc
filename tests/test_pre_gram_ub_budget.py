"""The C128 verdict has to be executable, not remembered.

Plan section 11.33 froze the layered-C128 route on two facts: the current fused
stage-1 kernel does not fit at M = 128 (UB, L0A and L0B all over their caps),
and the step that would get there ("halve the chunk count, double M") had
already turned K1 negative at C = 32 -> C = 64.  The first fact is arithmetic
over the kernel's own InitBuffer list, so it lives in a tool
(``gen_ub_l1_budget.pre_gram_ub``) and is pinned here:

  * the model has to agree with the headroom the kernel itself measured at
    C = 64 (its header records "under 8 KB of UB headroom left"), or it is
    just a second hand-kept comment;
  * the three budgets that break at M = 128 stay broken - if a future edit
    makes C = 128 fit, this test failing is the *point*: the decision rule in
    section 11.33 has to be re-run, not silently invalidated;
  * the model parses the source rather than carrying constants, so an
    InitBuffer edit moves the number (checked by editing a copy).

Host-side: no device, no RTC.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gen_ub_l1_budget as ub  # noqa: E402

KB = 1024
KERNEL = "kernels/v1/k1_pre_gram_mix.cpp"


def test_c64_matches_the_headroom_the_kernel_measured():
    pg = ub.pre_gram_ub(64)
    free = pg["ub_cap"] - pg["ub"]
    assert 0 < free < 8 * KB, "C=64 UB %d, cap %d" % (pg["ub"], pg["ub_cap"])


def test_c128_is_over_three_hardware_budgets():
    pg = ub.pre_gram_ub(128)
    assert pg["ub"] - pg["ub_cap"] > 64 * KB     # measured fault: aicore exception
    assert pg["l0a"] > 64 * KB
    assert pg["l0b"] > 64 * KB
    assert pg["l0c"] == 128 * KB                 # exactly full, not over
    # and the terms a layered build would have to band (section 11.33.1.d)
    assert {"bT0", "qgin", "qgmk", "qgout"} <= {n for n, _ in pg["ub_biggest"]}


def test_the_number_follows_the_source(tmp_path, monkeypatch):
    """Parsing, not constants: shrink one buffer in a copy and watch it move."""
    dst = tmp_path / KERNEL
    dst.parent.mkdir(parents=True)
    src = (ROOT / KERNEL).read_text(encoding="utf-8-sig")
    assert "pipe.InitBuffer(bT0, N * 4);" in src
    dst.write_text(src.replace("pipe.InitBuffer(bT0, N * 4);",
                               "pipe.InitBuffer(bT0, NG * 4);"), encoding="utf-8")
    base = ub.pre_gram_ub(64)["ub"]
    monkeypatch.setattr(ub, "ROOT", tmp_path)
    moved = ub.pre_gram_ub(64)["ub"]
    # bT0 goes from M * D * 4 (32 KB at C=64) to NG * 4 = 16 * D * 4 (8 KB).
    assert base - moved == (64 * 128 - 16 * 128) * 4
