"""Aggregate one msopprof `PipeUtilization.csv` into the docs' SUMMARY form.

`msprof op --aic-metrics=PipeUtilization` writes one row per core with the
per-pipe busy times of the profiled kernel; the verdicts in docs section 11.42
were read off this file by hand, so this script fixes the reading: it prints
mean / min / max for the profiled kernel's pipe times and its scalar stall
breakdown, and the sum of the busy pipes against the block wall, which is the
number that says whether two pipes were ever busy at once.  The prefix is
auto-detected: AIC kernels (`aic_fixpipe`/`mte2`/`mte1`/`cube`/`scalar`) and
AIV kernels (`aiv_vec`/`mte2`/`mte3`/`scalar`) are both read.

  python3 tools/parse_msop_pipeutilization.py <OPPROF_dir_or_csv> [...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PIPES = ["aic_fixpipe_time(us)", "aic_mte2_time(us)", "aic_mte1_time(us)",
         "aic_cube_time(us)", "aic_scalar_time(us)"]
STALLS = ["aic_scalar_mte1_stall_time(us)", "aic_scalar_wait_ib_time(us)",
          "aic_scalar_single_time(us)", "aic_scalar_dual_time(us)",
          "aic_scalar_mte2_stall_time(us)", "aic_scalar_cube_stall_time(us)",
          "aic_scalar_mte3_stall_time(us)", "aic_scalar_wait_time(us)"]


def pipe_columns(rows) -> tuple[str, list[str]]:
    """The (prefix, pipe) columns the CSV actually carries.

    AIC kernels fill ``aic_*`` and leave ``aiv_*`` NA and vice versa, so the
    pipe set cannot be hard-coded: the wide solve is the AIV-side kernel
    (``aiv_vec``/``aiv_mte2``/``aiv_mte3``/``aiv_scalar``) and section 11.51 is
    read off it.  Pipe-level columns match ``<prefix>_<pipe>_time(us)`` with a
    single-token pipe name, which excludes the per-pipe scalar stall
    breakdowns (``..._scalar_single_time(us)``) by construction.
    """
    import re
    cols = list(rows[0].keys())
    for pref in ("aic", "aiv"):
        if rows[0].get("%s_time(us)" % pref) in (None, "", "NA"):
            continue
        pipes = [c for c in cols
                 if re.fullmatch(r"%s_[a-z0-9]+_time\(us\)" % pref, c)]
        if pipes:
            return pref, pipes
    return "aic", PIPES


def load(arg: str) -> tuple[Path, list[dict]]:
    p = Path(arg)
    if p.is_dir():
        hits = sorted(p.glob("**/PipeUtilization.csv"))
        if not hits:
            raise SystemExit("no PipeUtilization.csv under %s" % p)
        if len(hits) > 1:
            raise SystemExit("%d PipeUtilization.csv under %s" % (len(hits), p))
        p = hits[0]
    with p.open() as f:
        rows = [r for r in csv.DictReader(f)]
    return p, rows


def stats(rows, col):
    xs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "NA")]
    if not xs:
        return None
    return sum(xs) / len(xs), min(xs), max(xs)


def main() -> None:
    for arg in sys.argv[1:]:
        p, rows = load(arg)
        prefix, pipes = pipe_columns(rows)
        stall_cols = [c.replace("aic_", prefix + "_") for c in STALLS]
        print("== %s (%d rows, %s pipes)" % (p, len(rows), prefix))
        wall = stats(rows, "%s_time(us)" % prefix)
        total = 0.0
        for col in pipes:
            s = stats(rows, col)
            if s is None:
                continue
            total += s[0]
            print("  %-24s mean %8.4f  min %8.4f  max %8.4f  (%.3f of wall)"
                  % (col, s[0], s[1], s[2], s[0] / wall[0]))
        print("  %-24s mean %8.4f  min %8.4f  max %8.4f"
              % ("%s_time(us) [wall]" % prefix, wall[0], wall[1], wall[2]))
        print("  %-24s       %8.4f   (sum of %d pipes / wall = %.3f)"
              % ("sum of pipes", total, len(pipes), total / wall[0]))
        for col in stall_cols:
            s = stats(rows, col)
            if s is not None and s[0] > 0.005:
                print("  %-34s mean %8.4f" % (col, s[0]))


if __name__ == "__main__":
    main()
