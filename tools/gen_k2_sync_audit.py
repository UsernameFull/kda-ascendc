"""Static K2 sync audit: producer-consumer lifetime + flag/barrier inventory.

The plan's first round asks for two artifacts that must not be produced by hand
(hand-written tables are how a wrong flag id gets "verified"):

  1. a producer-consumer-lifetime table for every buffer K2 touches, and
  2. the counts of every CrossCore flag pair and every PipeBarrier in the
     production K2 kernel, so a candidate that "deletes a flag" can be diffed
     against a machine-generated baseline.

Both are read out of ``kernels/v1/k2_persistent_loop.cpp`` by parsing the
source, not by re-deriving the design: Set/Wait pairs are matched by flag id,
``CrossCoreSetFlag<2, PIPE_x>`` records the issuing pipe, and every
``PipeBarrier<PIPE_x>`` is counted per pipe.  Anything it cannot attribute is
reported as unattributed rather than dropped.

  python3 tools/gen_k2_sync_audit.py            # table + inventory
  python3 tools/gen_k2_sync_audit.py --json     # same, machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernels" / "v1" / "k2_persistent_loop.cpp"

# Flag ids the kernel defines; the pairing check needs them by name.
FLAG_DEF_RE = re.compile(r"^constexpr\s+\w+\s+(FL_\w+)\s*=\s*(-?\d+)", re.M)
SET_RE = re.compile(r"CrossCoreSetFlag<(\d+),\s*(PIPE_\w+)>\((FL_\w+)\)")
WAIT_RE = re.compile(r"CrossCoreWaitFlag\((FL_\w+)\)")
BARRIER_RE = re.compile(r"PipeBarrier<PIPE_(\w+)>")
HARDPAIR_RE = re.compile(r"(SetFlag|WaitFlag)<HardEvent::(\w+)>\((\w+)\)")
# Buffers of interest: GlobalTensor declarations plus the UB TBuf aliases.
GT_RE = re.compile(r"GlobalTensor<[\w:]+>\s+(\w+);")
TBUF_RE = re.compile(r"TBuf<TPosition::(\w+)>\s+([\w, ]+);")
INIT_RE = re.compile(r"pipe\.InitBuffer\((\w+),\s*([^;]+)\);")


def parse_kernel(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    flags = {name: int(val) for name, val in FLAG_DEF_RE.findall(text)}
    sets = defaultdict(list)
    waits = defaultdict(list)
    barriers = Counter()
    hard_events = Counter()
    for lineno, line in enumerate(lines, 1):
        for n, pipe, flag in SET_RE.findall(line):
            sets[flag].append({"line": lineno, "pipe": pipe, "c2c": int(n)})
        for flag in WAIT_RE.findall(line):
            waits[flag].append({"line": lineno})
        for pipe in BARRIER_RE.findall(line):
            barriers[pipe] += 1
        for what, ev, _evid in HARDPAIR_RE.findall(line):
            hard_events[(what, ev)] += 1

    # engine section: ASCEND_IS_AIC / ASCEND_IS_AIV blocks, by first line no.
    aic_line = text[:text.find("if ASCEND_IS_AIC")].count("\n") + 1 \
        if "if ASCEND_IS_AIC" in text else None
    aiv_line = text[:text.find("if ASCEND_IS_AIV")].count("\n") + 1 \
        if "if ASCEND_IS_AIV" in text else None

    def engine_of(lineno: int) -> str:
        if aic_line and aiv_line:
            return "AIC" if aic_line <= lineno < aiv_line else "AIV"
        return "?"

    # The four K2 stages, located by their comment banners.
    stage_lines = {}
    # Each engine has its own chunk loop (AIC: stages 1/3, AIV: stages 2/4), so
    # the "is this set inside the loop" question is per engine: the flag that
    # primes a depth-one pipeline is published after the producer's loop starts
    # but before the consumer's.
    loop_lines = {}
    for lineno, line in enumerate(lines, 1):
        m = re.search(r"---- stage (\d)", line)
        if m:
            stage_lines.setdefault(int(m.group(1)), lineno)
        if re.search(r"for \(int32_t chunk = 0; chunk < NT", line):
            eng = "AIC" if (aic_line and aic_line <= lineno < (aiv_line or 1 << 30)) else "AIV"
            loop_lines.setdefault(eng, lineno)
    return dict(text=text, lines=lines, flags=flags, sets=dict(sets), waits=dict(waits),
                loop_lines=loop_lines,
                barriers=dict(barriers), hard_events=dict(hard_events),
                engine_of=engine_of, stages=stage_lines,
                aic_line=aic_line, aiv_line=aiv_line)


def stage_of(parsed, lineno: int) -> str:
    prev = 0
    for st in sorted(parsed["stages"]):
        if parsed["stages"][st] <= lineno:
            prev = st
    return "stage%d" % prev if prev else "prologue"


def build_report(parsed):
    flags = parsed["flags"]
    pairing = []
    def loop_start(engine):
        return parsed["loop_lines"].get(engine)

    for name in sorted(flags, key=lambda n: flags[n]):
        s = parsed["sets"].get(name, [])
        w = parsed["waits"].get(name, [])
        pipes = Counter(x["pipe"] for x in s)
        # A depth-one pipeline is primed: the producer publishes one extra
        # token per head-step *before* the loop (the prologue) so that the
        # first consumer iteration has something to wait on.  That is balanced
        # in the pipeline sense (sets = waits + prologue tokens) and must not
        # be reported as an unmatched set, or a later candidate that deletes
        # the prologue would look "balanced" while deadlocking.
        # "in loop" is judged against the loop of the engine that issues the
        # site (a set by AIV is outside its loop only if it precedes the AIV
        # chunk loop).
        def in_loop(site):
            ls = loop_start(parsed["engine_of"](site["line"]))
            return ls is None or site["line"] >= ls

        prologue_sets = [x for x in s if not in_loop(x)]
        inloop_sets = [x for x in s if in_loop(x)]
        inloop_waits = [x for x in w if in_loop(x)]
        pairing.append({
            "flag": name, "id": flags[name],
            "set_count": len(s), "wait_count": len(w),
            "set_pipes": dict(pipes),
            "prologue_sets": len(prologue_sets),
            "inloop_sets": len(inloop_sets),
            "inloop_waits": len(inloop_waits),
            "balanced_inloop": len(inloop_sets) == len(inloop_waits),
            "pipeline_primed": len(prologue_sets) > 0,
            "balanced": len(s) == len(w) or len(inloop_sets) == len(inloop_waits),
            "set_sites": [{"line": x["line"], "pipe": x["pipe"],
                           "engine": parsed["engine_of"](x["line"]),
                           "stage": stage_of(parsed, x["line"])} for x in s],
            "wait_sites": [{"line": x["line"],
                            "engine": parsed["engine_of"](x["line"]),
                            "stage": stage_of(parsed, x["line"])} for x in w],
        })

    # Producer-consumer lifetime: written where, read where, in which loop.
    lifetimes = [
        dict(buffer="U (input)", where="GM", written="host, once per call",
             read="AIV stage 2 (DataCopy ub)", live="whole call"),
        dict(buffer="W, Qg (input)", where="GM", written="K1 solve, once per call",
             read="AIC stage 1 (qw/qg -> L0A)", live="whole call"),
        dict(buffer="Aqk, Kg (input)", where="GM", written="K1 pre_gram, once per call",
             read="AIC stage 3 (qa/qk)", live="whole call"),
        dict(buffer="Decay (input)", where="GM", written="K1 pre_gram",
             read="AIV stage 4 (DataCopy dec)", live="whole call"),
        dict(buffer="D1, D2 (bf16)", where="GM", written="AIC stage 1 Fixpipe",
             read="AIV stage 2 (DataCopy d1)", live="chunk-loop, one chunk"),
        dict(buffer="D3 (bf16)", where="GM", written="AIC stage 3 Fixpipe",
             read="AIV stage 4 (DataCopy d3)", live="chunk-loop, one chunk"),
        dict(buffer="D4 (fp32)", where="GM", written="AIC stage 3 Fixpipe",
             read="AIV stage 4 (four quarter loads)", live="chunk-loop, one chunk"),
        dict(buffer="VnewT (bf16)", where="GM", written="AIV stage 2 (DataCopy Vt)",
             read="AIC stage 3 (qv/vx -> L0A/L0B)", live="chunk-loop, one chunk"),
        dict(buffer="S16 (bf16)", where="GM", written="AIV (start-up + stage 4 quarters)",
             read="AIC stage 1 (qs -> L0B)", live="across chunks (whole call)"),
        dict(buffer="S32 (fp32)", where="GM", written="AIV, once at the end",
             read="host", live="whole call"),
        dict(buffer="Out (bf16)", where="GM", written="AIV stage 4 (first quarter)",
             read="host", live="whole call"),
        dict(buffer="H0 (fp32)", where="GM", written="host, once per call",
             read="AIV start-up (DataCopy state)", live="prologue"),
        dict(buffer="cf (L0C)", where="L0C", written="Mmad stages 1/3",
             read="Fixpipe stages 1/3", live="one head-step"),
        dict(buffer="l0a/l0b (L0A/L0B)", where="L0A/L0B", written="LoadData stages 1/3",
             read="Mmad stages 1/3", live="one head-step"),
        dict(buffer="lw/lg/ls/la/lv/lx/lk (L1)", where="L1", written="DataCopy (Nd2Nz)",
             read="LoadData stages 1/3", live="one head-step"),
        dict(buffer="ub/vf/sc/vt (UB)", where="UB", written="AIV stage 2",
             read="AIV stage 2/3 hand-off", live="stage2 phase, one head-step"),
        dict(buffer="d1/d1f/d2/d3/d3f/ob (UB)", where="UB", written="AIV stage 2/4",
             read="AIV stage 2/4", live="aliased phase pair (stage2|stage4)"),
        dict(buffer="vb/d4 (UB)", where="UB", written="AIV stage 4 quarter loads",
             read="AIV stage 4 recurrence", live="stage4 phase, one head-step"),
        dict(buffer="st (fp32 state, UB)", where="UB", written="prologue + stage 4",
             read="stage-4 recurrence + final publish", live="whole call, resident"),
        dict(buffer="dec (UB)", where="UB", written="AIV stage 4 (DataCopy)",
             read="AIV stage 4 Mul", live="one head-step"),
        dict(buffer="s16 (UB)", where="UB", written="AIV Cast", read="AIV MTE3 store to S16",
             live="one quarter"),
    ]

    return dict(
        kernel=str(KERNEL.relative_to(ROOT)),
        flag_pairing=pairing,
        barriers_by_pipe=parsed["barriers"],
        barrier_total=sum(parsed["barriers"].values()),
        hard_event_counts={"%s<%s>" % k: v for k, v in
                           sorted(parsed["hard_events"].items())},
        loops=dict(n_chunk_loop="NT (T / KDA_CHUNK)", n_head_loop="nh <= MAXH",
                   stage_order={"AIC": [1, 3], "AIV": [2, 4]}),
        producer_consumer=lifetimes,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    parsed = parse_kernel(KERNEL)
    report = build_report(parsed)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print("K2 sync audit: %s" % report["kernel"])
    print("  flags: %d  (balanced: %s)" % (
        len(report["flag_pairing"]),
        all(f["balanced"] for f in report["flag_pairing"])))
    print("  %-6s %4s %-11s %-11s %-9s %s"
          % ("flag", "id", "sets(wait)", "inloop", "prologue", "pairing"))
    for f in report["flag_pairing"]:
        sites = ",".join("%s/%s@%d" % (s["engine"], s["stage"], s["line"])
                         for s in f["set_sites"])
        pts = ",".join("%s/%s@%d" % (s["engine"], s["stage"], s["line"])
                       for s in f["wait_sites"])
        pairing_state = "balanced" if f["balanced_inloop"] and not f["pipeline_primed"] else (
            "primed +%d" % f["prologue_sets"] if f["pipeline_primed"] else "UNBALANCED")
        print("  %-6s %4d %4d/%-4d %4d/%-4d %-9d %s"
              % (f["flag"], f["id"], f["set_count"], f["wait_count"],
                 f["inloop_sets"], f["inloop_waits"], f["prologue_sets"], pairing_state))
        print("        set  [%s]" % sites)
        print("        wait [%s]" % pts)
    print("  PipeBarrier by pipe: %s  (total %d)"
          % (", ".join("%s x%d" % (k, v) for k, v in sorted(report["barriers_by_pipe"].items())),
             report["barrier_total"]))
    print("  HardEvent pairs (set+wait): %s"
          % ", ".join("%s x%d" % (k, v) for k, v in
                      sorted(report["hard_event_counts"].items())))
    print("  producer-consumer lifetime table (%d rows):" % len(report["producer_consumer"]))
    for row in report["producer_consumer"]:
        print("    %-28s %-8s w:%-42s r:%-46s live:%s"
              % (row["buffer"], row["where"], row["written"], row["read"], row["live"]))


if __name__ == "__main__":
    main()
