"""Device-event timeline of one e2e call, and the blocked scan's price inside it.

11.55's launch replay says the blocked gate-cumsum scan shortens the
``kda_pre_gram_mix`` kernel by ~0.1-0.2 ms (control 4.2 -> scan 4.0, MIN of 8,
reproduced three times), but the interleaved e2e arms of
tools/probe_pg_cumsum_scan_e2e.py read +0.007 ms: the 10.9 ms wall does not
move.  A host-wall MIN over the whole pipeline cannot resolve 0.1 ms, so this
tool is the instrument that can: every launch of one call is wrapped in a pair
of device events (recorded on the launch's own stream), and after a sync the
per-launch span, the gap in front of each launch, and the whole-call device
wall are read back.  The same timeline is taken with the pre_gram launch
swapped for the probe clone's serial arm and its blocked-scan arm in
alternating order, so both questions - "does the kernel get faster inside the
pipeline" and "if so, where does the time go" - are answered on the device
instead of through the host.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_scan_timeline.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
REPS = int(os.environ.get("KDA_PG_TL_REPS", "8"))
DO_ARMS = os.environ.get("KDA_PG_TL_ARMS", "1") == "1"
PROBE = "kda_pg_addwork_probe"
ARMS = [("serial", 0, 0), ("scan", 0, 8)]


def stream_map():
    cur = torch.npu.current_stream()
    sa, sb = api._solve_streams(DEV)
    return {cur.npu_stream: cur, sa.npu_stream: sa, sb.npu_stream: sb}


def run_call(q, k, v, g, beta, kw, arm=None):
    """One pipeline call; every launch is bracketed by device events."""
    orig = api._launch
    log = []
    streams = stream_map()

    def spy(kernel, blocks, args, stream):
        obj = streams.get(stream)
        if obj is None:
            return orig(kernel, blocks, args, stream)
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record(obj)
        if kernel == "kda_pre_gram_mix" and arm is not None:
            api.launch_argsarray_engine(PROBE, int(blocks), stream,
                                        list(args) + [api._i(arm[0]), api._i(arm[1])], 0)
        else:
            orig(kernel, blocks, args, stream)
        ev1.record(obj)
        log.append({"kernel": kernel, "blocks": int(blocks), "stream": stream,
                    "ev0": ev0, "ev1": ev1})
        return None

    api._launch = spy
    torch.npu.synchronize()
    t0 = time.perf_counter()
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    host_ms = (time.perf_counter() - t0) * 1e3
    return out, st, log, host_ms


def spans(log):
    """(kernel, blocks, span, gap) per launch, gaps within each stream."""
    seq, prev_end = [], {}
    for rec in log:
        h = rec["stream"]
        gap = None if h not in prev_end else prev_end[h].elapsed_time(rec["ev0"])
        seq.append((rec["kernel"], rec["blocks"],
                    rec["ev0"].elapsed_time(rec["ev1"]), gap, h))
        prev_end[h] = rec["ev1"]
    return seq


def main() -> None:
    api._rtc("kernels/v1/k1_pg_addwork_probe.cpp", PROBE)

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    run_call(q, k, v, g, beta, kw)
    run_call(q, k, v, g, beta, kw)

    main_handle = torch.npu.current_stream().npu_stream

    # ---- the shipped pipeline, one call, full timeline
    out, st, log, host_ms = run_call(q, k, v, g, beta, kw)
    seq = spans(log)
    main_recs = [(i, s) for i, s in enumerate(seq) if s[4] == main_handle]
    first = log[main_recs[0][0]]["ev0"]
    last = log[main_recs[-1][0]]["ev1"]
    wall = first.elapsed_time(last)
    print("production timeline: %d launches, host wall %.3f ms, device wall "
          "(main stream) %.3f ms" % (len(seq), host_ms, wall), flush=True)
    print("  #  kernel                          blk      span_ms    gap_ms  off_ms")
    for i, (kn, blk, sp, gp, h) in enumerate(seq):
        off = first.elapsed_time(log[i]["ev0"])
        tag = " " if h == main_handle else "s"
        print("  %2d%s %-30s %4d   %8.3f  %8s  %7.3f"
              % (i, tag, kn, blk, sp, "      -" if gp is None else "%8.3f" % gp, off),
              flush=True)
    print("  sums: spans %.3f + gaps %.3f = %.3f ms"
          % (sum(s[2] for s in seq if s[4] == main_handle),
             sum(s[3] for s in seq if s[4] == main_handle and s[3]),
             wall), flush=True)

    # ---- interleaved arms, timeline per round
    if not DO_ARMS:
        return
    best_pg, best_wall, best_seq = {}, {}, {}
    for r in range(REPS):
        order = ARMS if r % 2 == 0 else ARMS[::-1]
        parts = []
        for name, add, abl in order:
            out, st, log, host_ms = run_call(q, k, v, g, beta, kw, arm=(add, abl))
            seq = spans(log)
            main_recs = [(i, s) for i, s in enumerate(seq) if s[4] == main_handle]
            wall = log[main_recs[0][0]]["ev0"].elapsed_time(log[main_recs[-1][0]]["ev1"])
            pg = next(s for i, s in main_recs if s[0] == "kda_pre_gram_mix")
            parts.append("%s pg %.3f wall %.3f host %.3f" % (name, pg[2], wall, host_ms))
            best_pg[name] = min(best_pg.get(name, 1e9), pg[2])
            if wall < best_wall.get(name, 1e9):
                best_wall[name] = wall
                best_seq[name] = seq
        print("  round %d: %s" % (r, "   |   ".join(parts)), flush=True)
    print("pre_gram span MIN-of-%d: serial %.3f  scan %.3f  delta %+.3f ms | "
          "wall (event span, MIN round) serial %.3f  scan %.3f  delta %+.3f ms"
          % (REPS, best_pg["serial"], best_pg["scan"],
             best_pg["scan"] - best_pg["serial"],
             best_wall["serial"], best_wall["scan"],
             best_wall["scan"] - best_wall["serial"]), flush=True)

    # ---- per-launch comparison of the two arms (MIN round of each)
    sa_seq, sc_seq = best_seq["serial"], best_seq["scan"]
    print("  #  kernel                          blk   serial_ms   scan_ms    delta",
          flush=True)
    for i, (a, b) in enumerate(zip(sa_seq, sc_seq)):
        if a[0] != b[0]:
            print("  %2d  name mismatch %s vs %s" % (i, a[0], b[0]), flush=True)
            continue
        mark = "*" if abs(b[2] - a[2]) > 0.05 else " "
        print("  %2d%s %-30s %4d   %8.3f  %8.3f  %+8.3f"
              % (i, mark, a[0], a[1], a[2], b[2], b[2] - a[2]), flush=True)


if __name__ == "__main__":
    main()
