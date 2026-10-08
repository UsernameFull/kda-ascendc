"""End-to-end A/B for the gate-cumsum blocked scan (docs 11.55).

k1_pre_gram_mix's gate prefix sum is a 63-step serial chain of 128-wide Adds
per chunk; 11.55's probe priced its deletion at -0.3 ms of the pre_gram
launch and a write-read-separated radix-8 blocked scan at -0.1 ms of the
launch, numerically valid but not bit-identical (out max|d| 3.05e-5 against
the gate's own 8.6e-3 margin).  This driver is the e2e step:

  * e2e MIN of N of the whole production pipeline, in the same process as
  * an interleaved pre_gram A/B through the launch spy - the probe clone's
    serial arm (addWork 0 / ablate 0, bit-identical to the shipped kernel)
    against its blocked-scan arm (ablate 8) - order rotated per round, so
    the delta is free of cross-process drift;
  * with --ref, the max|d| and relative error of the current production
    build against the saved serial baseline (out and final state).

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_cumsum_scan_e2e.py --save /tmp/pg_scan_ref.pt
  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_cumsum_scan_e2e.py --ref  /tmp/pg_scan_ref.pt
"""
from __future__ import annotations

import argparse
import faulthandler
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
REPS = int(__import__("os").environ.get("KDA_PG_E2E_REPS", "8"))
PROBE = "kda_pg_addwork_probe"
# the per-call event pair around the pre_gram launch is 2.4 ms of host
# time in the arms (measured with tools/probe_pg_host_floor.py), which is
# enough to make the arm call host-paced; KDA_PG_E2E_EVENTS=0 drops it
# for the wall verdict runs.
EVENTS = __import__("os").environ.get("KDA_PG_E2E_EVENTS", "1") == "1"
# arms as name:addWork:ablate; the ablation bits are documented in
# k1_pg_addwork_probe.cpp (1 = gate cumsum, 2 = sigmoid pass loop, 4 =
# post_gram, 8 = *serial* cumsum, the pre-11.56 form; the default is the
# blocked scan that shipped in 11.56).  Ablated arms are wrong by
# construction and are timing-only *controls* for the wall's sensitivity
# floor: if deleting 0.3-0.6 ms of device work from pre_gram does not move
# the e2e wall either, then the 0.1 ms scan never had a wall to move.
# NOTE (round 2, docs 11.58): the probe grew two more trailing ints and the
# bit-8 meaning flipped when the scan became production - the arms below
# carry the mapping for both tools.
ARMS = [tuple(a.split(":")) for a in
        __import__("os").environ.get(
            "KDA_PG_E2E_ARMS", "serial:0:8,scan:0:0").split(",")]
ARMS = [(n, int(a), int(b)) for n, a, b in ARMS]


def timed_call(call, reps):
    best = None
    host_best = None
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out, st = call()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        best = dt if best is None else min(best, dt)
        hd = (t1 - t0) * 1e3
        host_best = hd if host_best is None else min(host_best, hd)
    return best, host_best, out, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    call = lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    call()  # compile + warm
    call()
    e2e, host_ms, out, st = timed_call(call, REPS)
    print("production e2e MIN of %d: wall %.3f ms (host enqueue %.3f of it)  "
          "(compile %s)" % (REPS, e2e, host_ms, api.compile_config()), flush=True)

    # Interleaved pre_gram A/B through the launch spy: same process, same
    # inputs, order rotated - the serial arm is the probe clone at
    # (addWork, ablate) = (0, 0), the scan arm at (0, 8).
    api._rtc("kernels/v1/k1_pg_addwork_probe.cpp", PROBE)
    orig = api._launch

    def run_arm(add, abl):
        pg_ev = {}

        def spy(kernel, blocks, ksargs, stream):
            if kernel == "kda_pre_gram_mix":
                if EVENTS:
                    ev0 = torch.npu.Event(enable_timing=True)
                    ev1 = torch.npu.Event(enable_timing=True)
                    cur = torch.npu.current_stream()
                    ev0.record(cur)
                api.launch_argsarray_engine(PROBE, int(blocks), stream,
                                            list(ksargs) + [api._i(add), api._i(abl), api._i(0), api._i(0)], 0)
                if EVENTS:
                    ev1.record(cur)
                    pg_ev["pg"] = (ev0, ev1)
                return
            return orig(kernel, blocks, ksargs, stream)

        api._launch = spy
        try:
            out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        finally:
            api._launch = orig
        torch.npu.synchronize()
        pg_ms = (pg_ev["pg"][0].elapsed_time(pg_ev["pg"][1])
                 if "pg" in pg_ev else float("nan"))
        return out, st, pg_ms

    best, best_host, best_pg = {}, {}, {}
    for r in range(REPS):
        order = ARMS if r % 2 == 0 else ARMS[::-1]
        round_ms = []
        for name, add, abl in order:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _, _, pg = run_arm(add, abl)
            t1 = time.perf_counter()
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            hd = (t1 - t0) * 1e3
            round_ms.append("%s wall %.3f host %.3f pg %.3f" % (name, dt, hd, pg))
            best[name] = dt if name not in best else min(best[name], dt)
            best_host[name] = hd if name not in best_host else min(best_host[name], hd)
            best_pg[name] = pg if name not in best_pg else min(best_pg[name], pg)
        print("  round %d: %s" % (r, "  |  ".join(round_ms)), flush=True)
    for name, _, _ in ARMS:
        print("arm %-10s e2e MIN of %d %8.3f ms (host enqueue %7.3f)   "
              "pre_gram span MIN %7.3f ms"
              % (name, REPS, best[name], best_host[name], best_pg[name]), flush=True)

    if args.save:
        torch.save({"out": out.cpu(), "state": st.cpu()}, args.save)
        print("baseline saved to %s" % args.save, flush=True)
    if args.ref:
        ref = torch.load(args.ref, map_location=DEV, weights_only=True)
        for name, a, b in (("out", out, ref["out"]), ("state", st, ref["state"])):
            d = (a.float() - b.float())
            absd = float(d.abs().max().item())
            rel = float(torch.linalg.vector_norm(d) /
                        torch.linalg.vector_norm(b.float()).clamp_min(1e-30))
            print("%s vs baseline: rel %.3e abs %.3e" % (name, rel, absd), flush=True)


if __name__ == "__main__":
    main()
