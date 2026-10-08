"""Wave-balance sweep of pre_gram's launch grid, on device events through the production path.

Context: the 910B3 reports cube_core_num 20 (torch_npu device properties) and
the device agrees - the shipped 96 MIX blocks run ceil(96/20) = 5 waves
(msprof Task 4112 us at an 819.4 us block wall; device span ~4.0 ms), and a
24-block replay lands at ~6.3 ms, i.e. 2 waves of 256 steps, not one.  The
"24 AICs" reading in api.py's old sweep comment is the old 910_9382 golden
device.  So the shipped 96 x 64 grid burns ~12 of its 320 block-steps on the
last wave (16 blocks on 20 cores), and any grid divisible by 20 with the same
6144 total steps lands at 308-310 steps - 0.1-0.15 ms - even though R3 picked
96 because every reading in that old sweep was host-synced.

Instrument: full production calls with only KDA_PRE_UNROLL flipped between
calls (api reads it per call); the kda_pre_gram_mix launch of each call is
bracketed by a pair of device events on its own stream, arms interleave
round-robin and reduce by MIN over rounds.  A direct re-launch of the
captured kernel through api.launch_argsarray_engine was tried first and
dropped 7/12 arms with 0.000 spans (silently); do not use that path as an
instrument for this kernel.  The final block builds its output twice
(out-of-range chunks clamp to nchunk - 1), which stays bit-identical.

Arms, u -> grid = ceil(12288 / (2u)); makespan in block-steps:
  64 -> 96  = 5 x 64 = 320  (shipped, ~4.00 ms)
  77 -> 80  = 4 x 77 = 308
  62 -> 100 = 5 x 62 = 310
 103 -> 60  = 3 x 103 = 309
 154 -> 40  = 2 x 154 = 308
 308 -> 20  = 1 x 308 = 308
 128 -> 48  = 3 waves (C=20, 384) vs 2 waves (C=24, 256) - crosscheck
 256 -> 24  = 2 waves (C=20, 512) vs 1 wave  (C=24, 256) - crosscheck

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_grid_sweep.py
"""
from __future__ import annotations

import faulthandler
import os
import struct
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
CHUNKS = 128 * 96            # B * H * T/CHUNK = 12288 at KDA_CHUNK=64
ROUNDS = int(os.environ.get("KDA_PGS_ROUNDS", "4"))
KERNEL = "kda_pre_gram_mix"
U_OF_GRID = {96: 64, 80: 77, 100: 62, 60: 103, 40: 154, 20: 308,
             48: 128, 24: 256}


def run_span(q, k, v, g, beta, kw):
    """One full call; device-event span of its pre_gram launch in ms."""
    cur = torch.npu.current_stream()
    rec = {}
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == KERNEL:
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            rec["u"] = struct.unpack("<i", args[-5])[0]
            ev0.record(cur)
            orig(kernel, blocks, args, stream)
            ev1.record(cur)
            rec["grid"], rec["ev0"], rec["ev1"] = int(blocks), ev0, ev1
            return None
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    return rec, out, st


def main() -> None:
    props = torch_npu.npu.get_device_properties(0)
    print("device %s: cube_core_num %d, vector_core_num %d, L2 %.0f MB"
          % (props.name, props.cube_core_num, props.vector_core_num,
             props.L2_cache_size / 2 ** 20), flush=True)

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    os.environ["KDA_PRE_UNROLL"] = "64"
    run_span(q, k, v, g, beta, kw)          # warm + compile

    arms = [64, 77, 62, 103, 154, 308, 128, 256]
    span, grid_seen = {}, {}
    for r in range(ROUNDS):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        pend = []
        for u in order:
            os.environ["KDA_PRE_UNROLL"] = str(u)
            rec, _, _ = run_span(q, k, v, g, beta, kw)
            assert rec["u"] == u, "unroll slot read back as %d" % rec["u"]
            grid_seen[u] = rec["grid"]
            pend.append((u, rec["ev0"], rec["ev1"]))
        torch.npu.synchronize()
        for u, ev0, ev1 in pend:
            dt = float(ev0.elapsed_time(ev1))
            span[u] = dt if u not in span else min(span[u], dt)

    print("pre_gram span, full-call path, device events, MIN of %d:" % ROUNDS,
          flush=True)
    print("  %5s %5s %9s %9s %9s"
          % ("u", "grid", "span_ms", "waves20", "pred_ms20"), flush=True)
    for u in arms:
        gr = grid_seen[u]
        assert gr == -(-CHUNKS // (2 * u)), (u, gr)
        waves = -(-gr // 20)
        print("  %5d %5d %9.3f %9d %9.3f"
              % (u, gr, span[u], waves, waves * u * 0.01253), flush=True)

    # the production default (no env) should land on the winning shape
    os.environ.pop("KDA_PRE_UNROLL", None)
    rec, _, _ = run_span(q, k, v, g, beta, kw)
    print("default (no KDA_PRE_UNROLL): u %d grid %d" % (rec["u"], rec["grid"]),
          flush=True)

    # Paired e2e: alternate the shipped 64 against the winner in one process
    # (the 11.56 instrument - a cross-process host wall cannot resolve 0.1 ms).
    reps = int(os.environ.get("KDA_PGS_E2E_REPS", "10"))
    wall = {}
    for r in range(reps):
        for u in ((64, 77) if r % 2 == 0 else (77, 64)):
            os.environ["KDA_PRE_UNROLL"] = str(u)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            run_span(q, k, v, g, beta, kw)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            wall[u] = dt if u not in wall else min(wall[u], dt)
    print("e2e host wall, paired interleaved MIN of %d: u=64 %.3f  u=77 %.3f"
          "  delta %+.3f ms" % (reps, wall[64], wall[77], wall[77] - wall[64]),
          flush=True)

    # numerics: the shipped 64 against the best divisible-by-20 arm
    cand = [(u, span[u]) for u in (77, 62, 103, 154, 308)]
    best = min(cand, key=lambda t: t[1])[0]
    os.environ["KDA_PRE_UNROLL"] = "64"
    _, out_a, st_a = run_span(q, k, v, g, beta, kw)
    os.environ["KDA_PRE_UNROLL"] = str(best)
    _, out_b, st_b = run_span(q, k, v, g, beta, kw)
    print("numerics 64 vs %d: out diff %d / %d, state diff %d / %d"
          % (best, int((out_a != out_b).sum().item()), out_a.numel(),
             int((st_a != st_b).sum().item()), st_a.numel()), flush=True)


if __name__ == "__main__":
    main()
