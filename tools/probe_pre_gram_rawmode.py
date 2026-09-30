"""What do pre_gram's two raw fp32 Gram tiles cost to store?

kda_pre_gram_mix's paired Cube fixpipes the raw Grams into the Aqk32 and L
slots; the AIV band loop then reads both back to mask, scale and round them
into Aqk16 and L masked.  Both round trips are *intra-launch* - the AIC writes
a chunk and the AIV of the same MIX block reads it behind
CrossCoreWaitFlag(FL_DONE) - so they are priced by section 11.12's L2 constant,
not by section 11.46's 1.427 ms/GB cross-launch read.  Section 11.28 left the
pair on the table as "+0.04~0.10 ms, needs the AIV band loop touched"; this
prices the store half of it before anyone touches that loop.

Arms (k1_pre_gram_mix's rawMode runtime arg, api.pre_raw_mode(),
KDA_PRE_RAW_MODE):

  0  shipped, both fixpipes                       control
  1  drop the Aqk32 fixpipe                       -201.33 MB of stores
  2  drop the L fixpipe                           -201.33 MB of stores
  3  drop both                                    -402.65 MB of stores

A dropping arm is *wrong by construction*: the slots stay allocated and the AIV
still reads them, so it reads whatever the device memory holds.  That is this
repo's established delete-the-work-keep-the-clock method (section 11.35's cube
arm 2, section 4.7's a16Mode 2).  Nothing here is a shipping mode and no output
is checked.  rawMode 0 is the only value api.py ships.

Two granularities are reported because section 11.38 showed they disagree: the
stage (KDA_PROFILE=1's pre_gram_ms, which synchronises around the stage) and
the clean e2e wall (no profile, no extra syncs).

Guards, all printed: mode 3 must not be slower than mode 0 (deleting stores
cannot cost time), and mode 3's delta must be about the sum of modes 1 and 2
(the two tiles are independent stores; if they are not additive the price is a
scheduling artefact, not a byte price).

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pre_gram_rawmode.py
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
CH = api.CHUNK
MODES = [0, 1, 2, 3]
ROUNDS = int(os.environ.get("KDA_RAWMODE_ROUNDS", "5"))
TILE_BYTES = 201326592            # c * CHUNK * CHUNK * 4 at [1,8192,96,128], C=64
NOISE_STAGE = 0.006               # section 11.47's pre_gram noise floor


def call(kw):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    api.kda_bt16_fwd_ascendc(**kw)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    torch.manual_seed(20260928)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(q=q, k=k, v=v, g=g, beta=beta,
              A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    c = B * H * (T // CH)
    tile = c * CH * CH * 4
    print("pre_gram rawMode ablation, C=%d, [%d,%d,%d,%d], c=%d" %
          (CH, B, T, H, D, c), flush=True)
    print("  one raw fp32 tile = %.2f MB (expected %.2f MB); arms drop 0/1/1/2 of them"
          % (tile / 1e6, TILE_BYTES / 1e6), flush=True)
    if tile != TILE_BYTES:
        print("  note: geometry differs from the ledger's, so the ms/GB below uses "
              "the measured tile size", flush=True)

    print("warming / RTC ...", flush=True)
    os.environ["KDA_PRE_RAW_MODE"] = "0"
    os.environ["KDA_PROFILE"] = "0"
    call(kw)

    # Pass 1: stage granularity.  KDA_PROFILE=1 puts a torch.npu.synchronize()
    # around every stage, so its e2e wall is inflated and only the per-stage
    # numbers are usable here.
    os.environ["KDA_PROFILE"] = "1"
    stage = {m: 1e9 for m in MODES}
    stage_wall = {m: 1e9 for m in MODES}
    # Per-mode, not the loop's last call: the other two stages have to be
    # readable for the mode being reported, and a dropping arm feeds garbage
    # downstream, so "what did solve cost" is part of the evidence.
    prof = {m: {} for m in MODES}
    for _ in range(ROUNDS):
        for m in MODES:
            os.environ["KDA_PRE_RAW_MODE"] = str(m)
            w = call(kw)
            pr = api.get_last_profile()
            stage[m] = min(stage[m], float(pr["pre_gram_ms"]))
            stage_wall[m] = min(stage_wall[m], w)
            if float(pr["pre_gram_ms"]) == stage[m]:
                prof[m] = dict(pr)

    # Pass 2: clean e2e, no profile syncs.
    os.environ["KDA_PROFILE"] = "0"
    e2e = {m: 1e9 for m in MODES}
    for _ in range(ROUNDS):
        for m in MODES:
            os.environ["KDA_PRE_RAW_MODE"] = str(m)
            e2e[m] = min(e2e[m], call(kw))
    os.environ["KDA_PRE_RAW_MODE"] = "0"

    print()
    for m in MODES:
        print("  mode %d profiled stages: %s"
              % (m, ", ".join("%s %.3f" % (kk[:-3], vv)
                              for kk, vv in sorted(prof[m].items())
                              if kk.endswith("_ms") and kk != "total_ms")), flush=True)
    print()
    print("  mode  dropped stores    pre_gram_ms   d vs 0    ms/GB     e2e_ms   d vs 0")
    rows = {}
    for m in MODES:
        gb = (m & 1) * tile / 1e9 + (m >> 1) * tile / 1e9
        d_st = stage[m] - stage[0]
        d_e2 = e2e[m] - e2e[0]
        per = (d_st / gb) if gb > 0 else 0.0
        rows[m] = (gb, d_st, d_e2, per)
        print("  %4d  %10.2f MB   %10.3f  %+8.3f  %8.3f  %9.3f  %+7.3f"
              % (m, gb * 1e3, stage[m], d_st, per, e2e[m], d_e2), flush=True)

    print()
    st = {m: rows[m][1] for m in MODES}
    slower = all(st[m] > NOISE_STAGE for m in (1, 2, 3))
    faster = st[3] < -NOISE_STAGE
    free = abs(st[3]) <= NOISE_STAGE
    add = st[1] + st[2]
    additive = abs(st[3] - add) <= max(3 * NOISE_STAGE, 0.3 * abs(add) + NOISE_STAGE)

    if slower:
        print("  => every dropping arm is SLOWER, and monotonically so in the bytes "
              "dropped (+%.3f Aqk32, +%.3f L, +%.3f both; additive to %.3f).  "
              "Deleting a store cannot cost bandwidth, so this ablation is not a "
              "clean byte deletion: the two Fixpipes are also pacing points.  "
              "Dropping them lets FIX_M's Set/Wait pair retire with no fixpipe "
              "outstanding, so qc.FreeTensor(cf0/cf1) lands earlier and the next "
              "Mmad meets the L0C WAR hazard that the fixpipe's latency used to "
              "cover - and the AIC is handshaked to the AIV per step "
              "(FL_READY/FL_DONE), so a throttled AIC throttles the stage.  Same "
              "shape of finding as section 11.44's MTE3_V marker."
              % (st[1], st[2], st[3], add), flush=True)
        print()
        print("  => VERDICT: the 402.65 MB of raw fp32 Gram stores is NOT a "
              "candidate, and it is not free either - it is load-bearing.  "
              "Section 11.28's 'Aqk32 raw re-read, +0.04~0.10 ms' is CLOSED "
              "from the store side: do not split the fixpipe into quadrants and "
              "do not move mask/scale to the Cube on the strength of a byte "
              "count.  The ablation also cannot price the *read* side (the AIV "
              "reads both slots in every arm), so the read half of that "
              "candidate stays unpriced - and it is intra-launch, hence L2-priced "
              "at ~0.2 ms/GB, hence <= 0.08 ms for both tiles.", flush=True)
    elif faster:
        print("  => VERDICT: both raw tiles together are worth %.3f ms of the "
              "pre_gram stage (%.3f ms/GB) and %.3f ms of e2e.  Additive in the "
              "two tiles: %.3f vs %.3f + %.3f.  A quadrant-split fixpipe could "
              "capture the upper-right quadrant the downstream gathers never "
              "read (k1_solve_wu_wide.cpp:165 reads the diagonal sub-blocks and "
              ":173 the (1,0) block), i.e. ~25%%."
              % (-st[3], -rows[3][3], -rows[3][2], st[3], st[1], st[2]), flush=True)
    elif free:
        print("  => VERDICT: both raw tiles together are inside the %.3f ms noise "
              "floor of the pre_gram stage (%.3f ms), and %.3f ms in e2e.  The "
              "402.65 MB of raw Gram stores is free: the Cube is the hidden pipe "
              "of this stage, so deleting its stores buys nothing.  Section "
              "11.28's 'Aqk32 raw re-read' candidate is CLOSED."
              % (NOISE_STAGE, st[3], rows[3][2]), flush=True)
    else:
        print("  => MIXED: mode 3 is %.3f ms but modes 1/2 are %+.3f/%+.3f, so the "
              "two tiles do not behave alike.  Do not aggregate them; re-run with "
              "more rounds before drawing anything." % (st[3], st[1], st[2]),
              flush=True)

    if not additive:
        print("  => SUSPECT: mode 3 (%.3f) is not the sum of modes 1+2 (%.3f), so "
              "the two tiles interact and the per-tile numbers are not "
              "independent prices." % (st[3], add), flush=True)


if __name__ == "__main__":
    main()
