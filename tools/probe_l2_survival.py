"""Does an L2 line survive a launch boundary? (the assumption under plan 11.46)

Section 11.46 priced a cold cross-launch read at 1.427 ms/GB inside the Cube
solve - 7.1x the 0.2 ms/GB the ledger used - which repriced the Level 2/3
window streaming from 0.20 ms to 0.74 + 0.50 ms and said the ordering in
sections 11.28/11.29 should be revisited.  That repricing is an *upper bound*:
it holds only if a small double-window ring, written by launch N and read once
by launch N+1, is served from L2.

What 11.46's hot arm proved is weaker - a small window re-read by 6144 blocks
*inside one launch* stays resident, and that reuse is intra-launch (one miss
over ~384 hits), so it would look free even if every launch boundary
invalidated L2.  Section 11.25 route C points the other way (2.01 GB of handoff
at 1.047 ms/GB) but 2.01 GB does not fit in L2, so it says nothing about a
ring.  This measures the ring case.

``kernels/v1/k1_l2_survival_probe.cpp`` is one kernel with a fixed grid (6144,
the Cube solve's) and a fixed 4 KB granule (the Cube's RHS burst length).  Each
block owns the same 8-tile span and ``ntile`` says how much of it is touched,
so the footprint sweeps 25.2 -> 201.3 MB with the per-block shape and the
address layout unchanged:

  tR(ntile)   consumer alone, MIN over interleaved rounds
  tW(ntile)   producer alone
  tWR(ntile)  the two back to back - the Level 2/3 pattern, write-then-read
              across one launch boundary
  tR(0)       the floor: same grid, same UB init, same epilogue, no GM traffic

The reported price is the *leg above the floor*, never the raw time, because at
this footprint the launch floor (~0.26 ms) is larger than the traffic.  A first
cut of this probe reported flat times across an 8x byte range; that was slot
reuse letting the backend drop all but one store, which is why every slot is
now read by the epilogue and why the flatness check is printed rather than
assumed.

  knee in ms/GB  => L2 survives; Level 2/3 can capture the repriced 0.74 ms
  flat ms/GB     => it does not; streaming cannot turn a handoff into an L2 hit

STATUS 2026-09-28: **this instrument does not work and the script says so
instead of reporting a number.**  Three cuts, all flat:

  cut 1  epilogue consumed 32 B of each tile     +0.003 ms for +176 MB
  cut 2  epilogue stores the whole 32 KB tile    +0.002 ms for +201 MB
  cut 3  cut 2 plus an argument echo             +0.002 ms for +201 MB

Cut 3 settles the two explanations that were still open.  The scalars *are*
delivered - block 0 stamps the ntile and mode it received into out's tail and
the host reads them back as exactly (8,1), (3,1), (8,0), (1,3) - so this is not
an argument-packing bug.  And it is not the loads alone: the epilogue's own
201 MB of stores into the 4 MB ring also costs 0.000 ms (the floor was 0.220 ms
with a 1.57 MB epilogue and 0.219 ms with a 201 MB one), so *no* GM traffic in
this kernel is registering, in either direction.  The copies are being issued
and are not costing what 201 MB must cost; the cause is not identified.

So the cross-launch L2-survival question that 11.46's 0.74 ms upper bound rests
on is **open**.  The recommended next instrument is not another standalone
microbenchmark: ``kernels/v1/k1_solve_cube_rhs_probe.cpp`` is *known* to move
real bytes (its arms span 0.615-1.422 ms and are monotone in the bytes), so the
survival arm belongs there - have a first launch write a ring-sized RHS window
and the cube transcription read it as its RHS, against the same window read
cold.  The guard below is kept so this file cannot silently start reporting a
knee verdict off a flat sweep.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_l2_survival.py
"""
from __future__ import annotations

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

DEV = torch.device("npu:0")
GRID = 6144                  # the Cube solve's grid, so the launch shape matches
TILE_B = 4096                # bytes per DataCopy, the Cube's RHS granule
NSLOT = 8                    # tiles in a block's span; UB = NSLOT * 4 KB
OUT_SLOTS = 128              # epilogue ring, 128 x 32 KB = 4 MB (see the kernel)
NTILES = [0, 1, 2, 3, 4, 6, 8]
ROUNDS, REPS = 7, 3
L2_MS_PER_GB = 0.2           # section 11.12
COLD_MS_PER_GB = 1.427       # section 11.46, measured in the Cube solve


def main() -> None:
    api._rtc("kernels/v1/k1_l2_survival_probe.cpp", "kda_l2_survival_probe")
    buf = torch.randn(GRID * NSLOT * TILE_B // 2, device=DEV).to(torch.bfloat16)
    # The epilogue writes the whole NSLOT-tile span into a ring of OUT_SLOTS,
    # so this stays small on purpose: a full-grid output would stream 201 MB of
    # dirty lines through L2 and evict the span being measured.
    echo = OUT_SLOTS * NSLOT * TILE_B // 2   # element offset of the echo tail
    out = torch.zeros(echo + 512, dtype=torch.bfloat16, device=DEV)
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    argv = api._pack_ptrs([buf, out])

    def launch(mode: int, ntile: int) -> None:
        api._launch("kda_l2_survival_probe", GRID,
                    argv + [api._i(ntile), api._i(mode)], cur_h)

    def timeit(fn, reps=REPS):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    # ---- argument liveness: what did the kernel actually receive? ------------
    # Two cuts of this probe produced a read leg of +0.003 ms for +176 MB of
    # footprint, which cannot happen if the copies run, and inferring the cause
    # from timing or from what the producer left behind did not work: that
    # readback came out erratic at *every* ntile (0 tiles at ntile=1..4 and 7,
    # tile 4 at 5, tile 5 at 6, tiles 6-7 at 8), so it could not separate "the
    # scalar arrived as 0" from "the loop ran and the stores did not land".
    #
    # So the kernel now stamps the two scalars it received into the tail of
    # ``out``, at a position that encodes the value: block 0 copies 16 elements
    # of its UB (which the arming below has made all ones) to out[echo + v*16]
    # for v = ntile and to out[echo + 256 + v*16] for v = mode.  Zero the tail,
    # launch once, and the nonzero window *is* the value the kernel saw.  This
    # reads the argument, not its consequences.
    #
    # torch_npu does not order its own elementwise ops against the raw
    # aclrtLaunchKernel calls (api.py says so at the eye-tile cache), so the
    # arming is published with a synchronize before every diagnostic launch.
    t0e = TILE_B // 2                       # bf16 elements per tile
    span = NSLOT * t0e                      # elements in one block's span

    def decode(mode_arg, ntile_arg):
        buf.zero_()
        buf[:span] = 1.0        # block 0's span, i.e. its UB source, is all ones
        out.zero_()
        torch.npu.synchronize()
        launch(mode_arg, ntile_arg)
        torch.npu.synchronize()
        tail = out[echo:echo + 512].float().abs()
        wins = [tail[:256].reshape(16, 16).amax(dim=1).tolist(),
                tail[256:512].reshape(16, 16).amax(dim=1).tolist()]
        return [[i for i, v in enumerate(w) if v > 0.5] for w in wins]

    print()
    print("  argument echo (block 0 stamps the scalars it received into out's tail):")
    ok = True
    for mode_arg, ntile_arg in ((1, NSLOT), (1, 3), (0, NSLOT), (3, 1)):
        nt_seen, md_seen = decode(mode_arg, ntile_arg)
        good = nt_seen == [ntile_arg % 16] and md_seen == [mode_arg % 16]
        ok = ok and good
        print("     launch(mode=%d, ntile=%d) -> kernel saw ntile=%s mode=%s%s"
              % (mode_arg, ntile_arg, nt_seen or "none", md_seen or "none",
                 "" if good else "   MISMATCH"))
    if not ok:
        print("  => THE SCALARS ARE NOT ARRIVING AS PACKED.  Every number below would be")
        print("     a kernel doing no span traffic at some ntile, so the sweep is not run.")
        print("     launch_argsarray_engine hands one blob per parameter positionally, so")
        print("     the fault is in the arg list or the signature order, not the timing.")
        return
    buf.normal_()
    out.zero_()
    torch.npu.synchronize()

    print("L2 cross-launch survival, grid %d, %d B granule, %d-tile span, "
          "epilogue ring %d x %d KB = %.0f MB"
          % (GRID, TILE_B, NSLOT, OUT_SLOTS, NSLOT * TILE_B // 1024,
             OUT_SLOTS * NSLOT * TILE_B / 1e6), flush=True)
    for nt in NTILES:
        for m in (0, 1):
            launch(m, nt)
    torch.npu.synchronize()

    # Round-robin over every arm: the arms span 8x in footprint, so a drifting
    # clock would otherwise show up as the knee.
    tR = {nt: 1e9 for nt in NTILES}
    tW = {nt: 1e9 for nt in NTILES}
    tWR = {nt: 1e9 for nt in NTILES}
    for _ in range(ROUNDS):
        for nt in NTILES:
            tR[nt] = min(tR[nt], timeit(lambda n=nt: launch(0, n)))
            tW[nt] = min(tW[nt], timeit(lambda n=nt: launch(1, n)))
            tWR[nt] = min(tWR[nt], timeit(lambda n=nt: (launch(1, n), launch(0, n))))

    floor = tR[0]
    print()
    print("  launch floor tR(0) = %.3f ms (same grid, same UB init, same epilogue, no GM traffic)"
          % floor)
    print()
    print("  footprint    tR ms    tW ms   tWR ms   read leg   leg after W   ms/GB(R)   ms/GB(W->R)")
    rows = []
    for nt in NTILES:
        gb = GRID * nt * TILE_B / 1e9
        leg_r = tR[nt] - floor
        leg_wr = tWR[nt] - tW[nt] - floor
        p_r = leg_r / gb if gb > 0 else 0.0
        p_wr = leg_wr / gb if gb > 0 else 0.0
        if gb > 0:
            rows.append((gb * 1e3, nt, leg_r, leg_wr, p_r, p_wr))
        print("  %7.1f MB  %7.3f  %7.3f  %7.3f   %7.3f     %7.3f     %8.3f     %8.3f"
              % (gb * 1e3, tR[nt], tW[nt], tWR[nt], leg_r, leg_wr, p_r, p_wr), flush=True)

    print()
    print("verdict arithmetic")
    # The leg at the largest footprint has to be a real HBM read.  If it is not,
    # nothing below means anything and the script must not print a knee verdict.
    big_gb = rows[-1][0] / 1e3
    predicted = big_gb * COLD_MS_PER_GB
    if rows[-1][2] < 0.25 * predicted:
        print("  INSTRUMENT INVALID: the read leg at %.0f MB is %+.3f ms, but %.3f GB of"
              % (rows[-1][0], rows[-1][2], big_gb))
        print("  cold GM reads cannot cost less than ~%.3f ms at the 11.46 price of %.2f ms/GB."
              % (predicted, COLD_MS_PER_GB))
        print("  No GM traffic in this kernel is registering.  The argument echo above proves")
        print("  the scalars arrive, and the epilogue's own 201 MB of stores cost 0.000 ms too")
        print("  (the floor is the same with a 1.57 MB epilogue), so this is not a liveness or")
        print("  an argument-packing bug and it is not the loads alone.  Cause not identified.")
        print("  Next instrument: put the survival arm inside k1_solve_cube_rhs_probe.cpp,")
        print("  which is known to move real bytes, instead of a standalone microbenchmark.")
        return
    for name, li, pi in (("R alone (read-then-read)", 2, 4),
                         ("W then R (write-then-read)", 3, 5)):
        lo = min(rows, key=lambda r: r[pi])
        hi = rows[-1]
        ratio = hi[pi] / lo[pi] if lo[pi] > 0 else float("inf")
        print("  %-26s cheapest %.3f ms/GB at %.0f MB, dearest %.3f ms/GB at %.0f MB, "
              "spread %.1fx" % (name, lo[pi], lo[0], hi[pi], hi[0], ratio))
        print("     references: 11.12 L2-local %.2f ms/GB, 11.46 cold cross-launch %.2f ms/GB"
              % (L2_MS_PER_GB, COLD_MS_PER_GB))
        if ratio > 2.0 and hi[pi] > 2.0 * L2_MS_PER_GB:
            fit = [r for r in rows if r[pi] <= 1.5 * L2_MS_PER_GB]
            print("     => knee present.  Up to ~%.0f MB the read leg is at or below the L2"
                  % (fit[-1][0] if fit else rows[0][0]))
            print("        price, so a ring-sized window *is* served from L2 across a launch")
            print("        boundary and 11.46's repriced Level 2/3 (0.74 + 0.50 ms) is reachable.")
        elif ratio <= 2.0:
            print("     => flat at %.2f-%.2f ms/GB across an %.0fx footprint sweep: L2 does NOT"
                  % (lo[pi], hi[pi], rows[-1][0] / rows[0][0]))
            print("        survive the launch boundary.  Every cross-launch byte is HBM-priced,")
            print("        so window streaming cannot turn a handoff into an L2 hit and the")
            print("        repriced 0.74 ms is not capturable that way - the bytes have to be")
            print("        deleted, not rearranged.")
        else:
            print("     => partial: the small end is cheaper but never near the L2 price, so")
            print("        streaming would recover only part of the repriced 0.74 ms.")
    print("  flatness check (a flat leg means the loads were optimised away, not that they")
    print("  were free): tR grows %.2fx from %.0f MB to %.0f MB, i.e. %+.3f ms for %+.0f MB"
          % (tR[NTILES[-1]] / tR[NTILES[1]], rows[0][0], rows[-1][0],
             rows[-1][2] - rows[0][2], rows[-1][0] - rows[0][0]))


if __name__ == "__main__":
    main()
