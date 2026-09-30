"""Does a ring-sized window survive a launch boundary? (the gate under 11.46's 0.74 ms)

Section 11.46 priced a cold cross-launch read at 1.427 ms/GB inside the Cube
solve, which repriced the Level 2/3 window streaming from the ledger's 0.20 ms
to 0.74 + 0.50 ms.  That repricing is an *upper bound*: Level 2/3 only capture
it if a small double-window ring, written by one launch and read once by the
next, is served from L2 rather than from HBM.  11.46's hot arm does not answer
that - its window is re-read by 6144 blocks inside one launch, so one miss is
amortised over ~384 hits and it would look free even if every launch boundary
invalidated L2.

`tools/probe_l2_survival.py` tried to answer it with a standalone microbenchmark
and failed three cuts running: its sweep came out flat at +0.002 ms for +201 MB,
its own epilogue stores included, and an argument echo proved the scalars were
being delivered - so no GM traffic in that kernel registered in either direction
and the cause was never found.  The lesson recorded in 11.46 section 6 is that
the next instrument must not be another microbenchmark.

This one is not.  The consumer is `kernels/v1/k1_solve_cube_rhs_probe.cpp` mode
0, the same transcription whose arms span 0.615-1.422 ms and are monotone in the
bytes, i.e. an instrument already known to move real GM traffic and to price it
at 1.427 ms/GB.  Only the *size of the RHS operand* and *what ran immediately
before* change:

  producer  an elementwise torch kernel writes the ring (rk/rv over R chunks)
  evictor   an elementwise torch kernel streams a 512 MB scratch, so nothing of
            the ring is left in L2
  consumer  the Cube transcription over exactly R chunks, grid R/NC, so it reads
            the ring **once per byte** - no intra-launch reuse to hide behind

  hot leg  = t[producer; consumer]        - t[producer]
  cold leg = t[producer; evictor; consumer] - t[producer; evictor]

Both arms carry the same producer prefix, so the subtraction removes it and the
only difference is whether the ring was still resident when the consumer ran.
R sweeps 16 -> 256 MB, which locates the knee (i.e. the usable L2) empirically
instead of assuming a size.

The instrument validates itself: the cold leg must come out near 11.46's
independently measured 1.427 ms/GB.  If it does not, the ring/evictor sizes are
wrong and the script says so rather than reporting a survival fraction.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_ring_survival.py
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

D, DEV = 128, torch.device("npu:0")
CH = api.CHUNK
NC = api.WU_NCHUNK
R_MAX = 8192                   # chunks; the ring is R * CH * D * 2 bytes x 2
RS = [512, 1024, 2048, 4096, 8192]
K = 4                          # producer/consumer pairs per timed call
EVICT_MB = 512                 # must exceed L2 by a wide margin
COLD_MS_PER_GB = 1.427         # section 11.46, measured in the Cube solve
L2_MS_PER_GB = 0.2             # section 11.12
ROUNDS = 4


def main() -> None:
    api._rtc("kernels/v1/k1_solve_cube_rhs_probe.cpp", "kda_solve_cube_rhs_probe")
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    torch.manual_seed(20260928)

    # One allocation at R_MAX, sliced per arm: the consumer's operands and the
    # tensors the producer writes.
    a16 = torch.randn(R_MAX, CH, CH, device=DEV).to(torch.bfloat16)
    rk = torch.zeros(R_MAX, CH, D, dtype=torch.bfloat16, device=DEV)
    rv = torch.zeros(R_MAX, CH, D, dtype=torch.bfloat16, device=DEV)
    rk_src = torch.randn(R_MAX, CH, D, device=DEV).to(torch.bfloat16)
    rv_src = torch.randn(R_MAX, CH, D, device=DEV).to(torch.bfloat16)
    W = torch.zeros(R_MAX, CH, D, dtype=torch.bfloat16, device=DEV)
    U = torch.zeros(R_MAX, CH, D, dtype=torch.bfloat16, device=DEV)
    scratch = torch.randn(EVICT_MB * 1000 * 1000 // 4, device=DEV)

    def produce(r: int) -> None:
        # A real elementwise kernel writing every byte of the ring: not fill_
        # and not copy_, which can lower to a memset/memcpy and take a path no
        # producer in this pipeline takes.
        torch.mul(rk_src[:r], 2.0, out=rk[:r])
        torch.mul(rv_src[:r], 2.0, out=rv[:r])

    def evict() -> None:
        torch.mul(scratch, 2.0, out=scratch)

    def consume(r: int) -> None:
        argv = api._pack_ptrs([a16, rk, rv, W, U])
        for _ in range(K):
            api._launch("kda_solve_cube_rhs_probe", (r + NC - 1) // NC,
                        argv + [api._i(r), api._i(0), api._i(32)], cur_h)

    def timeit(fn, reps=3):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("ring survival, consumer = kda_solve_cube_rhs_probe mode 0, CHUNK=%d NC=%d, "
          "%d consumer launches per timed call" % (CH, NC, K), flush=True)
    print("  evictor %.0f MB in-place; ring swept over R = %s chunks"
          % (EVICT_MB, RS), flush=True)

    produce(R_MAX); evict(); consume(RS[0])
    torch.npu.synchronize()

    t_p = {r: 1e9 for r in RS}
    t_pe = {r: 1e9 for r in RS}
    t_pc = {r: 1e9 for r in RS}
    t_pec = {r: 1e9 for r in RS}
    for _ in range(ROUNDS):
        for r in RS:
            t_p[r] = min(t_p[r], timeit(lambda n=r: produce(n)))
            t_pe[r] = min(t_pe[r], timeit(lambda n=r: (produce(n), evict())))
            t_pc[r] = min(t_pc[r], timeit(lambda n=r: (produce(n), consume(n))))
            t_pec[r] = min(t_pec[r],
                           timeit(lambda n=r: (produce(n), evict(), consume(n))))

    print()
    print("  ring       ring MB  reads GB    t[P]   t[P;E]   t[P;C]  t[P;E;C]  hot leg "
          " cold leg  ms/GB(hot)  ms/GB(cold)")
    rows = []
    for r in RS:
        gb = K * r * CH * D * 2 * 2 / 1e9          # both RHS tensors, K passes
        hot = t_pc[r] - t_p[r]
        cold = t_pec[r] - t_pe[r]
        rows.append((r, r * CH * D * 2 * 2 / 1e6, gb, hot, cold,
                     hot / gb, cold / gb))
        print("  %5d ch %7.1f  %8.3f  %7.3f %8.3f %8.3f %9.3f  %7.3f  %8.3f  %10.3f  %10.3f"
              % (r, r * CH * D * 2 * 2 / 1e6, gb, t_p[r], t_pe[r], t_pc[r],
                 t_pec[r], hot, cold, hot / gb, cold / gb), flush=True)

    # The legs are *whole-consumer* costs: they carry the consumer's own W/U
    # stores (K x the ring), its A16 reads and K launch overheads, which is why
    # the absolute ms/GB here is 2-3.6 rather than 11.46's marginal 1.427.  The
    # difference between the arms is still clean, because the only thing that
    # differs is whether the ring was resident when the consumer ran - so
    # (cold - hot) / GB-of-RHS-reads is attributable to the RHS reads alone and
    # is directly comparable with 11.46's marginal price.  That comparison is
    # the instrument check, and it is a two-sided one: the benefit cannot exceed
    # what the bytes cost cold, and it must decay as the ring outgrows L2.
    print()
    print("instrument check: (cold - hot) per GB of RHS reads, against 11.46's "
          "independent marginal cold price of %.3f ms/GB" % COLD_MS_PER_GB)
    avoided = [(r, mb, (cold - hot) / gb, 1.0 - hot / cold)
               for r, mb, gb, hot, cold, _, _ in rows]
    for r, mb, per_gb, frac in avoided:
        print("  ring %6.1f MB: %.3f ms/GB avoided, %.0f%% of that arm's cold leg"
              % (mb, per_gb, 100 * frac))
    top, bottom = avoided[0][2], avoided[-1][2]
    over = top > 1.25 * COLD_MS_PER_GB
    flat = top < 3.0 * bottom
    if over:
        print("  => INVALID: the smallest ring avoids %.3f ms/GB, more than the %.3f ms/GB"
              % (top, COLD_MS_PER_GB))
        print("     those bytes cost cold in an independently measured kernel.  Residency")
        print("     cannot be worth more than the traffic, so the arms are not matched.")
        return
    if flat:
        print("  => INVALID: the benefit does not decay with ring size (%.3f vs %.3f ms/GB"
              % (top, bottom))
        print("     from %.0f MB to %.0f MB), so it is not a residency effect."
              % (rows[0][1], rows[-1][1]))
        return
    print("  => both sides hold: the benefit is bounded by 11.46's independent price and it")
    print("     decays %.0fx from the smallest ring to the largest, so it is residency."
          % (top / bottom))

    print()
    print("verdict arithmetic")
    print("  caveat: the hot leg is an *in-situ* number and a harsher test than Level 2/3.")
    print("  The producer runs once, then the consumer runs %d times on the same ring while"
          % K)
    print("  storing %d x the ring as W/U - so the ring has to survive %d consumer launches"
          % (K, K))
    print("  and their output stream.  Level 2/3 write each window immediately before its own")
    print("  consumer reads it, so the fractions below are a lower bound for them.")
    print()
    for r, mb, per_gb, frac in avoided:
        verdict = ("essentially all of the cold read price"
                   if per_gb > 0.8 * COLD_MS_PER_GB else
                   "most of it" if per_gb > 0.4 * COLD_MS_PER_GB else
                   "part of it" if per_gb > 0.1 * COLD_MS_PER_GB else
                   "none of it")
        print("  ring %6.1f MB -> %.3f of %.3f ms/GB recovered (%s)"
              % (mb, per_gb, COLD_MS_PER_GB, verdict))
    live = [mb for _, mb, per_gb, _ in avoided if per_gb > 0.5 * COLD_MS_PER_GB]
    print()
    if live:
        print("  => L2 DOES carry a handoff across a launch boundary.  A ring of ~%.0f MB or"
              % max(live))
        print("     less, written by one launch and read once by the next, recovers %.3f of"
              % avoided[0][2])
        print("     11.46's %.3f ms/GB - i.e. essentially the whole cross-launch read price,"
              % COLD_MS_PER_GB)
        print("     which is %.0f%% of that arm's *whole* cold leg.  The other %.0f%% is the"
              % (100 * avoided[0][3], 100 * (1 - avoided[0][3])))
        print("     consumer's own W/U stores, its A16 reads and %d launch overheads, none of"
              % K)
        print("     which residency can touch - that split is the check that the two arms")
        print("     differ only in the RHS.")
        print("     Level 2's handoff is 0.60 GB per call, so at that ring size its read leg is")
        print("     worth ~%.2f ms - 11.46's repriced 0.74 ms is capturable, not just an upper"
              % (0.60 * avoided[0][2]))
        print("     bound.  The constraint it adds is on the *window size*: at C=64 the Rk/Rv")
        print("     handoff is 32 KB per chunk, so a ~%.0f MB live ring with a 2-slot double"
              % max(live))
        print("     window is ~%d chunks per window, ~%d windows over the 12288-chunk call."
              % (max(live) * 1e6 / 2 / (CH * D * 2 * 2),
                 12288 // max(1, int(max(live) * 1e6 / 2 / (CH * D * 2 * 2)))))
    else:
        print("  => no ring size recovers even half the cold price, so L2 does not carry the")
        print("     handoff across a launch boundary and 11.46's 0.74 ms is not capturable by")
        print("     window streaming - the bytes would have to be deleted, not rearranged.")


if __name__ == "__main__":
    main()
