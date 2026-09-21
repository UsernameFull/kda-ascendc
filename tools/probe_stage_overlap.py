"""Stage-overlap bound: is there real device time in running the three stages
concurrently, and how much?

The route this measures against is "stream pre_gram -> solve -> K2 per
tile / head-group in one persistent super-kernel".  Plan section 11.17 rejected
stage/chunk pipelining with /tmp/sat.py, but that probe ran two copies of the
*same* full pipeline on two streams - a throughput statement about an already
full grid.  It says nothing about whether two *different* stages co-schedule,
and the three stages have different resource mixes (pre_gram: AIV vector bound,
Cube mostly idle; solve's AIC half: Cube only; K2: wait bound with a busy Cube).

This probe replays the captured production launch sequence by kernel name, so
every arm is the real kernel with the real args and the real grids - nothing is
rewritten, and no output is checked (the concurrent arms share their tensors on
purpose; this is the repo's established delete/keep-the-clock method).

Arms
  isolated    each stage alone
  concurrent  two stages, two streams, launched together
  floor       all four parts free at once (dependencies ignored: the ceiling)

Reported per pair: serial = the two isolated times added, hidden = serial minus
concurrent, i.e. the device time the pair actually hides.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_stage_overlap.py
"""
import faulthandler, os, sys, time
from pathlib import Path
import torch, torch_npu
faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
torch.manual_seed(1312)
q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
g = torch.randn(B, T, H, D, device=DEV) * 0.1
beta = torch.randn(B, T, H, device=DEV)
a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
bias = torch.randn(H, D, device=DEV) * 0.03
kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

seq = []
_orig = api._launch


def _spy(name, blocks, args, stream):
    seq.append((name, int(blocks), list(args)))
    return _orig(name, blocks, args, stream)


api._launch = _spy
print("warming / RTC (C=%d, H=%d)..." % (api.CHUNK, H), flush=True)
api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
torch.npu.synchronize()
api._launch = _orig
print("   captured %d launches (C=%d):" % (len(seq), api.CHUNK), flush=True)
for nm in dict.fromkeys(s[0] for s in seq):
    print("     %-32s %2d" % (nm, sum(1 for s in seq if s[0] == nm)))

PG   = [i for i, s in enumerate(seq) if s[0] == "kda_pre_gram_mix"]
SAIV = [i for i, s in enumerate(seq) if s[0] == "kda_solve_wu_wide"]
SAIC = [i for i, s in enumerate(seq) if s[0] in ("kda_solve_assemble",
                                                "kda_solve_wu_cube_kernel")]
K2   = [i for i, s in enumerate(seq) if s[0] == "kda_k2_persistent_loop"]

cur = torch_npu.npu.current_stream()
S = [torch_npu.npu.Stream(device=DEV) for _ in range(4)]


def play(idxs, stream):
    for i in idxs:
        nm, blk, ar = seq[i]
        api.launch_argsarray_engine(nm, blk, stream.npu_stream, ar, 0)


def timeit(fn, reps=7):
    xs = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return min(xs)


def one(idxs):
    return timeit(lambda: (S[0].wait_stream(cur), play(idxs, S[0]),
                           cur.wait_stream(S[0])))


def both(a, b):
    return timeit(lambda: (S[0].wait_stream(cur), S[1].wait_stream(cur),
                           play(a, S[0]), play(b, S[1]),
                           cur.wait_stream(S[0]), cur.wait_stream(S[1])))


def floor():
    def fn():
        for s in S:
            s.wait_stream(cur)
        play(PG, S[0]); play(SAIV, S[1]); play(SAIC, S[2]); play(K2, S[3])
        for s in S:
            cur.wait_stream(s)
    return timeit(fn)


parts = [("pre_gram", PG), ("solve:AIV-wide", SAIV), ("solve:AIC", SAIC), ("k2", K2)]
iso = {}
print()
print("=== isolated at the production grid (MIN of 7) ===")
for nm, idxs in parts:
    if not idxs:
        continue
    iso[nm] = one(idxs)
    print("   %-16s %7.3f ms  (%d launches)" % (nm, iso[nm], len(idxs)))

print()
print("=== concurrent, two streams ===")
for nm, (an, a), (bn, b) in [("PG || K2", ("pre_gram", PG), ("k2", K2)),
                             ("PG || solve:AIC", ("pre_gram", PG), ("solve:AIC", SAIC)),
                             ("solve:AIV || K2", ("solve:AIV-wide", SAIV), ("k2", K2)),
                             ("PG || solve:AIV", ("pre_gram", PG), ("solve:AIV-wide", SAIV)),
                             ("K2 || K2 (control)", ("k2", K2), ("k2", K2))]:
    if not a or not b:
        continue
    s = iso[an] + iso[bn]
    c = both(a, b)
    print("   %-20s serial %7.3f  concurrent %7.3f  hidden %6.3f (%4.1f%%)"
          % (nm, s, c, s - c, 100.0 * (s - c) / s))

print()
print("=== all four parts free at once (dependency-free floor) ===")
print("   %-20s stage sum %7.3f  concurrent %7.3f  hidden %6.3f (%4.1f%%)"
      % ("free-4stream", sum(iso.values()), floor(),
         sum(iso.values()) - floor(),
         100.0 * (sum(iso.values()) - floor()) / sum(iso.values())))
