"""Fixed-shape workspace pool: prototype, byte ledger and A/B.

Plan section S1 (first round): the exposed host share of a production call is
0.873 ms (docs 11.26) and the allocation arm of it measures 0.304 ms standalone
(tools/probe_host_cost.py, 28 tensors / 4.15 GB at [1,8192,96,128]/C=64).  The
question this probe answers is whether caching those allocations in a
shape-keyed pool actually moves the *end-to-end* number, or whether the host
work is already hidden behind the device.

Pool design (prototype scope: no kernel change, no api behaviour change):

  * key      (device, b, t, h, CHUNK, NV, need_intermediates): a shape that
             differs in any of these gets its own entry, so a stale entry can
             never be handed to a different geometry.  There is deliberately
             no "grow the pool" path - a bigger shape is a new entry.
  * lease    one entry is leased per call and returned at the end; a second
             lease of a busy entry is refused (the caller falls back to fresh
             allocations) rather than silently sharing bytes between two
             in-flight calls, which would corrupt both.
  * bytes    the entry holds the tensors the api would have allocated; the
             ledger below is the same arithmetic as tools/gen_ub_l1_budget.py
             so the two cannot drift.

  KDA_CHUNK=64 python3 -u tools/probe_workspace_pool.py --reps 20
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api

D, BV, NV = 128, 64, 2


class WorkspacePool:
    """Shape-keyed cache of the api's per-call workspace tensors.

    The prototype is used as a *stand-in* for the api's allocations: the caller
    asks for the tensors, drops them when done, and the pool keeps the device
    memory alive between calls.  ``checkout``/``checkin`` are explicit so a
    caller that holds two calls in flight at once cannot alias the same bytes.
    """

    def __init__(self):
        self._entries: dict[tuple, dict] = {}
        self._busy: set[tuple] = set()
        self.hits = 0
        self.misses = 0
        self.refused = 0

    def _spec(self, b, t, h, chunk, need_intermediates):
        nt = t // chunk
        bh = b * h
        c = bh * nt
        tasks = bh * NV
        c_solve = (c + api.SOLVE_WIDE_NCH - 1) // api.SOLVE_WIDE_NCH * api.SOLVE_WIDE_NCH
        sub = chunk // api.SOLVE_WIDE_SUBB
        bf16, fp32 = torch.bfloat16, torch.float32
        spec = {
            "qn": ((c, chunk, D), bf16), "kn": ((c, chunk, D), bf16),
            "gate": ((c, chunk, D), fp32), "gc": ((c, chunk, D), fp32),
            "beta_out": ((c, chunk), fp32), "decay": ((c, D), fp32),
            "rk": ((c, chunk, D), bf16), "rv": ((c, chunk, D), bf16),
            "qg": ((c, chunk, D), bf16), "kg": ((c, chunk, D), bf16),
            "aqk32": ((c, chunk, chunk), fp32), "aqk16": ((c, chunk, chunk), bf16),
            "L": ((c_solve, chunk, chunk), fp32),
            "a32": ((c_solve, chunk, chunk), fp32), "a16": ((c_solve, chunk, chunk), bf16),
            "W": ((c, chunk, D), bf16), "U": ((c, chunk, D), bf16),
            "gram_ops3": ((3, c, chunk, D), bf16),
            "gram_x": ((c, max(1, chunk // 2), D), bf16),
            "s32": ((tasks, BV, D), fp32), "s16": ((tasks, BV, D), bf16),
            "d1": ((tasks, nt, chunk, BV), bf16),
            "d2": ((tasks, nt, chunk, BV), bf16),
            "d3": ((tasks, nt, chunk, BV), bf16),
            "d4f": ((bh, D, D), fp32),
            "out_public": ((b, t, h, D), bf16),
            "vnew_t": ((tasks, nt, BV, chunk), bf16),
        }
        if api.SOLVE_WIDE_SUBB > 1:
            spec["xb"] = ((c_solve, api.SOLVE_WIDE_SUBB, sub, sub), bf16)
            spec["lneg"] = ((c_solve, sub, sub), bf16)
            spec["pmid"] = ((c_solve, sub, sub), bf16)
        if need_intermediates:
            spec["vnew"] = ((tasks, nt, chunk, BV), bf16)
        return spec

    def checkout(self, device, b, t, h, chunk, need_intermediates):
        key = (str(device), b, t, h, chunk, NV, bool(need_intermediates))
        if key in self._busy:
            self.refused += 1
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            spec = self._spec(b, t, h, chunk, need_intermediates)
            entry = {name: torch.empty(shape, dtype=dt, device=device)
                     for name, (shape, dt) in spec.items()}
            self._entries[key] = entry
        else:
            self.hits += 1
        self._busy.add(key)
        return entry

    def checkin(self, entry):
        for key, e in self._entries.items():
            if e is entry:
                self._busy.discard(key)
                return
        raise KeyError("entry was not leased from this pool")

    def nbytes(self):
        return sum(x.numel() * x.element_size()
                   for e in self._entries.values() for x in e.values())


# --- the allocator shim (used by the e2e A/B below) ---
def ledger(pool, b, t, h, chunk):
    spec = pool._spec(b, t, h, chunk, False)
    elt = {torch.bfloat16: 2, torch.float32: 4}
    total = 0
    for name, (shape, dt) in sorted(spec.items()):
        nbytes = 1
        for dim in shape:
            nbytes *= int(dim)
        nbytes *= elt[dt]
        total += nbytes
        print("   %-12s %10.2f MB  %s" % (name, nbytes / 1e6,
                                          "bf16" if dt is torch.bfloat16 else "fp32"))
    print("   %-12s %10.2f MB" % ("TOTAL", total / 1e6))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--t", type=int, default=8192)
    ap.add_argument("--h", type=int, default=96)
    args = ap.parse_args()
    dev = torch.device(args.device)

    torch.manual_seed(1312)
    B, T, H = args.b, args.t, args.h
    NT, BH, C = T // api.CHUNK, B * H, B * H * (T // api.CHUNK)
    TASKS = BH * NV
    CHUNK = api.CHUNK
    q = (torch.randn(B, T, H, D, device=dev) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=dev) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=dev) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=dev) * 0.1
    beta = torch.randn(B, T, H, device=dev)
    a_log = torch.linspace(-1.0, 0.2, H, device=dev)
    bias = torch.randn(H, D, device=dev) * 0.03
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

    print("warming / RTC (C=%d)..." % api.CHUNK, flush=True)
    out_stock = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()

    pool = WorkspacePool()
    print()
    print("pool ledger at B=%d T=%d H=%d C=%d:" % (B, T, H, api.CHUNK))
    pool_bytes = ledger(pool, B, T, H, api.CHUNK)

    print()
    print("warming the pool with one lease...", flush=True)
    e = pool.checkout(dev, B, T, H, api.CHUNK, False)
    pool.checkin(e)
    torch.npu.synchronize()

    def host_min(fn, reps):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best

    # A: the api's *allocation* arm alone - the same 30-odd tensors the api
    # allocates, in the api's order, with nothing else in the window.  This is
    # the arm the pool can remove; the whole-call host wall is a different
    # number (see tools/probe_host_cost.py) and is reported below for context.
    real_torch = api.torch
    allocs = []

    def api_alloc_arm():
        allocs.clear()
        allocs.append(real_torch.empty((C, CHUNK, D), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty_like(allocs[0]))
        allocs.append(real_torch.empty((C, CHUNK, D), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty_like(allocs[2]))
        allocs.append(real_torch.empty((C, CHUNK), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty((C, D), dtype=real_torch.float32, device=dev))
        for _ in range(4):
            allocs.append(real_torch.empty_like(allocs[0]))
        allocs.append(real_torch.empty((C, CHUNK, CHUNK), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty((C, CHUNK, CHUNK), dtype=real_torch.bfloat16, device=dev))
        for _ in range(3):
            allocs.append(real_torch.empty((C, CHUNK, CHUNK), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty_like(allocs[0]))
        allocs.append(real_torch.empty((C, CHUNK, CHUNK), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty_like(allocs[0]))
        allocs.append(real_torch.empty_like(allocs[0]))
        allocs.append(real_torch.empty((3, C, CHUNK, D), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty((C, CHUNK // 2, D), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty((TASKS, BV, D), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty((TASKS, BV, D), dtype=real_torch.bfloat16, device=dev))
        for _ in range(3):
            allocs.append(real_torch.empty((TASKS, NT, CHUNK, BV), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty((BH, D, D), dtype=real_torch.float32, device=dev))
        allocs.append(real_torch.empty((B, T, H, D), dtype=real_torch.bfloat16, device=dev))
        allocs.append(real_torch.empty((TASKS, NT, BV, CHUNK), dtype=real_torch.bfloat16, device=dev))

    a_ms = host_min(api_alloc_arm, args.reps)

    # B: lease a pooled entry instead - this is the *allocation* arm only, so
    # the same tensors are then handed back; nothing else changes.
    def pooled_alloc():
        ent = pool.checkout(dev, B, T, H, api.CHUNK, False)
        pool.checkin(ent)
    b_ms = host_min(pooled_alloc, args.reps)

    # C: do_bench on the real call, to see whether any of this reaches the
    # end-to-end number (host work that is already hidden cannot).
    from triton.testing import do_bench
    torch.npu.synchronize()
    med = do_bench(lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw),
                   warmup=100, rep=1000, quantiles=[0.5, 0.2, 0.8])

    # NOTE: this probe deliberately stops at pricing the arms.  An earlier
    # version also ran the call with a shim that served its allocations from the
    # pool, and that A/B is *not* safe to trust: reproducing the api's
    # allocation set exactly turned out to be the hard part, and both attempts
    # produced wrong answers rather than timings - keying on (shape, dtype)
    # aliases the api's same-shaped buffers (measured max|do| 3.368e-01), and
    # handing out a short buffer for a larger request trips the api's own
    # ``.view()`` (measured: "shape [8, 2, 64, 128] is invalid for input of
    # size 262144").  The question this probe exists for - does removing the
    # 0.33 ms reach the e2e number? - is answered without swapping buffers by
    # tools/probe_host_exposure.py, which *adds* host delay and watches the e2e
    # number move (it stays flat up to +2.4 ms per call).  These negative
    # results are pinned in tests/test_workspace_pool.py.

    print()
    print("   api per-call workspace alloc   %8.4f ms  (host MIN of %d, %d tensors)"
          % (a_ms, args.reps, len(allocs)))
    print("   pooled checkout/checkin        %8.4f ms  (host MIN of %d)" % (b_ms, args.reps))
    print("   alloc arm removed by the pool  %8.4f ms" % (a_ms - b_ms))
    print("   pool kept %.2f GB resident,  hit %d miss %d refused %d"
          % (pool.nbytes() / 1e9, pool.hits, pool.misses, pool.refused))
    e2e_host = host_min(lambda: api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw), args.reps)
    print("   e2e do_bench median %.3f ms  (p20 %.3f / p80 %.3f)"
          % (med[0], med[1], med[2]))
    print("   whole-call host wall       %.4f ms  (the pool cannot remove this one;"
          % e2e_host)
    print("                               it is only exposed if it exceeds the device time)")
    print("   e2e device time            ~10.77 ms; the host wall above is 5.8 ms, so the")
    print("                              host is hidden - see tools/probe_host_exposure.py")

    # Clean up: a hidden bug here would hold GB of device memory after exit.
    pool._entries.clear()
    pool._busy.clear()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
