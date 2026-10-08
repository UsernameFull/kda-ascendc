"""Do the solve's 24 slices pay a wave tail?  Slice-policy sweep, device events.

The two-level solve (docs 11.38-11.40) cuts c_solve into ``SOLVE_OVERLAP``
slices; the wide half (AIV) of slice i goes on stream sa, then that slice's
assemble + Cube (AIC) on sb behind an event.  At [1,8192,96,128], C=64 the
default 24 slices put (default overlap) grids of 128 wide blocks / 128
assemble / 256 Cube blocks per slice against this part's cores.  A wave model
that is not divisible by the core count makes every slice pay a partial last
wave: 128 blocks = 3.2 waves of 40 AIV cores (or 6.4 of 20 if the AIV-only
kernel maps to AI cores), 128 assemble = 6.4 waves of 20 AIC, 256 Cube = 12.8
waves of 20.  Docs 11.57's note left this as the one un-swept same-shape
structure ("slice boundaries cannot be divided by 20").

This probe re-slices the *same launches* (same pointers, same kernels) with
policies whose slice sizes are whole waves on both 20 and 40 cores - multiples
of 40 groups (a group = the lcm(4,4,2) = 4 chunks the two-level wiring uses) -
by monkeypatching ``api._launch_solve_two_level`` from the probe only; the
production function is untouched and the default arm calls it as shipped.

Read per arm (MIN of the rounds):
  * per-kernel device spans (wide / assemble / cube) summed over slices;
  * the solve stage span (first solve launch's event to the last one's),
    which is the number the e2e sees;
  * the e2e wall in a second, event-free pass (paired, order-rotated).

Slicing is pure scheduling - the arithmetic per chunk is identical - so every
arm's outputs must be bit-identical to the control; the check is printed.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_slices.py
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(5400, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
# (name, policy): "default" = the shipped function; "even" = same even split
# as production with a different count; "wave" = slices of that many chunks
# (must be a multiple of the wiring unit, 4) plus one remainder slice.
ARMS = [
    ("default(24)", ("default", 24)),
    ("wave 480", ("wave", 480)),
    ("wave 640", ("wave", 640)),
    ("wave 800", ("wave", 800)),
    ("wave 640f", ("wavef", 640)),
    ("wave 1280", ("wavef", 1280)),
]


def _parse_arm(spec):
    if spec == "default":
        return ("default(24)", ("default", 24))
    kind, val = spec.split(":")
    return (spec.replace(":", " "), (kind, int(val)))


_env_arms = os.environ.get("KDA_SLICE_ARMS", "")
if _env_arms:
    ARMS = [_parse_arm(x) for x in _env_arms.split(",")]
SPAN_ROUNDS = int(os.environ.get("KDA_SLICE_SPAN_ROUNDS", "3"))
WALL_ROUNDS = int(os.environ.get("KDA_SLICE_WALL_ROUNDS", "10"))
SOLVE_KERNELS = {"kda_solve_wu_wide", "kda_solve_assemble", "kda_solve_wu_cube_kernel"}


def make_two_level(kind, val):
    """A probe-local copy of api._launch_solve_two_level with custom pairs."""
    nch, asm_nchunk, wu_nchunk = api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK
    unit = math.lcm(nch, asm_nchunk, wu_nchunk)
    assert api.SOLVE_WIDE_SUBB == 2, "the probe only re-slices the SB=2 path"

    def impl(c_solve, c, nch_, asm_, wu_, overlap, L, eye, a32, a16, xb, lneg,
             pmid, rk, rv, W, U, stream, debug_stores, subb=2):
        assert (nch_, asm_, wu_) == (nch, asm_nchunk, wu_nchunk)
        ngrp = c_solve // unit
        if kind == "even":
            k = min(val, max(1, ngrp))
            pairs = [(i * ngrp // k, (i + 1) * ngrp // k) for i in range(k)]
        else:
            sgrp = val // unit
            assert sgrp > 0 and val % unit == 0
            nfull = ngrp // sgrp
            pairs = [(i * sgrp, (i + 1) * sgrp) for i in range(nfull)]
            if nfull * sgrp < ngrp:
                if kind == "wavef":
                    pairs.insert(0, (nfull * sgrp, ngrp))
                else:
                    pairs.append((nfull * sgrp, ngrp))
        cur = torch_npu.npu.current_stream()
        sa, sb = api._solve_streams(a16.device)
        sa.wait_stream(cur)
        for glo, ghi in pairs:
            lo, n = glo * unit, (ghi - glo) * unit
            wargs = api._pack_ptrs([L[lo:], eye, a32[lo:], a16[lo:], xb[lo:],
                                    lneg[lo:]]) + [
                api._i(n), api._i(api.a16_mode()), api._i(1 if debug_stores else 0)]
            aargs = api._pack_ptrs([a16[lo:], xb[lo:], lneg[lo:],
                                    None if pmid is None else pmid[lo:]]) + [
                api._i(n), api._i(api.asm_load_mode())]
            cargs = api._pack_ptrs([a16[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + [
                api._i(n), api._i(api.cube_a16_resident())]
            ncube = min(n, c - lo)
            api._launch("kda_solve_wu_wide", n // nch, wargs, sa.npu_stream)
            ev = torch_npu.npu.Event()
            ev.record(sa)
            sb.wait_event(ev)
            api._launch("kda_solve_assemble", (n + asm_nchunk - 1) // asm_nchunk,
                        aargs, sb.npu_stream)
            if ncube > 0:
                api._launch("kda_solve_wu_cube_kernel",
                            (ncube + wu_nchunk - 1) // wu_nchunk, cargs,
                            sb.npu_stream)
        cur.wait_stream(sb)

    return impl


def install(policy):
    """Returns the restore fn; the default arm leaves production installed."""
    kind, val = policy
    if kind == "default":
        return lambda: None
    orig = api._launch_solve_two_level
    api._launch_solve_two_level = make_two_level(kind, val)
    return lambda: setattr(api, "_launch_solve_two_level", orig)


def inputs():
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)
    return q, k, v, g, beta, kw


def call_with_spans(q, k, v, g, beta, kw):
    """One pipeline call; every solve launch bracketed by device events."""
    cur = torch_npu.npu.current_stream()
    sa, sb = api._solve_streams(DEV)
    streams = {cur.npu_stream: cur, sa.npu_stream: sa, sb.npu_stream: sb}
    log = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        obj = streams.get(stream)
        if obj is None or kernel not in SOLVE_KERNELS:
            return orig(kernel, blocks, args, stream)
        ev0 = torch_npu.npu.Event(enable_timing=True)
        ev1 = torch_npu.npu.Event(enable_timing=True)
        ev0.record(obj)
        ret = orig(kernel, blocks, args, stream)
        ev1.record(obj)
        log.append((kernel, int(blocks), ev0, ev1))
        return ret

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    spans = {}
    for kernel, blocks, ev0, ev1 in log:
        spans[kernel] = spans.get(kernel, 0.0) + ev0.elapsed_time(ev1)
    # Events are not orderable; use the first one as the clock reference.
    ref = log[0][2]
    first = min((ev0 for _, _, ev0, _ in log), key=ref.elapsed_time)
    last = max((ev1 for _, _, _, ev1 in log), key=ref.elapsed_time)
    stage = first.elapsed_time(last)
    return out, st, spans, stage, log


def call_timed(q, k, v, g, beta, kw):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return ((time.perf_counter() - t0) * 1e3, (t1 - t0) * 1e3, out, st)


def main() -> None:
    q, k, v, g, beta, kw = inputs()
    call_timed(q, k, v, g, beta, kw)          # compile + warm
    base_out, base_st = call_timed(q, k, v, g, beta, kw)[2:]

    c = B * H * (T // api.CHUNK)
    unit = math.lcm(api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK)
    print("solve slice-policy sweep: c=%d, unit=%d, NCH=%d ASM=%d WU=%d, "
          "cores: 20 AIC / 40 AIV" %
          (c, unit, api.SOLVE_WIDE_NCH, api.ASM_NCHUNK, api.WU_NCHUNK),
          flush=True)

    # ---- pass 1: device spans, controls the geometry print ----------------
    span = {}
    span_out = {}
    for name, policy in ARMS:
        restore = install(policy)
        try:
            best = None
            for _ in range(SPAN_ROUNDS):
                out, st, spans, stage, log = call_with_spans(q, k, v, g, beta, kw)
                if best is None or stage < best[1]:
                    best = (dict(spans), stage, log)
                    span_out[name] = (out, st)
            span[name] = best
        finally:
            restore()
    print()
    for name, policy in ARMS:
        spans, stage, log = span[name]
        grids = [b for _, b, _, _ in log]
        wide_n = sum(b for kk, b, _, _ in log if kk == "kda_solve_wu_wide")
        print("  %-12s stage %6.3f | wide %6.3f (%d blocks in %d slices) | "
              "asm %6.3f | cube %6.3f | launches %d"
              % (name, stage, spans.get("kda_solve_wu_wide", -1),
                 wide_n, sum(1 for kk, _, _, _ in log if kk == "kda_solve_wu_wide"),
                 spans.get("kda_solve_assemble", -1),
                 spans.get("kda_solve_wu_cube_kernel", -1), len(log)), flush=True)

    # ---- pass 2: the wall, no events --------------------------------------
    print()
    print("e2e wall (no events, MIN of %d, order rotated):" % WALL_ROUNDS, flush=True)
    wall, host = {}, {}
    for r in range(WALL_ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for name, policy in order:
            restore = install(policy)
            try:
                w, h = call_timed(q, k, v, g, beta, kw)[:2]
                wall[name] = min(wall.get(name, 1e9), w)
                host[name] = min(host.get(name, 1e9), h)
            finally:
                restore()
    wbase = wall[ARMS[0][0]]
    print("  %-12s %8s %8s %8s" % ("arm", "wall", "host", "delta"))
    for name, _ in ARMS:
        print("  %-12s %8.3f %8.3f %+8.3f" % (name, wall[name], host[name],
                                              wall[name] - wbase), flush=True)

    # ---- identity ----------------------------------------------------------
    print()
    for name, policy in ARMS:
        out, st = span_out[name]
        print("  identity %-12s out=%d st=%d"
              % (name, int((out != base_out).sum().item()),
                 int((st != base_st).sum().item())), flush=True)


if __name__ == "__main__":
    main()
