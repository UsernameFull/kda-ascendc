"""Round 2: does pre_gram's block wall follow the *AIC's* reply path per step?

Round 1 (docs 11.55) priced the AIV's vector stream at 31.7 ns per added
instruction and showed the block wall is that stream.  This round asks the
question the FL_DONE protocol decision needs: the AIC answers every step with
``CrossCoreSetFlag<2, PIPE_FIX>(FL_DONE)`` and the AIV waits that reply before
it can mask the previous chunk's Gram (the ~0.17 ms inside the -post_gram arm,
docs 11.55), on level-triggered flags with a strict one-step backlog.  A
deeper handshake (ping-pong ids, depth 2) would only pay if the AIC's *reply
latency* is what the AIV stalls on - so price that latency: inject work right
where the reply is produced and read the slope of launch time against it.

  aicWork    extra Mmad instructions per step, re-run onto the same fp32
             accumulator (cmatrixInitVal = true, so bit-exact), straight in
             front of M_FIX -> Fixpipe -> FL_DONE.
  aicScalar  dependent scalar ops immediately before the FL_DONE set.

The reference cost of one 64x64x128 Mmad on this device is ~56 ns (msprof
PipeUtilization of the shipped kernel: cube busy 28.9 us per 819.4 us block
over 8 Mmads x 64 steps, docs 11.55); a response of 56 ns per added Mmad
means the reply path is 100% wall, a response of ~0 means the AIC absorbs it.

The control arm replays the *production* arithmetic (blocked scan since docs
11.56; ablate bit 8 = the old 63-step serial cumsum as the timing reference)
on the captured production launch - grid 80 x u 77, the 20-AIC shape of docs
11.57 - so the deltas are directly comparable with production.  Carried over
from round 1: the AIV anchor (+32 Muls/pass), the delete arms (bit 1 gate
cumsum, 2 sigmoid, 3 both, 4 post_gram) and bit 16 (the AIC drops every Mmad;
wrong by construction, timing only).

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pre_gram_addwork.py
"""
from __future__ import annotations

import faulthandler
import math
import os
import struct
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
# (addWork, ablate, aicWork, aicScalar); labels are printed with the table.
ARMS = [
    (0, 0, 0, 0),      # control: production form (blocked scan)
    (32, 0, 0, 0),     # AIV anchor: +32 Muls/pass
    (0, 8, 0, 0),      # old 63-step serial cumsum (pre-11.56 form)
    (0, 1, 0, 0),      # -gate cumsum
    (0, 2, 0, 0),      # -sigmoid
    (0, 3, 0, 0),      # -cumsum,-sigmoid
    (0, 4, 0, 0),      # -post_gram (incl. the FL_DONE wait)
    (0, 0, 8, 0),      # AIC +8 Mmad/step
    (0, 0, 32, 0),     # AIC +32 Mmad/step
    (0, 0, 96, 0),     # AIC +96 Mmad/step
    (0, 0, 0, 512),    # AIC +512 dependent scalar ops/step
    (0, 16, 0, 0),     # AIC -every Mmad (wrong by construction)
    (0, 32, 0, 0),     # -only the FL_DONE wait (post_gram work kept)
]
LABEL = {(0, 0, 0, 0): "control(scan)", (32, 0, 0, 0): "AIV +32/pass",
         (0, 8, 0, 0): "serial cumsum", (0, 1, 0, 0): "-gate cumsum",
         (0, 2, 0, 0): "-sigmoid", (0, 3, 0, 0): "-cumsum,sigmoid",
         (0, 4, 0, 0): "-post_gram", (0, 0, 8, 0): "AIC +8 Mmad/step",
         (0, 0, 32, 0): "AIC +32 Mmad/step", (0, 0, 96, 0): "AIC +96 Mmad/step",
         (0, 0, 0, 512): "AIC +512 scal/step", (0, 16, 0, 0): "AIC -all Mmad",
         (0, 32, 0, 0): "-FL_DONE wait"}
# Arms whose pipeline output must be bit-exact against the control, plus the
# serial reference (tolerance, not identity).
IDENT = [(0, 0, 0, 0), (32, 0, 0, 0), (0, 0, 32, 0), (0, 0, 0, 512), (0, 8, 0, 0)]
ROUNDS = int(os.environ.get("KDA_ADDWORK_ROUNDS", "5"))
PROBE = "kda_pg_addwork_probe"
MMAD_NS_REF = 56.0   # msprof cube-busy price of one 64x64x128 Mmad (docs 11.55)


def capture(q, k, v, g, beta, kw):
    seq = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == "kda_pre_gram_mix":
            seq.append((int(blocks), list(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return seq[-1], out, st


def run_pipeline(q, k, v, g, beta, kw, arm):
    """Whole pipeline with the pre_gram launch replaced by the probe arm."""
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == "kda_pre_gram_mix":
            api.launch_argsarray_engine(
                PROBE, int(blocks), stream,
                list(args) + [api._i(arm[0]), api._i(arm[1]),
                              api._i(arm[2]), api._i(arm[3])], 0)
            return
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return out, st


def n_diff(a, b):
    return int((a != b).sum().item())


def tf32(a, b):
    return (float((a - b).abs().max().item()), float(a.abs().max().item()))


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

    (grid, args), out_p, st_p = capture(q, k, v, g, beta, kw)
    u = struct.unpack("<i", args[-5])[0]
    raw = struct.unpack("<i", args[-2])[0]
    dbg = struct.unpack("<i", args[-1])[0]
    cores = api._aic_core_count(DEV)
    waves = int(math.ceil(grid / cores))
    npass = max(api.CHUNK // 16, 1)
    print("captured kda_pre_gram_mix: grid %d, %d args, unroll %d, rawMode %d, "
          "debugStores %d; %d AIC cores -> %d wave(s), NP %d"
          % (grid, len(args), u, raw, dbg, cores, waves, npass), flush=True)

    stream = torch_npu.npu.current_stream().npu_stream
    times = {a: [] for a in ARMS}
    for r in range(ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for arm in order:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            api.launch_argsarray_engine(
                PROBE, grid, stream,
                args + [api._i(arm[0]), api._i(arm[1]),
                        api._i(arm[2]), api._i(arm[3])], 0)
            torch.npu.synchronize()
            times[arm].append((time.perf_counter() - t0) * 1e3)

    best = {a: min(vs) for a, vs in times.items()}
    base = best[ARMS[0]]
    ok = 0.5 < base < 20.0
    print("replay launch+sync wall (MIN of %d, ms), control = %.3f%s"
          % (ROUNDS, base, "" if ok else "  *** WARNING: out of range, the "
             "launch may have been dropped (docs 11.57) ***"), flush=True)
    for arm in ARMS:
        print("  %-18s %8.3f   delta %+8.3f   spread %.2f"
              % (LABEL[arm], best[arm], best[arm] - base,
                 max(times[arm]) - best[arm]), flush=True)

    d = {a: best[a] - base for a in ARMS}
    n_aiv = 32 * npass * u * waves      # one subcore's added-instruction stream
    steps = u * waves                   # one AIC's steps in the same units
    aiv_ns = d[(32, 0, 0, 0)] * 1e6 / n_aiv
    print()
    print("AIV anchor: +32 Muls/pass = %+0.3f ms over %d instr -> %.1f ns/instr "
          "(%.0f cyc, round 1 read 31.7 ns)"
          % (d[(32, 0, 0, 0)], n_aiv, aiv_ns, aiv_ns * 1.8), flush=True)
    print("AIC dose -> wall response over %d steps in one AIC's stream:" % steps,
          flush=True)
    for ai in (8, 32, 96):
        a = (0, 0, ai, 0)
        per_step = d[a] * 1e6 / steps
        print("  +%3d Mmad/step  = %+0.3f ms -> %+7.1f ns/step -> %+6.2f ns/Mmad "
              "(%.0f%% of the %0.0f ns reference)"
              % (ai, d[a], per_step, d[a] * 1e6 / (steps * ai),
                  d[a] * 1e6 / (steps * ai) / MMAD_NS_REF * 100.0, MMAD_NS_REF),
              flush=True)
    a = (0, 0, 0, 512)
    print("  +512 scal/step  = %+0.3f ms -> %+7.1f ns/step -> %+6.2f ns/op"
          % (d[a], d[a] * 1e6 / steps, d[a] * 1e6 / (steps * 512)), flush=True)
    print("  -all Mmad       = %+0.3f ms (delete direction, wrong by construction)"
          % d[(0, 16, 0, 0)], flush=True)

    print()
    print("delete arms re-run on the production arithmetic (control = scan):",
          flush=True)
    d1, d2, d3 = d[(0, 1, 0, 0)], d[(0, 2, 0, 0)], d[(0, 3, 0, 0)]
    print("  -gate cumsum %+0.3f | -sigmoid %+0.3f | both %+0.3f (sum of singles "
          "%+0.3f) | -post_gram %+0.3f | serial cumsum %+0.3f"
          % (d1, d2, d3, d1 + d2, d[(0, 4, 0, 0)], d[(0, 8, 0, 0)]), flush=True)
    dw = d[(0, 32, 0, 0)]
    print("  -FL_DONE wait only %+0.3f (keeps all of post_gram's work; this is "
          "the ceiling a deeper handshake could recover, and -post_gram minus "
          "this is the mask/scale/round work itself)" % dw, flush=True)

    resp32 = d[(0, 0, 32, 0)] * 1e6 / steps
    resp96 = d[(0, 0, 96, 0)] * 1e6 / steps
    resp8 = d[(0, 0, 8, 0)] * 1e6 / steps
    print()
    if resp8 < 30.0 and resp32 < 60.0:
        print("  => REPLY PATH HAS SLACK at this dose: +8 and +32 Mmads/step "
              "(%+.0f / %+.0f ns/step) are absorbed; the AIC can stand "
              ">= 32 Mmads (~1.8 us at 1:1) of extra reply latency per step "
              "before the AIV sees any of it." % (resp8, resp32), flush=True)
    elif resp32 >= 60.0:
        print("  => REPLY PATH IS WALL: +32 Mmads/step costs %+.1f ns/step, "
              "i.e. roughly 1:1 with the M work it adds; any protocol change "
              "that lengthens the AIC reply (or stops the AIV from hiding it) "
              "pays on the wall at this rate." % resp32, flush=True)
    if resp96 > 0.0 and resp32 > 0.0:
        slope = (resp96 - resp32) / (96.0 - 32.0)
        print("  => pairwise slope (32 -> 96 Mmads): %.2f ns/Mmad = %.0f%% of "
              "the %.0f ns reference" % (slope, slope / MMAD_NS_REF * 100.0,
                                         MMAD_NS_REF), flush=True)

    print()
    outs = {}
    for arm in IDENT:
        outs[arm] = run_pipeline(q, k, v, g, beta, kw, arm)
    out0, st0 = outs[(0, 0, 0, 0)]

    def cmp_ref(arm):
        o, st = outs[arm]
        return n_diff(o, out_p), n_diff(st, st_p)

    print("identity vs captured production pipeline (must be 0/0 - same scan):")
    print("  control(scan) out=%d st=%d | AIV+32 out=%d st=%d | AIC+32M out=%d st=%d"
          " | AIC+512s out=%d st=%d"
          % (cmp_ref((0, 0, 0, 0)) + cmp_ref((32, 0, 0, 0))
             + cmp_ref((0, 0, 32, 0)) + cmp_ref((0, 0, 0, 512))), flush=True)
    out8, st8 = outs[(0, 8, 0, 0)]
    mo, so = tf32(out8, out0)
    print("  serial cumsum vs scan (tolerance arm): out differ=%d/%d max|d|=%.3e "
          "(rel %.1e) | state differ=%d max|d|=%.3e"
          % (n_diff(out8, out0), out0.numel(), mo, mo / max(so, 1e-30),
             n_diff(st8, st0), float((st8 - st0).abs().max().item())), flush=True)


if __name__ == "__main__":
    main()
