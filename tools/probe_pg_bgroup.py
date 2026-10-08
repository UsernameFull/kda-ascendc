"""Round 3 on pre_gram's AIV wall: what is inside post_gram's own 0.331 ms?

Round 1 (docs 11.55) priced added vector instructions at 31.7 ns each; round 2
(docs 11.58) showed the AIC reply path is not the wall - deleting the whole
post_gram costs 0.331 ms, several times what its ~40 instructions per chunk
should cost at the measured price.  So the band loop spends its time in
*waits*, not only in issue.

This round compiles textual twins of the production kernel and replays the
captured production launch (grid 80 x unroll 77, the 20-AIC shape of docs
11.57) against them in one process, alternating, MIN of KDA_BG_ROUNDS.  Every
arm is timed twice: host wall around launch+sync and a device event span that
brackets exactly the kernel.

  ctrl    identical recompile (compile-noise anchor)
  flat    the band loads/stores move from the 2-D "(16, M/8)" descriptor to a
          single 1-D burst of 16*M elements; the band region is contiguous in
          GM, so this is the same bytes in the same order
  hoist   the gate sigmoid's Duplicate(t2, 1.0) leaves the NP loop (t2 is a
          constant and nothing writes it between the reads => bit-same)
  flat+   both
  pf      one-band-ahead prefetch: band 0's loads are issued before the loop;
          each iteration dequeues the current band first and issues the NEXT
          band's loads before computing on the current one, so one MTE2
          latency is covered by one band body instead of being waited on
  nostore timing-only: the three band stores dropped (queue protocol kept)
  nosel   timing-only: both mask Selects become plain Muls(x, 1.0)

Identity arms run the whole pipeline with the twin swapped in and diff
out/state against the captured production run; the two timing-only arms are
wrong by construction and are labelled so.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_bgroup.py
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

faulthandler.dump_traceback_later(7200, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
ROUNDS = int(os.environ.get("KDA_BG_ROUNDS", "8"))
PROD = "kda_pre_gram_mix"
# Every arm is a patch against the PRE-11.60 kernel, so the base text is pinned
# to the commit the round was measured on (KDA_BG_BASE overrides; "WORKTREE"
# reads the tree's copy, which is the post-11.60 form).
BASE = os.environ.get("KDA_BG_BASE", "7d628eb")
if BASE == "WORKTREE":
    SRC = (ROOT / "kernels/v1/k1_pre_gram_mix.cpp").read_text(encoding="utf-8-sig")
else:
    import subprocess
    SRC = subprocess.check_output(
        ["git", "show", BASE + ":kernels/v1/k1_pre_gram_mix.cpp"], cwd=ROOT
    ).decode("utf-8-sig")


def sub1(src: str, old: str, new: str) -> str:
    n = src.count(old)
    assert n == 1, "anchor not unique (%d): %r" % (n, old[:70])
    return src.replace(old, new)


# ---- patches ---------------------------------------------------------------
F_LOAD = ("""    DataCopy(ga32i, Aqk32[o], DataCopyParams(16, M / 8, 0, 0));
    DataCopy(gl32i, L[o], DataCopyParams(16, M / 8, 0, 0));""",
          """    DataCopy(ga32i, Aqk32[o], NB);
    DataCopy(gl32i, L[o], NB);""")
F_MASK = ("""        DataCopy(gmaskS, MaskS[mo], DataCopyParams(16, M / 8, 0, 0));
        DataCopy(gmaskL, MaskL[mo], DataCopyParams(16, M / 8, 0, 0));""",
          """        DataCopy(gmaskS, MaskS[mo], NB);
        DataCopy(gmaskL, MaskL[mo], NB);""")
F_STORE = ("""    if (keepAqk32) DataCopy(Aqk32[o], ga32s, DataCopyParams(16, M / 8, 0, 0));
    DataCopy(L[o], gl32s, DataCopyParams(16, M / 8, 0, 0));
    DataCopy(Aqk16[o], g16s, DataCopyParams(16, M / 16, 0, 0));""",
           """    if (keepAqk32) DataCopy(Aqk32[o], ga32s, NB);
    DataCopy(L[o], gl32s, NB);
    DataCopy(Aqk16[o], g16s, NB);""")

H_SIG = ("""    for (int32_t hp = 0; hp < NP; ++hp) {
    LocalTensor<float> gfs = gf[hp * NG];
    Muls(gfs, gfs, aexp, NG);
    Exp(gfs, gfs, NG);
    Adds(gfs, gfs, 1.0f, NG);
    Duplicate(t2, 1.0f, NG);
    PipeBarrier<PIPE_V>();""",
         """    Duplicate(t2, 1.0f, NG);
    for (int32_t hp = 0; hp < NP; ++hp) {
    LocalTensor<float> gfs = gf[hp * NG];
    Muls(gfs, gfs, aexp, NG);
    Exp(gfs, gfs, NG);
    Adds(gfs, gfs, 1.0f, NG);
    PipeBarrier<PIPE_V>();""")

def _pf(flat: bool):
    pre_a = "DataCopy(gpre, Aqk32[m0], DataCopyParams(16, M / 8, 0, 0));\n        DataCopy(gpre[16 * M], L[m0], DataCopyParams(16, M / 8, 0, 0));"
    pre_f = "DataCopy(gpre, Aqk32[m0], 16 * M);\n        DataCopy(gpre[16 * M], L[m0], 16 * M);"
    nx_a = "DataCopy(gnx, Aqk32[on], DataCopyParams(16, M / 8, 0, 0));\n        DataCopy(gnx[NB], L[on], DataCopyParams(16, M / 8, 0, 0));"
    nx_f = "DataCopy(gnx, Aqk32[on], NB);\n        DataCopy(gnx[NB], L[on], NB);"
    old, new = P_PF_T
    return (old, new.replace("__PRE__", pre_f if flat else pre_a)
            .replace("__NX__", nx_f if flat else nx_a))


P_PF_T = ("""    for (int32_t mm = 0; mm < KF; ++mm) {
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;
    const uint64_t mo = static_cast<uint64_t>(mm) * 16 * M;
    constexpr int32_t NB = 16 * M;   // elements in one band
    LocalTensor<float> gin = qin.AllocTensor<float>();
    LocalTensor<float> ga32i = gin, gl32i = gin[NB];
    DataCopy(ga32i, Aqk32[o], DataCopyParams(16, M / 8, 0, 0));
    DataCopy(gl32i, L[o], DataCopyParams(16, M / 8, 0, 0));
    qin.EnQue(gin);
    LocalTensor<float> gmk, gmaskS, gmaskL;
    if (buildMasks) {
        gmk = qmk.AllocTensor<float>();
        gmaskS = gmk; gmaskL = gmk[NB];
        DataCopy(gmaskS, MaskS[mo], DataCopyParams(16, M / 8, 0, 0));
        DataCopy(gmaskL, MaskL[mo], DataCopyParams(16, M / 8, 0, 0));
        qmk.EnQue(gmk);
    }
    LocalTensor<float> gA = qin.DeQue<float>();""",
        """    // One-band-ahead: band 0 is issued once before the loop and every
    // iteration dequeues the current band first, then issues the NEXT band's
    // loads before computing on the current one, so one MTE2 latency is
    // covered by one band body (the queue is FIFO, depth 2 is enough).
    {
        LocalTensor<float> gpre = qin.AllocTensor<float>();
        __PRE__;
        qin.EnQue(gpre);
    }
    for (int32_t mm = 0; mm < KF; ++mm) {
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;
    const uint64_t mo = static_cast<uint64_t>(mm) * 16 * M;
    constexpr int32_t NB = 16 * M;   // elements in one band
    LocalTensor<float> gA = qin.DeQue<float>();
    if (mm + 1 < KF) {
        LocalTensor<float> gnx = qin.AllocTensor<float>();
        const uint64_t on = m0 + static_cast<uint64_t>(mm + 1) * 16 * M;
        __NX__;
        qin.EnQue(gnx);
    }
    LocalTensor<float> gmk, gmaskS, gmaskL;
    if (buildMasks) {
        gmk = qmk.AllocTensor<float>();
        gmaskS = gmk; gmaskL = gmk[NB];
        DataCopy(gmaskS, MaskS[mo], DataCopyParams(16, M / 8, 0, 0));
        DataCopy(gmaskL, MaskL[mo], DataCopyParams(16, M / 8, 0, 0));
        qmk.EnQue(gmk);
    }""")

P_PF = _pf(False)
P_PF_FLAT = _pf(True)

A_NOSTORE = ((F_STORE[0],
              """    (void)ga32s; (void)gl32s; (void)g16s;"""))
A_NOSEL = (("""    Select(ga32o, gmaskBits, ga32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    Muls(ga32o, ga32o, scale, NB);""",
            """    Muls(ga32o, ga32, 1.0f, NB);
    Muls(ga32o, ga32o, scale, NB);"""),
           ("""    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);""",
            """    Muls(gl32o, gl32, 1.0f, NB);"""))

NB_BAR1 = ("""    Muls(ga32o, ga32o, scale, NB);
    PipeBarrier<PIPE_V>();""",
          """    Muls(ga32o, ga32o, scale, NB);""")
NB_BAR2 = ("""    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    PipeBarrier<PIPE_V>();""",
          """    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);""")

D_LOOP = ("""    for (int32_t mm = 0; mm < KF; ++mm) {
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;""",
          """    for (int32_t mmx = 0; mmx < 2 * KF; ++mmx) {
    const int32_t mm = mmx % KF;
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;""")

V_NONE = ("""    Select(ga32o, gmaskBits, ga32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    Muls(ga32o, ga32o, scale, NB);
    PipeBarrier<PIPE_V>();
    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> g16 = qo16.AllocTensor<bfloat16_t>();
    Cast(g16, ga32o, RoundMode::CAST_RINT, NB);""",
         """    (void)gmaskBits; (void)gmaskBitsL; (void)scale;
    LocalTensor<bfloat16_t> g16 = qo16.AllocTensor<bfloat16_t>();""")

ARMS = ["ctrl", "flat", "hoist", "flat+", "pf", "pf+", "nostore", "nosel",
        "nobar", "dbl", "noV"]
WRONG = {"nostore", "nosel", "dbl", "noV"}
PATCHES = {
    "ctrl": [],
    "flat": [F_LOAD, F_MASK, F_STORE],
    "hoist": [H_SIG],
    "flat+": [F_LOAD, F_MASK, F_STORE, H_SIG],
    "pf": [P_PF],
    "pf+": [P_PF_FLAT, F_MASK, F_STORE, H_SIG],
    "nostore": [A_NOSTORE],
    "nosel": list(A_NOSEL),
    "nobar": [NB_BAR1, NB_BAR2],
    "dbl": [D_LOOP],
    "noV": [V_NONE],
}


def twin_source(arm: str) -> str:
    src = SRC
    for old, new in PATCHES[arm]:
        src = sub1(src, old, new)
    return src.replace(PROD, "kda_pg_bg_" + arm.replace("+", "x"))


def capture(q, k, v, g, beta, kw):
    seq = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == PROD:
            seq.append((int(blocks), list(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return seq[-1], out, st


def run_pipeline(q, k, v, g, beta, kw, twin):
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        return orig(twin if kernel == PROD else kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    return out, st


def n_diff(a, b):
    return int((a != b).sum().item())


def main() -> None:
    only = os.environ.get("KDA_BG_ONLY")
    arms = [a for a in ARMS if only is None or a in only.split(",")]
    t_all = time.perf_counter()
    for arm in arms:
        t0 = time.perf_counter()
        api.rtc_compile(api._defines() + twin_source(arm),
                        "kda_pg_bg_" + arm.replace("+", "x"), "")
        print("compiled %-8s (%.0f s)" % (arm, time.perf_counter() - t0),
              flush=True)
    print("all compiles: %.0f s" % (time.perf_counter() - t_all), flush=True)

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
    print("captured %s: grid %d, unroll %d, %d args"
          % (PROD, grid, u, len(args)), flush=True)

    cur = torch_npu.npu.current_stream()
    stream = cur.npu_stream
    walls = {a: [] for a in arms}
    spans = {a: [] for a in arms}
    for r in range(ROUNDS):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for arm in order:
            torch.npu.synchronize()
            ev0 = torch_npu.npu.Event(enable_timing=True)
            ev1 = torch_npu.npu.Event(enable_timing=True)
            ev0.record(cur)
            t0 = time.perf_counter()
            api.launch_argsarray_engine(
                "kda_pg_bg_" + arm.replace("+", "x"), grid, stream, args, 0)
            ev1.record(cur)
            torch.npu.synchronize()
            walls[arm].append((time.perf_counter() - t0) * 1e3)
            spans[arm].append(ev0.elapsed_time(ev1))

    bw = {a: min(vs) for a, vs in walls.items()}
    bs = {a: min(vs) for a, vs in spans.items()}
    cw, cs = bw["ctrl"], bs["ctrl"]
    ok = 0.5 < cs < 20.0
    print("replay device span (MIN of %d, ms), ctrl = %.3f%s"
          % (ROUNDS, cs, "" if ok else "  *** WARNING: out of range, the "
             "launch may have been dropped (docs 11.57) ***"), flush=True)
    print("  %-8s %9s %9s %9s %9s %8s" %
          ("arm", "span", "d-span", "wall", "d-wall", "spread"), flush=True)
    for a in arms:
        print("  %-8s %9.3f %+9.3f %9.3f %+9.3f %8.3f"
              % (a, bs[a], bs[a] - cs, bw[a], bw[a] - cw,
                 max(spans[a]) - bs[a]), flush=True)
    print("(spread = max-min of the device spans; d = arm minus ctrl)", flush=True)
    steps = u * int((grid + api._aic_core_count(DEV) - 1) //
                    api._aic_core_count(DEV))
    print("per-chunk gains: a d-span of -0.077 ms = -1 us per subcore chunk "
          "(%d chunks per subcore; %d chunk-instance total)" % (u, 12288),
          flush=True)

    ident = [a for a in arms if a not in WRONG]
    print()
    print("identity vs captured production (out %d / state %d elements):"
          % (out_p.numel(), st_p.numel()), flush=True)
    for a in ident:
        out_t, st_t = run_pipeline(q, k, v, g, beta, kw,
                                   "kda_pg_bg_" + a.replace("+", "x"))
        print("  %-8s out=%d st=%d" % (a, n_diff(out_t, out_p),
                                       n_diff(st_t, st_p)), flush=True)
    print()
    print("timing-only arms (%s) are wrong by construction."
          % ",".join(sorted(WRONG)), flush=True)


if __name__ == "__main__":
    main()
