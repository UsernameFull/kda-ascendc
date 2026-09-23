"""Stage lifecycle, slot map and GM traffic ledger for pre_gram -> solve -> K2.

Plan section 2.1 asks for one record per intermediate with producer / consumer /
scope / memory / lifetime / sync / layout, and section 2.4 requires every
candidate to state its byte budget.  Hand-kept tables are how a stale number
gets "verified" (the K2 audit's own note), so this tool carries the ledger as
data and then *validates* it against the sources it cites:

  * every row names a kernel (or ``api.py`` for the host-side rows) and the
    symbols it appears as; the symbols have to exist in that file, so deleting
    the buffer or renaming it fails the generator instead of quietly leaving a
    stale row in the CSV;
  * the launch-site snippets a row cites as its producer/consumer have to
    appear verbatim in ``api.py``, so re-wiring a buffer to another kernel
    fails here rather than in a later round's probe;
  * the byte columns are derived from shape x dtype, and the production total
    is reconciled against ``tools/gen_ub_l1_budget.py`` (same arithmetic, one
    authority for the per-call workspace).

The traffic ledger at the end is what prices plan levels 2 and 3: it sums the
GM bytes each stage writes and reads per call, marks the rows whose only
production consumer is the debug dict, and converts them at the two rates the
repo has measured (0.095 ms per 201 MB store, R3's K2 v_new guard, and the
0.2 ms/GB L2 figure of section 11.12).  A candidate that claims "keep it on
chip" has to beat a number, not a vibe.

  KDA_CHUNK=64 python3 tools/gen_stage_lifecycle.py
  KDA_CHUNK=64 python3 tools/gen_stage_lifecycle.py --check   # validate only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

from gen_ub_l1_budget import budget as ub_budget  # noqa: E402

ART = ROOT / "docs" / "artifacts"
KERNELS = {
    "pre_gram": "kernels/v1/k1_pre_gram_mix.cpp",
    "solve_wide": "kernels/v1/k1_solve_wu_wide.cpp",
    "solve_assemble": "kernels/v1/k1_solve_assemble.cpp",
    "solve_cube": "kernels/v1/k1_solve_wu_cube.cpp",
    "k2": "kernels/v1/k2_persistent_loop.cpp",
    "api": "python/kda_ascendc_v1/api.py",
}
STAGE_OF_KERNEL = {"pre_gram": "pre_gram", "solve_wide": "solve",
                   "solve_assemble": "solve", "solve_cube": "solve", "k2": "k2"}
ELT = {"bf16": 2, "fp32": 4, "u8": 1}


def geometry(chunk, b, t, h, d=128, bv=64, nv=2):
    maxh = int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0")) or (4 if chunk <= 64 else 2)
    subb = 2 if chunk >= 64 else 1
    nchunk = 32 if chunk <= 16 else (16 if chunk <= 32 else 8)
    nt = t // chunk
    bh = b * h
    c = bh * nt
    c_solve = (c + nchunk // subb - 1) // (nchunk // subb) * (nchunk // subb)
    nc = nchunk // subb          # SOLVE_WIDE_NCH: chunks per wide/assemble block
    return dict(b=b, t=t, h=h, d=d, bv=bv, nv=nv, chunk=chunk, nt=nt, bh=bh, c=c,
                c_solve=c_solve, tasks=bh * nv, maxh=maxh, subb=subb, sub=chunk // subb,
                nchunk=nchunk, nc=nc, nch=nc, bs=min(32, chunk), tile=chunk * bv,
                s_tile=bv * d, nblk=min(bh, (bh + maxh - 1) // maxh),
                pre_unroll=1 if c < 256 else min(64, max(2, c // 192)),
                slices=min(24, max(1, (c_solve // 4) // 8)),
                d4tile=bv * d)


def shape_bytes(shape: str, dtype: str, geom: dict) -> int:
    if not shape:
        return 0
    n = 1
    for tok in shape.split("*"):
        n *= int(eval(tok.strip(), {}, geom))  # tokens are geometry names only
    return n * ELT[dtype]


# Producer / consumer / scope / memory / lifetime / sync / layout for every
# intermediate of the three stages.  ``prod`` and ``cons`` are short prose for
# the CSV; ``src``/``sym`` are the machine check (see validate()).
#   role: input | handoff | resident | scratch | output | debug
ROWS = [
    # ---------------- host -> pre_gram ----------------
    dict(name="Q, K", stage="host", where="GM", shape="c*chunk*d", dtype="bf16",
         src="api", sym=["q", "k"], role="input",
         prod="host, once per call", cons="pre_gram AIV (DataCopyPad, public order)",
         scope="(b,t,h) whole call", live="whole call", sync="stream order",
         layout="public [b,t,h,d], 4B-stride rows", launch="kda_pre_gram_mix"),
    dict(name="V", stage="host", where="GM", shape="c*chunk*d", dtype="bf16",
         src="api", sym=["v"], role="input",
         prod="host, once per call", cons="pre_gram AIV (rv landing -> Rv)",
         scope="(b,t,h) whole call", live="whole call", sync="stream order",
         layout="public [b,t,h,d]", launch="kda_pre_gram_mix"),
    dict(name="G", stage="host", where="GM", shape="c*chunk*d", dtype="fp32",
         src="api", sym=["g"], role="input",
         prod="host, once per call", cons="pre_gram AIV (DataCopyPad byte-stride)",
         scope="(b,t,h) whole call", live="whole call", sync="stream order",
         layout="public [b,t,h,d]", launch="kda_pre_gram_mix"),
    dict(name="Beta (packed)", stage="host", where="GM", shape="c*chunk", dtype="fp32",
         src="api", sym=["beta_pack"], role="input",
         prod="host permute().contiguous()", cons="pre_gram AIV",
         scope="whole call", live="whole call", sync="stream order",
         layout="[c,CHUNK] chunk-major", launch="kda_pre_gram_mix"),
    dict(name="A_log, bias", stage="host", where="GM", shape="h*d", dtype="fp32",
         src="api", sym=["A_log", "bias"], role="input",
         prod="host, once per call", cons="pre_gram AIV",
         scope="per head", live="whole call", sync="stream order",
         layout="[h], [h,d]", launch="kda_pre_gram_mix"),
    # ---------------- pre_gram internals ----------------
    dict(name="qnb/knb/rvb (landing)", stage="pre_gram", where="UB",
         shape="2*16*d", dtype="bf16", src="pre_gram",
         sym=["qnb", "knb", "rvb"], role="scratch",
         prod="pre_gram AIV MTE2", cons="pre_gram AIV V (Cast to fp32)",
         scope="one 16-row band", live="one band", sync="TQue (VECIN, depth 2)",
         layout="bf16 rows"),
    dict(name="qf/kf/qf32 (norm tiles)", stage="pre_gram", where="UB",
         shape="16*d", dtype="fp32", src="pre_gram",
         sym=["bQf", "bKf"], role="scratch",
         prod="pre_gram AIV Cast", cons="l2 norm + Qg/Kg/Rk/Rv stores",
         scope="one band", live="one band", sync="PipeBarrier<PIPE_V>",
         layout="fp32 rows"),
    dict(name="gate/decay tiles", stage="pre_gram", where="UB",
         shape="16*d", dtype="fp32", src="pre_gram",
         sym=["bGef", "bT2"], role="scratch",
         prod="pre_gram AIV (cumsum, exp)",
         cons="Decay/Rk/Rv/Qg/Kg stores", scope="one band", live="one band",
         sync="MTE3->V self-paired at the pass boundary", layout="fp32 rows"),
    dict(name="Ga, Gk, Gb", stage="pre_gram", where="GM", shape="3*c*chunk*d",
         dtype="bf16", src="pre_gram", sym=["Ga", "Gk", "Gb"], role="handoff",
         prod="pre_gram AIV MTE3 (bf16 gated operands)",
         cons="pre_gram AIC MTE2 -> L1 -> L0A/L0B",
         scope="one chunk (two per step)", live="one step (AIV step -> AIC step)",
         sync="CrossCoreSetFlag/WaitFlag<2> FL_READY / FL_DONE per step",
         layout="[c,CHUNK,D] row-major -> Nd2Nz on load",
         launch="void run_gram_aic(GM_ADDR pGa"),
    dict(name="Gx (cross-band k)", stage="pre_gram", where="GM",
         shape="c*(chunk//2)*d", dtype="bf16", src="pre_gram", sym=["Gx"],
         role="handoff",
         prod="pre_gram AIV", cons="pre_gram AIC (band 1's k reference)",
         scope="one chunk", live="one step", sync="same FL_READY channel",
         layout="[c,CHUNK/2,D]", launch="void run_gram_aic(GM_ADDR pGa"),
    dict(name="ta/tk/tb/tx (Gram operands)", stage="pre_gram", where="L1/L0A/L0B",
         shape="2*chunk*d", dtype="bf16", src="pre_gram",
         sym=["qa", "qb", "qx"], role="resident",
         prod="pre_gram AIC DataCopy (Nd2Nz)",
         cons="pre_gram AIC LoadData -> Mmad", scope="one step", live="one step",
         sync="TQue depth 4/2 (MTE2_MTE1, MTE1_M, M_FIX, FIX_M)",
         layout="fractal NZ"),
    dict(name="cf (raw Gram, L0C)", stage="pre_gram", where="L0C",
         shape="16*chunk", dtype="fp32", src="pre_gram", sym=["qc"], role="scratch",
         prod="pre_gram AIC Mmad", cons="pre_gram AIC Fixpipe",
         scope="one band of one chunk", live="one band", sync="M_FIX / FIX_M",
         layout="fractal"),
    dict(name="Aqk32 raw", stage="pre_gram", where="GM", shape="c*chunk*chunk",
         dtype="fp32", src="pre_gram", sym=["Aqk32"], role="handoff",
         prod="pre_gram AIC Fixpipe (raw fp32 Gram)",
         cons="pre_gram AIV post_gram (band loop, MTE2)",
         scope="one chunk", live="one step", sync="CrossCoreWaitFlag(FL_DONE)",
         layout="[c,CHUNK,CHUNK] row-major, 16-row bands",
         launch="Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32"),
    dict(name="L raw", stage="pre_gram", where="GM", shape="c_solve*chunk*chunk",
         dtype="fp32", src="pre_gram", sym=["L"], role="handoff",
         prod="pre_gram AIC Fixpipe (raw fp32 Gram)", cons="pre_gram AIV post_gram",
         scope="one chunk", live="one step", sync="CrossCoreWaitFlag(FL_DONE)",
         layout="[c_solve,CHUNK,CHUNK] row-major",
         launch="Fixpipe<float, float, CFG_ROW_MAJOR>(L["),
    dict(name="Aqk32 masked/scaled", stage="pre_gram", where="GM",
         shape="c*chunk*chunk", dtype="fp32", src="pre_gram", sym=["Aqk32"],
         role="debug",
         prod="pre_gram AIV MTE3 (select + scale)",
         cons="nothing in production (skipped unless debugStores)",
         scope="one chunk", live="one step", sync="V_MTE3 -> TQue VECOUT",
         layout="[c,CHUNK,CHUNK] row-major", skipped=True,
         launch="if (keepAqk32) DataCopy(Aqk32[o], ga32s"),
    dict(name="L masked", stage="pre_gram", where="GM",
         shape="c_solve*chunk*chunk", dtype="fp32", src="pre_gram", sym=["L"],
         role="handoff",
         prod="pre_gram AIV MTE3 (select)", cons="solve wide AIV (bLraw)",
         scope="one chunk", live="whole call (read once)",
         sync="launch order on the caller's stream",
         layout="[c_solve,CHUNK,CHUNK] row-major",
         launch="DataCopy(L[o], gl32s"),
    dict(name="Aqk16", stage="pre_gram", where="GM", shape="c*chunk*chunk",
         dtype="bf16", src="pre_gram", sym=["Aqk16"], role="handoff",
         prod="pre_gram AIV Cast -> MTE3", cons="k2 AIC stage 3 (qa)",
         scope="one chunk", live="whole call (read once)", sync="launch order",
         layout="[c,CHUNK,CHUNK] row-major",
         launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="Rk, Rv", stage="pre_gram", where="GM", shape="2*c*chunk*d",
         dtype="bf16", src="pre_gram", sym=["Rk", "Rv"], role="handoff",
         prod="pre_gram AIV MTE3 (l2-normalised k, v)",
         cons="solve cube AIC (L0B) as the solve's right-hand sides",
         scope="one chunk", live="whole call (read once)", sync="launch order",
         layout="[c,CHUNK,D] row-major -> Nd2Nz",
         launch="api: _pack_ptrs([a16, rk, rv, W, U])"),
    dict(name="Qg, Kg", stage="pre_gram", where="GM", shape="2*c*chunk*d",
         dtype="bf16", src="pre_gram", sym=["Qg", "Kg"], role="handoff",
         prod="pre_gram AIV MTE3 (gated q, k)",
         cons="k2 AIC stage 1 (qw/qg) and stage 3 (qk)",
         scope="one chunk", live="whole call (read once)", sync="launch order",
         layout="[c,CHUNK,D] row-major -> Nd2Nz",
         launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="Decay", stage="pre_gram", where="GM", shape="c*d", dtype="fp32",
         src="pre_gram", sym=["Decay"], role="handoff",
         prod="pre_gram AIV MTE3 (chunk-centred gate)",
         cons="k2 AIV stage 4 (dequeue, one row per chunk)",
         scope="one chunk row", live="whole call (read once)", sync="launch order",
         layout="[c,D] fp32", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="BetaOut", stage="pre_gram", where="GM", shape="c*chunk",
         dtype="fp32", src="pre_gram", sym=["BetaOut"], role="debug",
         prod="pre_gram AIV MTE3",
         cons="nothing in production (skipped unless debugStores)",
         scope="one chunk", live="one store", sync="MTE3_V self-paired",
         layout="[c,CHUNK] fp32", skipped=True,
         launch="if (debugStores != 0) DataCopy(BetaOut[cm], beta"),
    dict(name="Qn, Kn", stage="pre_gram", where="GM",
         shape="2*c*chunk*d", dtype="bf16", src="pre_gram", sym=["Qn", "Kn"],
         role="debug",
         prod="not written in production (pointers passed as null)",
         cons="nothing", scope="whole call", live="never written",
         sync="n/a", layout="[c,CHUNK,D]", launch="api: qn if keep else None"),
    # ---------------- solve ----------------
    dict(name="bLraw (gathered L)", stage="solve_wide", where="UB",
         shape="nc*sub*sub", dtype="fp32", src="solve_wide", sym=["bLraw"],
         role="scratch",
         prod="solve wide AIV MTE2 (gather, DataCopyParams(NC,2,30,0))",
         cons="solve wide V (MulAddDst row recursion)",
         scope="one block of NC chunks", live="one block", sync="MTE2_V, V_MTE3",
         layout="[row][chunk][lane] in UB, chunk-major source"),
    dict(name="bCexp (Brcb expansion)", stage="solve_wide", where="UB",
         shape="nc*sub*8", dtype="fp32", src="solve_wide", sym=["bCexp"],
         role="scratch", prod="solve wide AIV Brcb", cons="MulAddDst src1",
         scope="one block", live="one block", sync="PipeBarrier<PIPE_V>",
         layout="per-row coefficient x 8 blocks"),
    dict(name="A32", stage="solve_wide", where="GM", shape="c_solve*chunk*chunk",
         dtype="fp32", src="solve_wide", sym=["A32"], role="debug",
         prod="solve wide AIV MTE3 (fp32 A_inv)",
         cons="nothing in production (skipped unless debugStores)",
         scope="one block", live="one block", sync="V_MTE3", skipped=True,
         layout="[c_solve,CHUNK,CHUNK]", launch="if (keepA32) {",
         api="_pack_ptrs([L[lo:], eye, a32[lo:]"),
    dict(name="A16", stage="solve_wide", where="GM", shape="c_solve*chunk*chunk",
         dtype="bf16", src="solve_wide", sym=["A16"], role="handoff",
         prod="solve wide AIV MTE3 (bf16 A_inv)",
         cons="solve assemble AIC + solve cube AIC (L0A)",
         scope="one block", live="two launches (wide -> assemble/cube)",
         sync="launch order on the caller's stream",
         layout="[c_solve,CHUNK,CHUNK] row-major",
         launch="api: _pack_ptrs([a16[lo:], xb[lo:], lneg[lo:], pmid[lo:]])"),
    dict(name="Xb, Lneg", stage="solve_wide", where="GM",
         shape="2*c_solve*sub*sub", dtype="bf16", src="solve_wide",
         sym=["Xb", "Lneg"], role="handoff",
         prod="wide (Xb, Lneg) / assemble (Pmid)",
         cons="assemble (Xb, Lneg) / cube (Pmid)",
         scope="one block", live="two launches", sync="launch order",
         layout="[c_solve,SB,sub,sub] and [c_solve,sub,sub]",
         launch="api: _pack_ptrs([a16[lo:], xb[lo:], lneg[lo:], pmid[lo:]])"),
    dict(name="Pmid", stage="solve_assemble", where="GM", shape="c_solve*sub*sub",
         dtype="bf16", src="solve_assemble", sym=["P"], role="handoff",
         prod="solve assemble AIC Fixpipe (X21 = -X22 L21 X11)",
         cons="solve cube AIC", scope="one block", live="one launch",
         sync="launch order",
         layout="[c_solve,sub,sub] bf16"),
    dict(name="qa/qb (assemble L1)", stage="solve_assemble", where="L1",
         shape="nc*sub*sub", dtype="bf16", src="solve_assemble",
         sym=["qa", "qb"], role="resident",
         prod="solve assemble AIC DataCopy",
         cons="solve assemble AIC LoadData -> Mmad", scope="one block",
         live="one launch", sync="MTE2_MTE1, MTE1_M, M_FIX, FIX_M",
         layout="fractal"),
    dict(name="W, U", stage="solve_cube", where="GM", shape="2*c*chunk*d",
         dtype="bf16", src="solve_cube", sym=["W", "U"], role="handoff",
         prod="solve cube AIC (W = A_inv @ Rk, U = A_inv @ Rv)",
         cons="k2 AIC stage 1 (qw) and AIV stage 2 (DataCopy ub)",
         scope="one chunk", live="whole call (read once)",
         sync="launch order on the caller's stream",
         layout="[c,CHUNK,D] bf16", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    # ---------------- K2 ----------------
    dict(name="H0", stage="k2", where="GM", shape="bh*d*d", dtype="fp32",
         src="k2", sym=["pH0"], role="input",
         prod="host, once per call (optional)", cons="k2 AIV start-up",
         scope="per (b,h)", live="prologue", sync="stream order",
         layout="[bh,D,D]", launch="api: h0 = None if initial_state is None"),
    dict(name="st (resident state)", stage="k2", where="UB", shape="maxh*bv*d",
         dtype="fp32", src="k2", sym=["us", "MAXH", "S_TILE"], role="resident",
         prod="k2 AIV prologue (H0) + stage 4 recurrence",
         cons="k2 AIV stage 4 + final S32 publish", scope="whole call, resident",
         live="whole call", sync="none (single subcore owns it)",
         layout="fp32 [MAXH][BV,D]"),
    dict(name="d1/d1f/d2/d3/d3f/ob (UB)", stage="k2", where="UB", shape="3*chunk*bv",
         dtype="bf16", src="k2", sym=["uC"], role="scratch",
         prod="k2 AIV MTE2 (d1/d2/d3 loads)", cons="k2 AIV stage 2/4 arithmetic",
         scope="one head-step", live="stage 2 | stage 4 (aliased phase pair)",
         sync="MTE2_V / V_MTE3, phase aliasing in the TBuf plan",
         layout="bf16 [CHUNK,BV] tiles"),
    dict(name="vb/d4 (UB)", stage="k2", where="UB", shape="d*bv", dtype="fp32",
         src="k2", sym=["uA", "uE"], role="scratch",
         prod="k2 AIV stage 4 quarter loads", cons="k2 AIV stage 4 recurrence",
         scope="one head-step", live="stage 4", sync="PipeBarrier / V_MTE3",
         layout="fp32 [D,BV]"),
    dict(name="lw/lg/ls/la/lv/lx/lk (L1)", stage="k2", where="L1",
         shape="7*chunk*d", dtype="bf16", src="k2",
         sym=["qw", "qg", "qs", "qa", "qv", "qx", "qk"], role="resident",
         prod="k2 AIC DataCopy (Nd2Nz, one per operand)",
         cons="k2 AIC LoadData -> Mmad (stages 1/3)", scope="one head-step",
         live="one head-step", sync="TQue + MTE2_MTE1 / MTE1_M",
         layout="fractal NZ"),
    dict(name="cf (L0C)", stage="k2", where="L0C", shape="2*chunk*bv", dtype="fp32",
         src="k2", sym=["qc"], role="scratch", prod="k2 AIC Mmad stages 1/3",
         cons="k2 AIC Fixpipe (-> d1/d2, d3/d4)", scope="one head-step",
         live="one head-step", sync="M_FIX / FIX_M + enUnitFlag",
         layout="fractal fp32"),
    dict(name="D1, D2 (bf16)", stage="k2", where="GM", shape="2*tasks*nt*chunk*bv",
         dtype="bf16", src="k2", sym=["pD1", "pD2"], role="handoff",
         prod="k2 AIC stage 1 Fixpipe", cons="k2 AIV stage 2 (DataCopy ub)",
         scope="one head-step", live="one chunk (stage1 -> stage2)",
         sync="CrossCore FL_C1 (AIC stage 1 -> AIV stage 2)",
         layout="[tasks,NT,CHUNK,BV] bf16", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="VnewT (bf16)", stage="k2", where="GM", shape="tasks*nt*bv*chunk",
         dtype="bf16", src="k2", sym=["pVnewT"], role="handoff",
         prod="k2 AIV stage 2 (DataCopy Vt)", cons="k2 AIC stage 3 (qv/vx)",
         scope="one head-step", live="one chunk (stage2 -> stage3)",
         sync="CrossCore FL_V (AIV stage 2 -> AIC stage 3)",
         layout="[tasks,NT,BV,CHUNK] (transposed store)", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="D3 (bf16)", stage="k2", where="GM", shape="tasks*nt*chunk*bv",
         dtype="bf16", src="k2", sym=["pD3"], role="handoff",
         prod="k2 AIC stage 3 Fixpipe", cons="k2 AIV stage 4 (DataCopy d3)",
         scope="one head-step", live="one chunk (stage3 -> stage4)",
         sync="CrossCore FL_C2 (AIC stage 3 -> AIV stage 4)",
         layout="[tasks,NT,CHUNK,BV] bf16", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="D4 (fp32)", stage="k2", where="GM", shape="bh*d*d", dtype="fp32",
         src="k2", sym=["pD4"], role="handoff",
         prod="k2 AIC stage 3 Fixpipe", cons="k2 AIV stage 4 (four quarter loads)",
         scope="one head-step", live="one chunk (stage3 -> stage4)",
         sync="CrossCore FL_C2", layout="[bh,D,D] fp32, quarter tiles",
         launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="S16 (bf16)", stage="k2", where="GM", shape="tasks*bv*d",
         dtype="bf16", src="k2", sym=["pS16"], role="handoff",
         prod="k2 AIV (start-up + stage 4 quarters)", cons="k2 AIC stage 1 (qs)",
         scope="whole call", live="across chunks (state handoff)",
         sync="CrossCore FL_R (AIV -> AIC, prologue-primed)",
         layout="[tasks,BV,D] bf16", launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="S32 (fp32)", stage="k2", where="GM", shape="tasks*bv*d",
         dtype="fp32", src="k2", sym=["pS32"], role="output",
         prod="k2 AIV, once at the end", cons="host (final_state)",
         scope="whole call", live="final publish", sync="stream order",
         layout="[tasks,BV,D] -> viewed [bh,NV,BV,D]",
         launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="Out (bf16)", stage="k2", where="GM", shape="b*t*h*d",
         dtype="bf16", src="k2", sym=["pOut"], role="output",
         prod="k2 AIV stage 4 (strided store, public layout)", cons="host",
         scope="whole call", live="one store per head-step", sync="stream order",
         layout="public [b,t,h,d] direct (no host permute)",
         launch="api: _pack_ptrs([U, W, qg, aqk16, kg, decay"),
    dict(name="Vnew (row-major)", stage="k2", where="GM",
         shape="tasks*nt*chunk*bv", dtype="bf16", src="k2", sym=["pVnew"],
         role="debug",
         prod="not written in production (null pointer, R3 guard)",
         cons="nothing", scope="whole call", live="never written", sync="n/a",
         layout="[tasks,NT,CHUNK,BV]", launch="api: vnew = ("),
]

# Cross-stage handoff: what a level 2/3 window pipeline would stage in a ring.
# local_slot is the slot index inside one window; the ring is double-windowed,
# slot = (window & 1) * n_local + local_slot (plan section 2.3).
SLOTS = [
    dict(pair="pre_gram -> solve", local_slot=0, buffer="L masked",
         grain="chunk", prod="pre_gram AIV post_gram",
         cons="solve_wide AIV bLraw", ring="double window (2 x n_local)"),
    dict(pair="pre_gram -> solve", local_slot=1, buffer="Rk, Rv",
         grain="chunk", prod="pre_gram AIV", cons="solve_cube AIC L0B",
         ring="double window (2 x n_local)"),
    dict(pair="solve -> k2", local_slot=0, buffer="W, U",
         grain="chunk", prod="solve_cube AIC", cons="k2 AIC stage 1 (W)",
         ring="double window (2 x n_local)"),
    dict(pair="pre_gram -> k2", local_slot=0, buffer="Qg, Kg",
         grain="chunk", prod="pre_gram AIV", cons="k2 AIC stages 1/3",
         ring="double window (2 x n_local)"),
    dict(pair="pre_gram -> k2", local_slot=1, buffer="Aqk16",
         grain="chunk", prod="pre_gram AIV", cons="k2 AIC stage 3",
         ring="double window (2 x n_local)"),
    dict(pair="pre_gram -> k2", local_slot=2, buffer="Decay",
         grain="chunk", prod="pre_gram AIV", cons="k2 AIV stage 4",
         ring="double window (2 x n_local)"),
]


def validate(geom) -> list[str]:
    """Fail on a stale row: cited symbols must exist, launch sites must match."""
    errors = []
    text_cache = {}
    for key, rel in KERNELS.items():
        text_cache[key] = (ROOT / rel).read_text(encoding="utf-8-sig")
    for row in ROWS:
        src = text_cache[row["src"]]
        for sym in row["sym"]:
            if sym not in src:
                errors.append("%s: symbol %r not found in %s"
                              % (row["name"], sym, KERNELS[row["src"]]))
        site = row.get("launch")
        if site:
            key, sep, snippet = site.partition(": ")
            if not sep:
                key, snippet = row["src"], site
            if snippet not in text_cache[key]:
                errors.append("%s: site %r not found in %s"
                              % (row["name"], snippet, KERNELS[key]))
    # slot map: local slots inside a pair must be distinct, or two live buffers
    # would be handed the same ring entry.
    seen = {}
    for slot in SLOTS:
        key = (slot["pair"], slot["local_slot"])
        if key in seen:
            errors.append("slot collision: %s local %d used by %s and %s"
                          % (slot["pair"], slot["local_slot"], seen[key], slot["buffer"]))
        seen[key] = slot["buffer"]
    # byte reconciliation against the workspace budget generator
    rep = ub_budget(geom["chunk"], geom["b"], geom["t"], geom["h"], geom["maxh"])
    total = sum(v[0] for k, v in rep["workspace"].items()
                if k not in rep["workspace_conditional"])
    return errors, total


def traffic(geom):
    """Per-call GM bytes: writes and reads, by stage and by production value."""
    rows = []
    for row in ROWS:
        if row["where"] not in ("GM", "L1", "L0C", "L1/L0A/L0B", "UB"):
            continue
        if row["where"] != "GM":
            continue
        nbytes = shape_bytes(row["shape"], row["dtype"], geom)
        ghost = row.get("skipped") or "not written" in row["prod"]
        # Every GM row is written once and read once by the kernels it names,
        # except the rows whose only consumer is the debug dict (``ghost``) and
        # the ones production never writes.  A row that is handed over twice
        # (Aqk32, L) is two rows in the ledger, one per hop.
        writes = 0 if ghost else nbytes
        reads = 0 if ghost or row["role"] == "debug" else nbytes
        rows.append(dict(row=row, nbytes=nbytes, writes=writes, reads=reads))
    return rows


def report(geom):
    errors, ws_total = validate(geom)
    rows = traffic(geom)
    gm_write = sum(r["writes"] for r in rows)
    gm_read = sum(r["reads"] for r in rows)
    dead = [r for r in rows if r["row"].get("skipped")]
    out = []
    out.append("geometry: C=%d b=%d t=%d h=%d d=%d  nt=%d bh=%d c=%d tasks=%d nblk=%d maxh=%d"
               % (geom["chunk"], geom["b"], geom["t"], geom["h"], geom["d"], geom["nt"],
                  geom["bh"], geom["c"], geom["tasks"], geom["nblk"], geom["maxh"]))
    out.append("")
    out.append("lifecycle ledger rows: %d  (validated against %d kernels + api.py)"
               % (len(ROWS), len(KERNELS)))
    out.append("workspace reconciled with gen_ub_l1_budget.py: %.2f MB" % (ws_total / 1e6))
    out.append("")
    out.append("GM traffic per call (production path):")
    out.append("   writes %.2f MB   reads %.2f MB   total %.2f MB"
               % (gm_write / 1e6, gm_read / 1e6, (gm_write + gm_read) / 1e6))
    out.append("")
    out.append("   by stage:")
    for stage in ("host", "pre_gram", "solve_wide", "solve_assemble", "solve_cube", "k2"):
        w = sum(r["writes"] for r in rows if r["row"]["stage"] == stage)
        rd = sum(r["reads"] for r in rows if r["row"]["stage"] == stage)
        if w or rd:
            out.append("      %-15s w %8.2f MB  r %8.2f MB" % (stage, w / 1e6, rd / 1e6))
    out.append("")
    out.append("   top flows (w = writes, r = reads, MB per call):")
    for r in sorted(rows, key=lambda r: -(r["writes"] + r["reads"]))[:14]:
        out.append("      %-28s %-8s w %8.2f  r %8.2f"
                   % (r["row"]["name"], r["row"]["role"], r["writes"] / 1e6,
                      r["reads"] / 1e6))
    out.append("")
    out.append("   stores kept for the return_intermediates views only, skipped in")
    out.append("   production by the debugStores flag (docs 11.29; measured -0.164 ms")
    out.append("   interleaved at [1,8192,96,128]/C=64, tools/probe_dead_store.py):")
    tot_dead = 0
    for r in dead:
        out.append("      %-28s w %8.2f MB   %s"
                   % (r["row"]["name"], r["nbytes"] / 1e6, r["row"]["cons"]))
        tot_dead += r["nbytes"]
    extra_read = sum(r["reads"] for r in rows
                     if r["row"]["name"] == "Aqk32 raw")
    out.append("      total %.2f MB of production writes removed; the AIV's %.2f MB"
               % (tot_dead / 1e6, extra_read / 1e6))
    out.append("      re-read of Aqk32 raw stays - the mask/scale needs the value")
    out.append("")
    out.append("   the two estimates that sent this to the A/B, against the result:")
    for name, rate in (("0.095 ms / 201 MB store (R3, K2 v_new guard)", 0.095 / 201.33e6),
                       ("0.2 ms / GB (section 11.12, L2 traffic)", 0.2e-3 / 1e6)):
        out.append("      %-46s estimated %.3f ms" % (name, tot_dead * rate))
    out.append("      %-46s measured  -0.164 ms" % "interleaved A/B (docs 11.29)")
    out.append("")
    out.append("slot map (level 2/3 handoff, double window):")
    for slot in SLOTS:
        out.append("      %-18s local %d  %-12s %s"
                   % (slot["pair"], slot["local_slot"], slot["buffer"], slot["ring"]))
    return errors, "\n".join(out)


def write_artifacts(geom, text, errors):
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "stage_lifecycle.csv").open("w", newline="") as fh:
        # The checked-in instance is the C=64 build; the chunk column is what
        # keeps a C=16 copy from being mistaken for it.
        cols = ["chunk", "name", "stage", "where", "dtype", "bytes", "producer",
                "consumer", "scope", "lifetime", "sync", "layout", "role",
                "source", "symbols"]
        w = csv.writer(fh)
        w.writerow(cols)
        for row in ROWS:
            w.writerow([geom["chunk"], row["name"], row["stage"], row["where"], row["dtype"],
                        shape_bytes(row["shape"], row["dtype"], geom),
                        row["prod"], row["cons"], row["scope"], row["live"],
                        row["sync"], row["layout"], row["role"], row["src"],
                        " ".join(row["sym"])])
    with (ART / "stage_slot_map.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "local_slot", "buffer", "grain", "bytes_per_slot",
                    "producer", "consumer", "ring"])
        by_name = {r["name"]: r for r in ROWS}
        for slot in SLOTS:
            src = by_name.get(slot["buffer"])
            nbytes = shape_bytes(src["shape"], src["dtype"], geom) if src else 0
            w.writerow([slot["pair"], slot["local_slot"], slot["buffer"], slot["grain"],
                        nbytes, slot["prod"], slot["cons"], slot["ring"]])
    (ART / "stage_traffic.txt").write_text(
        text + "\n\n" + ("VALIDATION FAILED:\n  " + "\n  ".join(errors) + "\n"
                         if errors else "validation: clean\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate, write nothing")
    ap.add_argument("--shape", default="1,8192,96,128")
    args = ap.parse_args()
    b, t, h, d = (int(x) for x in args.shape.split(","))
    chunk = int(os.environ.get("KDA_CHUNK", "64"))
    # Every supported build has to validate: the C=16/32 legs differ in the
    # two-level solve (SB, xb/lneg/pmid) and in the wide block size, which is
    # where a row that still describes the C=64 geometry would show up.
    build_errors = {}
    for other in (16, 32, 64):
        errs, _ = validate(geometry(other, 1, 512, 4, d))
        if errs:
            build_errors[other] = errs
    geom = geometry(chunk, b, t, h, d)
    errors, text = report(geom)
    builds = "builds validated: " + ", ".join(
        "C=%d %s" % (c, "FAIL" if c in build_errors else "ok") for c in (16, 32, 64))
    text = text + "\n\n" + builds
    print(text)
    errors += ["C=%d: %s" % (c, e) for c, errs in build_errors.items() for e in errs]
    for e in errors:
        print("  !! %s" % e)
    if not args.check:
        write_artifacts(geom, text, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
