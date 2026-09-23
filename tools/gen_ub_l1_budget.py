"""UB / L1 / workspace byte budget at the production geometry.

The plan's first round wants the byte budget as a generated artifact rather
than a hand-kept comment, because the aliasing in K2 (and the pools the S1-S3
route would add) is exactly where a stale hand-written number lets an overrun
through: a UB overrun is a kernel-side error, not a host one, so the arithmetic
has to be executable.

Three budgets, all from the accepted production geometry:

  UB   K2's resident state (MAXH x S_TILE fp32) + its aliased staging + the
       L1 queues it also charges to UB, against the 192 KB part.  The amounts
       are read out of the kernel's own comments/constants where possible.
  L1   the seven B1 queues and the CO1 tile, against 512 KB.
  work the per-call workspace the api allocates, per tensor and in total, plus
       the peak-live set (what a fixed-shape pool would have to hold).

  KDA_CHUNK=64 python3 tools/gen_ub_l1_budget.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

D, BV, NV = 128, 64, 2


def budget(chunk: int, b: int, t: int, h: int, maxh: int) -> dict:
    FR = 16
    M = chunk
    N = BV
    NB = M // FR
    NG = 2 * BV // M
    TILE = M * BV              # one (chunk x value-half) tile
    S_TILE = BV * D            # fp32 state, one head
    K = M
    L0A_ELEMS = M * D * 2      # two operand tiles: W and Qg
    ub_state = maxh * S_TILE * 4
    # Staging from the kernel header: A|E|B|C|D with the phase aliasing.
    # HALF_BYTES is max(TILE, S_TILE / 2) *elements* converted to bytes: at
    # C=64 both branches are 8 KB, while at C=16 the bf16 state half (8 KB) is
    # four times a 16-row tile (2 KB), which is why the max is there at all.
    HALF_BYTES = max(TILE * 2, S_TILE)   # B, D  (bf16 TILE or the state half)
    staging = (TILE * 4                        # A  fp32 TILE (vf | of)
               + 2 * TILE * 2                  # E  sc, vt (+ d1f/d2f/d3f view)
               + 2 * HALF_BYTES                # B, D
               + TILE * 2                      # C  d1/d3/ob
               + D * 4)                        # dec
    # The resident fp32 state is `us` = MAXH * S_TILE fp32 (the whole thing,
    # not a half: S_TILE is one head's BV x D state, so `us` is UB budget 1).
    l1 = (M * D * 2 * 2                 # qw, qg
          + NV * BV * D * 2             # qs
          + M * K * 2                   # qa
          + NG * M * K * 2              # qv
          + NV * BV * K * 2             # qx
          + D * K * 2)                  # qk
    co1 = (NG * M * N + NV * M * N) * 4
    l0a = L0A_ELEMS * 2
    l0b = NV * BV * D * 2

    nt = t // chunk
    bh = b * h
    c = bh * nt
    tasks = bh * NV
    SOLVE_SUBB = 2 if chunk >= 64 else 1
    sub = chunk // SOLVE_SUBB
    ws = {
        "qn": (c * chunk * D, "bf16"), "kn": (c * chunk * D, "bf16"),
        "gate": (c * chunk * D, "fp32"), "gc": (c * chunk * D, "fp32"),
        "beta_out": (c * chunk, "fp32"), "decay": (c * D, "fp32"),
        "rk": (c * chunk * D, "bf16"), "rv": (c * chunk * D, "bf16"),
        "qg": (c * chunk * D, "bf16"), "kg": (c * chunk * D, "bf16"),
        "aqk32": (c * chunk * chunk, "fp32"), "aqk16": (c * chunk * chunk, "bf16"),
        "L": (c * chunk * chunk, "fp32"),
        "a32": (c * chunk * chunk, "fp32"), "a16": (c * chunk * chunk, "bf16"),
        "W": (c * chunk * D, "bf16"), "U": (c * chunk * D, "bf16"),
        # The two-level solve (SOLVE_WIDE_SUBB > 1) only exists at C >= 64, so
        # its three staging buffers are build-dependent and are added below.
        "xb": None, "lneg": None, "pmid": None,
        "s32": (tasks * BV * D, "fp32"), "s16": (tasks * BV * D, "bf16"),
        # The route-1 split's chunk-entry-state snapshot (k2_state_loop writes
        # it, k2_out_parallel reads it).  It replaces the fused loop's single
        # S16 slot, which is why the split's extra GM traffic is the *read*
        # side only: [tasks, nt, BV, D] bf16 = 402.65 MB at the target shape.
        "hsnap": (tasks * nt * BV * D, "bf16"),
        "d1": (tasks * nt * chunk * BV, "bf16"),
        "d2": (tasks * nt * chunk * BV, "bf16"),
        "d3": (tasks * nt * chunk * BV, "bf16"),
        "d4f": (bh * D * D, "fp32"),
        "out_public": (b * t * h * D, "bf16"),
        "vnew_t": (tasks * nt * BV * chunk, "bf16"),
        "vnew": (tasks * nt * chunk * BV, "bf16"),
    }
    # `vnew` and `vnew_t` exist only for return_intermediates (production passes
    # a null pointer and skips the row-major store), so they are reported but
    # not counted in the production total.
    conditional = {"vnew", "hsnap"}
    elt = {"bf16": 2, "fp32": 4}
    if chunk >= 64:
        ws["xb"] = (c * SOLVE_SUBB * sub * sub, "bf16")
        ws["lneg"] = (c * sub * sub, "bf16")
        ws["pmid"] = (c * sub * sub, "bf16")
    priced: dict[str, tuple[int, str]] = {}
    for name, spec in ws.items():
        if spec is None:
            continue          # build-dependent buffer not present at this chunk
        n, dt = spec
        priced[name] = (int(n) * elt[dt], dt)

    # `vnew` and `vnew_t` exist only for return_intermediates (production passes
    # a null pointer and skips the row-major store), so they are reported but
    # not counted in the production total.
    conditional = {"vnew", "hsnap"}
    elt = {"bf16": 2, "fp32": 4}
    if chunk >= 64:
        ws["xb"] = (c * SOLVE_SUBB * sub * sub, "bf16")
        ws["lneg"] = (c * sub * sub, "bf16")
        ws["pmid"] = (c * sub * sub, "bf16")
    ws_prod = {k: v for k, v in priced.items() if k not in conditional}
    return dict(chunk=chunk, shape=(b, t, h, D), maxh=maxh,
                ub=dict(state_bytes=ub_state, staging_bytes=staging,
                        total=ub_state + staging, cap=192 * 1024),
                l1=dict(queues=l1, co1=co1, l0a=l0a, l0b=l0b,
                        l1_total=l1, l1_cap=512 * 1024,
                        l0c_cap=128 * 1024, l0c_used=co1),
                workspace=priced, workspace_conditional=sorted(conditional),
                workspace_total=sum(v[0] for v in ws_prod.values()))
    return dict(chunk=chunk, shape=(b, t, h, D), maxh=maxh,
                ub=dict(state_bytes=ub_state, staging_bytes=staging,
                        total=ub_state + staging, cap=192 * 1024),
                l1=dict(queues=l1, co1=co1, l0a=l0a, l0b=l0b,
                        l1_total=l1, l1_cap=512 * 1024,
                        l0c_cap=128 * 1024, l0c_used=co1),
                workspace=ws, workspace_total=sum(v[0] for v in ws.values()))


def split_budget(chunk: int, maxh: int) -> dict:
    """UB / L1 / L0 of the two route-1 kernels (k2_state_loop + k2_out_parallel).

    Same arithmetic as ``budget`` - the numbers are the kernels' own
    ``InitBuffer`` calls, restated in the tool because the tools cannot parse
    the source - so the split's admission check is executable like the fused
    loop's.  ``k2_state_loop`` is the fused loop's staging minus the output
    side (no ``ob``/``of``/d2f/d3f: its uA is vf only), and ``k2_out_parallel``
    has no resident state at all, which is why the output half costs 56 KB of UB
    against the fused loop's 184.5 KB.
    """
    FR, BV, NV, D = 16, 64, 2, 128
    M = chunk
    NB = M // FR
    NG = 2 * BV // M
    TILE = M * BV
    S_TILE = BV * D
    HALF_BYTES = max(TILE * 2, S_TILE)          # uB/uD of the state loop
    state_ub = (maxh * S_TILE * 4               # us: resident fp32 state
                + TILE * 4                      # uA vf
                + HALF_BYTES                    # uB ub | s16
                + TILE * 2                      # uC d1
                + HALF_BYTES                    # uD vb | d4
                + TILE * 4                      # uE sc/vt | d1f
                + D * 4)                        # udec
    state_l1 = (M * D * 2                       # qw
                + NV * BV * D * 2               # qs
                + NG * M * M * 2                # qv
                + D * M * 2)                    # qk
    state_l0a = max(M * D, NG * M * M) * 2
    state_l0b = NV * BV * D * 2
    state_l0c = (NG * M * D + NV * M * BV) * 4
    out_ub = 2 * TILE * 4 + 3 * TILE * 2        # of, scratch | d2, d3, ob
    out_l1 = (M * D * 2 + NV * BV * D * 2 + M * M * 2 + NV * BV * M * 2)
    out_l0a = (M * D + M * M) * 2
    out_l0b = (NV * BV * D + NV * BV * M) * 2
    out_l0c = 2 * NV * M * BV * 4
    return {
        "state": dict(ub=state_ub, l1=state_l1, l0a=state_l0a, l0b=state_l0b, l0c=state_l0c),
        "out": dict(ub=out_ub, l1=out_l1, l0a=out_l0a, l0b=out_l0b, l0c=out_l0c),
        "caps": dict(ub=192 * 1024, l1=512 * 1024, l0a=64 * 1024,
                     l0b=64 * 1024, l0c=128 * 1024),
    }


def pre_gram_ub(chunk: int) -> dict:
    """What the fused stage-1 kernel charges, at a given KDA_CHUNK.

    Route 3 of the 2026-09-23 order is "layered C128": keep the chunk-generic
    kernels, move the logical chunk to 128, lay the work out so no whole
    [128, 128] intermediate is resident.  The first half of that is a
    *feasibility* question - does the existing kernel even fit at M = 128 - and
    it is arithmetic, not a measurement: the kernel's InitBuffer list is an
    expression list in M, D, N = M*D, NG = 16*D and BS = min(32, M), so the
    total can be evaluated instead of argued about.

    Read out of the source rather than re-typed: the two terms that break first
    (bT0 = N * 4, the whole-chunk gate cumsum, and the four qg* queues =
    2 * 16 * M * 4 per slot, the band staging) are exactly the ones a
    hand-kept copy drifts on.  The two budget regions are kept apart because
    they are: the AIV half (kda_pre_gram_mix's own TPipe, the 192 KB UB part)
    and the paired Cube's A1/B1/CO1 queues (run_gram_aic's TPipe, L1), which
    are allocated in the same file but not in the same memory.
    """
    import re
    src = (ROOT / "kernels/v1/k1_pre_gram_mix.cpp").read_text(encoding="utf-8-sig")
    aiv = src[src.index("kda_pre_gram_mix("):]
    aic = src[src.index("run_gram_aic("):src.index("kda_pre_gram_mix(")]
    env = {"M": chunk, "D": D, "N": chunk * D, "MM": chunk * chunk,
           "NG": 16 * D, "MT": 16 if chunk > 16 else chunk,
           "BS": 32 if chunk > 32 else chunk, "KF": max(1, chunk // 16)}

    def parse(text):
        rows = []
        for m in re.finditer(r"pipe\.InitBuffer\((\w+),\s*(\d+),\s*([^;]+)\);", text):
            rows.append((m.group(1), int(m.group(2)) * eval(m.group(3), {}, env)))
        for m in re.finditer(r"pipe\.InitBuffer\((\w+),(?!\s*\d+\s*,)\s*([^;]+)\);", text):
            rows.append((m.group(1), eval(m.group(2), {}, env)))
        return rows

    aiv_rows, aic_rows = parse(aiv), parse(aic)
    # The two L0 operand slabs and the L0C slot are declared as raw
    # LocalTensor<uint8_t> views rather than through InitBuffer, and at M = 128
    # they are the *other* wall (L0A 128 KB and L0B 80 KB against 64 KB each,
    # L0C exactly 128 KB) - see api.py's refusal note for the same arithmetic.
    xband = chunk > 32
    bs = 32 if xband else chunk
    l0a = 4 * chunk * D * 2
    l0b = 2 * chunk * D * 2 + (2 * bs * D * 2 if xband else 0)
    l0c = 2 * chunk * chunk * 4
    return {"chunk": chunk, "l0a": l0a, "l0b": l0b, "l0c": l0c,
            "ub": sum(b for _, b in aiv_rows), "ub_cap": 192 * 1024,
            "l1": sum(b for _, b in aic_rows), "l1_cap": 512 * 1024,
            "ub_biggest": sorted(aiv_rows, key=lambda r: -r[1])[:6],
            "l1_biggest": sorted(aic_rows, key=lambda r: -r[1])[:4]}


def main():
    chunk = int(os.environ.get("KDA_CHUNK", "64"))
    print("fused stage-1 (k1_pre_gram_mix.cpp) across the chunk sizes,"
          " InitBuffer evaluated from source:")
    print("   %-6s %22s %22s" % ("", "AIV half (UB)", "Cube half (L1 queues)"))
    for c in (16, 32, 64, 96, 128):
        pg = pre_gram_ub(c)
        def fmt(used, cap):
            over = used - cap
            return ("%7.1f / %3d KB  %s" % (used / 1024, cap // 1024,
                    ("%+.1f OVER" % (over / 1024)) if over > 0 else
                    ("-%.1f free" % (-over / 1024))))
        print("   C=%-4d %22s %22s" % (c, fmt(pg["ub"], pg["ub_cap"]),
                                       fmt(pg["l1"], pg["l1_cap"])))
    pg = pre_gram_ub(128)
    print("   C=128 AIV terms: " + ", ".join("%s %.1f KB" % (n, b / 1024)
                                             for n, b in pg["ub_biggest"]))
    print("   C=128 L1 terms:  " + ", ".join("%s %.1f KB" % (n, b / 1024)
                                             for n, b in pg["l1_biggest"]))
    for c in (16, 32, 64, 96, 128):
        pg = pre_gram_ub(c)
        print("   C=%-4d L0: A2 %.1f / 64 KB, B2 %.1f / 64 KB, CO1 %.1f / 128 KB"
              % (c, pg["l0a"] / 1024, pg["l0b"] / 1024, pg["l0c"] / 1024))
    print()
    # MAXH is derived the same way api.py derives it, so this tool does not
    # have to import the api (which pulls in the RTC launcher extension): a
    # budget generator that needs a compiled extension to run is one more thing
    # that can be stale.
    maxh = int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0")) or (4 if chunk <= 64 else 2)
    rep = budget(chunk, 1, 8192, 96, maxh)
    ub = rep["ub"]
    print("geometry: C=%d  shape=%s  MAXH=%d" % (rep["chunk"], rep["shape"], rep["maxh"]))
    print()
    print("UB budget (per AIV subcore, cap %d KB):" % (ub["cap"] // 1024))
    print("   resident fp32 state  MAXH * S_TILE      %8.1f KB" % (ub["state_bytes"] / 1024))
    print("   aliased staging                          %8.1f KB" % (ub["staging_bytes"] / 1024))
    print("   TOTAL                                    %8.1f KB   headroom %.1f KB"
          % (ub["total"] / 1024, (ub["cap"] - ub["total"]) / 1024))
    l1 = rep["l1"]
    print()
    print("L1 / L0 budget (per AIC):")
    print("   7 B1 queues                              %8.1f KB" % (l1["queues"] / 1024))
    print("   L1 cap                                   %8.1f KB   headroom %.1f KB"
          % (l1["l1_cap"] / 1024, (l1["l1_cap"] - l1["l1_total"]) / 1024))
    print("   CO1 (L0C)                                %8.1f KB of %d KB"
          % (l1["co1"] / 1024, l1["l0c_cap"] // 1024))
    print("   L0A                                      %8.1f KB of 64 KB" % (l1["l0a"] / 1024))
    print("   L0B                                      %8.1f KB of 64 KB" % (l1["l0b"] / 1024))
    print()
    print("workspace (allocated per call):")
    for k, (nbytes, dt) in sorted(rep["workspace"].items(), key=lambda kv: -kv[1][0]):
        cond = "  [intermediates only]" if k in rep["workspace_conditional"] else ""
        print("   %-12s %10.2f MB  %s%s" % (k, nbytes / 1e6, dt, cond))
    print("   %-12s %10.2f MB  (production)" % ("TOTAL", rep["workspace_total"] / 1e6))
    print("   peak live     %10.2f MB  (guard: the pool has to hold this)"
          % (max(v[0] for v in rep["workspace"].values()) / 1e6))

    # The route-1 split (k2_state_loop + k2_out_parallel).  It is a candidate,
    # not the shipped path, but it lives in the tree and its admission numbers
    # have to be executable like everything else - the measurement that
    # rejected it (docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md section 11.30) is
    # about time, not about a budget overrun.
    sb = split_budget(chunk, maxh)
    print()
    print("route-1 split (k2_state_loop + k2_out_parallel), same caps:")
    print("   %-22s %8s %8s %8s" % ("", "UB", "L1", "L0C"))
    for name in ("state", "out"):
        row = sb[name]
        print("   %-22s %7.1fK %7.1fK %7.1fK   L0A %.1fK  L0B %.1fK"
              % (name, row["ub"] / 1024, row["l1"] / 1024, row["l0c"] / 1024,
                 row["l0a"] / 1024, row["l0b"] / 1024))
    for name in ("state", "out"):
        row = sb[name]
        over = [k for k in ("ub", "l1", "l0a", "l0b", "l0c")
                if row[k] > sb["caps"][k]]
        print("   %-22s %s" % (name, "OVER: " + ", ".join(over) if over else "fits"))


if __name__ == "__main__":
    main()
