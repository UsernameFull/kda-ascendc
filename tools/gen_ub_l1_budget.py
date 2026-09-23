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
    conditional = {"vnew"}
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
    conditional = {"vnew"}
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


def main():
    chunk = int(os.environ.get("KDA_CHUNK", "64"))
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


if __name__ == "__main__":
    main()
