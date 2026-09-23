"""Route 2's floor: what does the *AIV half* of a wide-RHS solve cost?

The 2026-09-23 order asks for a C64 two-block prototype of

    (I + L) X = [Rk, Rv],   X_i = T_ii^-1 (R_i - sum_{j<i} L_ij X_j)

and fixes the decision rule: if the complete fused solve - layout, W/U store
and consumption included - is not under ~2.5 ms at [1,8192,96,128] (the
shipped two-level solve's 2.53 ms), the route stops.  This probe measures the
part of that price that no layout or pipeline choice can remove: the row
recursion itself, in the two shapes it can take.

  * the shipped shape (k1_solve_wu_wide.cpp): recurse on the M x M inverse,
    M = 32 lanes per instruction, 8 chunk-instances sideways in the repeat
    axis - 124 MulAddDst per chunk;
  * the fused shape: recurse on the [M, 256] RHS instead.  Same instruction
    count per instance, but the tile now holds 256 lanes per chunk, so the
    same UB holds four times fewer instances: 496 instructions per chunk for
    the two-block split, 240 for a leaf-16 split, both at 8 repeats.

Both run in one process, on 12288 chunk-instances (the production chunk
count), with no gathers, no casts and no stores but one 32 B epilogue - so each
number is a lower bound on that shape's AIV cost.  The script also replays the
captured production solve launches in the same process, so the shipped stage's
AIV/AIC split is measured on the same device in the same session rather than
quoted from the docs.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_rhs_substitution.py
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
B, T, H = 1, 8192, 96
CHUNKS = 12288          # (T / CHUNK) x H at CHUNK = 64
GRID = 1536             # the production wide-solve grid (CHUNKS / 8 instances)
MODES = [(0, "null          ", 2, 0), (1, "shipped shape ", 2, (32 - 1) * 256),
         (2, "fused 2x32    ", 4, (32 - 1) * 512), (3, "fused 4x16    ", 4, (16 - 1) * 512)]


def main() -> None:
    api._rtc("kernels/v1/k1_solve_rhs_probe.cpp", "kda_solve_rhs_probe")
    out = torch.zeros(GRID * 8, dtype=torch.float32, device=DEV)
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream

    def launch(mode: int, groups: int, store_off: int) -> None:
        api._launch("kda_solve_rhs_probe", GRID,
                    api._pack_ptrs([out]) + [api._i(groups), api._i(mode),
                                             api._i(store_off)], cur_h)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("recursion floors at CHUNK=%d, %d chunk-instances, grid %d"
          % (api.CHUNK, CHUNKS, GRID), flush=True)
    for mode, name, groups, store_off in MODES:
        launch(mode, groups, store_off)      # warm
    torch.npu.synchronize()
    floor = {}
    print()
    print("  arm            groups  instructions/chunk  repeats  ms (MIN of 5)  ns/chunk")
    for mode, name, groups, store_off in MODES:
        ms = timeit(lambda m=mode, g=groups, s=store_off: launch(m, g, s))
        floor[mode] = ms
        instr = {0: 0, 1: 124, 2: 496, 3: 240}[mode]
        print("  %s %5d   %8d            %5d   %8.3f        %8.1f"
              % (name, groups, instr, 8, ms, ms * 1e6 / CHUNKS), flush=True)
    del floor[0]

    # ---- the shipped solve, replayed from a captured production launch -------
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
    bias = torch.randn(H, D, device=DEV) * 0.03
    seq = []
    orig = api._launch

    def spy(name, blocks, args, stream):
        seq.append((name, int(blocks), list(args)))
        return orig(name, blocks, args, stream)

    api._launch = spy
    print()
    print("warming the production pipeline (RTC) ...", flush=True)
    api.kda_bt16_fwd_ascendc(q, k, v, g, beta, A_log=a_log, bias=bias,
                             lower_bound=-1.0, output_final_state=True)
    torch.npu.synchronize()
    api._launch = orig

    def play(names, stream):
        for nm, blk, ar in seq:
            if nm in names:
                api.launch_argsarray_engine(nm, blk, stream.npu_stream, ar, 0)

    sa = torch_npu.npu.Stream(device=DEV)
    s = torch_npu.npu.Stream(device=DEV)

    def one(names):
        return timeit(lambda: (s.wait_stream(cur), play(names, s), cur.wait_stream(s)), 5)

    aiv = one({"kda_solve_wu_wide"})
    aic = one({"kda_solve_assemble", "kda_solve_wu_cube_kernel"})
    both = timeit(lambda: (sa.wait_stream(cur), s.wait_stream(cur),
                           play({"kda_solve_wu_wide"}, sa),
                           play({"kda_solve_assemble", "kda_solve_wu_cube_kernel"}, s),
                           cur.wait_stream(sa), cur.wait_stream(s)), 5)
    print("shipped solve, same process (MIN of 5): AIV %.3f  AIC %.3f  overlapped %.3f ms"
          % (aiv, aic, both))
    print()
    print("verdict arithmetic")
    print("  shipped solve stage (this session)  %8.3f ms" % both)
    print("  shipped AIV half                    %8.3f ms" % aiv)
    print("  shipped AIC half (assemble+cube)    %8.3f ms" % aic)
    print("  probe: shipped recursion floor      %8.3f ms   (rest of the AIV half: "
          "gathers, casts, A16/Xb/Lneg stores)" % floor[1])
    print("  probe: fused 2x32 recursion floor   %8.3f ms   (%.2fx the shipped AIV half)"
          % (floor[2], floor[2] / aiv))
    print("  probe: fused 4x16 recursion floor   %8.3f ms   (%.2fx the shipped AIV half)"
          % (floor[3], floor[3] / aiv))
    print("  budget for the whole fused solve    %8.3f ms   (the shipped stage)" % both)
    for m, label in ((2, "2x32"), (3, "4x16")):
        extra = 0.0   # coupling cube + RHS gathers + W/U stores are all still to come
        print("  fused %s: recursion floor is %.2fx the whole stage, and still needs "
              "RHS gathers, W/U stores and (for 2x32) the coupling Cube step"
              % (label, floor[m] / both))


if __name__ == "__main__":
    main()
