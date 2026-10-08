"""K2's block count on a 20-core part: 24 blocks is not one wave any more.

api.py's K2 block heuristic was tuned on the 24-core golden device ("4
heads/block puts the whole 96-head grid on 24 blocks, i.e. one wave per AIC",
where 24 blocks of 4 heads beat 48 blocks of 2: 5.97 vs 6.23 ms).  This part
has 20 AIC cores (torch_npu cube_core_num), so those 24 blocks are 2 waves of
4-head blocks with 16 cores idle through the second: 8 head-steps where 32
blocks of 3 need 2 x 3 = 6 and 48 blocks of 2 need 3 x 2 = 6.  The kernel's
head map strides by the *runtime* block count ("for (h = blk; h < BH &&
nh < MAXH; h += NBLK)", k2_persistent_loop.cpp), so the shape is one nblk
away, no recompile - KDA_PERSIST_LOOP_BLOCKS pins it per call.

Arms (interleaved, one process, device events on the k2 launch, MIN over
rounds; outputs and fp32 state checked bitwise against the default):

  default   -> the api picker, which should now land on 32 blocks / 3 heads
  24 blocks -> the shipped shape (4 heads, 2 waves)
  48 blocks -> 2 heads, 3 waves
  96 blocks -> 1 head (the map needs the define cap >= 1: yes)

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_k2_maxh_wave.py
"""
from __future__ import annotations

import faulthandler
import os
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
K2 = "kda_k2_persistent_loop"
ROUNDS = int(os.environ.get("KDA_K2W_ROUNDS", "4"))
ARMS = [(0, "default (picker)"), (24, "24 blocks x4 heads"), (32, "32 blocks x3 heads"),
        (48, "48 blocks x2 heads"), (96, "96 blocks x1 head")]


def run_call(q, k, v, g, beta, kw):
    """One full call; device-event spans of the K2 (and pre_gram) launches."""
    cur = torch.npu.current_stream()
    rec = {}
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel in (K2, "kda_pre_gram_mix"):
            ev0 = torch.npu.Event(enable_timing=True)
            ev1 = torch.npu.Event(enable_timing=True)
            ev0.record(cur)
            orig(kernel, blocks, args, stream)
            ev1.record(cur)
            rec[kernel] = (int(blocks), ev0, ev1)
            return None
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    return rec, out, st


def main() -> None:
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    os.environ.pop("KDA_PERSIST_LOOP_BLOCKS", None)
    run_call(q, k, v, g, beta, kw)          # warm + compile

    span, nblk, wall = {}, {}, {}
    ref = None
    for r in range(ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for want, label in order:
            if want:
                os.environ["KDA_PERSIST_LOOP_BLOCKS"] = str(want)
            else:
                os.environ.pop("KDA_PERSIST_LOOP_BLOCKS", None)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            rec, out, st = run_call(q, k, v, g, beta, kw)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            wall[label] = dt if label not in wall else min(wall[label], dt)
            gr, ev0, ev1 = rec[K2]
            s = float(ev0.elapsed_time(ev1))
            span[label] = s if label not in span else min(span[label], s)
            nblk[label] = gr
            if label == ARMS[0][1] and ref is None:
                ref = (out, st)
            elif label == ARMS[0][1]:
                pass
            else:
                d_out = int((out != ref[0]).sum().item())
                d_st = int((st != ref[1]).sum().item())
                assert d_out == 0 and d_st == 0, (label, d_out, d_st)
    for _, label in ARMS:
        print("  %-18s nblk %3d  k2 span MIN %8.3f ms  e2e wall MIN %8.3f ms"
              % (label, nblk[label], span[label], wall[label]), flush=True)
    print("all arms bit-identical on out/state against the default", flush=True)


if __name__ == "__main__":
    main()
