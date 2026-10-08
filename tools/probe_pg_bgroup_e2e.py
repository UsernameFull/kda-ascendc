"""End-to-end pair for the docs-11.60 post_gram rewrite (pf+).

Compiles the pre-rewrite pre_gram kernel (git HEAD) as a twin and times the
whole pipeline alternately - production vs the twin swapped in for the
pre_gram launch - in one process, MIN of KDA_BG_E2E_ROUNDS, with the device
span of each pipeline call (first solve event .. last k2 event is not needed;
the span of the *whole* call via two events on the compute stream) and the
host wall.  Identity on out/state is asserted from the same runs.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_bgroup_e2e.py
"""
from __future__ import annotations

import faulthandler
import os
import subprocess
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
ROUNDS = int(os.environ.get("KDA_BG_E2E_ROUNDS", "12"))
PROD = "kda_pre_gram_mix"
TWIN = "kda_pg_bg_old"


def main() -> None:
    src = subprocess.check_output(
        ["git", "show", "HEAD:kernels/v1/k1_pre_gram_mix.cpp"], cwd=ROOT
    ).decode("utf-8-sig")
    api.rtc_compile(api._defines() + src.replace(PROD, TWIN), TWIN, "")
    print("compiled the pre-rewrite twin", flush=True)

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    def run(old: bool):
        orig = api._launch

        def spy(kernel, blocks, args, stream):
            return orig(TWIN if (old and kernel == PROD) else kernel,
                        blocks, args, stream)

        cur = torch_npu.npu.current_stream()
        api._launch = spy
        try:
            ev0 = torch_npu.npu.Event(enable_timing=True)
            ev1 = torch_npu.npu.Event(enable_timing=True)
            ev0.record(cur)
            t0 = time.perf_counter()
            out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
            ev1.record(cur)
            torch.npu.synchronize()
        finally:
            api._launch = orig
        return out, st, (time.perf_counter() - t0) * 1e3, ev0.elapsed_time(ev1)

    res = {"new": [], "old": []}
    outs = {}
    for r in range(ROUNDS):
        for arm in (("new", "old") if r % 2 == 0 else ("old", "new")):
            out, st, wall, span = run(arm == "old")
            res[arm].append((wall, span))
            outs[arm] = (out, st)
    nw = min(v[0] for v in res["new"])
    ow = min(v[0] for v in res["old"])
    ns = min(v[1] for v in res["new"])
    os_ = min(v[1] for v in res["old"])
    print("pipeline wall  (MIN of %d): new %.3f | old %.3f | delta %+.3f"
          % (ROUNDS, nw, ow, nw - ow), flush=True)
    print("pipeline span  (MIN of %d): new %.3f | old %.3f | delta %+.3f"
          % (ROUNDS, ns, os_, ns - os_), flush=True)
    print("walls new: %s" % " ".join("%.3f" % w for w, _ in res["new"]),
          flush=True)
    print("walls old: %s" % " ".join("%.3f" % w for w, _ in res["old"]),
          flush=True)
    a, b = outs["new"], outs["old"]
    print("identity new vs old: out %d, state %d (must be 0/0)"
          % (int((a[0] != b[0]).sum().item()), int((a[1] != b[1]).sum().item())),
          flush=True)


if __name__ == "__main__":
    main()
