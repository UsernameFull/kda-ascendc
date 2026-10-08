"""Bit-identity gate for the docs-11.60 post_gram rewrite (pf+).

Reconstructs the pre-rewrite kernel source by applying the landing patch pairs
in reverse to the tree's current file, compiles that as a twin, runs the whole
pipeline twice (production, and the twin swapped in for the pre_gram launch)
with ``return_intermediates=True`` and diffs every tensor.  Zero on all of
them at C=64 is the gate the other probe arms passed; this also covers C=16
and C=32 via KDA_CHUNK (KF=1 and KF=2 - the prefetch branch degenerates).

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/verify_pg_bgroup_ident.py
"""
from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
PROD = "kda_pre_gram_mix"
SRC = (ROOT / "kernels/v1/k1_pre_gram_mix.cpp").read_text(encoding="utf-8-sig")

# The pre-rewrite source comes straight from git: this file was untouched by
# the docs-11.57..11.59 work, so HEAD's copy is the comparison kernel.
import subprocess


def old_source() -> str:
    return subprocess.check_output(
        ["git", "show", "HEAD:kernels/v1/k1_pre_gram_mix.cpp"], cwd=ROOT
    ).decode("utf-8-sig")


def main() -> None:
    src = old_source()
    api.rtc_compile(api._defines() + src.replace(PROD, "kda_pg_bg_old"),
                    "kda_pg_bg_old", "")
    print("compiled the pre-rewrite twin", flush=True)

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True,
              return_intermediates=True)

    out_p, st_p, dbg_p = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()

    orig = api._launch

    def spy(kernel, blocks, args, stream):
        return orig("kda_pg_bg_old" if kernel == PROD else kernel, blocks,
                    args, stream)

    api._launch = spy
    try:
        out_o, st_o, dbg_o = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()

    worst = 0.0
    for name in sorted(set(dbg_p) | set(dbg_o)):
        a, b = dbg_p.get(name), dbg_o.get(name)
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            print("  %-12s SKIP (non-tensor)" % name)
            continue
        d = float((a.float() - b.float()).abs().max().cpu())
        worst = max(worst, d)
        print("  %-12s max|d|=%.3e %s" % (name, d, "SAME" if d == 0.0 else "**DIFF**"))
    for name, a, b in (("out", out_p, out_o), ("state", st_p, st_o)):
        d = float((a.float() - b.float()).abs().max().cpu())
        worst = max(worst, d)
        print("  %-12s max|d|=%.3e %s" % (name, d, "SAME" if d == 0.0 else "**DIFF**"))
    c = api.compile_config()
    print("C=%d: worst over all tensors %.3e -> %s"
          % (c["KDA_CHUNK"], worst,
             "BIT-IDENTICAL" if worst == 0.0 else "**NOT**"))


if __name__ == "__main__":
    main()
