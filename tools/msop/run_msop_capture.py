"""Profile a *production* kernel launch by replaying its captured arguments.

msopprof's injection interposes ``aclrtLaunchKernel*`` / ``rtKernelLaunch`` but
not ``aclrtLaunchKernelWithArgsArray`` - the call the production launcher uses -
so kernels launched by the real pipeline are invisible to it.
``tools/msop/run_msop.py`` hand-builds one geometry per kernel, which is fine
for the 3-7 argument solve kernels and unwieldy for the 36 arguments of
``kda_pre_gram_mix``.  This driver takes the other way: it lets the real
pipeline build one call's worth of arguments (a launch spy, the trick
``/tmp/wide_e2e.py`` uses for its replay), then re-launches the captured blob
through ``msop_shim`` so the profiler sees a kernel with production pointers,
production strides and production flags.  The capture call itself goes through
``aclrtLaunchKernelWithArgsArray`` and stays invisible, so ``--kernel-name`` /
``--launch-count`` select the replay only.

The captured launch is the *last* one of its name in the call (the pipeline
launches ``kda_pre_gram_mix`` once per call), so a multi-slice configuration
still profiles one well-formed launch rather than a mixture.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 \
  KDA_MSOPP_CAPTURE=kda_pre_gram_mix \
    msprof op --application="python3 tools/msop/run_msop_capture.py" \
              --aic-metrics=PipeUtilization \
              --kernel-name=kda_pre_gram_mix \
              --launch-count=1 --output=<dir>

Any kernel in ``api._SOURCES`` can be captured by name this way, including the
mixed AIC/AIV ones whose per-pipe columns come out as two row sets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build/msop_shim"))
import kda_ascendc_v1.api as api  # noqa: E402
import msop_shim  # noqa: E402

DEV = torch.device("npu:0")
D, B, T, H = 128, 1, 8192, 96


def main() -> None:
    name = os.environ.get("KDA_MSOPP_CAPTURE", "kda_pre_gram_mix")
    iters = int(os.environ.get("KDA_MSOPP_ITERS", "3"))
    srcs = dict((n, r) for r, n in api._SOURCES)
    if name not in srcs:
        raise SystemExit("%s is not one of api._SOURCES (%s)" % (name, sorted(srcs)))

    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)

    seq = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel == name:
            seq.append((int(blocks), b"".join(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        # One real call builds the arguments (and the pipeline's RTC cache).
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
    finally:
        api._launch = orig
    if not seq:
        raise SystemExit("%s never launched in this configuration" % name)
    grid, blob = seq[-1]
    del out, st

    src = (ROOT / srcs[name]).read_text(encoding="utf-8-sig")
    handle = msop_shim.rtc_handle(api._defines() + src, name)
    stream = torch_npu.npu.current_stream().npu_stream
    for _ in range(iters):
        msop_shim.launch(handle, grid, blob, stream)
    torch.npu.synchronize()
    print("captured %s: grid %d, %d arg bytes, launched x%d"
          % (name, grid, len(blob), iters), flush=True)


if __name__ == "__main__":
    main()
