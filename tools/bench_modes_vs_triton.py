"""End-to-end benchmark of every AscendC k2_mode against the Triton-Ascend path.

Usage:
    python tools/bench_modes_vs_triton.py [B,T,H [B,T,H ...]]

Environment:
    BENCH_WARM (default 1), BENCH_REPS (default 3)
    BENCH_SHAPES (default "1,32,2;2,1024,4;2,4096,8;1,8192,32")

Each call is timed end to end (host launch + device execution) between two
device synchronizations, and every mode is checked against the Triton output.
Per-shape JSON is written to results/bench/.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch, torch_npu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'src'))
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc
from kda_bt16 import kda_bt16_fwd

DEV = torch.device('npu:0')
D = 128
WARM = int(os.environ.get('BENCH_WARM', '1'))
REPS = int(os.environ.get('BENCH_REPS', '3'))
MODES = ['separated', 'cube_separated', 'cube_full_d4', 'mix_d12_vnew', 'mix_aic_1_2',
         'persistent', 'persistent_scan', 'triton_aiv']


def sync():
    torch.npu.synchronize()


def timed(fn, warm=WARM, reps=REPS):
    for _ in range(warm):
        fn(); sync()
    vals = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync()
        vals.append((time.perf_counter() - t0) * 1e3)
    vals.sort()
    return {'samples_ms': vals, 'median_ms': vals[len(vals) // 2]}


def main():
    shapes = ([tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]] or
              [tuple(int(x) for x in s.split(',')) for s in
               os.environ.get('BENCH_SHAPES', '1,32,2;2,1024,4;2,4096,8;1,8192,32').split(';')])
    torch.npu.set_device(DEV); torch.manual_seed(1312); torch.empty(1, device=DEV); sync()
    rows = {}
    for b, t, h in shapes:
        q = (torch.randn(b, t, h, D) * .2).to(torch.bfloat16).to(DEV)
        k = (torch.randn_like(q) * .2).to(torch.bfloat16).to(DEV)
        v = (torch.randn_like(q) * .1).to(torch.bfloat16).to(DEV)
        g = torch.randn(b, t, h, D, device=DEV) * .1
        beta = torch.randn(b, t, h, device=DEV)
        alog = torch.linspace(-1, .2, h, device=DEV)
        bias = torch.randn(h, D, device=DEV) * .03
        h0 = torch.zeros(b, h, D, D, device=DEV)
        kw = dict(A_log=alog, bias=bias, lower_bound=-1., initial_state=h0,
                  output_final_state=True)
        triton = lambda: kda_bt16_fwd(
            q, k, v, g, beta, initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-1.,
            A_log=alog, dt_bias=bias.reshape(-1))
        fns = {m: (lambda m=m: kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode=m, **kw))
               for m in MODES}
        fns['triton'] = triton
        for fn in fns.values():
            fn()
        sync()
        ref_o, ref_s = triton(); sync()
        row = {'shape': [b, t, h, D]}
        for name, fn in fns.items():
            stat = timed(fn)
            o, s = fn(); sync()
            stat['out_max_abs_vs_triton'] = float((o.float() - ref_o.float()).abs().max().cpu())
            stat['state_max_abs_vs_triton'] = float((s.float() - ref_s.float()).abs().max().cpu())
            stat['finite'] = bool(torch.isfinite(o).all() and torch.isfinite(s).all())
            row[name] = stat
            print(f"[{b},{t},{h}] {name:16s} {stat['median_ms']:9.3f} ms  "
                  f"out={stat['out_max_abs_vs_triton']:.2e} "
                  f"state={stat['state_max_abs_vs_triton']:.2e}", flush=True)
        row['speedup_vs_triton'] = {m: row['triton']['median_ms'] / row[m]['median_ms']
                                    for m in MODES if row[m]['median_ms'] > 0}
        rows[f'{b}x{t}x{h}'] = row
        torch.npu.empty_cache()
        outp = ROOT / 'results/bench'
        outp.mkdir(parents=True, exist_ok=True)
        (outp / f'modes_vs_triton_{b}_{t}_{h}.json').write_text(
            json.dumps(row, indent=2) + os.linesep)
    print('\n=== median ms ===')
    print(f"{'shape':12s}" + ''.join(f"{m:>16s}" for m in MODES + ['triton']))
    for key, row in rows.items():
        print(f"{key:12s}" + ''.join(f"{row[m]['median_ms']:16.3f}" for m in MODES + ['triton']))


if __name__ == '__main__':
    main()
