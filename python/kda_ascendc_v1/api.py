"""S07 separated AscendC device closure for KDA v1."""
from __future__ import annotations

import struct
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch_npu

from .layout import pack_tokens

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "build/S02_clean/torch_extensions/kda_ascendc_v1_launcher"
import sys
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
from kda_ascendc_v1_launcher import launch_argsarray_engine, rtc_compile

CHUNK = 16
D = 128
BV = 64
NV = 2
_COMPILED = False
_PERSISTENT_COMPILED = False
_PERSISTENT_SCAN_COMPILED = False
_PERSISTENT_SCAN_CUBE_COMPILED = False
_TRITON_AIV_COMPILED = False
_LAST_PROFILE: dict[str, object] = {}
_LAUNCH_COUNTS: dict[str, int] = {}
_LAUNCH_BLOCKS: dict[str, int] = {}


def _pack_ptrs(xs):
    return [struct.pack("<Q", 0 if x is None else int(x.data_ptr())) for x in xs]


def _i(x: int):
    return struct.pack("<i", int(x))


def _f(x: float):
    return struct.pack("<f", float(x))


def _launch(name: str, blocks: int, args: list[bytes], stream):
    _LAUNCH_COUNTS[name] = _LAUNCH_COUNTS.get(name, 0) + 1
    _LAUNCH_BLOCKS[name] = _LAUNCH_BLOCKS.get(name, 0) + int(blocks)
    launch_argsarray_engine(name, int(blocks), stream, args, 0)


def _compile_all() -> None:
    global _COMPILED
    if _COMPILED:
        return
    sources = [
        ("kernels/v1/preprocess.cpp", "kda_preprocess_kernel"),
        ("kernels/v1/k1_gram.cpp", "kda_gram_kernel"),
        ("kernels/v1/k1_solve_wu.cpp", "kda_solve_wu_kernel"),
        ("kernels/v1/k2_init.cpp", "kda_k2_init_kernel"),
        ("kernels/v1/k2_d12.cpp", "kda_k2_d12_kernel"),
        ("kernels/v1/k2_d12_cube.cpp", "kda_k2_d12_cube_kernel"),
        ("kernels/v1/k2_vnew.cpp", "kda_k2_vnew_kernel"),
        ("kernels/v1/k2_d34.cpp", "kda_k2_d34_kernel"),
        ("kernels/v1/k2_d3_cube_bv64.cpp", "kda_k2_d3_cube_bv64"),
        ("kernels/v1/k2_d4_only.cpp", "kda_k2_d4_only_kernel"),
        ("kernels/v1/k2_kg_transpose.cpp", "kda_kg_transpose"),
        ("kernels/v1/k2_d4_full.cpp", "kda_k2_d4_full"),
        ("kernels/v1/k2_mix_d4_outstate.cpp", "kda_k2_mix_d4_outstate"),
        ("kernels/v1/k2_mix_d12_vnew.cpp", "kda_k2_mix_d12_vnew"),
        ("kernels/v1/k2_mix_all_cube.cpp", "kda_k2_mix_all_cube"),
        ("kernels/v1/k2_outstate_full.cpp", "kda_k2_outstate_full_kernel"),
        ("kernels/v1/k2_outstate.cpp", "kda_k2_outstate_kernel"),
    ]
    for rel, name in sources:
        rtc_compile((ROOT / rel).read_text(), name, "")
    _COMPILED = True


def _compile_persistent() -> None:
    global _PERSISTENT_COMPILED
    if _PERSISTENT_COMPILED:
        return
    rtc_compile((ROOT / "kernels/v1/k2_persistent.cpp").read_text(),
                "kda_k2_persistent_kernel", "")
    _PERSISTENT_COMPILED = True


def _compile_persistent_scan() -> None:
    global _PERSISTENT_SCAN_COMPILED
    if _PERSISTENT_SCAN_COMPILED:
        return
    rtc_compile((ROOT / "kernels/v1/k2_persistent_scan.cpp").read_text(),
                "kda_k2_persistent_scan_kernel", "")
    _PERSISTENT_SCAN_COMPILED = True

def _compile_persistent_scan_cube() -> None:
    global _PERSISTENT_SCAN_CUBE_COMPILED
    if _PERSISTENT_SCAN_CUBE_COMPILED:
        return
    rtc_compile((ROOT / "kernels/v1/k2_persistent_scan_cube.cpp").read_text(),
                "kda_k2_persistent_scan_cube_kernel", "")
    _PERSISTENT_SCAN_CUBE_COMPILED = True

def _compile_triton_aiv() -> None:
    global _TRITON_AIV_COMPILED
    if _TRITON_AIV_COMPILED:
        return
    rtc_compile((ROOT / "kernels/v1/k2_triton_aiv.cpp").read_text(),
                "kda_k2_triton_aiv", "")
    _TRITON_AIV_COMPILED = True

def _check_inputs(q, k, v, g, beta, A_log, bias, initial_state):
    if q.device.type != "npu":
        raise ValueError("AscendC v1 requires NPU tensors")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q/k/v must be BF16")
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape) or tuple(q.shape) != tuple(v.shape):
        raise ValueError("q/k/v must have identical [B,T,H,128] shape")
    b, t, h, d = q.shape
    if d != D or t < CHUNK or t % CHUNK:
        raise ValueError("support is T>=16, T%16==0, D=128")
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("g must be FP32 [B,T,H,128]")
    if beta.shape != (b, t, h) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B,T,H]")
    if A_log.shape != (h,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if bias is not None and (bias.shape != (h, D) or bias.dtype != torch.float32):
        raise ValueError("bias must be FP32 [H,128]")
    if initial_state is not None and (initial_state.shape != (b, h, D, D) or initial_state.dtype != torch.float32):
        raise ValueError("initial_state must be FP32 [B,H,128,128]")
    return b, t, h


def get_last_profile() -> dict[str, object]:
    profile = dict(_LAST_PROFILE)
    profile["launch_counts"] = dict(_LAUNCH_COUNTS)
    profile["launch_blocks"] = dict(_LAUNCH_BLOCKS)
    profile["launch_total"] = sum(_LAUNCH_COUNTS.values())
    return profile


def kda_bt16_fwd_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    A_log: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    return_intermediates: bool = False,
    k2_mode: str = "separated",
):
    """Run an opt-in AscendC KDA path; all KDA arithmetic stays on device."""
    global _LAST_PROFILE
    global _LAUNCH_COUNTS, _LAUNCH_BLOCKS
    _LAUNCH_COUNTS = {}
    _LAUNCH_BLOCKS = {}
    profile = os.environ.get("KDA_PROFILE", "0") == "1"
    prof: dict[str, float] = {}
    def mark(name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = time.perf_counter()
    def finish(name: str, start_name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = (time.perf_counter() - prof[start_name]) * 1e3
    b, t, h = _check_inputs(q, k, v, g, beta, A_log, bias, initial_state)
    if k2_mode not in {"separated", "cube_separated", "cube_d3_separated", "cube_full_d4", "mix_aic_1_2", "mix_d12_vnew", "persistent", "persistent_scan", "persistent_scan_cube", "triton_aiv"}:
        raise ValueError("unsupported k2_mode")
    if scale is None:
        scale = D ** -0.5
    nt = t // CHUNK
    bh = b * h
    c = bh * nt
    tasks = bh * NV
    q, k, v, g, beta, A_log = [x.contiguous() for x in (q, k, v, g, beta, A_log)]
    if bias is not None:
        bias = bias.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    q_pack, k_pack, v_pack, g_pack = [pack_tokens(x) for x in (q, k, v, g)]
    beta_pack = beta.view(b, nt, CHUNK, h).permute(0, 3, 1, 2).contiguous().view(c, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    qn = torch.empty_like(q_pack); kn = torch.empty_like(k_pack)
    gate = torch.empty((c, CHUNK, D), dtype=torch.float32, device=q.device)
    gc = torch.empty_like(gate)
    beta_out = torch.empty((c, CHUNK), dtype=torch.float32, device=q.device)
    decay = torch.empty((c, D), dtype=torch.float32, device=q.device)
    rk = torch.empty_like(q_pack); rv = torch.empty_like(q_pack)
    qg = torch.empty_like(q_pack); kg = torch.empty_like(q_pack)
    # The preprocess kernel addresses every token in packed ``[c, CHUNK, D]``
    # order, so the public [B, T, H, D] tensors have to be packed first.
    pre_args = _pack_ptrs([q_pack, k_pack, v_pack, g_pack, beta_pack,
                           A_log, bias, qn, kn, gate, gc, beta_out, decay, rk, rv, qg, kg])
    pre_args += [_i(b), _i(t), _i(h), _f(lower_bound)]
    mark("pre_start")
    _launch("kda_preprocess_kernel", c, pre_args, stream)
    finish("preprocess_ms", "pre_start")

    aqk32 = torch.empty((c, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    aqk16 = torch.empty((c, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    L = torch.empty_like(aqk32)
    gram_args = _pack_ptrs([qn, kn, gc, beta_out, aqk32, aqk16, L]) + [_i(c), _f(scale)]
    mark("gram_start")
    _launch("kda_gram_kernel", c, gram_args, stream)
    finish("gram_ms", "gram_start")

    a32 = torch.empty_like(aqk32)
    a16 = torch.empty_like(aqk16)
    W = torch.empty_like(q_pack)
    U = torch.empty_like(q_pack)
    solve_args = _pack_ptrs([L, rk, rv, a32, a16, W, U]) + [_i(c)]
    mark("solve_start")
    _launch("kda_solve_wu_kernel", c, solve_args, stream)
    finish("solve_ms", "solve_start")

    if k2_mode == "persistent_scan_cube":
        k2_mode = "mix_all_cube"

    if k2_mode == "triton_aiv":
        _compile_triton_aiv()
        out_task = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        h0 = None if initial_state is None else initial_state.view(bh, D, D)
        htf = torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device) if output_final_state else None
        pargs = _pack_ptrs([W, qg, U, aqk16, kg, decay, h0, out_task, htf])
        pargs += [_i(bh), _i(nt), _i(NV), _f(scale)]
        mark("k2_start")
        _launch("kda_k2_triton_aiv", tasks, pargs, stream)
        finish("k2_ms", "k2_start")
        out_public = out_task.view(bh, NV, nt, CHUNK, BV).permute(0, 2, 3, 1, 4).contiguous().view(b, t, h, D)
        final_state = None if htf is None else htf.view(bh, NV, BV, D).reshape(b, h, D, D)
        if profile:
            prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
            _LAST_PROFILE = prof
        return out_public, final_state

    if k2_mode in {"persistent", "persistent_scan"}:
        if k2_mode == "persistent_scan":
            _compile_persistent_scan()
        else:
            _compile_persistent()
        out_task = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        h0 = None if initial_state is None else initial_state.view(bh, D, D)
        htf = torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device) if output_final_state else None
        pargs = _pack_ptrs([W, qg, U, aqk16, kg, decay, h0, out_task, htf])
        pargs += [_i(bh), _i(nt), _i(NV), _f(scale)]
        mark("k2_start")
        _launch("kda_k2_persistent_scan_kernel" if k2_mode == "persistent_scan" else "kda_k2_persistent_kernel", tasks, pargs, stream)
        finish("k2_ms", "k2_start")
        out_public = out_task.view(bh, NV, nt, CHUNK, BV).permute(0, 2, 3, 1, 4).contiguous().view(b, t, h, D)
        final_state = None if htf is None else htf.view(bh, NV, BV, D).reshape(b, h, D, D)
        if profile:
            prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
            _LAST_PROFILE = prof
        if not return_intermediates:
            return out_public, final_state
        debug = {"Qn": qn, "Kn": kn, "Gate": gate, "Gc": gc, "Beta": beta_out,
                 "Decay": decay, "Rk": rk, "Rv": rv, "Qg": qg, "Kg": kg,
                 "Aqk32": aqk32, "Aqk": aqk16, "L": L, "A32": a32, "A16": a16,
                 "W": W, "U": U, "persistent": True}
        return out_public, final_state, debug

    s32 = torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device)
    s16 = torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device)
    h0 = None if initial_state is None else initial_state.view(bh, D, D)
    init_args = _pack_ptrs([h0, s32, s16]) + [_i(bh), _i(NV)]
    mark("init_start")
    _launch("kda_k2_init_kernel", tasks, init_args, stream)
    finish("init_ms", "init_start")

    # Modes that route d3/d4 through the Cube kernels need the K-major kg^T
    # staging buffer produced by kda_kg_transpose.
    cube_d4_modes = {"cube_full_d4", "mix_aic_1_2", "mix_d12_vnew", "mix_all_cube"}
    separated_modes = {"separated", "cube_separated", "cube_d3_separated"}
    needs_kg_t = k2_mode in cube_d4_modes or k2_mode in separated_modes
    d1 = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.float32, device=q.device)
    d2 = torch.empty_like(d1)
    vnew = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
    vnew_t = torch.empty((tasks, nt, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
    d3 = torch.empty_like(d1)
    d4 = None if k2_mode in cube_d4_modes else torch.empty((c, D, D), dtype=torch.float32, device=q.device)
    d4_full = (torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
                if k2_mode in {"mix_aic_1_2", "mix_d12_vnew", "mix_all_cube"} else
                torch.empty((c, D, D), dtype=torch.float32, device=q.device)
                if k2_mode == "cube_full_d4" else None)
    kg_t = torch.empty((c, D, CHUNK), dtype=torch.bfloat16, device=q.device) if needs_kg_t else None
    out_task = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
    mark("k2_start")
    if needs_kg_t:
        mark("kg_start")
        _launch("kda_kg_transpose", c, _pack_ptrs([kg, kg_t]) + [_i(c)], stream)
        finish("kg_transpose_ms", "kg_start")
    for chunk in range(nt):
        common = [_i(bh), _i(nt), _i(NV), _i(chunk)]
        if k2_mode == "mix_all_cube":
            _launch("kda_k2_mix_all_cube", bh,
                    _pack_ptrs([U, W, qg, s16, aqk16, kg_t, decay,
                                d1, d2, d3, d4_full, s32, out_task,
                                vnew, vnew_t]) +
                    common + [_i(1), _f(scale)], stream)
            continue
        if k2_mode == "mix_d12_vnew":
            _launch("kda_k2_mix_d12_vnew", bh,
                    _pack_ptrs([U, W, qg, s16, d1, d2, vnew, vnew_t]) + common,
                    stream)
        else:
            d12_name = (
                "kda_k2_d12_cube_kernel"
                if k2_mode in {"cube_separated", "cube_d3_separated", "cube_full_d4", "mix_aic_1_2"}
                else "kda_k2_d12_kernel"
            )
            _launch(d12_name, tasks, _pack_ptrs([W, qg, s16, d1, d2]) + common, stream)
            _launch("kda_k2_vnew_kernel", tasks, _pack_ptrs([U, d1, vnew, vnew_t]) + common, stream)
        if k2_mode in {"mix_aic_1_2", "mix_d12_vnew"}:
            _launch("kda_k2_mix_d4_outstate", bh,
                    _pack_ptrs([aqk16, vnew_t, kg_t, d2, d3, d4_full,
                                s32, s16, decay, out_task]) +
                    common + [_i(1), _f(scale)], stream)
        elif k2_mode == "cube_full_d4":
            _launch("kda_k2_d3_cube_bv64", tasks, _pack_ptrs([aqk16, vnew_t, d3]) + common, stream)
            _launch("kda_k2_d4_full", bh, _pack_ptrs([vnew_t, kg_t, d4_full]) + common, stream)
            _launch("kda_k2_outstate_full_kernel", tasks, _pack_ptrs([d2, d3, d4_full, s32, s16, decay, out_task]) + common + [_f(scale)], stream)
        elif k2_mode == "cube_d3_separated":
            _launch("kda_k2_d3_cube_bv64", tasks, _pack_ptrs([aqk16, vnew_t, d3]) + common, stream)
            _launch("kda_k2_d4_only_kernel", tasks, _pack_ptrs([vnew_t, kg_t, d4]) + common, stream)
            _launch("kda_k2_outstate_kernel", tasks, _pack_ptrs([d2, d3, d4, s32, s16, decay, out_task]) + common + [_f(scale)], stream)
        else:
            _launch("kda_k2_d34_kernel", tasks, _pack_ptrs([aqk16, vnew_t, kg_t, d3, d4]) + common, stream)
            _launch("kda_k2_outstate_kernel", tasks, _pack_ptrs([d2, d3, d4, s32, s16, decay, out_task]) + common + [_f(scale)], stream)

    finish("k2_ms", "k2_start")
    out_public = out_task.view(bh, NV, nt, CHUNK, BV).permute(0, 2, 3, 1, 4).contiguous().view(b, t, h, D)
    final_state = None
    if profile:
        prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
        _LAST_PROFILE = prof
    if output_final_state:
        final_state = s32.view(bh, NV, BV, D).reshape(b, h, D, D)
    if not return_intermediates:
        return out_public, final_state
    debug = {
        "Qn": qn, "Kn": kn, "Gate": gate, "Gc": gc, "Beta": beta_out,
        "Decay": decay, "Rk": rk, "Rv": rv, "Qg": qg, "Kg": kg,
        "Aqk32": aqk32, "Aqk": aqk16, "L": L, "A32": a32, "A16": a16,
        "W": W, "U": U, "d1": d1, "d2": d2, "Vnew": vnew, "VnewT": vnew_t,
        "d3": d3, "d4": d4, "state_s32": s32,
        "d4_full": d4_full,
    }
    return out_public, final_state, debug
