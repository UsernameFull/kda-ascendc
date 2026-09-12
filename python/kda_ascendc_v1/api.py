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
SOLVE_NCHUNK = 8
SOLVE_WIDE_NCHUNK = 32
KGT_NCHUNK = 8
WU_NCHUNK = 4
D = 128
BV = 64
NV = 2
_COMPILED = False
_PERSISTENT_COMPILED = False
_PERSISTENT_SCAN_COMPILED = False
_TRITON_AIV_COMPILED = False
_LAST_PROFILE: dict[str, object] = {}
# Triangular 0/1 masks for the intra-chunk Gram kernel, built once per device.
_GRAM_MASKS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Identity tile read by the K1 solve kernel, built once per device.
_EYE_TILES: dict[torch.device, torch.Tensor] = {}
_LAUNCH_COUNTS: dict[str, int] = {}
_LAUNCH_BLOCKS: dict[str, int] = {}


def _tri_eye(device: torch.device) -> torch.Tensor:
    """Identity tile for the K1 solve: A_inv starts from it."""
    eye = _EYE_TILES.get(device)
    if eye is None:
        eye = torch.eye(CHUNK, device=device, dtype=torch.float32).contiguous()
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls, so publish the tile before any kernel reads it.
        torch.npu.synchronize()
        _EYE_TILES[device] = eye
    return eye


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
        ("kernels/v1/k1_pre_gram.cpp", "kda_pre_gram_kernel"),
        ("kernels/v1/k1_solve_wu.cpp", "kda_solve_wu_kernel"),
        ("kernels/v1/k1_solve_wu_wide.cpp", "kda_solve_wu_wide"),
        ("kernels/v1/k1_solve_wu_cube.cpp", "kda_solve_wu_cube_kernel"),
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
        ("kernels/v1/k2_persistent_loop.cpp", "kda_k2_persistent_loop"),
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

def _compile_triton_aiv() -> None:
    global _TRITON_AIV_COMPILED
    if _TRITON_AIV_COMPILED:
        return
    rtc_compile((ROOT / "kernels/v1/k2_triton_aiv.cpp").read_text(),
                "kda_k2_triton_aiv", "")
    _TRITON_AIV_COMPILED = True

def _tri_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = _GRAM_MASKS.get(device)
    if masks is None:
        idx = torch.arange(CHUNK, device=device)
        masks = ((idx[None, :] <= idx[:, None]).to(torch.float32).contiguous(),
                 (idx[None, :] < idx[:, None]).to(torch.float32).contiguous())
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls made by the launcher, so make sure the mask
        # tensors are on device before any kernel can read them.
        torch.npu.synchronize()
        _GRAM_MASKS[device] = masks
    return masks


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
    if k2_mode not in {"separated", "cube_separated", "cube_d3_separated", "cube_full_d4", "mix_aic_1_2", "mix_d12_vnew", "persistent", "persistent_scan", "persistent_loop", "persistent_scan_cube", "triton_aiv"}:
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
    # The four stage-1 inputs are no longer packed: pre_gram reads them out of
    # the public [B, T, H, D] layout with a byte-strided DataCopyPad, which
    # removes a 1.07 GB copy round trip per call (~0.6 ms at [1,8192,32]).  The
    # block/stride form of DataCopy cannot express that gather on this part.
    qk_row_bytes = h * D * 2 - D * 2
    g_row_bytes = h * D * 4 - D * 4
    beta_pack = beta.view(b, nt, CHUNK, h).permute(0, 3, 1, 2).contiguous().view(c, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    qn = torch.empty((c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    kn = torch.empty_like(qn)
    gate = torch.empty((c, CHUNK, D), dtype=torch.float32, device=q.device)
    gc = torch.empty_like(gate)
    beta_out = torch.empty((c, CHUNK), dtype=torch.float32, device=q.device)
    decay = torch.empty((c, D), dtype=torch.float32, device=q.device)
    rk = torch.empty_like(qn); rv = torch.empty_like(qn)
    qg = torch.empty_like(qn); kg = torch.empty_like(qn)
    aqk32 = torch.empty((c, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    aqk16 = torch.empty((c, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    # The wide solve kernel rounds the chunk count up to SOLVE_WIDE_NCHUNK and
    # walks whole chunk groups, so L and its two outputs are allocated with the
    # padded length: that keeps every gather and store of the last group inside
    # an allocation, and the values it computes for the padded chunks (never
    # read, and never handed out below) may be whatever uninitialised device
    # memory holds.  The debug dict hands out narrowed views so the shapes stay
    # ``[c, 16, 16]``.
    c_solve = (c + SOLVE_WIDE_NCHUNK - 1) // SOLVE_WIDE_NCHUNK * SOLVE_WIDE_NCHUNK
    L = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    mask_s, mask_l = _tri_masks(q.device)
    # Stages 1+2 run as one AIV block per chunk ("k1_pre_gram.cpp"): the fused
    # kernel consumes Qn/Kn/Gc out of UB instead of round-tripping 32 KB per
    # chunk through GM.  It addresses every token in packed ``[c, CHUNK, D]``
    # order, so the public [B, T, H, D] tensors have to be packed first, and
    # it only writes Qn/Kn/Gate/Gc when those pointers are non-null (the
    # ``return_intermediates`` debug path).
    keep = return_intermediates
    pre_args = _pack_ptrs([q, k, v, g, beta_pack,
                           A_log, bias,
                           qn if keep else None, kn if keep else None,
                           gate if keep else None, gc if keep else None,
                           beta_out, decay, rk, rv, qg, kg,
                           aqk32, aqk16, L, mask_s, mask_l])
    # One block walks `pre_unroll` consecutive chunks.  The stage is issue-bound
    # and pays a fixed per-block setup cost, so unrolling is worth 8-12% from a
    # few hundred chunks up (measured 2.679 -> 2.342 ms at [1,8192,32]); it is
    # capped at 8 chunks so at least ~256 blocks stay in flight, and tiny grids
    # (fewer than 256 chunks) stay at 1 because the loop wrapper itself costs a
    # few percent there.
    pre_unroll = 1 if c < 256 else min(8, max(2, c // 512))
    pre_args += [_i(b), _i(t), _i(h), _f(lower_bound), _f(scale), _i(pre_unroll),
                 _i(qk_row_bytes), _i(g_row_bytes)]
    mark("pre_gram_start")
    _launch("kda_pre_gram_kernel", (c + pre_unroll - 1) // pre_unroll, pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a32 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    a16 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty_like(qn)
    U = torch.empty_like(qn)
    solve_args = _pack_ptrs([L, _tri_eye(q.device), a32, a16]) + [_i(c)]
    mark("solve_start")
    # One AIV block solves SOLVE_WIDE_NCHUNK chunks with every vector
    # instruction (see the kernel header); the padded chunks are solved too but
    # land outside the first c.
    _launch("kda_solve_wu_wide", c_solve // SOLVE_WIDE_NCHUNK, solve_args, stream)
    # The Cube needs the bf16 A_inv the substitution just wrote, so the two
    # launches stay ordered on the stream.
    _launch("kda_solve_wu_cube_kernel", (c + WU_NCHUNK - 1) // WU_NCHUNK,
            _pack_ptrs([a16, rk, rv, W, U]) + [_i(c)], stream)
    finish("solve_ms", "solve_start")

    # Historical name: ``persistent_scan_cube`` has always been served by the
    # fused per-chunk Cube kernel, never by a single-kernel persistent loop.
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
        out_public = out_task.view(b, h, NV, nt, CHUNK, BV).permute(0, 3, 4, 1, 2, 5).contiguous().view(b, t, h, D)
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
        out_public = out_task.view(b, h, NV, nt, CHUNK, BV).permute(0, 3, 4, 1, 2, 5).contiguous().view(b, t, h, D)
        final_state = None if htf is None else htf.view(bh, NV, BV, D).reshape(b, h, D, D)
        if profile:
            prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
            _LAST_PROFILE = prof
        if not return_intermediates:
            return out_public, final_state
        debug = {"Qn": qn, "Kn": kn, "Gate": gate, "Gc": gc, "Beta": beta_out,
                 "Decay": decay, "Rk": rk, "Rv": rv, "Qg": qg, "Kg": kg,
                 "Aqk32": aqk32, "Aqk": aqk16, "L": L[:c], "A32": a32[:c], "A16": a16[:c],
                 "W": W, "U": U, "persistent": True}
        return out_public, final_state, debug

    if k2_mode == "persistent_loop":
        # One device-side chunk loop.  Each block owns up to MAXH=2 heads (both
        # AIV subcores run the same flag sequence and split the value dim), keeps
        # its fp32 state in UB across all chunks and never writes S32 until the
        # end, so the whole recurrence is a single MIX launch.  The block count
        # must keep the head map total: nblk * MAXH >= bh.
        # One head per block is ~4% faster than two while the heads still fit in
        # the AIC count (measured at [2,4096,8]: 4.38 vs 4.55 ms); past that,
        # two heads per block keep every block resident instead of queueing a
        # second wave (32 blocks at [1,8192,32] cost 17.2 vs 13.8 ms).
        aic_cores = 24
        want = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
        nblk = want if 0 < want <= bh else (bh if bh <= aic_cores else (bh + 1) // 2)
        nblk = max(nblk, (bh + 1) // 2)
        s32 = torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device)
        s16 = torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device)
        d1 = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.float32, device=q.device)
        d2 = torch.empty_like(d1)
        d3 = torch.empty_like(d1)
        d4f = torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
        out_task = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        vnew = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        vnew_t = torch.empty((tasks, nt, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
        h0 = None if initial_state is None else initial_state.view(bh, D, D)
        kg_t = torch.empty((c, D, CHUNK), dtype=torch.bfloat16, device=q.device)
        mark("k2_start")
        _launch("kda_kg_transpose", (c + KGT_NCHUNK - 1) // KGT_NCHUNK,
                _pack_ptrs([kg, kg_t]) + [_i(c)], stream)
        _launch("kda_k2_persistent_loop", nblk,
                _pack_ptrs([U, W, qg, aqk16, kg_t, decay, d1, d2, d3, d4f,
                            out_task, vnew, vnew_t, h0, s32, s16]) +
                [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale)], stream)
        finish("k2_ms", "k2_start")
        out_public = out_task.view(b, h, NV, nt, CHUNK, BV).permute(0, 3, 4, 1, 2, 5).contiguous().view(b, t, h, D)
        final_state = None if not output_final_state else s32.view(bh, NV, BV, D).reshape(b, h, D, D)
        if profile:
            prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
            _LAST_PROFILE = prof
        if not return_intermediates:
            return out_public, final_state
        debug = {"Qn": qn, "Kn": kn, "Gate": gate, "Gc": gc, "Beta": beta_out,
                 "Decay": decay, "Rk": rk, "Rv": rv, "Qg": qg, "Kg": kg,
                 "Aqk32": aqk32, "Aqk": aqk16, "L": L[:c], "A32": a32[:c], "A16": a16[:c],
                 "W": W, "U": U, "d1": d1, "d2": d2, "Vnew": vnew, "VnewT": vnew_t,
                 "d3": d3, "d4": d4f, "state_s32": s32, "persistent_loop": True}
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
    # Every K2 kernel of a chunk depends on the previous chunk's state update,
    # so the chain stays one launch per (chunk, stage); each batch kernel is
    # launched with nchunk=1 here.
    mark("k2_start")
    if needs_kg_t:
        mark("kg_start")
        _launch("kda_kg_transpose", (c + KGT_NCHUNK - 1) // KGT_NCHUNK,
                _pack_ptrs([kg, kg_t]) + [_i(c)], stream)
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
            _launch(d12_name, tasks, _pack_ptrs([W, qg, s16, d1, d2]) + common + [_i(1)], stream)
            _launch("kda_k2_vnew_kernel", tasks, _pack_ptrs([U, d1, vnew, vnew_t]) + common + [_i(1)], stream)
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
            _launch("kda_k2_d34_kernel", tasks, _pack_ptrs([aqk16, vnew_t, kg_t, d3, d4]) + common + [_i(1)], stream)
            _launch("kda_k2_outstate_kernel", tasks, _pack_ptrs([d2, d3, d4, s32, s16, decay, out_task]) + common + [_f(scale)], stream)

    finish("k2_ms", "k2_start")
    out_public = out_task.view(b, h, NV, nt, CHUNK, BV).permute(0, 3, 4, 1, 2, 5).contiguous().view(b, t, h, D)
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
        "Aqk32": aqk32, "Aqk": aqk16, "L": L[:c], "A32": a32[:c], "A16": a16[:c],
        "W": W, "U": U, "d1": d1, "d2": d2, "Vnew": vnew, "VnewT": vnew_t,
        "d3": d3, "d4": d4, "state_s32": s32,
        "d4_full": d4_full,
    }
    return out_public, final_state, debug
