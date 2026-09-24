"""S07 separated AscendC device closure for KDA v1."""
from __future__ import annotations

import math
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

# Chunk size of the whole pipeline.  It has to match the kernels' KDA_CHUNK
# (see _defines below): the AIV/Cube block sizes, the triangular masks and the
# identity tile all follow it.  16 is the default because it is the widest
# supported shape (T % CHUNK == 0, so C=16 accepts four times the sequence
# lengths of C=64); 64 is what every R3 number and the production baseline are
# measured at - there the per-chunk setup, the gate cumsum and the K2 chunk
# hand-offs are amortised over four times the rows, and the two-level solve
# (SOLVE_WIDE_SUBB = 2) only exists at C >= 64.  Long sequences should build
# with KDA_CHUNK=64.
CHUNK = int(os.environ.get("KDA_CHUNK", "16"))
# Chunk sizes the pipeline is actually checked at.  C=16, C=32 and C=64 all
# pass the whole matrix of tests/test_chunk_shape_matrix.py (B = 1/2,
# H = 2/32/48/96, short and long T, with and without an initial state, plus
# determinism); anything else is refused before the RTC compile - see
# _kda_fwd_impl's guard.
SUPPORTED_CHUNKS = frozenset({16, 32, 64})
SOLVE_NCHUNK = 8
# The wide solve keeps four live [NC, CHUNK, CHUNK] tiles plus the Brcb
# expansion, so the 192 KB of UB cap NC at 32/16/4 chunks for CHUNK =
# 16/32/64 (the 16 and 4 settings are 184 KB: NC x [CHUNK, CHUNK] fp32 twice
# over, its bf16 rounding copy and the Brcb expansion).  The sweep at
# [1,8192,96,128] measured the solve stage at 2.70 (NC 8) -> 2.43 (NC 16) ms
# for CHUNK = 32.  The solve-Cube's L0C holds one slot per (pass, chunk) unit
# (2 * NC * CHUNK * 128 fp32), which is why NC drops to 4 at CHUNK = 32 (the
# whole 128 KB of L0C, 2.56 ms) and to 2 at CHUNK = 64 (also 128 KB).
SOLVE_WIDE_NCHUNK = (int(os.environ.get("KDA_SOLVE_WIDE_NCHUNK", "0"))
                     or (32 if CHUNK <= 16 else (16 if CHUNK <= 32 else 8)))
# Two-level solve (R1): the wide kernel runs the row recursion on M = CHUNK/SB
# sub-blocks and the assemble kernel forms the coupling block on the Cube, so
# the substitution's M^3/2 vector lanes per chunk drop with SB^2 (64 -> 16 -> 4
# kLane for SB = 1/2/4).  SB = 2 needs M >= 16 for the 16-row fractal copies
# of the assemble kernel, i.e. CHUNK >= 32; the 64-wide chunk is the one the
# e2e spends its time in, so it is the one that gets the two-level path.
SOLVE_WIDE_SUBB = (int(os.environ.get("KDA_SOLVE_WIDE_SUBB", "0"))
                   or (2 if CHUNK >= 64 else 1))
# Chunks one wide block solves (the wide kernel's tile runs over sub-blocks x
# chunks, see its header): SB sub-blocks of every chunk share one tile.
SOLVE_WIDE_NCH = SOLVE_WIDE_NCHUNK // SOLVE_WIDE_SUBB
# Chunks one assemble block forms the coupling block of.  The shipped queue
# form (KDA_ASM_LOADS=0) issues the L1 loads of a whole pass before its
# arithmetic, so its queue has to be exactly this deep; the explicit-buffer
# forms (1, 2 - production) have no such constraint, and 6/8/12/16 were
# re-tested on them (tools/probe_solve_assemble_nc.py): they run and are
# bit-identical, but e2e is flat inside the noise floor (NC = 4/6/8 ->
# 10.389/10.390/10.392 ms), so 4 stays - the knob is free, not profitable.
ASM_NCHUNK = int(os.environ.get("KDA_ASM_NCHUNK", "0")) or 4
# The wide part is AIV-only and the assemble/Cube part AIC-only, so the two
# can run at the same time: the chunk range is cut into this many slices, the
# wide slices go on one stream and the assemble+Cube slices on another behind
# a per-slice event.  Measured at [1,8192,96,128]/CHUNK=64: 3.22 -> 2.51 ms
# for the whole solve, bit-identical outputs.  0 keeps a single stream.
# R3: 16 -> 24 slices.  Two same-process interleaved sweeps (MIN of 3) put
# solve_ms at 2.650 (16) / 2.525 (24) / 2.742 (8) / 3.463 (48): the AIV and
# AIC halves overlap better with smaller slices, but only down to the point
# where the per-slice launch pair (and the AIC side's tail) starts to show.
SOLVE_OVERLAP = int(os.environ.get("KDA_SOLVE_OVERLAP", "24"))
# Heads per block in the persistent K2 loop, and therefore the KDA_MAXH the
# kernel is compiled with (they must agree: the kernel's head map only walks
# MAXH heads per block, so a smaller define silently drops heads).
# The fp32 state is 32 KB of UB per head and stays resident, so all the
# staging has to fit in the rest of this part's 192 KB.  R3 halved that
# staging by aliasing it across the two phases the loop alternates (stage 2
# and stage 4, each of which opens with a PIPE_ALL drain): 112.5 KB -> 56.5 KB
# at C=64, 46.5 -> 26.5 KB at C=16 (kernels/v1/k2_persistent_loop.cpp carries
# the budget).  That is what makes 4 heads fit at every chunk size -
# 128 + 56.5 = 184.5 KB of 192 at C=64, 128 + 36.5 = 164.5 at C=32 - where
# the old layout capped C=64/32 at 2 (176.5 KB).  A UB overrun is a
# kernel-side aivec error, not a host-side allocation failure, so this is
# arithmetic, not a probe.
# Heads per block is not a free knob: 4 heads/block puts the whole 96-head
# grid on 24 blocks, i.e. one wave per AIC instead of two, and the second wave
# costs a whole second pass over the chunks.  Measured same-process A/B at
# [1,8192,96,128] (interleaved MIN of 4, bit-identical outputs incl. the fp32
# state): C=64 k2 6.23 (MAXH 2, 2 waves) -> 5.97 ms (MAXH 4, 1 wave); C=32
# 6.12 -> 5.81 ms.  The aliasing is *not* free at a fixed head count - it
# needs the stage-4 recurrence split into four 16-row quarters instead of two
# 32-row halves, worth +0.54 ms at C=64/MAXH 2 - so the MAXH 4 gain is what
# pays for it, and the pairing is the point (2 + 56.5 = 120.5 KB wastes the
# diet).
# nh = 1 is not just "one fewer head": the depth-one flag protocol only
# overlaps the two engines when a block has two or more heads in flight, so at
# C=64 the second head is worth k2 6.04 -> 3.91 ms and e2e 12.79 -> 10.74 ms
# at [1,8192,96,128] (MIN of 4), numerics unchanged to the bit (K2 stage gate
# 6.6e-36/9.6e-04/3.3e-08).  The env can lower MAXH, or raise it within the
# ceiling; C=16/32/64 are each checked end to end against the fp32 reference.
PERSIST_MAXH = max(1, min(4 if CHUNK <= 64 else 2,
                          int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0"))
                          or (4 if CHUNK <= 64 else 2)))
KGT_NCHUNK = 8
WU_NCHUNK = int(os.environ.get("KDA_WU_NCHUNK", "0")) or (4 if CHUNK <= 32 else 2)
D = 128
BV = 64
NV = 2
# The K2 recurrence has exactly one implementation that is part of the API:
# the device-side chunk loop walks KDA_CHUNK rows at a time, so it is the only
# one that follows the build (kernels/v1/k2_persistent_loop.cpp).  Every other
# mode in this file is a checkpoint of the S12-S15 experiments and carries the
# 16-row tile as a literal (``constexpr int32_t M = 16`` in k2_d12.cpp,
# k2_vnew.cpp, k2_d34.cpp, k2_outstate*.cpp, the k2_mix_* trio and the two
# persistent_scan kernels): they *are* the C=16 implementation, so at any
# other chunk size they would read 16 rows of a CHUNK-row chunk and return a
# plausible-looking wrong answer - which is how a C=64 build once got a
# "1.45 ms" number that was really the C=16 kernel.  They stay reachable for
# benchmarking through kda_ascendc_v1.experimental, and the C=16 check is
# enforced centrally in _kda_fwd_impl so no caller can bypass it.
PERSISTENT_LOOP = "persistent_loop"
# Route 1 of the 2026-09-22 redesign (docs/PREFILL_LIFECYCLE_REFACTOR_20260922.md
# section 4.1): the serial state chain (kda_k2_state_loop) and the output
# (kda_k2_out_parallel) as two kernels.  Only Z and the state gate the next
# chunk, so the output of every chunk is computable in parallel from the
# chunk-entry state the state kernel publishes; the pair is a candidate only
# if its *total* beats the fused loop (tools/probe_state_out_split.py).  It is
# not in C16_ONLY_K2_MODES: both kernels are KDA_CHUNK-generic, exactly like
# the loop they were split out of.
SPLIT_STATE_OUT = "split_state_out"
C16_ONLY_K2_MODES = frozenset({
    "separated", "cube_separated", "cube_d3_separated", "cube_full_d4",
    "mix_aic_1_2", "mix_d12_vnew", "persistent", "persistent_scan",
    "persistent_scan_cube", "triton_aiv",
})
K2_MODES = frozenset(C16_ONLY_K2_MODES | {PERSISTENT_LOOP, SPLIT_STATE_OUT})
_COMPILED = False
_PERSISTENT_COMPILED = False
_PERSISTENT_SCAN_COMPILED = False
_TRITON_AIV_COMPILED = False
_LAST_PROFILE: dict[str, object] = {}
# Triangular 0/1 masks for the intra-chunk Gram kernel, built once per device.
_GRAM_MASKS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Identity tile read by the K1 solve kernel, built once per (device, width).
_EYE_TILES: dict[tuple, torch.Tensor] = {}
_LAUNCH_COUNTS: dict[str, int] = {}
_LAUNCH_BLOCKS: dict[str, int] = {}


def _tri_eye(device: torch.device, n: int | None = None) -> torch.Tensor:
    """Identity tile for the K1 solve: A_inv starts from it.

    The wide kernel reads it as one dense n x n block (``DataCopyParams(n, n/8,
    0, 0)``), so the two-level path wants the M x M sub-block identity, not the
    CHUNK x CHUNK one: passing the chunk-sized tile there reads the first
    M*M floats of a CHUNK-wide identity packed M-per-row, which is a different
    matrix and puts the solve off by O(1) (measured 1.178e+00 against the
    fp64 inverse, vs 4.882e-04 with the right tile).
    """
    n = CHUNK if n is None else n
    key = (device, n)
    eye = _EYE_TILES.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=torch.float32).contiguous()
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls, so publish the tile before any kernel reads it.
        torch.npu.synchronize()
        _EYE_TILES[key] = eye
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


# The two-level solve runs its AIV half and its AIC half on their own streams
# (see _launch_solve_two_level); one pair per device, created once.
_SOLVE_STREAMS: dict = {}


def _solve_streams(device: torch.device):
    pair = _SOLVE_STREAMS.get(device)
    if pair is None:
        pair = (torch_npu.npu.Stream(device=device),
                torch_npu.npu.Stream(device=device))
        _SOLVE_STREAMS[device] = pair
    return pair


def a16_mode() -> int:
    """A16-store ablation mode for the wide solve (KDA_SOLVE_A16_MODE).

    0 is production: the wide kernel writes both the diagonal sub-blocks of the
    parent A_inv tile and its strict-upper blank.  1 drops the blank, 2 drops
    the whole parent tile - the Cube solve then reads a partially written
    operand, which is the point of the ablation (tools/probe_solve_a16_ablation.py
    prices the store) and never a production setting.  Read per call rather than
    frozen at import so a probe can flip it between two arms of one process; the
    kernel is unchanged either way, this is a runtime argument.
    """
    return int(os.environ.get("KDA_SOLVE_A16_MODE", "0"))


def asm_load_mode() -> int:
    """Load and intermediate shape for the coupling block (kda_solve_assemble).

    0 is the shipped form: one B1 queue per operand, six ND2NZ calls per chunk,
    and P through its own GM tile.  1 batches the same bytes - one call per
    block for the A operand, a chunk's two B bands merged, and pass 1's whole B
    operand in one call.  2 is 1 plus P on chip: pass 0's fixpipe writes P into
    L1 in NZ instead of GM and pass 1 builds L0B from it, which drops the tile's
    50.3 MB round trip (151.0 -> 100.7 MB per call of L1 traffic) and, since
    nothing touches the GM tile any more, the caller stops allocating its
    25.2 MB as well (section 11.40).  2 is production: measured in
    tools/probe_solve_assemble_loads.py at [1,8192,96,128], the three arms come
    out 0.902 / 0.692 / 0.512 ms isolated, AIC 2.614 / 2.305 / 2.197 and e2e
    10.624 / 10.445 / 10.408 ms.  All three land byte-identical
    operands (tools/probe_solve_assemble_coalesce.py and
    tools/probe_solve_assemble_l1p.py check P and A16 are equal before timing
    anything), so this is a scheduling knob, not a numeric one.  Read per call
    rather than frozen at import so a probe can flip it between arms of one
    process.
    """
    return int(os.environ.get("KDA_ASM_LOADS", "2"))


def cube_a16_resident() -> int:
    """A16 residency for the solve's Cube kernel (kda_solve_wu_cube_kernel).

    0 re-reads the block's A16 tile at the top of each of the two passes - the
    second read is an L2 hit (docs section 11.35 priced it at 0.070 ms).  1
    keeps the block's NC tiles in L1 for both passes, which is the same bytes
    of L1 as the queue it replaces.  1 is production: measured in
    tools/probe_solve_cube_a16_resident.py at [1,8192,96,128], the cube half
    goes 1.577 -> 1.511 ms and the stage 2.346 -> 2.327, with the pipeline
    bit-identical (a first attempt at this mode spun the block - its slots were
    still AllocTensor'd, which the kernel header records).  Read per call so a
    probe can flip it between two arms of one process; the arithmetic is
    identical either way.
    """
    return int(os.environ.get("KDA_CUBE_A16_RESIDENT", "1"))


def _launch_solve_two_level(c_solve, c, nch, asm_nchunk, wu_nchunk, overlap, L, eye,
                            a32, a16, xb, lneg, pmid, rk, rv, W, U, stream,
                            debug_stores) -> None:
    """R1 two-level solve: wide kernel (AIV), then assemble + Cube (AIC).

    The substitution runs on M = CHUNK/SB sub-blocks (k1_solve_wu_wide.cpp) and
    the coupling block X21 = -X22 L21 X11 is left to the Cube
    (k1_solve_assemble.cpp), which cuts the row recursion's M^3/2 vector lanes
    per chunk by SB^2 and leaves the Cube the part it is actually good at.

    The wide half occupies the vector cores and the assemble/Cube half the Cube
    cores, so the chunk range is cut into ``overlap`` slices and the two halves
    run on their own streams behind a per-slice event - at [1,8192,96,128] and
    CHUNK = 64 that takes the stage from 3.22 to 2.51 ms with bit-identical
    outputs.  Slices hold whole ``unit`` (wide block / assemble / Cube unit)
    groups, which is what keeps a padded tail from leaking into the next slice;
    ``overlap = 0`` runs everything on the caller's stream.  ``c`` is the real
    chunk count: the wide/assemble halves own c-sized-per-chunk buffers padded
    up to ``c_solve``, but rk/rv/W/U only carry the c real chunks, so the Cube
    launch - whose kernel clamps its loop to the count it is handed - has to be
    clamped to them (a padded count sends the tail block's stores past W/U).
    ``pmid`` (the assemble's P tile, 25.2 MB per call at [1,8192,96,128] - the
    50.3 MB in section 11.39 is its GM round trip, write plus read) is passed as
    None when the load mode keeps P on chip: the caller does not allocate it,
    ``_pack_ptrs`` writes a null pointer, and the kernel never dereferences it
    (docs section 11.40).  Modes 0/1 own the tile and need it non-null, so this
    is a property of the mode, not of the call site.
    """
    unit = math.lcm(nch, asm_nchunk, wu_nchunk)
    ngrp = c_solve // unit
    # Slices have to stay fat enough to keep a block's fixed cost amortised:
    # at most ngrp/8 of them, and never more than one per group.
    slices = min(overlap, max(1, ngrp // 8)) if overlap > 0 else 0
    ovl = slices >= 2
    overlap = slices
    cur = torch_npu.npu.current_stream()
    pairs = [(0, ngrp)] if not ovl else \
        [(i * ngrp // overlap, (i + 1) * ngrp // overlap) for i in range(overlap)]
    if ovl:
        sa, sb = _solve_streams(a16.device)
        # The wide slices read what the caller's stream produced (L, the aqk
        # Gram's bf16 tile) and the Cube slices write W/U that K2 reads there.
        sa.wait_stream(cur)
    for glo, ghi in pairs:
        lo, n = glo * unit, (ghi - glo) * unit
        wargs = _pack_ptrs([L[lo:], eye, a32[lo:], a16[lo:], xb[lo:],
                            lneg[lo:]]) + [_i(n), _i(a16_mode()), _i(1 if debug_stores else 0)]
        aargs = _pack_ptrs([a16[lo:], xb[lo:], lneg[lo:],
                            None if pmid is None else pmid[lo:]]) + \
            [_i(n), _i(asm_load_mode())]
        cargs = _pack_ptrs([a16[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + \
            [_i(n), _i(cube_a16_resident())]
        ncube = min(n, c - lo)
        if ovl:
            _launch("kda_solve_wu_wide", n // nch, wargs, sa.npu_stream)
            ev = torch_npu.npu.Event()
            ev.record(sa)
            sb.wait_event(ev)
            _launch("kda_solve_assemble", (n + asm_nchunk - 1) // asm_nchunk,
                    aargs, sb.npu_stream)
            if ncube > 0:
                _launch("kda_solve_wu_cube_kernel",
                        (ncube + wu_nchunk - 1) // wu_nchunk, cargs, sb.npu_stream)
        else:
            _launch("kda_solve_wu_wide", n // nch, wargs, stream)
            _launch("kda_solve_assemble", (n + asm_nchunk - 1) // asm_nchunk,
                    aargs, stream)
            if ncube > 0:
                _launch("kda_solve_wu_cube_kernel",
                        (ncube + wu_nchunk - 1) // wu_nchunk, cargs, stream)
    if ovl:
        cur.wait_stream(sb)


def _defines() -> str:
    """Source-side flags for the RTC compile.

    aclrtcCreateProg has no -D option, so the chunk size and the two
    chunk-dependent block sizes ride in front of the kernel source; every
    guarded kernel (KDA_CHUNK, KDA_SOLVE_WIDE_NCHUNK, KDA_WU_NCHUNK) picks
    them up through its own #ifndef default.

    Compiling a KDA_CHUNK kernel without this prefix is not a build error: the
    kernel silently keeps its ``#ifndef KDA_CHUNK 16`` default and answers a
    C=32/64 host call with 16 rows per chunk (fast, plausible, wrong).  Every
    compile in this package therefore goes through ``_rtc``, and
    ``tests/test_rtc_compile_config.py`` fails the build if a call site
    compiles a kernel without the prefix.
    """
    return ("#define KDA_CHUNK %d\n#define KDA_SOLVE_WIDE_NCHUNK %d\n"
            "#define KDA_SOLVE_WIDE_SUBB %d\n#define KDA_ASM_NCHUNK %d\n"
            "#define KDA_WU_NCHUNK %d\n#define KDA_MAXH %d\n"
            % (CHUNK, SOLVE_WIDE_NCHUNK, SOLVE_WIDE_SUBB, ASM_NCHUNK,
               WU_NCHUNK, PERSIST_MAXH))


def compile_config() -> dict[str, int]:
    """The geometry the kernels were compiled with, as a plain dict.

    ``get_last_profile()`` carries this next to the timings: the RTC compile
    leaves no trace in the .o of how big KDA_CHUNK / KDA_MAXH / the solve block
    sizes were, so a measurement without them cannot be compared against
    another one.
    """
    return {
        "KDA_CHUNK": CHUNK,
        "KDA_MAXH": PERSIST_MAXH,
        "KDA_SOLVE_WIDE_NCHUNK": SOLVE_WIDE_NCHUNK,
        "KDA_SOLVE_WIDE_SUBB": SOLVE_WIDE_SUBB,
        "KDA_ASM_NCHUNK": ASM_NCHUNK,
        "KDA_WU_NCHUNK": WU_NCHUNK,
        "KDA_SOLVE_OVERLAP": SOLVE_OVERLAP,
        "nv": NV,
    }


def _rtc(rel: str, name: str) -> None:
    """Compile ``kernels/v1/<rel>`` as ``name``, defines prefix included.

    Every RTC compile of this package has to come through here (see
    ``_defines``); the C=16-only kernels of the S12-S15 experiments ignore the
    prefix, but they get it too so that the invariant is mechanical.

    The source is read as ``utf-8-sig``: a UTF-8 BOM in front of the first
    ``#include`` is not a warning here but a hard compile error
    (``unexpected character <U+FEFF>``, followed by every type name in the
    file turning unknown), and it is invisible in an editor - a BOM once made
    ``k2_persistent_scan.cpp`` uncompilable with no trace of why.  Reading it
    away in the one place every compile goes through keeps that from coming
    back through any of the kernels.
    """
    rtc_compile(_defines() + (ROOT / rel).read_text(encoding="utf-8-sig"),
                name, "")


_SOURCES: list[tuple[str, str]] = [
    ("kernels/v1/preprocess.cpp", "kda_preprocess_kernel"),
    ("kernels/v1/k1_gram.cpp", "kda_gram_kernel"),
    ("kernels/v1/k1_pre_gram.cpp", "kda_pre_gram_kernel"),
    ("kernels/v1/k1_pre_gram_mix.cpp", "kda_pre_gram_mix"),
    ("kernels/v1/k1_solve_wu.cpp", "kda_solve_wu_kernel"),
    ("kernels/v1/k1_solve_wu_wide.cpp", "kda_solve_wu_wide"),
    ("kernels/v1/k1_solve_assemble.cpp", "kda_solve_assemble"),
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
    ("kernels/v1/k2_state_loop.cpp", "kda_k2_state_loop"),
    ("kernels/v1/k2_out_parallel.cpp", "kda_k2_out_parallel"),
    ("kernels/v1/k2_outstate_full.cpp", "kda_k2_outstate_full_kernel"),
    ("kernels/v1/k2_outstate.cpp", "kda_k2_outstate_kernel"),
]


def _compile_all() -> None:
    global _COMPILED
    if _COMPILED:
        return
    for rel, name in _SOURCES:
        _rtc(rel, name)
    _COMPILED = True


def _compile_persistent() -> None:
    global _PERSISTENT_COMPILED
    if _PERSISTENT_COMPILED:
        return
    _rtc("kernels/v1/k2_persistent.cpp", "kda_k2_persistent_kernel")
    _PERSISTENT_COMPILED = True


def _compile_persistent_scan() -> None:
    global _PERSISTENT_SCAN_COMPILED
    if _PERSISTENT_SCAN_COMPILED:
        return
    _rtc("kernels/v1/k2_persistent_scan.cpp", "kda_k2_persistent_scan_kernel")
    _PERSISTENT_SCAN_COMPILED = True

def _compile_triton_aiv() -> None:
    global _TRITON_AIV_COMPILED
    if _TRITON_AIV_COMPILED:
        return
    _rtc("kernels/v1/k2_triton_aiv.cpp", "kda_k2_triton_aiv")
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
        raise ValueError("support is T>=%d, T%%%d==0, D=128" % (CHUNK, CHUNK))
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
    # The compile geometry rides with every profile: an RTC kernel carries no
    # trace of the defines it was built with, and a timing without them cannot
    # be compared against another one.
    profile["compile"] = compile_config()
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
    k2_mode: str | None = None,
):
    """Run an opt-in AscendC KDA path; all KDA arithmetic stays on device.

    ``q/k/v`` are BF16 ``[B, T, H, 128]``, ``g`` is FP32 ``[B, T, H, 128]``,
    ``beta`` and ``A_log``/``bias`` are FP32; returns ``(out, final_state)``,
    plus the stage tensors when ``return_intermediates`` is set.

    ``k2_mode`` names the K2 (state recurrence) implementation and defaults to
    ``"persistent_loop"``: one MIX launch runs the whole chunk loop and keeps
    the fp32 state in UB, and it is the only implementation that follows the
    build's ``KDA_CHUNK``.  The historical per-chunk modes are C=16-only
    kernels; they are reachable through ``kda_ascendc_v1.experimental`` and
    are rejected here so that a caller cannot silently run one at another
    chunk size.
    """
    if k2_mode is None:
        k2_mode = PERSISTENT_LOOP
    if k2_mode != PERSISTENT_LOOP:
        if k2_mode in C16_ONLY_K2_MODES:
            raise ValueError(
                "k2_mode=%r is an experimental C=16-only kernel; the public "
                "API serves %r only (use "
                "kda_ascendc_v1.experimental.kda_bt16_fwd_ascendc_experimental "
                "to benchmark the historical modes)" % (k2_mode, PERSISTENT_LOOP))
        raise ValueError("unsupported k2_mode %r (choose from %s)"
                         % (k2_mode, ", ".join(sorted(K2_MODES))))
    return _kda_fwd_impl(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, A_log=A_log, bias=bias,
        lower_bound=lower_bound, return_intermediates=return_intermediates,
        k2_mode=k2_mode)


def _kda_fwd_impl(
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
    k2_mode: str = PERSISTENT_LOOP,
):
    """The one implementation both entry points call; ``k2_mode`` is validated.

    The public wrapper only lets ``persistent_loop`` through; this is where the
    remaining modes are checked against the build, because the C=16-only
    kernels would otherwise silently compute a 16-row answer.
    """
    # Builds outside the validated set are refused before anything is compiled
    # or launched: a silently wrong answer is the one failure mode this package
    # cannot afford (see C16_ONLY_K2_MODES for the other half of the same rule).
    if CHUNK not in SUPPORTED_CHUNKS and os.environ.get(
            "KDA_ALLOW_UNSUPPORTED_CHUNK", "0") != "1":
        raise ValueError(
            "KDA_CHUNK=%d is an unsupported build: the kernels do follow the "
            "chunk size (M = KDA_CHUNK, and stage 3 contracts over K = M), but "
            "only the sizes in api.SUPPORTED_CHUNKS (16, 32, 64) are gated by "
            "the CHUNK x shape correctness matrix in "
            "tests/test_chunk_shape_matrix.py.  Anything else is unchecked, and "
            "a tile that is wrong for its chunk size is a silently wrong answer "
            "here (or a kernel-side aivec error), never a host-side report - "
            "C=128, for instance, sizes its L0C queue and its L0A allocation to "
            "exactly 128 KB and 64 KB, the whole of both buffers.  Build with "
            "KDA_CHUNK=16, 32 or 64; set KDA_ALLOW_UNSUPPORTED_CHUNK=1 to run it "
            "anyway (debugging only)." % CHUNK)
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
    if k2_mode not in K2_MODES:
        raise ValueError("unsupported k2_mode %r" % (k2_mode,))
    if k2_mode in C16_ONLY_K2_MODES and CHUNK != 16:
        raise ValueError(
            "k2_mode=%r is a C=16 implementation (M = 16 is a literal in its "
            "kernels) but this build has KDA_CHUNK=%d: it would solve 16 rows "
            "of every %d-row chunk.  Rebuild with KDA_CHUNK=16 or use %r."
            % (k2_mode, CHUNK, CHUNK, PERSISTENT_LOOP))
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
    c_solve = (c + SOLVE_WIDE_NCH - 1) // SOLVE_WIDE_NCH * SOLVE_WIDE_NCH
    L = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    if c_solve != c:
        # The wide kernel is launched in whole SOLVE_WIDE_NCH-chunk blocks, so
        # the chunk count is rounded up and the tail chunks are solved as well -
        # on the strength of L's tail being zero, which is what makes them come
        # out as a harmless identity (k1_solve_wu_wide.cpp).  The Gram only
        # writes the c real chunks, so the tail has to be zeroed here.
        L[c:].zero_()
    mask_s, mask_l = _tri_masks(q.device)
    # Stages 1+2 run as one AIV block per chunk ("k1_pre_gram.cpp"): the fused
    # kernel consumes Qn/Kn/Gc out of UB instead of round-tripping 32 KB per
    # chunk through GM.  It addresses every token in packed ``[c, CHUNK, D]``
    # order, so the public [B, T, H, D] tensors have to be packed first, and
    # it only writes Qn/Kn/Gate/Gc when those pointers are non-null (the
    # ``return_intermediates`` debug path).
    keep = return_intermediates
    # Three of the buffers a call allocates exist for the
    # ``return_intermediates`` views and are read by nothing on the device:
    # Aqk32's masked fp32 copy (the band loop reads the raw one; Aqk16 is
    # rounded from registers), A32 (the wide solve's fp32 A_inv - the Cube
    # solve reads A16) and BetaOut.  Writing them costs 405 MB per call at
    # C=64 and 0.19 ms at R3's measured store rate, so the two kernels take a
    # ``debugStores`` flag and production passes 0.  It is a flag and not a
    # nulled pointer because the raw Aqk32 slot stays live either way: the
    # pre_gram band loop reads it back before the masked copy exists.
    # Measured interleaved at [1,8192,96,128]/C=64 (tools/probe_dead_store.py,
    # 3 rounds x 800 ms): 10.765 -> 10.601 ms for the three stores together, so
    # the flag defaults to off and KDA_DEBUG_STORES=1 is the way back to the
    # old behaviour (docs 11.29).
    keep_debug = keep or os.environ.get("KDA_DEBUG_STORES", "0") == "1"
    pre_head = [q, k, v, g, beta_pack, A_log, bias,
                qn if keep else None, kn if keep else None,
                gate if keep else None, gc if keep else None,
                beta_out, decay, rk, rv, qg, kg]
    pre_tail = [aqk32, aqk16, L, mask_s, mask_l]
    # One block walks `pre_unroll` consecutive chunks.  The stage is issue-bound
    # and pays a fixed per-block setup cost, so unrolling is worth 8-12% from a
    # few hundred chunks up (measured 2.679 -> 2.342 ms at [1,8192,32]); tiny
    # grids (fewer than 256 chunks) stay at 1 because the loop wrapper itself
    # costs a few percent there.
    # R3: the block count is what costs - every 24-block wave re-pays the
    # block prologue - so aim at a fixed number of *waves* rather than at a
    # block-count floor.  The earlier sweep (768 blocks (pu 8) 4.163, 384 (16)
    # 4.099, 192 (32) 4.043, 96 (64) 4.031 at pre_gram ~4.0 ms) pointed at
    # ~4 waves of 24 AICs = 96 blocks, i.e. one block per 128 chunks
    # (pu = c / 192); re-measured on the current stage (3.31 ms baseline,
    # interleaved MIN of 3: pu 8 3.518, 16 3.374, 32 3.315, 64 3.271, and past
    # the cap 96 3.771, 128 3.508, 192 4.872, 256 3.416), so the 4-wave target
    # holds and pu = 64 is the optimum at c = 12288.  c // 192 keeps 96 blocks
    # (pu capped at 64) for every larger shape and leaves the tiny grids at 1.
    pre_unroll = (int(os.environ.get("KDA_PRE_UNROLL", "0"))
                  or (1 if c < 256 else min(64, max(2, c // 192))))
    mark("pre_gram_start")
    if os.environ.get("KDA_PRE_GRAM", "mix") == "aiv":
        # The vector-only experiment path has no debugStores flag: it keeps
        # writing its Aqk32/BetaOut tiles, exactly as before this change.
        pre_args = _pack_ptrs(pre_head + pre_tail)
        pre_args += [_i(b), _i(t), _i(h), _f(lower_bound), _f(scale), _i(pre_unroll),
                     _i(qk_row_bytes), _i(g_row_bytes)]
        _launch("kda_pre_gram_kernel", (c + pre_unroll - 1) // pre_unroll, pre_args, stream)
    else:
        # The two intra-chunk Grams run on the paired Cube: the AIVs publish
        # ga/gk1/gb in bf16 (12 KB per chunk) and the AIC does both 16x16x128
        # Mmads per step.  The Gram half is 1.05 ms of the vector-only block's
        # 2.21 ms and the vector pipe is the bottleneck of the whole stage, so
        # the Cube's work is hidden behind the AIV's remaining ~3.3 us per chunk
        # (see kernels/v1/k1_pre_gram_mix.cpp).
        gram_ops = torch.empty((3, c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
        # The cross-band k side of the (1, 0) Gram block: a CHUNK = 64 gate
        # needs one decay reference per 32-row band, and the Gram block that
        # spans two bands is served by a second copy of band 0's k rows
        # published under band 1's centre (the kernel note).  A single-band
        # chunk writes no tile here, so CHUNK <= 32 keeps a one-row stand-in.
        gram_x = torch.empty((c, max(1, CHUNK // 2), D), dtype=torch.bfloat16,
                             device=q.device)
        pre_args = _pack_ptrs(pre_head + [gram_ops[0], gram_ops[1], gram_ops[2], gram_x]
                              + pre_tail)
        pre_args += [_i(b), _i(t), _i(h), _f(lower_bound), _f(scale), _i(pre_unroll),
                     _i(qk_row_bytes), _i(g_row_bytes), _i(1 if keep_debug else 0)]
        # One MIX block pairs an AIC with two AIV subcores, so it covers
        # 2 * pre_unroll chunks.
        _launch("kda_pre_gram_mix", (c + 2 * pre_unroll - 1) // (2 * pre_unroll),
                pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a32 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    a16 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty_like(qn)
    U = torch.empty_like(qn)
    mark("solve_start")
    if SOLVE_WIDE_SUBB > 1:
        # Two-level solve: the wide kernel writes the diagonal sub-blocks (Xb
        # next to the parent tile) and the negated coupling triangle, the
        # assemble kernel forms X21 on the Cube, and the Cube solve consumes
        # the assembled tile.  See _launch_solve_two_level.
        sub = CHUNK // SOLVE_WIDE_SUBB
        bf16 = torch.bfloat16
        xb = torch.empty((c_solve, SOLVE_WIDE_SUBB, sub, sub), dtype=bf16, device=q.device)
        lneg = torch.empty((c_solve, sub, sub), dtype=bf16, device=q.device)
        # Load mode 2 keeps P on chip (section 11.39), so its GM tile is
        # neither written nor read: the 25.17 MB allocation at [1,8192,96,128]
        # (24 MiB = 12288 chunks of [32, 32] bf16 - the 50.3 MB section 11.39
        # quotes is the round trip that tile used to carry) is dead memory
        # under mode 2 and only modes 0/1 need it.  The mode is read per call,
        # so a probe that flips the knob back gets the tile allocated again on
        # the next call.
        pmid = (None if asm_load_mode() >= 2 else
                torch.empty((c_solve, sub, sub), dtype=bf16, device=q.device))
        _launch_solve_two_level(c_solve, c, SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK,
                                SOLVE_OVERLAP, L,
                                _tri_eye(q.device, CHUNK // SOLVE_WIDE_SUBB),
                                a32, a16,
                                xb, lneg, pmid, rk, rv, W, U, stream,
                                1 if keep_debug else 0)
    else:
        # One AIV block solves SOLVE_WIDE_NCHUNK chunks with every vector
        # instruction (see the kernel header); the padded chunks are solved too
        # but land outside the first c.  The two-level operands are null here:
        # the kernel takes them in every build, and a short argument list would
        # leave it reading C out of the args array's tail.
        solve_args = _pack_ptrs([L, _tri_eye(q.device), a32, a16, None, None])
        solve_args += [_i(c), _i(a16_mode()), _i(1 if keep_debug else 0)]
        _launch("kda_solve_wu_wide", c_solve // SOLVE_WIDE_NCHUNK, solve_args, stream)
        # The Cube needs the bf16 A_inv the substitution just wrote, so the two
        # launches stay ordered on the stream.
        _launch("kda_solve_wu_cube_kernel", (c + WU_NCHUNK - 1) // WU_NCHUNK,
                _pack_ptrs([a16, rk, rv, W, U]) + [_i(c), _i(cube_a16_resident())],
                stream)
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

    if k2_mode in (PERSISTENT_LOOP, SPLIT_STATE_OUT):
        split = k2_mode == SPLIT_STATE_OUT
        # One device-side chunk loop.  Each block owns up to MAXH=4 heads (both
        # AIV subcores run the same flag sequence and split the value dim), keeps
        # its fp32 state in UB across all chunks and never writes S32 until the
        # end, so the whole recurrence is a single MIX launch.  The block count
        # must keep the head map total: nblk * MAXH >= bh.
        # One head per block is ~4% faster than two while the heads still fit in
        # the AIC count (measured at [2,4096,8]: 4.38 vs 4.55 ms); past that,
        # two heads per block keep every block resident instead of queueing a
        # second wave (32 blocks at [1,8192,32] cost 17.2 vs 13.8 ms).
        aic_cores = 24
        # Heads per block.  The kernel keeps up to MAXH=4 heads resident and one
        # block per AIC is all the parallelism this part has, so ask for as few
        # blocks as the head count allows while still filling all 24 AICs - a
        # second wave costs a whole pass over the chunks.  Measured at
        # [1,8192,96,128]: 4 heads/block 6.53-6.56 ms against 6.71-6.88 for
        # 2 heads/block (MIN of 3 in-process rounds, 3 rounds each), so the
        # auto value is ceil(bh / 24) capped at the kernel's MAXH.
        maxh = max(1, min(PERSIST_MAXH, (bh + aic_cores - 1) // aic_cores))
        want = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
        nblk = want if 0 < want <= bh else (bh if bh <= aic_cores
                                           else (bh + maxh - 1) // maxh)
        nblk = max(nblk, (bh + maxh - 1) // maxh)
        s32 = torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device)
        # The fused loop publishes the bf16 state into one slot it overwrites
        # every chunk; the split's state kernel publishes the same bytes into a
        # per-chunk slot, which is what the output kernel reads as H[c].  The
        # 384 MiB snapshot is the split's whole extra allocation and the only
        # reason its write side is free (see k2_state_loop.cpp).
        s16 = (None if split else
               torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device))
        hsnap = (torch.empty((tasks, nt, BV, D), dtype=torch.bfloat16, device=q.device)
                 if split else None)
        # d1/d2/d3 cross to the vector side as bf16 (the loop's fixpipe rounds
        # them), so they are allocated bf16 here as well: the loop indexes the
        # buffers in bf16 elements and an fp32 allocation silently half-filled
        # them and returned garbage through ``return_intermediates``.
        d1 = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        d2 = torch.empty_like(d1)
        d3 = torch.empty_like(d1)
        d4f = torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
        # The loop writes the public [B, T, H, D] layout directly: one 128 B
        # run per (chunk row) with a NH*D-element stride between rows, which
        # removes the 186 us `permute(0,3,4,1,2,5).contiguous()` (67 MB in +
        # 67 MB out) the api used to run over the task layout.  The strided
        # store costs 0.04 ms of device time, the same-store check is
        # bit-exact against the old layout + host permute.
        out_public = torch.empty((b, t, h, D), dtype=torch.bfloat16, device=q.device)
        # `vnew` is the row-major twin of `vnew_t`, and the kernel reads only
        # `vnew_t` - the row-major copy is written solely for
        # ``return_intermediates`` (the pQn/pKn pattern).  Production passes a
        # null pointer and skips both the store (0.095 ms of K2 at
        # [1,8192,96,128], measured interleaved) and the 201 MB allocation.
        vnew = (torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
                if return_intermediates else None)
        vnew_t = torch.empty((tasks, nt, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
        h0 = None if initial_state is None else initial_state.view(bh, D, D)
        # kg goes to the loop in its public [c, CHUNK, D] layout: the AIC loads
        # it into L1 with Nd2Nz and transposes each 16x16 fractal on the way
        # into L0B (LoadDataWithTranspose), which is what kg_t fed and what
        # kda_kg_transpose built.  Dropping that launch saves its 0.16 ms of
        # device time (and the 67 MB round trip) per pass.
        mark("k2_start")
        if split:
            # Two launches, both on the caller's stream: the output kernel
            # needs every chunk's H[c], and a device-side "chunk c is ready"
            # hand-off between two launches is the Level 4 scheduler, not
            # this candidate.  The pair's total is the number that decides.
            mark("k2_state_start")
            _launch("kda_k2_state_loop", nblk,
                    _pack_ptrs([U, W, kg, decay, d1, d4f, hsnap,
                                vnew, vnew_t, h0, s32]) +
                    [_i(bh), _i(nt), _i(NV), _i(nblk), _i(h)], stream)
            # Not "..._ms": total_ms sums every *_ms key, and these two are
            # inside the k2_ms span.
            finish("k2_state", "k2_state_start")
            mark("k2_out_start")
            # `d2`/`d3` are the same per-(task, chunk) bf16 slots the fused
            # loop's fixpipes wrote, so the arithmetic (and the rounding
            # positions) of out = d2*scale + d3 are unchanged.
            _launch("kda_k2_out_parallel", nblk,
                    _pack_ptrs([qg, aqk16, kg, d2, d3, hsnap, out_public, vnew_t]) +
                    [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale), _i(h)], stream)
            finish("k2_out", "k2_out_start")
        else:
            _launch("kda_k2_persistent_loop", nblk,
                    _pack_ptrs([U, W, qg, aqk16, kg, decay, d1, d2, d3, d4f,
                                out_public, vnew, vnew_t, h0, s32, s16]) +
                    [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale), _i(h)], stream)
        finish("k2_ms", "k2_start")
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
                 "d3": d3, "d4": d4f, "state_s32": s32,
                 "persistent_loop": not split, "split_state_out": split,
                 "S16snap": hsnap}
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
