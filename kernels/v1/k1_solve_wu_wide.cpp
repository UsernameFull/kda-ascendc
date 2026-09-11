// K1 stage 3: 16x16 forward-substitution solve, WIDE_NCHUNK chunks per vector
// instruction.
//
//   A_inv = (I + L)^{-1},  A_inv[i,:] = e_i + sum_{j<i} (-L[i][j]) A_inv[j,:]
//
// Same recursion as k1_solve_wu.cpp, but the vector lane width is the chunk
// axis instead of the 16-element row.  On this core a count-mode vector
// instruction costs ~25-30 cycles of issue however few lanes it fills (three
// independent probes agree: +8 one-repeat Adds per chunk in the fused kernel,
// the unroll 4/8/16 sweep there, and the variant timings here), so the old
// per-chunk Axpy loop paid that 120 times per chunk.  Here one MulAddDst
// updates row i of *thirty-two* chunks at once: 120 instructions per 32 chunks
// instead of 120 per chunk.
//
// The three pieces of layout that make it work (all verified on hardware with
// a probe kernel, see docs/ASCENDC_V1_KERNELS.md):
//   1. Brcb(dst, src, 2*NC, BrcbRepeatParams(1, 8)) reads one 32B block (eight
//      columns of one chunk) per repeat and writes eight blocks, so block
//      (16*ch + j) of the expansion holds -L[ch][i][j] eight times over.
//   2. A binary vector op with src1BlkStride = 0 reads that one block for both
//      32B halves of a 16-lane repeat, i.e. it broadcasts the coefficient over
//      exactly the sixteen lanes of one chunk's row, and src1RepStride = 16
//      blocks walks the chunks.
//   3. The coefficients come from DataCopyParams(NC, 2, 30, 0), a gather of
//      two-block rows 32 blocks apart, which turns the chunk-major L of GM
//      into the per-row layout Brcb wants.  The transposed layout only exists
//      in UB: GM traffic is unchanged (2 KB per row of 32 chunks = the same
//      1 KB per chunk that L occupies).
//
// The upper triangle of L is zero (the Gram kernel writes only the lower one
// and the mask zeroes the diagonal), so the j >= i terms contribute nothing;
// they are simply not issued.
//
// Measured at [1,8192,32] against the per-chunk version, bit-identical on both
// outputs (fp32 max diff 0.000e+00, every bf16 value equal):
//   NC =  8  0.2235 ms      NC = 16  0.1285 ms      NC = 32  0.1006 ms
//   the old kernel                 0.9120 ms
// NC = 64 does not fit: the four live tiles would need ~190 KB of the 192 KB.
// What is left is memory traffic - 40 MB per pass (16 MB L read, 16 MB fp32
// A_inv write, 8 MB bf16 A_inv write) at ~370 GB/s is ~0.11 ms, so the kernel
// is now at the bandwidth floor rather than the issue floor.
//
// C is rounded up to a multiple of NC by the caller, which zeroes the tail rows
// of L so the padded chunks solve to a harmless identity.
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t MM = M * M;
#ifndef KDA_SOLVE_WIDE_NCHUNK
#define KDA_SOLVE_WIDE_NCHUNK 32
#endif
constexpr int32_t NC = KDA_SOLVE_WIDE_NCHUNK;
constexpr int32_t CH = NC * MM;   // floats in one block's A_inv
constexpr int32_t RW = NC * M;    // floats in one gathered row of L

extern "C" __global__ __aicore__ void kda_solve_wu_wide(
    GM_ADDR pL, GM_ADDR pEye, GM_ADDR pA32, GM_ADDR pA16, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> bLraw, bCexp, bAf, bAb, bEye;
    pipe.InitBuffer(bLraw, CH * 4);
    pipe.InitBuffer(bCexp, 2 * NC * 64 * 4);
    pipe.InitBuffer(bAf, CH * 4);
    pipe.InitBuffer(bAb, CH * 2);
    pipe.InitBuffer(bEye, MM * 4);
    LocalTensor<float> lraw = bLraw.Get<float>(), cexp = bCexp.Get<float>();
    LocalTensor<float> af = bAf.Get<float>(), eye = bEye.Get<float>();
    LocalTensor<bfloat16_t> ab = bAb.Get<bfloat16_t>();
    GlobalTensor<float> L, Eye, A32;
    GlobalTensor<bfloat16_t> A16;
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pEye));
    A32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pA32));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));

    DataCopy(eye, Eye[0], DataCopyParams(M, 2, 0, 0));
    for (int32_t i = 0; i < M; ++i) {
        DataCopy(lraw[i * RW], L[static_cast<uint64_t>(c0) * MM + i * M],
                 DataCopyParams(NC, 2, 30, 0));
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    // Fold the sign of the coefficient into the load: one instruction for the
    // whole gathered tile.
    Muls(lraw, lraw, -1.0f, CH);
    PipeBarrier<PIPE_V>();

    for (int32_t i = 0; i < M; ++i) {
        // Row i starts from e_i for every chunk (one instruction, 16 chunks).
        Adds(af[i * M], eye[i * M], 0.0f, M, NC, UnaryRepeatParams(1, 1, 32, 0));
        Brcb(cexp, lraw[i * RW], 2 * NC, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < i; ++j) {
            MulAddDst(af[i * M], af[j * M], cexp[j * 8], M, NC,
                      BinaryRepeatParams(1, 1, 0, 32, 32, 16));
        }
        PipeBarrier<PIPE_V>();
    }
    Cast(ab, af, RoundMode::CAST_RINT, CH);
    PipeBarrier<PIPE_V>();
    Cast(af, ab, RoundMode::CAST_NONE, CH);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
    for (int32_t ch = 0; ch < NC; ++ch) {
        DataCopy(A32[static_cast<uint64_t>(c0 + ch) * MM], af[ch * MM],
                 DataCopyParams(M, 2, 0, 0));
        DataCopy(A16[static_cast<uint64_t>(c0 + ch) * MM], ab[ch * MM],
                 DataCopyParams(M, 1, 0, 0));
    }
    PipeBarrier<PIPE_ALL>();
}
