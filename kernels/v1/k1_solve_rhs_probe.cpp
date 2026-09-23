// Timing probe (NOT a production kernel): how much does the *AIV half* of a
// wide-RHS forward substitution cost, against the recursion the shipped
// two-level solve already runs?
//
// The order of 2026-09-23 (route 2) asks for a C64 two-block prototype of
//
//     (I + L) X = [Rk, Rv]        X_i = T_ii^-1 (R_i - sum_{j<i} L_ij X_j)
//
// instead of the shipped A_inv = (I+L)^-1 followed by W/U = A_inv [Rk, Rv].
// The premise is that the fused form drops the full inverse materialisation.
// What it does not drop - and what governs the price - is the row recursion
// itself: both forms run
//
//     for i: for j < i:  X_i += L_ij * X_j
//
// over the same L, so the *instruction count* per chunk is set by the block
// split (sum_i i = M(M-1)/2 per diagonal block) while the *lane width* is set
// by what the instruction's other operand is: the shipped form recurses on the
// M x M inverse (M lanes, wide over chunks via the tile's instance axis), the
// fused form recurses on the [M, 256] RHS (256 lanes, and with the same UB it
// fits four times fewer instances).
//
// This kernel runs both recursion shapes back to back in one process, on the
// real chunk count, with no gathers, no stores and no casts - i.e. it measures
// a *lower bound* on each shape's AIV cost, which is the number the decision
// rule needs: if the fused shape's floor is already at or above the whole
// current solve stage (2.53 ms at [1,8192,96,128]/C=64), no layout, pipeline or
// Cube-side split can bring the route back.
//
// Modes (all four process the same 12288 chunk-instances per launch):
//   0  null: the tile init and the epilogue store only
//   1  shipped shape: SB=2 sub-blocks of M=32, tile = [row][8 instances][32
//      lanes], 124 MulAddDst per chunk, 8 repeats each - a transcription of
//      k1_solve_wu_wide.cpp's inner loop with the L gather and the A16/Xb/Lneg
//      stores removed
//   2  fused two-block shape: SB=2 sub-blocks of M=32 against the 256-wide
//      RHS, tile = [sub-block][row][2 chunks][256 lanes] (128 KB of UB), 496
//      instructions per chunk, 8 repeats each
//   3  fused leaf-16 shape: SB=4 sub-blocks of M=16 against the 256-wide RHS,
//      same 128 KB tile, 240 instructions per chunk, 8 repeats each
//
// Mode 1's number calibrates the probe against the shipped kernel's measured
// 1.84 ms (its recursion is not all of that; the gathers and the A16 store are
// the rest).  Modes 2/3 are the lower bounds the fused route has to fit inside
// 2.53 ms together with every gather, store and Cube step it also needs.
#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr int32_t LANES = 256;   // the wide RHS: Rk (128) | Rv (128)
constexpr int32_t CM = 64 * 255; // largest count-mode calCount

// One iteration = NCH chunks of one parent chunk, SB diagonal sub-blocks of
// M rows each.  The tile is [sub-block][row][chunk][LANES] fp32; the
// coefficient tile holds one 32 B block per (row, chunk, lane group), so a
// repeat broadcasts its block (src1BlkStride = 0) and walks one block per
// repeat (src1RepStride = 1).
template <int32_t M, int32_t SB, int32_t NCH>
__aicore__ inline void rhs_recursion(TPipe& pipe, LocalTensor<float>& x,
                                     LocalTensor<float>& cexp) {
    constexpr int32_t RW = NCH * LANES;              // floats in one row
    constexpr int32_t REP = NCH * (LANES / 64);      // 64-lane repeats
    constexpr int32_t NB = SB * M * RW;              // tile floats
    static_assert(NB * 4 <= 128 * 1024, "tile over 128 KB of UB");
    static_assert(NB > 0, "unused");
    for (int32_t s = 0; s < SB; ++s) {
        LocalTensor<float> xs = x[s * M * RW];
        for (int32_t i = 0; i < M; ++i) {
            // "X_i starts from R_i": in the fused kernel this is the RHS load.
            Duplicate(xs[i * RW], 0.01f, NCH * LANES);
            PipeBarrier<PIPE_V>();
            for (int32_t j = 0; j < i; ++j) {
                MulAddDst(xs[i * RW], xs[j * RW], cexp[j * 8], 64, REP,
                          BinaryRepeatParams(1, 1, 0, 8, 8, 1));
            }
            PipeBarrier<PIPE_V>();
        }
    }
}

// The shipped shape: the tile is [row][instance][M lanes] with the instances
// being (sub-block, chunk) pairs, so one instruction is wide over four chunks
// *and* both sub-blocks at once, at M-lane granularity.  Transcribed from
// k1_solve_wu_wide.cpp (M = 32, NC = 8 instances), with the eye source
// replaced by a duplicated row.
__aicore__ inline void inverse_recursion(TPipe& pipe, LocalTensor<float>& x,
                                         LocalTensor<float>& cexp,
                                         LocalTensor<float>& eye) {
    constexpr int32_t M = 32;
    constexpr int32_t NC = 8;              // instances = 2 sub-blocks x 4 chunks
    constexpr int32_t RW = NC * M;         // 256 floats in one row
    for (int32_t i = 0; i < M; ++i) {
        Adds(x[i * RW], eye[i * M], 0.0f, M, NC, UnaryRepeatParams(1, 1, M / 8, 0));
        Brcb(cexp, eye, NC * M / 8, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < i; ++j) {
            MulAddDst(x[i * RW], x[j * RW], cexp[j * 8], M, NC,
                      BinaryRepeatParams(1, 1, 0, M / 8, M / 8, M));
        }
        PipeBarrier<PIPE_V>();
    }
}
}  // namespace

extern "C" __global__ __aicore__ void kda_solve_rhs_probe(
    GM_ADDR pOut, int32_t groups, int32_t mode, int32_t storeOff) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t bid = static_cast<int32_t>(GetBlockIdx());
    TPipe pipe;
    TBuf<TPosition::VECCALC> bX, bC;
    // 128 KB tile + 16 KB of coefficients/eye: the fused shapes' own budget.
    pipe.InitBuffer(bX, 32768 * 4);
    pipe.InitBuffer(bC, 4096 * 4);
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    LocalTensor<float> x = bX.Get<float>();
    LocalTensor<float> c = bC.Get<float>();
    Duplicate(c, 0.01f, 1024);
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pOut));
    for (int32_t g = 0; g < groups; ++g) {
        if (mode == 0) {
            Duplicate(x, 0.01f, 2048);
            PipeBarrier<PIPE_V>();
        } else if (mode == 1) {
            LocalTensor<float> eye = c[512];
            inverse_recursion(pipe, x, c, eye);
        } else if (mode == 2) {
            rhs_recursion<32, 2, 2>(pipe, x, c);
        } else {
            rhs_recursion<16, 4, 2>(pipe, x, c);
        }
        SetFlag<HardEvent::V_MTE3>(e3);
        WaitFlag<HardEvent::V_MTE3>(e3);
        DataCopy(out[static_cast<uint64_t>(bid) * 8], x[storeOff],
                 DataCopyParams(1, 1, 0, 0));
    }
}
