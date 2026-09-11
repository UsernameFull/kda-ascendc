// K1 stage 3: 16x16 forward-substitution solve.
//
//   A_inv = (I + L)^{-1}      (L carries the strict lower triangle of Akk)
//
// The inverse follows the row recursion
//   A_inv[i] = e_i - sum_{j<i} L[i][j] A_inv[j]
// with every row resident in UB, so one (i, j) pair is a single 16-lane Axpy
// and the whole solve runs on the vector unit.  The scalar version this
// replaces pushed ~380 values through GetValue/SetValue at ~30 cycles each.
//
// The 120 Axpies of one chunk form a serial chain and a single chain only
// keeps the vector pipe partly busy, so each block solves NCHUNK chunks at
// once and interleaves the chains: one barrier covers the (i, j) step of every
// chunk.  Measured at [1,8192,32]: 2.44 ms scalar -> 1.10 ms vector (NCHUNK=1)
// -> 0.97 ms (NCHUNK=8); NCHUNK=16 is not faster, and the remaining time is
// the per-chunk vector throughput (120 Axpy + 120 coefficients per chunk).
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t MM = M * M;
#ifndef KDA_SOLVE_NCHUNK
#define KDA_SOLVE_NCHUNK 8
#endif
constexpr int32_t NC = KDA_SOLVE_NCHUNK;

extern "C" __global__ __aicore__ void kda_solve_wu_kernel(
    GM_ADDR pL, GM_ADDR pEye, GM_ADDR pA32, GM_ADDR pA16, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> bL, bEye, bAf, bAb;
    pipe.InitBuffer(bL, NC * MM * 4);
    pipe.InitBuffer(bEye, MM * 4);
    pipe.InitBuffer(bAf, NC * MM * 4);
    pipe.InitBuffer(bAb, NC * MM * 2);
    LocalTensor<float> lv = bL.Get<float>();
    LocalTensor<float> eye = bEye.Get<float>();
    LocalTensor<float> af = bAf.Get<float>();
    LocalTensor<bfloat16_t> ab = bAb.Get<bfloat16_t>();
    GlobalTensor<float> L, Eye, A32;
    GlobalTensor<bfloat16_t> A16;
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pEye));
    A32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pA32));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    DataCopy(eye, Eye[0], DataCopyParams(M, 2, 0, 0));
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(lv[ch * MM], L[static_cast<uint64_t>(c0 + ch) * MM], DataCopyParams(M, 2, 0, 0));
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);

    // ---- A_inv starts from the identity ----------------------------------
    // A_inv[i] = e_i - sum_{j<i} L[i][j] A_inv[j]; the whole row is kept in UB
    // so the subtractive updates below are 16-lane Axpies.
    for (int32_t ch = 0; ch < nch; ++ch) {
        Adds(af[ch * MM], eye, 0.0f, MM);
    }
    PipeBarrier<PIPE_V>();

    // Row i only accumulates into its own row and rows j < i are final by the
    // time they are read, so the coefficients come straight from the (never
    // written) L tile.  The chunks are independent: one barrier per (i, j)
    // step covers all of them.
    for (int32_t i = 1; i < M; ++i) {
        for (int32_t j = 0; j < i; ++j) {
            for (int32_t ch = 0; ch < nch; ++ch) {
                const float cij = lv[ch * MM].GetValue(i * M + j);
                Axpy(af[ch * MM + i * M], af[ch * MM + j * M], -cij, M);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    for (int32_t ch = 0; ch < nch; ++ch) {
        Cast(ab[ch * MM], af[ch * MM], RoundMode::CAST_RINT, MM);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t ch = 0; ch < nch; ++ch) {
        Cast(af[ch * MM], ab[ch * MM], RoundMode::CAST_NONE, MM);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(A32[static_cast<uint64_t>(c0 + ch) * MM], af[ch * MM], DataCopyParams(M, 2, 0, 0));
        DataCopy(A16[static_cast<uint64_t>(c0 + ch) * MM], ab[ch * MM], DataCopyParams(M, 1, 0, 0));
    }
    PipeBarrier<PIPE_ALL>();
}
