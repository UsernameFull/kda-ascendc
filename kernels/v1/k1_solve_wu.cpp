// K1 stage 3: 16x16 forward-substitution solve.
//
//   A_inv = (I + strict_lower(Akk))^{-1}      (Akk arrives as L)
//
// The solve runs in scalar registers (256 fp32 values) and emits the fp32
// A_inv together with its bf16 rounding.  The two right-hand sides
// w = A16 @ rk and u = A16 @ rv are formed on the Cube by
// kda_solve_wu_cube_kernel, which is much cheaper than the vector row
// broadcast plus column reduction this kernel used to run.
// One AIV block per (batch, head, chunk).
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t MM = M * M;

extern "C" __global__ __aicore__ void kda_solve_wu_kernel(
    GM_ADDR pL, GM_ADDR pA32, GM_ADDR pA16, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c = GetBlockIdx();
    if (c >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID sv = pipe.AllocEventID<HardEvent::S_V>();
    TBuf<TPosition::VECCALC> bL, bA16f, bA16b;
    pipe.InitBuffer(bL, MM * 4);
    pipe.InitBuffer(bA16f, MM * 4);
    pipe.InitBuffer(bA16b, MM * 2);
    LocalTensor<float> lv = bL.Get<float>();
    LocalTensor<float> a16f = bA16f.Get<float>();
    LocalTensor<bfloat16_t> a16b = bA16b.Get<bfloat16_t>();
    GlobalTensor<float> L, A32;
    GlobalTensor<bfloat16_t> A16;
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    A32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pA32));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    const uint64_t m0 = static_cast<uint64_t>(c) * MM;

    DataCopy(lv, L[m0], DataCopyParams(M, 2, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);

    // ---- forward substitution: Ai = (I + L)^{-1} -------------------------
    // Only the strict lower triangle is read; the rest of aiv stays zero.  The
    // scalar UB reads (GetValue) are the expensive part of this kernel, so read
    // each entry once: the coefficients of row i are exactly aiv[i][j<i], which
    // the first loop already loaded and nothing has overwritten yet.
    float aiv[MM];
    for (int32_t k = 0; k < MM; ++k) aiv[k] = 0.0f;
    for (int32_t i = 1; i < M; ++i) {
        for (int32_t j = 0; j < i; ++j) aiv[i * M + j] = -lv.GetValue(i * M + j);
    }
    for (int32_t i = 2; i < M; ++i) {
        float arow[M];
        float raw[M];
        for (int32_t j = 0; j < M; ++j) {
            raw[j] = (j < i) ? aiv[i * M + j] : 0.0f;
            arow[j] = raw[j];
        }
        for (int32_t j = 0; j < i; ++j) {
            // The substitution coefficients are the raw row entries; the
            // accumulating arow must not feed back into later coefficients.
            const float aj = raw[j];
            if (aj == 0.0f) continue;
            for (int32_t k = 0; k < M; ++k) {
                arow[k] += aj * aiv[j * M + k];
            }
        }
        for (int32_t k = 0; k < M; ++k) {
            aiv[i * M + k] = arow[k];
        }
    }
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < M; ++j) {
            a16f.SetValue(i * M + j, aiv[i * M + j] + (i == j ? 1.0f : 0.0f));
        }
    }
    SetFlag<HardEvent::S_V>(sv);
    WaitFlag<HardEvent::S_V>(sv);
    Cast(a16b, a16f, RoundMode::CAST_RINT, MM);
    PipeBarrier<PIPE_V>();
    Cast(a16f, a16b, RoundMode::CAST_NONE, MM);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
    DataCopy(A32[m0], a16f, DataCopyParams(M, 2, 0, 0));
    DataCopy(A16[m0], a16b, DataCopyParams(M, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
