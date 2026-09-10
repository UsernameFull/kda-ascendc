// K1 stage 3: 16x16 forward-substitution solve plus the w/u right-hand sides.
//
//   A_inv = (I + strict_lower(Akk))^{-1}      (Akk arrives as L)
//   w = bf16(A_inv) @ (k * beta * exp2(gate))
//   u = bf16(A_inv) @ (v * beta)
//
// The solve runs in scalar registers (256 fp32 values); w/u are then formed
// on the vector unit with a per-row broadcast of A_inv and a column reduction,
// so no Cube engine is required.  One AIV block per (batch, head, chunk).
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t N = M * D;
constexpr int32_t MM = M * M;

// out[0:128] = sum_j tile[j, 0:128]
//
// The fp32 vector mask covers 64 elements per repeat, so every tree level has
// to be issued twice: once for columns 0:64 and once for 64:128.
static __aicore__ inline void ColSum(LocalTensor<float> tile) {
    Add(tile, tile, tile[8 * D], 64, 8, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    Add(tile[64], tile[64], tile[8 * D + 64], 64, 8, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[4 * D], 64, 4, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    Add(tile[64], tile[64], tile[4 * D + 64], 64, 4, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[2 * D], 64, 2, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    Add(tile[64], tile[64], tile[2 * D + 64], 64, 2, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[D], 64, 1, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    Add(tile[64], tile[64], tile[D + 64], 64, 1, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
}

// t0[j, 0:64]  += ab[j] * r[j, 0:64]
// t0[j, 64:128]+= ab[j] * r[j, 64:128]
static __aicore__ inline void ScaleRows(LocalTensor<float> t0, const LocalTensor<float> ab,
                                        const LocalTensor<float> r) {
    Mul(t0, ab, r, 64, M, BinaryRepeatParams(1, 1, 1, 16, 8, 16));
    Mul(t0[64], ab, r[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 8, 16));
}

// Sum the 16 rows of a 16x16 fp32 tile into its first row (one block per row).
static __aicore__ inline void ColSum16(LocalTensor<float> tile) {
    Add(tile, tile, tile[8 * M], 16, 8, BinaryRepeatParams(1, 1, 1, 2, 2, 2));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[4 * M], 16, 4, BinaryRepeatParams(1, 1, 1, 2, 2, 2));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[2 * M], 16, 2, BinaryRepeatParams(1, 1, 1, 2, 2, 2));
    PipeBarrier<PIPE_V>();
    Add(tile, tile, tile[M], 16, 1, BinaryRepeatParams(1, 1, 1, 2, 2, 2));
    PipeBarrier<PIPE_V>();
}

// Both right-hand sides share the broadcast of the A_inv row, so W and U are
// formed in the same pass (halves the scalar broadcast work of MatVec).
static __aicore__ inline void MatVec2(LocalTensor<bfloat16_t> outw, LocalTensor<bfloat16_t> outu,
                                      LocalTensor<float> t0, LocalTensor<float> t1,
                                      const LocalTensor<float> ab, const LocalTensor<float> rk,
                                      const LocalTensor<float> rv, const LocalTensor<float> af) {
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < M; ++j) {
            Duplicate(ab[j * 64], af.GetValue(i * M + j), 64);
        }
        PipeBarrier<PIPE_V>();
        ScaleRows(t0, ab, rk);
        ScaleRows(t1, ab, rv);
        PipeBarrier<PIPE_V>();
        ColSum(t0);
        ColSum(t1);
        Cast(outw[i * D], t0, RoundMode::CAST_RINT, D);
        Cast(outu[i * D], t1, RoundMode::CAST_RINT, D);
        PipeBarrier<PIPE_V>();
    }
}

extern "C" __global__ __aicore__ void kda_solve_wu_kernel(
    GM_ADDR pL, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pA32, GM_ADDR pA16,
    GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c = GetBlockIdx();
    if (c >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID sv = pipe.AllocEventID<HardEvent::S_V>();
    TBuf<TPosition::VECCALC> bL, bRkf, bRvf, bAb, bT0, bT1, bA16f, bWb, bUb, bRkb,
        bRvb, bA16b;
    pipe.InitBuffer(bL, MM * 4);
    pipe.InitBuffer(bT1, N * 4);
    pipe.InitBuffer(bRkf, N * 4); pipe.InitBuffer(bRvf, N * 4);
    pipe.InitBuffer(bAb, M * 64 * 4); pipe.InitBuffer(bT0, N * 4);
    pipe.InitBuffer(bA16f, MM * 4); pipe.InitBuffer(bA16b, MM * 2);
    pipe.InitBuffer(bWb, N * 2); pipe.InitBuffer(bUb, N * 2);
    pipe.InitBuffer(bRkb, N * 2); pipe.InitBuffer(bRvb, N * 2);
    LocalTensor<float> lv = bL.Get<float>();
    LocalTensor<float> rkf = bRkf.Get<float>(), rvf = bRvf.Get<float>();
    LocalTensor<float> ab = bAb.Get<float>(), t0 = bT0.Get<float>(), t1 = bT1.Get<float>();
    LocalTensor<float> a16f = bA16f.Get<float>();
    LocalTensor<bfloat16_t> a16b = bA16b.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> wb = bWb.Get<bfloat16_t>(), ub = bUb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> rkb = bRkb.Get<bfloat16_t>(), rvb = bRvb.Get<bfloat16_t>();
    GlobalTensor<float> L, A32;
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    A32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pA32));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t m0 = static_cast<uint64_t>(c) * MM;

    DataCopy(lv, L[m0], DataCopyParams(M, 2, 0, 0));
    DataCopy(rkb, Rk[x0], DataCopyParams(M, 8, 0, 0));
    DataCopy(rvb, Rv[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);

    // ---- forward substitution: Ai = (I + L)^{-1} -------------------------
    float aiv[MM];
    for (int32_t k = 0; k < MM; ++k) {
        const int32_t i = k / M, j = k % M;
        aiv[k] = (j < i) ? -lv.GetValue(k) : 0.0f;
    }
    for (int32_t i = 2; i < M; ++i) {
        float arow[M];
        float raw[M];
        for (int32_t j = 0; j < M; ++j) {
            raw[j] = (j < i) ? -lv.GetValue(i * M + j) : 0.0f;
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
    Cast(rkf, rkb, RoundMode::CAST_NONE, N);
    Cast(rvf, rvb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    MatVec2(wb, ub, t0, t1, ab, rkf, rvf, a16f);

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
    DataCopy(A32[m0], a16f, DataCopyParams(M, 2, 0, 0));
    DataCopy(A16[m0], a16b, DataCopyParams(M, 1, 0, 0));
    DataCopy(W[x0], wb, DataCopyParams(M, 8, 0, 0));
    DataCopy(U[x0], ub, DataCopyParams(M, 8, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
