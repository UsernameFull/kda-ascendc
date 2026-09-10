// K1 stage 2: intra-chunk Gram matrices.
//
//   Aqk[i,j] = scale * sum_d (qn[i,d] e^{gc[i,d]}) * (kn[j,d] e^{-gc[j,d]})
//   Akk[i,j] = beta_i * sum_d (kn[i,d] e^{gc[i,d]}) * (kn[j,d] e^{-gc[j,d]})
//
// Aqk is masked to i >= j, Akk (exported as L) to i > j.  gc is the gate
// recentred on the chunk mid row, so the exponents stay near zero.
// One AIV block per (batch, head, chunk); the reduction runs on the vector
// unit row by row (WholeReduceSum over the contiguous head dimension).
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t N = M * D;
constexpr float LN2 = 0.6931471805599453f;

// rs[j] = sum_d tile[j, d]; tmp is an N-float scratch tile.
static __aicore__ inline void RowDot(LocalTensor<float> rs, LocalTensor<float> tmp,
                                     const LocalTensor<float> tile) {
    Add(tmp, tile, tile[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    WholeReduceSum(rs, tmp, 64, M, 1, 1, 16);
}

// t0[j, 0:64]  = a[0:64]  * b[j, 0:64]
// t0[j, 64:128]= a[64:128]* b[j, 64:128]
static __aicore__ inline void RowBroadcastMul(LocalTensor<float> t0, const LocalTensor<float> a,
                                              const LocalTensor<float> b) {
    Mul(t0, a, b, 64, M, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    Mul(t0[64], a[64], b[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
}

extern "C" __global__ __aicore__ void kda_gram_kernel(
    GM_ADDR pQn, GM_ADDR pKn, GM_ADDR pGc, GM_ADDR pBeta,
    GM_ADDR pAqk32, GM_ADDR pAqk16, GM_ADDR pL, GM_ADDR pMaskS, GM_ADDR pMaskL,
    int32_t C, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c = GetBlockIdx();
    if (c >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID sv = pipe.AllocEventID<HardEvent::S_V>();
    TBuf<TPosition::VECCALC> bQf, bKf, bEf, bEfn, bA, bK1, bB, bT0,
        bA32, bL32, bA16, bBeta, bQnb, bKnb, bRedA, bRedK, bMaskS, bMaskL, bTb;
    pipe.InitBuffer(bQf, N * 4); pipe.InitBuffer(bKf, N * 4);
    pipe.InitBuffer(bEf, N * 4); pipe.InitBuffer(bEfn, N * 4);
    pipe.InitBuffer(bA, N * 4); pipe.InitBuffer(bK1, N * 4); pipe.InitBuffer(bB, N * 4);
    pipe.InitBuffer(bT0, N * 4);
    pipe.InitBuffer(bRedA, M * M * 4); pipe.InitBuffer(bRedK, M * M * 4);
    pipe.InitBuffer(bMaskS, M * M * 4); pipe.InitBuffer(bMaskL, M * M * 4);
    pipe.InitBuffer(bTb, D * 4);
    pipe.InitBuffer(bA32, M * M * 4); pipe.InitBuffer(bL32, M * M * 4);
    pipe.InitBuffer(bA16, M * M * 2);
    pipe.InitBuffer(bBeta, M * 4); pipe.InitBuffer(bQnb, N * 2); pipe.InitBuffer(bKnb, N * 2);
    LocalTensor<float> qf = bQf.Get<float>(), kf = bKf.Get<float>();
    LocalTensor<float> ef = bEf.Get<float>(), efn = bEfn.Get<float>();
    LocalTensor<float> a = bA.Get<float>(), k1 = bK1.Get<float>(), b = bB.Get<float>();
    LocalTensor<float> t0 = bT0.Get<float>();
    LocalTensor<float> redA = bRedA.Get<float>(), redK = bRedK.Get<float>();
    LocalTensor<float> maskS = bMaskS.Get<float>(), maskL = bMaskL.Get<float>();
    LocalTensor<float> tb = bTb.Get<float>();
    LocalTensor<float> a32 = bA32.Get<float>(), l32 = bL32.Get<float>();
    LocalTensor<bfloat16_t> a16 = bA16.Get<bfloat16_t>();
    LocalTensor<float> beta = bBeta.Get<float>();
    LocalTensor<bfloat16_t> qnb = bQnb.Get<bfloat16_t>(), knb = bKnb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Qn, Kn, Aqk16;
    GlobalTensor<float> Gc, Beta, Aqk32, L, MaskS, MaskL;
    Qn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQn));
    Kn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKn));
    Aqk16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk16));
    Gc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pGc));
    Beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBeta));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    MaskS.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskS));
    MaskL.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskL));
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;

    DataCopy(qnb, Qn[x0], DataCopyParams(M, 8, 0, 0));
    DataCopy(knb, Kn[x0], DataCopyParams(M, 8, 0, 0));
    DataCopy(ef, Gc[x0], DataCopyParams(M, 16, 0, 0));
    DataCopy(beta, Beta[static_cast<uint64_t>(c) * M], DataCopyParams(1, 2, 0, 0));
    DataCopy(maskS, MaskS[0], DataCopyParams(M, 2, 0, 0));
    DataCopy(maskL, MaskL[0], DataCopyParams(M, 2, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(qf, qnb, RoundMode::CAST_NONE, N);
    Cast(kf, knb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    // exp2(gc) and exp2(-gc) computed from the same raw input, matching the
    // reference which evaluates both exponents independently.
    Muls(ef, ef, LN2, N);
    Exp(ef, ef, N);
    PipeBarrier<PIPE_V>();
    DataCopy(t0, Gc[x0], DataCopyParams(M, 16, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Muls(t0, t0, -LN2, N);
    Exp(efn, t0, N);
    PipeBarrier<PIPE_V>();
    Mul(a, qf, ef, N);
    Mul(k1, kf, ef, N);
    Mul(b, kf, efn, N);
    PipeBarrier<PIPE_V>();

    // Row i of the Gram matrix lands in row i of redA/redK, so the triangular
    // masks are applied with a single elementwise multiply instead of 512
    // scalar SetValue/GetValue pairs per chunk.
    for (int32_t i = 0; i < M; ++i) {
        RowBroadcastMul(t0, a[i * D], b);
        PipeBarrier<PIPE_V>();
        RowDot(redA[i * M], ef, t0);
        PipeBarrier<PIPE_V>();
        Muls(tb, k1[i * D], beta.GetValue(i), D);
        PipeBarrier<PIPE_V>();
        RowBroadcastMul(t0, tb, b);
        PipeBarrier<PIPE_V>();
        RowDot(redK[i * M], ef, t0);
        PipeBarrier<PIPE_V>();
    }
    Muls(redA, redA, scale, M * M);
    PipeBarrier<PIPE_V>();
    Mul(a32, redA, maskS, M * M);
    Mul(l32, redK, maskL, M * M);
    PipeBarrier<PIPE_V>();
    Cast(a16, a32, RoundMode::CAST_RINT, M * M);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
    DataCopy(Aqk32[m0], a32, DataCopyParams(M, 2, 0, 0));
    DataCopy(L[m0], l32, DataCopyParams(M, 2, 0, 0));
    DataCopy(Aqk16[m0], a16, DataCopyParams(M, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
