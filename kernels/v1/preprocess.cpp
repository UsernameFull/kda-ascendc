// K1 stage 1: input normalisation + gate/decay preparation.
//
// One AIV block per (batch, head, chunk).  Produces, in the packed
// ``c = (b * H + h) * NT + chunk`` order used by every other v1 kernel:
//   qn, kn  = l2-normalised q/k (bf16, no sqrt(D) scaling)
//   gate    = chunk-local cumsum of  lower_bound * sigmoid(exp(A_log) * (g + bias))
//             carried in the log2 domain (x RCP_LN2)
//   gc      = gate re-centred on the BT/2 row (used by the gram kernel)
//   beta    = sigmoid(beta)
//   decay   = exp2(gate_last)  (per-chunk state decay, fp32)
//   qg      = qn * exp2(gate)              (output term)
//   rk      = kn * beta * exp2(gate)       (w right-hand side)
//   rv      = v  * beta                    (u right-hand side)
//   kg      = kn * exp2(gate_last - gate)  (state update)
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t N = M * D;
constexpr float RCP_LN2 = 1.4426950216f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float EPS = 1e-6f;

// Row-wise sum of a [M, D] fp32 tile: rs[i] holds sum_d tile[i, d].
// The first Add halves every row in place (strided repeats keep each row's
// data inside its own 128-element slot); WholeReduceSum then collapses the
// remaining 64 elements per row.
static __aicore__ inline void RowReduce(LocalTensor<float> rs, LocalTensor<float> tmp,
                                        const LocalTensor<float> tile) {
    Add(tmp, tile, tile[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    WholeReduceSum(rs, tmp, 64, M, 1, 1, 16);
}

extern "C" __global__ __aicore__ void kda_preprocess_kernel(
    GM_ADDR pQ, GM_ADDR pK, GM_ADDR pV, GM_ADDR pG, GM_ADDR pBeta,
    GM_ADDR pAlog, GM_ADDR pBias,
    GM_ADDR pQn, GM_ADDR pKn, GM_ADDR pGate, GM_ADDR pGc, GM_ADDR pBetaOut,
    GM_ADDR pDecay, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pQg, GM_ADDR pKg,
    int32_t B, int32_t T, int32_t H, float lower_bound) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t nt = T / M;
    const int32_t c = GetBlockIdx();
    if (c >= B * H * nt) return;
    const int32_t bh = c / nt;
    const int32_t head = bh % H;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID evs = pipe.AllocEventID<HardEvent::V_S>();
    TBuf<TPosition::VECCALC> bQf, bKf, bT0, bT2, bEf, bRed,
        bQnb, bKnb, bRkb, bRvb, bQgb, bKgb, bBias, bBeta, bBb, bAlog;
    pipe.InitBuffer(bQf, N * 4); pipe.InitBuffer(bKf, N * 4);
    pipe.InitBuffer(bT0, N * 4); pipe.InitBuffer(bT2, N * 4);
    pipe.InitBuffer(bEf, N * 4); pipe.InitBuffer(bRed, 384 * 4);
    pipe.InitBuffer(bQnb, N * 2); pipe.InitBuffer(bKnb, N * 2); pipe.InitBuffer(bRkb, N * 2);
    pipe.InitBuffer(bRvb, N * 2); pipe.InitBuffer(bQgb, N * 2); pipe.InitBuffer(bKgb, N * 2);
    pipe.InitBuffer(bBias, D * 4); pipe.InitBuffer(bBeta, M * 4);
    pipe.InitBuffer(bBb, M * 64 * 4); pipe.InitBuffer(bAlog, 8 * 4);
    LocalTensor<float> qf = bQf.Get<float>(), kf = bKf.Get<float>();
    LocalTensor<float> gf = bT0.Get<float>(), t2 = bT2.Get<float>();
    LocalTensor<float> ef = bEf.Get<float>(), red = bRed.Get<float>();
    LocalTensor<float> bias = bBias.Get<float>(), beta = bBeta.Get<float>();
    LocalTensor<float> bb = bBb.Get<float>(), alog = bAlog.Get<float>();
    LocalTensor<bfloat16_t> qnb = bQnb.Get<bfloat16_t>(), knb = bKnb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> rkb = bRkb.Get<bfloat16_t>(), rvb = bRvb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> qgb = bQgb.Get<bfloat16_t>(), kgb = bKgb.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> Q, K, V, Qn, Kn, Rk, Rv, Qg, Kg;
    GlobalTensor<float> G, Beta, Alog, Bias, Gate, Gc, BetaOut, Decay;
    Q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQ));
    K.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pK));
    V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pV));
    Qn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQn));
    Kn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKn));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    G.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pG));
    Beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBeta));
    Alog.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAlog));
    Bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBias));
    Gate.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pGate));
    Gc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pGc));
    BetaOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBetaOut));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));

    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t cm = static_cast<uint64_t>(c) * M;

    DataCopy(alog, Alog[head], 8);
    DataCopy(beta, Beta[cm], DataCopyParams(1, 2, 0, 0));
    DataCopy(qnb, Q[x0], DataCopyParams(M, 8, 0, 0));
    DataCopy(gf, G[x0], DataCopyParams(M, 16, 0, 0));
    DataCopy(rvb, V[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(qf, qnb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- q l2 norm -------------------------------------------------------
    Mul(t2, qf, qf, N);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    Adds(red, red, EPS, M);
    Rsqrt(red, red, M);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    for (int32_t i = 0; i < M; ++i) {
        Muls(qf[i * D], qf[i * D], red.GetValue(i), D);
    }
    PipeBarrier<PIPE_V>();
    Cast(qnb, qf, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();
    Cast(qf, qnb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- k l2 norm -------------------------------------------------------
    DataCopy(knb, K[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(kf, knb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, kf, N);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    Adds(red, red, EPS, M);
    Rsqrt(red, red, M);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    for (int32_t i = 0; i < M; ++i) {
        Muls(kf[i * D], kf[i * D], red.GetValue(i), D);
    }
    PipeBarrier<PIPE_V>();
    Cast(knb, kf, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();
    Cast(kf, knb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- beta sigmoid ----------------------------------------------------
    Muls(beta, beta, -1.0f, M);
    Exp(beta, beta, M);
    Adds(beta, beta, 1.0f, M);
    Duplicate(red, 1.0f, M);
    PipeBarrier<PIPE_V>();
    Div(beta, red, beta, M);
    PipeBarrier<PIPE_V>();

    // ---- gate (cumsum carried in the log2 domain) ------------------------
    if (pBias != nullptr) {
        DataCopy(bias, Bias[head * D], DataCopyParams(1, 16, 0, 0));
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        Add(gf, gf, bias, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        Add(gf[64], gf[64], bias[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        PipeBarrier<PIPE_V>();
    }
    Exp(alog, alog, 8);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    const float aexp = alog.GetValue(0);
    Muls(gf, gf, aexp, N);
    Muls(gf, gf, -1.0f, N);
    Exp(gf, gf, N);
    Adds(gf, gf, 1.0f, N);
    Duplicate(t2, 1.0f, N);
    PipeBarrier<PIPE_V>();
    Div(gf, t2, gf, N);
    Muls(gf, gf, lower_bound, N);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 1; i < M; ++i) {
        Add(gf[i * D], gf[i * D], gf[(i - 1) * D], D);
        PipeBarrier<PIPE_V>();
    }
    Muls(gf, gf, RCP_LN2, N);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Gate[x0], gf, DataCopyParams(M, 16, 0, 0));
    PipeBarrier<PIPE_ALL>();

    // ---- decay = exp2(gate_last) ----------------------------------------
    Muls(t2, gf[15 * D], LN2, D);
    Exp(t2, t2, D);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Decay[static_cast<uint64_t>(c) * D], t2, DataCopyParams(1, 16, 0, 0));
    PipeBarrier<PIPE_ALL>();

    // ---- gc = gate - gate[mid] ------------------------------------------
    for (int32_t i = 0; i < M; ++i) {
        Sub(t2[i * D], gf[i * D], gf[8 * D], D);
    }
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Gc[x0], t2, DataCopyParams(M, 16, 0, 0));
    PipeBarrier<PIPE_ALL>();

    // ---- exp2(gate) ------------------------------------------------------
    Muls(ef, gf, LN2, N);
    Exp(ef, ef, N);
    PipeBarrier<PIPE_V>();

    // ---- beta broadcast over the D axis ---------------------------------
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    for (int32_t i = 0; i < M; ++i) {
        Duplicate(bb[i * 64], beta.GetValue(i), 64);
    }
    PipeBarrier<PIPE_V>();

    // ---- qg = qn * exp2(gate) --------------------------------------------
    Mul(t2, qf, ef, N);
    PipeBarrier<PIPE_V>();
    Cast(qgb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- rk = kn * beta * exp2(gate) -------------------------------------
    Mul(t2, kf, ef, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bb, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 8));
    Mul(t2[64], t2[64], bb, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 8));
    PipeBarrier<PIPE_V>();
    Cast(rkb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- rv = v * beta ---------------------------------------------------
    Cast(t2, rvb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bb, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 8));
    Mul(t2[64], t2[64], bb, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 8));
    PipeBarrier<PIPE_V>();
    Cast(rvb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- kg = kn * exp2(gate_last - gate) --------------------------------
    for (int32_t i = 0; i < M; ++i) {
        Sub(t2[i * D], gf[15 * D], gf[i * D], D);
    }
    PipeBarrier<PIPE_V>();
    Muls(t2, t2, LN2, N);
    Exp(t2, t2, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, kf, N);
    PipeBarrier<PIPE_V>();
    Cast(kgb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- stores ----------------------------------------------------------
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Qn[x0], qnb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Kn[x0], knb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Qg[x0], qgb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Kg[x0], kgb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Rk[x0], rkb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Rv[x0], rvb, DataCopyParams(M, 8, 0, 0));
    DataCopy(BetaOut[cm], beta, DataCopyParams(1, 2, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
