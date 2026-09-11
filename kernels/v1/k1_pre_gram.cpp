// K1 stages 1+2 fused: input normalisation, gate/decay preparation and the
// intra-chunk Gram matrices in one AIV block per (batch, head, chunk).
//
// The standalone pair costs 3.7 ms at [1,8192,32] and is DMA bound: the
// preprocess half alone moves 61 KB per chunk (1.0 GB per pass) at ~570 GB/s,
// and a stub that keeps every DataCopy but drops the whole compute chain still
// takes 1.79 of its 1.96 ms.  "Qn"/"Kn"/"Gc" are 32 KB per chunk of that
// traffic and their only consumer is the Gram kernel on the next launch, so
// folding the two stages together deletes the round trip.
//
// The Gram half is unchanged arithmetic and bit-identical to the old
// kernel's "Aqk32"/"Aqk"/"L" outputs: qf/kf here are the same bf16-rounded,
// l2-normalised rows the old kernel loaded back from "Qn"/"Kn", and the
// chunk-centred gate is still the same fp32 value.
//
// The GM copies of Qn/Kn/Gate/Gc are debug-only and skipped when the pointer
// is null ("api.py" passes them only for "return_intermediates").
//
// Per-row scalars (the l2 norms, beta, the chunk-centred gate) are broadcast
// with "Brcb" instead of a 16-iteration "Muls"/"Duplicate" loop: Brcb turns the
// 16 values into a tile with eight 32B blocks per row, and the consumer then
// runs 64 lanes x 16 repeats with "src1RepStride = 1" block (i.e. one scalar
// per repeat; "dstRepStride = 16" blocks walks the 128-lane rows), once for the
// low half and once for the high half.  That is 3 instructions instead of 16
// plus the 16 scalar "GetValue" reads, and it is bit-identical (measured
// max(abs(diff)) = 0.000e+00 on all 13 outputs): 3.114 -> 2.679 ms at
// [1,8192,32] and 1.590 -> 1.373 ms at [1,4096,32].
//
// The kernel is issue-bound, not FLOP-bound (msprof ArithmeticUtilization: the
// fp32 vector ALU is busy 20% of the block, while a micro-benchmark puts a
// fixed ~25-35 cycles on every vector instruction, plus ~1-7 cycles per repeat
// depending on the form).  Three changes follow from that, all bit-identical
// and worth 2.69 -> 2.59 ms at [1,8192,32]:
//   * every input load is issued up front behind its own MTE2->V flag and the
//     wait is deferred to the consumer, so the q/k norms run while G/V/the
//     masks are still in flight (the block used to wait for all five copies);
//   * Qg/Kg/Rk/Rv/BetaOut are stored as soon as they exist, so their MTE3
//     traffic drains behind the Gram loop instead of at the end of the block;
//   * the two "V_S" sync pairs around the l2 norms are dead (nothing reads
//     those reductions with the scalar unit any more) and cost ~2%.
//
// The block walks `unroll` consecutive chunks (the body is byte-identical,
// just wrapped in the loop) because the whole kernel is issue-bound and every
// block pays a fixed setup cost worth ~10% of one chunk's work: 2.679 ->
// 2.342 ms at [1,8192,32] with unroll 8, 0.333 -> 0.308 ms at [1,1024,32]
// with unroll 4, 0.0953 -> 0.0932 ms at [2,1024,4] with unroll 2, all
// bit-identical.  Back-to-back-launch measurements (host launch overhead
// hidden) put the optimum at >= 256 blocks; at chunk counts under 256 the
// loop wrapper itself costs ~6%, so `api.py` leaves those at unroll 1.
//
// A last pass deleted four more instructions per chunk, again bit-identical
// (0.000e+00 on all 13 outputs, 2.289 -> 2.262 ms at [1,8192,32] with the same
// back-to-back harness):
//   * "Muls(gf, gf, aexp)" followed by "Muls(gf, gf, -1)" is one "Muls" with
//     the negated scalar (a sign flip is exact, so this is not a rounding
//     change);
//   * "Muls(t2, t2, -1.0f)" after "gf - gf[15]" is the same as computing
//     "gf[15] - gf" directly - swap the two "Sub" operands *and* their repeat
//     strides (src0 is the fixed row, so its stride is the 0 one);
//   * the A-side triangular mask multiply is a no-op: the Gram loop writes
//     exactly the lower triangle and "Duplicate" already put zeros above it,
//     so "Muls(ga32, redA, scale)" is the whole post-processing - and the
//     "MaskS" tile no longer has to be fetched.  "MaskL" has to stay, because
//     the K-side loop does write L's diagonal and the mask is what zeroes it.
//
// Cost model from probes on this kernel (+8 instructions per chunk, launch
// time at [1,8192,32]): +0.040 ms for 8 one-repeat "Adds" => ~27 cycles per
// vector instruction, +0.056 ms for 8 "Mul"s with 16 repeats => ~0.6 cycles
// per extra repeat, +0.051 ms for 8 one-repeat "Exp" => the special-function
// unit is dearer per repeat but does not dominate.  msprof agrees: the vector
// pipe is busy 70% of this kernel and the instruction count, not the lane
// count, is what it waits on.
//
// Negative results, worth not retrying: dropping the 15 barriers of the gate
// cumsum and/or the 64 of the Gram loop changes nothing (the vector pipe is
// in-order, so they are free); giving the Gram loop a second product tile so
// the A and K halves stop serialising on one buffer changes nothing (the pipe
// is issue-limited, not latency-limited); 256B UB padding on "gb" or "redA"
// costs 4.5% and on "ga"/"gtb" nothing, so the natural layout is already the
// good one; unroll 4/16/32 are all worse than 8; and moving the Gram to the
// Cube would need 24 KB per chunk of extra GM traffic in each direction to
// save ~4.7k of the ~13.7k core cycles per chunk, a wash at this kernel's
// ~370 GB/s.
//
// NOTE: the body below is sensitive to how it is written - an equivalent
// rewrite that only reformats the buffer declarations or moves the
// "PipeBarrier<PIPE_V>" after the gate re-centring loop trips an aivec error
// (mte error info 0x8030860ef, pc offset ~0x3d8) on this CANN runtime.  Keep
// the layout of this file; verified working on 910_9382 / CANN 9.1.0.
//
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

static __aicore__ inline void RowDot(LocalTensor<float> rs, LocalTensor<float> tmp,
                                     uint8_t rows) {
    WholeReduceSum(rs, tmp, 64, rows, 1, 1, 16);
}
static __aicore__ inline void RowBroadcastMul(LocalTensor<float> t0, const LocalTensor<float> a,
                                              const LocalTensor<float> b, uint8_t rows) {
    Mul(t0, a, b, 64, rows, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    MulAddDst(t0, a[64], b[64], 64, rows, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
}
extern "C" __global__ __aicore__ void kda_pre_gram_kernel(
    GM_ADDR pQ, GM_ADDR pK, GM_ADDR pV, GM_ADDR pG, GM_ADDR pBeta,
    GM_ADDR pAlog, GM_ADDR pBias,
    GM_ADDR pQn, GM_ADDR pKn, GM_ADDR pGate, GM_ADDR pGc, GM_ADDR pBetaOut,
    GM_ADDR pDecay, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pQg, GM_ADDR pKg,
    GM_ADDR pAqk32, GM_ADDR pAqk16, GM_ADDR pL, GM_ADDR pMaskS, GM_ADDR pMaskL,
    int32_t B, int32_t T, int32_t H, float lower_bound, float scale, int32_t unroll) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t nt = T / M;
    const int32_t nchunk = B * H * nt;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vq = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vk = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vg = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vv = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vs = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID evs = pipe.AllocEventID<HardEvent::V_S>();
    TBuf<TPosition::VECCALC> bQf, bKf, bT0, bT2, bEf, bRed,
        bQnb, bKnb, bRkb, bRvb, bQgb, bKgb, bBias, bBeta, bBb, bAlog, bZz,
        bGef, bGefn, bGa, bGk1, bGb, bRedA, bRedK, bGmaskS, bGmaskL, bGtb, bGa32, bGl32, bGa16;
    pipe.InitBuffer(bQf, N * 4); pipe.InitBuffer(bKf, N * 4);
    pipe.InitBuffer(bT0, N * 4); pipe.InitBuffer(bT2, N * 4);
    pipe.InitBuffer(bEf, N * 4); pipe.InitBuffer(bRed, 384 * 4);
    pipe.InitBuffer(bQnb, N * 2); pipe.InitBuffer(bKnb, N * 2); pipe.InitBuffer(bRkb, N * 2);
    pipe.InitBuffer(bRvb, N * 2); pipe.InitBuffer(bQgb, N * 2); pipe.InitBuffer(bKgb, N * 2);
    pipe.InitBuffer(bBias, D * 4); pipe.InitBuffer(bBeta, M * 4);
    pipe.InitBuffer(bBb, M * 64 * 4); pipe.InitBuffer(bAlog, 8 * 4);
    pipe.InitBuffer(bZz, N * 4);
    pipe.InitBuffer(bGef, N * 4);
    pipe.InitBuffer(bGefn, N * 4);
    pipe.InitBuffer(bGa, N * 4);
    pipe.InitBuffer(bGk1, N * 4);
    pipe.InitBuffer(bGb, N * 4);
    pipe.InitBuffer(bRedA, M * M * 4);
    pipe.InitBuffer(bRedK, M * M * 4);
    pipe.InitBuffer(bGmaskS, M * M * 4);
    pipe.InitBuffer(bGmaskL, M * M * 4);
    pipe.InitBuffer(bGtb, D * 4);
    pipe.InitBuffer(bGa32, M * M * 4);
    pipe.InitBuffer(bGl32, M * M * 4);
    pipe.InitBuffer(bGa16, M * M * 2);
    LocalTensor<float> qf = bQf.Get<float>(), kf = bKf.Get<float>();
    LocalTensor<float> gf = bT0.Get<float>(), t2 = bT2.Get<float>();
    LocalTensor<float> ef = bEf.Get<float>(), red = bRed.Get<float>();
    LocalTensor<float> bias = bBias.Get<float>(), beta = bBeta.Get<float>();
    LocalTensor<float> bb = bBb.Get<float>(), alog = bAlog.Get<float>();
    LocalTensor<float> zz = bZz.Get<float>();
    LocalTensor<float> gef = bGef.Get<float>(), gefn = bGefn.Get<float>();
    LocalTensor<float> ga = bGa.Get<float>(), gk1 = bGk1.Get<float>(), gb = bGb.Get<float>();
    LocalTensor<float> redA = bRedA.Get<float>(), redK = bRedK.Get<float>();
    LocalTensor<float> gmaskS = bGmaskS.Get<float>(), gmaskL = bGmaskL.Get<float>();
    LocalTensor<float> gtb = bGtb.Get<float>();
    LocalTensor<float> ga32 = bGa32.Get<float>(), gl32 = bGl32.Get<float>();
    LocalTensor<bfloat16_t> ga16 = bGa16.Get<bfloat16_t>();

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

    GlobalTensor<bfloat16_t> Aqk16;
    GlobalTensor<float> Aqk32, L, MaskS, MaskL;
    Aqk16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk16));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    MaskS.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskS));
    MaskL.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskL));
    // One block walks `unroll` consecutive chunks: the kernel is issue-bound
    // (~40 cycles per vector instruction) and pays a fixed per-block cost that
    // is a tenth of a block's work, so amortizing it over 2-8 chunks is worth
    // 2.68 -> 2.34 ms at [1,8192,32] (unroll 8) with the body unchanged and
    // bit-identical output.  `api.py` picks the unroll from the chunk count.
    for (int32_t u = 0; u < unroll; ++u) {
    const int32_t c = GetBlockIdx() * unroll + u;
    if (c >= nchunk) break;
    const int32_t bh = c / nt;
    const int32_t head = bh % H;
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t cm = static_cast<uint64_t>(c) * M;

    DataCopy(qnb, Q[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vq);
    DataCopy(knb, K[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vk);
    DataCopy(gf, G[x0], DataCopyParams(M, 16, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vg);
    DataCopy(rvb, V[x0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vv);
    DataCopy(alog, Alog[head], 8);
    DataCopy(beta, Beta[cm], DataCopyParams(1, 2, 0, 0));
    DataCopy(gmaskL, MaskL[0], DataCopyParams(M, 2, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vs);
    WaitFlag<HardEvent::MTE2_V>(e2vq);
    Cast(qf, qnb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- q l2 norm -------------------------------------------------------
    Mul(t2, qf, qf, N);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, M);
    Rsqrt(red, red, M);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(qf, qf, red[64], 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(qf[64], qf[64], red[64], 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(qnb, qf, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();
    Cast(qf, qnb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- k l2 norm -------------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vk);
    Cast(kf, knb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, kf, N);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, M);
    Rsqrt(red, red, M);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(kf, kf, red[64], 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(kf[64], kf[64], red[64], 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(knb, kf, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();
    Cast(kf, knb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();

    // ---- beta sigmoid ----------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vs);
    WaitFlag<HardEvent::MTE2_V>(e2vg);
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
    const float aexp = -alog.GetValue(0);
    Muls(gf, gf, aexp, N);
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
    if (pGate != nullptr) {
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gate[x0], gf, DataCopyParams(M, 16, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }

    // ---- decay = exp2(gate_last) ----------------------------------------
    Muls(t2, gf[15 * D], LN2, D);
    Exp(t2, t2, D);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Decay[static_cast<uint64_t>(c) * D], t2, DataCopyParams(1, 16, 0, 0));
    PipeBarrier<PIPE_ALL>();

    // ---- gc = gate - gate[mid] ------------------------------------------
    Sub(zz, gf, gf[8 * D], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Sub(zz[64], gf[64], gf[8 * D + 64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    if (pGc != nullptr) {
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gc[x0], zz, DataCopyParams(M, 16, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }

    // ---- exp2(gate) ------------------------------------------------------
    Muls(ef, gf, LN2, N);
    Exp(ef, ef, N);
    PipeBarrier<PIPE_V>();

    // ---- beta broadcast over the D axis ---------------------------------
    PipeBarrier<PIPE_V>();
    Brcb(bb, beta, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();

    // ---- qg = qn * exp2(gate) --------------------------------------------
    Mul(t2, qf, ef, N);
    PipeBarrier<PIPE_V>();
    Cast(qgb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- rk = kn * beta * exp2(gate) -------------------------------------
    Mul(t2, kf, ef, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rkb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- rv = v * beta ---------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vv);
    Cast(t2, rvb, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rvb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();

    // ---- kg = kn * exp2(gate_last - gate) --------------------------------
    Sub(t2, gf[15 * D], gf, 64, M, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    Sub(t2[64], gf[15 * D + 64], gf[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    PipeBarrier<PIPE_V>();
    Muls(t2, t2, LN2, N);
    Exp(t2, t2, N);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, kf, N);
    PipeBarrier<PIPE_V>();
    Cast(kgb, t2, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();


    // early stores: let MTE3 drain behind the Gram work
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Qg[x0], qgb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Kg[x0], kgb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Rk[x0], rkb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Rv[x0], rvb, DataCopyParams(M, 8, 0, 0));
    DataCopy(BetaOut[cm], beta, DataCopyParams(1, 2, 0, 0));

    // ---- Gram half (prep only) ----
    Muls(gef, zz, LN2, N);
    Exp(gef, gef, N);
    Muls(gefn, zz, -LN2, N);
    Exp(gefn, gefn, N);
    PipeBarrier<PIPE_V>();
    Mul(ga, qf, gef, N);
    Mul(gk1, kf, gef, N);
    Mul(gb, kf, gefn, N);
    PipeBarrier<PIPE_V>();
    Mul(gk1, gk1, bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(gk1[64], gk1[64], bb, 64, M, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Duplicate(redA, 0.0f, M * M);
    Duplicate(redK, 0.0f, M * M);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < M; ++i) {
        const uint8_t rows = static_cast<uint8_t>(i + 1);
        RowBroadcastMul(gtb, ga[i * D], gb, rows);
        PipeBarrier<PIPE_V>();
        RowDot(redA[i * M], gtb, rows);
        PipeBarrier<PIPE_V>();
        RowBroadcastMul(gtb, gk1[i * D], gb, rows);
        PipeBarrier<PIPE_V>();
        RowDot(redK[i * M], gtb, rows);
        PipeBarrier<PIPE_V>();
    }
    Muls(ga32, redA, scale, M * M);
    Mul(gl32, redK, gmaskL, M * M);
    PipeBarrier<PIPE_V>();
    Cast(ga16, ga32, RoundMode::CAST_RINT, M * M);
    PipeBarrier<PIPE_V>();
    // ---- stores ----------------------------------------------------------
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    if (pQn != nullptr) DataCopy(Qn[x0], qnb, DataCopyParams(M, 8, 0, 0));
    if (pKn != nullptr) DataCopy(Kn[x0], knb, DataCopyParams(M, 8, 0, 0));
    DataCopy(Aqk32[m0], ga32, DataCopyParams(M, 2, 0, 0));
    DataCopy(L[m0], gl32, DataCopyParams(M, 2, 0, 0));
    DataCopy(Aqk16[m0], ga16, DataCopyParams(M, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
    }
}
