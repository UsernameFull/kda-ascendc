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
// The one "PipeBarrier<PIPE_ALL>" left on the fast path (the "Gate"/"Gc" ones
// only run when the caller asks for intermediates) guards the "Decay" store:
// "t2" is reused by the "qg" product a few instructions later, so only the
// MTE3->V half of it is needed - "SetFlag<HardEvent::MTE3_V>" after the copy,
// "WaitFlag<HardEvent::MTE3_V>" before the "Mul" that overwrites the tile.
// 2.259 -> 2.209 ms at [1,8192,32] (R=10, unroll 8, bit-identical), and
// in-pipeline "pre_gram" 2.285 -> 2.237 ms of a 7.463 -> 7.433 ms pass.
//
// The tail "PipeBarrier<PIPE_ALL>" is the bigger prize (deleting it is 2.134
// ms) but it has no flag formulation that survives this runtime: a
// loop-carried "SetFlag"/"WaitFlag" pair - MTE3->MTE2, or MTE3->V waited at
// the top of the body or at the first stored-tile write - hangs the *first*
// launch, even though the same pattern runs in a toy micro-kernel, the
// in-body pairs above are fine and the compiled code is the same size.  That
// one needs the loads and the stores to stop sharing UB, i.e. a double
// buffered TQue, not a flag.  See docs/ASCENDC_V1_KERNELS.md.
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
// good one; unroll 4/16/32 are all worse than 8.  (An earlier revision of this
// note priced "move the Gram to the Cube" as a wash at this kernel's own
// ~370 GB/s; that was wrong - the publishes ride an idle MTE3, see below.)
//
// ---------------------------------------------------------------------------
// MIX variant (this file): the two intra-chunk Grams move to the paired Cube.
//
// A stub of the vector-only block with just the 32-iteration Gram loop deleted
// runs 1.158 ms against 2.205 ms for the whole block at [1,8192,32]
// (back-to-back launches, MIN of 10), so the Gram half is 1.05 ms, 47% of the
// stage, and it is the one part of the chunk the vector unit is bad at: 32
// row-broadcast multiplies plus 32 whole-row reductions against a handful of
// instructions for every other step.  msprof agrees the block has no vector
// headroom left (aiv_vec_ratio 0.845).
//
// The split needs three things:
//   * the operands (ga = qf*exp2(gz), gk1 = ga's K side times beta,
//     gb = kf*exp2(-gz)) rounded to bf16 and published to GM: 12 KB per chunk.
//     A store-headroom probe measured 24 KB/chunk of extra MTE3 at 0.047 ms of
//     the 2.2 ms block, so the traffic is nearly free;
//   * the Cube side: Aqk = ga @ gb^T and L = gk1 @ gb^T as two 16x16x128 Mmads
//     that share one L0B operand.  Neither side needs a transpose: Nd2Nz of the
//     [16,128] tile plus a *plain* LoadData, read as B[K=128,N=16] with
//     MmadParams(16,16,128), is the transposed operand (probe in
//     /tmp/kdaval/cubeprobe);
//   * a software pipeline so the cross-core round trip is off the critical
//     path.  The AIV masks chunk u-1's Gram (which the Cube finished while it
//     computed chunk u) and only then publishes chunk u, so the Cube has a
//     whole chunk of vector work to answer in; the AIC is paired with two AIVs
//     (MIX_AIC_1_2) and computes both subcores' Grams per step, which leaves it
//     at ~25% duty.  The Aqk32/L slots double as the
//     Cube->AIV channel (2 KB read + 2 KB write per chunk) so no new buffer is
//     needed for the Gram results; the mask, the scale and the bf16 cast stay
//     on the AIV.
//
// Precision: the Cube forms the same products the vector loop did (both
// operands are exactly bf16 after the cast) with an fp32 accumulator, so the
// only difference is the operand rounding - measured end to end through the
// whole KDA pipeline in docs/ASCENDC_V1_KERNELS.md.
//
// The AIV publishes through a two-deep UB ring: the slot written for chunk u
// was last read by the Cube at step u-2, and the DONE wait for chunk u-1 sits
// between them, so the WAR is covered by the handshake rather than by an extra
// loop-carried flag (those hang in this runtime, see the note below).
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
// Cross-core flag channels (mode 2 = this AIC and its two AIVs).  The
// AIV->AIC direction is an *AND over the group's subcores*: a wait on the AIC
// is satisfied only once both AIVs have set that flagId, and it consumes both
// bits at once (probe /tmp/miniflag.cpp: one subcore setting, or two channels
// with one set each, never satisfies a wait; both setting the same id always
// does).  So a per-subcore channel cannot carry a step, and an AIC that waits
// twice per step instead of once deadlocks.  Both subcores therefore set one
// shared READY per step and the AIC waits it once.  The flag is also a level,
// not a count: the publish has to sit *after* the previous step's DONE wait,
// otherwise a subcore that runs a step ahead donates its next set to the
// current wait, the pairing drifts one step per chunk and the long shapes hang
// (which is what the original version did).
// A mode-2 AIV->AIC flag fires only when *both* subcores of the group have set
// it (a per-subcore channel can never be satisfied by one producer alone, and
// the flag is a level rather than a count), so the handshake uses one shared
// channel per step with both subcores setting it, and the AIC waits it once.
constexpr uint16_t FL_READY = 8;    // both AIVs -> AIC: step u's operands are in GM
constexpr uint16_t FL_DONE = 9;     // AIC -> both AIVs: step u's raw Gram is in GM

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



// AIV side of the handoff: the Cube has just written the raw fp32 Gram of
// chunk `c` into the Aqk32/L slots.  Mask, scale and the Aqk16 rounding stay
// on the vector unit (2 KB read + 2 KB write per chunk, no extra GM buffer).
static __aicore__ inline void post_gram(const LocalTensor<float> ga32,
                                        const LocalTensor<float> gl32,
                                        const LocalTensor<bfloat16_t> ga16,
                                        const LocalTensor<float> gmaskS,
                                        const LocalTensor<float> gmaskL,
                                        const GlobalTensor<float> Aqk32,
                                        const GlobalTensor<float> L,
                                        const GlobalTensor<bfloat16_t> Aqk16,
                                        int32_t c, float scale, TEventID e2v, TEventID ev3) {
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    CrossCoreWaitFlag(FL_DONE);
    DataCopy(ga32, Aqk32[m0], DataCopyParams(M, 2, 0, 0));
    DataCopy(gl32, L[m0], DataCopyParams(M, 2, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Mul(ga32, ga32, gmaskS, M * M);
    Muls(ga32, ga32, scale, M * M);
    Mul(gl32, gl32, gmaskL, M * M);
    PipeBarrier<PIPE_V>();
    Cast(ga16, ga32, RoundMode::CAST_RINT, M * M);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Aqk32[m0], ga32, DataCopyParams(M, 2, 0, 0));
    DataCopy(L[m0], gl32, DataCopyParams(M, 2, 0, 0));
    DataCopy(Aqk16[m0], ga16, DataCopyParams(M, 1, 0, 0));
}

// The paired Cube: for every step u it computes the two Grams of the two
// chunks the AIVs of this group just published (Aqk = ga @ gb^T and
// L = gk1 @ gb^T), writing the raw fp32 results into the Aqk32/L slots that
// the AIV then masks, scales and rounds.
static __aicore__ inline void run_gram_aic(GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb,
                                           GM_ADDR pAqk32, GM_ADDR pL,
                                           int32_t nchunk, int32_t unroll) {
    const int32_t base = GetBlockIdx() * 2 * unroll;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 4> qa, qb;
    pipe.InitBuffer(qa, 4, M * D * 2);
    pipe.InitBuffer(qb, 4, M * D * 2);
    TQue<QuePosition::CO1, 2> qc;
    pipe.InitBuffer(qc, 2, M * M * 4);
    // L0A/L0B slots: two per chunk (ga, gk1) and one shared gb.
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 4 * M * D * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * M * D * 2);
    GlobalTensor<bfloat16_t> Ga, Gk, Gb;
    GlobalTensor<float> Aqk32, L;
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));

    for (int32_t u = 0; u < unroll; ++u) {
        // One wait per AIV channel: neither wait can be satisfied by the
        // other AIV.  The handshake stays strictly paired (backlog <= 1)
        // because the AIV that is about to publish step u+1 must first pass
        // the FL_DONE of step u's Gram, which this AIC can only send after it
        // has consumed that AIV's step-u flag.
        // One shared channel: both AIV subcores set it once per step, and a
        // mode-2 AIV->AIC flag only fires when *both* of them have (see the
        // header note), so this is the only channel that can carry a step.
        CrossCoreWaitFlag(FL_READY);
        // Both subcores clamp an out-of-range chunk to the last one, so the
        // pair stays in step; the duplicate work is idempotent.
        int32_t c0 = base + 2 * u;
        int32_t c1 = c0 + 1;
        if (c0 >= nchunk) c0 = nchunk - 1;
        if (c1 >= nchunk) c1 = nchunk - 1;
        const uint64_t o0 = static_cast<uint64_t>(c0) * N;
        const uint64_t o1 = static_cast<uint64_t>(c1) * N;
        auto ta0 = qa.AllocTensor<bfloat16_t>();
        auto tk0 = qa.AllocTensor<bfloat16_t>();
        auto ta1 = qa.AllocTensor<bfloat16_t>();
        auto tk1 = qa.AllocTensor<bfloat16_t>();
        auto tb0 = qb.AllocTensor<bfloat16_t>();
        auto tb1 = qb.AllocTensor<bfloat16_t>();
        DataCopy(ta0, Ga[o0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(tk0, Gk[o0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(tb0, Gb[o0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(ta1, Ga[o1], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(tk1, Gk[o1], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(tb1, Gb[o1], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        qa.EnQue(ta0);
        qa.EnQue(tk0);
        qa.EnQue(ta1);
        qa.EnQue(tk1);
        qb.EnQue(tb0);
        qb.EnQue(tb1);
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        ta0 = qa.DeQue<bfloat16_t>();
        tk0 = qa.DeQue<bfloat16_t>();
        ta1 = qa.DeQue<bfloat16_t>();
        tk1 = qa.DeQue<bfloat16_t>();
        tb0 = qb.DeQue<bfloat16_t>();
        tb1 = qb.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> la0 = a8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk0 = a8[M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> la1 = a8[2 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk1 = a8[3 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb0 = b8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb1 = b8[M * D * 2].ReinterpretCast<bfloat16_t>();
        LoadData(la0, ta0, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(lk0, tk0, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(la1, ta1, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(lk1, tk1, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(lb0, tb0, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(lb1, tb1, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(e1m);
        WaitFlag<HardEvent::MTE1_M>(e1m);
        auto ip = FixpipeParamsV220(M, M, 1, M, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        for (int32_t s = 0; s < 2; ++s) {
            LocalTensor<float> cf0 = qc.AllocTensor<float>();
            LocalTensor<float> cf1 = qc.AllocTensor<float>();
            if (s == 0) {
                Mmad(cf0, la0, lb0, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk0, lb0, MmadParams(M, M, D, 0, false, true));
            } else {
                Mmad(cf0, la1, lb1, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk1, lb1, MmadParams(M, M, D, 0, false, true));
            }
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            const uint64_t m0 = static_cast<uint64_t>(s == 0 ? c0 : c1) * M * M;
            Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32[m0], cf0, ip);
            Fixpipe<float, float, CFG_ROW_MAJOR>(L[m0], cf1, ip);
            SetFlag<HardEvent::FIX_M>(efm);
            WaitFlag<HardEvent::FIX_M>(efm);
            qc.FreeTensor(cf0);
            qc.FreeTensor(cf1);
        }
        qa.FreeTensor(ta0);
        qa.FreeTensor(tk0);
        qa.FreeTensor(ta1);
        qa.FreeTensor(tk1);
        qb.FreeTensor(tb0);
        qb.FreeTensor(tb1);
        CrossCoreSetFlag<2, PIPE_FIX>(FL_DONE);
    }
}

extern "C" __global__ __aicore__ void kda_pre_gram_mix(
    GM_ADDR pQ, GM_ADDR pK, GM_ADDR pV, GM_ADDR pG, GM_ADDR pBeta,
    GM_ADDR pAlog, GM_ADDR pBias,
    GM_ADDR pQn, GM_ADDR pKn, GM_ADDR pGate, GM_ADDR pGc, GM_ADDR pBetaOut,
    GM_ADDR pDecay, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pQg, GM_ADDR pKg,
    GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb,
    GM_ADDR pAqk32, GM_ADDR pAqk16, GM_ADDR pL, GM_ADDR pMaskS, GM_ADDR pMaskL,
    int32_t B, int32_t T, int32_t H, float lower_bound, float scale, int32_t unroll,
    int32_t xRowBytes, int32_t gRowBytes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        run_gram_aic(pGa, pGk, pGb, pAqk32, pL, B * H * (T / M), unroll);
    }
    if ASCEND_IS_AIV {
    const int32_t nt = T / M;
    const DataCopyPadExtParams<bfloat16_t> padNop(false, 0, 0, 0);
    const DataCopyPadExtParams<float> padNopF(false, 0, 0, 0);
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
    TEventID e3d = pipe.AllocEventID<HardEvent::MTE3_V>();
    TEventID e2vg2 = pipe.AllocEventID<HardEvent::MTE2_V>();
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
    // Two-deep UB ring for the published Gram operands (see the header note).
    TBuf<TPosition::VECCALC> bPga0, bPga1, bPgk0, bPgk1, bPgb0, bPgb1;
    pipe.InitBuffer(bPga0, N * 2); pipe.InitBuffer(bPga1, N * 2);
    pipe.InitBuffer(bPgk0, N * 2); pipe.InitBuffer(bPgk1, N * 2);
    pipe.InitBuffer(bPgb0, N * 2); pipe.InitBuffer(bPgb1, N * 2);
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

    GlobalTensor<bfloat16_t> Q, K, V, Qn, Kn, Rk, Rv, Qg, Kg, Ga, Gk, Gb;
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
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
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
    // In MIX_AIC_1_2 the AIV sees 2N raw blocks for N AIC blocks: the raw index
    // picks the group and GetSubBlockIdx() the subcore.  The two subcores walk
    // interleaved chunks (2u + sub) so the paired Cube reads two adjacent
    // operand tiles per step.  Out-of-range chunks clamp to the last one, which
    // keeps both subcores' flag counts balanced and the duplicate work
    // idempotent.
    const int32_t ratio = static_cast<int32_t>(GetTaskRation());
    const int32_t group = ratio == 0 ? static_cast<int32_t>(GetBlockIdx())
                                     : static_cast<int32_t>(GetBlockIdx()) / ratio;
    const int32_t sub = static_cast<int32_t>(GetSubBlockIdx());
    int32_t cprev = -1;
    for (int32_t u = 0; u < unroll; ++u) {
    int32_t c = group * 2 * unroll + 2 * u + sub;
    if (c >= nchunk) c = nchunk - 1;
    const int32_t bh = c / nt;
    const int32_t head = bh % H;
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t cm = static_cast<uint64_t>(c) * M;
    // The four stage-1 inputs are read straight out of the public [B, T, H, D]
    // layout instead of a packed [c, M, D] copy: one token is D contiguous
    // elements and consecutive tokens of a chunk sit H*D elements apart, so a
    // strided DataCopy replaces the pack (which cost a 0.67 ms round trip of
    // 1.07 GB at [1,8192,32]).  The downstream buffers keep the packed order.
    const int32_t b = bh / H;
    const int32_t ck = c - bh * nt;
    const uint64_t xb = (static_cast<uint64_t>(b) * T + static_cast<uint64_t>(ck) * M) *
                            static_cast<uint64_t>(H) * D +
                        static_cast<uint64_t>(head) * D;

    // The block/stride form of DataCopy is broken for a strided GM source on
    // this part (halves the bursts land unwritten, probe
    // /tmp/kdaval/probe_stride4.py); DataCopyPad's byte-stride form reads the
    // same gather correctly, so the four stage-1 inputs use it.
    DataCopyPad(qnb, Q[xb], DataCopyExtParams(M, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vq);
    DataCopyPad(knb, K[xb], DataCopyExtParams(M, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vk);
    DataCopyPad(gf, G[xb], DataCopyExtParams(M, D * 4, gRowBytes, 0, 0), padNopF);
    SetFlag<HardEvent::MTE2_V>(e2vg);
    DataCopyPad(rvb, V[xb], DataCopyExtParams(M, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vv);
    DataCopy(alog, Alog[head], 8);
    DataCopy(beta, Beta[cm], DataCopyParams(1, 2, 0, 0));
    DataCopy(gmaskL, MaskL[0], DataCopyParams(M, 2, 0, 0));
    // The A-side triangle comes back: the Cube writes the full Gram.
    DataCopy(gmaskS, MaskS[0], DataCopyParams(M, 2, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2vg2);
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

    WaitFlag<HardEvent::MTE2_V>(e2vg2);
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
    // "t2" is reused by the qg product below, so this one store needs an
    // MTE3->V flag instead of a full PIPE_ALL.
    SetFlag<HardEvent::MTE3_V>(e3d);

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
    WaitFlag<HardEvent::MTE3_V>(e3d);
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

    // ---- Gram half: bf16 operands for the paired Cube ---------------------
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
    LocalTensor<bfloat16_t> pga = (u & 1) == 0 ? bPga0.Get<bfloat16_t>() : bPga1.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgk = (u & 1) == 0 ? bPgk0.Get<bfloat16_t>() : bPgk1.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgb = (u & 1) == 0 ? bPgb0.Get<bfloat16_t>() : bPgb1.Get<bfloat16_t>();
    Cast(pga, ga, RoundMode::CAST_RINT, N);
    Cast(pgk, gk1, RoundMode::CAST_RINT, N);
    Cast(pgb, gb, RoundMode::CAST_RINT, N);
    PipeBarrier<PIPE_V>();
    // ---- previous chunk's Gram: mask, scale, round, store -----------------
    if (cprev >= 0) {
        post_gram(ga32, gl32, ga16, gmaskS, gmaskL, Aqk32, L, Aqk16, cprev, scale, e2vs, ev3);
    }
    // ---- publish + hand off ---------------------------------------------
    // The publish sits *after* the previous step's DONE wait so that at most
    // one publication per subcore is outstanding: the flag is a level, not a
    // count, so a subcore that runs a step ahead would otherwise donate its
    // next set to the current wait and the pairings would drift (which is what
    // hangs the long shapes).
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Ga[x0], pga, DataCopyParams(M, 8, 0, 0));
    DataCopy(Gk[x0], pgk, DataCopyParams(M, 8, 0, 0));
    DataCopy(Gb[x0], pgb, DataCopyParams(M, 8, 0, 0));
    if (pQn != nullptr) DataCopy(Qn[x0], qnb, DataCopyParams(M, 8, 0, 0));
    if (pKn != nullptr) DataCopy(Kn[x0], knb, DataCopyParams(M, 8, 0, 0));
    CrossCoreSetFlag<2, PIPE_MTE3>(FL_READY);
    cprev = c;
    PipeBarrier<PIPE_ALL>();
    }
    // Drain the pipeline: the last chunk's Gram has no later chunk to hide
    // behind (~1/unroll of the stage, and the Cube is idle by then).
    post_gram(ga32, gl32, ga16, gmaskS, gmaskL, Aqk32, L, Aqk16, cprev, scale, e2vs, ev3);
    PipeBarrier<PIPE_ALL>();
    }
}
