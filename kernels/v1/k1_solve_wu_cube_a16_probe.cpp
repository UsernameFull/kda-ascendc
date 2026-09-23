// Timing probe (NOT a production kernel): what does the Cube solve's per-pass
// A16 re-read cost?
//
// ``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.34 measured the
// solve stage as max(AIV, AIC) = 2.535 ms, with the AIC half split into
// assemble 0.928 + this kernel 1.586, and left two candidates on the table.
// The first one is that ``kda_solve_wu_cube_kernel`` loads A16 from GM in
// *each* of its two passes - 8 KB per chunk per pass, 16 KB per chunk in
// total, 98.3 MB of its own 983 MB per call at [1,8192,96,128]/C=64 - and the
// open question is whether those bytes are on the critical path at all: if the
// kernel is DMA-bound, dropping them is worth ~98/983 of 1.586 ms (~0.16 ms),
// and if it is bound by waves, L0 crossings or the Mmad/Fixpipe chain, it is
// worth nothing.
//
// A synthetic bandwidth ladder would only answer that by inference, so this
// probe *is* the candidate.  Mode 0 transcribes the shipped load structure
// (control), mode 1 keeps the block's NC tiles resident in L1 (NC x 8 KB) and
// re-issues only the L1 -> L0A crossing, mode 2 drops the A16 loads entirely
// (garbage operands, timing only) to give the slope a second point.  Every
// other instruction - the RHS loads, both LoadData crossings, the Mmad and the
// Fixpipe - is identical across the three modes, and all three pay for the
// same InitBuffer arithmetic, so the differences are the A16 GM traffic and
// nothing else.
//
// Transcribed 2026-09-23 from k1_solve_wu_cube.cpp @ 9cf5dcb; the caller hands
// it production-sized operands and interleaves the arms.
#include "kernel_operator.h"
using namespace AscendC;
#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
constexpr int32_t M = KDA_CHUNK, K = KDA_CHUNK, D = 128;
constexpr int32_t KF = K / 16;   // 16-row bands of a chunk-sized tile
constexpr int32_t DF = D / 16;
#ifndef KDA_WU_NCHUNK
#define KDA_WU_NCHUNK 4
#endif
constexpr int32_t NC = KDA_WU_NCHUNK;

extern "C" __global__ __aicore__ void kda_solve_wu_cube_a16_probe(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU,
    int32_t C, int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, M * K * 2);
    pipe.InitBuffer(qb, NC, D * K * 2);
    // Modes 1/2 read their A16 operand from here instead of from a queue slot;
    // the allocation is unconditional so all three modes carry the same L1
    // arithmetic.
    TBuf<TPosition::B1> bRes;
    pipe.InitBuffer(bRes, NC * M * K * 2);
    // One L0C slot per (pass, chunk) unit, as in the shipped kernel.
    LocalTensor<float> cfall(TPosition::CO1, 0, 2 * NC * M * D);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 2 * NC * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * NC * D * K * 2);
    LocalTensor<bfloat16_t> resident = bRes.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));

    if (mode == 1) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            DataCopy(resident[ch * M * K],
                     A16[static_cast<uint64_t>(c0 + ch) * M * K],
                     Nd2NzParams(1, M, K, 0, K, M, 1, 0));
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }

    for (int32_t pass = 0; pass < 2; ++pass) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto lb = qb.AllocTensor<bfloat16_t>();
            if (mode == 0) {
                auto la = qa.AllocTensor<bfloat16_t>();
#if KDA_CHUNK > 16
                DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K],
                         Nd2NzParams(1, M, K, 0, K, M, 1, 0));
#else
                DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
#endif
                qa.EnQue(la);
            }
            GlobalTensor<bfloat16_t> &rhs = (pass == 0) ? Rk : Rv;
#if KDA_CHUNK > 16
            for (int32_t mm = 0; mm < KF; ++mm) {
                DataCopy(lb[mm * DF * 256],
                         rhs[static_cast<uint64_t>(c0 + ch) * M * D + mm * 16 * D],
                         Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
            }
#else
            DataCopy(lb, rhs[static_cast<uint64_t>(c0 + ch) * M * D],
                     Nd2NzParams(1, M, D, 0, D, M, 1, 0));
#endif
            qb.EnQue(lb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la;
            if (mode == 0) {
                la = qa.DeQue<bfloat16_t>();
            } else {
                la = resident[ch * M * K];
            }
            auto lb = qb.DeQue<bfloat16_t>();
            const int32_t slot = pass * NC + ch;
            LocalTensor<bfloat16_t> a = a8[slot * M * K * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b = b8[slot * D * K * 2].ReinterpretCast<bfloat16_t>();
            for (int32_t dd = 0; dd < K / 16; ++dd) {
                for (int32_t mm = 0; mm < KF; ++mm) {
                    LoadData(a[(mm * (K / 16) + dd) * 256],
                             la[(dd * KF + mm) * 256],
                             LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                }
            }
            LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * DF, 1, 0, 0));
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            LocalTensor<float> cf = cfall[slot * M * D];
            Mmad(cf, a, b, MmadParams(M, D, K, 0, false, true));
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            auto ip = FixpipeParamsV220(D, M, M, D, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            if (pass == 0) {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    W[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            } else {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    U[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            }
            if (mode == 0) {
                qa.FreeTensor(la);
            }
            qb.FreeTensor(lb);
        }
    }
}
