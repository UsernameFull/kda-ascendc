// Timing probe (NOT a production kernel): what sets kda_solve_assemble's time?
//
// ``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.35 left the
// coupling-block kernel as the largest unexplained item on the AIC side: it
// moves 151.0 MB per call at [1,8192,96,128]/C=64 in 0.928 ms, i.e. 163 GB/s,
// where the cube solve's cold-path marginal rate is 839 GB/s - so ~0.75 ms of
// it is *structure*, not bytes.  The structural candidates are the small
// transfers (per chunk: one 2 KB A load, two 1 KB B loads, and in pass 1 two
// more 2 KB loads), the two-pass barrier with the P round trip through GM, and
// the per-chunk L0 chain (LoadData -> Mmad -> Fixpipe with two flag waits).
//
// This kernel is a transcription of k1_solve_assemble.cpp whose arms remove one
// candidate at a time, so each delta has a single cause:
//
//   mode 0  the shipped structure (control)
//   mode 1  loads only: no LoadData, no Mmad, no Fixpipe - the price of the
//           GM traffic as it is actually issued (6 DMA calls per chunk-pass)
//   mode 2  loads + Fixpipe (no LoadData, no Mmad): the store side is back, the
//           L0 chain is not
//   mode 3  control minus the P round trip: pass 0 does not store P and pass 1
//           does not load it, everything else unchanged (the arm an
//           L0C -> L1 -> L0B single-pass structure would take)
//   mode 4  pass 0 only: half the chunk-passes, same per-chunk work - a
//           per-block fixed cost (the two-pass barrier) would show up as
//           time > half of mode 0
//
// Modes 1/2 feed the Fixpipe and the Mmad whatever the buffer happens to hold,
// so their *outputs* are meaningless; they are measurement arms.
//
// Transcribed 2026-09-23 from k1_solve_assemble.cpp @ 5b07a0f.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;  // parent chunk size
constexpr int32_t M = PC / 2;      // diagonal sub-block size
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;     // 16-row fractal bands
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;   // one [M, M] bf16 operand
constexpr int32_t SLOTS = PASSES * NC;

extern "C" __global__ __aicore__ void kda_solve_assemble_probe(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C,
    int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, LBSZ);
    pipe.InitBuffer(qb, NC, LBSZ);
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));

    const int32_t passes = (mode == 4) ? 1 : PASSES;
    const bool l0chain = (mode != 1) && (mode != 2);
    const bool store = (mode != 1);
    for (int32_t pass = 0; pass < passes; ++pass) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto la = qa.AllocTensor<bfloat16_t>();
            auto lb = qb.AllocTensor<bfloat16_t>();
            if (pass == 0) {
                DataCopy(la, Lneg[static_cast<uint64_t>(c0 + ch) * MM],
                         Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                for (int32_t mm = 0; mm < KF; ++mm) {
                    DataCopy(lb[mm * KF * 256],
                             Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + mm * 16 * M],
                             Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                }
            } else {
                DataCopy(la, Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + MM],
                         Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                if (mode != 3) {   // 3: the whole P round trip is gone
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        DataCopy(lb[mm * KF * 256],
                                 P[static_cast<uint64_t>(c0 + ch) * MM + mm * 16 * M],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                }
            }
            qa.EnQue(la);
            qb.EnQue(lb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            auto la = qa.DeQue<bfloat16_t>();
            auto lb = qb.DeQue<bfloat16_t>();
            const int32_t slot = pass * NC + ch;
            LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            if (l0chain) {
                for (int32_t dd = 0; dd < KF; ++dd) {
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                                 LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                    }
                }
                LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                LocalTensor<float> cf = cfall[slot * MM];
                Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
            }
            if (store) {
                auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
                ip.quantPre = QuantMode_t::F322BF16;
                ip.unitFlag = 0;
                if (pass == 0) {
                    if (mode != 3) {
                        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                            P[static_cast<uint64_t>(c0 + ch) * MM],
                            cfall[slot * MM], ip);
                    }
                } else {
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC],
                        cfall[slot * MM], ip);
                }
            }
            qa.FreeTensor(la);
            qb.FreeTensor(lb);
        }
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
