// Timing probe (NOT a production kernel): can the assemble's loads be coalesced?
//
// ``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.36 left one measured
// item on the assemble's list: with the L0 chain and the stores taken out, the
// GM traffic alone is 0.339 ms for 100.7 MB (297 GB/s), i.e. 52% of the shipped
// structure's 0.648 ms, while the cube solve's cold-path marginal rate is
// 839 GB/s.  The pattern is 6 ND2NZ calls per chunk (ave 1.37 KB): one 2 KB
// whole-tile A load plus two 1 KB B bands per pass.  This kernel holds the
// bytes fixed at 100.7 MB and varies only how many calls carry them:
//
//   mode 0  loads only, the shipped pattern            6 calls/chunk   73728
//   mode 1  loads only, the two B bands merged        4 calls/chunk   49152
//   mode 2  loads only, the A load batched per block  2.5 calls/chunk 15360
//   mode 3  loads only, both operands batched from a
//           packed source (needs a re-layout)         1 call/chunk    12288
//   mode 4  loads only, the whole block-pass in one
//           call from a packed source (ceiling)       0.5 calls/chunk  6144
//   mode 5  the full structure (chain + stores), shipped pattern (anchor for
//           11.36's mode 0)
//   mode 6  the full structure with mode 2's load pattern - the production
//           candidate, but with the single-flag (TBuf) load path modes 0-5 use
//   mode 7  the shipped kernel verbatim: per-chunk L1 queue (EnQue/DeQue per
//           chunk, the shipped kernel's own overlap structure), shipped calls
//   mode 8  the shipped queue structure with mode 6's calls - the candidate in
//           the shipped sync shape
//
// Modes 5 and 7 are the same call pattern and the same arithmetic with two
// different L1/sync structures, so 5 vs 7 separates the structure from the
// device; 8 vs 7 is the candidate's true price.
//
// Modes 0-4 are measurement arms: nothing reads what they load.  Modes 5 and 6
// are the real thing, so the probe can also check that mode 6 is bit-identical
// to mode 5 - batched ND2NZ must land the same bytes in L1 or the chain would
// read a different operand.
//
// The batched forms are legal today because the sources are uniformly strided:
// Lneg is chunk-contiguous (MM elements apart), Xb's second block is 2*MM apart
// and P is contiguous, so Nd2NzParams(ndNum, nValue, dValue, srcNdMatrixStride,
// srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride) can take several
// ND matrices per call (ndNum was 1 everywhere in the shipped kernel), and the
// per-chunk 2-band split merges by the same route (2 x [16, M] at 512 elements).
// Modes 3/4 additionally need the operands re-laid out (the per-block bands
// packed contiguously), which is why they are ceilings and not candidates.
//
// Written 2026-09-23 next to k1_solve_assemble_probe.cpp (mode 0/5 there are
// this file's mode 1/0 in the other naming - both are anchors).
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;  // parent chunk size
constexpr int32_t M = PC / 2;      // diagonal sub-block size
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;     // 16-row fractal bands
constexpr int32_t BANDE = KF * 256;  // elements of one [16, M] band (512 at M=32)
constexpr int32_t BAND = BANDE * 2;  // bytes of one band (1 KB at M=32)
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;   // bytes of one [M, M] bf16 operand

extern "C" __global__ __aicore__ void kda_solve_assemble_coalesce_probe(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, GM_ADDR pPack,
    int32_t C, int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    const bool chain = (mode == 5) || (mode == 6) || (mode >= 7);
    const bool batchedA = (mode == 2) || (mode == 6) || (mode == 8);
    const bool mergedB = (mode == 1) || (mode == 2) || (mode == 6) || (mode == 8);
    const bool packed = (mode == 3) || (mode == 4);    // re-laid-out source
    const bool queued = (mode >= 7);                   // shipped L1 queue shape
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    // One L1 region for the whole block-pass: the A tiles of the block first
    // (contiguous, LBSZ apart - what the batched call wants), then the B bands
    // (two per chunk, BAND apart - the band-major order LoadDataWithTranspose
    // wants, and what one merged call per chunk produces).
    TBuf<TPosition::B1> buf;
    pipe.InitBuffer(buf, 2 * NC * LBSZ);
    // Modes 7/8: the shipped kernel's own L1 shape - one queue per operand,
    // NC chunks deep, EnQue/DeQue per chunk.  Mode 8 feeds the A queue with a
    // single batched call into its first slot, which only works if the queue's
    // slots are contiguous in allocation order; the probe's bit-identity check
    // against mode 7 is what verifies that.
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, LBSZ);
    pipe.InitBuffer(qb, NC, LBSZ);
    LocalTensor<float> cfall(TPosition::CO1, 0, PASSES * NC * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, PASSES * NC * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, PASSES * NC * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P, Pack;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));
    Pack.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pPack));

    LocalTensor<uint8_t> l1 = buf.Get<uint8_t>();
    if (queued) {
        for (int32_t pass = 0; pass < PASSES; ++pass) {
            LocalTensor<bfloat16_t> las[NC];
            LocalTensor<bfloat16_t> lbs[NC];
            for (int32_t ch = 0; ch < nch; ++ch) {
                las[ch] = qa.AllocTensor<bfloat16_t>();
                lbs[ch] = qb.AllocTensor<bfloat16_t>();
            }
            if (batchedA) {
                const uint64_t stride = (pass == 0) ? MM : 2 * MM;
                const uint64_t off = (pass == 0)
                    ? static_cast<uint64_t>(c0) * MM
                    : static_cast<uint64_t>(c0) * 2 * MM + MM;
                DataCopy(las[0], (pass == 0) ? Lneg[off] : Xb[off],
                         Nd2NzParams(nch, M, M, stride, M, M, 1, MM));
            }
            for (int32_t ch = 0; ch < nch; ++ch) {
                if (!batchedA) {
                    const uint64_t off = (pass == 0)
                        ? static_cast<uint64_t>(c0 + ch) * MM
                        : static_cast<uint64_t>(c0 + ch) * 2 * MM + MM;
                    DataCopy(las[ch], (pass == 0) ? Lneg[off] : Xb[off],
                             Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                }
                if (mergedB) {
                    const uint64_t off = (pass == 0)
                        ? static_cast<uint64_t>(c0 + ch) * 2 * MM
                        : static_cast<uint64_t>(c0 + ch) * MM;
                    DataCopy(lbs[ch], (pass == 0) ? Xb[off] : P[off],
                             Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                } else {
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        const uint64_t off = (pass == 0)
                            ? static_cast<uint64_t>(c0 + ch) * 2 * MM + mm * 16 * M
                            : static_cast<uint64_t>(c0 + ch) * MM + mm * 16 * M;
                        DataCopy(lbs[ch][mm * BANDE], (pass == 0) ? Xb[off] : P[off],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                }
            }
            for (int32_t ch = 0; ch < nch; ++ch) {
                qa.EnQue(las[ch]);
                qb.EnQue(lbs[ch]);
            }
            SetFlag<HardEvent::MTE2_MTE1>(e21);
            WaitFlag<HardEvent::MTE2_MTE1>(e21);
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<bfloat16_t> la = qa.DeQue<bfloat16_t>();
                LocalTensor<bfloat16_t> lb = qb.DeQue<bfloat16_t>();
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> a =
                    a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
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
                auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
                ip.quantPre = QuantMode_t::F322BF16;
                ip.unitFlag = 0;
                if (pass == 0) {
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
                } else {
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
                qa.FreeTensor(la);
                qb.FreeTensor(lb);
            }
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
    for (int32_t pass = 0; pass < PASSES; ++pass) {
        LocalTensor<bfloat16_t> la0 =
            l1[0].ReinterpretCast<bfloat16_t>();           // A tiles of the block
        LocalTensor<bfloat16_t> lb0 =
            l1[NC * LBSZ].ReinterpretCast<bfloat16_t>();   // B bands of the block
        if (packed) {
            // Re-laid-out source: the block-pass's 16 KB are contiguous, so the
            // whole batch (A tiles as 2 bands each + B bands) is one or two
            // uniform-band calls.
            // One block-pass is 2*NC*MM elements (16 KB at M=32, NC=4): the
            // A tiles first (NC*MM), then the B bands (2*NC*BANDE = NC*MM).
            const uint64_t base = static_cast<uint64_t>(GetBlockIdx()) *
                                      (PASSES * 2 * NC * MM) +
                                  pass * (2 * NC * MM);
            if (mode == 4) {
                // The whole block-pass (the four A tiles = 8 bands, then the
                // eight B bands) in one call of 16 KB.
                DataCopy(la0, Pack[base],
                         Nd2NzParams(2 * (2 * NC), 16, M, BANDE, M, 16, 1, BANDE));
            } else {
                DataCopy(la0, Pack[base],
                         Nd2NzParams(2 * NC, 16, M, BANDE, M, 16, 1, BANDE));
                DataCopy(lb0, Pack[base + NC * MM],
                         Nd2NzParams(2 * NC, 16, M, BANDE, M, 16, 1, BANDE));
            }
        } else {
            // A operand: one call for the block (mode 2/6) or one per chunk.
            if (batchedA) {
                const uint64_t stride = (pass == 0) ? MM : 2 * MM;
                const uint64_t off = (pass == 0)
                    ? static_cast<uint64_t>(c0) * MM
                    : static_cast<uint64_t>(c0) * 2 * MM + MM;
                DataCopy(la0, (pass == 0) ? Lneg[off] : Xb[off],
                         Nd2NzParams(nch, M, M, stride, M, M, 1, MM));
            }
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<bfloat16_t> la = l1[ch * LBSZ].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> lb =
                    l1[NC * LBSZ + ch * LBSZ].ReinterpretCast<bfloat16_t>();
                if (!batchedA) {
                    // The shipped call: one whole-tile Nd2Nz per chunk.
                    if (pass == 0) {
                        DataCopy(la, Lneg[static_cast<uint64_t>(c0 + ch) * MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    } else {
                        DataCopy(la, Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    }
                }
                // B operand: two 1 KB bands, merged into one call where the arm
                // says so (the per-band calls of the shipped kernel).
                // Merging these two calls per chunk is legal for both operands
                // (the two bands of a chunk are 1 KB apart in Xb's block 0 and
                // in P); batching them across the block is not, because Xb's
                // chunk stride is 4 KB - that is what modes 3/4 re-lay out.
                if (mergedB) {
                    const uint64_t off = (pass == 0)
                        ? static_cast<uint64_t>(c0 + ch) * 2 * MM
                        : static_cast<uint64_t>(c0 + ch) * MM;
                    DataCopy(lb, (pass == 0) ? Xb[off] : P[off],
                             Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                } else {
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        const uint64_t off = (pass == 0)
                            ? static_cast<uint64_t>(c0 + ch) * 2 * MM + mm * 16 * M
                            : static_cast<uint64_t>(c0 + ch) * MM + mm * 16 * M;
                        DataCopy(lb[mm * BANDE], (pass == 0) ? Xb[off] : P[off],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                }
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        if (chain) {
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<bfloat16_t> la = l1[ch * LBSZ].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> lb =
                    l1[NC * LBSZ + ch * LBSZ].ReinterpretCast<bfloat16_t>();
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> a =
                    a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
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
                auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
                ip.quantPre = QuantMode_t::F322BF16;
                ip.unitFlag = 0;
                if (pass == 0) {
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
                } else {
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
