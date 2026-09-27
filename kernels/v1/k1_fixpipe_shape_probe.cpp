// Timing probe (NOT a production kernel): what sets the assemble's Fixpipe
// time?  The on-board PipeUtilization collection (docs section 11.42) measured
// 1.812 us of Fixpipe busy per block = 49.6% of the block wall, from 8 calls
// of 2 KB each (4 chunks x 2 passes) - 226.5 ns per call, i.e. 9 GB/s.
//
// Two readings are possible: the per-call fixed cost is high (then batching
// calls wins), or the destination pattern is the price (then only the pattern
// can change).  The arms separate them; the arithmetic (loads + Mmad) is the
// same in every arm and arm 5 removes only the stores, so each store form's
// marginal cost is visible directly.
//
//   mode 0  production store structure: 4x [M,M] NZ -> L1 (pass 0) then
//           4x [M,M] row-major -> GM with dstStride = PC (pass 1)   = 8 calls
//   mode 1  the batched candidate: the same 4 L1 stores, then ONE row-major
//           call with ndNum = nch (srcNdStride/dstNdStride batch the four
//           strided blocks into one descriptor).  Its A16 output must be
//           bit-identical to mode 0's before it is timed.
//   mode 2  8x NZ -> L1: the per-call cost of the pass-0 destination
//   mode 3  8x row-major -> GM with dstStride = M (2 KB contiguous): the
//           per-call cost of the pass-1 bytes without the row stride
//   mode 4  8x row-major -> GM with dstStride = PC (production rows)
//   mode 5  no stores at all: the loads + Mmad floor
//
// The srcNdStride of mode 1 is a runtime argument (it is in 1 KB L0C fractal
// units; the four slots are 4 KB apart, so 4 is the first guess) - the driver
// sweeps it and only times the value that reproduces mode 0 bit for bit.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;
constexpr int32_t M = PC / 2;
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;
constexpr int32_t BANDE = KF * 256;
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;
constexpr int32_t SLOTS = PASSES * NC;

extern "C" __global__ __aicore__ void kda_fixpipe_shape_probe(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pDst3, int32_t C,
    int32_t mode, int32_t srcNd) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufA, bufB, bufP;
    pipe.InitBuffer(bufA, NC * LBSZ);
    pipe.InitBuffer(bufB, NC * LBSZ);
    pipe.InitBuffer(bufP, SLOTS * LBSZ);
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, Dst3;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    Dst3.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pDst3));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lbAll = bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();

    // Loads: the production batched form, both passes' operands up front.
    DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
             Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(lbAll[ch * MM], Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM],
                 Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
    }
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);

    for (int32_t slot = 0; slot < SLOTS; ++slot) {
        const int32_t ch = slot % NC;
        LocalTensor<bfloat16_t> la = laAll[ch * MM];
        LocalTensor<bfloat16_t> lb = lbAll[ch * MM];
        LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
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

    if (mode == 5) {
        // floor: arithmetic only
    } else if (mode == 0) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto ipnz = FixpipeParamsV220(M, M, M, M, false);
            ipnz.quantPre = QuantMode_t::F322BF16;
            ipnz.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cfall[ch * MM], ipnz);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto ip = FixpipeParamsV220(M, M, M, PC, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC],
                cfall[(NC + ch) * MM], ip);
        }
    } else if (mode == 1) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto ipnz = FixpipeParamsV220(M, M, M, M, false);
            ipnz.quantPre = QuantMode_t::F322BF16;
            ipnz.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cfall[ch * MM], ipnz);
        }
        auto ip = FixpipeParamsV220(M, M, M, PC, false, QuantMode_t::F322BF16, 0,
                                    static_cast<uint16_t>(nch),
                                    static_cast<uint16_t>(srcNd),
                                    static_cast<uint16_t>(PC * PC), 0);
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            A16[static_cast<uint64_t>(c0) * PC * PC + M * PC],
            cfall[NC * MM], ip);
    } else if (mode == 2) {
        for (int32_t slot = 0; slot < SLOTS; ++slot) {
            auto ipnz = FixpipeParamsV220(M, M, M, M, false);
            ipnz.quantPre = QuantMode_t::F322BF16;
            ipnz.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[slot * MM], cfall[slot * MM], ipnz);
        }
    } else if (mode == 3) {
        for (int32_t slot = 0; slot < SLOTS; ++slot) {
            auto ip = FixpipeParamsV220(M, M, M, M, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                Dst3[(static_cast<uint64_t>(c0) * SLOTS + slot) * MM], cfall[slot * MM], ip);
        }
    } else {  // mode 4
        for (int32_t slot = 0; slot < SLOTS; ++slot) {
            auto ip = FixpipeParamsV220(M, M, M, PC, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                Dst3[(static_cast<uint64_t>(c0) * SLOTS + slot) * MM], cfall[slot * MM], ip);
        }
    }
    PipeBarrier<PIPE_ALL>();
}
