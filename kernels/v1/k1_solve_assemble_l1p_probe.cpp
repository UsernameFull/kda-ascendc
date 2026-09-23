// Timing probe (NOT a production kernel): can the assemble's P round trip
// through GM be replaced by L0C -> L1 -> L0B?
//
// ``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.36 priced the P
// round trip at 0.137 ms of the coupling block's 0.648 ms (an arm that simply
// stopped storing and loading it).  This kernel builds the structure that would
// actually take that back, because the two are not the same thing: removing the
// round trip removes 50.3 MB of GM traffic, while *replacing* it keeps the
// per-chunk dependency and moves the intermediate on chip.
//
//   mode 0  the shipped two-pass structure: pass 0 fixpipes P row-major to its
//           own GM tile, pass 1 loads it back and transposes it into L0B
//   mode 1  mode 0 minus the store and the load (the section 11.36 ceiling -
//           wrong by construction, it feeds the Mmad whatever L1 holds)
//   mode 2  the candidate: pass 0 fixpipes P into an L1 region in NZ
//           (CFG_NZ, same F322BF16 quantization), pass 1 builds L0B from that
//           region with LoadDataWithTranspose
//
// The load pattern is the production one (section 11.37's batched form: one
// Nd2Nz per block for the A operand, a chunk's two B bands merged, pass 1's
// whole B operand in one call), so the arms differ only in where P lives.
//
// The layout is the whole question.  The shipped path reaches L0B through
// Nd2Nz's band-major arrangement [k-band][n-block][16 rows][16 elements] with
// four 512-byte fractals contiguous, which is why the shipped call is
// LoadDataWithTranspose(0, KF*KF, 1, 0, 0).  A CFG_NZ fixpipe writes the same
// four fractals in the other block order - [n-block][k-block] - so a plain
// (0, KF*KF, 1, 0, 0) would feed the Mmad transposed.  Mode 2 therefore loads
// them one fractal at a time with the block order put right by hand:
//
//   L0B fractal 0 <- source block 0      (kb=0, nb=0)
//   L0B fractal 1 <- source block 2      (kb=0, nb=1)
//   L0B fractal 2 <- source block 1      (kb=1, nb=0)
//   L0B fractal 3 <- source block 3      (kb=1, nb=1)
//
// and the probe's bit-identity check against mode 0 is what says whether that
// mapping is right (a wrong one is not slow, it is wrong).
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;
constexpr int32_t M = PC / 2;
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;
constexpr int32_t BANDE = KF * 256;   // elements of one [16, M] band
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;      // bytes of one [M, M] bf16 operand
constexpr int32_t SLOTS = PASSES * NC;
constexpr int32_t FRAC = 256;         // elements of one bf16 16x16 fractal

extern "C" __global__ __aicore__ void kda_solve_assemble_l1p_probe(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C, int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    const bool onchip = (mode == 2);
    const bool roundtrip = (mode == 0);
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufA, bufB, bufP;
    pipe.InitBuffer(bufA, NC * LBSZ);
    pipe.InitBuffer(bufB, NC * LBSZ);
    // Mode 2's P tiles: one [M, M] bf16 tile per chunk of the block (8 KB at
    // NC = 4, M = 32), written by the fixpipe and read back by the load.
    pipe.InitBuffer(bufP, NC * LBSZ);
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lbAll = bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();

    for (int32_t pass = 0; pass < PASSES; ++pass) {
        // The section 11.37 load pattern: the block's A tiles in one call, a
        // chunk's two B bands merged, and in pass 1 the whole block's B
        // operand in one call.
        if (pass == 0) {
            DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                     Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
            for (int32_t ch = 0; ch < nch; ++ch) {
                DataCopy(lbAll[ch * MM], Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM],
                         Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
            }
        } else {
            DataCopy(laAll, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                     Nd2NzParams(nch, M, M, 2 * MM, M, M, 1, MM));
            if (roundtrip) {
                DataCopy(lbAll, P[static_cast<uint64_t>(c0) * MM],
                         Nd2NzParams(KF * nch, 16, M, BANDE, M, 16, 1, BANDE));
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t slot = pass * NC + ch;
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
            if (pass == 0 || roundtrip) {
                // Pass 0's B operand is X11 in both structures; pass 1's is P,
                // which mode 0 reads from its GM tile like pass 0's X11.
                LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
            } else if (!onchip) {
                // Mode 1: pass 1 loads nothing for B - the arm is the ceiling
                // the round trip could ever be worth, not a candidate.
            } else {
                // Mode 2: P out of L1 in NZ order.  The four 512-byte fractals
                // are contiguous but ordered [nb][kb] instead of [kb][nb], so
                // the L0B slots take them in the order 0, 2, 1, 3.
                const int32_t src = ch * MM;
                LoadDataWithTranspose(b, lpAll[src],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[2 * FRAC], lpAll[src + FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[1 * FRAC], lpAll[src + 2 * FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[3 * FRAC], lpAll[src + 3 * FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
            }
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            LocalTensor<float> cf = cfall[slot * MM];
            Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            if (pass == 0) {
                if (roundtrip) {
                    auto ip = FixpipeParamsV220(M, M, M, M, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
                } else if (onchip) {
                    auto ip = FixpipeParamsV220(M, M, M, M, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ip);
                }
            } else {
                auto ip = FixpipeParamsV220(M, M, M, PC, false);
                ip.quantPre = QuantMode_t::F322BF16;
                ip.unitFlag = 0;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
            }
        }
        // Pass 1 reads what pass 0 wrote (GM in mode 0, L1 in mode 2).
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
