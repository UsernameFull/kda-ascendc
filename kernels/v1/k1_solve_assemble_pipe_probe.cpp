// Timing probe (NOT a production kernel): the assemble's per-unit chain is a
// serial chain of full drains - LoadData/LoadDataWithTranspose -> SetFlag/
// WaitFlag<MTE1_M> -> Mmad -> SetFlag/WaitFlag<M_FIX> -> Fixpipe, eight times -
// so the on-board profile (docs section 11.42) shows the block wall is the SUM
// of the pipe busy times (fixpipe 1.812 + mte2 1.201 + scalar 0.610 + mte1
// 0.162 + cube 0.098 = 3.884 us against a 3.652 us wall): no two pipes are
// ever busy at once.  The units are independent (one L0/L0C slot per unit, no
// WAR), so the drains are the only thing serialising them.
//
// This kernel is the production mode-2 structure with the drains removed one
// layer at a time; every arm computes and stores the same thing, so A16 has to
// stay bit-identical across all of them.
//
//   mode 0  control: the production chain (per-unit drains, pass barrier,
//           pass 1's la issued at the top of pass 1)
//   mode 1  hoist: pass 1's la issued before pass 0's loads (control
//           otherwise) - the pass-1 MTE2 is hideable under pass 0
//   mode 2  phases: hoist + per pass: all fills, one MTE1_M wait, all Mmads,
//           one M_FIX wait, all Fixpipes
//   mode 3  interleave: hoist + ping-pong M_FIX ids, so Fixpipe ch overlaps
//           Mmad ch+1 (still <= 1 outstanding set per id)
//   mode 4  phases without the hoist (separates the drain change from it)
//   mode 6  rolling: each stage one chunk ahead of its consumer, so fill /
//           lift / mad / fix all hold a different chunk at once (mode 5 rolls
//           only MTE2; the phases arms wait for the whole pass)
//   mode 7  coalesced lift: the per-chunk lift as repeat-2 calls (source
//           fractals (0,2) -> L0A (0,1), (1,3) -> (2,3)), 5+8 instructions per
//           chunk-pass cut to 3+4; the permutation the four single-fractal
//           calls spell out is srcStride 2 with consecutive destinations
//   mode 8  the same permutation with the other knob: consecutive source
//           fractals (0,1) -> (0,2) with one 512 B destination gap
//   mode 9  whole-window fill: the block's Xb span is contiguous in GM, so
//           read it (both bands, all chunks) as 4*nch 16-row ND matrices in
//           ONE call - band 0 is pass 0's B, band 1 is pass 1's A, which
//           therefore needs no fill at all (6 MTE2 calls/block -> 2)
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
constexpr int32_t FRAC = 256;

// Mode 6's body: one pass of the chain with every stage a chunk ahead of its
// consumer - fill(ch+2) (MTE2 -> L1), lift(ch+1) (L1 -> L0A/L0B), mad(ch)
// (L0C) and fix(ch-1) (L0C -> L1/GM) issue in that order in one iteration, so
// in steady state all four pipes hold a different chunk at once.  This is the
// double-buffer pattern of the reference page applied inside one pass rather
// than between calls; modes 0-5 never actually overlapped MTE1/M/FIX with
// their predecessors (phases waits for the whole pass's lifts, depth-1 rolls
// MTE2 only), so it is the first arm that tests whether the sum-of-pipes wall
// is an artefact of the drains at all.  Two ping-pong ids for MTE1_M and two
// for M_FIX; MTE2_MTE1 stays the per-chunk pair mode 5 uses.  Bounds are
// checked per stage, so one body covers the prologue (k < 0), the steady
// state and the drain (k >= nch).  The pass boundary keeps the existing
// barrier, i.e. the P buffer's FIX -> MTE1 dependency is unchanged.
static __aicore__ inline void kda_asm_rolling_pass(
    int32_t pass, int32_t nch, int32_t c0, LocalTensor<bfloat16_t> laAll,
    LocalTensor<bfloat16_t> lbAll, LocalTensor<bfloat16_t> lpAll,
    LocalTensor<uint8_t>& a8, LocalTensor<uint8_t>& b8,
    LocalTensor<float>& cfall, GlobalTensor<bfloat16_t>& A16,
    GlobalTensor<bfloat16_t>& Xb, GlobalTensor<bfloat16_t>& Lneg,
    TEventID e21, TEventID e21b, TEventID e1m, TEventID e1mb,
    TEventID f0, TEventID f1) {
    for (int32_t k = -2; k <= nch; ++k) {
        if (k + 2 < nch) {
            const int32_t nx = k + 2;
            if (pass == 0) {
                DataCopy(laAll[nx * MM],
                         Lneg[static_cast<uint64_t>(c0 + nx) * MM],
                         Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                DataCopy(lbAll[nx * MM],
                         Xb[static_cast<uint64_t>(c0 + nx) * 2 * MM],
                         Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
            } else {
                DataCopy(laAll[nx * MM],
                         Xb[static_cast<uint64_t>(c0 + nx) * 2 * MM + MM],
                         Nd2NzParams(1, M, M, 0, M, M, 1, 0));
            }
            SetFlag<HardEvent::MTE2_MTE1>((nx % 2) ? e21b : e21);
        }
        if (k + 1 >= 0 && k + 1 < nch) {
            const int32_t nx = k + 1;
            WaitFlag<HardEvent::MTE2_MTE1>((nx % 2) ? e21b : e21);
            const int32_t slot = pass * NC + nx;
            LocalTensor<bfloat16_t> a =
                a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b =
                b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            for (int32_t dd = 0; dd < KF; ++dd) {
                for (int32_t mm = 0; mm < KF; ++mm) {
                    LoadData(a[(mm * KF + dd) * 256],
                             laAll[(nx * MM) + (dd * KF + mm) * 256],
                             LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                }
            }
            if (pass == 0) {
                LoadDataWithTranspose(b, lbAll[nx * MM],
                                      LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
            } else {
                const int32_t src = nx * MM;
                LoadDataWithTranspose(b, lpAll[src],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[2 * FRAC], lpAll[src + FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[1 * FRAC], lpAll[src + 2 * FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
                LoadDataWithTranspose(b[3 * FRAC], lpAll[src + 3 * FRAC],
                                      LoadData2dTransposeParams(0, 1, 1, 0, 0));
            }
            SetFlag<HardEvent::MTE1_M>((nx % 2) ? e1mb : e1m);
        }
        if (k >= 0 && k < nch) {
            WaitFlag<HardEvent::MTE1_M>((k % 2) ? e1mb : e1m);
            const int32_t slot = pass * NC + k;
            LocalTensor<bfloat16_t> a =
                a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b =
                b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            Mmad(cfall[slot * MM], a, b, MmadParams(M, M, M, 0, false, true));
            SetFlag<HardEvent::M_FIX>((k % 2) ? f1 : f0);
        }
        if (k - 1 >= 0 && k - 1 < nch) {
            const int32_t nx = k - 1;
            WaitFlag<HardEvent::M_FIX>((nx % 2) ? f1 : f0);
            LocalTensor<float> cf = cfall[(pass * NC + nx) * MM];
            if (pass == 0) {
                auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                ipnz.quantPre = QuantMode_t::F322BF16;
                ipnz.unitFlag = 0;
                Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[nx * MM], cf, ipnz);
            } else {
                auto ip = FixpipeParamsV220(M, M, M, PC, false);
                ip.quantPre = QuantMode_t::F322BF16;
                ip.unitFlag = 0;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    A16[static_cast<uint64_t>(c0 + nx) * PC * PC + M * PC], cf, ip);
            }
        }
    }
}

extern "C" __global__ __aicore__ void kda_solve_assemble_pipe_probe(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C,
    int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID e1mb = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID e21b = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID f0 = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID f1 = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufA, bufA2, bufB, bufP, bufX;
    pipe.InitBuffer(bufA, NC * LBSZ);
    pipe.InitBuffer(bufA2, NC * LBSZ);   // hoisted pass-1 la
    pipe.InitBuffer(bufB, NC * LBSZ);
    pipe.InitBuffer(bufP, NC * LBSZ);
    pipe.InitBuffer(bufX, NC * 2 * MM * 2);   // mode 9's whole-Xb window
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> la1 = bufA2.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lbAll = bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lxAll = bufX.Get<bfloat16_t>();

    if (mode == 5) {
        // Depth-1 load pipeline: chunk k+1's loads issue before chunk k is
        // computed and each chunk waits only its own flag (ping-pong ids), so
        // MTE2 of chunk k+1 runs under MTE1/M/FIX of chunk k instead of the
        // whole pass's loads standing in front of all its arithmetic.
        for (int32_t pass = 0; pass < PASSES; ++pass) {
            for (int32_t ch = 0; ch < nch; ++ch) {
                if (ch == 0) {
                    if (pass == 0) {
                        DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                        DataCopy(lbAll, Xb[static_cast<uint64_t>(c0) * 2 * MM],
                                 Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                    } else {
                        DataCopy(laAll, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(e21);
                }
                if (ch + 1 < nch) {
                    const int32_t nx = ch + 1;
                    if (pass == 0) {
                        DataCopy(laAll[nx * MM],
                                 Lneg[static_cast<uint64_t>(c0 + nx) * MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                        DataCopy(lbAll[nx * MM],
                                 Xb[static_cast<uint64_t>(c0 + nx) * 2 * MM],
                                 Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                    } else {
                        DataCopy(laAll[nx * MM],
                                 Xb[static_cast<uint64_t>(c0 + nx) * 2 * MM + MM],
                                 Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    }
                    SetFlag<HardEvent::MTE2_MTE1>((nx % 2) ? e21b : e21);
                }
                WaitFlag<HardEvent::MTE2_MTE1>((ch % 2) ? e21b : e21);
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> a =
                    a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                for (int32_t dd = 0; dd < KF; ++dd) {
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        LoadData(a[(mm * KF + dd) * 256], laAll[(ch * MM) + (dd * KF + mm) * 256],
                                 LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                    }
                }
                if (pass == 0) {
                    LoadDataWithTranspose(b, lbAll[ch * MM],
                                          LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                } else {
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
                SetFlag<HardEvent::M_FIX>(f0);
                WaitFlag<HardEvent::M_FIX>(f0);
                if (pass == 0) {
                    auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                    ipnz.quantPre = QuantMode_t::F322BF16;
                    ipnz.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ipnz);
                } else {
                    auto ip = FixpipeParamsV220(M, M, M, PC, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if (mode == 6) {
        for (int32_t pass = 0; pass < PASSES; ++pass) {
            kda_asm_rolling_pass(pass, nch, c0, laAll, lbAll, lpAll, a8, b8,
                                 cfall, A16, Xb, Lneg, e21, e21b, e1m, e1mb,
                                 f0, f1);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if (mode == 9) {
        // The block's Xb span [c0*2*MM, (c0+nch)*2*MM) is contiguous in GM:
        // read all of it as 4*nch ND matrices of 16 rows at a uniform 512
        // element stride.  Band 0 lands first in each chunk's 2*MM window
        // (pass 0's B), band 1 second (pass 1's A), so pass 1 has no fill and
        // the block's MTE2 is two calls: the batched Lneg fill and this one.
        DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                 Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
        DataCopy(lxAll, Xb[static_cast<uint64_t>(c0) * 2 * MM],
                 Nd2NzParams(4 * nch, 16, M, BANDE, M, 16, 1, BANDE));
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        for (int32_t pass = 0; pass < PASSES; ++pass) {
            for (int32_t ch = 0; ch < nch; ++ch) {
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> la =
                    (pass == 0) ? laAll[ch * MM] : lxAll[ch * 2 * MM + MM];
                LocalTensor<bfloat16_t> a =
                    a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                if (pass == 0) {
                    // Lneg comes in the 32-row (k-block major) NZ order, which
                    // is the permutation the four single-fractal calls spell
                    // out; band 1 of the whole-window call is the 16-row
                    // (n-block major) order, so it reads straight in one call.
                    for (int32_t dd = 0; dd < KF; ++dd) {
                        for (int32_t mm = 0; mm < KF; ++mm) {
                            LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                                     LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                        }
                    }
                    LoadDataWithTranspose(b, lxAll[ch * 2 * MM],
                                          LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                } else {
                    LoadData(a, lxAll[ch * 2 * MM + MM],
                             LoadData2dParams(0, KF * KF, 1, 0, 0, false, 0));
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
                SetFlag<HardEvent::M_FIX>(f0);
                WaitFlag<HardEvent::M_FIX>(f0);
                if (pass == 0) {
                    auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                    ipnz.quantPre = QuantMode_t::F322BF16;
                    ipnz.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ipnz);
                } else {
                    auto ip = FixpipeParamsV220(M, M, M, PC, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    const bool hoist = (mode == 1) || (mode == 2) || (mode == 3);
    const bool phases = (mode == 2) || (mode == 4);
    const bool interleave = (mode == 3);

    // pass 0 loads (always), pass 1's la hoisted when asked
    DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
             Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(lbAll[ch * MM], Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM],
                 Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
    }
    if (hoist) {
        DataCopy(la1, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                 Nd2NzParams(nch, M, M, 2 * MM, M, M, 1, MM));
    }

    for (int32_t pass = 0; pass < PASSES; ++pass) {
        if (!hoist) {
            if (pass == 1) {
                DataCopy(laAll, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                         Nd2NzParams(nch, M, M, 2 * MM, M, M, 1, MM));
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        LocalTensor<bfloat16_t> laSrc = (pass == 0 || !hoist) ? laAll : la1;

        if (phases) {
            for (int32_t ch = 0; ch < nch; ++ch) {
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> la = laSrc[ch * MM];
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
                if (pass == 0) {
                    LoadDataWithTranspose(b, lbAll[ch * MM],
                                          LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                } else {
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
            }
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<bfloat16_t> a =
                    a8[(pass * NC + ch) * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[(pass * NC + ch) * MM * 2].ReinterpretCast<bfloat16_t>();
                Mmad(cfall[(pass * NC + ch) * MM], a, b,
                     MmadParams(M, M, M, 0, false, true));
            }
            SetFlag<HardEvent::M_FIX>(f0);
            WaitFlag<HardEvent::M_FIX>(f0);
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<float> cf = cfall[(pass * NC + ch) * MM];
                if (pass == 0) {
                    auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                    ipnz.quantPre = QuantMode_t::F322BF16;
                    ipnz.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ipnz);
                } else {
                    auto ip = FixpipeParamsV220(M, M, M, PC, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
            }
        } else if (interleave) {
            // fills for the whole pass, then a legal interleave: Mmad ch+2 is
            // issued right after Fixpipe ch's wait consumed its id, so the FIX
            // pipe of ch overlaps the M pipe of ch+1 with <= 1 outstanding set
            // per id.
            for (int32_t ch = 0; ch < nch; ++ch) {
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> la = laSrc[ch * MM];
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
                if (pass == 0) {
                    LoadDataWithTranspose(b, lbAll[ch * MM],
                                          LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                } else {
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
            }
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            for (int32_t ch = 0; ch < nch; ++ch) {
                LocalTensor<bfloat16_t> a =
                    a8[(pass * NC + ch) * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[(pass * NC + ch) * MM * 2].ReinterpretCast<bfloat16_t>();
                Mmad(cfall[(pass * NC + ch) * MM], a, b,
                     MmadParams(M, M, M, 0, false, true));
                if (ch == 0) SetFlag<HardEvent::M_FIX>(f0);
                if (ch == 1) SetFlag<HardEvent::M_FIX>(f1);
            }
            for (int32_t ch = 0; ch < nch; ++ch) {
                if (ch % 2 == 0) {
                    WaitFlag<HardEvent::M_FIX>(f0);
                } else {
                    WaitFlag<HardEvent::M_FIX>(f1);
                }
                LocalTensor<float> cf = cfall[(pass * NC + ch) * MM];
                if (pass == 0) {
                    auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                    ipnz.quantPre = QuantMode_t::F322BF16;
                    ipnz.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ipnz);
                } else {
                    auto ip = FixpipeParamsV220(M, M, M, PC, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
                // re-arm: the wait above consumed id ch%2, so ch+2 may reuse it
                if (ch + 2 < nch) {
                    LocalTensor<bfloat16_t> a =
                        a8[(pass * NC + ch + 2) * MM * 2].ReinterpretCast<bfloat16_t>();
                    LocalTensor<bfloat16_t> b =
                        b8[(pass * NC + ch + 2) * MM * 2].ReinterpretCast<bfloat16_t>();
                    Mmad(cfall[(pass * NC + ch + 2) * MM], a, b,
                         MmadParams(M, M, M, 0, false, true));
                    if (ch % 2 == 0) SetFlag<HardEvent::M_FIX>(f0);
                    else SetFlag<HardEvent::M_FIX>(f1);
                }
            }
        } else {
            for (int32_t ch = 0; ch < nch; ++ch) {
                const int32_t slot = pass * NC + ch;
                LocalTensor<bfloat16_t> la = laSrc[ch * MM];
                LocalTensor<bfloat16_t> a =
                    a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                LocalTensor<bfloat16_t> b =
                    b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
                if (mode == 7 && KF == 2) {
                    // source fractals (0,2) -> L0A (0,1) and (1,3) -> (2,3):
                    // srcStride 2, consecutive destinations (a fractal is
                    // 256 bf16, so the second base is element 256)
                    LoadData(a[0], la[0], LoadData2dParams(0, 2, 2, 0, 0, false, 0));
                    LoadData(a[2 * 256], la[1 * 256],
                             LoadData2dParams(0, 2, 2, 0, 0, false, 0));
                } else if (mode == 8 && KF == 2) {
                    // the other knob for the same permutation: consecutive
                    // source fractals (0,1) -> L0A (0,2), one 512 B destination
                    // gap, so the second base is L0A fractal 1
                    LoadData(a[0], la[0], LoadData2dParams(0, 2, 1, 0, 1, false, 0));
                    LoadData(a[1 * 256], la[2 * 256],
                             LoadData2dParams(0, 2, 1, 0, 1, false, 0));
                } else {
                    for (int32_t dd = 0; dd < KF; ++dd) {
                        for (int32_t mm = 0; mm < KF; ++mm) {
                            LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                                     LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                        }
                    }
                }
                if (pass == 0) {
                    LoadDataWithTranspose(b, lbAll[ch * MM],
                                          LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
                } else if (mode == 7 && KF == 2) {
                    const int32_t src = ch * MM;
                    LoadDataWithTranspose(b, lpAll[src],
                                          LoadData2dTransposeParams(0, 2, 2, 0, 0));
                    LoadDataWithTranspose(b[2 * FRAC], lpAll[src + FRAC],
                                          LoadData2dTransposeParams(0, 2, 2, 0, 0));
                } else if (mode == 8 && KF == 2) {
                    const int32_t src = ch * MM;
                    LoadDataWithTranspose(b, lpAll[src],
                                          LoadData2dTransposeParams(0, 2, 1, 0, 1));
                    LoadDataWithTranspose(b[1 * FRAC], lpAll[src + 2 * FRAC],
                                          LoadData2dTransposeParams(0, 2, 1, 0, 1));
                } else {
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
                SetFlag<HardEvent::M_FIX>(f0);
                WaitFlag<HardEvent::M_FIX>(f0);
                if (pass == 0) {
                    auto ipnz = FixpipeParamsV220(M, M, M, M, false);
                    ipnz.quantPre = QuantMode_t::F322BF16;
                    ipnz.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_NZ>(lpAll[ch * MM], cf, ipnz);
                } else {
                    auto ip = FixpipeParamsV220(M, M, M, PC, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
