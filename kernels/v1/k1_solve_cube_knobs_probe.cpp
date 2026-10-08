// K1 stage 3b twin for the cube's MTE2 load shape (docs section 11.63).
//
// `kda_solve_wu_cube_kernel` with one extra runtime argument.  Everything
// else - queue protocol, LoadData crossings, the transpose walk, Mmad,
// Fixpipe, slot arithmetic, InitBuffer sizes - is byte-for-byte the shipped
// kernel, so a mode that keeps W/U bit-identical is a load-form change and
// nothing else.  The on-board account (msprof op PipeUtilization, archive
// /data/models/Qwen3-4B/kda_msprof_20261008_cube) reads the shipped block
// wall as 4.382 us of which MTE2 is 3.271 (74.6%): 16 calls of 4 KB (the
// per-band RHS loop) plus 2 of 8 KB (A16) per block.
//
//   loadMode 0  shipped       KF Nd2Nz calls per chunk-pass, per-chunk A16
//   loadMode 1  band-merged   one Nd2Nz (ndNum = KF) per chunk-pass: the four
//               [16, D] bands of a chunk are one strided ND run
//   loadMode 2  block-merged  mode 1 plus the block's chunks in one call per
//               pass (ndNum = KF * nch) and the A16 fill in one call per block
//               (ndNum = nch)
//
// The merged calls keep each matrix's internal layout (same nValue / dValue /
// dstNzC0Stride / dstNzNStride as the one-matrix calls) and span the matrices
// with srcNdMatrixStride / dstNzMatrixStride = the ship-class per-band spacing
// (16 * D elements), which is why mode 1's L1 bytes - and the transpose walk
// that follows them - do not move at all; mode 2 additionally relies on the
// qb slots being L1-adjacent, which the bit gate checks.
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
constexpr int32_t BAND_E = 16 * D;   // elements of one [16, D] band

extern "C" __global__ __aicore__ void kda_solve_cube_knobs_probe(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C,
    int32_t a16Mode, int32_t loadMode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    const bool a16res = (a16Mode != 0);
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, M * K * 2);
    pipe.InitBuffer(qb, NC, D * K * 2);
    TBuf<TPosition::B1> bufA16;
    pipe.InitBuffer(bufA16, NC * M * K * 2);
    LocalTensor<float> cfall(TPosition::CO1, 0, 2 * NC * M * D);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 2 * NC * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * NC * D * K * 2);
    LocalTensor<bfloat16_t> a16l1 = bufA16.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));

    if (a16res) {
        // One read of the block's A16 for both passes.
        for (int32_t ch = 0; ch < nch; ++ch) {
#if KDA_CHUNK > 16
            if (loadMode >= 2 && nch > 1) {
                break;  // the whole block in one call, below
            }
            DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K],
                     Nd2NzParams(1, M, K, 0, K, M, 1, 0));
#else
            DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
#endif
        }
#if KDA_CHUNK > 16
        if (loadMode >= 2 && nch > 1) {
            DataCopy(a16l1, A16[static_cast<uint64_t>(c0) * M * K],
                     Nd2NzParams(nch, M, K, M * K, K, M, 1, M * K));
        }
#endif
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }
    for (int32_t pass = 0; pass < 2; ++pass) {
        // Issue every chunk's L1 load before touching L0: the GM round trips of
        // one chunk then overlap the arithmetic of the previous one.  Each pass
        // loads its own right-hand side (pass 0 = rk -> W, pass 1 = rv -> U).
        for (int32_t ch = 0; ch < nch; ++ch) {
            if (!a16res) {
                auto la = qa.AllocTensor<bfloat16_t>();
                DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K],
                         Nd2NzParams(1, M, K, 0, K, M, 1, 0));
                qa.EnQue(la);
            }
#if KDA_CHUNK > 16
            GlobalTensor<bfloat16_t> &rhs = (pass == 0) ? Rk : Rv;
            if (loadMode >= 2 && nch > 1) {
                // The block's chunks in one call: matrices [ch][mm] at the
                // uniform band stride, landing in the two L1-adjacent slots.
                if (ch == 0) {
                    auto lb0 = qb.AllocTensor<bfloat16_t>();
                    auto lb1 = qb.AllocTensor<bfloat16_t>();
                    DataCopy(lb0, rhs[static_cast<uint64_t>(c0) * M * D],
                             Nd2NzParams(nch * KF, 16, D, BAND_E, D, 16, 1, BAND_E));
                    qb.EnQue(lb0);
                    qb.EnQue(lb1);
                }
            } else if (loadMode >= 1) {
                auto lb = qb.AllocTensor<bfloat16_t>();
                DataCopy(lb, rhs[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(KF, 16, D, BAND_E, D, 16, 1, BAND_E));
                qb.EnQue(lb);
            } else {
                auto lb = qb.AllocTensor<bfloat16_t>();
                for (int32_t mm = 0; mm < KF; ++mm) {
                    DataCopy(lb[mm * DF * 256],
                             rhs[static_cast<uint64_t>(c0 + ch) * M * D + mm * 16 * D],
                             Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
                }
                qb.EnQue(lb);
            }
#else
            auto lb = qb.AllocTensor<bfloat16_t>();
            if (pass == 0) {
                DataCopy(lb, Rk[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            } else {
                DataCopy(lb, Rv[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            }
            qb.EnQue(lb);
#endif
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la;
            if (a16res) {
                la = a16l1[ch * M * K];
            } else {
                la = qa.DeQue<bfloat16_t>();
            }
            auto lb = qb.DeQue<bfloat16_t>();
            // L0A/L0B are indexed by the pass as well: the previous pass's mmads
            // may still be reading the same chunk slot.
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
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(W[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            } else {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(U[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            }
            if (!a16res) {
                qa.FreeTensor(la);
            }
            qb.FreeTensor(lb);
        }
    }
}
