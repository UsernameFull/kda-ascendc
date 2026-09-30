// Timing probe (NOT a production kernel): what is the *marginal price per GB*
// of the Cube solve's right-hand-side reads?
//
// The ledger carries two different prices for a byte that crosses a launch
// boundary, and nothing in it reconciles them:
//
//   * section 11.12 measured GM traffic that a *later block of the same
//     launch* re-reads at 0.2 ms/GB (an L2 hit);
//   * section 11.25 route C measured 2.01 GB of cross-launch handoff costing
//     2.104 ms, i.e. 0.955 GB/ms = 1.047 ms/GB;
//   * section 11.35 priced the Cube's *second* A16 read at 1439 GB/s marginal
//     against the first read's 839 GB/s cold rate - and that second read is
//     also cross-launch, issued one pass later by the same block.
//
// Every arithmetic projection built on the cheap price is optimistic by up to
// 5x, and the one that matters here is section 11.28's rejection of the Level
// 2/3 window streaming: it priced 0.60 GB of extra handoff at 0.12 + 0.08 ms
// using the L2 rate.  At the route-C rate the same bytes are 0.52 + 0.34 ms.
// The sign of that correction decides whether the verdict stands, so this probe
// measures the rate directly instead of inferring it.
//
// It does that the way sections 11.35/11.36 did: the probe *is* the kernel.
// ``kernels/v1/k1_solve_wu_cube.cpp`` reads two right-hand sides -
// rk (pass 0) and rv (pass 1), each [CHUNK, 128] bf16 - so at
// [1,8192,96,128]/C=64 it pulls 402.65 MB of cross-launch bytes per call, in
// 16 Nd2Nz DataCopy calls per block.  Those bytes were written by earlier
// launches (preprocess / the wide solve) and the working set is 25x the L2, so
// the expectation is that they are HBM traffic; the arms below say what that
// costs, in this kernel, on this device, today.
//
// Arms (all four keep A16 resident in L1, i.e. a16Mode=1, the shipped default,
// and every other instruction - the qa/qb queue protocol, both LoadData
// crossings, the Mmad, the Fixpipe, the L0/L0C slot arithmetic - is identical):
//
//   0  control: RHS read from this block's own chunks, ``rhs[c0 + ch]``.  This
//      is the shipped address pattern, so the arm is production.
//   1  L2-hot: same call count, same shapes, same instruction stream, but the
//      address is ``rhs[(c0 % hot) + ch]``, so the whole grid reads one
//      ``hot``-chunk window (hot=32 => 1.0 MB across both RHS tensors) that
//      stays resident.  (arm 0 - arm 1) is the price of 402.65 MB of cold
//      cross-launch bytes; ``hot`` is an argument, not a constant, so the
//      harness can sweep it and show the price is a property of the bytes and
//      not of the window size.  The window is 16 chunk bases x KF bands of
//      4 KB, i.e. deliberately *not* the single-address pattern section 10
//      caught being pathologically slow.
//   2  floor: the RHS DataCopy calls are gone entirely (the L1 slot is still
//      AllocTensor'd, EnQue'd, DeQue'd and crossed into L0B, so the queue
//      protocol and the transpose stay).  (arm 1 - arm 2) is the price of the
//      calls themselves with no bytes behind them.
//   3  half: pass 0 reads cold, pass 1 reads hot - exactly half of arm 0's cold
//      bytes.  This is the linearity point: if the price is per-byte, (0-3) and
//      (3-1) are equal and each is half of (0-1).  Section 11.35 ran the same
//      check on the A16 re-read and found it linear.
//
// Arms 1/2/3 compute garbage on purpose (they multiply A_inv by the wrong
// right-hand side, or by an uninitialised one), so there is no bit-identity
// gate here - the same arrangement as mode 2 of
// kernels/v1/k1_solve_wu_cube_a16_probe.cpp.  They are measurement arms; the
// caller ties arm 0 to reality by replaying the shipped
// ``kda_solve_wu_cube_kernel`` in the same process and printing it beside them.
//
// Transcribed 2026-09-28 from k1_solve_wu_cube.cpp; the RHS load path is the
// only difference from the shipped kernel.
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

extern "C" __global__ __aicore__ void kda_solve_cube_rhs_probe(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU,
    int32_t C, int32_t mode, int32_t hot) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    // Arms 1/3 take their hot RHS from this window; it is computed on every arm
    // so the select below is the same instruction everywhere.
    const int32_t h0 = c0 % hot;
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

    // A16 resident: one read of the block's tiles for both passes, which is
    // what the shipped kernel does at KDA_CUBE_A16_RESIDENT=1.
    for (int32_t ch = 0; ch < nch; ++ch) {
#if KDA_CHUNK > 16
        DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K],
                 Nd2NzParams(1, M, K, 0, K, M, 1, 0));
#else
        DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
#endif
    }
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);

    for (int32_t pass = 0; pass < 2; ++pass) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto lb = qb.AllocTensor<bfloat16_t>();
            if (mode != 2) {
                GlobalTensor<bfloat16_t> &rhs = (pass == 0) ? Rk : Rv;
                // mode 0: both passes cold (shipped).  mode 1: both hot.
                // mode 3: pass 0 cold, pass 1 hot - half the cold bytes.
                const bool cold = (mode == 0) || (mode == 3 && pass == 0);
                const int32_t base = cold ? (c0 + ch) : (h0 + ch);
#if KDA_CHUNK > 16
                for (int32_t mm = 0; mm < KF; ++mm) {
                    DataCopy(lb[mm * DF * 256],
                             rhs[static_cast<uint64_t>(base) * M * D + mm * 16 * D],
                             Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
                }
#else
                DataCopy(lb, rhs[static_cast<uint64_t>(base) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
#endif
            }
            qb.EnQue(lb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la = a16l1[ch * M * K];
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
            qb.FreeTensor(lb);
        }
    }
}
