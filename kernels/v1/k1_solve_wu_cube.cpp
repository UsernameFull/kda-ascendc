// K1 stage 3b: the two solve right-hand sides on the Cube.
//
//   w = bf16(A_inv) @ (k * beta * exp2(gate))   A16 [16,16] bf16, rk [16,128] bf16
//   u = bf16(A_inv) @ (v * beta)                A16 [16,16] bf16, rv [16,128] bf16
//
// This replaces the vector row-broadcast + column reduction (MatVec2) that used
// to run inside kda_solve_wu_kernel.  rk/rv are consumed in their natural
// [16,128] layout: Nd2Nz turns them into 16x16 fractals in L1 and
// LoadDataWithTranspose transposes each fractal into the B2 zN layout the Cube
// wants, so no transposed staging buffer is needed.
//
// One AIC block handles NCHUNK chunks: the matmul itself is far too small to
// hide a block's fixed cost (measured 1.26 us per block at [1,8192,32]), so the
// loads of every chunk are issued up front and the mmads follow.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128;
#ifndef KDA_WU_NCHUNK
#define KDA_WU_NCHUNK 4
#endif
constexpr int32_t NC = KDA_WU_NCHUNK;

extern "C" __global__ __aicore__ void kda_solve_wu_cube_kernel(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, M * K * 2);
    pipe.InitBuffer(qb, NC, D * K * 2);
    TQue<QuePosition::CO1, NC> qc;
    pipe.InitBuffer(qc, NC, M * D * 4);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 2 * NC * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * NC * D * K * 2);
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));

    for (int32_t pass = 0; pass < 2; ++pass) {
        // Issue every chunk's L1 load before touching L0: the GM round trips of
        // one chunk then overlap the arithmetic of the previous one.  Each pass
        // loads its own right-hand side (pass 0 = rk -> W, pass 1 = rv -> U).
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto la = qa.AllocTensor<bfloat16_t>();
            auto lb = qb.AllocTensor<bfloat16_t>();
            DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
            if (pass == 0) {
                DataCopy(lb, Rk[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            } else {
                DataCopy(lb, Rv[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            }
            qa.EnQue(la);
            qb.EnQue(lb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            auto la = qa.DeQue<bfloat16_t>();
            auto lb = qb.DeQue<bfloat16_t>();
            // L0A/L0B are indexed by the pass as well: the previous pass's mmads
            // may still be reading the same chunk slot.
            const int32_t slot = pass * NC + ch;
            LocalTensor<bfloat16_t> a = a8[slot * M * K * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b = b8[slot * D * K * 2].ReinterpretCast<bfloat16_t>();
            LoadData(a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, 8, 1, 0, 0));
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            LocalTensor<float> cf = qc.AllocTensor<float>();
            Mmad(cf, a, b, MmadParams(M, D, K, 0, false, true));
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            auto ip = FixpipeParamsV220(D, M, 16, D, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            if (pass == 0) {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(W[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            } else {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(U[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            }
            SetFlag<HardEvent::FIX_M>(efm);
            WaitFlag<HardEvent::FIX_M>(efm);
            qc.FreeTensor(cf);
            qa.FreeTensor(la);
            qb.FreeTensor(lb);
        }
    }
}
