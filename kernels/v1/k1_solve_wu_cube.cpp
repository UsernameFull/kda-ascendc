// K1 stage 3b: the two solve right-hand sides on the Cube.
//
//   w = bf16(A_inv) @ (k * beta * exp2(gate))   A16 [16,16] bf16, rk [16,128] bf16
//   u = bf16(A_inv) @ (v * beta)                A16 [16,16] bf16, rv [16,128] bf16
//
// This replaces the vector row-broadcast + column reduction (MatVec2) that used
// to run inside kda_solve_wu_kernel.  rk/rv are consumed in their natural
// [16,128] layout: Nd2Nz turns them into 16x16 fractals in L1 and
// LoadDataWithTranspose transposes each fractal into the B2 zN layout the Cube
// wants, so no transposed staging buffer is needed.  One AIC block per chunk.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128;

extern "C" __global__ __aicore__ void kda_solve_wu_cube_kernel(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c = GetBlockIdx();
    if (c >= C) return;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> qa, qb;
    pipe.InitBuffer(qa, 1, M * K * 2);
    pipe.InitBuffer(qb, 1, D * K * 2);
    TQue<QuePosition::CO1, 1> qc;
    pipe.InitBuffer(qc, 1, M * D * 4);
    LocalTensor<float> cf = qc.AllocTensor<float>();
    LocalTensor<uint8_t> a8(TPosition::A2, 0, M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, D * K * 2);
    LocalTensor<bfloat16_t> a = a8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
    auto la = qa.AllocTensor<bfloat16_t>();
    auto lb = qb.AllocTensor<bfloat16_t>();
    DataCopy(la, A16[static_cast<uint64_t>(c) * M * K], M * K);
    DataCopy(lb, Rk[static_cast<uint64_t>(c) * M * D], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
    qa.EnQue(la); qb.EnQue(lb);
    la = qa.DeQue<bfloat16_t>(); lb = qb.DeQue<bfloat16_t>();
    LoadData(a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, 8, 1, 0, 0));
    SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
    Mmad(cf, a, b, MmadParams(M, D, K, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
    {
        auto ip = FixpipeParamsV220(D, M, 16, D, false);
        ip.quantPre = QuantMode_t::F322BF16;
        ip.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(W[static_cast<uint64_t>(c) * M * D], cf, ip);
    }
    SetFlag<HardEvent::FIX_M>(efm); WaitFlag<HardEvent::FIX_M>(efm);
    qa.FreeTensor(la);
    la = qa.AllocTensor<bfloat16_t>();
    DataCopy(lb, Rv[static_cast<uint64_t>(c) * M * D], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
    qa.EnQue(la); qb.EnQue(lb);
    la = qa.DeQue<bfloat16_t>(); lb = qb.DeQue<bfloat16_t>();
    LoadData(a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, 8, 1, 0, 0));
    SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
    Mmad(cf, a, b, MmadParams(M, D, K, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
    {
        auto ip = FixpipeParamsV220(D, M, 16, D, false);
        ip.quantPre = QuantMode_t::F322BF16;
        ip.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(U[static_cast<uint64_t>(c) * M * D], cf, ip);
    }
    SetFlag<HardEvent::FIX_M>(efm); WaitFlag<HardEvent::FIX_M>(efm);
    qa.FreeTensor(la); qb.FreeTensor(lb); qc.FreeTensor(cf);
}
