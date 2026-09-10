// d3 = Aqk[16,16] @ v_new^T^T = Aqk @ v_new
// B = v_newT [128,16] (pre-transposed by AIV)
// LoadData(false) + Mmad(transB=true) = A @ B^T = Aqk @ v_new
// ZERO LoadData(transpose=true) calls
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;

extern "C" __global__ __aicore__ void k2_d3_btrans(GM_ADDR paqk, GM_ADDR pvnewT, GM_ADDR pd3, GM_ADDR ws, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, 16 * 16 * 2);
    pipe.InitBuffer(l1BQue, 1, 128 * 16 * 2);
    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, 16 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 8 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 64 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gA, gB;
    GlobalTensor<float> gC;
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)paqk);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gC.SetGlobalBuffer((__gm__ float *)pd3);

    // A = Aqk [16,16]: Nd2Nz(n=16, d=16, srcD=16, C0Stride=16, NStride=1)
    // B = v_newT [128,16]: Nd2Nz(n=128, d=16, srcD=16, C0Stride=128, NStride=1)
    //   B logical = [K=128, N=16], transB=true → A @ B^T = [16,16]@[16,128] → [16,128]
    LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
    DataCopy(la, gA, Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
    DataCopy(lb, gB, Nd2NzParams(1, 128, 16, 0, 16, 128, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(la); l1BQue.EnQue(lb);
    la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
    // A: [16,16] = 1 fractal → repeat=1, transpose=FALSE
    LoadData(l0a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    // B: [128,16] = 8 fractals (128/16=8 K-blocks, 1 N-block) → repeat=8, transpose=FALSE
    LoadData(l0b, lb, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    // Mmad(M=16, N=128, K=16, transB=true) → A @ B^T = Aqk @ v_new
    // Wait: B=[128,16] means K=128, N=16. With transB=true: B^T=[16,128]
    // Mmad(M=16, N=128, K=128, false, true) → A[16,16] @ B^T[16,128]? No!
    // M=16, N=128, K must match A's K and B^T's K
    // A=[16,16] → M=16, K=16
    // B=[128,16] → K=128, N=16. B^T=[16,128] → K=16, N=128
    // So Mmad(16, 128, 128, false, true) → K=128 but A has K=16 → MISMATCH!
    // Actually with transB=true: B is [K=128, N=16], B^T=[K=16, N=128]
    // But Mmad K must match A's K (=16) and B^T's K (=16). OK!
    // Mmad params: M=16, N=128, K=128? Or K=16?
    // B original [128,16]: physical K=128, N=16
    // transB=true: logical B = B^T = [K=16, N=128]
    // But L0B physically has [K=128, N=16] fractals
    // Mmad reads L0B as transposed: treats physical [K=128,N=16] as [K=16,N=128]
    // Wait that doesn't work - K and N are different
    // 
    // Actually transB=true means: the B matrix in L0B is already transposed
    // So if L0B has [128 rows, 16 cols] = [K=128, N=16] in physical
    // transB=true tells Mmad to treat it as [N=128, K=16] → B^T = [K=16, N=128]
    // Then Mmad computes A[16,16] @ B^T[16,128] → [16,128] ✓
    // Mmad params: M=16, N=128, K=16
    // But B physical has 128×16 = 8 K-blocks × 1 N-block
    // With transB=true: these become 1 K-block × 8 N-blocks
    // K=16 (1 block), N=128 (8 blocks) → Mmad(16, 128, 16, false, true) ✓
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 16, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    for (int b = 0; b < 8; b++) {
        auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
        ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
    }
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
    l0CQue.FreeTensor(l0cf);
}
