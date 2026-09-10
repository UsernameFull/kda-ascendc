// d4 = v_newT[128,16] @ kg^T^T = v_newT @ kg
// 8x fully-unrolled Mmad(16,128,16, transB=true), LoadData(false) for ALL
// Each block: fresh A+B LoadData, fresh C accumulation, Fixpipe to row-block
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;

extern "C" __global__ __aicore__ void k2_d4_unroll8(GM_ADDR pvnewT, GM_ADDR pkgT, GM_ADDR pd4, GM_ADDR ws, GM_ADDR tiling) {
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
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)pkgT);
    gC.SetGlobalBuffer((__gm__ float *)pd4);

    // A = v_newT[128,16], B = kgT[128,16]
    // 8 blocks, each: A_m = v_newT[m*16:(m+1)*16, :], B = kgT (same for all)
    // Mmad(16,128,16, false, true) = A_m @ B^T = v_newT[m] @ kg

#define D4_BLOCK(m) { \
    LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>(); \
    LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>(); \
    DataCopy(la, gA[m * 256], Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0)); \
    DataCopy(lb, gB, Nd2NzParams(1, 128, 16, 0, 16, 128, 1, 0)); \
    SetFlag<HardEvent::MTE2_MTE1>(ev21); \
    WaitFlag<HardEvent::MTE2_MTE1>(ev21); \
    l1AQue.EnQue(la); l1BQue.EnQue(lb); \
    la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>(); \
    LoadData(l0a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0)); \
    LoadData(l0b, lb, LoadData2dParams(0, 8, 1, 0, 0, false, 0)); \
    SetFlag<HardEvent::MTE1_M>(ev1m); \
    WaitFlag<HardEvent::MTE1_M>(ev1m); \
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 16, 0, false, true)); \
    SetFlag<HardEvent::M_FIX>(evmfix); \
    WaitFlag<HardEvent::M_FIX>(evmfix); \
    for (int nb = 0; nb < 8; nb++) { \
        auto ip = FixpipeParamsV220(16, 16, 1, 128, false); \
        ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0; \
        Fixpipe<float, float, CFG_ROW_MAJOR>(gC[m * 2048 + nb * 16], l0cf[nb * 256], ip); \
    } \
    SetFlag<HardEvent::FIX_M>(evfixm); \
    WaitFlag<HardEvent::FIX_M>(evfixm); \
    l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb); \
    AscendC::PipeBarrier<PIPE_ALL>(); \
}
    D4_BLOCK(0)
    D4_BLOCK(1)
    D4_BLOCK(2)
    D4_BLOCK(3)
    D4_BLOCK(4)
    D4_BLOCK(5)
    D4_BLOCK(6)
    D4_BLOCK(7)
#undef D4_BLOCK
    l0CQue.FreeTensor(l0cf);
}
