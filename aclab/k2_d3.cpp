// k2_d3: d3 = Aqk[16,16] @ v_new[16,128] -> [16,128] fp32 (AIC, smoke P3 pattern)
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void k2_d3_kernel(
    GM_ADDR pAqk, GM_ADDR pV, GM_ADDR pC3, int32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, 16 * 16 * 2);
    pipe.InitBuffer(l1BQue, 1, 16 * 128 * 2);
    LocalTensor<bfloat16_t> l1a = l1AQue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> l1b = l1BQue.AllocTensor<bfloat16_t>();
    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, 16 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 1 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 8 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gA, gB;
    GlobalTensor<float> gC;
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pAqk);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)pV);
    gC.SetGlobalBuffer((__gm__ float *)pC3);

    // P3: Aqk[16,16] @ v_new[16,128] (k=16, n=128)
    DataCopy(l1a, gA, Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
    DataCopy(l1b, gB, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(l1a);
    l1BQue.EnQue(l1b);
    LocalTensor<bfloat16_t> l1a2 = l1AQue.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> l1b2 = l1BQue.DeQue<bfloat16_t>();
    LoadData(l0a, l1a2, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadData(l0b, l1b2, LoadData2dParams(0, 8, 1, 0, 0, true, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 16, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    for (int b = 0; b < 8; b++) {
        auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
    }
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    l1AQue.FreeTensor(l1a);
    l1BQue.FreeTensor(l1b);
    l0CQue.FreeTensor(l0cf);
}
