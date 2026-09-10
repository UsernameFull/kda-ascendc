// k2_glue_final: out = scale*d2+d3, h_new = h*exp2(g_last)[broadcast]+d4 (AIV-only)
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
constexpr int32_t H = 128 * 128;
constexpr float SCALE = 0.08838834764f;
constexpr float LN2 = 0.6931471805599453f;

extern "C" __global__ __aicore__ void k2_glue_final_kernel(
    GM_ADDR pg_last, GM_ADDR ph, GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4,
    GM_ADDR pout, GM_ADDR phnew, int32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> ubD2, ubD3, ubOut, ubOutb, ubG, ubG2, ubT, ubH, ubD4;
    pipe.InitBuffer(ubD2, E * 4);
    pipe.InitBuffer(ubD3, E * 4);
    pipe.InitBuffer(ubOut, E * 4);
    pipe.InitBuffer(ubOutb, E * 2);
    pipe.InitBuffer(ubG, 128 * 4);
    pipe.InitBuffer(ubG2, 128 * 4);
    pipe.InitBuffer(ubT, 128 * 4);
    pipe.InitBuffer(ubH, H * 4);
    pipe.InitBuffer(ubD4, H * 4);
    LocalTensor<float> tD2 = ubD2.Get<float>(), tD3 = ubD3.Get<float>(), tOut = ubOut.Get<float>(),
        tG = ubG.Get<float>(), tG2 = ubG2.Get<float>(), tT = ubT.Get<float>(),
        tH = ubH.Get<float>(), tD4 = ubD4.Get<float>();
    LocalTensor<bfloat16_t> tOutb = ubOutb.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gOut;
    GlobalTensor<float> gG, gH, gD2, gD3, gD4, gHnew;
    gOut.SetGlobalBuffer((__gm__ bfloat16_t *)pout);
    gG.SetGlobalBuffer((__gm__ float *)pg_last);
    gH.SetGlobalBuffer((__gm__ float *)ph);
    gD2.SetGlobalBuffer((__gm__ float *)pd2);
    gD3.SetGlobalBuffer((__gm__ float *)pd3);
    gD4.SetGlobalBuffer((__gm__ float *)pd4);
    gHnew.SetGlobalBuffer((__gm__ float *)phnew);

    DataCopy(tD2, gD2, E);
    DataCopy(tD3, gD3, E);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);
    Muls(tOut, tD2, SCALE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Add(tOut, tOut, tD3, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tOutb, tOut, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gOut, tOutb, E);

    // exp2(g_last) coef
    DataCopy(tG, gG, 128);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);
    Muls(tG2, tG, LN2, 128);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tG2, 128);
    AscendC::PipeBarrier<PIPE_V>();

    // h_new = h*coef + d4 (single big load)
    DataCopy(tH, gH[0], DataCopyParams(16, 128, 0, 0));
    DataCopy(tD4, gD4[0], DataCopyParams(16, 128, 0, 0));
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::PipeBarrier<PIPE_V>();
    for (int r = 0; r < 128; r++) {
        Mul(tH[r * 128], tH[r * 128], tT, 128);
        AscendC::PipeBarrier<PIPE_V>();
    }
    Add(tH, tH, tD4, H);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gHnew[0], tH, DataCopyParams(16, 128, 0, 0));
}
