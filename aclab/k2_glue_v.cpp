// k2_glue_v: v_new = u - d1 (AIV-only, verified vecop pattern)
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;

extern "C" __global__ __aicore__ void k2_glue_v_kernel(
    GM_ADDR pu, GM_ADDR pd1, GM_ADDR pvnew, int32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> ubUb, ubU, ubD1, ubV, ubVb;
    pipe.InitBuffer(ubUb, E * 2);
    pipe.InitBuffer(ubU, E * 4);
    pipe.InitBuffer(ubD1, E * 4);
    pipe.InitBuffer(ubV, E * 4);
    pipe.InitBuffer(ubVb, E * 2);
    LocalTensor<bfloat16_t> tUb = ubUb.Get<bfloat16_t>();
    LocalTensor<float> tU = ubU.Get<float>(), tD1 = ubD1.Get<float>(), tV = ubV.Get<float>();
    LocalTensor<bfloat16_t> tVb = ubVb.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gU, gVnew;
    GlobalTensor<float> gD1;
    gU.SetGlobalBuffer((__gm__ bfloat16_t *)pu);
    gVnew.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gD1.SetGlobalBuffer((__gm__ float *)pd1);

    DataCopy(tUb, gU, E);
    DataCopy(tD1, gD1, E);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);
    Cast(tU, tUb, RoundMode::CAST_NONE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Sub(tV, tU, tD1, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tVb, tV, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gVnew, tVb, E);
}
