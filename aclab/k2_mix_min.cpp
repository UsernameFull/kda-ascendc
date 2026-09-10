// MIX cross-core data handoff test: AIC computes d1=w@h^T -> GM,
// AIV reads d1, computes v_new=u-d1 -> GM. Verifies AIC->AIV data visibility.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;

extern "C" __global__ __aicore__ void k2_mix_min(GM_ADDR pw, GM_ADDR ph, GM_ADDR pu,
                                                 GM_ADDR pd1, GM_ADDR pvnew, GM_ADDR ws, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C220_CUBE__)
    // ---- AIC: d1 = w @ h^T -> GM pd1 ----
    TPipe pipe;
    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, 16 * 128 * 2);
    pipe.InitBuffer(l1BQue, 1, 128 * 128 * 2);
    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, 16 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 8 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 64 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gA, gB;
    GlobalTensor<float> gD1;
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pw);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);
    gD1.SetGlobalBuffer((__gm__ float *)pd1);
    LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
    DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(la); l1BQue.EnQue(lb);
    la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
    LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    for (int b = 0; b < 8; b++) {
        auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
        ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gD1[b * 16], l0cf[b * 256], ip);
    }
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb); l0CQue.FreeTensor(l0cf);
    // signal AIV: d1 ready
    CrossCoreSetFlag<2, PIPE_MTE3>(8);
    // wait AIV: v_new ready
    CrossCoreWaitFlag<2, PIPE_MTE3>(9);
#elif defined(__DAV_C220_VEC__)
    // ---- AIV: wait d1, compute v_new = u - d1 -> GM pvnew ----
    CrossCoreWaitFlag<2, PIPE_MTE2>(8);
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
    // signal AIC: v_new ready
    CrossCoreSetFlag<2, PIPE_MTE3>(9);
#endif
}
