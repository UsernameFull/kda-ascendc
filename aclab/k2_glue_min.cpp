// M1.3 AIV glue kernel (AIV-only, verified vecop pattern).
// Inputs (GM): u[16,128] bf16, g_last[128] fp32, h[128,128] fp32,
//              d1[16,128] fp32, d2[16,128] fp32, d3[16,128] fp32, d4[128,128] fp32
// Outputs (GM): vnew[16,128] bf16, out[16,128] bf16, hnew[128,128] fp32
// Math:
//   v_new = u - d1
//   out   = scale*d2 + d3
//   h_new = h*exp2(g_last)[broadcast] + d4
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t E = 16 * 128;
constexpr float SCALE = 0.08838834764f;
constexpr float LN2 = 0.6931471805599453f;
constexpr int32_t H = 128 * 128;   // full state elements

extern "C" __global__ __aicore__ void k2_glue_min_kernel(
    GM_ADDR pu, GM_ADDR pg_last, GM_ADDR ph, GM_ADDR pd1, GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4,
    GM_ADDR pvnew, GM_ADDR pout, GM_ADDR phnew, int32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    // small 16x128 buffers (fp32 8KB each, bf16 4KB)
    TBuf<TPosition::VECIN> ubUb, ubU, ubD1, ubD2, ubD3, ubV, ubVb, ubOut, ubOutb, ubG, ubG2, ubT;
    // big 128x128 buffers (fp32 64KB each)
    TBuf<TPosition::VECIN> ubH, ubD4;
    pipe.InitBuffer(ubUb, E * 2);
    pipe.InitBuffer(ubU, E * 4);
    pipe.InitBuffer(ubD1, E * 4);
    pipe.InitBuffer(ubD2, E * 4);
    pipe.InitBuffer(ubD3, E * 4);
    pipe.InitBuffer(ubV, E * 4);
    pipe.InitBuffer(ubVb, E * 2);
    pipe.InitBuffer(ubOut, E * 4);
    pipe.InitBuffer(ubOutb, E * 2);
    pipe.InitBuffer(ubG, 128 * 4);
    pipe.InitBuffer(ubG2, 128 * 4);
    pipe.InitBuffer(ubT, 128 * 4);
    pipe.InitBuffer(ubH, H * 4);
    pipe.InitBuffer(ubD4, H * 4);

    LocalTensor<bfloat16_t> tUb = ubUb.Get<bfloat16_t>();
    LocalTensor<float> tU = ubU.Get<float>(), tD1 = ubD1.Get<float>(), tD2 = ubD2.Get<float>(),
        tD3 = ubD3.Get<float>(), tV = ubV.Get<float>(), tOut = ubOut.Get<float>(),
        tG = ubG.Get<float>(), tG2 = ubG2.Get<float>(), tT = ubT.Get<float>(),
        tH = ubH.Get<float>(), tD4 = ubD4.Get<float>();
    LocalTensor<bfloat16_t> tVb = ubVb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> tOutb = ubOutb.Get<bfloat16_t>();

    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();

    GlobalTensor<bfloat16_t> gU, gVnew, gOut;
    GlobalTensor<float> gG, gH, gD1, gD2, gD3, gD4, gHnew;
    gU.SetGlobalBuffer((__gm__ bfloat16_t *)pu);
    gVnew.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t *)pout);
    gG.SetGlobalBuffer((__gm__ float *)pg_last);
    gH.SetGlobalBuffer((__gm__ float *)ph);
    gD1.SetGlobalBuffer((__gm__ float *)pd1);
    gD2.SetGlobalBuffer((__gm__ float *)pd2);
    gD3.SetGlobalBuffer((__gm__ float *)pd3);
    gD4.SetGlobalBuffer((__gm__ float *)pd4);
    gHnew.SetGlobalBuffer((__gm__ float *)phnew);

    // ---- load 16x128 inputs ----
    DataCopy(tUb, gU, E);
    DataCopy(tD1, gD1, E);
    DataCopy(tD2, gD2, E);
    DataCopy(tD3, gD3, E);
    DataCopy(tG, gG, 128);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);

    Cast(tU, tUb, RoundMode::CAST_NONE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Sub(tV, tU, tD1, E);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(tOut, tD2, SCALE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Add(tOut, tOut, tD3, E);
    AscendC::PipeBarrier<PIPE_V>();

    Cast(tVb, tV, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gVnew, tVb, E);
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    Cast(tOutb, tOut, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gOut, tOutb, E);

    // ---- exp2(g_last) coefficients ----
    Muls(tG2, tG, LN2, 128);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tG2, 128);
    AscendC::PipeBarrier<PIPE_V>();

    // ---- h update: h_new = h*exp2(g_last)[broadcast] + d4 ----
    DataCopy(tH, gH[0], DataCopyParams(16, 128, 0, 0));
    DataCopy(tD4, gD4[0], DataCopyParams(16, 128, 0, 0));
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::PipeBarrier<PIPE_V>();
    for (int r = 0; r < 128; r++) {
        Mul(tH[r * 128], tH[r * 128], tT, 128);   // h*coef (in-place row)
        AscendC::PipeBarrier<PIPE_V>();
    }
    Add(tH, tH, tD4, H);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gHnew[0], tH, DataCopyParams(16, 128, 0, 0));
    AscendC::PipeBarrier<PIPE_MTE3>();
}
