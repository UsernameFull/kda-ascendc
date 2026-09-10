// MIX1: AIC d1=w@h^T -> flag8 -> AIV v_new/qg/kg -> flag9.
// AIV buffers are independent (no phase-2), so no aliasing.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
constexpr float LN2 = 0.6931471805599453f;

extern "C" __global__ __aicore__ void k2_mix1(GM_ADDR pw, GM_ADDR ph, GM_ADDR pu,
    GM_ADDR pq, GM_ADDR pk, GM_ADDR pg, GM_ADDR pglast,
    GM_ADDR pd1, GM_ADDR pvnew, GM_ADDR pvnewT, GM_ADDR pqg, GM_ADDR pkg, GM_ADDR ws, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C220_CUBE__)
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
    CrossCoreSetFlag<2, PIPE_MTE3>(8);
    CrossCoreWaitFlag<2, PIPE_MTE3>(9);
#elif defined(__DAV_C220_VEC__)
    CrossCoreWaitFlag<2, PIPE_MTE2>(8);
    TPipe pipe;
    TBuf<TPosition::VECIN> ubUb, ubU, ubD1, ubV, ubVb, ubQ, ubK, ubG, ubQb, ubKb, ubT, ubT2, ubGl;
    pipe.InitBuffer(ubUb, E * 2);
    pipe.InitBuffer(ubU, E * 4);
    pipe.InitBuffer(ubD1, E * 4);
    pipe.InitBuffer(ubV, E * 4);
    pipe.InitBuffer(ubVb, E * 2);
    pipe.InitBuffer(ubQ, E * 4);
    pipe.InitBuffer(ubK, E * 4);
    pipe.InitBuffer(ubG, E * 4);
    pipe.InitBuffer(ubQb, E * 2);
    pipe.InitBuffer(ubKb, E * 2);
    pipe.InitBuffer(ubT, E * 4);
    pipe.InitBuffer(ubT2, E * 4);
    pipe.InitBuffer(ubGl, 128 * 4);
    TBuf<TPosition::VECIN> ubTg, ubTg2;
    pipe.InitBuffer(ubTg, 16 * 16 * 2);
    pipe.InitBuffer(ubTg2, 16 * 16 * 2);
    LocalTensor<bfloat16_t> tUb = ubUb.Get<bfloat16_t>();
    LocalTensor<float> tU = ubU.Get<float>(), tD1 = ubD1.Get<float>(), tV = ubV.Get<float>(),
        tQ = ubQ.Get<float>(), tK = ubK.Get<float>(), tG = ubG.Get<float>(),
        tT = ubT.Get<float>(), tT2 = ubT2.Get<float>(), tGl = ubGl.Get<float>();
    LocalTensor<bfloat16_t> tVb = ubVb.Get<bfloat16_t>(), tQb = ubQb.Get<bfloat16_t>(), tKb = ubKb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> tTg = ubTg.Get<bfloat16_t>(), tTg2 = ubTg2.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gU, gQ, gK, gVnew, gVnewT, gQqg, gKg;
    GlobalTensor<float> gD1, gG, gGl;
    gU.SetGlobalBuffer((__gm__ bfloat16_t *)pu);
    gQ.SetGlobalBuffer((__gm__ bfloat16_t *)pq);
    gK.SetGlobalBuffer((__gm__ bfloat16_t *)pk);
    gVnew.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gQqg.SetGlobalBuffer((__gm__ bfloat16_t *)pqg);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gD1.SetGlobalBuffer((__gm__ float *)pd1);
    gG.SetGlobalBuffer((__gm__ float *)pg);
    gGl.SetGlobalBuffer((__gm__ float *)pglast);

    DataCopy(tUb, gU, E);
    DataCopy(tD1, gD1, E);
    DataCopy(tQb, gQ, E);
    DataCopy(tKb, gK, E);
    DataCopy(tG, gG, E);
    DataCopy(tGl, gGl, 128);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);
    Cast(tU, tUb, RoundMode::CAST_NONE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tQ, tQb, RoundMode::CAST_NONE, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tK, tKb, RoundMode::CAST_NONE, E);
    AscendC::PipeBarrier<PIPE_V>();
    // v_new = u - d1
    Sub(tV, tU, tD1, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tVb, tV, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gVnew, tVb, E);
    // vnewT = v_new^T via 8x block transpose (each 16x16 block)
    // v_new[16,128]: block b = cols 16b..16b+15 (each row 128 apart in memory,
    // so gather into contiguous 16x16 first). Use tUb as scratch [16,16].
    for (int b = 0; b < 8; b++) {
        // gather column-block b of tVb into tUb[16,16]: tUb[r][c] = tVb[r][16b+c]
        // tVb is [16,128]; element (r, 16b+c) at r*128 + 16b + c. Not contiguous.
        // Use per-row DataCopy of 16 elems.
        for (int r = 0; r < 16; r++) {
            DataCopy(tTg[r * 16], tVb[r * 128 + b * 16], DataCopyParams(1, 1, 0, 0));
        }
        SetFlag<HardEvent::MTE2_V>(ev2v);
        WaitFlag<HardEvent::MTE2_V>(ev2v);
        AscendC::Transpose(tTg2, tTg);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evv3);
        WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(gVnewT[b * 256], tTg2, DataCopyParams(1, 16, 0, 0));
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(9);
    // qg = q * exp2(g)  (non-inplace exp: tU scratch; output to tQb)
    Muls(tU, tG, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tU, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tU, tQ, tT, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tQb, tU, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gQqg, tQb, E);
    CrossCoreSetFlag<2, PIPE_MTE3>(9);
    CrossCoreSetFlag<2, PIPE_MTE3>(9);
    // qg = q * exp2(g)  (non-inplace exp: tU scratch; output to tQb)
    Muls(tU, tG, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tU, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tU, tQ, tT, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tQb, tU, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gQqg, tQb, E);
    // kg = k * exp2(g_last - g); output to tKb
    for (int r = 0; r < 16; r++) {
        Sub(tT2[r * 128], tGl, tG[r * 128], 128);
        AscendC::PipeBarrier<PIPE_V>();
    }
    Muls(tD1, tT2, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT2, tD1, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tD1, tK, tT2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tKb, tD1, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gKg, tKb, E);
    // kg = k * exp2(g_last - g); output to tKb
    for (int r = 0; r < 16; r++) {
        Sub(tT2[r * 128], tGl, tG[r * 128], 128);
        AscendC::PipeBarrier<PIPE_V>();
    }
    Muls(tD1, tT2, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT2, tD1, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tD1, tK, tT2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tKb, tD1, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gKg, tKb, E);
    CrossCoreSetFlag<2, PIPE_MTE3>(9);
#endif
}
