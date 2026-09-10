// MIX2: AIC d2=qg@h^T, d3=Aqk@v_new, d4=v_new^T@kg -> flag10 -> AIV out/state.
// AIV buffers independent (no phase-1), no aliasing.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
constexpr int32_t H = 128 * 128;
constexpr float SCALE = 0.08838834764f;
constexpr float LN2 = 0.6931471805599453f;

extern "C" __global__ __aicore__ void k2_mix2(GM_ADDR ph, GM_ADDR pqg, GM_ADDR paqk, GM_ADDR pvnew,
    GM_ADDR pvnewT, GM_ADDR pkg, GM_ADDR pglast, GM_ADDR phstate,
    GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4, GM_ADDR pout, GM_ADDR phnew, GM_ADDR ws, GM_ADDR tiling) {
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
    pipe.InitBuffer(l0CQue, 1, 128 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 8 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 64 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gA, gB, gAqk, gV, gKg;
    GlobalTensor<float> gC;
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);
    gAqk.SetGlobalBuffer((__gm__ bfloat16_t *)paqk);
    gV.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gC.SetGlobalBuffer((__gm__ float *)pd2);

    auto dot16x128 = [&](GlobalTensor<bfloat16_t> &gAin, GlobalTensor<float> &gCout) __aicore__ {
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gAin, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
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
            Fixpipe<float, float, CFG_ROW_MAJOR>(gCout[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
    };

    // d2 = qg @ h^T
    dot16x128(gA, gC);
    // d3 = Aqk[16,16] @ v_new[16,128]
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)paqk);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gC.SetGlobalBuffer((__gm__ float *)pd3);
    {
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 8, 1, 0, 0, true, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
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
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    // d4 = v_new^T @ kg -> [128,128] (m-tiled 8x)
    gV.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gC.SetGlobalBuffer((__gm__ float *)pd4);
    {
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(lb, gKg, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1BQue.EnQue(lb);
        LoadData(l0b, l1BQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 8, 1, 0, 0, true, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        for (int i = 0; i < 8; i++) {
            LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
            DataCopy(la, gV[i * 256], Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
            SetFlag<HardEvent::MTE2_MTE1>(ev21);
            WaitFlag<HardEvent::MTE2_MTE1>(ev21);
            l1AQue.EnQue(la);
            LoadData(l0a, l1AQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            SetFlag<HardEvent::MTE1_M>(ev1m);
            WaitFlag<HardEvent::MTE1_M>(ev1m);
            Mmad(l0cf[i * 2048], l0a, l0b, MmadParams(16, 128, 16, 0, false, true));
            SetFlag<HardEvent::M_FIX>(evmfix);
            WaitFlag<HardEvent::M_FIX>(evmfix);
            l1AQue.FreeTensor(la);
        }
        AscendC::PipeBarrier<PIPE_M>();
        for (int mb = 0; mb < 8; mb++)
            for (int nb = 0; nb < 8; nb++) {
                auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
                ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
                Fixpipe<float, float, CFG_ROW_MAJOR>(gC[mb * 2048 + nb * 16], l0cf[(mb * 8 + nb) * 256], ip);
            }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(10);
    CrossCoreWaitFlag<2, PIPE_MTE3>(11);
    l0CQue.FreeTensor(l0cf);
#elif defined(__DAV_C220_VEC__)
    CrossCoreWaitFlag<2, PIPE_MTE2>(10);
    TPipe pipe;
    TBuf<TPosition::VECIN> ubD2, ubD3, ubOut, ubOutb, ubH, ubD4, ubG, ubG2, ubT, ubT2;
    pipe.InitBuffer(ubD2, E * 4);
    pipe.InitBuffer(ubD3, E * 4);
    pipe.InitBuffer(ubOut, E * 4);
    pipe.InitBuffer(ubOutb, E * 2);
    pipe.InitBuffer(ubH, H * 4);
    pipe.InitBuffer(ubD4, H * 4);
    pipe.InitBuffer(ubG, 128 * 4);
    pipe.InitBuffer(ubG2, 128 * 4);
    pipe.InitBuffer(ubT, 128 * 4);
    pipe.InitBuffer(ubT2, 128 * 4);
    LocalTensor<float> tD2 = ubD2.Get<float>(), tD3 = ubD3.Get<float>(), tOut = ubOut.Get<float>(),
        tH = ubH.Get<float>(), tD4 = ubD4.Get<float>(), tG = ubG.Get<float>(), tG2 = ubG2.Get<float>(),
        tT = ubT.Get<float>(), tT2 = ubT2.Get<float>();
    LocalTensor<bfloat16_t> tOutb = ubOutb.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gOut, gHbf;
    GlobalTensor<float> gG, gH, gD2, gD3, gD4, gHnew;
    gOut.SetGlobalBuffer((__gm__ bfloat16_t *)pout);
    gHbf.SetGlobalBuffer((__gm__ bfloat16_t *)ph);  // state feedback: write h_new as bf16 back to ph
    gG.SetGlobalBuffer((__gm__ float *)pglast);
    gH.SetGlobalBuffer((__gm__ float *)phstate);
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
    // h_new = h*exp2(g_last) + d4
    DataCopy(tG, gG, 128);
    SetFlag<HardEvent::MTE2_V>(ev2v);
    WaitFlag<HardEvent::MTE2_V>(ev2v);
    Muls(tG2, tG, LN2, 128);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tG2, 128);
    AscendC::PipeBarrier<PIPE_V>();
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
    // State feedback: write h_new as bf16 back to ph for next chunk's d1/d2
    Cast(tOutb, tH, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    // Write h_new as bf16 to ph, in 8 blocks of 16x128 (reuse small outb buffer)
    for (int hb = 0; hb < 8; hb++) {
        Cast(tOutb, tH[hb * 2048], RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evv3);
        WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(gHbf[hb * 2048], tOutb, DataCopyParams(2, 64, 0, 0));
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(11);
#endif
}
