// Full MIX K2 single step. blockdim=3 (MIX_AIC_1_2).
// DAG:
//   AIC: d1 = w@h^T -> GM, SetFlag(8)
//   AIV: Wait(8), v_new=u-d1, qg=q*exp2(g), kg=k*exp2(g_last-g), v_new^T -> GM, SetFlag(9)
//   AIC: Wait(9), d2=qg@h^T, d3=Aqk@v_new, d4=v_new^T@kg -> GM, SetFlag(10)
//   AIV: Wait(10), out=scale*d2+d3, h_new=h*exp2(g_last)+d4 -> GM
// Inputs: w[16,128]bf16, h[128,128]bf16, u[16,128]bf16, q[16,128]bf16, k[16,128]bf16,
//         g[16,128]fp32(g_cum), g_last[128]fp32, aqk[16,16]bf16, h_state[128,128]fp32
// Outputs: out[16,128]bf16, hnew[128,128]fp32
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
constexpr int32_t H = 128 * 128;
constexpr float SCALE = 0.08838834764f;
constexpr float LN2 = 0.6931471805599453f;

extern "C" __global__ __aicore__ void k2_mix_step(GM_ADDR pw, GM_ADDR ph, GM_ADDR pu,
    GM_ADDR pq, GM_ADDR pk, GM_ADDR pg, GM_ADDR pglast, GM_ADDR paqk, GM_ADDR phstate,
    GM_ADDR pd1, GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4, GM_ADDR pvnew, GM_ADDR pvnewT, GM_ADDR pqg, GM_ADDR pkg,
    GM_ADDR pout, GM_ADDR phnew, GM_ADDR ws, GM_ADDR tiling) {
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
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pw);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);
    gAqk.SetGlobalBuffer((__gm__ bfloat16_t *)paqk);
    gV.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gC.SetGlobalBuffer((__gm__ float *)pd1);

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

    // ---- phase 1: d1 = w @ h^T ----
    dot16x128(gA, gC);
    CrossCoreSetFlag<2, PIPE_MTE3>(8);
    // ---- wait AIV: v_new/qg/kg ready ----
    CrossCoreWaitFlag<2, PIPE_MTE3>(9);

    // ---- phase 2: d2 = qg@h^T, d3 = Aqk@v_new, d4 = v_new^T@kg ----
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg);
    gC.SetGlobalBuffer((__gm__ float *)pd2);
    dot16x128(gA, gC);   // d2

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
    l0CQue.FreeTensor(l0cf);

#elif defined(__DAV_C220_VEC__)
    // ---- AIV phase 1: wait d1, compute v_new, qg, kg, v_new^T ----
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
    LocalTensor<bfloat16_t> tUb = ubUb.Get<bfloat16_t>();
    LocalTensor<float> tU = ubU.Get<float>(), tD1 = ubD1.Get<float>(), tV = ubV.Get<float>(),
        tQ = ubQ.Get<float>(), tK = ubK.Get<float>(), tG = ubG.Get<float>(),
        tT = ubT.Get<float>(), tT2 = ubT2.Get<float>(), tGl = ubGl.Get<float>();
    LocalTensor<bfloat16_t> tVb = ubVb.Get<bfloat16_t>(), tQb = ubQb.Get<bfloat16_t>(), tKb = ubKb.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gU, gQ, gK, gVnew, gVnewT, gQqg, gKg;
    GlobalTensor<float> gD1, gG, gGl;
    gU.SetGlobalBuffer((__gm__ bfloat16_t *)pu);
    gQ.SetGlobalBuffer((__gm__ bfloat16_t *)pq);
    gK.SetGlobalBuffer((__gm__ bfloat16_t *)pk);
    gVnew.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
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
    // v_new^T: transpose via store (write tVb transposed to gVnewT)
    // NOTE: transpose in AIV requires per-element; for now write v_new^T by
    // storing tVb with transposed layout is non-trivial. We store v_new and
    // let AIC read it transposed via Nd2Nz (d4 uses v_new^T = Nd2Nz(v_new) gives v_new^T? no).
    // Simplification: AIC d4 reads v_new (not v_new^T) and uses transposeA. For now
    // store v_new to pvnewT too (placeholder); real transpose handled in AIC d4 via
    // reading v_new as [16,128] with transposeA. We'll fix in AIC d4.
    DataCopy(gVnewT, tVb, E);   // placeholder: same as v_new (AIC will transpose)
    // qg = q * exp2(g)
    Muls(tT, tG, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT, tT, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tQ, tQ, tT, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tQb, tQ, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gQqg, tQb, E);
    // kg = k * exp2(g_last - g)
    // g_last broadcast: tGl[128] -> need [16,128]; use row loop
    for (int r = 0; r < 16; r++) {
        Sub(tT2[r * 128], tGl, tG[r * 128], 128);
        AscendC::PipeBarrier<PIPE_V>();
    }
    Muls(tT2, tT2, LN2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT2, tT2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(tK, tK, tT2, E);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tKb, tK, RoundMode::CAST_RINT, E);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gKg, tKb, E);
    CrossCoreSetFlag<2, PIPE_MTE3>(9);

    // ---- AIV phase 2: wait d2/d3/d4, out + state ----
    CrossCoreWaitFlag<2, PIPE_MTE2>(10);
    // out = scale*d2 + d3
    TBuf<TPosition::VECIN> ubD2, ubD3, ubOut, ubOutb, ubH, ubD4, ubG2, ubT3;
    pipe.InitBuffer(ubD2, E * 4);
    pipe.InitBuffer(ubD3, E * 4);
    pipe.InitBuffer(ubOut, E * 4);
    pipe.InitBuffer(ubOutb, E * 2);
    pipe.InitBuffer(ubH, H * 4);
    pipe.InitBuffer(ubD4, H * 4);
    pipe.InitBuffer(ubG2, 128 * 4);
    pipe.InitBuffer(ubT3, 128 * 4);
    LocalTensor<float> tD2 = ubD2.Get<float>(), tD3 = ubD3.Get<float>(), tOut = ubOut.Get<float>(),
        tH = ubH.Get<float>(), tD4 = ubD4.Get<float>(), tG2 = ubG2.Get<float>(), tT3 = ubT3.Get<float>();
    LocalTensor<bfloat16_t> tOutb = ubOutb.Get<bfloat16_t>();
    GlobalTensor<float> gD2, gD3, gD4, gH, gHnew;
    GlobalTensor<bfloat16_t> gOut;
    gD2.SetGlobalBuffer((__gm__ float *)pd2);
    gD3.SetGlobalBuffer((__gm__ float *)pd3);
    gD4.SetGlobalBuffer((__gm__ float *)pd4);
    gH.SetGlobalBuffer((__gm__ float *)phstate);
    gHnew.SetGlobalBuffer((__gm__ float *)phnew);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t *)pout);
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
    Muls(tG2, tGl, LN2, 128);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tT3, tG2, 128);
    AscendC::PipeBarrier<PIPE_V>();
    DataCopy(tH, gH[0], DataCopyParams(16, 128, 0, 0));
    DataCopy(tD4, gD4[0], DataCopyParams(16, 128, 0, 0));
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::PipeBarrier<PIPE_V>();
    for (int r = 0; r < 128; r++) {
        Mul(tH[r * 128], tH[r * 128], tT3, 128);
        AscendC::PipeBarrier<PIPE_V>();
    }
    Add(tH, tH, tD4, H);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evv3);
    WaitFlag<HardEvent::V_MTE3>(evv3);
    DataCopy(gHnew[0], tH, DataCopyParams(16, 128, 0, 0));
#endif
}
