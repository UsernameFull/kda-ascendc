#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t K = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t N = 128;
constexpr int32_t TILE = M * BV;

static __aicore__ inline void run_d3_aic(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pD3,
    int32_t BH, int32_t NT, int32_t chunk) {
    int32_t bh = GetBlockIdx();
    if (bh >= BH) {
        return;
    }
    constexpr int32_t N3 = 64;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> qa, qb;
    pipe.InitBuffer(qa, 1, M * K * 2);
    pipe.InitBuffer(qb, 1, BV * K * 2);
    TQue<QuePosition::CO1, 1> qc;
    pipe.InitBuffer(qc, 1, M * D * 4);
    LocalTensor<float> cf = qc.AllocTensor<float>();
    LocalTensor<uint8_t> a8(TPosition::A2, 0, M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, BV * K * 2);
    LocalTensor<bfloat16_t> a = a8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> Aqk, Vt;
    GlobalTensor<float> D3;
    Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
    int32_t c = bh * NT + chunk;
    for (int iv = 0; iv < 2; ++iv) {
        int32_t task = bh * 2 + iv;
        auto la = qa.AllocTensor<bfloat16_t>();
        auto lb = qb.AllocTensor<bfloat16_t>();
        DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K],
                 Nd2NzParams(1, M, K, 0, K, M, 1, 0));
        DataCopy(lb, Vt[(static_cast<uint64_t>(task) * NT + chunk) * BV * K],
                 Nd2NzParams(1, BV, K, 0, K, BV, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        qa.EnQue(la);
        qb.EnQue(lb);
        la = qa.DeQue<bfloat16_t>();
        lb = qb.DeQue<bfloat16_t>();
        LoadData(a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
        LoadData(b, lb, LoadData2dParams(0, 4, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(e1m);
        WaitFlag<HardEvent::MTE1_M>(e1m);
        Mmad(cf, a, b, MmadParams(M, N3, K, 0, false, true));
        SetFlag<HardEvent::M_FIX>(emf);
        WaitFlag<HardEvent::M_FIX>(emf);
        for (int nb = 0; nb < 4; ++nb) {
            auto ip = FixpipeParamsV220(M, M, 1, N3, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(
                D3[(static_cast<uint64_t>(task) * NT + chunk) * M * BV + nb * M],
                cf[nb * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(efm);
        WaitFlag<HardEvent::FIX_M>(efm);
        qa.FreeTensor(la);
        qb.FreeTensor(lb);
        PipeBarrier<PIPE_ALL>();
    }
    qc.FreeTensor(cf);
}

static __aicore__ inline void run_d4_aic(
    GM_ADDR pVnewT, GM_ADDR pKgT, GM_ADDR pD4,
    int32_t BH, int32_t NT, int32_t chunk, int32_t d4_reuse) {
    int32_t bh = GetBlockIdx();
    if (bh >= BH) {
        return;
    }
    int32_t c = bh * NT + chunk;

    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();

    TQue<QuePosition::B1, 1> qa, qb;
    pipe.InitBuffer(qa, 1, M * K * 2);
    pipe.InitBuffer(qb, 1, D * K * 2);
    TQue<QuePosition::CO1, 1> qc;
    pipe.InitBuffer(qc, 1, M * N * 4);

    LocalTensor<float> cf = qc.AllocTensor<float>();
    LocalTensor<uint8_t> a8(TPosition::A2, 0, M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, D * K * 2);
    LocalTensor<bfloat16_t> a = a8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8.ReinterpretCast<bfloat16_t>();

    GlobalTensor<bfloat16_t> Vt;
    GlobalTensor<bfloat16_t> Kt;
    GlobalTensor<float> D4;
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

    for (int mb = 0; mb < 8; ++mb) {
        int iv = mb / 4;
        int rr = mb % 4;
        uint64_t ao = (static_cast<uint64_t>(bh * 2 + iv) * NT + chunk) *
                          64 * M + static_cast<uint64_t>(rr) * M * K;
        auto la = qa.AllocTensor<bfloat16_t>();
        auto lb = qb.AllocTensor<bfloat16_t>();
        DataCopy(la, Vt[ao], Nd2NzParams(1, M, K, 0, K, M, 1, 0));
        DataCopy(lb, Kt[static_cast<uint64_t>(c) * D * K],
                 Nd2NzParams(1, D, K, 0, K, D, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        qa.EnQue(la);
        qb.EnQue(lb);
        la = qa.DeQue<bfloat16_t>();
        lb = qb.DeQue<bfloat16_t>();
        LoadData(a, la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
        LoadData(b, lb, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(e1m);
        WaitFlag<HardEvent::MTE1_M>(e1m);
        Mmad(cf, a, b, MmadParams(M, N, K, 0, false, true));
        SetFlag<HardEvent::M_FIX>(emf);
        WaitFlag<HardEvent::M_FIX>(emf);
        uint64_t d4c = d4_reuse != 0 ? static_cast<uint64_t>(bh) : static_cast<uint64_t>(c);
        for (int nb = 0; nb < 8; ++nb) {
            auto ip = FixpipeParamsV220(M, M, 1, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(
                D4[d4c * D * D + mb * M * D + nb * M],
                cf[nb * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(efm);
        WaitFlag<HardEvent::FIX_M>(efm);
        qa.FreeTensor(la);
        qb.FreeTensor(lb);
        PipeBarrier<PIPE_ALL>();
    }
    qc.FreeTensor(cf);
}


static __aicore__ inline void run_outstate_aiv(
    GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4, GM_ADDR pS32, GM_ADDR pS16,
    GM_ADDR pDecay, GM_ADDR pOut,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk, int32_t d4_reuse, float scale) {
    int32_t raw_block = GetBlockIdx();
    int32_t ratio = static_cast<int32_t>(GetTaskRation());
    int32_t bh = ratio == 0 ? raw_block : raw_block / ratio;
    int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
    if (bh >= BH || iv >= NV) {
        return;
    }
    int32_t task = bh * NV + iv;
    int32_t c = bh * NT + chunk;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> ud2, ud3, ud4, us, udec, uo, uof, us16;
    pipe.InitBuffer(ud2, TILE * 4);
    pipe.InitBuffer(ud3, TILE * 4);
    pipe.InitBuffer(ud4, BV * D * 4);
    pipe.InitBuffer(us, BV * D * 4);
    pipe.InitBuffer(udec, D * 4);
    pipe.InitBuffer(uo, TILE * 2);
    pipe.InitBuffer(uof, TILE * 4);
    pipe.InitBuffer(us16, BV * D * 2);

    LocalTensor<float> d2 = ud2.Get<float>();
    LocalTensor<float> d3 = ud3.Get<float>();
    LocalTensor<float> d4 = ud4.Get<float>();
    LocalTensor<float> s = us.Get<float>();
    LocalTensor<float> dec = udec.Get<float>();
    LocalTensor<float> of = uof.Get<float>();
    LocalTensor<bfloat16_t> ob = uo.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> s16 = us16.Get<bfloat16_t>();

    GlobalTensor<float> D2, D3, D4, S32, Decay;
    GlobalTensor<bfloat16_t> S16, Out;
    D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd2));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd3));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd4));
    S32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pS32));
    S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
    Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));

    uint64_t t0 = (static_cast<uint64_t>(task) * NT + chunk) * TILE;
    uint64_t s0 = static_cast<uint64_t>(task) * BV * D;
    uint64_t c0 = static_cast<uint64_t>(c) * D;
    uint64_t d4c = d4_reuse != 0 ? static_cast<uint64_t>(bh) : static_cast<uint64_t>(c);
    uint64_t d4base = d4c * D * D + static_cast<uint64_t>(iv) * BV * D;

    DataCopy(d2, D2[t0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d3, D3[t0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d4, D4[d4base], DataCopyParams(BV, 16, 0, 0));
    DataCopy(s, S32[s0], DataCopyParams(BV, 16, 0, 0));
    DataCopy(dec, Decay[c0], DataCopyParams(1, 16, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);

    for (int i = 0; i < M; ++i) {
        for (int v = 0; v < BV; ++v) {
            of.SetValue(i * BV + v,
                        scale * d2.GetValue(i * BV + v) +
                        d3.GetValue(i * BV + v));
        }
    }
    Cast(ob, of, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();

    for (int v = 0; v < BV; ++v) {
        for (int k = 0; k < D; ++k) {
            s.SetValue(v * D + k,
                        s.GetValue(v * D + k) * dec.GetValue(k) +
                        d4.GetValue(v * D + k));
        }
    }
    Cast(s16, s, RoundMode::CAST_RINT, BV * D);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Out[t0], ob, DataCopyParams(M, 4, 0, 0));
    DataCopy(S32[s0], s, DataCopyParams(BV, 16, 0, 0));
    DataCopy(S16[s0], s16, DataCopyParams(BV, 8, 0, 0));
    PipeBarrier<PIPE_ALL>();
}

constexpr uint16_t SYNC_AIC_AIV_FLAG_LOCAL = 9;

extern "C" __global__ __aicore__ void kda_k2_mix_d4_outstate(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pKgT,
    GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4,
    GM_ADDR pS32, GM_ADDR pS16, GM_ADDR pDecay, GM_ADDR pOut,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk, int32_t d4_reuse, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        run_d3_aic(pAqk, pVnewT, pd3, BH, NT, chunk);
        run_d4_aic(pVnewT, pKgT, pd4, BH, NT, chunk, d4_reuse);
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_LOCAL);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_LOCAL);
        run_outstate_aiv(pd2, pd3, pd4, pS32, pS16, pDecay, pOut,
                         BH, NT, NV, chunk, d4_reuse, scale);
    }
}
