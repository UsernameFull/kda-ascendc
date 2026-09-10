#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t N = 64;
constexpr int32_t TILE = M * BV;
constexpr uint16_t SYNC_D12_VNEW = 8;

static __aicore__ inline void run_d12_aic(
    GM_ADDR pW, GM_ADDR pQg, GM_ADDR pS16,
    GM_ADDR pD1, GM_ADDR pD2,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    const int32_t bh = GetBlockIdx();
    if (bh >= BH) {
        return;
    }

    TPipe pipe;
    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(l1BQue, 1, BV * D * sizeof(bfloat16_t));
    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, M * BV * sizeof(float));

    LocalTensor<float> l0c = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aBytes(TPosition::A2, 0, M * D * sizeof(bfloat16_t));
    LocalTensor<uint8_t> l0bBytes(TPosition::B2, 0, BV * D * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> l0a = l0aBytes.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bBytes.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> W, Qg, S16;
    GlobalTensor<float> D1, D2;
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
    D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));

    for (int32_t iv = 0; iv < NV; ++iv) {
        const int32_t task = bh * NV + iv;
        const uint64_t a0 = static_cast<uint64_t>(bh * NT + chunk) * M * D;
        const uint64_t s0 = static_cast<uint64_t>(task) * BV * D;
        const uint64_t o0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();

        DataCopy(la, W[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        DataCopy(lb, S16[s0], Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la);
        l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>();
        lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 32, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0c, l0a, l0b, MmadParams(M, N, D, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int32_t nb = 0; nb < 4; ++nb) {
            auto ip = FixpipeParamsV220(M, N / 4, 1, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0 + nb * M], l0c[nb * M * (N / 4)], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la);
        l1BQue.FreeTensor(lb);
        // The next Cube reuses L0B/L0C for Qg. Ensure the d1 fixpipe has
        // released those local resources before the second matrix load.
        PipeBarrier<PIPE_ALL>();

        la = l1AQue.AllocTensor<bfloat16_t>();
        lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, Qg[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la);
        la = l1AQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0c, l0a, l0b, MmadParams(M, N, D, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int32_t nb = 0; nb < 4; ++nb) {
            auto ip = FixpipeParamsV220(M, N / 4, 1, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0 + nb * M], l0c[nb * M * (N / 4)], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la);
        l1BQue.FreeTensor(lb);
        PipeBarrier<PIPE_ALL>();
    }
    l0CQue.FreeTensor(l0c);
}

static __aicore__ inline void run_vnew_aiv(
    GM_ADDR pU, GM_ADDR pD1, GM_ADDR pVnew, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    const int32_t raw = GetBlockIdx();
    const int32_t ratio = static_cast<int32_t>(GetTaskRation());
    const int32_t bh = ratio == 0 ? raw : raw / ratio;
    const int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
    if (bh >= BH || iv >= NV) {
        return;
    }

    const int32_t task = bh * NV + iv;
    const int32_t c = bh * NT + chunk;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> uu, ud, uv, ut, uf;
    pipe.InitBuffer(uu, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(ud, TILE * sizeof(float));
    pipe.InitBuffer(uv, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(ut, TILE * sizeof(bfloat16_t));
    // Keep the full U row tile in FP32. The old standalone kernel allocated
    // only TILE floats but cast M*D values into it before slicing the tile.
    pipe.InitBuffer(uf, M * D * sizeof(float));

    LocalTensor<bfloat16_t> ub = uu.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vb = uv.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vt = ut.Get<bfloat16_t>();
    LocalTensor<float> d = ud.Get<float>();
    LocalTensor<float> vf = uf.Get<float>();
    GlobalTensor<bfloat16_t> U, V, Vt;
    GlobalTensor<float> D1;
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
    V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnew));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));

    const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
    const uint64_t out0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
    DataCopy(ub, U[u0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d, D1[out0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(vf, ub, RoundMode::CAST_NONE, M * D);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t v = 0; v < BV; ++v) {
            vf.SetValue(i * BV + v,
                        vf.GetValue(i * D + iv * BV + v) - d.GetValue(i * BV + v));
        }
    }
    Cast(vb, vf, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();
    for (int32_t v = 0; v < BV; ++v) {
        for (int32_t i = 0; i < M; ++i) {
            vt.SetValue(v * M + i, vb.GetValue(i * BV + v));
        }
    }
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(V[out0], vb, DataCopyParams(M, 4, 0, 0));
    DataCopy(Vt[out0], vt, DataCopyParams(BV, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void kda_k2_mix_d12_vnew(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pS16,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pVnew, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        run_d12_aic(pW, pQg, pS16, pD1, pD2, BH, NT, NV, chunk);
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_D12_VNEW);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(SYNC_D12_VNEW);
        run_vnew_aiv(pU, pD1, pVnew, pVnewT, BH, NT, NV, chunk);
    }
}
