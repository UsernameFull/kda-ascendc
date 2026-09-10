#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t N = 64;
constexpr int32_t K = 16;
constexpr int32_t TILE = M * BV;
constexpr uint16_t SYNC_D12_VNEW = 8;
constexpr uint16_t SYNC_VNEW_READY = 10;
constexpr uint16_t SYNC_D34_READY = 12;
constexpr uint16_t SYNC_STATE_READY = 13;

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



static __aicore__ inline void run_state_aiv(
    GM_ADDR pU, GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4,
    GM_ADDR pAqk, GM_ADDR pDecay, GM_ADDR pS16, GM_ADDR pH0, GM_ADDR pOut, GM_ADDR pHt,
    GM_ADDR pVnew, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, float scale) {
    int32_t raw_block = GetBlockIdx();
    int32_t ratio = static_cast<int32_t>(GetTaskRation());
    int32_t bh = ratio == 0 ? raw_block : raw_block / ratio;
    if (bh >= BH || GetSubBlockIdx() != 0) return;
    TPipe pipe;
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> us0,us1,udec,ud1,ud2,ud3,ud4,uv,uu,uub,uvb,uvt,uo,uof,uhf;
    pipe.InitBuffer(us0,BV*D*4); pipe.InitBuffer(us1,BV*D*4); pipe.InitBuffer(udec,D*4); pipe.InitBuffer(ud1,TILE*4); pipe.InitBuffer(ud2,TILE*4);
    pipe.InitBuffer(ud3,TILE*4); pipe.InitBuffer(ud4,BV*D*4); pipe.InitBuffer(uv,TILE*4); pipe.InitBuffer(uu,M*D*4); pipe.InitBuffer(uub,M*D*2);
    pipe.InitBuffer(uvb,TILE*2); pipe.InitBuffer(uvt,BV*M*2); pipe.InitBuffer(uo,TILE*2); pipe.InitBuffer(uof,TILE*4); pipe.InitBuffer(uhf,BV*D*2);
    LocalTensor<float> h0=us0.Get<float>(), h1=us1.Get<float>(), dec=udec.Get<float>(), d1=ud1.Get<float>(), d2=ud2.Get<float>(), d3=ud3.Get<float>(), d4=ud4.Get<float>();
    LocalTensor<float> vf=uv.Get<float>(), uf32=uu.Get<float>(), outf=uof.Get<float>();
    LocalTensor<bfloat16_t> ub=uub.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vb=uvb.Get<bfloat16_t>(), vt=uvt.Get<bfloat16_t>(), outb=uo.Get<bfloat16_t>(), hbf=uhf.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> U,S16,Vnew,VnewT,Out; GlobalTensor<float> D1,D2,D3,D4,Decay,H0,Ht;
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU)); S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    Vnew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnew)); VnewT.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT)); Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
    D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1)); D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2)); D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3)); D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay)); H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0)); Ht.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pHt));
    uint64_t state_base=static_cast<uint64_t>(bh)*NV*BV*D;
    if(pH0==nullptr) { Duplicate(h0,0.0f,BV*D); Duplicate(h1,0.0f,BV*D); }
    else {
        DataCopy(h0,H0[state_base],DataCopyParams(BV,16,0,0));
        DataCopy(h1,H0[state_base+static_cast<uint64_t>(BV)*D],DataCopyParams(BV,16,0,0));
        SetFlag<HardEvent::MTE2_V>(ev2v); WaitFlag<HardEvent::MTE2_V>(ev2v);
    }
    for(int chunk=0; chunk<NT; ++chunk) {
        CrossCoreWaitFlag(SYNC_D12_VNEW);
        uint64_t ubase=(static_cast<uint64_t>(bh)*NT+chunk)*M*D;
        DataCopy(ub,U[ubase],DataCopyParams(M,8,0,0));
        SetFlag<HardEvent::MTE2_V>(ev2v); WaitFlag<HardEvent::MTE2_V>(ev2v);
        Cast(uf32,ub,RoundMode::CAST_NONE,M*D);
        for(int iv=0;iv<NV;iv++) {
            int32_t task=bh*NV+iv;
            uint64_t base=(static_cast<uint64_t>(task)*NT+chunk)*TILE;
            DataCopy(d1,D1[base],DataCopyParams(M,8,0,0));
            SetFlag<HardEvent::MTE2_V>(ev2v); WaitFlag<HardEvent::MTE2_V>(ev2v);
            Cast(vf,uf32[iv*BV],RoundMode::CAST_NONE,TILE);
            Sub(vf,vf,d1,TILE);
            Cast(vb,vf,RoundMode::CAST_RINT,TILE);
            for(int i=0;i<M;i++) for(int v=0;v<BV;v++) vt.SetValue(v*M+i,vb.GetValue(i*BV+v));
            SetFlag<HardEvent::V_MTE3>(evv3); WaitFlag<HardEvent::V_MTE3>(evv3);
            DataCopy(Vnew[base],vb,DataCopyParams(M,4,0,0));
            DataCopy(VnewT[(static_cast<uint64_t>(task)*NT+chunk)*BV*M],vt,DataCopyParams(BV,1,0,0));
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_VNEW_READY);
        CrossCoreWaitFlag(SYNC_D34_READY);
        DataCopy(dec,Decay[(static_cast<uint64_t>(bh)*NT+chunk)*D],DataCopyParams(1,16,0,0));
        SetFlag<HardEvent::MTE2_V>(ev2v); WaitFlag<HardEvent::MTE2_V>(ev2v);
        for(int iv=0;iv<NV;iv++) {
            int32_t task=bh*NV+iv;
            uint64_t base=(static_cast<uint64_t>(task)*NT+chunk)*TILE;
            LocalTensor<float> state = iv == 0 ? h0 : h1;
            DataCopy(d2,D2[base],DataCopyParams(M,8,0,0));
            DataCopy(d3,D3[base],DataCopyParams(M,8,0,0));
            DataCopy(d4,D4[static_cast<uint64_t>(bh)*D*D+static_cast<uint64_t>(iv)*BV*D],DataCopyParams(BV,16,0,0));
            SetFlag<HardEvent::MTE2_V>(ev2v); WaitFlag<HardEvent::MTE2_V>(ev2v);
            Muls(outf,d2,scale,TILE); Add(outf,outf,d3,TILE); Cast(outb,outf,RoundMode::CAST_RINT,TILE);
            SetFlag<HardEvent::V_MTE3>(evv3); WaitFlag<HardEvent::V_MTE3>(evv3); DataCopy(Out[base],outb,DataCopyParams(M,4,0,0));
            for(int v=0;v<BV;v++) for(int k=0;k<D;k++) state.SetValue(v*D+k,state.GetValue(v*D+k)*dec.GetValue(k)+d4.GetValue(v*D+k));
            Cast(hbf,state,RoundMode::CAST_RINT,BV*D); SetFlag<HardEvent::V_MTE3>(evv3); WaitFlag<HardEvent::V_MTE3>(evv3);
            DataCopy(S16[static_cast<uint64_t>(task)*BV*D],hbf,DataCopyParams(BV,16,0,0));
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_STATE_READY);
    }
    if(pHt!=nullptr){
        SetFlag<HardEvent::V_MTE3>(evv3); WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(Ht[static_cast<uint64_t>(bh)*NV*BV*D],h0,DataCopyParams(BV,16,0,0));
        DataCopy(Ht[static_cast<uint64_t>(bh)*NV*BV*D+static_cast<uint64_t>(BV)*D],h1,DataCopyParams(BV,16,0,0));
    }
}

extern "C" __global__ __aicore__ void kda_k2_persistent_scan_cube_kernel(
    GM_ADDR pW,GM_ADDR pQg,GM_ADDR pU,GM_ADDR pAqk,GM_ADDR pKgT,GM_ADDR pDecay,
    GM_ADDR pH0,GM_ADDR pS16,GM_ADDR pD1,GM_ADDR pD2,GM_ADDR pD3,GM_ADDR pD4,
    GM_ADDR pOut,GM_ADDR pHt,GM_ADDR pVnew,GM_ADDR pVnewT,
    int32_t BH,int32_t NT,int32_t NV,float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        int32_t bh=GetBlockIdx();
        if(bh < BH) {
            for(int chunk=0; chunk<NT; ++chunk) {
                if(chunk>0) { CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_STATE_READY); }
                run_d12_aic(pW,pQg,pS16,pD1,pD2,BH,NT,NV,chunk);
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_D12_VNEW);
                CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_VNEW_READY);
                run_d3_aic(pAqk,pVnewT,pD3,BH,NT,chunk);
                run_d4_aic(pVnewT,pKgT,pD4,BH,NT,chunk,1);
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_D34_READY);
            }
        }
    }
    if ASCEND_IS_AIV {
        run_state_aiv(pU,pD1,pD2,pD3,pD4,pAqk,pDecay,pS16,pH0,pOut,pHt,pVnew,pVnewT,BH,NT,NV,scale);
    }
}
