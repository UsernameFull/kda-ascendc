#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t N = 64;
constexpr int32_t TILE = M * BV;
constexpr uint16_t SYNC_D12_VNEW = 8;
constexpr int32_t K = 16;
constexpr int32_t N_D4 = 128;


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
        {
            auto ip = FixpipeParamsV220(N, M, 16, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0], l0c, ip);
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
        {
            auto ip = FixpipeParamsV220(N, M, 16, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0], l0c, ip);
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
    TEventID evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
    TBuf<TPosition::VECCALC> uu, ud, uv, ut, uf, usc;
    pipe.InitBuffer(uu, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(ud, TILE * sizeof(float));
    pipe.InitBuffer(uv, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(ut, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(usc, (BV / M) * M * M * sizeof(bfloat16_t));
    // Keep the full U row tile in FP32. The old standalone kernel allocated
    // only TILE floats but cast M*D values into it before slicing the tile.
    pipe.InitBuffer(uf, M * D * sizeof(float));

    LocalTensor<bfloat16_t> ub = uu.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vb = uv.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vt = ut.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> sc = usc.Get<bfloat16_t>();
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
        Sub(vf[i * BV], vf[i * D + iv * BV], d[i * BV], BV);
    }
    Cast(vb, vf, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();
    // v_new^T via the UB 16x16 transpose unit: gather the four column blocks
    // of v_new into contiguous 16x16 tiles, transpose, then store 64 rows.
    SetFlag<HardEvent::V_MTE2>(evm2);
    WaitFlag<HardEvent::V_MTE2>(evm2);
    for (int32_t bl = 0; bl < BV / M; ++bl) {
        for (int32_t r = 0; r < M; ++r) {
            DataCopy(sc[bl * M * M + r * M], vb[r * BV + bl * M], DataCopyParams(1, 1, 0, 0));
        }
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    for (int32_t bl = 0; bl < BV / M; ++bl) {
        AscendC::Transpose(vt[bl * M * M], sc[bl * M * M]);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(V[out0], vb, DataCopyParams(M, 4, 0, 0));
    DataCopy(Vt[out0], vt, DataCopyParams(BV, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}

// One AIC block stages Aqk, both v_new^T tiles and kg^T once, then issues all
// ten Mmads (eight 16-row d4 groups plus the two d3 tiles) back to back and
// drains the M/FIX pipes once.  Like the rest of this kernel it assumes
// NV == 2 (V == 128 with BV == 64), so the whole d4 tile is 8 16-row groups.
static __aicore__ inline void run_d34_aic(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pKgT, GM_ADDR pD3, GM_ADDR pD4,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk, int32_t d4_reuse) {
    int32_t bh = GetBlockIdx();
    if (bh >= BH) {
        return;
    }
    int32_t c = bh * NT + chunk;
    constexpr int32_t N3 = 64;
    constexpr int32_t NG = 2 * BV / M;          // 16-row groups in the whole d4 tile
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> qa, qv, qk;
    pipe.InitBuffer(qa, 1, M * K * 2);
    pipe.InitBuffer(qv, 1, NV * BV * K * 2);
    pipe.InitBuffer(qk, 1, D * K * 2);
    TQue<QuePosition::CO1, 1> qc;
    pipe.InitBuffer(qc, 1, NG * M * N_D4 * 4 + NV * M * N3 * 4);
    LocalTensor<float> cf = qc.AllocTensor<float>();
    LocalTensor<uint8_t> a8(TPosition::A2, 0, (NG + 1) * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, (D + NV * BV) * K * 2);
    LocalTensor<bfloat16_t> a = a8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> Aqk, Vt, Kt;
    GlobalTensor<float> D3, D4;
    Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

    // 16-column operands: the ND layout already is the NZ layout the Cube
    // wants, so plain burst copies into L1 replace the Nd2Nz descriptors.
    auto la = qa.AllocTensor<bfloat16_t>();
    auto lv = qv.AllocTensor<bfloat16_t>();
    auto lk = qk.AllocTensor<bfloat16_t>();
    DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], M * K);
    for (int32_t iv = 0; iv < NV; ++iv) {
        int32_t task = bh * NV + iv;
        DataCopy(lv[iv * BV * K], Vt[(static_cast<uint64_t>(task) * NT + chunk) * BV * K], BV * K);
    }
    DataCopy(lk, Kt[static_cast<uint64_t>(c) * D * K], D * K);
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);
    qa.EnQue(la);
    qv.EnQue(lv);
    qk.EnQue(lk);
    la = qa.DeQue<bfloat16_t>();
    lv = qv.DeQue<bfloat16_t>();
    lk = qk.DeQue<bfloat16_t>();

    LoadData(a, lv, LoadData2dParams(0, NG, 1, 0, 0, false, 0));
    LoadData(a[NG * M * K], la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadData(b, lk, LoadData2dParams(0, D / M, 1, 0, 0, false, 0));
    for (int32_t iv = 0; iv < NV; ++iv) {
        LoadData(b[D * K + iv * BV * K], lv[iv * BV * K], LoadData2dParams(0, BV / M, 1, 0, 0, false, 0));
    }
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);

    // d4 = v_new^T @ kg, one 16 x 128 C tile per 16-row group.
    for (int32_t g = 0; g < NG; ++g) {
        Mmad(cf[g * M * N_D4], a[g * M * K], b, MmadParams(M, N_D4, K, 0, false, true));
    }
    // d3[i][v] = Aqk[i][j] v_new[j][v] for each value tile.
    for (int32_t iv = 0; iv < NV; ++iv) {
        Mmad(cf[NG * M * N_D4 + iv * M * N3], a[NG * M * K], b[D * K + iv * BV * K],
             MmadParams(M, N3, K, 0, false, true));
    }
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);

    uint64_t d4c = d4_reuse != 0 ? static_cast<uint64_t>(bh) : static_cast<uint64_t>(c);
    for (int32_t g = 0; g < NG; ++g) {
        auto ip = FixpipeParamsV220(N_D4, M, 16, N_D4, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(D4[d4c * D * D + static_cast<uint64_t>(g) * M * D],
                                             cf[g * M * N_D4], ip);
    }
    for (int32_t iv = 0; iv < NV; ++iv) {
        auto ip = FixpipeParamsV220(N3, M, 16, N3, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(
            D3[(static_cast<uint64_t>(bh * NV + iv) * NT + chunk) * M * BV],
            cf[NG * M * N_D4 + iv * M * N3], ip);
    }
    SetFlag<HardEvent::FIX_M>(efm);
    WaitFlag<HardEvent::FIX_M>(efm);
    qa.FreeTensor(la);
    qv.FreeTensor(lv);
    qk.FreeTensor(lk);
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

    Muls(of, d2, scale, TILE);
    Add(of, of, d3, TILE);
    Cast(ob, of, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();

    // s[v][k] = s[v][k]*dec[k] + d4[v][k] in two strided Mul repeats (the
    // decay vector is reused with a zero source-repeat stride).
    Mul(s, s, dec, 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Mul(s[64], s[64], dec[64], 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Add(s, s, d4, BV * D);
    Cast(s16, s, RoundMode::CAST_RINT, BV * D);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Out[t0], ob, DataCopyParams(M, 4, 0, 0));
    DataCopy(S32[s0], s, DataCopyParams(BV, 16, 0, 0));
    DataCopy(S16[s0], s16, DataCopyParams(BV, 8, 0, 0));
    PipeBarrier<PIPE_ALL>();
}

constexpr uint16_t SYNC_VNEW_READY = 10;
constexpr uint16_t SYNC_D34_READY = 11;

extern "C" __global__ __aicore__ void kda_k2_mix_all_cube(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pS16,
    GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4,
    GM_ADDR pS32, GM_ADDR pOut, GM_ADDR pVnew, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk,
    int32_t d4_reuse, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        run_d12_aic(pW, pQg, pS16, pD1, pD2, BH, NT, NV, chunk);
        CrossCoreSetFlag<2, PIPE_FIX>(8);
        CrossCoreWaitFlag(9);
        run_d34_aic(pAqk, pVnewT, pKgT, pD3, pD4, BH, NT, NV, chunk, d4_reuse);
        CrossCoreSetFlag<2, PIPE_FIX>(10);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(8);
        run_vnew_aiv(pU, pD1, pVnew, pVnewT, BH, NT, NV, chunk);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
        CrossCoreWaitFlag(10);
        run_outstate_aiv(pD2, pD3, pD4, pS32, pS16, pDecay, pOut, BH, NT, NV, chunk, d4_reuse, scale);
    }
}

