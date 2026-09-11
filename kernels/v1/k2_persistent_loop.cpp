// Persistent K2: one MIX_AIC_1_2 launch runs the whole chunk recurrence on the
// device instead of eight host launches per chunk.
//
// Each block owns one or two heads (bh) and every AIV subcore owns one value
// tile (iv), so a block has nh * 2 independent state tiles in flight and the
// AIC is never blocked by its own chain.  Per chunk the two engines run
//     AIC: [wait R; d12(h); set C1] x nh   then   [wait V; d34(h); set C2] x nh
//     AIV: [wait C1; vnew(h); set V] x nh  then   [wait C2; out(h); set R] x nh
// with both subcores executing the same flag sequence (the unit of the
// protocol is a head, because that is the only split both subcores share).
// Four flag ids carry the whole loop, the protocol is depth one, and the AIV
// pre-sets R once per head before the loop so the first chunk needs no special
// case.  See docs/VLLM_ASCEND_KDA_REVIEW_20260911.md for why that matters.
//
// The fp32 state never leaves the AIV: it is loaded from H0 at start-up, kept
// in UB across the whole loop and stored to S32 once at the end.  Only the bf16
// copy of the state (S16) is published to GM for the Cube operands.
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t K = 16;
constexpr int32_t N = 64;
constexpr int32_t N_D4 = 128;
constexpr int32_t TILE = M * BV;
constexpr int32_t S_TILE = BV * D;
constexpr int32_t NG = 2 * BV / M;   // 16-row groups in the whole d4 tile
constexpr int32_t MAXH = 2;          // heads owned by one block
// L0A holds W and Qg side by side (2*M*D bf16) and is then reused by d34's
// NG+1 16-column operands, so the raw allocation is whichever is larger.
constexpr int32_t L0A_ELEMS =
    2 * M * D > (NG + 1) * M * K ? 2 * M * D : (NG + 1) * M * K;
// Depth-one loop protocol, ids stay inside the usable range (<= 7).
constexpr uint16_t FL_C1 = 0;
constexpr uint16_t FL_V = 1;
constexpr uint16_t FL_C2 = 2;
constexpr uint16_t FL_R = 3;

extern "C" __global__ __aicore__ void kda_k2_persistent_loop(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4,
    GM_ADDR pOut, GM_ADDR pVnew, GM_ADDR pVnewT,
    GM_ADDR pH0, GM_ADDR pS32, GM_ADDR pS16,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int32_t nv = NV;

    if ASCEND_IS_AIC {
        const int32_t blk = static_cast<int32_t>(GetBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> W, Qg, Aqk, Vt, Kt, S16;
        GlobalTensor<float> D1, D2, D3, D4;
        W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
        Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
        Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

        TPipe pipe;
        TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
        TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
        TQue<QuePosition::B1, 1> qw, qg, qs, qa, qv, qk;
        pipe.InitBuffer(qw, 1, M * D * 2);
        pipe.InitBuffer(qg, 1, M * D * 2);
        pipe.InitBuffer(qs, 1, nv * BV * D * 2);
        pipe.InitBuffer(qa, 1, M * K * 2);
        pipe.InitBuffer(qv, 1, nv * BV * K * 2);
        pipe.InitBuffer(qk, 1, D * K * 2);
        TQue<QuePosition::CO1, 1> qc;
        pipe.InitBuffer(qc, 1, (NG * M * N_D4 + nv * M * N) * 4);
        LocalTensor<float> cf = qc.AllocTensor<float>();
        LocalTensor<uint8_t> a8(TPosition::A2, 0, L0A_ELEMS * 2);
        // L0B: the two state tiles (64 fractals), reused by d34's 16.
        LocalTensor<uint8_t> b8(TPosition::B2, 0, nv * BV * D * 2);
        LocalTensor<bfloat16_t> l0a = a8.ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b = b8.ReinterpretCast<bfloat16_t>();

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // ---- stage 1: d1 = W @ S16^T, d2 = Qg @ S16^T for every head
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                CrossCoreWaitFlag(FL_R);
                const uint64_t a0 = static_cast<uint64_t>(bh * NT + chunk) * M * D;
                auto lw = qw.AllocTensor<bfloat16_t>();
                auto lg = qg.AllocTensor<bfloat16_t>();
                auto ls = qs.AllocTensor<bfloat16_t>();
                DataCopy(lw, W[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
                DataCopy(lg, Qg[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t s0 = static_cast<uint64_t>(bh * nv + iv) * BV * D;
                    DataCopy(ls[iv * BV * D], S16[s0], Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qw.EnQue(lw);
                qg.EnQue(lg);
                qs.EnQue(ls);
                lw = qw.DeQue<bfloat16_t>();
                lg = qg.DeQue<bfloat16_t>();
                ls = qs.DeQue<bfloat16_t>();
                LoadData(l0a, lw, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
                LoadData(l0a[M * D], lg, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[iv * BV * D], ls[iv * BV * D],
                             LoadData2dParams(0, 32, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[iv * M * N], l0a, l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                    Mmad(cf[(nv + iv) * M * N], l0a[M * D], l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t o0 = static_cast<uint64_t>(bh * nv + iv) * NT * TILE +
                                        static_cast<uint64_t>(chunk) * TILE;
                    auto ip1 = FixpipeParamsV220(N, M, 16, N, false);
                    ip1.quantPre = QuantMode_t::NoQuant;
                    ip1.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0], cf[iv * M * N], ip1);
                    auto ip2 = FixpipeParamsV220(N, M, 16, N, false);
                    ip2.quantPre = QuantMode_t::NoQuant;
                    ip2.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0], cf[(nv + iv) * M * N], ip2);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qw.FreeTensor(lw);
                qg.FreeTensor(lg);
                qs.FreeTensor(ls);
                // L0A/L0B/L0C are reused by the next stage on this same core;
                // the FIX_M event alone is not enough on this runtime.
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C1);
            }
            // ---- stage 3: d3 = Aqk @ v_new, d4 = v_new^T @ kg for every head
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t c = bh * NT + chunk;
                CrossCoreWaitFlag(FL_V);
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lv = qv.AllocTensor<bfloat16_t>();
                auto lk = qk.AllocTensor<bfloat16_t>();
                DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], M * K);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const int32_t task = bh * nv + iv;
                    DataCopy(lv[iv * BV * K],
                             Vt[(static_cast<uint64_t>(task) * NT + chunk) * BV * K], BV * K);
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
                LoadData(l0a, lv, LoadData2dParams(0, NG, 1, 0, 0, false, 0));
                LoadData(l0a[NG * M * K], la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                LoadData(l0b, lk, LoadData2dParams(0, D / M, 1, 0, 0, false, 0));
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[D * K + iv * BV * K], lv[iv * BV * K],
                             LoadData2dParams(0, BV / M, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                // L0C and L0B/L0A are free again once the d12 fixpipes retired.
                for (int32_t g = 0; g < NG; ++g) {
                    Mmad(cf[g * M * N_D4], l0a[g * M * K], l0b,
                         MmadParams(M, N_D4, K, 0, false, true));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[NG * M * N_D4 + iv * M * N], l0a[NG * M * K],
                         l0b[D * K + iv * BV * K], MmadParams(M, N, K, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t g = 0; g < NG; ++g) {
                    auto ip = FixpipeParamsV220(N_D4, M, 16, N_D4, false);
                    ip.quantPre = QuantMode_t::NoQuant;
                    ip.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D4[static_cast<uint64_t>(bh) * D * D + static_cast<uint64_t>(g) * M * D],
                        cf[g * M * N_D4], ip);
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    auto ip = FixpipeParamsV220(N, M, 16, N, false);
                    ip.quantPre = QuantMode_t::NoQuant;
                    ip.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D3[(static_cast<uint64_t>(bh * nv + iv) * NT + chunk) * TILE],
                        cf[NG * M * N_D4 + iv * M * N], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qa.FreeTensor(la);
                qv.FreeTensor(lv);
                qk.FreeTensor(lk);
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C2);
            }
        }
        qc.FreeTensor(cf);
        return;
    }

    if ASCEND_IS_AIV {
        const int32_t ratio = static_cast<int32_t>(GetTaskRation());
        const int32_t raw = static_cast<int32_t>(GetBlockIdx());
        const int32_t blk = ratio == 0 ? raw : raw / ratio;
        // Both subcores execute the same flag sequence; only the value tile
        // they work on differs.
        const int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> U, V, Vt, S16, Out;
        GlobalTensor<float> D1, D2, D3, D4, Decay, H0, S32;
        U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
        V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnew));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
        Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
        Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
        H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
        S32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pS32));

        TPipe pipe;
        TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        TEventID evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
        TBuf<TPosition::VECCALC> uu, ud1, uv, ut, uf, usc, uo, uof, ud2, ud3, ud4, udec, us, us16;
        pipe.InitBuffer(uu, M * D * sizeof(bfloat16_t));
        pipe.InitBuffer(ud1, TILE * sizeof(float));
        pipe.InitBuffer(uv, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(ut, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uf, M * D * sizeof(float));
        pipe.InitBuffer(usc, (BV / M) * M * M * sizeof(bfloat16_t));
        pipe.InitBuffer(uo, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uof, TILE * sizeof(float));
        pipe.InitBuffer(ud2, TILE * sizeof(float));
        pipe.InitBuffer(ud3, TILE * sizeof(float));
        pipe.InitBuffer(ud4, S_TILE * sizeof(float));
        pipe.InitBuffer(udec, D * sizeof(float));
        pipe.InitBuffer(us, MAXH * S_TILE * sizeof(float));
        pipe.InitBuffer(us16, S_TILE * sizeof(bfloat16_t));

        LocalTensor<bfloat16_t> ub = uu.Get<bfloat16_t>();
        LocalTensor<float> d1 = ud1.Get<float>();
        LocalTensor<bfloat16_t> vb = uv.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vt = ut.Get<bfloat16_t>();
        LocalTensor<float> vf = uf.Get<float>();
        LocalTensor<bfloat16_t> sc = usc.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> ob = uo.Get<bfloat16_t>();
        LocalTensor<float> of = uof.Get<float>();
        LocalTensor<float> d2 = ud2.Get<float>();
        LocalTensor<float> d3 = ud3.Get<float>();
        LocalTensor<float> d4 = ud4.Get<float>();
        LocalTensor<float> dec = udec.Get<float>();
        LocalTensor<float> st = us.Get<float>();
        LocalTensor<bfloat16_t> s16 = us16.Get<bfloat16_t>();

        // ---- start-up: bring the fp32 state on chip, publish the bf16 copy
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            LocalTensor<float> state = st[s * S_TILE];
            if (pH0 == nullptr) {
                Duplicate(state, 0.0f, S_TILE);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(state, H0[static_cast<uint64_t>(task) * S_TILE],
                         DataCopyParams(BV, 16, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
            }
            Cast(s16, state, RoundMode::CAST_RINT, S_TILE);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(ev3);
            WaitFlag<HardEvent::V_MTE3>(ev3);
            DataCopy(S16[static_cast<uint64_t>(task) * S_TILE], s16, DataCopyParams(BV, 8, 0, 0));
            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t s = 0; s < nh; ++s) {
            CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);   // prologue: first d12 waits
        }

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // ---- stage 2: v_new = u - d1 (and its transpose) per head
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                CrossCoreWaitFlag(FL_C1);
                // The previous iteration's vector work may still be reading the
                // UB staging buffers (ub/vf/vb/sc) that this iteration loads
                // into.  A one-chunk-per-launch kernel never sees this WAR
                // hazard; a loop does.
                PipeBarrier<PIPE_ALL>();
                const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
                const uint64_t out0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
                DataCopy(ub, U[u0], DataCopyParams(M, 8, 0, 0));
                DataCopy(d1, D1[out0], DataCopyParams(M, 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                Cast(vf, ub, RoundMode::CAST_NONE, M * D);
                PipeBarrier<PIPE_V>();
                for (int32_t i = 0; i < M; ++i) {
                    Sub(vf[i * BV], vf[i * D + iv * BV], d1[i * BV], BV);
                }
                Cast(vb, vf, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
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
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_V);
            }
            // ---- stage 4: out = d2*scale + d3 and the state recurrence
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                CrossCoreWaitFlag(FL_C2);
                // Same WAR hazard as stage 2: d2/d3/d4/dec, ob and s16 are all
                // rewritten here while the previous iteration's reads of them
                // (and its MTE3 copies out of ob/s16) may still be in flight.
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> state = st[s * S_TILE];
                const uint64_t t0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
                const uint64_t d4base = static_cast<uint64_t>(bh) * D * D +
                                        static_cast<uint64_t>(iv) * BV * D;
                DataCopy(d2, D2[t0], DataCopyParams(M, 8, 0, 0));
                DataCopy(d3, D3[t0], DataCopyParams(M, 8, 0, 0));
                DataCopy(d4, D4[d4base], DataCopyParams(BV, 16, 0, 0));
                DataCopy(dec, Decay[static_cast<uint64_t>(c) * D], DataCopyParams(1, 16, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                // out = d2 * scale + d3 is accumulated in fp32 and only then
                // rounded to bf16, exactly like the separated outstate kernel.
                Muls(of, d2, scale, TILE);
                Add(of, of, d3, TILE);
                Cast(ob, of, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                Mul(state, state, dec, 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                Mul(state[64], state[64], dec[64], 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                Add(state, state, d4, S_TILE);
                PipeBarrier<PIPE_V>();
                Cast(s16, state, RoundMode::CAST_RINT, S_TILE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                DataCopy(Out[t0], ob, DataCopyParams(M, 4, 0, 0));
                DataCopy(S16[static_cast<uint64_t>(task) * S_TILE], s16, DataCopyParams(BV, 8, 0, 0));
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
            }
        }
        // ---- finally: publish the fp32 state
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            DataCopy(S32[static_cast<uint64_t>(task) * S_TILE], st[s * S_TILE],
                     DataCopyParams(BV, 16, 0, 0));
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
