// K2 split, half A: the serial state chain, with the output removed from it.
//
// Route 1 of the 2026-09-22 redesign (docs/PREFILL_LIFECYCLE_REFACTOR_20260922.md
// section 4.1): the chunk recurrence is
//
//     Z     = U - W S
//     O     = Q S + A Z
//     Snext = diag(d) S + Kg^T Z
//
// and only Z and Snext gate the next chunk.  O is late-computable, so this
// kernel keeps the chain and drops every piece of the output: no Qg load, no
// Qg @ S^T Mmad, no Aqk load, no d3 = Aqk @ v_new Mmad, and stage 4 is the
// state recurrence alone (no d2/d3 read, no scale-add-cast, no out store).
//
// It is the same block/protocol skeleton as k2_persistent_loop.cpp - one
// block owns up to MAXH heads, each AIV subcore owns one value tile, four
// flags carry the depth-one loop - with two address changes that pay for the
// split's snapshot:
//
//   * the bf16 state the AIC reads is published *per chunk* instead of into
//     one slot: slot (task, chunk) holds the chunk-entry state H[c] (the
//     same bytes the loop used to keep in S16, just at a chunk-indexed
//     address), so the snapshot costs the chain no extra write at all;
//   * the fp32 state still never leaves the AIV until the end (S32), and the
//     only reader of the snapshot inside this kernel is stage 1 itself.
//
// The parallel half is k2_out_parallel.cpp: O[c] = Q[c] H[c] + A[c] Z[c] over
// every (head, value tile, chunk) at once.  The pair is only worth switching
// to if the *sum* of the two kernels, the snapshot traffic and the layout
// conversions beats the fused loop - tools/probe_state_out_split.py measures
// exactly that, interleaved in one process, and asserts the arms are
// bit-identical before reporting.
//
// Flags: FL_C1 AIC->AIV d1 ready, FL_V AIV->AIC v_new^T published, FL_C2
// AIC->AIV d4 ready, FL_R AIV->AIC state published (the depth-one back
// pressure that also keeps the snapshot in L2).
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
#ifndef KDA_MAXH
#define KDA_MAXH 4
#endif
constexpr int32_t M = KDA_CHUNK;
constexpr int32_t FR = 16;
constexpr int32_t NB = M / FR;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t K = M;
constexpr int32_t N = 64;
constexpr int32_t N_D4 = 128;
constexpr int32_t TILE = M * BV;
constexpr int32_t S_TILE = BV * D;
constexpr int32_t NG = 2 * BV / M;
constexpr int32_t MAXH = KDA_MAXH;
constexpr int32_t L0A_ELEMS = (M * D > NG * M * K) ? M * D : NG * M * K;

constexpr uint16_t FL_C1 = 0;
constexpr uint16_t FL_V = 1;
constexpr uint16_t FL_C2 = 2;
constexpr uint16_t FL_R = 3;

extern "C" __global__ __aicore__ void kda_k2_state_loop(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD4, GM_ADDR pHsnap,
    GM_ADDR pVnew, GM_ADDR pVnewT, GM_ADDR pH0, GM_ADDR pS32,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, int32_t NH) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int32_t nv = NV;

    if ASCEND_IS_AIC {
        const int32_t blk = static_cast<int32_t>(GetBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> W, Vt, Kt, Hsnap, D1;
        GlobalTensor<float> D4;
        W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
        Hsnap.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pHsnap));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD1));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

        TPipe pipe;
        TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
        TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
        TEventID em1 = pipe.AllocEventID<HardEvent::M_MTE1>();
        TQue<QuePosition::B1, 1> qw, qs, qv, qk;
        pipe.InitBuffer(qw, 1, M * D * 2);
        pipe.InitBuffer(qs, 1, nv * BV * D * 2);
        pipe.InitBuffer(qv, 1, NG * M * K * 2);
        pipe.InitBuffer(qk, 1, D * K * 2);
        TQue<QuePosition::CO1, 1> qc;
        pipe.InitBuffer(qc, 1, (NG * M * N_D4 + nv * M * N) * 4);
        LocalTensor<float> cf = qc.AllocTensor<float>();
        LocalTensor<uint8_t> a8(TPosition::A2, 0, L0A_ELEMS * 2);
        LocalTensor<uint8_t> b8(TPosition::B2, 0, nv * BV * D * 2);
        LocalTensor<bfloat16_t> l0a = a8.ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b = b8.ReinterpretCast<bfloat16_t>();

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // ---- stage 1: d1 = W @ H[c]^T for every head (no d2: the output
            // kernel owns Q @ H).
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const uint64_t a0 = static_cast<uint64_t>(bh * NT + chunk) * M * D;
                auto lw = qw.AllocTensor<bfloat16_t>();
                auto ls = qs.AllocTensor<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lw[b * FR * D], W[a0 + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_R);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t s0 = (static_cast<uint64_t>(bh * nv + iv) * NT + chunk) * S_TILE;
                    DataCopy(ls[iv * BV * D], Hsnap[s0],
                             Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qw.EnQue(lw);
                qs.EnQue(ls);
                lw = qw.DeQue<bfloat16_t>();
                ls = qs.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[b * FR * D], lw[b * FR * D],
                             LoadData2dParams(0, D / FR, 1, 0, 0, false, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[iv * BV * D], ls[iv * BV * D],
                             LoadData2dParams(0, BV * D / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[iv * M * N], l0a, l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t o0 = static_cast<uint64_t>(bh * nv + iv) * NT * TILE +
                                        static_cast<uint64_t>(chunk) * TILE;
                    auto ip1 = FixpipeParamsV220(N, M, M, N, false);
                    ip1.quantPre = QuantMode_t::F322BF16;
                    ip1.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(D1[o0], cf[iv * M * N], ip1);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qw.FreeTensor(lw);
                qs.FreeTensor(ls);
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C1);
            }
            // ---- stage 3: d4 = v_new^T @ kg (the state's delta only; Aqk and
            // the d3 = Aqk @ v_new Mmad moved to the output kernel).
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t c = bh * NT + chunk;
                auto lv = qv.AllocTensor<bfloat16_t>();
                auto lk = qk.AllocTensor<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lk[b * FR * D], Kt[static_cast<uint64_t>(c) * M * D + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_V);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const int32_t task = bh * nv + iv;
                    const uint64_t t0 = (static_cast<uint64_t>(task) * NT + chunk) * BV * K;
                    DataCopy(lv[iv * BV * K], Vt[t0], BV * K);
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qv.EnQue(lv);
                qk.EnQue(lk);
                lv = qv.DeQue<bfloat16_t>();
                lk = qk.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NG * NB; ++b) {
                    LoadData(l0a[b * FR * K], lv[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                }
                for (int32_t b = 0; b < NB; ++b) {
                    LoadDataWithTranspose(l0b[b * (D / FR) * 256], lk[b * FR * D],
                                          LoadData2dTransposeParams(0, D / FR, 1, 0, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                Mmad(cf, l0a, l0b, MmadParams(NG * M, N_D4, K, 0, false, true));
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                {
                    auto ip = FixpipeParamsV220(N_D4, NG * M, NG * M, N_D4, false);
                    ip.quantPre = QuantMode_t::NoQuant;
                    ip.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D4[static_cast<uint64_t>(bh) * D * D], cf[0], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qv.FreeTensor(lv);
                qk.FreeTensor(lk);
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
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
        const int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> U, V, Vt, Hsnap, D1;
        GlobalTensor<float> D4, Decay, H0, S32;
        U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
        V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnew));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Hsnap.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pHsnap));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD1));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
        Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
        H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
        S32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pS32));

        TPipe pipe;
        TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        TEventID evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
        TEventID e3v = pipe.AllocEventID<HardEvent::MTE3_V>();
        constexpr int32_t HALF_BYTES = (TILE * 2 > S_TILE) ? TILE * 2 : S_TILE;
        // Same phase-aliasing as the fused loop, minus the output side: A is
        // vf only, E is sc/vt (stage 2) | d1f (stage 4 has no casts left).
        TBuf<TPosition::VECCALC> uA, uB, uC, uD, uE, udec, us;
        pipe.InitBuffer(uA, TILE * sizeof(float));
        pipe.InitBuffer(uB, HALF_BYTES);
        pipe.InitBuffer(uC, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uD, HALF_BYTES);
        pipe.InitBuffer(uE, TILE * sizeof(float));
        pipe.InitBuffer(udec, D * sizeof(float));
        pipe.InitBuffer(us, MAXH * S_TILE * sizeof(float));

        LocalTensor<float> vf = uA.Get<float>();
        LocalTensor<bfloat16_t> sc = uE.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vt = sc[TILE];
        LocalTensor<float> d1f = uE.Get<float>();
        LocalTensor<bfloat16_t> ub = uB.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> s16 = uB.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> d1 = uC.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vb = uD.Get<bfloat16_t>();
        LocalTensor<float> d4 = uD.Get<float>();
        LocalTensor<float> dec = udec.Get<float>();
        LocalTensor<float> st = us.Get<float>();

        // ---- start-up: fp32 state on chip, publish the chunk-0 snapshot
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            LocalTensor<float> state = st[s * S_TILE];
            if (pH0 == nullptr) {
                Duplicate(state, 0.0f, S_TILE);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(state, H0[static_cast<uint64_t>(task) * S_TILE],
                         DataCopyParams(BV, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
            }
            for (int32_t hf = 0; hf < 2; ++hf) {
                if (hf == 1) PipeBarrier<PIPE_ALL>();
                LocalTensor<float> sh = state[hf * (S_TILE / 2)];
                Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 2);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                DataCopy(Hsnap[static_cast<uint64_t>(task) * NT * S_TILE + hf * (BV / 2) * D], s16,
                         DataCopyParams(BV / 2, D / 16, 0, 0));
            }
            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t s = 0; s < nh; ++s) {
            CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
        }

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // ---- stage 2: v_new = u - d1 (and its packed transpose)
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                PipeBarrier<PIPE_ALL>();
                const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
                const uint64_t out0 = (static_cast<uint64_t>(task) * NT + chunk) * TILE;
                DataCopy(ub, U[u0 + iv * BV], DataCopyParams(M, BV / 16, (D - BV) / 16, 0));
                CrossCoreWaitFlag(FL_C1);
                DataCopy(d1, D1[out0], DataCopyParams(M, BV / 16, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                Cast(vf, ub, RoundMode::CAST_NONE, M * BV);
                Cast(d1f, d1, RoundMode::CAST_NONE, TILE);
                PipeBarrier<PIPE_V>();
                Sub(vf, vf, d1f, BV, M,
                    BinaryRepeatParams(1, 1, 1, BV / 8, BV / 8, BV / 8));
                Cast(vb, vf, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(evm2);
                WaitFlag<HardEvent::V_MTE2>(evm2);
                for (int32_t j0 = 0; j0 < BV / FR; ++j0) {
                    for (int32_t m0 = 0; m0 < NB; ++m0) {
                        DataCopy(sc[(j0 * NB + m0) * FR * FR], vb[m0 * FR * BV + j0 * FR],
                                 DataCopyParams(FR, 1, BV / FR - 1, 0));
                    }
                }
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                for (int32_t bl = 0; bl < (BV / FR) * NB; ++bl) {
                    AscendC::Transpose(vt[bl * FR * FR], sc[bl * FR * FR]);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                if (pVnew != nullptr) DataCopy(V[out0], vb, DataCopyParams(M, BV / 16, 0, 0));
                DataCopy(Vt[out0], vt, BV * M);
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_V);
            }
            // ---- stage 4: S = diag(decay) S + d4, and publish H[chunk + 1]
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> state = st[s * S_TILE];
                const uint64_t d4base = static_cast<uint64_t>(bh) * D * D +
                                        static_cast<uint64_t>(iv) * BV * D;
                DataCopy(dec, Decay[static_cast<uint64_t>(c) * D], DataCopyParams(1, D / 8, 0, 0));
                CrossCoreWaitFlag(FL_C2);
                DataCopy(d4, D4[d4base], DataCopyParams(BV / 4, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                const bool publish = (chunk + 1 < NT);
                const uint64_t next0 = (static_cast<uint64_t>(task) * NT + chunk + 1) * S_TILE;
                for (int32_t hf = 0; hf < 4; ++hf) {
                    if (hf > 0) {
                        WaitFlag<HardEvent::MTE2_V>(e2v);
                        WaitFlag<HardEvent::MTE3_V>(e3v);
                    }
                    LocalTensor<float> sh = state[hf * (S_TILE / 4)];
                    Mul(sh, sh, dec, 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Mul(sh[64], sh[64], dec[64], 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Add(sh, sh, d4, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(ev3);
                    WaitFlag<HardEvent::V_MTE3>(ev3);
                    if (publish) {
                        DataCopy(Hsnap[next0 + hf * (BV / 4) * D], s16,
                                 DataCopyParams(BV / 4, D / 16, 0, 0));
                    }
                    if (hf < 3) {
                        SetFlag<HardEvent::MTE3_V>(e3v);
                        SetFlag<HardEvent::V_MTE2>(evm2);
                        WaitFlag<HardEvent::V_MTE2>(evm2);
                        DataCopy(d4, D4[d4base + (hf + 1) * (BV / 4) * D],
                                 DataCopyParams(BV / 4, D / 8, 0, 0));
                        SetFlag<HardEvent::MTE2_V>(e2v);
                    }
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
            }
        }
        // ---- finally: publish the fp32 state
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            DataCopy(S32[static_cast<uint64_t>(task) * S_TILE], st[s * S_TILE],
                     DataCopyParams(BV, D / 8, 0, 0));
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
