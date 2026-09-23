// K2 split, half B: the output, with the serial state chain removed from it.
//
//     O[c] = Q[c] H[c] + A[c] Z[c]
//
// H[c] is the chunk-entry state the state kernel publishes per chunk (slot
// (task, chunk) of pHsnap) and Z[c] is its v_new^T; neither depends on any
// other chunk's output, so every (head, value tile, chunk) tile here is
// independent.  The block/flag skeleton is still the loop kernel's - one block
// owns up to MAXH heads, each AIV subcore one value tile, and FL_C1/FL_C2/FL_R
// are the depth-one hand-off - but there is no FL_V, because the AIV produces
// nothing the Cube consumes: the two Mmads of a tile (Qg @ H^T and Aqk @ Z)
// read only GM operands, and the vector side only combines them.
//
// The arithmetic is the fused loop's, unchanged: the Cube rounds d2 and d3 to
// bf16 exactly as k2_persistent_loop.cpp does, and the AIV accumulates
// out = d2 * scale + d3 in fp32 and rounds once, so the split is bit-exact
// against the fused kernel (tools/probe_state_out_split.py asserts it).
//
// The back-pressure flag is not needed for correctness (every D2/D3 slot is
// written once), only to bound the flag backlog and keep the D2/D3 reads in
// L2 - it is the same depth-one throttle the fused loop runs.
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
constexpr int32_t TILE = M * BV;
constexpr int32_t S_TILE = BV * D;
constexpr int32_t NG = 2 * BV / M;
constexpr int32_t MAXH = KDA_MAXH;
// L0A: Qg (M x D) then Aqk (M x K).  L0B: the nv state tiles (BV x D) then the
// nv v_new tiles (BV x K).  Both are far inside the 64 KB parts (24 KB / 48 KB
// at C = 64) because the output has no resident state.
constexpr int32_t L0A_ELEMS = M * D + M * K;
constexpr int32_t L0B_ELEMS = 2 * BV * D + 2 * BV * K;

constexpr uint16_t FL_C1 = 0;   // AIC -> AIV: d2 = Qg @ H^T ready
constexpr uint16_t FL_C2 = 2;   // AIC -> AIV: d3 = Aqk @ v_new ready
constexpr uint16_t FL_R = 3;    // AIV -> AIC: slot consumed (back pressure)

extern "C" __global__ __aicore__ void kda_k2_out_parallel(
    GM_ADDR pQg, GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pD2, GM_ADDR pD3,
    GM_ADDR pHsnap, GM_ADDR pOut, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, float scale, int32_t NH) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int32_t nv = NV;

    if ASCEND_IS_AIC {
        const int32_t blk = static_cast<int32_t>(GetBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> Qg, Aqk, Vt, Hsnap, D2, D3;
        Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
        Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Hsnap.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pHsnap));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD3));

        TPipe pipe;
        TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
        TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
        TEventID em1 = pipe.AllocEventID<HardEvent::M_MTE1>();
        TQue<QuePosition::B1, 1> qg, qs, qa, qx;
        pipe.InitBuffer(qg, 1, M * D * 2);
        pipe.InitBuffer(qs, 1, nv * BV * D * 2);
        pipe.InitBuffer(qa, 1, M * K * 2);
        pipe.InitBuffer(qx, 1, nv * BV * K * 2);
        TQue<QuePosition::CO1, 1> qc;
        pipe.InitBuffer(qc, 1, 2 * nv * M * N * 4);
        LocalTensor<float> cf = qc.AllocTensor<float>();
        LocalTensor<uint8_t> a8(TPosition::A2, 0, L0A_ELEMS * 2);
        LocalTensor<uint8_t> b8(TPosition::B2, 0, L0B_ELEMS * 2);
        LocalTensor<bfloat16_t> l0a = a8.ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b = b8.ReinterpretCast<bfloat16_t>();

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t c = bh * NT + chunk;
                const uint64_t a0 = static_cast<uint64_t>(c) * M * D;
                // ---- d2 = Qg @ H[c]^T
                auto lq = qg.AllocTensor<bfloat16_t>();
                auto ls = qs.AllocTensor<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lq[b * FR * D], Qg[a0 + b * FR * D],
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
                qg.EnQue(lq);
                qs.EnQue(ls);
                lq = qg.DeQue<bfloat16_t>();
                ls = qs.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[b * FR * D], lq[b * FR * D],
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
                    auto ip = FixpipeParamsV220(N, M, M, N, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(D2[o0], cf[iv * M * N], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qg.FreeTensor(lq);
                qs.FreeTensor(ls);
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C1);
                // ---- d3 = Aqk @ v_new
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lx = qx.AllocTensor<bfloat16_t>();
                if (K == FR) {
                    DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], M * K);
                } else {
                    for (int32_t b = 0; b < NB; ++b) {
                        DataCopy(la[b * FR * K],
                                 Aqk[static_cast<uint64_t>(c) * M * K + b * FR * K],
                                 Nd2NzParams(1, FR, K, 0, K, FR, 1, 0));
                    }
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const int32_t task = bh * nv + iv;
                    const uint64_t t0 = (static_cast<uint64_t>(task) * NT + chunk) * BV * K;
                    if (K == FR) {
                        DataCopy(lx[iv * BV * K], Vt[t0], BV * K);
                    } else {
                        for (int32_t cc = 0; cc < K / FR; ++cc) {
                            DataCopy(lx[iv * BV * K + cc * (BV / FR) * FR * FR],
                                     Vt[t0 + cc * FR * FR],
                                     DataCopyParams(BV / FR, FR * FR / 16,
                                                    (K / FR) * FR * FR / 16 - FR * FR / 16, 0));
                        }
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qa.EnQue(la);
                qx.EnQue(lx);
                la = qa.DeQue<bfloat16_t>();
                lx = qx.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[M * D + b * FR * K], la[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[2 * BV * D + iv * BV * K], lx[iv * BV * K],
                             LoadData2dParams(0, BV * K / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[nv * M * N + iv * M * N], l0a[M * D], l0b[2 * BV * D + iv * BV * K],
                         MmadParams(M, N, K, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t o0 = static_cast<uint64_t>(bh * nv + iv) * NT * TILE +
                                        static_cast<uint64_t>(chunk) * TILE;
                    auto ip = FixpipeParamsV220(N, M, M, N, false);
                    ip.quantPre = QuantMode_t::F322BF16;
                    ip.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(D3[o0], cf[nv * M * N + iv * M * N], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qa.FreeTensor(la);
                qx.FreeTensor(lx);
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

        GlobalTensor<bfloat16_t> D2, D3, Out;
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pD3));
        Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));

        TPipe pipe;
        TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        // No resident state and no transposes: five tiles and one event pair
        // are the whole vector side (56 KB of the 192 KB part).  of and the
        // cast scratch have to be *different* fp32 tiles: out = d2*scale + d3
        // is accumulated in fp32, so the scaled d2 has to survive until d3 is
        // in registers.
        TBuf<TPosition::VECCALC> uA, uE, uB, uC, uD;
        pipe.InitBuffer(uA, TILE * sizeof(float));
        pipe.InitBuffer(uE, TILE * sizeof(float));
        pipe.InitBuffer(uB, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uC, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uD, TILE * sizeof(bfloat16_t));
        LocalTensor<float> of = uA.Get<float>();
        LocalTensor<float> scratch = uE.Get<float>();
        LocalTensor<bfloat16_t> d2 = uB.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> d3 = uC.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> ob = uD.Get<bfloat16_t>();

        for (int32_t s = 0; s < nh; ++s) {
            CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);   // prologue: first d2 waits
        }

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                // Same WAR guard as the fused loop's stage 4: d2/d3/ob/of are
                // rewritten here while the previous tile's MTE3 copies out of
                // them may still be in flight.
                PipeBarrier<PIPE_ALL>();
                const uint64_t t0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
                // Both reads are gated on the *current* tile's flags (unlike
                // the fused loop, whose stage 4 could hoist d2 because stage 2
                // had already waited FL_C1); the two waits sit back to back so
                // the second MTE2 read still issues before the math starts.
                CrossCoreWaitFlag(FL_C1);
                DataCopy(d2, D2[t0], DataCopyParams(M, BV / 16, 0, 0));
                CrossCoreWaitFlag(FL_C2);
                DataCopy(d3, D3[t0], DataCopyParams(M, BV / 16, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                // out = d2 * scale + d3 is accumulated in fp32 and only then
                // rounded to bf16, exactly like the fused outstate stage.
                Cast(scratch, d2, RoundMode::CAST_NONE, TILE);
                Muls(of, scratch, scale, TILE);
                Cast(scratch, d3, RoundMode::CAST_NONE, TILE);
                Add(of, of, scratch, TILE);
                Cast(ob, of, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                const int32_t hh = bh - (bh / NH) * NH;
                const uint64_t obase = (static_cast<uint64_t>(bh / NH) * NT + chunk) *
                                           (static_cast<uint64_t>(M) * NH * D) +
                                       static_cast<uint64_t>(hh) * D + iv * BV;
                DataCopy(Out[obase], ob,
                         DataCopyParams(M, BV / 16, 0,
                                        static_cast<uint16_t>(NH * D / 16 - BV / 16)));
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
