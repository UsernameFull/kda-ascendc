// Fused d3 + d4 Cube kernel for the "separated" K2 path.
//   d3[i][v] = sum_j Aqk[i][j] * v_new[j][v]   (task owns a 16 x 64 tile)
//   d4[v][k] = sum_i v_new[i][v] * kg[i][k]    (task owns rows iv*64..iv*64+63
//                                               of the per-chunk D4[c][D][D])
// One AIC block per (b, h, v-tile).  Both dots reuse the operand sequences of
// the validated kda_k2_d3_cube_bv64 / kda_k2_d4_full kernels, so the transposed
// v_new^T / kg^T staging buffers are consumed as-is (no in-kernel transpose).
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128, BV = 64, N3 = 64, N = 128;

extern "C" __global__ __aicore__ void kda_k2_d34_kernel(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pKgT, GM_ADDR pD3, GM_ADDR pD4,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t task = GetBlockIdx();
    if (task >= BH * NV) return;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;
    const int32_t c = bh * NT + chunk;
    const uint64_t t0 = static_cast<uint64_t>(task) * NT + chunk;
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
    GlobalTensor<bfloat16_t> Aqk, Vt, Kt;
    GlobalTensor<float> D3, D4;
    Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

    // ---- d4 = v_new^T @ kg, rows of this v-tile only
    const uint64_t d4c = static_cast<uint64_t>(c) * D * D +
                         static_cast<uint64_t>(iv) * BV * D;
    for (int rr = 0; rr < 4; ++rr) {
        auto la = qa.AllocTensor<bfloat16_t>();
        auto lb = qb.AllocTensor<bfloat16_t>();
        DataCopy(la, Vt[t0 * BV * K + static_cast<uint64_t>(rr) * M * K],
                 Nd2NzParams(1, M, K, 0, K, M, 1, 0));
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
        for (int nb = 0; nb < 8; ++nb) {
            auto ip = FixpipeParamsV220(M, M, 1, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(
                D4[d4c + static_cast<uint64_t>(rr) * M * D + nb * M],
                cf[nb * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(efm);
        WaitFlag<HardEvent::FIX_M>(efm);
        qa.FreeTensor(la);
        qb.FreeTensor(lb);
        PipeBarrier<PIPE_ALL>();
    }

    // ---- d3 = Aqk @ v_new for this v-tile
    {
        auto la = qa.AllocTensor<bfloat16_t>();
        auto lb = qb.AllocTensor<bfloat16_t>();
        DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K],
                 Nd2NzParams(1, M, K, 0, K, M, 1, 0));
        DataCopy(lb, Vt[t0 * BV * K], Nd2NzParams(1, BV, K, 0, K, BV, 1, 0));
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
                D3[t0 * M * BV + nb * M], cf[nb * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(efm);
        WaitFlag<HardEvent::FIX_M>(efm);
        qa.FreeTensor(la);
        qb.FreeTensor(lb);
    }
    qc.FreeTensor(cf);
}
