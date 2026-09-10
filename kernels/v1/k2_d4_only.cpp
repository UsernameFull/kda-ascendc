// d4[v][k] = sum_i v_new[i][v] * kg[i][k] for the v-tile owned by one block.
// One AIC block per (b, h, v-tile); the result fills the bottom (iv == 0) or
// top (iv == 1) half of the per-chunk D4[c][D][D] buffer, which is the layout
// the separated outstate kernel reads (d4_reuse == 0).
//
// The operand sequence mirrors the validated kda_k2_d4_full / mix d4 path:
// A = v_new^T [16, 16] (bf16), B = kg^T [128, 16] (bf16) -> [16, 128] fp32.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128, BV = 64, N = 128;

extern "C" __global__ __aicore__ void kda_k2_d4_only_kernel(
    GM_ADDR pVnewT, GM_ADDR pKgT, GM_ADDR pD4,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t task = GetBlockIdx();
    if (task >= BH * NV) return;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;
    const int32_t c = bh * NT + chunk;
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
    GlobalTensor<bfloat16_t> Vt, Kt;
    GlobalTensor<float> D4;
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
    const uint64_t d4c = static_cast<uint64_t>(c) * D * D +
                         static_cast<uint64_t>(iv) * BV * D;
    for (int rr = 0; rr < 4; ++rr) {
        const uint64_t ao = (static_cast<uint64_t>(task) * NT + chunk) * BV * K +
                            static_cast<uint64_t>(rr) * M * K;
        auto la = qa.AllocTensor<bfloat16_t>();
        auto lb = qb.AllocTensor<bfloat16_t>();
        DataCopy(la, Vt[ao], M * K);
        DataCopy(lb, Kt[static_cast<uint64_t>(c) * D * K], D * K);
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
        {
            auto ip = FixpipeParamsV220(N, M, 16, N, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(
                D4[d4c + static_cast<uint64_t>(rr) * M * D], cf, ip);
        }
        SetFlag<HardEvent::FIX_M>(efm);
        WaitFlag<HardEvent::FIX_M>(efm);
        qa.FreeTensor(la);
        qb.FreeTensor(lb);
        PipeBarrier<PIPE_ALL>();
    }
    qc.FreeTensor(cf);
}
