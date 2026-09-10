// d3 = Aqk @ v_new (Aqk [16,16] bf16, v_newT [BV,16] bf16 -> [16,BV] fp32).
// One block per (b, h, v-tile); matches the Cube sequence used by the
// combined MIX kernels so the results are bit-identical.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128, BV = 64, N3 = 64;

extern "C" __global__ __aicore__ void kda_k2_d3_cube_bv64(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pD3,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t task = GetBlockIdx();
    if (task >= BH * NV) return;
    const int32_t bh = task / NV;
    const int32_t c = bh * NT + chunk;
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
    qc.FreeTensor(cf);
}
