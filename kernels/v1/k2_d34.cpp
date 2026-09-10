// Fused d3 + d4 Cube kernel for the "separated" K2 path.
//   d3[i][v] = sum_j Aqk[i][j] * v_new[j][v]   (task owns a 16 x 64 tile)
//   d4[v][k] = sum_i v_new[i][v] * kg[i][k]    (task owns rows iv*64..iv*64+63
//                                               of the per-chunk D4[c][D][D])
// Both dots consume the same two staged tiles (v_new^T [64, 16] and
// kg^T [128, 16]), so the kernel loads each one once, stages every L0 operand
// with a single LoadData, issues the five Mmads back to back and drains the
// M/FIX pipes only once.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, K = 16, D = 128, BV = 64, N3 = 64, N = 128;
constexpr int32_t VT = BV / M;  // 16-row groups of v_new^T

extern "C" __global__ __aicore__ void kda_k2_d34_kernel(
    GM_ADDR pAqk, GM_ADDR pVnewT, GM_ADDR pKgT, GM_ADDR pD3, GM_ADDR pD4,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk0, int32_t nchunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t tasks = BH * NV;
    const int32_t blk = GetBlockIdx();
    if (blk >= tasks * nchunk) return;
    const int32_t task = blk % tasks;
    const int32_t chunk = chunk0 + blk / tasks;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;
    const int32_t c = bh * NT + chunk;
    const uint64_t t0 = static_cast<uint64_t>(task) * NT + chunk;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> qv, qk, qa;
    pipe.InitBuffer(qv, 1, BV * K * 2);
    pipe.InitBuffer(qk, 1, D * K * 2);
    pipe.InitBuffer(qa, 1, M * K * 2);
    TQue<QuePosition::CO1, 1> qc;
    pipe.InitBuffer(qc, 1, (VT + 1) * M * N * 4);
    LocalTensor<float> cf = qc.AllocTensor<float>();
    LocalTensor<uint8_t> a8(TPosition::A2, 0, (VT + 1) * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, (D + BV) * K * 2);
    LocalTensor<bfloat16_t> a = a8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> Aqk, Vt, Kt;
    GlobalTensor<float> D3, D4;
    Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

    // ---- stage the three operands once: v_new^T, kg^T and Aqk
    auto lv = qv.AllocTensor<bfloat16_t>();
    auto lk = qk.AllocTensor<bfloat16_t>();
    auto la = qa.AllocTensor<bfloat16_t>();
    DataCopy(lv, Vt[t0 * BV * K], Nd2NzParams(1, BV, K, 0, K, BV, 1, 0));
    DataCopy(lk, Kt[static_cast<uint64_t>(c) * D * K], Nd2NzParams(1, D, K, 0, K, D, 1, 0));
    DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], Nd2NzParams(1, M, K, 0, K, M, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);
    qv.EnQue(lv);
    qk.EnQue(lk);
    qa.EnQue(la);
    lv = qv.DeQue<bfloat16_t>();
    lk = qk.DeQue<bfloat16_t>();
    la = qa.DeQue<bfloat16_t>();

    LoadData(a, lv, LoadData2dParams(0, VT, 1, 0, 0, false, 0));
    LoadData(a[VT * M * K], la, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadData(b, lk, LoadData2dParams(0, D / M, 1, 0, 0, false, 0));
    LoadData(b[D * K], lv, LoadData2dParams(0, VT, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);

    // ---- d4 = v_new^T @ kg: one 16 x 128 C tile per 16-row group
    for (int32_t rr = 0; rr < VT; ++rr) {
        Mmad(cf[rr * M * N], a[rr * M * K], b, MmadParams(M, N, K, 0, false, true));
    }
    // ---- d3 = Aqk @ v_new
    Mmad(cf[VT * M * N], a[VT * M * K], b[D * K], MmadParams(M, N3, K, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);

    const uint64_t d4c = static_cast<uint64_t>(c) * D * D +
                         static_cast<uint64_t>(iv) * BV * D;
    // One Fixpipe per 16 x 128 C tile: nSize covers the whole tile and
    // srcStride=16 walks the 16 x 16 C0 fractals.  Per-fractal Fixpipe calls
    // cost ~6 us per launch, which dominated this kernel.
    for (int32_t rr = 0; rr < VT; ++rr) {
        auto ip = FixpipeParamsV220(N, M, 16, N, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(
            D4[d4c + static_cast<uint64_t>(rr) * M * D], cf[rr * M * N], ip);
    }
    {
        auto ip = FixpipeParamsV220(N3, M, 16, N3, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(D3[t0 * M * BV], cf[VT * M * N], ip);
    }
    SetFlag<HardEvent::FIX_M>(efm);
    WaitFlag<HardEvent::FIX_M>(efm);
    qv.FreeTensor(lv);
    qk.FreeTensor(lk);
    qa.FreeTensor(la);
    qc.FreeTensor(cf);
    PipeBarrier<PIPE_ALL>();
}
