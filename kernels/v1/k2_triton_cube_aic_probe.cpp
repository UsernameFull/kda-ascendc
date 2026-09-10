#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void kda_k2_triton_cube_aic_probe(
    GM_ADDR pA, GM_ADDR pB, GM_ADDR pOut, int32_t blocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t block = GetBlockIdx();
    if (block >= blocks) return;
    constexpr int32_t M = 16;
    constexpr int32_t N = 64;
    constexpr int32_t K = 128;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::B1> a_buf, b_buf;
    TBuf<TPosition::CO1> c_buf;
    TBuf<TPosition::VECCALC> v_buf;
    pipe.InitBuffer(a_buf, M * K * sizeof(bfloat16_t));
    pipe.InitBuffer(b_buf, N * K * sizeof(bfloat16_t));
    pipe.InitBuffer(c_buf, M * N * sizeof(float));
    pipe.InitBuffer(v_buf, M * N * sizeof(float));
    LocalTensor<bfloat16_t> a = a_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b_buf.Get<bfloat16_t>();
    LocalTensor<uint8_t> a2_bytes(TPosition::A2, 0, M * K * sizeof(bfloat16_t));
    LocalTensor<uint8_t> b2_bytes(TPosition::B2, 0, N * K * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> a2 = a2_bytes.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b2 = b2_bytes.ReinterpretCast<bfloat16_t>();
    LocalTensor<float> c = c_buf.Get<float>();
    LocalTensor<float> v = v_buf.Get<float>();
    GlobalTensor<bfloat16_t> A, B;
    GlobalTensor<float> Out;
    A.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA));
    B.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pB));
    Out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pOut));
    const uint64_t offset_a = static_cast<uint64_t>(block) * M * K;
    const uint64_t offset_b = static_cast<uint64_t>(block) * N * K;
    const uint64_t offset_o = static_cast<uint64_t>(block) * M * N;
    DataCopy(a, A[offset_a], Nd2NzParams(1, M, K, 0, K, M, 1, 0));
    DataCopy(b, B[offset_b], Nd2NzParams(1, N, K, 0, K, N, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);
    LoadData(a2, a, LoadData2dParams(0, K / 16, 1, 0, 0, false, 0));
    LoadData(b2, b, LoadData2dParams(0, 32, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);
    Mmad(c, a2, b2, MmadParams(M, N, K, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);
    Muls(v, c, 2.0f, M * N);
    Add(v, v, c, M * N);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    for (int32_t nb = 0; nb < 4; ++nb) {
        auto ip = FixpipeParamsV220(M, N / 4, 1, N, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(Out[offset_o + nb * 16], c[nb * M * 16], ip);
    }
    SetFlag<HardEvent::FIX_M>(efm);
    WaitFlag<HardEvent::FIX_M>(efm);
}
