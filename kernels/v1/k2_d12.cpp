// Standalone per-(b,h,tile) W/Qg @ S16^T kernel for the "separated" K2 path.
// Same math (and same Cube instruction sequence) as k2_d12_cube.cpp, so the
// "separated" and "cube_separated" modes agree to the last bit.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BV = 64, D = 128, M = 16, N = 64;

extern "C" __global__ __aicore__ void kda_k2_d12_kernel(
    GM_ADDR pW, GM_ADDR pQg, GM_ADDR pS16, GM_ADDR pd1, GM_ADDR pd2,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t task = GetBlockIdx();
    const int32_t tasks = BH * NV;
    if (task >= tasks) return;
    const int32_t bh = task / NV;
    const int32_t c = bh * NT + chunk;
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
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(TPosition::A2, 0, M * D * sizeof(bfloat16_t));
    LocalTensor<uint8_t> l0bU8(TPosition::B2, 0, BV * D * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> W, Qg, S16;
    GlobalTensor<float> D1, D2;
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd1));
    D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd2));
    const uint64_t a0 = static_cast<uint64_t>(c) * M * D;
    const uint64_t s0 = static_cast<uint64_t>(task) * BV * D;
    const uint64_t o0 = static_cast<uint64_t>(task * NT + chunk) * M * BV;
    LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
    DataCopy(la, W[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
    DataCopy(lb, S16[s0], Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21); WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(la); l1BQue.EnQue(lb);
    la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
    LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    LoadData(l0b, lb, LoadData2dParams(0, 32, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m); WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(M, N, D, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix); WaitFlag<HardEvent::M_FIX>(evmfix);
    for (int32_t nb = 0; nb < 4; ++nb) {
        auto ip = FixpipeParamsV220(M, N / 4, 1, N, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0 + nb * M], l0cf[nb * M * (N / 4)], ip);
    }
    SetFlag<HardEvent::FIX_M>(evfixm); WaitFlag<HardEvent::FIX_M>(evfixm);
    l1AQue.FreeTensor(la);
    PipeBarrier<PIPE_ALL>();
    la = l1AQue.AllocTensor<bfloat16_t>();
    DataCopy(la, Qg[a0], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21); WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(la); la = l1AQue.DeQue<bfloat16_t>();
    LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m); WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(M, N, D, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix); WaitFlag<HardEvent::M_FIX>(evmfix);
    for (int32_t nb = 0; nb < 4; ++nb) {
        auto ip = FixpipeParamsV220(M, N / 4, 1, N, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0 + nb * M], l0cf[nb * M * (N / 4)], ip);
    }
    SetFlag<HardEvent::FIX_M>(evfixm); WaitFlag<HardEvent::FIX_M>(evfixm);
    l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb); l0CQue.FreeTensor(l0cf);
}
