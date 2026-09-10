#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BV = 64, D = 128, TILE = BV * D;

extern "C" __global__ __aicore__ void kda_k2_init_kernel(
    GM_ADDR pH0, GM_ADDR pS32, GM_ADDR pS16, int32_t BH, int32_t NV) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t task = GetBlockIdx();
    if (task >= BH * NV) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> uf, ub;
    pipe.InitBuffer(uf, TILE * sizeof(float));
    pipe.InitBuffer(ub, TILE * sizeof(bfloat16_t));
    LocalTensor<float> s = uf.Get<float>();
    LocalTensor<bfloat16_t> s16 = ub.Get<bfloat16_t>();
    GlobalTensor<float> H0, S32;
    GlobalTensor<bfloat16_t> S16;
    H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
    S32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pS32));
    S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    if (pH0 == nullptr) {
        Duplicate(s, 0.0f, TILE);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopy(s, H0[static_cast<uint64_t>(task) * TILE], DataCopyParams(BV, 16, 0, 0));
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
    }
    Cast(s16, s, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(S32[static_cast<uint64_t>(task) * TILE], s, DataCopyParams(BV, 16, 0, 0));
    DataCopy(S16[static_cast<uint64_t>(task) * TILE], s16, DataCopyParams(BV, 8, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
