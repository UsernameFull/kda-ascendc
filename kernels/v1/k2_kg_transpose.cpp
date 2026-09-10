// kg [chunk, 16, 128] -> kg^T [chunk, 128, 16] (bf16), used by the Cube
// d4 path which expects the K-major operand layout.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, D = 128, TILE = M * D;

extern "C" __global__ __aicore__ void kda_kg_transpose(
    GM_ADDR pKg, GM_ADDR pKgT, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c = GetBlockIdx();
    if (c >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> uin, uout;
    pipe.InitBuffer(uin, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(uout, TILE * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> in = uin.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> out = uout.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Kg, KgT;
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    KgT.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    DataCopy(in, Kg[static_cast<uint64_t>(c) * TILE], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t d = 0; d < D; ++d) {
            out.SetValue(d * M + i, in.GetValue(i * D + d));
        }
    }
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(KgT[static_cast<uint64_t>(c) * TILE], out, DataCopyParams(D, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
