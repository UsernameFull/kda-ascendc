// v_new = u - w @ h  for one (b, h, v-tile) of one chunk.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, D = 128, BV = 64, TILE = M * BV;

extern "C" __global__ __aicore__ void kda_k2_vnew_kernel(
    GM_ADDR pU, GM_ADDR pD1, GM_ADDR pVnew, GM_ADDR pVnewT,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk0, int32_t nchunk) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t tasks = BH * NV;
    const int32_t blk = GetBlockIdx();
    if (blk >= tasks * nchunk) return;
    const int32_t task = blk % tasks;
    const int32_t chunk = chunk0 + blk / tasks;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;
    const int32_t c = bh * NT + chunk;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev2 = pipe.AllocEventID<HardEvent::V_MTE2>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> uu, ud, uv, ut, uf, usc;
    pipe.InitBuffer(uu, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(ud, TILE * sizeof(float));
    pipe.InitBuffer(uv, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(ut, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(uf, M * D * sizeof(float));
    // Staging for the 16x16 block transpose that builds v_new^T.
    pipe.InitBuffer(usc, (BV / M) * M * M * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> ub = uu.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vb = uv.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> vt = ut.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> sc = usc.Get<bfloat16_t>();
    LocalTensor<float> d = ud.Get<float>();
    LocalTensor<float> vf = uf.Get<float>();
    GlobalTensor<bfloat16_t> U, V, Vt;
    GlobalTensor<float> D1;
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
    V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnew));
    Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
    D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
    const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
    const uint64_t out0 = static_cast<uint64_t>(task * NT + chunk) * TILE;
    DataCopy(ub, U[u0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d, D1[out0], DataCopyParams(M, 8, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(vf, ub, RoundMode::CAST_NONE, M * D);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < M; ++i) {
        Sub(vf[i * BV], vf[i * D + iv * BV], d[i * BV], BV);
    }
    Cast(vb, vf, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();
    // v_new^T via 16x16 block transpose on the UB vtranspose unit: gather the
    // four 16-column blocks of v_new into contiguous 16x16 tiles, transpose,
    // then store all 64 rows at once.
    SetFlag<HardEvent::V_MTE2>(ev2);
    WaitFlag<HardEvent::V_MTE2>(ev2);
    for (int32_t bl = 0; bl < BV / M; ++bl) {
        for (int32_t r = 0; r < M; ++r) {
            DataCopy(sc[bl * M * M + r * M], vb[r * BV + bl * M], DataCopyParams(1, 1, 0, 0));
        }
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    for (int32_t bl = 0; bl < BV / M; ++bl) {
        AscendC::Transpose(vt[bl * M * M], sc[bl * M * M]);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(V[out0], vb, DataCopyParams(M, 4, 0, 0));
    DataCopy(Vt[out0], vt, DataCopyParams(BV, 1, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
