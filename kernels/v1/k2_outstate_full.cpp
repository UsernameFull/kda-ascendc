// Output + per-chunk state update for one (b, h, v-tile) of one chunk.
// Output + per-chunk state update for one (b, h, v-tile) of one chunk.
// d4 is addressed per (b, h) here (the "d4_full" reuse layout).
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, D = 128, BV = 64, TILE = M * BV;

extern "C" __global__ __aicore__ void kda_k2_outstate_full_kernel(
    GM_ADDR pd2, GM_ADDR pd3, GM_ADDR pd4, GM_ADDR pS32, GM_ADDR pS16,
    GM_ADDR pDecay, GM_ADDR pOut,
    int32_t BH, int32_t NT, int32_t NV, int32_t chunk, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t task = GetBlockIdx();
    if (task >= BH * NV) return;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;
    const int32_t c = bh * NT + chunk;
    (void)c;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> ud2, ud3, ud4, us, udec, uo, uof, us16;
    pipe.InitBuffer(ud2, TILE * 4);
    pipe.InitBuffer(ud3, TILE * 4);
    pipe.InitBuffer(ud4, BV * D * 4);
    pipe.InitBuffer(us, BV * D * 4);
    pipe.InitBuffer(udec, D * 4);
    pipe.InitBuffer(uo, TILE * 2);
    pipe.InitBuffer(uof, TILE * 4);
    pipe.InitBuffer(us16, BV * D * 2);
    LocalTensor<float> d2 = ud2.Get<float>();
    LocalTensor<float> d3 = ud3.Get<float>();
    LocalTensor<float> d4 = ud4.Get<float>();
    LocalTensor<float> s = us.Get<float>();
    LocalTensor<float> dec = udec.Get<float>();
    LocalTensor<float> of = uof.Get<float>();
    LocalTensor<bfloat16_t> ob = uo.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> s16 = us16.Get<bfloat16_t>();
    GlobalTensor<float> D2, D3, D4, S32, Decay;
    GlobalTensor<bfloat16_t> S16, Out;
    D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd2));
    D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd3));
    D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pd4));
    S32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pS32));
    S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
    Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
    const uint64_t t0 = (static_cast<uint64_t>(task) * NT + chunk) * TILE;
    const uint64_t s0 = static_cast<uint64_t>(task) * BV * D;
    const uint64_t c0 = static_cast<uint64_t>(c) * D;
    const uint64_t d4base = static_cast<uint64_t>(bh) * D * D + static_cast<uint64_t>(iv) * BV * D;
    DataCopy(d2, D2[t0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d3, D3[t0], DataCopyParams(M, 8, 0, 0));
    DataCopy(d4, D4[d4base], DataCopyParams(BV, 16, 0, 0));
    DataCopy(s, S32[s0], DataCopyParams(BV, 16, 0, 0));
    DataCopy(dec, Decay[c0], DataCopyParams(1, 16, 0, 0));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Muls(of, d2, scale, TILE);
    Add(of, of, d3, TILE);
    Cast(ob, of, RoundMode::CAST_RINT, TILE);
    PipeBarrier<PIPE_V>();
    // s[v][k] *= dec[k] for all BV rows in two strided Mul repeats (the decay
    // vector is reused with a zero source-repeat stride).
    Mul(s, s, dec, 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Mul(s[64], s[64], dec[64], 64, BV, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Add(s, s, d4, BV * D);
    Cast(s16, s, RoundMode::CAST_RINT, BV * D);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Out[t0], ob, DataCopyParams(M, 4, 0, 0));
    DataCopy(S32[s0], s, DataCopyParams(BV, 16, 0, 0));
    DataCopy(S16[s0], s16, DataCopyParams(BV, 8, 0, 0));
    PipeBarrier<PIPE_ALL>();
}
