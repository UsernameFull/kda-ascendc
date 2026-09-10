#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M = 16;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t TILE = M * BV;

extern "C" __global__ __aicore__ void kda_k2_triton_aiv(
    GM_ADDR pW, GM_ADDR pQg, GM_ADDR pU, GM_ADDR pAqk, GM_ADDR pKg,
    GM_ADDR pDecay, GM_ADDR pH0, GM_ADDR pOut, GM_ADDR pHt,
    int32_t BH, int32_t NT, int32_t NV, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t task = GetBlockIdx();
    const int32_t tasks = BH * NV;
    if (task >= tasks) return;
    const int32_t bh = task / NV;
    const int32_t iv = task - bh * NV;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> state_buf, w_buf, q_buf, u_buf, k_buf, a_buf,
        decay_buf, wf_buf, qf_buf, uf_buf, kf_buf, af_buf, d1_buf, d2_buf,
        v_buf, d3_buf, d4_buf, out_buf, out_bf_buf;
    pipe.InitBuffer(state_buf, BV * D * sizeof(float));
    pipe.InitBuffer(w_buf, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(q_buf, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(u_buf, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(k_buf, M * D * sizeof(bfloat16_t));
    pipe.InitBuffer(a_buf, M * M * sizeof(bfloat16_t));
    pipe.InitBuffer(decay_buf, D * sizeof(float));
    pipe.InitBuffer(wf_buf, M * D * sizeof(float));
    pipe.InitBuffer(qf_buf, M * D * sizeof(float));
    pipe.InitBuffer(uf_buf, M * D * sizeof(float));
    pipe.InitBuffer(kf_buf, M * D * sizeof(float));
    pipe.InitBuffer(af_buf, M * M * sizeof(float));
    pipe.InitBuffer(d1_buf, TILE * sizeof(float));
    pipe.InitBuffer(d2_buf, TILE * sizeof(float));
    pipe.InitBuffer(v_buf, TILE * sizeof(float));
    pipe.InitBuffer(d3_buf, TILE * sizeof(float));
    pipe.InitBuffer(d4_buf, BV * D * sizeof(float));
    pipe.InitBuffer(out_buf, TILE * sizeof(float));
    pipe.InitBuffer(out_bf_buf, TILE * sizeof(bfloat16_t));

    LocalTensor<float> state = state_buf.Get<float>();
    LocalTensor<float> wf = wf_buf.Get<float>(), qf = qf_buf.Get<float>();
    LocalTensor<float> uf = uf_buf.Get<float>(), kf = kf_buf.Get<float>();
    LocalTensor<float> af = af_buf.Get<float>();
    LocalTensor<float> decay = decay_buf.Get<float>();
    LocalTensor<float> d1 = d1_buf.Get<float>(), d2 = d2_buf.Get<float>();
    LocalTensor<float> vnew = v_buf.Get<float>(), d3 = d3_buf.Get<float>();
    LocalTensor<float> d4 = d4_buf.Get<float>(), output = out_buf.Get<float>();
    LocalTensor<bfloat16_t> wb = w_buf.Get<bfloat16_t>(), qb = q_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> ub = u_buf.Get<bfloat16_t>(), kb = k_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> ab = a_buf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> output_bf = out_bf_buf.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> W, Qg, U, Kg, Aqk, Out;
    GlobalTensor<float> Decay, H0, Ht;
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
    Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
    H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
    Ht.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pHt));

    const uint64_t state_offset = static_cast<uint64_t>(bh) * NV * BV * D +
                                  static_cast<uint64_t>(iv) * BV * D;
    if (pH0 == nullptr) {
        Duplicate(state, 0.0f, BV * D);
    } else {
        DataCopy(state, H0[state_offset], DataCopyParams(BV, 16, 0, 0));
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
    }

    for (int32_t chunk = 0; chunk < NT; ++chunk) {
        const int32_t c = bh * NT + chunk;
        const uint64_t x_offset = static_cast<uint64_t>(c) * M * D;
        const uint64_t a_offset = static_cast<uint64_t>(c) * M * M;
        const uint64_t out_offset = (static_cast<uint64_t>(task) * NT + chunk) * TILE;
        DataCopy(wb, W[x_offset], DataCopyParams(M, 8, 0, 0));
        DataCopy(qb, Qg[x_offset], DataCopyParams(M, 8, 0, 0));
        DataCopy(ub, U[x_offset], DataCopyParams(M, 8, 0, 0));
        DataCopy(kb, Kg[x_offset], DataCopyParams(M, 8, 0, 0));
        DataCopy(ab, Aqk[a_offset], DataCopyParams(M, 1, 0, 0));
        DataCopy(decay, Decay[static_cast<uint64_t>(c) * D], DataCopyParams(1, 16, 0, 0));
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        Cast(wf, wb, RoundMode::CAST_NONE, M * D);
        Cast(qf, qb, RoundMode::CAST_NONE, M * D);
        Cast(uf, ub, RoundMode::CAST_NONE, M * D);
        Cast(kf, kb, RoundMode::CAST_NONE, M * D);
        Cast(af, ab, RoundMode::CAST_NONE, M * M);
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < BV; ++j) {
                float d1_value = 0.0f, d2_value = 0.0f;
                for (int32_t k = 0; k < D; ++k) {
                    const float s = state.GetValue(j * D + k);
                    d1_value += wf.GetValue(i * D + k) * s;
                    d2_value += qf.GetValue(i * D + k) * s;
                }
                d1.SetValue(i * BV + j, d1_value);
                d2.SetValue(i * BV + j, d2_value);
                vnew.SetValue(i * BV + j, uf.GetValue(i * D + iv * BV + j) - d1_value);
            }
        }
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < BV; ++j) {
                float value = 0.0f;
                for (int32_t r = 0; r < M; ++r) value += af.GetValue(i * M + r) * vnew.GetValue(r * BV + j);
                d3.SetValue(i * BV + j, value);
            }
        }
        for (int32_t j = 0; j < BV; ++j) {
            for (int32_t k = 0; k < D; ++k) {
                float value = 0.0f;
                for (int32_t i = 0; i < M; ++i) value += vnew.GetValue(i * BV + j) * kf.GetValue(i * D + k);
                d4.SetValue(j * D + k, value);
            }
        }
        Muls(output, d2, scale, TILE);
        Add(output, output, d3, TILE);
        Cast(output_bf, output, RoundMode::CAST_RINT, TILE);
        for (int32_t j = 0; j < BV; ++j) {
            Mul(state[j * D], state[j * D], decay, D);
        }
        Add(state, state, d4, BV * D);
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Out[out_offset], output_bf, DataCopyParams(M, 4, 0, 0));
    }
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    if (pHt != nullptr) DataCopy(Ht[state_offset], state, DataCopyParams(BV, 16, 0, 0));
}
