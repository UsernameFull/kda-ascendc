// kg [chunk, 16, 128] -> kg^T [chunk, 128, 16] (bf16), used by the Cube
// d4 path which expects the K-major operand layout.
//
// The transpose is done with the UB "vtranspose" unit: each 16x16 block is
// gathered with one strided copy, transposed, and written back as one
// contiguous 16-row store.
//
// The 16x16 block is small, so a block's fixed cost dominates this kernel:
// each block now transposes KG_NCHUNK chunks and pays it once.
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t M = 16, D = 128, TILE = M * D;
constexpr int32_t NB = D / M;  // 16x16 blocks along the row
#ifndef KDA_KGT_NCHUNK
#define KDA_KGT_NCHUNK 8
#endif
constexpr int32_t NC = KDA_KGT_NCHUNK;

extern "C" __global__ __aicore__ void kda_kg_transpose(
    GM_ADDR pKg, GM_ADDR pKgT, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> uin, uout, usc;
    pipe.InitBuffer(uin, NC * TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(uout, NC * TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(usc, NC * NB * M * M * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> in = uin.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> out = uout.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> sc = usc.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Kg, KgT;
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    KgT.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(in[ch * TILE], Kg[static_cast<uint64_t>(c0 + ch) * TILE], DataCopyParams(M, 8, 0, 0));
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    for (int32_t ch = 0; ch < nch; ++ch) {
        // Column block b of the chunk, rows 0..15: one strided copy per block.
        for (int32_t b = 0; b < NB; ++b) {
            DataCopy(sc[ch * NB * M * M + b * M * M], in[ch * TILE + b * M], DataCopyParams(M, 1, 7, 0));
        }
    }
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    for (int32_t ch = 0; ch < nch; ++ch) {
        for (int32_t b = 0; b < NB; ++b) {
            AscendC::Transpose(out[ch * TILE + b * M * M], sc[ch * NB * M * M + b * M * M]);
        }
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    for (int32_t ch = 0; ch < nch; ++ch) {
        DataCopy(KgT[static_cast<uint64_t>(c0 + ch) * TILE], out[ch * TILE], DataCopyParams(D, 1, 0, 0));
    }
    PipeBarrier<PIPE_ALL>();
}
