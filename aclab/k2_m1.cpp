#include "kernel_operator.h"
#include "hardware.h"
#include "layout.h"
#include "mem.h"
#include "common_func.h"

using namespace AscendC;

constexpr uint32_t M16 = 16, K128 = 128, N128 = 128;
constexpr uint16_t USER_FLAG = 8;
constexpr uint32_t LS_A = 4096;              // 16x128 bf16 NZ = 8 fractals x 512B
constexpr uint32_t LS_B = 32768;             // 128x128 bf16 NZ = 64 fractals x 512B

static constexpr uint32_t BLOCK_16 = 16;     // elems per 32B block (bf16)
static constexpr uint32_t FRACTAL = 256;     // elems per fractal (16x16 bf16 = 512B)

// -------- CUBE side: two matmuls, results to GM --------
#if defined(__DAV_C220_CUBE__)
__aicore__ inline void cube_dot(GlobalTensor<bfloat16_t>& gmA, GlobalTensor<bfloat16_t>& gmB,
                                GlobalTensor<bfloat16_t>& gmC,
                                LocalTensor<bfloat16_t>& l1a, LocalTensor<bfloat16_t>& l1b,
                                LocalTensor<bfloat16_t>& l0a, LocalTensor<bfloat16_t>& l0b,
                                LocalTensor<float>& l0c) {
    // A: [16,128] ND -> L1 NZ
    DataCopy(l1a, gmA, Nd2NzParams(1, M16, K128, 0, K128, M16, 1, 0));
    // B: [128(K),128(N)] ND -> L1 NZ  (B stored k-major; we feed h^T layout)
    DataCopy(l1b, gmB, Nd2NzParams(1, K128, N128, 0, N128, K128, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);

    // L1(ZN) -> L0A(ZZ), no transpose: m=16,k=128
    for (uint32_t i = 0; i < M16 / BLOCK_16; ++i) {
        LoadData(l0a[i * 8 * FRACTAL], l1a[i * 1 * FRACTAL],
                 LoadData2dParams(0, K128 / BLOCK_16, 1, 0, 0, false, 0));
    }
    // L1(ZN) -> L0B(NZ), no transpose: n=128,k=128
    for (uint32_t i = 0; i < K128 / BLOCK_16; ++i) {
        LoadData(l0b[i * 8 * FRACTAL], l1b[i * 1 * FRACTAL],
                 LoadData2dParams(0, N128 / BLOCK_16, 8, 0, 0, true, 0));
    }
    SetFlag<HardEvent::MTE1_M>(1);
    WaitFlag<HardEvent::MTE1_M>(1);

    // C = A @ B
    Mmad(l0c, l0a, l0b, MmadParams(M16, N128, K128, 0, false, true));
    PipeBarrier<PIPE_M>();

    // L0C -> GM bf16 ND
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
    FixpipeParamsV220 fp(N128, M16, M16, N128, false);
    fp.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gmC, l0c, fp);
    SetFlag<HardEvent::FIX_M>(0);
    PipeBarrier<PIPE_MTE3>();
}
#endif

extern "C" __global__ __aicore__ void k2_m1(GM_ADDR w, GM_ADDR hT, GM_ADDR qg,
                                            GM_ADDR c1, GM_ADDR c2, GM_ADDR out,
                                            GM_ADDR debug) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

#if defined(__DAV_C220_CUBE__)
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto l1a = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(0);
    auto l1b = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(LS_A);
    auto l0a = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(0);
    auto l0b = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(0);
    auto l0c = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    GlobalTensor<bfloat16_t> gw, gh, gq, gc1, gc2;
    gw.SetGlobalBuffer((__gm__ bfloat16_t *)w);
    gh.SetGlobalBuffer((__gm__ bfloat16_t *)hT);
    gq.SetGlobalBuffer((__gm__ bfloat16_t *)qg);
    gc1.SetGlobalBuffer((__gm__ bfloat16_t *)c1);
    gc2.SetGlobalBuffer((__gm__ bfloat16_t *)c2);

    cube_dot(gw, gh, gc1, l1a, l1b, l0a, l0b, l0c);
    cube_dot(gq, gh, gc2, l1a, l1b, l0a, l0b, l0c);

    auto d0 = reinterpret_cast<__gm__ uint32_t *>(debug);
    d0[0] = 0xA1C0u;
    CrossCoreSetFlag<2, PIPE_MTE3>(USER_FLAG);

#elif defined(__DAV_C220_VEC__)
    CrossCoreWaitFlag<2, PIPE_MTE2>(USER_FLAG);
    auto d1 = reinterpret_cast<__gm__ uint32_t *>(debug);
    d1[1] = 0xA1F0u;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, M16 * K128 * sizeof(bfloat16_t));
    pipe.InitBuffer(outQue, 1, M16 * K128 * sizeof(bfloat16_t));

    GlobalTensor<bfloat16_t> gc1, go;
    gc1.SetGlobalBuffer((__gm__ bfloat16_t *)c1);
    go.SetGlobalBuffer((__gm__ bfloat16_t *)out);

    LocalTensor<bfloat16_t> s1 = inQue.AllocTensor<bfloat16_t>();
    DataCopy(s1, gc1, M16 * K128);
    inQue.EnQue(s1); s1 = inQue.DeQue<bfloat16_t>();

    LocalTensor<bfloat16_t> dst = outQue.AllocTensor<bfloat16_t>();
    // echo c1 -> out (no bf16 Add support; sum tested via python instead)
    DataCopy(dst, s1, M16 * K128);
    outQue.EnQue(dst); dst = outQue.DeQue<bfloat16_t>();
    DataCopy(go, dst, M16 * K128);
    outQue.FreeTensor(dst); inQue.FreeTensor(s1);
#else
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, M16 * K128 * sizeof(bfloat16_t));
    pipe.InitBuffer(outQue, 1, M16 * K128 * sizeof(bfloat16_t));
    GlobalTensor<bfloat16_t> gc1, go;
    gc1.SetGlobalBuffer((__gm__ bfloat16_t *)c1);
    go.SetGlobalBuffer((__gm__ bfloat16_t *)out);
    LocalTensor<bfloat16_t> s1 = inQue.AllocTensor<bfloat16_t>();
    DataCopy(s1, gc1, M16 * K128);
    inQue.EnQue(s1); s1 = inQue.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> dst = outQue.AllocTensor<bfloat16_t>();
    DataCopy(dst, s1, M16 * K128);
    outQue.EnQue(dst); dst = outQue.DeQue<bfloat16_t>();
    DataCopy(go, dst, M16 * K128);
    outQue.FreeTensor(dst); inQue.FreeTensor(s1);
#endif
}
