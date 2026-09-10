#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t N_ELEMS = 1024;
constexpr uint16_t USER_FLAG = 8;   // 避开 4 个 Matmul 内部的 0~7

// MIX sync sanity test (user hypothesis #3):
//   AIC : write debug[0]=0xA1C0, then SetFlag<mode2,MTE3>(8)
//   AIV : WaitFlag<mode2,MTE2>(8), write debug[1]=0xA1F0, y = 3*x
//   fallback (no split): y = 2*x   -> tells us if __DAV_C220_* split failed
extern "C" __global__ __aicore__ void k2_mix(GM_ADDR x, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR debug) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

#if defined(__DAV_C220_CUBE__)
    // ---- cube pass ----
    auto d0 = reinterpret_cast<__gm__ uint32_t *>(debug);
    d0[0] = 0xA1C0u;
    CrossCoreSetFlag<2, PIPE_MTE3>(USER_FLAG);
#elif defined(__DAV_C220_VEC__)
    // ---- vector pass ----
    CrossCoreWaitFlag<2, PIPE_MTE2>(USER_FLAG);
    auto d1 = reinterpret_cast<__gm__ uint32_t *>(debug);
    d1[1] = 0xA1F0u;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, N_ELEMS * sizeof(float));
    pipe.InitBuffer(outQue, 1, N_ELEMS * sizeof(float));
    GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer((__gm__ float *)x);
    gy.SetGlobalBuffer((__gm__ float *)y);
    LocalTensor<float> src = inQue.AllocTensor<float>();
    DataCopy(src, gx, N_ELEMS);
    inQue.EnQue(src); src = inQue.DeQue<float>();
    LocalTensor<float> dst = outQue.AllocTensor<float>();
    Muls(dst, src, 3.0f, N_ELEMS);
    outQue.EnQue(dst); dst = outQue.DeQue<float>();
    DataCopy(gy, dst, N_ELEMS);
    outQue.FreeTensor(dst); inQue.FreeTensor(src);
#else
    // ---- fallback (split macros not active) ----
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, N_ELEMS * sizeof(float));
    pipe.InitBuffer(outQue, 1, N_ELEMS * sizeof(float));
    GlobalTensor<float> gx, gy;
    gx.SetGlobalBuffer((__gm__ float *)x);
    gy.SetGlobalBuffer((__gm__ float *)y);
    LocalTensor<float> src = inQue.AllocTensor<float>();
    DataCopy(src, gx, N_ELEMS);
    inQue.EnQue(src); src = inQue.DeQue<float>();
    LocalTensor<float> dst = outQue.AllocTensor<float>();
    Muls(dst, src, 2.0f, N_ELEMS);
    outQue.EnQue(dst); dst = outQue.DeQue<float>();
    DataCopy(gy, dst, N_ELEMS);
    outQue.FreeTensor(dst); inQue.FreeTensor(src);
#endif
}
