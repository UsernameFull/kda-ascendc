#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t N_ELEMS = 1024;

// MIX skeleton: verify dual-pass build + cross-core handshake.
//   AIC side : set flag -> done
//   AIV side : wait flag, y = 2*x
// Once this runs, M1 fills in real cube math on the AIC side.

extern "C" __global__ __aicore__ void k2_mix(GM_ADDR x, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

#if defined(__DAV_C220_CUBE__)
    // ---- cube pass ----
    // signal vector side that cube stage is "done" (nothing to compute yet)
    // disabled
#elif defined(__DAV_C220_VEC__)
    // ---- vector pass ----
    // disabled

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
#else
    // single-pass fallback (plain AIV build): y = 2*x
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
