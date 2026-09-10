#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t N_ELEMS = 1024;

extern "C" __global__ __aicore__ void hello_copy(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace; (void)tiling;
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
    inQue.EnQue(src);
    src = inQue.DeQue<float>();

    LocalTensor<float> dst = outQue.AllocTensor<float>();
    Muls(dst, src, 2.0f, N_ELEMS);
    outQue.EnQue(dst);
    dst = outQue.DeQue<float>();

    DataCopy(gy, dst, N_ELEMS);
    outQue.FreeTensor(dst);
    inQue.FreeTensor(src);
}
