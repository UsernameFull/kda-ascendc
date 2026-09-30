// Minimal bisect for the fixpipe shapes k1_solve_assemble4 uses.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;

extern "C" __global__ __aicore__ void kda_fixpipe16_probe(
    GM_ADDR pOut, int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    TBuf<TPosition::B1> buf;
    pipe.InitBuffer(buf, 8192);
    LocalTensor<float> cf(TPosition::CO1, 0, 4096);
    LocalTensor<bfloat16_t> lb = buf.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
    const int32_t b = GetBlockIdx();
    auto ip = FixpipeParamsV220(16, 16, 16, PC, false);
    ip.quantPre = QuantMode_t::F322BF16;
    ip.unitFlag = 0;
    if (mode == 0) {          // 16x16 RM at a 1 KB-aligned base
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096], cf[0], ip);
    } else if (mode == 1) {   // 16x16 RM at base + 1040 elements (2080 B)
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096 + 1040], cf[0], ip);
    } else if (mode == 2) {   // NZ -> L1
        auto ipnz = FixpipeParamsV220(16, 16, 16, 16, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lb[0], cf[0], ipnz);
    } else if (mode == 3) {   // the trio: ndNum = 3, srcNd = 1 KB, dstNd = 1040
        auto ipt = FixpipeParamsV220(16, 16, 16, PC, false,
                                     QuantMode_t::F322BF16, 0, 3, 1, 1040, 0);
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096], cf[0], ipt);
    } else if (mode == 4) {   // control: 32 rows x 32 cols at a 4 KB base
        auto ip32 = FixpipeParamsV220(32, 32, 32, PC, false);
        ip32.quantPre = QuantMode_t::F322BF16;
        ip32.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096 + 32 * PC], cf[0], ip32);
    } else if (mode == 5) {   // 16x16 RM at base + 2064 elements (4128 B)
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096 + 2064], cf[0], ip);
    } else if (mode == 6) {   // the pair: ndNum = 2 at base 2048 elements
        auto ipp = FixpipeParamsV220(16, 16, 16, PC, false,
                                     QuantMode_t::F322BF16, 0, 2, 1, 1040, 0);
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(out[b * 4096 + 32 * PC], cf[0], ipp);
    }
    PipeBarrier<PIPE_ALL>();
}
