#include "kernel_operator.h"
#include "hardware.h"
#include "layout.h"
#include "mem.h"
#include "common_func.h"

using namespace AscendC;
constexpr uint32_t M16 = 16, K16 = 16, N16 = 16;

extern "C" __global__ __aicore__ void k2_min(GM_ADDR ffts, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                             GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_C220_CUBE__)
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto l1a = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(0);
    auto l1b = buf.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(512);
    auto l0a = buf.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(0);
    auto l0b = buf.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(0);
    auto l0c = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    GlobalTensor<bfloat16_t> ga, gb, gc;
    ga.SetGlobalBuffer((__gm__ bfloat16_t *)a);
    gb.SetGlobalBuffer((__gm__ bfloat16_t *)b);
    gc.SetGlobalBuffer((__gm__ bfloat16_t *)c);

    DataCopy(l1a, ga, Nd2NzParams(1, M16, K16, 0, K16, M16, 1, 0));
    DataCopy(l1b, gb, Nd2NzParams(1, K16, N16, 0, N16, K16, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

    LoadData(l0a, l1a, LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadData(l0b, l1b, LoadData2dParams(0, 1, 1, 0, 0, true, 0));
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

    Mmad(l0c, l0a, l0b, MmadParams(M16, N16, K16, 0, false, true));
    PipeBarrier<PIPE_M>();
    PipeBarrier<PIPE_FIX>();
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

    FixpipeParamsV220 fp(N16, M16, M16, N16, false);
    fp.quantPre = QuantMode_t::F322BF16;
    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gc, l0c, fp);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
#endif
}
