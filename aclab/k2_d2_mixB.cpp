#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
extern "C" __global__ __aicore__ void k2_d2_mixB(GM_ADDR pqg, GM_ADDR ph, GM_ADDR pd2, GM_ADDR ws, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C220_CUBE__)
    TPipe pipe;
    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, 16 * 128 * 2);
    pipe.InitBuffer(l1BQue, 1, 128 * 128 * 2);
    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, 16 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();
    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 8 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 64 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gA, gB;
    GlobalTensor<float> gC;
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);

    { // block 0
        uint64_t base = 0 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 1
        uint64_t base = 1 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 2
        uint64_t base = 2 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 3
        uint64_t base = 3 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 4
        uint64_t base = 4 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 5
        uint64_t base = 5 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 6
        uint64_t base = 6 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }

    { // block 7
        uint64_t base = 7 * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg + base);
        gC.SetGlobalBuffer((__gm__ float *)pd2 + base);
        LocalTensor<bfloat16_t> la = l1AQue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> lb = l1BQue.AllocTensor<bfloat16_t>();
        DataCopy(la, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
        DataCopy(lb, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(la); l1BQue.EnQue(lb);
        la = l1AQue.DeQue<bfloat16_t>(); lb = l1BQue.DeQue<bfloat16_t>();
        LoadData(l0a, la, LoadData2dParams(0, 8, 1, 0, 0, false, 0));
        LoadData(l0b, lb, LoadData2dParams(0, 64, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0cf[b * 256], ip);
        }
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }
    l0CQue.FreeTensor(l0cf);
#elif defined(__DAV_C220_VEC__)

    { // AIV block 0 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 1 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 2 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 3 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 4 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 5 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 6 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }

    { // AIV block 7 (no-op, just flag response)
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }
#endif
}