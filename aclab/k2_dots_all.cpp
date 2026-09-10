#include "kernel_operator.h"
using namespace AscendC;

// M1.3: all four K2 dots in one AIC kernel, results to GM (fp32).
//   d1 = w  @ h^T       [16,128]  (A=w, B=h raw -> Nd2Nz implicit h^T)
//   d2 = qg @ h^T       [16,128]
//   d3 = Aqk @ v_new    [16,128]  (Aqk[16,16] @ v_new[16,128], k=16)
//   d4 = v_new^T @ kg   [128,128] (m-tiled 8x mmad(16,128,16))
// Inputs (bf16): pw, pqg, ph[128,128], pAqk[16,16], pvnew[16,128], pkg[16,128]
// Outputs (fp32): pc1[16,128], pc2[16,128], pc3[16,128], pc4[128,128]
extern "C" __global__ __aicore__ void k2_dots_all_kernel(
    GM_ADDR pw, GM_ADDR pqg, GM_ADDR ph, GM_ADDR pAqk,
    GM_ADDR pvnew, GM_ADDR pvnewT, GM_ADDR pkg,
    GM_ADDR pc1, GM_ADDR pc2, GM_ADDR pc3, GM_ADDR pc4, int32_t mode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;

    TEventID ev21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID ev1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID evmfix = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID evfixm = pipe.AllocEventID<HardEvent::FIX_M>();

    TQue<QuePosition::B1, 1> l1AQue, l1BQue;
    pipe.InitBuffer(l1AQue, 1, 16 * 128 * 2);
    pipe.InitBuffer(l1BQue, 1, 128 * 128 * 2);
    LocalTensor<bfloat16_t> l1a = l1AQue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> l1b = l1BQue.AllocTensor<bfloat16_t>();

    TQue<QuePosition::CO1, 1> l0CQue;
    pipe.InitBuffer(l0CQue, 1, 128 * 128 * 4);
    LocalTensor<float> l0cf = l0CQue.AllocTensor<float>();

    LocalTensor<uint8_t> l0aU8(AscendC::TPosition::A2, 0, 8 * 512);
    LocalTensor<uint8_t> l0bU8(AscendC::TPosition::B2, 0, 64 * 512);
    LocalTensor<bfloat16_t> l0a = l0aU8.ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> l0b = l0bU8.ReinterpretCast<bfloat16_t>();

    GlobalTensor<bfloat16_t> gA, gB, gAqk, gV, gKg;
    GlobalTensor<float> gC1, gC2, gC3, gC4;
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pw);
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);
    gAqk.SetGlobalBuffer((__gm__ bfloat16_t *)pAqk);
    gV.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gC1.SetGlobalBuffer((__gm__ float *)pc1);
    gC2.SetGlobalBuffer((__gm__ float *)pc2);
    gC3.SetGlobalBuffer((__gm__ float *)pc3);
    gC4.SetGlobalBuffer((__gm__ float *)pc4);

    auto fixpipe16 = [&](GlobalTensor<float> &gC, LocalTensor<float> &l0c, uint32_t dstStride) __aicore__ {
        auto ip = FixpipeParamsV220(16, 16, 1, dstStride, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gC[0], l0c, ip);
    };
    auto fixpipe128 = [&](GlobalTensor<float> &gC, LocalTensor<float> &l0c) __aicore__ {
        for (int b = 0; b < 8; b++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC[b * 16], l0c[b * 256], ip);
        }
    };

    // d1 = w @ h^T
    DataCopy(l1a, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    DataCopy(l1b, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(l1a);
    l1BQue.EnQue(l1b);
    LoadData(l0a, l1AQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    LoadData(l0b, l1BQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 64, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    fixpipe128(gC1, l0cf);
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    AscendC::PipeBarrier<PIPE_ALL>();

    // d2 = qg @ h^T (reload A and B independently)
    if (mode >= 1) {
    gA.SetGlobalBuffer((__gm__ bfloat16_t *)pqg);
    DataCopy(l1a, gA, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    DataCopy(l1b, gB, Nd2NzParams(1, 128, 128, 0, 128, 128, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(l1a);
    l1BQue.EnQue(l1b);
    LoadData(l0a, l1AQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 8, 1, 0, 0, false, 0));
    LoadData(l0b, l1BQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 64, 1, 0, 0, false, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 128, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    fixpipe128(gC2, l0cf);
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    AscendC::PipeBarrier<PIPE_ALL>();
    }

    // d3 = Aqk[16,16] @ v_new[16,128] (k=16)
    if (mode >= 2) {
    DataCopy(l1a, gAqk, Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
    DataCopy(l1b, gV, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1AQue.EnQue(l1a);
    l1BQue.EnQue(l1b);
    LoadData(l0a, l1AQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 1, 1, 0, 0, false, 0));
    LoadData(l0b, l1BQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 8, 1, 0, 0, true, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    Mmad(l0cf, l0a, l0b, MmadParams(16, 128, 16, 0, false, true));
    SetFlag<HardEvent::M_FIX>(evmfix);
    WaitFlag<HardEvent::M_FIX>(evmfix);
    fixpipe128(gC3, l0cf);
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    AscendC::PipeBarrier<PIPE_ALL>();
    }

    // d4 = v_new^T @ kg -> [128,128] (A = v_new^T [128,16], m-tiled 8x)
    if (mode >= 3) {
    // B = kg [16,128] once
    DataCopy(l1b, gKg, Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0));
    SetFlag<HardEvent::MTE2_MTE1>(ev21);
    WaitFlag<HardEvent::MTE2_MTE1>(ev21);
    l1BQue.EnQue(l1b);
    LoadData(l0b, l1BQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 8, 1, 0, 0, true, 0));
    SetFlag<HardEvent::MTE1_M>(ev1m);
    WaitFlag<HardEvent::MTE1_M>(ev1m);
    // A = v_new^T: each m-block i reads v_new[i, :] as 16 rows -> A[i,16] = v_new[i,:]
    // v_new[16,128] row-major; A[m-block i, k] = v_new[i, k]?? No: A[128,16], A[m,k]=vnew[k?].
    // We need A = v_new^T so A[m, k] = v_new[k, m]. m-block i covers rows 16i..16i+15 of A
    // = columns 16i..16i+15 of v_new. So for m-block i, load v_new[:, 16i:16i+16] transposed.
    // kda_k2_m128 loads gA[i*256] as [16,16] contiguous blocks, i.e. A is [128,16] row-major
    // where A[i*16+j] = vnew[j*128 + i]? That's NOT a contiguous block of v_new.
    // So host must supply A = v_new^T contiguous [128,16]. We do that (pvnew holds v_new^T here
    // for d4 path, laid out as [128,16]).
    gV.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);  // v_new^T [128,16] contiguous
    for (int i = 0; i < 8; i++) {
        DataCopy(l1a, gV[i * 256], Nd2NzParams(1, 16, 16, 0, 16, 16, 1, 0));
        SetFlag<HardEvent::MTE2_MTE1>(ev21);
        WaitFlag<HardEvent::MTE2_MTE1>(ev21);
        l1AQue.EnQue(l1a);
        LoadData(l0a, l1AQue.DeQue<bfloat16_t>(), LoadData2dParams(0, 1, 1, 0, 0, false, 0));
        SetFlag<HardEvent::MTE1_M>(ev1m);
        WaitFlag<HardEvent::MTE1_M>(ev1m);
        Mmad(l0cf[i * 2048], l0a, l0b, MmadParams(16, 128, 16, 0, false, true));
        SetFlag<HardEvent::M_FIX>(evmfix);
        WaitFlag<HardEvent::M_FIX>(evmfix);
    }
    AscendC::PipeBarrier<PIPE_M>();
    for (int mb = 0; mb < 8; mb++) {
        for (int nb = 0; nb < 8; nb++) {
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant;
            ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gC4[mb * 2048 + nb * 16], l0cf[(mb * 8 + nb) * 256], ip);
        }
    }
    SetFlag<HardEvent::FIX_M>(evfixm);
    WaitFlag<HardEvent::FIX_M>(evfixm);
    AscendC::PipeBarrier<PIPE_ALL>();
    }

    l1AQue.FreeTensor(l1a);
    l1BQue.FreeTensor(l1b);
    l0CQue.FreeTensor(l0cf);
}
