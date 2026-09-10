"""Generate B-step-unrolled MIX1_blockB and MIX2_blockB kernels.
Each block is the verified single-step MIX1/MIX2 logic fully unrolled
(compile-time), dodging the dynamic-loop+cube compiler bug.

Output: k2_mix1_blockB.cpp next to this script (override with OUT_DIR).
"""
import os
from pathlib import Path

OUT_DIR = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent))

B = 8
E = 16 * 128

def gensig(name, params):
    return f'extern "C" __global__ __aicore__ void {name}({params}, GM_ADDR ws, GM_ADDR tiling) {{'

# ---------------- MIX1_blockB ----------------
parts = []
parts.append('''#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t E = 16 * 128;
constexpr float LN2 = 0.6931471805599453f;

''')
parts.append(gensig("k2_mix1_blockB",
    "GM_ADDR pw, GM_ADDR ph, GM_ADDR pu, GM_ADDR pq, GM_ADDR pk, GM_ADDR pg, GM_ADDR pglast, "
    "GM_ADDR pd1, GM_ADDR pvnew, GM_ADDR pvnewT, GM_ADDR pqg, GM_ADDR pkg"))
parts.append('''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
''')

# AIC side: B unrolled d1 blocks, each setflag8/waitflag9
parts.append('''#if defined(__DAV_C220_CUBE__)
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
    GlobalTensor<float> gD1;
    gB.SetGlobalBuffer((__gm__ bfloat16_t *)ph);
''')
for u in range(B):
    parts.append(f'''
    {{ // AIC block {u}
        uint64_t base = {u} * E;
        gA.SetGlobalBuffer((__gm__ bfloat16_t *)pw + base);
        gD1.SetGlobalBuffer((__gm__ float *)pd1 + base);
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
        for (int b = 0; b < 8; b++) {{
            auto ip = FixpipeParamsV220(16, 16, 1, 128, false);
            ip.quantPre = QuantMode_t::NoQuant; ip.unitFlag = 0;
            Fixpipe<float, float, CFG_ROW_MAJOR>(gD1[b * 16], l0cf[b * 256], ip);
        }}
        SetFlag<HardEvent::FIX_M>(evfixm);
        WaitFlag<HardEvent::FIX_M>(evfixm);
        l1AQue.FreeTensor(la); l1BQue.FreeTensor(lb);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(8);
        CrossCoreWaitFlag<2, PIPE_MTE3>(9);
    }}''')
parts.append('''    l0CQue.FreeTensor(l0cf);
''')

# AIV side: B unrolled v_new/qg/kg/vnewT blocks
parts.append('''#elif defined(__DAV_C220_VEC__)
    TPipe pipe;
    TBuf<TPosition::VECIN> ubUb, ubU, ubD1, ubV, ubVb, ubQ, ubK, ubG, ubQb, ubKb, ubT, ubT2, ubGl, ubTg, ubTg2;
    pipe.InitBuffer(ubUb, E * 2);
    pipe.InitBuffer(ubU, E * 4);
    pipe.InitBuffer(ubD1, E * 4);
    pipe.InitBuffer(ubV, E * 4);
    pipe.InitBuffer(ubVb, E * 2);
    pipe.InitBuffer(ubQ, E * 4);
    pipe.InitBuffer(ubK, E * 4);
    pipe.InitBuffer(ubG, E * 4);
    pipe.InitBuffer(ubQb, E * 2);
    pipe.InitBuffer(ubKb, E * 2);
    pipe.InitBuffer(ubT, E * 4);
    pipe.InitBuffer(ubT2, E * 4);
    pipe.InitBuffer(ubGl, 128 * 4);
    pipe.InitBuffer(ubTg, 16 * 16 * 2);
    pipe.InitBuffer(ubTg2, 16 * 16 * 2);
    LocalTensor<bfloat16_t> tUb = ubUb.Get<bfloat16_t>();
    LocalTensor<float> tU = ubU.Get<float>(), tD1 = ubD1.Get<float>(), tV = ubV.Get<float>(),
        tQ = ubQ.Get<float>(), tK = ubK.Get<float>(), tG = ubG.Get<float>(),
        tT = ubT.Get<float>(), tT2 = ubT2.Get<float>(), tGl = ubGl.Get<float>();
    LocalTensor<bfloat16_t> tVb = ubVb.Get<bfloat16_t>(), tQb = ubQb.Get<bfloat16_t>(), tKb = ubKb.Get<bfloat16_t>(),
        tTg = ubTg.Get<bfloat16_t>(), tTg2 = ubTg2.Get<bfloat16_t>();
    TEventID ev2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID evv3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    GlobalTensor<bfloat16_t> gU, gQ, gK, gVnew, gVnewT, gQqg, gKg;
    GlobalTensor<float> gD1, gG, gGl;
    gU.SetGlobalBuffer((__gm__ bfloat16_t *)pu);
    gQ.SetGlobalBuffer((__gm__ bfloat16_t *)pq);
    gK.SetGlobalBuffer((__gm__ bfloat16_t *)pk);
    gVnew.SetGlobalBuffer((__gm__ bfloat16_t *)pvnew);
    gVnewT.SetGlobalBuffer((__gm__ bfloat16_t *)pvnewT);
    gQqg.SetGlobalBuffer((__gm__ bfloat16_t *)pqg);
    gKg.SetGlobalBuffer((__gm__ bfloat16_t *)pkg);
    gD1.SetGlobalBuffer((__gm__ float *)pd1);
    gG.SetGlobalBuffer((__gm__ float *)pg);
    gGl.SetGlobalBuffer((__gm__ float *)pglast);
''')
for u in range(B):
    parts.append(f'''
    {{ // AIV block {u}
        uint64_t base = {u} * E;
        CrossCoreWaitFlag<2, PIPE_MTE2>(8);
        DataCopy(tUb, gU[base], E);
        DataCopy(tD1, gD1[base], E);
        DataCopy(tQb, gQ[base], E);
        DataCopy(tKb, gK[base], E);
        DataCopy(tG, gG[base], E);
        DataCopy(tGl, gGl[{u} * 128], 128);
        SetFlag<HardEvent::MTE2_V>(ev2v);
        WaitFlag<HardEvent::MTE2_V>(ev2v);
        Cast(tU, tUb, RoundMode::CAST_NONE, E);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tQ, tQb, RoundMode::CAST_NONE, E);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tK, tKb, RoundMode::CAST_NONE, E);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(tV, tU, tD1, E);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tVb, tV, RoundMode::CAST_RINT, E);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evv3);
        WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(gVnew[base], tVb, E);
        for (int b = 0; b < 8; b++) {{
            for (int r = 0; r < 16; r++) {{
                DataCopy(tTg[r * 16], tVb[r * 128 + b * 16], DataCopyParams(1, 1, 0, 0));
            }}
            SetFlag<HardEvent::MTE2_V>(ev2v);
            WaitFlag<HardEvent::MTE2_V>(ev2v);
            AscendC::Transpose(tTg2, tTg);
            AscendC::PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(evv3);
            WaitFlag<HardEvent::V_MTE3>(evv3);
            DataCopy(gVnewT[base + b * 256], tTg2, DataCopyParams(1, 16, 0, 0));
        }}
        Muls(tU, tG, LN2, E);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(tT, tU, E);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tU, tQ, tT, E);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tQb, tU, RoundMode::CAST_RINT, E);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evv3);
        WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(gQqg[base], tQb, E);
        for (int r = 0; r < 16; r++) {{
            Sub(tT2[r * 128], tGl, tG[r * 128], 128);
            AscendC::PipeBarrier<PIPE_V>();
        }}
        Muls(tD1, tT2, LN2, E);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(tT2, tD1, E);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tD1, tK, tT2, E);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tKb, tD1, RoundMode::CAST_RINT, E);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evv3);
        WaitFlag<HardEvent::V_MTE3>(evv3);
        DataCopy(gKg[base], tKb, E);
        AscendC::PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(9);
    }}''')
parts.append('''#endif
}''')
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "k2_mix1_blockB.cpp"
out.write_text('\n'.join(parts))
print(f"MIX1_blockB generated (B={B}) -> {out}")
