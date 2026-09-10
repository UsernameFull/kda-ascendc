#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BV=64,D=128,M=16,TILE=M*BV;
extern "C" __global__ __aicore__ void kda_k2_persistent_kernel(
    GM_ADDR pW,GM_ADDR pQg,GM_ADDR pU,GM_ADDR pAqk,GM_ADDR pKg,GM_ADDR pDecay,
    GM_ADDR pH0,GM_ADDR pOut,GM_ADDR pHt,
    int32_t BH,int32_t NT,int32_t NV,float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int32_t task=GetBlockIdx(), tasks=BH*NV; if(task>=tasks)return;
    int32_t bh=task/NV, iv=task-bh*NV;
    TPipe pipe; TEventID e2v=pipe.AllocEventID<HardEvent::MTE2_V>(); TEventID ev3=pipe.AllocEventID<HardEvent::V_MTE3>(); TEventID evs=pipe.AllocEventID<HardEvent::V_S>();
    TBuf<TPosition::VECCALC> us,us16,uw,uq,uu,uk,ua,udec,uf,d1b,d2b,vb,d3b,d4b,ob,vfb,ofb,wfb,qfb,ufb,kfb,afb;
    pipe.InitBuffer(us,BV*D*4); pipe.InitBuffer(us16,BV*D*2); pipe.InitBuffer(uw,M*D*2); pipe.InitBuffer(uq,M*D*2); pipe.InitBuffer(uu,M*D*2); pipe.InitBuffer(uk,M*D*2); pipe.InitBuffer(ua,M*M*2); pipe.InitBuffer(udec,D*4); pipe.InitBuffer(uf,BV*D*4); pipe.InitBuffer(d1b,TILE*4); pipe.InitBuffer(d2b,TILE*4); pipe.InitBuffer(vb,TILE*2); pipe.InitBuffer(d3b,TILE*4); pipe.InitBuffer(d4b,BV*D*4); pipe.InitBuffer(ob,TILE*2); pipe.InitBuffer(vfb,TILE*4); pipe.InitBuffer(ofb,TILE*4); pipe.InitBuffer(wfb,M*D*4); pipe.InitBuffer(qfb,M*D*4); pipe.InitBuffer(ufb,M*D*4); pipe.InitBuffer(kfb,M*D*4); pipe.InitBuffer(afb,M*M*4);
    LocalTensor<float> s=us.Get<float>(), dec=udec.Get<float>(), sf=us.Get<float>(), d1=d1b.Get<float>(), d2=d2b.Get<float>(), d3=d3b.Get<float>(), d4=d4b.Get<float>(), vf=vfb.Get<float>(), of=ofb.Get<float>(), wf=wfb.Get<float>(), qf=qfb.Get<float>(), uf32=ufb.Get<float>(), kf=kfb.Get<float>(), af=afb.Get<float>();
    LocalTensor<bfloat16_t> sb=us16.Get<bfloat16_t>(), wb=uw.Get<bfloat16_t>(), qb=uq.Get<bfloat16_t>(), ub=uu.Get<bfloat16_t>(), kb=uk.Get<bfloat16_t>(), ab=ua.Get<bfloat16_t>(), vbv=vb.Get<bfloat16_t>(), outb=ob.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> W,Qg,U,Kg,Aqk; GlobalTensor<float> Decay,H0,Ht; GlobalTensor<bfloat16_t> Out;
    W.SetGlobalBuffer((__gm__ bfloat16_t*)pW); Qg.SetGlobalBuffer((__gm__ bfloat16_t*)pQg); U.SetGlobalBuffer((__gm__ bfloat16_t*)pU); Kg.SetGlobalBuffer((__gm__ bfloat16_t*)pKg); Aqk.SetGlobalBuffer((__gm__ bfloat16_t*)pAqk); Decay.SetGlobalBuffer((__gm__ float*)pDecay); H0.SetGlobalBuffer((__gm__ float*)pH0); Out.SetGlobalBuffer((__gm__ bfloat16_t*)pOut); Ht.SetGlobalBuffer((__gm__ float*)pHt);
    uint64_t s0=(uint64_t)task*BV*D;
    if(pH0==nullptr) Duplicate(s,0.0f,BV*D); else { uint64_t h0b=((uint64_t)bh*D*D+(uint64_t)iv*BV*D); DataCopy(s,H0[h0b],DataCopyParams(BV,16,0,0)); SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v); }
    for(int chunk=0;chunk<NT;chunk++) {
        int c=bh*NT+chunk; uint64_t ce=(uint64_t)c*M*D, cm=(uint64_t)c*M*M, co=(uint64_t)(task*NT+chunk)*TILE, cd=(uint64_t)c*D;
        DataCopy(wb,W[ce],DataCopyParams(M,8,0,0)); DataCopy(qb,Qg[ce],DataCopyParams(M,8,0,0)); DataCopy(ub,U[ce],DataCopyParams(M,8,0,0)); DataCopy(kb,Kg[ce],DataCopyParams(M,8,0,0)); DataCopy(ab,Aqk[cm],DataCopyParams(M,1,0,0)); DataCopy(dec,Decay[cd],DataCopyParams(1,16,0,0));
        SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
        Cast(wf,wb,RoundMode::CAST_NONE,M*D); Cast(qf,qb,RoundMode::CAST_NONE,M*D); Cast(uf32,ub,RoundMode::CAST_NONE,M*D); Cast(kf,kb,RoundMode::CAST_NONE,M*D); Cast(af,ab,RoundMode::CAST_NONE,M*M);
        PipeBarrier<PIPE_V>();
        Duplicate(d1,0.0f,TILE); Duplicate(d2,0.0f,TILE); PipeBarrier<PIPE_V>();
        for(int i=0;i<M;i++) for(int v=0;v<BV;v++){ float x=0,y=0; for(int k=0;k<D;k++){float st=sf.GetValue(v*D+k); x+=wf.GetValue(i*D+k)*st; y+=qf.GetValue(i*D+k)*st;} d1.SetValue(i*BV+v,x); d2.SetValue(i*BV+v,y); }
        for(int i=0;i<M;i++) for(int v=0;v<BV;v++) vf.SetValue(i*BV+v,uf32.GetValue(i*D+iv*BV+v)-d1.GetValue(i*BV+v));
        Duplicate(d3,0.0f,TILE); Duplicate(d4,0.0f,BV*D); PipeBarrier<PIPE_V>();
        for(int i=0;i<M;i++) for(int v=0;v<BV;v++){float z=0; for(int j=0;j<M;j++) z+=af.GetValue(i*M+j)*vf.GetValue(j*BV+v); d3.SetValue(i*BV+v,z);}
        for(int v=0;v<BV;v++) for(int k=0;k<D;k++){float z=0; for(int i=0;i<M;i++) z+=vf.GetValue(i*BV+v)*kf.GetValue(i*D+k); d4.SetValue(v*D+k,z);}
        Muls(of,d2,scale,TILE); Add(of,of,d3,TILE);
        Cast(outb,of,RoundMode::CAST_RINT,TILE);
        for(int v=0;v<BV;v++) Mul(s[v*D],s[v*D],dec,D);
        Add(s,s,d4,BV*D);
        SetFlag<HardEvent::V_S>(evs); WaitFlag<HardEvent::V_S>(evs);
        SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3); DataCopy(Out[co],outb,DataCopyParams(M,4,0,0));
    }
    SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3); if(pHt!=nullptr) DataCopy(Ht[s0],s,DataCopyParams(BV,16,0,0));
}
