// K1 stage 3c: the coupling block of the two-level solve.
//
//   X21 = X22 @ Lneg21 @ X11                (all three [M, M], M = KDA_CHUNK/2)
//
// With KDA_SOLVE_WIDE_SUBB = 2 the wide kernel solves the two diagonal M x M
// triangles of every KDA_CHUNK-sized chunk instead of the whole chunk: the
// row recursion costs M^3/2 vector lanes per chunk, so halving the depth
// quarters the vector work, and the instances of its tile become (sub-block,
// chunk) pairs.  What is left of the chunk-sized inverse is the coupling
// block of the 2x2 block inverse
//
//   A_inv = [[X11, 0], [-X22 L21 X11, X22]]
//
// which holds for I + L split as [[I+L11, 0], [L21, I+L22]] (L strictly
// lower).  X11 / X22 arrive contiguous from the wide kernel (Xb) next to the
// parent-tile copies the Cube solve already consumes, and Lneg21 is its
// negated bf16 export, so this kernel only has to form that block and write
// it into the parent A16 tile - kda_solve_wu_cube_kernel stays untouched.
// This kernel's output is the only writer of the parent tile's lower-left
// block; the wide kernel writes its diagonal and zeroes its strict upper
// triangle (the Cube solve reads all of it).
//
// The intermediate P = Lneg21 @ X11 goes through GM because L0C has no path
// back into L0A/L0B; splitting the two Mmads into two passes keeps that round
// trip off the per-chunk critical path (only the boundary between the passes
// has to drain), and both passes stream like the Cube solve does: the L1
// loads of every chunk of a pass are issued before its arithmetic starts.
//
// Load path (section 11.37): the shipped form issues six ND2NZ calls per chunk
// (one 2 KB whole-tile A load plus two 1 KB B bands per pass, 1.37 KB and
// 4.25 ns per call) and the block's traffic alone measured 0.313-0.339 ms for
// 100.7 MB against the Cube solve's 839 GB/s cold-path marginal rate, so the
// transfers - not the bytes - were the floor.  The sources are uniformly
// strided (Lneg chunk-contiguous, Xb's pass-1 block 2*MM apart, P
// chunk-contiguous, and a [M, M] tile's two 16-row bands are 1 KB apart in all
// three), so Nd2NzParams' ndNum can carry several ND matrices per call:
//
//   mode 0  shipped: one B1 queue per operand (per-chunk EnQue/DeQue gives the
//           per-chunk load -> LoadData pipelining), per-chunk calls
//   mode 1  batched: one explicit B1 buffer per operand, the A operand in one
//           call per block, a chunk's two B bands merged (ndNum = KF), and in
//           pass 1 the whole block's B operand in one 8 KB call.  Pass 0 is
//           1 + NC calls and pass 1 is 2, against 3 * NC each.
//   mode 2  mode 1 plus P on chip (docs section 11.39): pass 0's fixpipe
//           writes P into L1 in NZ (CFG_NZ, the same F322BF16 quantization)
//           instead of its own GM tile, pass 1 builds L0B from that region
//           with LoadDataWithTranspose, and the GM tile is neither written nor
//           read - 100.7 MB per call instead of 151.0, and the probe measured
//           0.096 ms of the round trip's 0.107 ms ceiling.  The NZ fractal
//           order is [n-block][k-block] where the Nd2Nz band-major source was
//           [k-block][n-block], so the four 16x16 fractals go into L0B in the
//           order 0, 2, 1, 3 - A16 is bit-identical to mode 0, which is what
//           says the mapping is right.
//
// Both paths are the same arithmetic in the same L1 layout;
// tools/probe_solve_assemble_coalesce.py checks the two are bit-identical
// (P and A16) before it times either, and the mode is a runtime argument
// (api.asm_load_mode(), KDA_ASM_LOADS) so the same binary can be flipped.
//
// Modes 5-8 of that probe also measured the two structures separately: at the
// shipped call pattern the queue is worth 0.196 ms over an explicit buffer
// (0.706 vs 0.901 ms - the per-chunk pipelining hides part of the chain), but
// with the batched calls the explicit buffer wins (0.531 vs 0.606 ms), which
// is why mode 1 pairs the batching with the explicit buffer.
//
// The queue's NC-deep shape is what pinned KDA_ASM_NCHUNK at 4 (6 and 8
// deadlock, measured); the explicit buffer has no such depth constraint, so
// re-testing a fatter block is a follow-up, not a claim.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;  // parent chunk size (the wide kernel's PC)
constexpr int32_t M = PC / 2;      // diagonal sub-block size
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;     // 16-row fractal bands
constexpr int32_t BANDE = KF * 256;  // elements of one [16, M] band (512 at M=32)
constexpr int32_t BAND = BANDE * 2;  // bytes of one band (1 KB at M=32)
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;   // one [M, M] bf16 operand
constexpr int32_t SLOTS = PASSES * NC;  // one L0 slot per (pass, chunk) unit

// One chunk's L0 chain and store, shared by both load paths.
static __aicore__ inline void assemble_chunk(
    int32_t pass, int32_t slot, bool onchip, LocalTensor<bfloat16_t> la,
    LocalTensor<bfloat16_t> lb, LocalTensor<bfloat16_t> lp,
    LocalTensor<uint8_t>& a8, LocalTensor<uint8_t>& b8, LocalTensor<float>& cfall,
    GlobalTensor<bfloat16_t>& P, GlobalTensor<bfloat16_t>& A16,
    int32_t c0, int32_t ch, TEventID e1m, TEventID emf) {
    LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    for (int32_t dd = 0; dd < KF; ++dd) {
        for (int32_t mm = 0; mm < KF; ++mm) {
            LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                     LoadData2dParams(0, 1, 1, 0, 0, false, 0));
        }
    }
    if (pass == 0 || !onchip) {
        LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
    } else {
        // P out of L1 in NZ order: the four 512-byte fractals are ordered
        // [n-block][k-block] instead of the Nd2Nz source's [k-block][n-block],
        // so L0B's slots take source blocks 0, 2, 1, 3.
        const int32_t src = ch * MM;
        LoadDataWithTranspose(b, lp[src], LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[2 * 256], lp[src + 256],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[1 * 256], lp[src + 512],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[3 * 256], lp[src + 768],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
    }
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);
    LocalTensor<float> cf = cfall[slot * MM];
    Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);
    // Pass 0 writes the intermediate row-major; pass 1 writes the parent
    // tile's lower-left block, whose rows are PC elements apart instead of M.
    auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
    ip.quantPre = QuantMode_t::F322BF16;
    ip.unitFlag = 0;
    if (pass == 0 && onchip) {
        auto ipnz = FixpipeParamsV220(M, M, M, M, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lp[ch * MM], cf, ipnz);
    } else if (pass == 0) {
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
    } else {
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
    }
}

extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C,
    int32_t loadMode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    const bool batched = (loadMode != 0);
    const bool onchip = (loadMode >= 2);
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    // Mode 0's L1 shape: one queue per chunk of the block.  The loads of a
    // whole pass are issued before its arithmetic starts, so a two-deep queue
    // would stall on the third AllocTensor (that is a deadlock, not a
    // slowdown - measured).
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, LBSZ);
    pipe.InitBuffer(qb, NC, LBSZ);
    // Mode 1's L1 shape: the same 2 * NC * LBSZ bytes as explicit buffers, one
    // per operand, written by one batched call each.  Both shapes are declared
    // because the mode is a runtime argument; together they are 32 KB of L1.
    TBuf<TPosition::B1> bufA, bufB;
    pipe.InitBuffer(bufA, NC * LBSZ);
    pipe.InitBuffer(bufB, NC * LBSZ);
    // Mode 2's P tiles, one [M, M] bf16 per chunk of the block (8 KB at NC = 4,
    // M = 32): written by pass 0's fixpipe, read by pass 1's L0B fill.  It
    // replaces the GM round trip, not L1 - the buffer only exists when the mode
    // is on, so mode 0/1 leave it as 8 KB of unused L1.
    TBuf<TPosition::B1> bufP;
    pipe.InitBuffer(bufP, NC * LBSZ);
    // One L0 slot per (pass, chunk) unit: 2 * NC * 4 KB = 32 KB of the 128 KB
    // L0C, 16 KB of each of the 64 KB L0A/L0B.
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lbAll = bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();

    for (int32_t pass = 0; pass < PASSES; ++pass) {
        if (batched) {
            // Same bytes, fewer calls: the A tiles of the block in one Nd2Nz
            // (the source stride is MM for Lneg and 2*MM for Xb's pass-1
            // block), a chunk's two B bands merged, and in pass 1 the whole
            // block's B operand in one call (P is chunk-contiguous, so its
            // eight 1 KB bands are one 8 KB transfer).  1 + NC calls for pass
            // 0, 2 for pass 1.
            if (pass == 0) {
                DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                         Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
                for (int32_t ch = 0; ch < nch; ++ch) {
                    DataCopy(lbAll[ch * MM],
                             Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM],
                             Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                }
            } else {
                DataCopy(laAll, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                         Nd2NzParams(nch, M, M, 2 * MM, M, M, 1, MM));
                if (!onchip) {
                    // Mode 2 has no P in GM: pass 1's B operand comes from the
                    // L1 tiles pass 0's fixpipe just wrote.
                    DataCopy(lbAll, P[static_cast<uint64_t>(c0) * MM],
                             Nd2NzParams(KF * nch, 16, M, BANDE, M, 16, 1, BANDE));
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(e21);
            WaitFlag<HardEvent::MTE2_MTE1>(e21);
            for (int32_t ch = 0; ch < nch; ++ch) {
                assemble_chunk(pass, pass * NC + ch, onchip, laAll[ch * MM],
                               lbAll[ch * MM], lpAll, a8, b8, cfall, P, A16,
                               c0, ch, e1m, emf);
            }
        } else {
            for (int32_t ch = 0; ch < nch; ++ch) {
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lb = qb.AllocTensor<bfloat16_t>();
                // The A operand is one whole-tile Nd2Nz call (the layout the
                // crossed LoadData indices below undo, as in k1_solve_wu_cube);
                // the B operand goes through LoadDataWithTranspose, which wants
                // the band-major order the per-band calls here produce.
                if (pass == 0) {
                    DataCopy(la, Lneg[static_cast<uint64_t>(c0 + ch) * MM],
                             Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        DataCopy(lb[mm * KF * 256],
                                 Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + mm * 16 * M],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                } else {
                    DataCopy(la, Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + MM],
                             Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        DataCopy(lb[mm * KF * 256],
                                 P[static_cast<uint64_t>(c0 + ch) * MM + mm * 16 * M],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                }
                qa.EnQue(la);
                qb.EnQue(lb);
            }
            SetFlag<HardEvent::MTE2_MTE1>(e21);
            WaitFlag<HardEvent::MTE2_MTE1>(e21);
            for (int32_t ch = 0; ch < nch; ++ch) {
                auto la = qa.DeQue<bfloat16_t>();
                auto lb = qb.DeQue<bfloat16_t>();
                assemble_chunk(pass, pass * NC + ch, false, la, lb, lpAll, a8, b8,
                               cfall, P, A16, c0, ch, e1m, emf);
                qa.FreeTensor(la);
                qb.FreeTensor(lb);
            }
        }
        // Pass 1 reads the P tiles pass 0 wrote.
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
