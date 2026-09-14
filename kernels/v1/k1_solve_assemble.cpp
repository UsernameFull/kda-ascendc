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
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;  // parent chunk size (the wide kernel's PC)
constexpr int32_t M = PC / 2;      // diagonal sub-block size
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;     // 16-row fractal bands
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;   // one [M, M] bf16 operand
constexpr int32_t SLOTS = PASSES * NC;  // one L0 slot per (pass, chunk) unit

extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    // One L1 buffer per chunk of the block: the loads of a whole pass are
    // issued before its arithmetic starts, so a two-deep queue would stall on
    // the third AllocTensor (that is a deadlock, not a slowdown - measured).
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, LBSZ);
    pipe.InitBuffer(qb, NC, LBSZ);
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

    for (int32_t pass = 0; pass < PASSES; ++pass) {
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
            const int32_t slot = pass * NC + ch;
            LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
            for (int32_t dd = 0; dd < KF; ++dd) {
                for (int32_t mm = 0; mm < KF; ++mm) {
                    LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                             LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                }
            }
            LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            LocalTensor<float> cf = cfall[slot * MM];
            Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            // Pass 0 writes the intermediate row-major; pass 1 writes the
            // parent tile's lower-left block, whose rows are PC elements
            // apart instead of M.
            auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            if (pass == 0) {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
            } else {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
            }
            qa.FreeTensor(la);
            qb.FreeTensor(lb);
        }
        // Pass 1 reads the P tiles pass 0 wrote.
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
