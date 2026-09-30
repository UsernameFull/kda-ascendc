// K1 stage 3c, SB = 4 build: the six coupling blocks of the four-level solve,
// computed entirely out of 16x16 operands on the Cube.
//
// With KDA_SOLVE_WIDE_SUBB = 4 the wide kernel solves the four diagonal 16x16
// triangles of every 64-wide chunk (k1_solve_wu_wide.cpp) and exports the six
// negated strictly-lower blocks of the chunk in one 1536-element bundle:
//
//   [ (1,0) | (3,2) | rows 32..63 x cols 0..31 ]
//
// This kernel forms the six coupling blocks X[s2,s1] (s2 > s1) of the
// chunk-sized inverse I_free = A_inv - I, i.e. the strictly-lower part of
//
//   X = [[X0,0,0,0],[X10,X1,0,0],[X20,X21,X2,0],[X30,X31,X32,X3]]
//
// by block forward substitution, all on chip:
//
//   X10 = X1  Ln10 X0                    X21 = X2 Ln21 X1
//   X32 = X3  Ln32 X2
//   X20 = X2 (Ln20 X0 + Ln21 X10)
//   X31 = X3 (Ln31 X1 + Ln32 X21)
//   X30 = X3 (Ln30 X0 + Ln31 X10 + Ln32 X20)
//
// (each line is the exact rearrangement of the 32-level form: e.g. the 32x32
// product X22m @ Lneg32big @ X11m expands to these six, with X32 = X3 Ln32 X2
// substituted wherever X22m's lower-left quadrant appears).  16 base Mmads of
// 16^3 per chunk - the same count as the 32-level schedule's two 32^3 - and
// every operand is a single 16x16 fractal, so no 32-wide composite is ever
// materialised and the L0A/L0B fills have no fractal-order permutation.
//
// Nine results round trip through L1 in NZ (the mode-2 path of the 32-level
// kernel): L0C has no read path back into L0A/L0B and three of the couplings
// feed later levels (X10 -> X20 -> X30, X21 -> X31, X20 -> X30), so they are
// fixpipe'd to L1 - the NZ write measured ~free next to a row-major store -
// and lifted back with LoadDataWithTranspose (B operands) or LoadData (A).
//
// Store shape (the point section 11.52 flagged for SB = 4): the six outputs
// sit in the parent tile at (16,0), (32,16), (48,32) - the 1-below-diagonal
// trio, a uniform 1040-element push - (32,0), (48,16) - the 2-below pair - and
// (48,0).  Their L0C slots are laid out in the same order, so each group
// leaves through ONE fixpipe call (ndNum = 3 / 2 / 1, srcNdStride = one 1 KB
// slot, dstNdStride = 1040 elements) instead of six 16-row stores: three calls
// per chunk against the production 32-level kernel's one 32-row call, at 1.5x
// the bytes.  storeMode 1 is the per-piece fallback (six calls), 2 drops the
// stores (an ablation - the tile is left half-written, never correct).
//
// Sync: M_FIX orders every Mad before its fixpipe and MTE1_M every LoadData
// before its Mad; the fixpipe -> LoadData edge - L1 tiles written by Fixpipe
// and read back by the level above's LoadData - is a PipeBarrier<PIPE_ALL>()
// at the head of each level, the pattern k1_solve_assemble's mode 2 uses where
// pass 1 reads the P tiles pass 0 fixpiped into L1.  The barrier form is a
// measured choice, not a style one.  PipeBarrier<PIPE_FIX> came first and only
// orders the FIX queue against itself: the next level's LoadData still beats
// the fixpipe that feeds it, and the raw gate comes back 5.0e-01 wrong exactly
// at the blocks whose operands cross that edge ((2,0), (3,0), (3,1)) while
// every L0C dump of a level's own Mads stays exact.  HardEvent::FIX_MTE1 is
// not the way out either: it exists in the header but appears nowhere in the
// CANN implementation, and with the tokens this kernel needs (one per level
// and chunk, one id per consumer stream) the core wedges at the first level
// that waits on a fixpipe output - measured at C = 4, NC = 4 as an aicore
// timeout with fixp errors set at level 3.  A level's LoadData calls only read
// tiles written by earlier levels, so one drain per level covers all of its
// waits, and the chunks still pipeline inside a level.  Every L0A/L0B/L0C slot
// belongs to exactly one (chunk, mmad), so nothing is rewritten inside a
// block; the bundle and Xb tiles are read-only after one MTE2_MTE1 token.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif

constexpr int32_t PC = KDA_CHUNK;  // parent chunk size (the wide kernel's PC)
constexpr int32_t M = PC / 4;      // SB = 4 sub-block size
constexpr int32_t MM = M * M;      // elements of one 16x16 tile
constexpr int32_t CE = 6 * MM;     // the Lneg bundle per chunk
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t NM = 16;         // Mmads per chunk (one A/B slot each)
constexpr int32_t RN = 12;         // L0C result slots per chunk

// One fractal move into L0A (A operands stay as ND2NZ/fixpipe wrote them).
static __aicore__ inline void load_a(int32_t ab, const LocalTensor<uint8_t>& a8, int32_t slot,
                                     const LocalTensor<bfloat16_t>& src, int32_t el) {
    if (ab & 4) return;
    LoadData(a8[slot * MM * 2].ReinterpretCast<bfloat16_t>(), src[el],
             LoadData2dParams(0, 1, 1, 0, 0, false, 0));
}

// One transposing fractal move into L0B: both the Nd2Nz bands and the fixpipe
// NZ output are read into L0B through LoadDataWithTranspose in the 32-level
// kernel, and a single 16x16 fractal is the same story with one repeat.
static __aicore__ inline void load_b(int32_t ab, const LocalTensor<uint8_t>& b8, int32_t slot,
                                     const LocalTensor<bfloat16_t>& src, int32_t el) {
    if (ab & 8) return;
    LoadDataWithTranspose(b8[slot * MM * 2].ReinterpretCast<bfloat16_t>(), src[el],
                          LoadData2dTransposeParams(0, 1, 1, 0, 0));
}

static __aicore__ inline void mad(int32_t ab, const LocalTensor<float>& cfall, int32_t res,
                                  const LocalTensor<uint8_t>& a8, int32_t a_slot,
                                  const LocalTensor<uint8_t>& b8, int32_t b_slot,
                                  bool init) {
    if (ab & 16) return;
    Mmad(cfall[res * MM],
         a8[a_slot * MM * 2].ReinterpretCast<bfloat16_t>(),
         b8[b_slot * MM * 2].ReinterpretCast<bfloat16_t>(),
         MmadParams(16, 16, 16, 0, false, init));
}

// A whole chain as ONE mad: the group's terms are loaded into consecutive
// L0A/L0B slots (k-block order), so C = A0 B0 + A1 B1 (+ A2 B2) is a single
// K = 16 * terms product with cmatrixInitVal = true.  This is not a style
// choice: two Mmads into the same L0C slot back to back are a hazard on this
// part - the second one's C read (init = false) can beat the first one's C
// write (measured: Ea/Gr/Gl came out 0.3-0.5 wrong, everything else exact,
// with the CANN matmul library's own workaround being an unconditional
// PipeBarrier<PIPE_M> after every small Mmad, mmad_compute.h).  Stacking the K
// blocks removes the read-modify-write instead of fencing it, and it is four
// Mads fewer per chunk: the wide cube kernel's K = 128 operand shows the
// fractal order used here (k blocks at consecutive 512 B slots).
static __aicore__ inline void madk(int32_t ab, const LocalTensor<float>& cfall, int32_t res,
                                   const LocalTensor<uint8_t>& a8, int32_t a_slot,
                                   const LocalTensor<uint8_t>& b8, int32_t b_slot,
                                   int32_t k) {
    if (ab & 16) return;
    Mmad(cfall[res * MM],
         a8[a_slot * MM * 2].ReinterpretCast<bfloat16_t>(),
         b8[b_slot * MM * 2].ReinterpretCast<bfloat16_t>(),
         MmadParams(16, 16, k, 0, false, true));
}

// L0C -> L1 in NZ (the same F322BF16 quantization mode 2 uses).
static __aicore__ inline void nz_out(int32_t ab, const LocalTensor<bfloat16_t>& lz, int32_t dst_el,
                                     const LocalTensor<float>& cfall, int32_t res) {
    if (ab & 2) return;
    auto ipnz = FixpipeParamsV220(16, 16, 16, 16, false);
    ipnz.quantPre = QuantMode_t::F322BF16;
    ipnz.unitFlag = 0;
    Fixpipe<bfloat16_t, float, CFG_NZ>(lz[dst_el], cfall[res * MM], ipnz);
}

extern "C" __global__ __aicore__ void kda_solve_assemble4(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, int32_t C, int32_t storeMode,
    int32_t ab, int32_t level) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufLn, bufXb, bufZ;
    pipe.InitBuffer(bufLn, NC * 6 * MM * 2);
    pipe.InitBuffer(bufXb, NC * 4 * MM * 2);
    pipe.InitBuffer(bufZ, NC * 9 * MM * 2);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, NC * NM * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, NC * NM * MM * 2);
    LocalTensor<float> cfall(TPosition::CO1, 0, NC * RN * MM);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    LocalTensor<bfloat16_t> ln = bufLn.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lx = bufXb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lz = bufZ.Get<bfloat16_t>();

    // The six bundle pieces (n10, n32, Ln20, Ln21, Ln30, Ln31) as nch ND
    // matrices each, and the four Xb tiles of every chunk in one call (their
    // per-chunk 1024-element block is four 256-element tiles back to back).
    if (!(ab & 1)) {
        const uint64_t lb0 = static_cast<uint64_t>(c0) * CE;
        DataCopy(ln[0 * NC * MM], Lneg[lb0 + 0 * MM],
                 Nd2NzParams(nch, 16, 16, CE, 16, 16, 1, MM));
        DataCopy(ln[1 * NC * MM], Lneg[lb0 + 1 * MM],
                 Nd2NzParams(nch, 16, 16, CE, 16, 16, 1, MM));
        DataCopy(ln[2 * NC * MM], Lneg[lb0 + 2 * MM],
                 Nd2NzParams(nch, 16, 16, CE, 32, 16, 1, MM));
        DataCopy(ln[3 * NC * MM], Lneg[lb0 + 2 * MM + 16],
                 Nd2NzParams(nch, 16, 16, CE, 32, 16, 1, MM));
        DataCopy(ln[4 * NC * MM], Lneg[lb0 + 2 * MM + 16 * 32],
                 Nd2NzParams(nch, 16, 16, CE, 32, 16, 1, MM));
        DataCopy(ln[5 * NC * MM], Lneg[lb0 + 2 * MM + 16 * 32 + 16],
                 Nd2NzParams(nch, 16, 16, CE, 32, 16, 1, MM));
        DataCopy(lx, Xb[static_cast<uint64_t>(c0) * 4 * MM],
                 Nd2NzParams(4 * nch, 16, 16, MM, 16, 16, 1, MM));
    }
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);

        if (level >= 1) {
    // ---- level 0: X10 = X1 Ln10 X0, X21 = X2 Ln21 X1, X32 = X3 Ln32 X2 ------
        // (the L-chains are the two-Mmad form of each coupling; the leaf products
        //  are Q10 / Ert / Q32, kept in L1 for the level-1 Mmads)
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            load_a(ab, a8, s + 0, ln, 0 * NC * MM + ch * MM);   // Ln10
            load_b(ab, b8, s + 0, lx, ch * 4 * MM + 0 * MM);    // X0
            load_a(ab, a8, s + 1, ln, 1 * NC * MM + ch * MM);   // Ln32
            load_b(ab, b8, s + 1, lx, ch * 4 * MM + 2 * MM);    // X2
            load_a(ab, a8, s + 2, ln, 3 * NC * MM + ch * MM);   // Ln21
            load_b(ab, b8, s + 2, lx, ch * 4 * MM + 1 * MM);    // X1
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            mad(ab, cfall, r + 0, a8, s + 0, b8, s + 0, true);   // Q10
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 0 * NC * MM + ch * MM, cfall, r + 0);
            mad(ab, cfall, r + 1, a8, s + 1, b8, s + 1, true);   // Q32
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 1 * NC * MM + ch * MM, cfall, r + 1);
            mad(ab, cfall, r + 2, a8, s + 2, b8, s + 2, true);   // Ert
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 2 * NC * MM + ch * MM, cfall, r + 2);
        }
    
        
    }

    if (level >= 2) {
        PipeBarrier<PIPE_ALL>();
    // ---- level 1: the three couplings at L0C slots r+3..r+5 ----------------
        // A trio store's source slots are contiguous and its three tile offsets a
        // uniform 1040 elements apart, so the group is one ndNum = 3 fixpipe.
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            // Q10/Q32/Ert are level 0's fixpipe outputs; the level-head barrier
            // (PIPE_ALL, above) is what orders those writes before these reads.
            load_a(ab, a8, s + 3, lx, ch * 4 * MM + 1 * MM);     // X1
            load_b(ab, b8, s + 3, lz, 0 * NC * MM + ch * MM);    // Q10
            load_a(ab, a8, s + 5, lx, ch * 4 * MM + 3 * MM);     // X3
            load_b(ab, b8, s + 5, lz, 1 * NC * MM + ch * MM);    // Q32
            load_a(ab, a8, s + 4, lx, ch * 4 * MM + 2 * MM);     // X2
            load_b(ab, b8, s + 4, lz, 2 * NC * MM + ch * MM);    // Ert
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            mad(ab, cfall, r + 3, a8, s + 3, b8, s + 3, true);   // X10
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 3 * NC * MM + ch * MM, cfall, r + 3);
            mad(ab, cfall, r + 4, a8, s + 4, b8, s + 4, true);   // X21
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 4 * NC * MM + ch * MM, cfall, r + 4);
            mad(ab, cfall, r + 5, a8, s + 5, b8, s + 5, true);   // X32
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            if (storeMode != 2 && !(ab & 32)) {
                if (storeMode == 0) {
                    auto ipt = FixpipeParamsV220(16, 16, 16, PC, false,
                                                 QuantMode_t::F322BF16, 0, 3, 1, 1040, 0);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 16 * PC],
                        cfall[(r + 3) * MM], ipt);
                } else {
                    auto ip1 = FixpipeParamsV220(16, 16, 16, PC, false);
                    ip1.quantPre = QuantMode_t::F322BF16;
                    ip1.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 16 * PC],
                        cfall[(r + 3) * MM], ip1);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 32 * PC + 16],
                        cfall[(r + 4) * MM], ip1);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 48 * PC + 32],
                        cfall[(r + 5) * MM], ip1);
                }
            }
        }
    
        
    }

    if (level >= 3) {
        PipeBarrier<PIPE_ALL>();
    // ---- level 2: Ea = Ln20 X0 + Ln21 X10, Gr = Ln31 X1 + Ln32 X21 --------
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            load_a(ab, a8, s + 6, ln, 2 * NC * MM + ch * MM);    // Ln20
            load_b(ab, b8, s + 6, lx, ch * 4 * MM + 0 * MM);     // X0
            load_a(ab, a8, s + 7, ln, 3 * NC * MM + ch * MM);    // Ln21
            load_b(ab, b8, s + 7, lz, 3 * NC * MM + ch * MM);    // X10
            load_a(ab, a8, s + 8, ln, 5 * NC * MM + ch * MM);    // Ln31
            load_b(ab, b8, s + 8, lx, ch * 4 * MM + 1 * MM);     // X1
            load_a(ab, a8, s + 9, ln, 1 * NC * MM + ch * MM);    // Ln32
            load_b(ab, b8, s + 9, lz, 4 * NC * MM + ch * MM);    // X21
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            // Ea = [Ln20 | Ln21] @ [[X0], [X10]]: one K = 32 Mad (slots s+6,
            // s+7 already hold exactly that order).
            madk(ab, cfall, r + 6, a8, s + 6, b8, s + 6, 32);
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 5 * NC * MM + ch * MM, cfall, r + 6);
            // Gr = [Ln31 | Ln32] @ [[X1], [X21]], one K = 32 Mad.
            madk(ab, cfall, r + 7, a8, s + 8, b8, s + 8, 32);
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 7 * NC * MM + ch * MM, cfall, r + 7);
        }
    
        
    }

    if (level >= 4) {
        PipeBarrier<PIPE_ALL>();
    // ---- level 3: X20 = X2 Ea, X31 = X3 Gr ---------------------------------
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            load_a(ab, a8, s + 10, lx, ch * 4 * MM + 2 * MM);    // X2
            load_b(ab, b8, s + 10, lz, 5 * NC * MM + ch * MM);   // Ea
            load_a(ab, a8, s + 11, lx, ch * 4 * MM + 3 * MM);    // X3
            load_b(ab, b8, s + 11, lz, 7 * NC * MM + ch * MM);   // Gr
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            mad(ab, cfall, r + 8, a8, s + 10, b8, s + 10, true);  // X20
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 6 * NC * MM + ch * MM, cfall, r + 8);
            mad(ab, cfall, r + 9, a8, s + 11, b8, s + 11, true);  // X31
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            if (storeMode != 2 && !(ab & 64)) {
                if (storeMode == 0) {
                    auto ipp = FixpipeParamsV220(16, 16, 16, PC, false,
                                                 QuantMode_t::F322BF16, 0, 2, 1, 1040, 0);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 32 * PC],
                        cfall[(r + 8) * MM], ipp);
                } else {
                    auto ip1 = FixpipeParamsV220(16, 16, 16, PC, false);
                    ip1.quantPre = QuantMode_t::F322BF16;
                    ip1.unitFlag = 0;
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 32 * PC],
                        cfall[(r + 8) * MM], ip1);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 48 * PC + 16],
                        cfall[(r + 9) * MM], ip1);
                }
            }
        }
    
        
    }

    if (level >= 5) {
        PipeBarrier<PIPE_ALL>();
    // ---- level 4: Gl = Ln30 X0 + Ln31 X10 + Ln32 X20 -----------------------
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            load_a(ab, a8, s + 12, ln, 4 * NC * MM + ch * MM);   // Ln30
            load_b(ab, b8, s + 12, lx, ch * 4 * MM + 0 * MM);    // X0
            load_a(ab, a8, s + 13, ln, 5 * NC * MM + ch * MM);   // Ln31
            load_b(ab, b8, s + 13, lz, 3 * NC * MM + ch * MM);   // X10
            load_a(ab, a8, s + 14, ln, 1 * NC * MM + ch * MM);   // Ln32
            load_b(ab, b8, s + 14, lz, 6 * NC * MM + ch * MM);   // X20
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            // Gl = [Ln30 | Ln31 | Ln32] @ [[X0], [X10], [X20]], one K = 48 Mad.
            madk(ab, cfall, r + 10, a8, s + 12, b8, s + 12, 48);
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            nz_out(ab, lz, 8 * NC * MM + ch * MM, cfall, r + 10);
        }
    
        
    }

    if (level >= 6) {
        PipeBarrier<PIPE_ALL>();
    // ---- level 5: X30 = X3 Gl ----------------------------------------------
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM;
            load_a(ab, a8, s + 15, lx, ch * 4 * MM + 3 * MM);    // X3
            load_b(ab, b8, s + 15, lz, 8 * NC * MM + ch * MM);   // Gl
            SetFlag<HardEvent::MTE1_M>(e1m);
        }
        for (int32_t ch = 0; ch < nch; ++ch) {
            const int32_t s = ch * NM, r = ch * RN;
            WaitFlag<HardEvent::MTE1_M>(e1m);
            mad(ab, cfall, r + 11, a8, s + 15, b8, s + 15, true);  // X30
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            if (storeMode != 2 && !(ab & 128)) {
                auto ip1 = FixpipeParamsV220(16, 16, 16, PC, false);
                ip1.quantPre = QuantMode_t::F322BF16;
                ip1.unitFlag = 0;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    A16[static_cast<uint64_t>(c0 + ch) * PC * PC + 48 * PC],
                    cfall[(r + 11) * MM], ip1);
            }
        }
    }

    PipeBarrier<PIPE_ALL>();
}
