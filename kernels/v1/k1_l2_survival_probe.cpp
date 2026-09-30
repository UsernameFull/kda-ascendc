// Timing probe (NOT a production kernel): does an L2 line survive a launch
// boundary?
//
// Section 11.46 measured the marginal price of a *cold* cross-launch read at
// 1.427 ms/GB inside the Cube solve - 7.1x the 0.2 ms/GB the ledger had been
// using since section 11.12 - and that repriced the Level 2/3 window streaming
// from 0.20 ms to 0.74 + 0.50 ms.  The repricing is an upper bound, and it
// rests on one property nobody has measured: those two levels only pay off if a
// small double-window ring, written by launch N and read **once** by launch
// N+1, is served from L2 rather than from HBM.
//
// What 11.46's hot arm proved is weaker: a small window re-read by 6144 blocks
// *inside one launch* stays resident.  That reuse is intra-launch (one miss
// amortised over ~384 hits), so it would look free even if every launch
// boundary invalidated L2.  Section 11.25 route C points the other way
// (2.01 GB of handoff at 1.047 ms/GB, no L2 help) but 2.01 GB does not fit in
// L2, so it says nothing about a ring.
//
// This isolates the property.  One kernel, fixed grid, fixed 4 KB granule (the
// burst length the Cube's Nd2Nz RHS loads use); each block owns the same
// NSLOT-tile span and ``ntile`` says how much of it is touched, so the
// footprint sweeps 25.2 -> 201.3 MB with the per-block instruction shape and
// the address layout unchanged:
//
//   mode 0  consumer: reads ``ntile`` of its tiles, each byte once
//   mode 1  producer: writes the same ``ntile`` tiles from UB
//   ntile=0 the floor: same grid, same epilogue, no span traffic
//
// STATUS 2026-09-28: **this instrument does not work.**  The sweep is flat at
// +0.002 ms for +201 MB of footprint, and it stayed flat through three cuts:
// a 32 B-per-tile epilogue, a whole-tile epilogue, and a whole-tile epilogue
// plus the argument echo below.  The echo settles what the flatness is *not*:
// block 0 stamps the ntile and mode it received into out's tail and the host
// reads back exactly (8,1), (3,1), (8,0), (1,3), so the scalars are delivered
// and this is not an argument-packing bug.  It is also not the loads alone -
// the epilogue's own 201 MB of stores into the 4 MB ring cost 0.000 ms (the
// floor is 0.220 ms with a 1.57 MB epilogue and 0.219 ms with a 201 MB one) -
// so no GM traffic in this kernel registers, in either direction, and the
// cause is not identified.  tools/probe_l2_survival.py detects this and prints
// INSTRUMENT INVALID instead of a verdict.
//
// The design notes are kept because they are the reasons the file looks the way
// it does, and because the next attempt needs them:
//
//   (a) *Liveness.* Every loaded byte is stored by the epilogue, so no load can
//       be shrunk or dropped as dead.  Section 11.46's Cube probe does not need
//       this because there the loads are load-bearing all the way through
//       LoadDataWithTranspose -> Mmad -> Fixpipe - and that is the property
//       this microbenchmark has not managed to reproduce.
//   (b) *Self-eviction.* The epilogue stores into a small ring (OUT_SLOTS x
//       32 KB = 4 MB) rather than a full-size output, so 201 MB of store
//       traffic does not evict the very span whose residency is being measured
//       before the next launch reads it.  The store cost is identical on every
//       arm, so it cancels in the leg above the floor.
//   (c) The address stride is NSLOT tiles per block regardless of ntile, so the
//       span a block owns never moves; only the touched part of it changes.
//
// The recommended next instrument is not another standalone microbenchmark:
// kernels/v1/k1_solve_cube_rhs_probe.cpp is *known* to move real bytes (its
// arms span 0.615-1.422 ms and are monotone in the bytes), so the survival arm
// belongs there - a first launch writes a ring-sized RHS window, the cube
// transcription reads it as its RHS, against the same window read cold.
//
// Reading: if L2 survives a launch boundary, ms/GB has a knee - cheap while the
// footprint fits, HBM-priced once it does not.  If it does not survive, ms/GB
// is flat at the HBM price and Level 2/3 cannot capture the repriced 0.74 ms
// by streaming.  tools/probe_l2_survival.py checks the largest leg against what
// the cold price predicts and prints INSTRUMENT INVALID rather than a verdict
// if the loads turn out not to be reaching GM again.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_L2P_TILE
#define KDA_L2P_TILE 4096          // bytes per DataCopy: the Cube's RHS granule
#endif
#ifndef KDA_L2P_OUTSLOTS
#define KDA_L2P_OUTSLOTS 128       // ring slots for the epilogue, see trap (b)
#endif
constexpr int32_t TILE_B = KDA_L2P_TILE;
constexpr int32_t TILE_E = KDA_L2P_TILE / 2;   // bf16 elements per tile
constexpr int32_t NSLOT = 8;                   // tiles in a block's span
constexpr int32_t OUT_SLOTS = KDA_L2P_OUTSLOTS;

extern "C" __global__ __aicore__ void kda_l2_survival_probe(
    GM_ADDR pBuf, GM_ADDR pOut, int32_t ntile, int32_t mode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t bid = static_cast<int32_t>(GetBlockIdx());
    TPipe pipe;
    TBuf<TPosition::VECCALC> bU;
    pipe.InitBuffer(bU, NSLOT * TILE_B);
    LocalTensor<bfloat16_t> u = bU.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> buf, out;
    buf.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pBuf));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
    // Fixed stride: every block owns the same NSLOT tiles whether it touches
    // one of them or all eight.
    const uint64_t base = static_cast<uint64_t>(bid) * NSLOT * TILE_E;
    if (mode == 1) {
        // One read back as the source of the writes, so the stores are not
        // garbage.  bf16 has no float-scalar Duplicate.
        DataCopy(u, buf[0], NSLOT * TILE_E);
        PipeBarrier<PIPE_MTE2>();
        for (int32_t i = 0; i < ntile; ++i) {
            DataCopy(buf[base + static_cast<uint64_t>(i) * TILE_E],
                     u[i * TILE_E], TILE_E);
        }
        PipeBarrier<PIPE_MTE3>();
    } else {
        for (int32_t i = 0; i < ntile; ++i) {
            DataCopy(u[i * TILE_E],
                     buf[base + static_cast<uint64_t>(i) * TILE_E], TILE_E);
        }
        PipeBarrier<PIPE_MTE2>();
    }
    // Trap (a): the whole tile goes out, so every loaded byte is consumed.
    // Trap (b): it goes into a 4 MB ring, so it does not evict what is being
    // measured.
    DataCopy(out[static_cast<uint64_t>(bid % OUT_SLOTS) * NSLOT * TILE_E], u,
             NSLOT * TILE_E);
    // Argument echo (diagnostic).  A sweep of this probe came out flat - +0.003
    // ms for +176 MB of footprint - and inferring why from timing or from what
    // the producer left behind did not work: the readback was erratic at every
    // ntile, so it could not separate "the scalar arrived as 0" from "the loop
    // ran and the stores did not land".  So block 0 stamps the two scalars it
    // actually received into the tail of ``out``, at a position that encodes the
    // value; the host zeroes that tail, launches once, and reads the position
    // back.  It uses only APIs this kernel already uses (a bf16 scalar cannot
    // be written directly - there is no float-scalar Duplicate for bf16), and
    // the position is taken mod 16 so a garbage scalar cannot write out of
    // bounds.  ``out`` must be OUT_SLOTS * NSLOT * TILE_E + 512 elements.
    if (bid == 0) {
        const uint64_t echo = static_cast<uint64_t>(OUT_SLOTS) * NSLOT * TILE_E;
        const int32_t nt = ((ntile % 16) + 16) % 16;
        const int32_t md = ((mode % 16) + 16) % 16;
        DataCopy(out[echo + static_cast<uint64_t>(nt) * 16], u, 16);
        DataCopy(out[echo + 256 + static_cast<uint64_t>(md) * 16], u, 16);
    }
}
