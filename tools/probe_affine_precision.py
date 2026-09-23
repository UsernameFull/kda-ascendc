"""Route-2 gate 1: does the dense-affine recurrence survive bf16 at all?

The second route of the 2026-09-22 redesign rewrites the chunk recurrence as

    E = diag(d) - Kg^T W      F = Kg^T U
    P = Q - A W               R = A U
    Snext = E S + F           O   = P S + R

which is the *same algebraic map* as the shipped

    Z = U - W S   O = Q S + A Z   Snext = diag(d) S + Kg^T Z

but not the same *bf16* map: the shipped chain multiplies the state by an exact
fp32 decay and adds an fp32-accumulated correction, while the affine chain's
whole transition - decay included - has to become a bf16 operand for the Cube's
Mmad.  bf16 has 8 significand bits, so a diagonal entry near 1 carries ~2^-9 of
rounding error, and a weak gate (decay ~ 1) turns that into a systematic drift
over the sequence.  That is a numerical question, not a kernel question, and it
is answerable in torch before anything is built - which is the order the plan
asks for ("先过长链精度再测完整成本").

This tool runs the chunk math once in fp32 (the reference) and then re-runs it
under four rounding disciplines that all consume the *same* fp32 intermediates:

  fp32        no intermediate rounding (the reference itself)
  shipped     the kernels' discipline: bf16 W/U/Aqk/Qg/kg, bf16 S16 snapshot,
              bf16 d1, bf16 Z, exact fp32 decay, fp32-accumulated state
  affine      E/P rounded to bf16 (the Cube operand), F/R fp32
  affine_dec  same, but the decay stays *outside* the rounded matrix
              (E_off = -Kg^T W in bf16, Snext = diag(d) S + E_off S + F)

and reports, at several checkpoints, the relative error of the state and of the
output against the fp32 reference.  The gate is the one the plan states: the
affine variants have to stay inside the shipped chain's error band, on weak
decay, long sequences and a non-zero initial state.

  ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_affine_precision.py
  ... --shapes weak --chunk 64 --t 8192 --h 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
D = 128
RCP_LN2 = 1.44269504089


def gate(q, k, v, g, beta, a_log, bias, lower_bound):
    """q/k/v/g/beta -> the chunk-side operands, fp32, as the kernels build them."""
    eps = 1e-6
    qn = q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + eps)
    kn = k.float() / torch.sqrt((k.float() ** 2).sum(-1, keepdim=True) + eps)
    beta_s = beta.float().sigmoid()
    if a_log is not None:
        gi = g.float() + bias[None, None, :, :]
        gate_ = lower_bound * torch.sigmoid(torch.exp(a_log)[None, None, :, None] * gi) * RCP_LN2
    else:
        gate_ = lower_bound * torch.sigmoid(g.float()) * RCP_LN2
    return qn, kn, beta_s, gate_


def chunk_operands(qn, kn, v, beta_s, gc, scale):
    """[BH, C, D] operands of one chunk, fp32: A_inv, W, U, Aqk, Kg, decay, Qg."""
    C = qn.shape[1]
    mid = C // 2
    g_rec = gc - gc[:, mid:mid + 1]                      # [BH, C, D]
    e_pos = torch.exp2(g_rec)
    k_pos = kn * e_pos
    q_pos = qn * e_pos
    aqk = torch.tril(torch.bmm(q_pos, (kn * torch.exp2(-g_rec)).transpose(1, 2)) * scale)
    akk = torch.tril(torch.bmm(k_pos, (kn * torch.exp2(-g_rec)).transpose(1, 2))
                     * beta_s[:, :, None], diagonal=-1)
    bh, n = akk.shape[0], C
    inv = torch.eye(n, device=akk.device, dtype=torch.float32).expand(bh, n, n).clone()
    for i in range(1, n):
        inv[:, i, :] = -torch.bmm(akk[:, i:i + 1, :i], inv[:, :i, :])[:, 0, :]
        inv[:, i, i] += 1.0
    k_w = kn * beta_s[:, :, None] * torch.exp2(gc)
    v_w = v.float() * beta_s[:, :, None]
    w = torch.bmm(inv, k_w)
    u = torch.bmm(inv, v_w)
    kg = kn * torch.exp2(gc[:, -1:, :] - gc)              # [BH, C, D]
    qg = q_pos
    dec = torch.exp2(gc[:, -1, :])                       # [BH, D]
    return w, u, aqk, kg, qg, dec


def bf(x):
    return x.to(torch.bfloat16).float()


def run(q, k, v, g, beta, a_log, bias, lower_bound, scale, chunk, mode,
        initial_state=None, checkpoints=()):
    """One formulation's chain.  Returns out, final_state, {chunk: state rel err}."""
    B, T, H, Dd = q.shape
    nt = T // chunk
    qn, kn, beta_s, gate_ = gate(q, k, v, g, beta, a_log, bias, lower_bound)
    qn = qn.reshape(B * H, T, Dd)
    kn = kn.reshape(B * H, T, Dd)
    vv = v.float().reshape(B * H, T, Dd)
    bb = beta_s.reshape(B * H, T)
    gt = gate_.reshape(B * H, T, Dd)
    dev = q.device
    state = torch.zeros(B * H, Dd, Dd, dtype=torch.float32, device=dev)
    if initial_state is not None:
        state = initial_state.float().reshape(B * H, Dd, Dd).clone()
    out = torch.zeros(B * H, T, Dd, dtype=torch.float32, device=dev)
    trace = {}
    ref_state = None
    for ic in range(nt):
        sl = slice(ic * chunk, (ic + 1) * chunk)
        gc = torch.cumsum(gt[:, sl], dim=1)
        w, u, aqk, kg, qg, dec = chunk_operands(qn[:, sl], kn[:, sl], vv[:, sl],
                                                bb[:, sl], gc, scale)
        if mode == "fp32":
            w_, u_, a_ = w, u, aqk
            kg_, qg_ = kg, qg
            s16 = state
            z = u_ - torch.bmm(w_, s16.transpose(1, 2))
            o = (bf(torch.bmm(qg_, s16.transpose(1, 2))) * scale
                 + bf(torch.bmm(a_, z)))
            state = state * dec[:, None, :] + torch.bmm(z.transpose(1, 2), kg_)
        elif mode == "shipped":
            w_, u_, a_, kg_, qg_ = bf(w), bf(u), bf(aqk), bf(kg), bf(qg)
            s16 = bf(state)
            d1 = bf(torch.bmm(w_, s16.transpose(1, 2)))
            z = bf(u_ - d1)
            # the fixpipe rounds both of the output Mmads to bf16 before the
            # vector side accumulates them, exactly as the fused K2 does
            d2 = bf(torch.bmm(qg_, s16.transpose(1, 2)))
            d3 = bf(torch.bmm(a_, z))
            o = d2 * scale + d3
            state = state * dec[:, None, :] + torch.bmm(z.transpose(1, 2), kg_)
        else:
            kg_, qg_, a_, u_, w_ = bf(kg), bf(qg), bf(aqk), bf(u), bf(w)
            # The state is stored as S^T (V-first, the kernels' layout), so
            # every affine operand is written transposed as well:
            #   E S  ->  S^T E^T ,  E^T = diag(d) - W^T Kg
            #   F    ->  F^T = U^T Kg ,  F = Kg^T U
            #   P S  ->  P @ S reads the transposed state directly
            ee = -torch.bmm(w_.transpose(1, 2), kg_)      # [BH, K, K] = E^T
            ff = torch.bmm(u_.transpose(1, 2), kg_)       # [BH, V, K] = F^T
            # P = scale*Q - A W: the q half of the output carries the scale
            # (Aqk already does), and a scale folded into P is what lets the
            # Cube accumulate Q*scale and -A W*q in one L0C tile.
            pp = qg_ * scale - torch.bmm(a_, w_)          # [BH, C, K] = P
            rr = torch.bmm(a_, u_)                        # [BH, C, V] = R
            s16 = bf(state)
            if mode == "affine":
                e16 = bf(ee + torch.diag_embed(dec))      # both operands bf16
                state = torch.bmm(s16, e16) + ff
            elif mode == "affine_dec":
                state = (state * dec[:, None, :]
                         + torch.bmm(s16, bf(ee)) + ff)
            elif mode == "affine_fp32":
                state = torch.bmm(state, ee + torch.diag_embed(dec)) + ff
            else:
                raise ValueError(mode)
            o = torch.bmm(bf(pp), s16.transpose(1, 2)) + rr
        out[:, sl] = o
        if mode == "fp32" and ref_state is None and ic == nt - 1:
            ref_state = state
        if (ic + 1) in checkpoints:
            trace[ic + 1] = state
    return out, state, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--t", type=int, default=8192)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=1)
    ap.add_argument("--shapes", default="long,mid,init",
                    help="which input regimes to run")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = torch.device("npu", args.device)
    torch.npu.set_device(args.device)
    T, H, B = args.t, args.h, args.b
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def rnd(*shape, std=1.0):
        return torch.randn(*shape, generator=gen).to(dev, dtype=torch.float32) * std

    # The gate the kernels build is
    #     gate = lower_bound * sigmoid(A_log * g + bias) * ln2^-1
    # so the per-token decay 2^gate is ~1 only where A_log * g is well
    # negative: that (long memory) is the regime where a rounded transition
    # matrix can drift, and it is the one the plan's "weak decay" means.  With
    # the repo's default g = logsigmoid(randn) and A_log = randn(H) the state
    # is gone within a chunk (2^(sum gate) ~ 0), so a study that only ran that
    # would be unable to see anything.
    long_g = lambda: torch.full((B, T, H, D), -8.0, device=dev) + rnd(B, T, H, D, std=0.05)
    mid_g = lambda: torch.nn.functional.logsigmoid(rnd(B, T, H, D, std=0.35) - 2.6).clamp_min(-5.0)
    regimes = {
        "long": dict(g=long_g(), alog=torch.full((H,), 2.0, device=dev)),
        "mid": dict(g=mid_g(), alog=torch.full((H,), 2.0, device=dev)),
        "init": dict(g=long_g(), alog=torch.full((H,), 2.0, device=dev), init=True),
    }

    bias = rnd(H, D, std=0.2) * 0.1
    scale = D ** -0.5
    want = [x for x in args.shapes.split(",") if x in regimes]
    cps = sorted({max(1, (T // args.chunk) // 4),
                  max(1, (T // args.chunk) // 2),
                  max(1, 3 * (T // args.chunk) // 4), T // args.chunk})

    print("chunk=%d T=%d H=%d B=%d  dtypes: fp32 accum, bf16 where the kernels round"
          % (args.chunk, T, H, B))
    print()
    hdr = "%-8s %-12s %10s %10s %10s" % ("regime", "formulation", "state@", "out", "note")
    print(hdr)
    print("-" * len(hdr))
    for name in want:
        reg = regimes[name]
        q = rnd(B, T, H, D).to(torch.bfloat16)
        k = rnd(B, T, H, D).to(torch.bfloat16)
        v = rnd(B, T, H, D).to(torch.bfloat16)
        beta = rnd(B, T, H)
        init = None
        if reg.get("init"):
            init = rnd(B * H, D, D, std=0.05)
        g = reg["g"]
        a_log = reg["alog"]
        # effective decay over one chunk (from the reference's own gate), so the
        # table says which regime it is actually testing
        qn0, kn0, bs0, gt0 = gate(q, k, v, g, beta, a_log, bias, -5.0)
        gcc = torch.cumsum(gt0.reshape(B * H, T, D)[:, :args.chunk], dim=1)
        chunk_decay = torch.exp2(gcc[:, -1, :]).mean().item()
        common = (q, k, v, g, beta, a_log, bias, -5.0, scale, args.chunk)
        out_ref, st_ref, tr_ref = run(*common, "fp32", initial_state=init, checkpoints=cps)
        base_out = out_ref.abs().max().item()
        base_st = st_ref.abs().max().item()
        res = {}
        for mode in ("shipped", "affine", "affine_dec", "affine_fp32"):
            out, st, tr = run(*common, mode, initial_state=init, checkpoints=cps)
            res[mode] = (out, st, tr)
            eo = (out - out_ref).abs().max().item() / max(base_out, 1e-30)
            es = (st - st_ref).abs().max().item() / max(base_st, 1e-30)
            note = "upper bound (no E rounding)" if mode == "affine_fp32" else ""
            if mode != "shipped":
                d_st = (st - res["shipped"][1]).abs().max().item() / max(base_st, 1e-30)
                d_out = ((out - res["shipped"][0]).abs().max().item()
                         / max(base_out, 1e-30))
                note = ("vs shipped: %.2e / %.2e  " % (d_st, d_out)) + note
            print("%-8s %-12s %10.3e %10.3e %s" % (name, mode, es, eo, note))
        # where the divergence starts: relative state error at each checkpoint
        print("%-8s %-12s %s   (state error vs the fp32 reference, by chunk)"
              % ("", "state@chunk", "  ".join("%9d" % c for c in cps)))
        for mode in ("shipped", "affine", "affine_dec", "affine_fp32"):
            tr = res[mode][2]
            row = []
            for c in cps:
                if c in tr and c in tr_ref:
                    den = max(tr_ref[c].abs().max().item(), 1e-30)
                    row.append("%9.3e" % ((tr[c] - tr_ref[c]).abs().max().item() / den))
                else:
                    row.append("%9s" % "-")
            print("%-8s %-12s %s" % ("", mode, "  ".join(row)))
        print()
    print("gate: an affine variant only counts if its state/out error stays inside")
    print("the shipped chain's band (it may be *smaller* - the affine output path")
    print("drops two roundings - but it must not be larger).")


if __name__ == "__main__":
    main()
