# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ratio=128 CSA fused-compressor numerics contract gate (see docs/fe-oss-apis/csa.md).

Runs the contract over the full validation envelope — 15 union cases (coff {1, 2} x
head_dim {128, 512} x {8k / 3x8k / ragged / edge / 64k / 128k token packs}; the 128k
d=512 packs select the large-bucket schedules), 1 scaled-input case and 4 padding
cases (these 20 finite-intermediate cases each run all three gates), plus 1
overflow-intermediate case (fp32 score+ape == +Inf: determinism + eager NaN-pattern
faithfulness, fp64 comparator explicitly skipped) — 21 cases — and one ratio=4/coff=2
reference cross-check. The case list is asserted to select every shipped
(config, schedule) kernel, forward and backward, through the dispatch tables
themselves.

Per case, three gates:

  GATE 1 (deterministic; all supported inputs): 3 runs per direction (zero-initialized
    + 2 NaN-prefilled output buffers): outputs bitwise-identical across runs — an
    unwritten slot would disagree between the zero-init and prefilled runs, so the
    pair proves kernel-side writes cover every slot — and no NaN survives on the 20
    finite-intermediate cases (their fp32 reference is NaN-free; the overflow case
    instead replays its NaN pattern bit-stable on integer views). dAPE is exempt
    (fp32 atomics); its replay delta is recorded.
  GATE 2 (faithful to the fp32-intermediate eager reference; tolerances calibrated on
    the DOCUMENTED INPUT DISTRIBUTION below): forward out and backward dKV/dScore:
    n_diff <= max(1, 0.1% numel) AND max_abs <= 1.6e-2; dAPE max_abs <= 1e-3. On the
    overflow-intermediate case the same faithfulness is checked as non-finite
    propagation instead: the fused NaN mask must EQUAL the eager reference's on all
    four outputs (both sides compute in fp32, so the fp32 overflow poisons both
    alike).
  GATE 3 (fp64-oracle parity; inputs whose fp32 intermediates stay finite): for the
    deterministic outputs (out/dKV/dScore), per tensor,
    max|fused - fp64| <= max|eager_fp32 - fp64| * (1 + 1e-6) + 1e-4. For dAPE the
    same check is gated on the unit-scale cases only: dAPE is an fp32 atomic
    accumulation whose reduction-order error fluctuates run to run AROUND the eager
    error (observed within ~1.5x either way), so on scaled inputs — where the 1e-4
    absolute term no longer dominates — its parity outcome is a coin flip and is
    recorded instead of gated. On the overflow-intermediate case the comparator is
    SKIPPED with the reason recorded: the fp64 oracle stays finite where fp32
    overflows (asserted by the case), so a NaN-vs-finite distance gates nothing.

GATE 2's absolute thresholds and every published worst-observed number are properties
of the gate input distribution, NOT of all supported inputs: bf16's grid is relative,
so absolute deviations scale with the input magnitudes (multiplying kv/grad_out by
2^k scales every out/dKV/dScore deviation exactly 2^k). The gate distribution is
    kv ~ N(0, 1) bf16, score ~ N(0, 1.5^2) bf16, ape ~ N(0, 0.25^2) fp32,
    grad_out ~ N(0, 1) bf16 (seeds 1234 / 7; padding cases 11 / 13).
The scaled-input case (kv and grad_out x64, an exact bf16 exponent shift) turns the
scale dependence into committed evidence: differing-element counts and fp64 parity
of the deterministic outputs are invariant, absolute deviations scale exactly (its
recorded max_abs values exceed the unit-scale thresholds by design, so GATE 2 checks
scale-adjusted bounds there: out x64, dKV x64, dScore/dAPE x4096).

The overflow-intermediate case commits the fp32-boundary side of the contract:
kv = 1, score = bf16 max, ape = fp32 max on one 128-token c1d128 segment drives the
fp32 score + ape add to +Inf, so every output of the kernels AND of the fp32 eager
reference is NaN while the fp64 oracle stays finite (out == 1.0, asserted). The case
gates bit-stable replay of the NaN pattern and NaN-mask equality vs eager on all four
outputs; the non-finite clause of the contract is THIS committed case, not an
order-independence theorem — near fp32 max the fused evaluation order can saturate
earlier than the eager one (un-normalized chunk partials; see the Numerics caveat in
docs/fe-oss-apis/csa.md).

Padding cases put static row-capacity padding (`total_comp > cu_seqlens_comp[-1]`)
AND token-capacity padding (`kv/score` rows beyond `cu_seqlens[-1]`) on a ragged
pack, and run the same three gates on the full padded shapes: the eager reference
replicates the row-0 window on capacity rows exactly like the kernels, its autograd
returns exact zeros on all never-consumed token rows, and the fused side runs with
NONZERO grad_out on the padding rows while the reference zeroes them (proving the
kernel ignores them). The token-padding gradient rows are additionally asserted to
come back exactly zero.

Also RECORDS (not gates) whether out/dKV/dScore happen to be bitwise-equal to eager
per case, so docs can state honestly which parts remain bitwise. The eager reference
is the fp32-intermediate mirror of the Megatron-LM eager pooling region, identical to
the one in test/python/fe_api/csa/test_CSA_compressor.py; it is cross-checked here
against the production ratio=4 backward (bitwise dKV/dScore) before being trusted.

Requires a CC 10.0 GPU and the ``cudnn[cutedsl]`` install. Not collected by pytest.
Run, e.g.::

    CUDA_VISIBLE_DEVICES=0 python benchmark/csa/gate_csa_compressor_r128.py --json gate.json
"""

import argparse
import json
import math
import sys

import torch

RATIO = 128
CONFIGS = {"c1d128": (1, 128), "c2d128": (2, 128), "c1d512": (1, 512), "c2d512": (2, 512)}
EDGE_PACK = [127, 8192, 0, 129, 128, 3, 515, 1024]
CASES = [
    ("c1d128", "1x8192", [8192]),
    ("c1d128", "3x8192", [8192] * 3),
    ("c1d128", "ragged3", [1023, 2048, 509]),
    ("c1d128", "edgepack", EDGE_PACK),
    ("c1d128", "1x65536", [65536]),
    ("c1d128", "1x131072", [131072]),
    ("c2d128", "1x8192", [8192]),
    ("c2d128", "edgepack", EDGE_PACK),
    ("c2d128", "1x65536", [65536]),
    ("c1d512", "1x8192", [8192]),
    ("c1d512", "1x65536", [65536]),
    ("c1d512", "1x131072", [131072]),
    ("c2d512", "1x8192", [8192]),
    ("c2d512", "1x65536", [65536]),
    ("c2d512", "1x131072", [131072]),
]
# (config, shape_name, lens, scale): kv and grad_out multiplied by `scale` (a power of
# two, so the bf16 inputs shift exponents without re-rounding). See the module
# docstring: proves determinism + fp64 parity of the deterministic outputs are
# invariant under exact exponent shifts (while the fp32 intermediates stay finite)
# and absolute deviations scale with magnitude.
SCALED_CASES = [("c2d128", "1x8192", [8192], 64.0)]
PADDING_CASES = [("c1d512", 8, 37), ("c2d512", 8, 21), ("c1d128", 5, 37), ("c2d128", 5, 21)]
ABS_TOL = 1.6e-2
DIFF_FRAC = 0.001
DAPE_TOL = 1e-3


# ---------------------------------------------------------------------------
# Eager reference (fp32-intermediate mirror of the Megatron-LM eager region;
# identical to the reference in test_CSA_compressor.py)
# ---------------------------------------------------------------------------


def _batch_of_row(cu_seqlens, total):
    n_seg = cu_seqlens.shape[0] - 1
    row_idx = torch.arange(total, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens[1:], right=True).clamp(max=max(n_seg - 1, 0))


def _overlap_transform_thd(tensor, is_first_in_seg, head_dim, fill_value=0):
    n, ratio, b_dim, _ = tensor.size()
    d = head_dim
    new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
    new_tensor[:, ratio:] = tensor[:, :, :, d:]
    prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
    prev_data[is_first_in_seg] = fill_value
    new_tensor[:, :ratio] = prev_data
    return new_tensor


def _eager_pool(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, d, coff, mode):
    device = kv.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = _batch_of_row(cu_seqlens_comp, total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets  # (total_comp, ratio)

    if mode == "fp32":
        kv = kv.float()
        score = score.float()
    elif mode == "fp64":
        kv = kv.double()
        score = score.double()
        ape = ape.double()

    kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
    score_grouped = score[gather_idx]
    score_grouped = score_grouped + ape.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        kv_grouped = _overlap_transform_thd(kv_grouped, is_first, d, fill_value=0)
        score_grouped = _overlap_transform_thd(score_grouped, is_first, d, fill_value=float("-inf"))

    if mode == "fp32":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    else:  # fp64 oracle
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float64)
        out = (kv_grouped * weights).sum(dim=1)
    return out  # (total_comp, 1, d)


def run_eager_bwd(kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, mode):
    """Forward + backward through the eager reference; returns (out, dKV, dScore, dAPE)."""
    dtype = torch.float64 if mode == "fp64" else None
    kv_l = (kv.to(dtype) if dtype else kv.clone()).requires_grad_(True)
    score_l = (score.to(dtype) if dtype else score.clone()).requires_grad_(True)
    ape_l = (ape.to(dtype) if dtype else ape.clone()).requires_grad_(True)
    out = _eager_pool(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff, mode)
    out.backward(go.to(out.dtype))
    torch.cuda.synchronize()
    return out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach()


def _make_inputs(lens, d, ratio, coff, seed=1234):
    total = sum(lens)
    w = coff * d
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, 1, w, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    score = (torch.randn(total, 1, w, generator=gen, dtype=torch.float32).mul_(1.5)).to(torch.bfloat16)
    ape = torch.randn(ratio, w, generator=gen, dtype=torch.float32).mul_(0.25)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0] + list(seg_comp.cumsum(0)), dtype=torch.int32, device="cuda")
    total_comp = int(cuc[-1].item())
    return kv.cuda(), score.cuda(), ape.cuda(), cu, cuc, total_comp


def _make_go(total_comp, d, seed=7):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(total_comp, 1, d, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def tol_check(fused, eager, tol=ABS_TOL):
    diff = (fused.float() - eager.float()).abs()
    n_diff = int((diff > 0).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    ok = n_diff <= max(1, int(DIFF_FRAC * fused.numel())) and max_abs <= tol
    return ok, n_diff, max_abs


def oracle_check(fused, eager, oracle):
    err_f = float((fused.double() - oracle.double()).abs().max().item())
    err_e = float((eager.double() - oracle.double()).abs().max().item())
    return err_f <= err_e * (1 + 1e-6) + 1e-4, err_f, err_e


def _case_schedules(M, d, coff, nb_total):
    """The (fwd, bwd) schedules the dispatch tables select for this case."""
    return M._fwd_schedule_r128(RATIO, d, coff, nb_total), M._bwd_schedule_r128(RATIO, d, coff, nb_total)


def _shipped_schedules(M):
    """Every (config, schedule) pair the dispatch tables can select, per direction —
    derived from the tables' own boundaries so the coverage assertion tracks edits."""
    fwd, bwd = set(), set()
    for coff, d in CONFIGS.values():
        for nb in (1, M._SMALL_NB_MAX, M._SMALL_NB_MAX + 1, M._BWD_SMALL_NB_MAX, M._BWD_SMALL_NB_MAX + 1, M._LARGE_NB_MIN - 1, M._LARGE_NB_MIN):
            fwd.add(((coff, d), M._fwd_schedule_r128(RATIO, d, coff, nb)))
            bwd.add(((coff, d), M._bwd_schedule_r128(RATIO, d, coff, nb)))
    return fwd, bwd


def gate_case(M, config, shape_name, lens, rec_list, covered, scale=1.0):
    coff, d = CONFIGS[config]
    kv, score, ape, cu, cuc, tc = _make_inputs(lens, d, RATIO, coff)
    go = _make_go(tc, d)
    scaled = scale != 1.0
    if scaled:
        assert math.log2(scale).is_integer(), "scale must be a power of two (exact bf16 exponent shift)"
        kv = (kv.float() * scale).to(torch.bfloat16)
        go = (go.float() * scale).to(torch.bfloat16)
    total = kv.shape[0]
    w = coff * d
    kvf, scf, gof = kv.view(total, w), score.view(total, w), go.view(tc, d)
    sched_fwd, sched_bwd = _case_schedules(M, d, coff, tc)
    covered["fwd"].add(((coff, d), sched_fwd))
    covered["bwd"].add(((coff, d), sched_bwd))

    # ---------- forward: 3 runs (zero-init, then 2 NaN-prefilled) ----------
    out = torch.empty(tc, d, dtype=torch.bfloat16, device="cuda")
    outs = []
    for rep in range(3):
        if rep == 0:
            out.zero_()
        else:
            out.fill_(float("nan"))
        M.run_fwd_r128(kvf, scf, ape, cu, cuc, out, tc, RATIO, d, coff)
        torch.cuda.synchronize()
        outs.append(out.clone())
    fwd_stable = all(torch.equal(outs[0], o) for o in outs[1:])
    fwd_nan_ok = not any(torch.isnan(o).any().item() for o in outs)
    ref32 = _eager_pool(kv, score, ape, cu, cuc, tc, RATIO, d, coff, "fp32").view(tc, d)
    ref64 = _eager_pool(kv, score, ape, cu, cuc, tc, RATIO, d, coff, "fp64").view(tc, d)
    fwd_tol_ok, fwd_nd, fwd_ma = tol_check(outs[0], ref32, ABS_TOL * scale)
    fwd_or_ok, fwd_errf, fwd_erre = oracle_check(outs[0], ref32.double(), ref64)
    fwd_bitwise = int((outs[0].float() - ref32.float()).abs().max().item() == 0)

    # ---------- backward: 3 runs (zero-init, then 2 NaN-prefilled) ----------
    runs = []
    for rep in range(3):
        gkv = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        gs = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        if rep == 0:
            gkv.zero_()
            gs.zero_()
        else:
            gkv.fill_(float("nan"))
            gs.fill_(float("nan"))
        gape = torch.zeros_like(ape)
        M.run_bwd_r128(kvf, scf, ape, cu, cuc, gof, gkv, gs, gape, tc, RATIO, d, coff)
        torch.cuda.synchronize()
        runs.append((gkv, gs, gape))
    bwd_stable = all(torch.equal(runs[0][0], r[0]) and torch.equal(runs[0][1], r[1]) for r in runs[1:])
    bwd_nan_ok = not any(torch.isnan(r[0]).any().item() or torch.isnan(r[1]).any().item() for r in runs)
    ape_replay = max(float((runs[0][2] - r[2]).abs().max().item()) for r in runs[1:])
    gkv, gs, gape = runs[0]

    _, ekv, es, eape = run_eager_bwd(kv, score, ape, cu, cuc, tc, RATIO, d, coff, go, mode="fp32")
    _, okv, os_, oape = run_eager_bwd(kv, score, ape, cu, cuc, tc, RATIO, d, coff, go, mode="fp64")
    # Scaled inputs scale the absolute deviations of each output by its input factors
    # (kv enters out linearly; grad_out enters dKV linearly; both enter dScore/dAPE).
    kv_tol_ok, kv_nd, kv_ma = tol_check(gkv.view_as(ekv), ekv, ABS_TOL * scale)
    s_tol_ok, s_nd, s_ma = tol_check(gs.view_as(es), es, ABS_TOL * scale * scale)
    ape_ma = float((gape - eape).abs().max().item())
    ape_tol_ok = ape_ma <= DAPE_TOL * scale * scale
    kv_or_ok, kv_errf, kv_erre = oracle_check(gkv.view_as(ekv), ekv, okv)
    s_or_ok, s_errf, s_erre = oracle_check(gs.view_as(es), es, os_)
    ape_or_ok, ape_errf, ape_erre = oracle_check(gape, eape, oape)
    kv_bitwise = int(torch.equal(gkv.view_as(ekv), ekv))
    s_bitwise = int(torch.equal(gs.view_as(es), es))

    g1 = fwd_stable and fwd_nan_ok and bwd_stable and bwd_nan_ok
    g2 = fwd_tol_ok and kv_tol_ok and s_tol_ok and ape_tol_ok
    # dAPE parity is gated at unit scale only (recorded when scaled; see module docstring).
    g3 = fwd_or_ok and kv_or_ok and s_or_ok and (ape_or_ok or scaled)
    ok = g1 and g2 and g3
    rec_list.append(
        dict(
            config=config,
            shape=shape_name,
            scale=scale,
            sched_fwd=list(sched_fwd),
            sched_bwd=list(sched_bwd),
            ok=ok,
            g1_deterministic=g1,
            g2_tolerance=g2,
            g3_oracle=g3,
            fwd=dict(
                stable=fwd_stable,
                nan_ok=fwd_nan_ok,
                n_diff=fwd_nd,
                numel=outs[0].numel(),
                max_abs=fwd_ma,
                bitwise_vs_eager=fwd_bitwise,
                err_fused=fwd_errf,
                err_eager=fwd_erre,
            ),
            dkv=dict(n_diff=kv_nd, numel=int(ekv.numel()), max_abs=kv_ma, bitwise_vs_eager=kv_bitwise, err_fused=kv_errf, err_eager=kv_erre),
            dscore=dict(n_diff=s_nd, max_abs=s_ma, bitwise_vs_eager=s_bitwise, err_fused=s_errf, err_eager=s_erre),
            dape=dict(max_abs=ape_ma, replay_delta=ape_replay, parity_gated=not scaled, parity_ok=bool(ape_or_ok), err_fused=ape_errf, err_eager=ape_erre),
            bwd=dict(stable=bwd_stable, nan_ok=bwd_nan_ok),
        )
    )
    label = f"{shape_name}*{scale:g}" if scaled else shape_name
    print(
        f"  {config:8s} {label:10s} {'PASS' if ok else 'FAIL'}  "
        f"g1={'Y' if g1 else 'N'} g2={'Y' if g2 else 'N'} g3={'Y' if g3 else 'N'} | "
        f"fwd nd={fwd_nd}/{outs[0].numel()} ma={fwd_ma:.2e}{' BITW' if fwd_bitwise else ''} | "
        f"dKV nd={kv_nd} ma={kv_ma:.2e}{' BITW' if kv_bitwise else ''} | "
        f"dS nd={s_nd} ma={s_ma:.2e}{' BITW' if s_bitwise else ''} | dAPE {ape_ma:.1e}",
        flush=True,
    )
    return ok


def gate_padding(M, config, pad, tok_pad, rec_list, covered):
    """Static row-capacity padding (+pad rows, incoming grads ignored) + token-capacity
    padding (+tok_pad kv/score rows, gradients exactly zero), all three gates on the
    full padded shapes. The eager reference models the capacity rows (row-0 window
    replication) and runs with the padding-row grad_out zeroed — the fused side keeps
    it NONZERO, so matching it proves the kernel ignores padding-row gradients."""
    coff, d = CONFIGS[config]
    lens = [1023, 2048, 509]
    kv, score, ape, cu, cuc, total_true = _make_inputs(lens, d, RATIO, coff)
    total_comp = total_true + pad
    go = _make_go(total_comp, d, seed=11)
    go_zero_pad = go.clone()
    go_zero_pad[total_true:] = 0
    gen = torch.Generator(device="cpu").manual_seed(13)
    w = coff * d
    kv2 = torch.cat([kv.view(kv.shape[0], -1), torch.randn(tok_pad, w, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()])
    score2 = torch.cat([score.view(score.shape[0], -1), torch.randn(tok_pad, w, generator=gen, dtype=torch.float32).to(torch.bfloat16).cuda()])
    total = kv2.shape[0]
    gof = go.view(total_comp, d)
    sched_fwd, sched_bwd = _case_schedules(M, d, coff, total_comp)
    covered["fwd"].add(((coff, d), sched_fwd))
    covered["bwd"].add(((coff, d), sched_bwd))

    def run_dir(poison):
        out = torch.empty(total_comp, d, dtype=torch.bfloat16, device="cuda")
        gkv = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        gs = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        for t in (out, gkv, gs):
            if poison:
                t.fill_(float("nan"))
            else:
                t.zero_()
        gape = torch.zeros_like(ape)
        M.run_fwd_r128(kv2, score2, ape, cu, cuc, out, total_comp, RATIO, d, coff)
        M.run_bwd_r128(kv2, score2, ape, cu, cuc, gof, gkv, gs, gape, total_comp, RATIO, d, coff)
        torch.cuda.synchronize()
        return out, gkv, gs, gape

    a = run_dir(False)
    b = run_dir(True)
    c = run_dir(True)
    stable = all(torch.equal(a[i], b[i]) and torch.equal(a[i], c[i]) for i in range(3))
    nan_ok = not any(torch.isnan(t).any().item() for r in (a, b, c) for t in r[:3])
    ape_replay = max(float((a[3] - r[3]).abs().max().item()) for r in (b, c))
    out, gkv, gs, gape = a

    # Eager reference at full padded shape (kv/score with the token-padding rows,
    # total_comp with the capacity rows, padding-row grad_out zeroed).
    kv_full = kv2.view(total, 1, w)
    score_full = score2.view(total, 1, w)
    ref32 = _eager_pool(kv_full, score_full, ape, cu, cuc, total_comp, RATIO, d, coff, "fp32").view(total_comp, d)
    ref64 = _eager_pool(kv_full, score_full, ape, cu, cuc, total_comp, RATIO, d, coff, "fp64").view(total_comp, d)
    _, ekv, es, eape = run_eager_bwd(kv_full, score_full, ape, cu, cuc, total_comp, RATIO, d, coff, go_zero_pad, mode="fp32")
    _, okv, os_, oape = run_eager_bwd(kv_full, score_full, ape, cu, cuc, total_comp, RATIO, d, coff, go_zero_pad, mode="fp64")

    fwd_tol_ok, fwd_nd, fwd_ma = tol_check(out, ref32)
    kv_tol_ok, kv_nd, kv_ma = tol_check(gkv.view_as(ekv), ekv)
    s_tol_ok, s_nd, s_ma = tol_check(gs.view_as(es), es)
    ape_ma = float((gape - eape).abs().max().item())
    fwd_or_ok, fwd_errf, fwd_erre = oracle_check(out, ref32.double(), ref64)
    kv_or_ok, kv_errf, kv_erre = oracle_check(gkv.view_as(ekv), ekv, okv)
    s_or_ok, s_errf, s_erre = oracle_check(gs.view_as(es), es, os_)
    ape_or_ok, ape_errf, ape_erre = oracle_check(gape, eape, oape)
    pad_zero_ok = bool((gkv[kv.shape[0] :] == 0).all().item() and (gs[kv.shape[0] :] == 0).all().item())

    g1 = stable and nan_ok
    g2 = fwd_tol_ok and kv_tol_ok and s_tol_ok and ape_ma <= DAPE_TOL
    g3 = fwd_or_ok and kv_or_ok and s_or_ok and ape_or_ok
    ok = g1 and g2 and g3 and pad_zero_ok
    rec_list.append(
        dict(
            kind="padding",
            config=config,
            pad=pad,
            tok_pad=tok_pad,
            sched_fwd=list(sched_fwd),
            sched_bwd=list(sched_bwd),
            ok=ok,
            g1_deterministic=g1,
            g2_tolerance=g2,
            g3_oracle=g3,
            fwd=dict(stable=stable, nan_ok=nan_ok, n_diff=fwd_nd, numel=out.numel(), max_abs=fwd_ma, err_fused=fwd_errf, err_eager=fwd_erre),
            dkv=dict(n_diff=kv_nd, numel=int(ekv.numel()), max_abs=kv_ma, err_fused=kv_errf, err_eager=kv_erre),
            dscore=dict(n_diff=s_nd, max_abs=s_ma, err_fused=s_errf, err_eager=s_erre),
            dape=dict(max_abs=ape_ma, replay_delta=ape_replay, parity_gated=True, parity_ok=bool(ape_or_ok), err_fused=ape_errf, err_eager=ape_erre),
            pad_zero_ok=pad_zero_ok,
        )
    )
    print(
        f"  padding {config} +{pad} rows +{tok_pad} tokens: {'PASS' if ok else 'FAIL'} "
        f"g1={'Y' if g1 else 'N'} g2={'Y' if g2 else 'N'} g3={'Y' if g3 else 'N'} | "
        f"fwd nd={fwd_nd} ma={fwd_ma:.1e} | dKV nd={kv_nd} ma={kv_ma:.1e} | dS nd={s_nd} ma={s_ma:.1e} | "
        f"dAPE {ape_ma:.1e} padzero={pad_zero_ok}",
        flush=True,
    )
    return ok


def gate_overflow(M, rec_list, covered):
    """Overflow-intermediate case (see the module docstring): kv = 1, score = bf16 max,
    ape = fp32 max on one 128-token c1d128 segment. The fp32 ``score + ape`` add is
    +Inf, so the kernels and the fp32 eager reference return all-NaN outputs alike
    (both compute in fp32) while the fp64 oracle stays finite (out == 1.0): GATE 2
    becomes NaN-mask equality vs eager on all four outputs, GATE 3 is skipped with the
    reason recorded (oracle finiteness asserted so the reason stays honest), and
    GATE 1's bitwise replay compares int16 bit views (torch.equal is false on NaN);
    the zero-init/NaN-prefill pair still proves kernel-side writes cover every slot,
    because an unwritten slot would disagree between the zero-init and prefilled
    runs."""
    config = "c1d128"
    coff, d = CONFIGS[config]
    total, w = 128, coff * d
    tc = total // RATIO
    kv = torch.full((total, 1, w), 1.0, dtype=torch.float32).to(torch.bfloat16).cuda()
    score = torch.full((total, 1, w), torch.finfo(torch.bfloat16).max, dtype=torch.float32).to(torch.bfloat16).cuda()
    ape = torch.full((RATIO, w), torch.finfo(torch.float32).max, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    cuc = torch.tensor([0, tc], dtype=torch.int32, device="cuda")
    go = torch.full((tc, 1, d), 1.0, dtype=torch.float32).to(torch.bfloat16).cuda()
    kvf, scf, gof = kv.view(total, w), score.view(total, w), go.view(tc, d)
    sched_fwd, sched_bwd = _case_schedules(M, d, coff, tc)
    covered["fwd"].add(((coff, d), sched_fwd))
    covered["bwd"].add(((coff, d), sched_bwd))

    runs = []
    for rep in range(3):
        out = torch.empty(tc, d, dtype=torch.bfloat16, device="cuda")
        gkv = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        gs = torch.empty(total, w, dtype=torch.bfloat16, device="cuda")
        for t in (out, gkv, gs):
            if rep == 0:
                t.zero_()
            else:
                t.fill_(float("nan"))
        gape = torch.zeros_like(ape)
        M.run_fwd_r128(kvf, scf, ape, cu, cuc, out, tc, RATIO, d, coff)
        M.run_bwd_r128(kvf, scf, ape, cu, cuc, gof, gkv, gs, gape, tc, RATIO, d, coff)
        torch.cuda.synchronize()
        runs.append((out, gkv, gs, gape))
    out, gkv, gs, gape = runs[0]
    # dAPE (fp32 atomics) is exempt from the bitwise replay as everywhere else; its
    # NaN mask is order-independent (NaN is absorbing under addition) and checked below.
    stable = all(all(torch.equal(runs[0][i].view(torch.int16), r[i].view(torch.int16)) for i in range(3)) for r in runs[1:])

    ref32 = _eager_pool(kv, score, ape, cu, cuc, tc, RATIO, d, coff, "fp32").view(tc, d)
    ref64 = _eager_pool(kv, score, ape, cu, cuc, tc, RATIO, d, coff, "fp64").view(tc, d)
    _, ekv, es, eape = run_eager_bwd(kv, score, ape, cu, cuc, tc, RATIO, d, coff, go, mode="fp32")
    _, okv, os_, oape = run_eager_bwd(kv, score, ape, cu, cuc, tc, RATIO, d, coff, go, mode="fp64")

    pairs = (("out", out, ref32), ("dkv", gkv.view_as(ekv), ekv), ("dscore", gs.view_as(es), es), ("dape", gape, eape))
    masks_equal = {name: bool((torch.isnan(f) == torch.isnan(e)).all().item()) for name, f, e in pairs}
    # This input poisons every slot (every window sees the +Inf add); asserting the
    # eager masks are FULL pins the case so it cannot silently hollow out.
    masks_full = {name: bool(torch.isnan(e).all().item()) for name, _, e in pairs}
    # GATE 3 skip: assert the reason (fp64 does not overflow on these inputs), then
    # record it instead of running the meaningless NaN-vs-finite comparison.
    oracle_finite = all(bool(torch.isfinite(t).all().item()) for t in (ref64, okv, os_, oape))
    skip_reason = "fp32 intermediates overflow while the fp64 oracle stays finite (asserted): a NaN-vs-finite distance gates nothing"

    g1 = stable
    g2 = all(masks_equal.values()) and all(masks_full.values())
    ok = g1 and g2 and oracle_finite
    rec_list.append(
        dict(
            kind="overflow",
            config=config,
            shape="1x128ovf",
            sched_fwd=list(sched_fwd),
            sched_bwd=list(sched_bwd),
            ok=ok,
            g1_deterministic=g1,
            g2_eager_nan_pattern=g2,
            g3_oracle_skipped=skip_reason,
            oracle_finite=oracle_finite,
            nan_numel={name: [int(torch.isnan(f).sum().item()), int(f.numel())] for name, f, _ in pairs},
        )
    )
    print(
        f"  {config:8s} {'1x128ovf':10s} {'PASS' if ok else 'FAIL'}  "
        f"g1={'Y' if g1 else 'N'} g2={'Y' if g2 else 'N'} g3=SKIP(oracle finite: {'Y' if oracle_finite else 'N'}) | "
        f"NaN out {int(torch.isnan(out).sum())}/{out.numel()} dKV {int(torch.isnan(gkv).sum())}/{gkv.numel()} "
        f"dS {int(torch.isnan(gs).sum())}/{gs.numel()} dAPE {int(torch.isnan(gape).sum())}/{gape.numel()} == eager",
        flush=True,
    )
    return ok


def validate_reference():
    """One ratio=4/coff=2 cross-check: the production backward is bitwise against the
    eager reference above, proving the reference before the r128 cases trust it."""
    from cudnn.csa.compressor import compressor_sm100 as K4

    kv, score, ape, cu, cuc, tc = _make_inputs([2048], 128, 4, 2)
    go = _make_go(tc, 128)
    total = kv.shape[0]
    kvf, scf, gof = kv.view(total, 256), score.view(total, 256), go.view(tc, 128)
    gkv = torch.zeros_like(kvf)
    gs = torch.zeros_like(scf)
    gape = torch.zeros_like(ape)
    K4.run_bwd(kvf, scf, ape, cu, cuc, gof, gkv, gs, gape, tc, 4, 128, 2)
    torch.cuda.synchronize()
    _, ekv, es, eape = run_eager_bwd(kv, score, ape, cu, cuc, tc, 4, 128, 2, go, mode="fp32")
    okv = torch.equal(gkv.view_as(ekv), ekv)
    os_ = torch.equal(gs.view_as(es), es)
    oape = float((gape - eape).abs().max().item())
    print(f"  reference validation @ r4/c2: dKV bitwise={okv} dScore bitwise={os_} dAPE {oape:.1e}", flush=True)
    return okv and os_ and oape <= DAPE_TOL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="run only the first two union cases (smoke)")
    ap.add_argument("--json", default=None, help="write the per-case records to this path")
    args = ap.parse_args()
    from cudnn.csa.compressor import compressor_sm100_r128 as M

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0), "requires a CC 10.0 GPU"
    print(f"r128 contract gate on {torch.cuda.get_device_name()}", flush=True)
    recs = []
    covered = {"fwd": set(), "bwd": set()}
    ok = validate_reference()
    for config, shape_name, lens in CASES if not args.quick else CASES[:2]:
        ok = gate_case(M, config, shape_name, lens, recs, covered) and ok
    if not args.quick:
        for config, shape_name, lens, scale in SCALED_CASES:
            ok = gate_case(M, config, shape_name, lens, recs, covered, scale=scale) and ok
        for config, pad, tok_pad in PADDING_CASES:
            ok = gate_padding(M, config, pad, tok_pad, recs, covered) and ok
        ok = gate_overflow(M, recs, covered) and ok
        # Every shipped (config, schedule) kernel must have been selected by >= 1 case
        # (derived from the dispatch tables, so adding a bucket fails the gate until a
        # case covers it).
        ship_fwd, ship_bwd = _shipped_schedules(M)
        missing = [("fwd", *m) for m in sorted(ship_fwd - covered["fwd"])] + [("bwd", *m) for m in sorted(ship_bwd - covered["bwd"])]
        print(
            f"  schedule coverage: fwd {len(covered['fwd'])}/{len(ship_fwd)} bwd {len(covered['bwd'])}/{len(ship_bwd)}"
            + (f" MISSING {missing}" if missing else ""),
            flush=True,
        )
        ok = ok and not missing
    n_pass = sum(1 for r in recs if r["ok"])
    print(f"GATE {'PASS' if ok else 'FAIL'} {n_pass}/{len(recs)}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(all_pass=ok, cases=recs), f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
