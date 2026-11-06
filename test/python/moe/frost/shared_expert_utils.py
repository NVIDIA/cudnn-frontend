# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the shared-expert decode pair's test modules (``test_shared_expert_decode.py``, ``test_shared_expert_decode_determinism.py``):
the operand builders, the fp64 reference of the bf16 operands, the harness band check, the readable fp32 model of the documented reduction
order (a second reference, never a byte oracle) and the device gate.  Not a test module (no ``test_`` prefix); imported by name from the
sibling modules (pytest puts their directory on ``sys.path``)."""

import math

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.moe.frost import shared_expert_decode as entry  # noqa: E402
from cudnn.moe.frost.kernel import shared_expert_decode_config as kcfg  # noqa: E402

GEOMETRIES = [kcfg.Geometry(8192, 256), kcfg.Geometry(8192, 128)]
SM100_READING = (kcfg.SM100_SMEM_PER_SM_BYTES, kcfg.SM100_SMEM_OPTIN_BYTES)  # 233,472 / 232,448
# the harness band for a bf16 multi-stage arm against an fp32 reference (never widened): rel-L2 and the per-element budget
BAND = dict(rel_l2=2e-2, atol_rms=2**-5, rtol=2**-6, max_frac=1e-2)


def _gid(g):
    return f"h{g.HIDDEN}i{g.INTERMEDIATE}"


def make_operands(tokens=6, device="cpu", seed=0, geometry=None, x_rows=None, structured=False):
    """x [tokens, H] (optionally inside a taller row-strided slab), W_gu [2 I, H], W_d [H, I], W_g [H], bf16, std 1 / sqrt(fan_in) for the
    weights (the harness's synth); ``structured`` = sign-structured data (a few large entries per row) beside the random draws."""
    g = geometry or kcfg.DEFAULT_GEOMETRY
    H, I = g.HIDDEN, g.INTERMEDIATE
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape, std=1.0):
        return (torch.randn(*shape, generator=gen) * std).to(device)

    rows = x_rows or tokens
    x = rn(rows, H).bfloat16()
    if structured:
        x[:, ::97] *= 8.0
        x[:, 1::131] = -x[:, 1::131]
    return dict(
        x=x[:tokens],
        w_gate_up=rn(2 * I, H, std=1.0 / math.sqrt(H)).bfloat16(),
        w_down=rn(H, I, std=1.0 / math.sqrt(I)).bfloat16(),
        w_gate=rn(H, std=1.0 / math.sqrt(H)).bfloat16(),
    )


def reference(ops):
    """The op in fp64 from the bf16 operands (a hand-rolled fp32 reference is a TF32 reference): ``sigmoid(x Wg) * ((silu(x Wgate^T) *
    (x Wup^T)) Wd^T)``."""
    x = ops["x"].double()
    I = ops["w_down"].shape[1]  # noqa: E741
    h = x @ ops["w_gate_up"].double().t()
    act = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
    y = act @ ops["w_down"].double().t()
    gate = torch.sigmoid(x @ ops["w_gate"].double().reshape(-1, 1))
    return y * gate


def band_metrics(out, ref):
    """The harness's comparison: rel-L2 and the fraction of elements beyond ``2^-5 rms(ref) + 2^-6 |ref|``."""
    o, r = out.double().reshape(-1), ref.double().reshape(-1)
    diff = (o - r).abs()
    rel_l2 = float(diff.norm() / r.norm().clamp(min=1e-30))
    rms = float(r.pow(2).mean().sqrt().clamp(min=1e-30))
    frac = float((diff > BAND["atol_rms"] * rms + BAND["rtol"] * r.abs()).double().mean())
    return rel_l2, frac


def assert_in_band(out, ref, label=""):
    rel_l2, frac = band_metrics(out, ref)
    assert rel_l2 <= BAND["rel_l2"], f"{label}: rel-L2 {rel_l2:.3e} > {BAND['rel_l2']}"
    assert frac <= BAND["max_frac"], f"{label}: {frac:.3e} of the elements beyond the element budget (> {BAND['max_frac']})"


def order_model(ops, cfg):
    """The readable fp32 model of the documented reduction ORDER (the kernel module's docstring): per output element, the chain of
    warp ``w``'s (chunk, step) k16 pieces inside each K slice, the warp sequence, the slice sequence; the gate logit as the GATE_PARTS-leaf
    tree, built on the operands' device.  NOT bitwise with the tensor core (its k16 contraction order is the hardware's): a second
    reference that states the order."""
    x = ops["x"].float()
    dev = x.device
    T, H = x.shape
    I = cfg.I  # noqa: E741
    wgu, wd, wg = ops["w_gate_up"].float(), ops["w_down"].float(), ops["w_gate"].float()
    h = torch.zeros(T, 2 * I, device=dev)
    for ks in range(cfg.K_SPLIT):
        slice_sum = torch.zeros(T, 2 * I, device=dev)
        warp_parts = []
        for w in range(cfg.WARPS):
            acc = torch.zeros(T, 2 * I, device=dev)
            for chunk in range(cfg.X_CHUNKS):
                for s in range(4 * w, 4 * w + 4):
                    k0 = ks * cfg.K_SLICE + chunk * kcfg.CHUNK_K + s * kcfg.MMA_K
                    acc = acc + x[:, k0 : k0 + kcfg.MMA_K] @ wgu[:, k0 : k0 + kcfg.MMA_K].t()
            warp_parts.append(acc)
        for part in warp_parts:  # ((C_0 + C_1) + C_2) + ..
            slice_sum = slice_sum + part
        h = h + slice_sum  # ((S_0 + S_1) + ..)
    act = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).bfloat16().float()
    y = torch.zeros(T, H, device=dev)
    for s in range(cfg.K_STEPS2):
        k0 = s * kcfg.MMA_K
        y = y + act[:, k0 : k0 + kcfg.MMA_K] @ wd[:, k0 : k0 + kcfg.MMA_K].t()
    parts = torch.zeros(T, cfg.GATE_PARTS, device=dev)
    for q in range(cfg.GATE_PARTS):
        ks, c = divmod(q, cfg.CLUSTERS)
        k0 = ks * cfg.K_SLICE + c * cfg.GATE_COLS
        parts[:, q] = (x[:, k0 : k0 + cfg.GATE_COLS] * wg[k0 : k0 + cfg.GATE_COLS]).sum(dim=1)
    gate = torch.sigmoid(parts.sum(dim=1, keepdim=True))
    return (y * gate).bfloat16()


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the shared-expert decode pair serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_version_error()
    if msg is not None:
        pytest.skip(msg)
    return dev
