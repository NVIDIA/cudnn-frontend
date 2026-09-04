# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``kda_cake`` engine: FlashInfer's CAKE-generated C16 recurrent-KDA
training kernels behind ``kimi_delta_attention``.

The engine serves one contract (bf16 io and gates, K = V = 128, fused l2norm /
beta-sigmoid / safe gate with ``a_log`` + ``dt_bias`` at lower bound -5,
checkpoint cadence 16) and is opt-in (``CUDNN_FRONTEND_ENABLE_FROST_ENGINES``).
These tests pin ``plan_name="kda_cake"`` on that contract and compare against
``kda_frost`` on identical inputs.

Known state of the frozen kernels (measured against FLA 0.5.2 ``chunk_kda`` with
the same flags, bf16, FlashInfer's own input regime): the recurrent state and the
data gradients dq/dk/dv/dbeta/d_initial_state agree to ~5e-3 relative RMS, but
the forward token output ``o`` is ~0.12 relative RMS off (cos 0.993), and the
gate parameter gradients dg/d_a_log/d_dt_bias are 3-5x further off than FROST's.
The engine reproduces FlashInfer's own results bit-for-bit, so this is a property
of the kernels; the affected assertions are strict xfails so they flip loudly
when the kernels are fixed.
"""

import math
import os

os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cudnn  # noqa: E402
from cudnn.linear_attention import ops as la_ops  # noqa: E402

from .reference_kda import kda_reference  # noqa: E402

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

SEED = 4636
LB = -5.0
SCALE = 1.0 / math.sqrt(128)
CAKE = "kda_cake"
FROST = "kda_frost"
KERNEL_FORWARD_GAP = "CAKE C16 forward output is ~0.12 rel-RMS off FLA/FROST (state and data gradients agree); see module docstring"
KERNEL_PARAM_GRAD_GAP = "CAKE C16 gate-parameter gradients are 3-5x further from FLA than FROST's; see module docstring"

# (seq_lens, H, HV): aligned, grouped, partial tails, a single long sequence (split work items)
SHAPES = [
    ([256, 256], 2, 2),
    ([512, 512], 2, 4),
    ([500, 300, 224], 2, 2),
    ([4096], 4, 4),
]


def _sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _rms_ratio(out, want):
    out, want = out.double().reshape(-1), want.double().reshape(-1)
    return ((out - want).norm() / want.norm().clamp_min(1e-12)).item()


def make_inputs(seq_lens, H, HV, *, seed=SEED, device="cuda"):
    torch.manual_seed(seed)
    total, N, K = sum(seq_lens), len(seq_lens), 128
    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    q = torch.randn(total, H, K, device=device).to(torch.bfloat16)
    k = torch.randn(total, H, K, device=device).to(torch.bfloat16)
    v = torch.randn(total, HV, K, device=device).to(torch.bfloat16)
    g = (torch.randn(total, HV, K, device=device) * 0.1).to(torch.bfloat16)
    beta = torch.randn(total, HV, device=device).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, device=device) + 1.0)
    dt_bias = torch.randn(HV, K, device=device) * 0.1
    state0 = torch.randn(N, HV, K, K, device=device) * 0.02
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu=cu, a_log=a_log, dt_bias=dt_bias, state0=state0)


def run_training_step(plan_name, inputs, *, seed=SEED + 1):
    """Forward + backward through the op with ``plan_name`` pinned; returns the
    outputs and every gradient as a dict."""
    torch.manual_seed(seed)
    leaves = {name: inputs[name].detach().clone().requires_grad_(True) for name in ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "state0")}
    o, fs, ckpt = la_ops.kimi_delta_attention(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        inputs["cu"],
        initial_state=leaves["state0"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=LB,
        a_log=leaves["a_log"],
        dt_bias=leaves["dt_bias"],
        checkpoint_every_n_tokens=16,
        plan_name=plan_name,
    )
    do = (torch.randn_like(o, dtype=torch.float32) * 0.1).to(o.dtype)
    dfs = torch.randn_like(fs) * 0.02
    grads = torch.autograd.grad((o, fs), [leaves[n] for n in ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "state0")], grad_outputs=(do, dfs))
    names = ("dq", "dk", "dv", "dg", "dbeta", "da_log", "ddt_bias", "dstate0")
    return dict(o=o, fs=fs, ckpt=ckpt, **dict(zip(names, grads)))


@pytest.fixture(scope="module")
def cake_available():
    if _sm() not in (100, 103):
        pytest.skip("kda_cake serves exact SM100 / SM103 only")
    from cudnn.linear_attention.cake import compiler

    try:
        compiler.cuda_include_dirs()
    except compiler.CakeCompileError as exc:
        pytest.skip(f"NVRTC needs the CUDA headers: {exc}")
    return True


@pytest.fixture(scope="module")
def steps(cake_available):
    """One cake and one frost training step per shape, shared by the parity tests."""
    out = {}
    for seq_lens, H, HV in SHAPES:
        inputs = make_inputs(seq_lens, H, HV)
        try:
            cake = run_training_step(CAKE, inputs)
        except cudnn.cudnnGraphNotSupportedError as exc:
            pytest.skip(f"kda_cake declined: {exc}")
        out[str(seq_lens)] = (inputs, cake, run_training_step(FROST, inputs))
    return out


def test_vendored_bodies_match_sha256sums():
    """The generated kernel bodies are frozen; the checksums are the contract."""
    from cudnn.linear_attention.cake import compiler

    pinned = {}
    for line in (compiler.KERNEL_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        pinned[name] = digest
    digests = compiler.source_digests()
    for name, digest in pinned.items():
        assert digests.get(name) == digest, f"{name} differs from its pinned digest"


def test_manifest_offers_kda_cake_when_opted_in():
    from cudnn.engines import manifest

    family = next(f for f in manifest.MANIFEST if f.name == "kda")
    assert "kda_cake" in family.offered_ids()
    engines = manifest.instantiate(family, family.offered_ids())
    assert any(e.name == "kda_cake" for e in engines), "kda_cake did not instantiate (missing cuda-python?)"


def test_plan_matches_flashinfer_layout():
    """Work-item planning is pure Python; pin its shape on a mixed batch so the
    port stays aligned with the upstream metadata builder (verified equal to
    FlashInfer's ``_build_c16_metadata`` on its C16 routes)."""
    from cudnn.linear_attention.cake.kda_host import CHUNK, WORK_ITEM_FIELDS, plan_c16

    plan = plan_c16([500, 300, 224], 2, 148)
    assert plan.total_chunks == 32 + 19 + 14
    assert plan.checkpoint_starts == (0, 32, 51, 65)
    assert plan.offsets == (0, 500, 800, 1024)
    assert not plan.aligned
    assert len(plan.work_rows) == plan.total_work_items * WORK_ITEM_FIELDS
    rows = [plan.work_rows[i : i + WORK_ITEM_FIELDS] for i in range(0, len(plan.work_rows), WORK_ITEM_FIELDS)]
    spans = [row[3] - row[2] for row in rows]
    assert spans == sorted(spans, reverse=True), "work items are ordered longest first"
    for sequence, head, write_start, write_end, compute_start, compute_end, bos, eos in rows:
        assert 0 <= sequence < 3 and 0 <= head < 2
        assert 0 <= compute_start <= write_start < write_end <= compute_end
        assert (bos, eos) == (plan.offsets[sequence], plan.offsets[sequence + 1])
    aligned = plan_c16([512, 512], 4, 148)
    assert aligned.aligned and aligned.total_chunks * CHUNK == 1024


@pytest.mark.parametrize("seq_lens,H,HV", SHAPES, ids=lambda s: str(s))
def test_state_and_data_gradients_match_frost(steps, seq_lens, H, HV):
    _inputs, cake, frost = steps[str(seq_lens)]
    assert torch.isfinite(cake["o"]).all() and torch.isfinite(cake["fs"]).all()
    assert cake["ckpt"].numel() > 0 and cake["ckpt"].dtype == torch.bfloat16
    assert _rms_ratio(cake["fs"], frost["fs"]) < 2e-2, "final_state"
    for name in ("dq", "dk", "dv", "dbeta", "dstate0"):
        assert torch.isfinite(cake[name]).all(), name
        assert _rms_ratio(cake[name], frost[name]) < 4e-2, f"{name}: rms ratio {_rms_ratio(cake[name], frost[name]):.4g}"


@pytest.mark.xfail(strict=True, reason=KERNEL_FORWARD_GAP)
@pytest.mark.parametrize("seq_lens,H,HV", SHAPES, ids=lambda s: str(s))
def test_forward_output_matches_frost(steps, seq_lens, H, HV):
    _inputs, cake, frost = steps[str(seq_lens)]
    assert _rms_ratio(cake["o"], frost["o"]) < 2e-2


@pytest.mark.xfail(strict=True, reason=KERNEL_PARAM_GRAD_GAP)
@pytest.mark.parametrize("seq_lens,H,HV", SHAPES, ids=lambda s: str(s))
def test_gate_parameter_gradients_match_frost(steps, seq_lens, H, HV):
    _inputs, cake, frost = steps[str(seq_lens)]
    for name in ("dg", "da_log", "ddt_bias"):
        assert _rms_ratio(cake[name], frost[name]) < 4e-2, name


@pytest.mark.parametrize("seq_lens,H,HV", SHAPES[:3], ids=lambda s: str(s))
def test_final_state_matches_reference(steps, seq_lens, H, HV):
    inputs, cake, _frost = steps[str(seq_lens)]
    q = F.normalize(inputs["q"].float(), dim=-1)
    k = F.normalize(inputs["k"].float(), dim=-1)
    o_ref, fs_ref = kda_reference(
        q[None],
        k[None],
        inputs["v"][None],
        inputs["g"][None],
        inputs["beta"][None],
        scale=SCALE,
        initial_state=inputs["state0"],
        cu_seqlens=inputs["cu"],
        safe_gate=True,
        gate_lower_bound=LB,
        a_log=inputs["a_log"],
        dt_bias=inputs["dt_bias"],
        use_beta_sigmoid=True,
    )
    assert _rms_ratio(cake["fs"], fs_ref) < 2e-2
    # The forward output gap against the fp64 reference, recorded rather than hidden.
    assert _rms_ratio(cake["o"], o_ref) < 0.5


@pytest.mark.parametrize(
    "override",
    [
        dict(g_dtype=torch.float32),
        dict(checkpoint_every_n_tokens=0),
        dict(gate_lower_bound=-3.0),
        dict(use_qk_l2norm_in_kernel=False),
        dict(scale=1.0),
    ],
    ids=["fp32-g", "no-checkpoints", "lower-bound", "no-l2norm", "scale"],
)
def test_declines_outside_the_contract(cake_available, override):
    inputs = make_inputs([256], 2, 2)
    g = inputs["g"].to(override.pop("g_dtype", torch.bfloat16))
    kw = dict(
        initial_state=inputs["state0"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=LB,
        a_log=inputs["a_log"],
        dt_bias=inputs["dt_bias"],
        checkpoint_every_n_tokens=16,
        plan_name=CAKE,
    )
    kw.update(override)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        la_ops.kimi_delta_attention(inputs["q"], inputs["k"], inputs["v"], g, inputs["beta"], inputs["cu"], **kw)
