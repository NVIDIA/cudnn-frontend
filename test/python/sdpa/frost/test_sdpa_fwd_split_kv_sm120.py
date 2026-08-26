# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV split on SM120, driven through the adapter rather than the template.

KNOB ROUTE: the adapter never decides the split itself — the heuristic's
chooser picks a value and the test passes it explicitly as the ``split_kv``
constructor knob, exactly as ``lower_dsl_prefill`` forwards a plan's knobs.
The chooser and the two-launch execute are shared with SM100; what differs here
is the geometry (one CTA per tile, no cluster) and that the config bars a split
under the flattened LPT schedulers.
"""

import math
from typing import NamedTuple, Optional

import pytest
import torch

from frost_test_utils import make_dense_bias, requires_dsl

pytestmark = requires_dsl


class _ApiCaseResult(NamedTuple):
    split: int
    output: torch.Tensor
    reference: torch.Tensor
    workspace_bytes: int
    expected_split: int
    stats: Optional[torch.Tensor]


def _expected_split(api):
    """What the chooser asks for on THIS device, from the adapter's OWN tile
    geometry. SM120 runs one CTA per tile (no cluster), and the adapter may pick
    either q_tile, so reading them off `api` keeps the expectation tied to the
    launch that actually happens rather than to one part's SM count."""
    from cudnn._device import device_info
    from cudnn.sdpa.fwd.heuristics import choose_split_kv

    return choose_split_kv(
        q_tiles=-(-api.s_q_max // api.q_tile),
        heads_q=api.h_q,
        batch=api.batch_size,
        kv_tiles=-(-api.s_k_max // api.kv_tile),
        sm_count=device_info(torch.cuda.current_device()).sm_count,
        ctas_per_tile=1,
        # The combine's grid is (S_q, H, B) — see choose_split_kv.
        combine_rows=api.batch_size * api.h_q * api.s_q_max,
    )


def _sm120_case(
    h_q,
    h_kv,
    s_q,
    s_kv,
    *,
    with_lse=False,
    workspace=True,
    causal=False,
    causal_bottom_right=False,
    window_size_left=None,
    window_size_right=None,
    lse_layout="contiguous",
    split_kv=None,
    bias_dtype=None,
    bias_layout="contiguous",
    pack_gqa=None,
):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, d, dev = 1, 128, "cuda"
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s_q, d, device=dev, dtype=torch.float16)
    k = torch.randn(b, h_kv, s_kv, d, device=dev, dtype=torch.float16)
    v = torch.randn(b, h_kv, s_kv, d, device=dev, dtype=torch.float16)
    bias = make_dense_bias(h_q, s_q, s_kv, bias_dtype, bias_layout) if bias_dtype is not None else None
    o = torch.zeros_like(q)
    lse_storage = None
    if not with_lse:
        lse = None
    elif lse_layout == "contiguous":
        lse = torch.zeros(b, h_q, s_q, device=dev, dtype=torch.float32)
    elif lse_layout == "strided":
        lse_storage = torch.full((s_q + 7, h_q + 2, b), -12345.0, device=dev, dtype=torch.float32)
        lse = lse_storage.permute(2, 1, 0)[:, :h_q, :s_q]
    else:
        raise ValueError(f"unknown LSE layout {lse_layout!r}")

    kw = dict(
        sample_bias=bias,
        is_causal=causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        pack_gqa=pack_gqa,
    )
    # Probe pass: the chooser reads the adapter's own tile geometry.
    probe = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, **kw)
    assert probe.check_support()
    expected = _expected_split(probe) if split_kv is None else split_kv
    # Knob route: the chosen split arrives as an explicit constructor knob
    # (split sets ride SCHED_NATURAL — the config bars a split under the LPT
    # remaps a causal graph would otherwise derive).
    if expected > 1:
        kw.update(split_kv=expected, sched_policy=0)
    api = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, sample_lse=lse, **kw)
    assert api.check_support()
    split = api.split_kv
    ws_bytes = api.scratch_workspace_bytes()
    api.compile()
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev) if (workspace and ws_bytes) else None
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, bias_tensor=bias, workspace=ws)
    torch.cuda.synchronize()

    qb, kb, vb = q.float(), k.float(), v.float()
    if h_q != h_kv:
        kb = kb.repeat_interleave(h_q // h_kv, dim=1)
        vb = vb.repeat_interleave(h_q // h_kv, dim=1)
    scores = torch.matmul(qb, kb.transpose(-1, -2)) / math.sqrt(d)
    if bias is not None:
        scores = scores + bias.float()
    if causal or window_size_left is not None or window_size_right is not None:
        i = torch.arange(s_q, device=scores.device).view(s_q, 1)
        j = torch.arange(s_kv, device=scores.device).view(1, s_kv)
        diagonal = i + (s_kv - s_q) if causal_bottom_right else i
        masked = torch.zeros((s_q, s_kv), dtype=torch.bool, device=scores.device)
        if causal:
            masked |= j > diagonal
        if window_size_right is not None:
            masked |= j > diagonal + window_size_right
        if window_size_left is not None:
            masked |= j < diagonal - window_size_left
        scores = scores.masked_fill(masked, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    if lse is not None:
        torch.testing.assert_close(lse, torch.logsumexp(scores, dim=-1), rtol=3e-2, atol=5e-2)
    if lse_storage is not None:
        gaps = torch.ones_like(lse_storage, dtype=torch.bool)
        gaps[:s_q, :h_q, :] = False
        assert torch.all(lse_storage[gaps] == -12345.0), "the combine LSE store touched padding outside its declared view"
    return _ApiCaseResult(split, o.float(), torch.matmul(p, vb), ws_bytes, expected, None if lse is None else lse.clone())


@pytest.mark.L0
def test_sm120_splits_a_decode_shape():
    """A decode shape the chooser wants split -- and the adapter must honor it."""
    result = _sm120_case(8, 1, 128, 32768)
    if result.expected_split == 1:
        pytest.skip("this part is small enough that the shape already fills it")
    assert result.split == result.expected_split > 1
    assert result.workspace_bytes > 0
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_sm120_does_not_split_a_full_part():
    """A launch that fills the part is left alone. Whether 1024x64 fills it
    depends on the SM count, so the expectation comes from the chooser rather
    than a fixed 1 that only holds on one device."""
    result = _sm120_case(64, 8, 1024, 16384)
    assert result.split == result.expected_split
    assert (result.workspace_bytes > 0) == (result.split > 1), "workspace is needed exactly when we split"
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.L0
@pytest.mark.parametrize("workspace", [True, False], ids=["carved", "standalone"])
def test_sm120_split_with_and_without_workspace(workspace):
    result = _sm120_case(8, 1, 128, 32768, workspace=workspace)
    assert result.split == result.expected_split
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_sm120_split_writes_the_recombined_lse():
    result = _sm120_case(8, 1, 128, 32768, with_lse=True)
    assert result.split == result.expected_split
    assert (result.output - result.reference).abs().max().item() <= 2e-2


@pytest.mark.L0
def test_sm120_split_writes_strided_recombined_lse():
    contiguous = _sm120_case(8, 1, 128, 1024, with_lse=True, split_kv=2)
    strided = _sm120_case(8, 1, 128, 1024, with_lse=True, lse_layout="strided", split_kv=2)
    assert strided.split == strided.expected_split == 2
    assert (strided.output - strided.reference).abs().max().item() <= 2e-2
    torch.testing.assert_close(strided.stats, contiguous.stats, atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32], ids=["bias-fp16", "bias-fp32"])
def test_sm120_split_attention_bias_uses_absolute_kv_columns(bias_dtype):
    result = _sm120_case(8, 2, 96, 1024, with_lse=True, split_kv=2, bias_dtype=bias_dtype, bias_layout="strided")
    assert result.split == 2
    assert (result.output - result.reference).abs().max().item() <= 3e-2


@pytest.mark.L1
@pytest.mark.parametrize(
    ("pack_gqa", "mask_kind"),
    [(False, "causal"), (True, "bottom_right"), (True, "swa")],
    ids=["gqa-causal", "pack-gqa-bottom-right", "pack-gqa-swa"],
)
def test_sm120_split_attention_bias_mask_combinations(pack_gqa, mask_kind):
    """Split-KV bias composes with every mask geometry supported by SM120 split-KV."""

    s_q, s_kv = (512, 512) if mask_kind == "swa" else (384, 1024)
    mask_kwargs = dict(causal=True)
    if mask_kind == "bottom_right":
        mask_kwargs["causal_bottom_right"] = True
    elif mask_kind == "swa":
        mask_kwargs["window_size_left"] = 128

    result = _sm120_case(
        8,
        2,
        s_q,
        s_kv,
        split_kv=2,
        bias_dtype=torch.float32,
        bias_layout="strided",
        pack_gqa=pack_gqa,
        **mask_kwargs,
    )
    assert result.split == 2
    assert (result.output - result.reference).abs().max().item() <= 3e-2


@pytest.mark.L0
def test_sm120_split_attention_bias_with_padding_is_rejected():
    """SM120 split-KV currently serves only unpadded dense bias graphs."""

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    batch, heads, sequence, head_dim = 1, 4, 128, 128
    q = torch.randn(batch, heads, sequence, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.empty_like(q)
    bias = make_dense_bias(heads, sequence, sequence, torch.float32, "contiguous")
    api = SdpaFwdDslSm120(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_bias=bias,
        seq_kv_lens_present=True,
        split_kv=2,
        sched_policy=0,
    )

    with pytest.raises(ValueError, match="serves unpadded dense graphs only"):
        api.check_support()


@pytest.mark.L0
def test_sm120_causal_split_requires_the_natural_scheduler():
    """The config bars a split under the LPT remaps a causal graph derives.
    Knob-route contract, both directions: a split WITHOUT an explicit
    scheduler lets the adapter derive LPT and must fail loudly at compile
    (honored-or-error, never silently degraded), while split + explicit
    NATURAL — what the heuristic actually emits — compiles and matches."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 part required")
    b, d, dev = 1, 128, "cuda"
    torch.manual_seed(0)
    q = torch.randn(b, 8, 512, d, device=dev, dtype=torch.float16)
    k = torch.randn(b, 1, 8192, d, device=dev, dtype=torch.float16)
    v = torch.randn(b, 1, 8192, d, device=dev, dtype=torch.float16)
    o = torch.zeros_like(q)

    bad = SdpaFwdDslSm120(sample_q=q, sample_k=k, sample_v=v, sample_o=o, is_causal=True, split_kv=2)
    assert bad.check_support()
    with pytest.raises(ValueError, match="split_kv"):
        bad.compile()  # derived LPT + split: the config backstop rejects

    result = _sm120_case(8, 1, 512, 8192, causal=True)
    if result.expected_split == 1:
        pytest.skip("this part is small enough that the causal shape already fills it")
    assert result.split == result.expected_split > 1, "the causal arm must actually exercise the split"
    assert (result.output - result.reference).abs().max().item() <= 2e-2
