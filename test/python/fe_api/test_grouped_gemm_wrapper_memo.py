# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The wrapper memo must never answer for operands it did not see.

The memo skips the wrapper's derivation on a repeat call. These tests pin the
property that makes that safe: the key is the operands' metadata, so anything that
changes what the derivation would produce takes a different key. In particular a
freshly allocated tensor is not trusted on account of its address -- CPython recycles
those immediately, so an identity-keyed memo would serve a stale entry to a tensor of
a different shape.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        pytest.skip("SM100 is required")


N_OUT, K, EXPERTS = 2048, 2048, 8


def _weights():
    b = torch.randn(EXPERTS, N_OUT, K, dtype=torch.bfloat16, device="cuda")
    b_ptrs = torch.tensor([b[i].data_ptr() for i in range(EXPERTS)], dtype=torch.int64, device="cuda")
    return b, b_ptrs


def _operands(m):
    return dict(
        a_tensor=torch.randn(m, K, 1, dtype=torch.bfloat16, device="cuda"),
        padded_offsets=torch.arange(m // EXPERTS, m + 1, m // EXPERTS, dtype=torch.int32, device="cuda"),
        alpha_tensor=torch.randn(EXPERTS, dtype=torch.float32, device="cuda"),
        prob_tensor=torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").reshape(m, 1, 1),
    )


def _call(m, b_ptrs, a=None, operands=None):
    from cudnn import grouped_gemm_wrapper_sm100

    operands = dict(operands if operands is not None else _operands(m))
    if a is not None:
        operands["a_tensor"] = a
    return grouped_gemm_wrapper_sm100(
        **operands,
        b_ptrs=b_ptrs,
        n=N_OUT,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
    )


@pytest.mark.L0
def test_memo_hit_matches_cold_path():
    """A memo hit produces the same bytes as the same call with the memo cleared."""
    from cudnn.gemm.cutedsl.grouped.unfused.api import _wrapper_memo

    b, b_ptrs = _weights()
    operands = _operands(2048)

    _wrapper_memo.clear()
    cold = _call(2048, b_ptrs, operands=operands)["d_tensor"].clone()
    assert _wrapper_memo, "the call should have populated the memo"
    warm = _call(2048, b_ptrs, operands=operands)["d_tensor"]

    torch.testing.assert_close(warm, cold, rtol=0, atol=0)


@pytest.mark.L0
def test_memo_misses_when_m_changes():
    """Alternating token counts with freshly built operands each step.

    Every operand is a new object per call, so their addresses are recycled; only the
    metadata in the key distinguishes the two shapes.
    """
    b, b_ptrs = _weights()
    for m in (2048, 2048, 4096, 2048, 4096, 4096, 2048, 4096):
        assert tuple(_call(m, b_ptrs)["d_tensor"].shape) == (m, N_OUT, 1)


@pytest.mark.L0
def test_memo_does_not_swallow_a_dtype_mismatch():
    """A fresh, wrongly-typed a_tensor is still rejected after the memo is warm."""
    b, b_ptrs = _weights()
    _call(2048, b_ptrs)

    bad = torch.empty(2048, K, 1, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        _call(2048, b_ptrs, a=bad)


@pytest.mark.L0
def test_memo_misses_on_a_transposed_operand():
    """Same shape and dtype, different strides -- a different key, and still rejected."""
    b, b_ptrs = _weights()
    _call(2048, b_ptrs)

    transposed = torch.randn(K, 2048, 1, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    assert tuple(transposed.shape) == (2048, K, 1)
    with pytest.raises(ValueError):
        _call(2048, b_ptrs, a=transposed)


@pytest.mark.L0
def test_glu_memo_hit_matches_cold_path():
    from cudnn import grouped_gemm_glu_wrapper_sm100
    from cudnn.gemm.cutedsl.grouped.glu.api import _glu_wrapper_memo

    m = 2048
    b, b_ptrs = _weights()
    kwargs = dict(
        a_tensor=torch.randn(m, K, 1, dtype=torch.bfloat16, device="cuda"),
        sfa_tensor=None,
        padded_offsets=torch.arange(m // EXPERTS, m + 1, m // EXPERTS, dtype=torch.int32, device="cuda"),
        alpha_tensor=torch.randn(EXPERTS, dtype=torch.float32, device="cuda"),
        prob_tensor=torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").reshape(m, 1, 1),
        b_ptrs=b_ptrs,
        n=N_OUT,
        b_dtype=torch.bfloat16,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
    )

    _glu_wrapper_memo.clear()
    cold = grouped_gemm_glu_wrapper_sm100(**kwargs)["d_tensor"].clone()
    assert _glu_wrapper_memo, "the call should have populated the memo"
    warm = grouped_gemm_glu_wrapper_sm100(**kwargs)["d_tensor"]

    torch.testing.assert_close(warm, cold, rtol=0, atol=0)


@pytest.mark.L0
def test_memo_key_covers_every_wrapper_parameter():
    """Every parameter that can change the result must appear in that wrapper's memo key.

    A parameter added to a wrapper but not to its key makes two different calls collide,
    and the failure is silent: the second gets the first's compiled op and output spec.
    This caught `sf_fp8_dtype_override` going missing from the GLU key during a rebase.

    Auto-discovers every CuTeDSL wrapper that has a memo, so an op gains coverage the
    moment one is added to it and no edit here is needed.
    """
    import importlib
    import inspect
    import pkgutil
    import re

    import cudnn.gemm.cutedsl as cutedsl

    # linear_offset is not part of any op cache key either; the wrappers resolve and
    # forward the caller's value on every call, hit or miss. current_stream is per-call
    # state, not a cache dimension.
    NOT_CACHE_DIMENSIONS = {"current_stream", "linear_offset"}

    checked = []
    for module_info in pkgutil.walk_packages(cutedsl.__path__, cutedsl.__name__ + "."):
        if not module_info.name.endswith(".api"):
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for name, func in vars(module).items():
            if not (name.endswith("_wrapper_sm100") and inspect.isfunction(func)):
                continue
            try:
                source = inspect.getsource(func)
            except OSError:
                continue
            if "memo_key = (" not in source:
                continue  # no memo on this op yet
            start = source.index("memo_key = (") + len("memo_key = (")
            key_source = source[start : source.index("\n    )", start)]
            params = set(inspect.signature(func).parameters) - NOT_CACHE_DIMENSIONS
            missing = sorted(p for p in params if not re.search(rf"\b{p}\b", key_source))
            assert not missing, f"{name} parameters missing from its memo key: {missing}"
            checked.append(name)

    assert checked, "no memoized wrappers discovered -- the discovery logic is broken"


# ---- Block-scaled (MXFP8) wrappers: glu, dglu, quant, wgrad ----------------------
#
# Same metadata, different data and a different valid token count (the padded_offsets
# contents) must hit the memo, and a hit must produce the bytes the cold path produces.

TENSOR_M, N_BS, K_BS, L_BS = 2048, 512, 512, 4
MXFP8 = dict(ab_dtype=torch.float8_e4m3fn, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=32, m_aligned=256)


def _mxfp8_inputs(group_m_list, l=L_BS, b_major="k"):
    from fe_api.grouped_gemm.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors

    return allocate_grouped_gemm_input_tensors(n=N_BS, k=K_BS, l=l, group_m_list=group_m_list, permuted_m=TENSOR_M, b_major=b_major, **MXFP8)


def _bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


class _CountingMemo(dict):
    """Only a memo miss stores; a hit never writes. `stores` therefore counts misses."""

    stores = 0

    def __setitem__(self, key, value):
        self.stores += 1
        super().__setitem__(key, value)


def _install_memo(monkeypatch, module, name):
    memo = _CountingMemo()
    monkeypatch.setattr(module, name, memo)
    return memo


def _assert_hit_matches_cold(monkeypatch, module, memo_name, call, inputs_a, inputs_b, pick):
    memo = _install_memo(monkeypatch, module, memo_name)
    call(inputs_a)
    assert memo.stores == 1 and len(memo) == 1, "the first call should miss and populate the memo"
    warm = _bytes(pick(call(inputs_b))).clone()
    assert memo.stores == 1, "the second call should hit the memo"
    memo.clear()
    cold = _bytes(pick(call(inputs_b)))
    assert memo.stores == 2
    torch.cuda.synchronize()
    assert torch.equal(warm, cold)


def _glu_bs_call(inputs, **overrides):
    from cudnn import grouped_gemm_glu_wrapper_sm100

    kwargs = dict(
        a_tensor=inputs["a_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        norm_const_tensor=inputs["norm_const_tensor"],
        prob_tensor=inputs["prob_tensor"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=32,
    )
    kwargs.update(overrides)
    return grouped_gemm_glu_wrapper_sm100(**kwargs)


@pytest.mark.L0
def test_glu_block_scaled_memo_hit_matches_cold_path(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.glu import api as glu_api

    inputs_a = _mxfp8_inputs([256] * L_BS)
    inputs_b = _mxfp8_inputs([512, 256, 512, 256])
    assert inputs_a["valid_m"] != inputs_b["valid_m"]
    _assert_hit_matches_cold(monkeypatch, glu_api, "_glu_wrapper_memo", _glu_bs_call, inputs_a, inputs_b, lambda out: out["d_tensor"][: inputs_b["valid_m"]])


@pytest.mark.L0
def test_glu_block_scaled_memo_misses_on_plan_time_change(monkeypatch):
    """Expert count, output dtype and operand stride order each take a different key."""
    from cudnn.gemm.cutedsl.grouped.glu import api as glu_api

    memo = _install_memo(monkeypatch, glu_api, "_glu_wrapper_memo")
    _glu_bs_call(_mxfp8_inputs([256] * L_BS))
    _glu_bs_call(_mxfp8_inputs([256] * L_BS))
    assert memo.stores == 1

    fewer_experts = _glu_bs_call(_mxfp8_inputs([256] * 2, l=2))
    assert memo.stores == 2 and fewer_experts["sfd_row_tensor"] is not None

    bf16_out = _glu_bs_call(_mxfp8_inputs([256] * L_BS), d_dtype=torch.bfloat16)
    assert memo.stores == 3 and tuple(bf16_out["amax_tensor"].shape) == (L_BS, 1)

    # Same shapes and dtypes, different strides: a different key, and still rejected.
    with pytest.raises(ValueError):
        _glu_bs_call(_mxfp8_inputs([256] * L_BS, b_major="n"))
    assert memo.stores == 3 and len(memo) == 3


def _dglu_bs_inputs(group_m_list):
    from fe_api.grouped_gemm.test_grouped_gemm_dswiglu_utils import allocate_grouped_gemm_dswiglu_tensors

    inputs = _mxfp8_inputs(group_m_list)
    inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=TENSOR_M,
        n=N_BS,
        l=L_BS,
        ab_dtype=MXFP8["ab_dtype"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        cd_major="n",
        sf_dtype=MXFP8["sf_dtype"],
        sf_vec_size=MXFP8["sf_vec_size"],
        input_tensors=inputs,
    )
    inputs["dprob_tensor"] = outputs["dprob_tensor"]
    return inputs


def _dglu_bs_call(inputs):
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    inputs["dprob_tensor"].zero_()
    return grouped_gemm_dglu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=inputs["dprob_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        norm_const_tensor=inputs["norm_const_tensor"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=32,
    )


@pytest.mark.L0
def test_dglu_block_scaled_memo_hit_matches_cold_path(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.dglu import api as dglu_api

    inputs_a = _dglu_bs_inputs([256] * L_BS)
    inputs_b = _dglu_bs_inputs([512, 256, 512, 256])
    _assert_hit_matches_cold(
        monkeypatch, dglu_api, "_dglu_wrapper_memo", _dglu_bs_call, inputs_a, inputs_b, lambda out: out["d_row_tensor"][: inputs_b["valid_m"]]
    )


def _quant_call(inputs, **overrides):
    from cudnn import grouped_gemm_quant_wrapper_sm100

    kwargs = dict(
        a_tensor=inputs["a_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        norm_const_tensor=inputs["norm_const_tensor"],
        prob_tensor=inputs["prob_tensor"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=32,
    )
    kwargs.update(overrides)
    return grouped_gemm_quant_wrapper_sm100(**kwargs)


@pytest.mark.L0
def test_quant_memo_hit_matches_cold_path(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.quant import api as quant_api

    inputs_a = _mxfp8_inputs([256] * L_BS)
    inputs_b = _mxfp8_inputs([512, 256, 512, 256])
    _assert_hit_matches_cold(monkeypatch, quant_api, "_quant_wrapper_memo", _quant_call, inputs_a, inputs_b, lambda out: out["d_tensor"][: inputs_b["valid_m"]])


@pytest.mark.L0
def test_quant_memo_misses_on_plan_time_change(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.quant import api as quant_api

    memo = _install_memo(monkeypatch, quant_api, "_quant_wrapper_memo")
    _quant_call(_mxfp8_inputs([256] * L_BS))
    _quant_call(_mxfp8_inputs([256] * L_BS))
    assert memo.stores == 1
    _quant_call(_mxfp8_inputs([256] * 2, l=2))
    assert memo.stores == 2
    bf16_out = _quant_call(_mxfp8_inputs([256] * L_BS), d_dtype=torch.bfloat16)
    assert memo.stores == 3 and tuple(bf16_out["amax_tensor"].shape) == (L_BS, 1) and bf16_out["sfd_row_tensor"] is None
    assert len(memo) == 3


def _wgrad_inputs(group_k_list):
    from fe_api.grouped_gemm.test_grouped_gemm_wgrad_utils import allocate_grouped_gemm_wgrad_tensors, grouped_gemm_wgrad_init

    cfg = grouped_gemm_wgrad_init(
        ab_dtype=torch.float8_e4m3fn,
        wgrad_dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
        sf_dtype=torch.float8_e8m0fnu,
    )
    cfg["group_k_list"] = group_k_list
    return allocate_grouped_gemm_wgrad_tensors(cfg)


def _wgrad_call(inputs, **overrides):
    from cudnn import grouped_gemm_wgrad_wrapper_sm100

    kwargs = dict(
        a_tensor=inputs["a_tensor"],
        b_tensor=inputs["b_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        offsets_tensor=inputs["offsets_tensor"],
        wgrad_dtype=torch.bfloat16,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        sf_vec_size=32,
    )
    kwargs.update(overrides)
    return grouped_gemm_wgrad_wrapper_sm100(**kwargs)


@pytest.mark.L0
def test_wgrad_memo_hit_matches_cold_path(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api

    inputs_a = _wgrad_inputs([256, 384])
    inputs_b = _wgrad_inputs([384, 256])
    _assert_hit_matches_cold(monkeypatch, wgrad_api, "_wgrad_wrapper_memo", _wgrad_call, inputs_a, inputs_b, lambda out: out["wgrad_tensor"])


@pytest.mark.L0
def test_wgrad_memo_misses_on_plan_time_change(monkeypatch):
    from cudnn.gemm.cutedsl.grouped.wgrad import api as wgrad_api

    memo = _install_memo(monkeypatch, wgrad_api, "_wgrad_wrapper_memo")
    _wgrad_call(_wgrad_inputs([256, 384]))
    _wgrad_call(_wgrad_inputs([384, 256]))
    assert memo.stores == 1
    accumulate = _wgrad_call(_wgrad_inputs([256, 384]), accumulate_on_output=True)
    assert memo.stores == 2
    fp16_out = _wgrad_call(_wgrad_inputs([256, 384]), wgrad_dtype=torch.float16)
    assert memo.stores == 3 and fp16_out["wgrad_tensor"].dtype == torch.float16 and accumulate["wgrad_tensor"].dtype == torch.bfloat16
    assert len(memo) == 3
