# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch
from cutlass import Boolean

import cudnn
import cudnn.flex_attention as flex_attention
import cudnn.flex_attention.api as flex_attention_api
from cudnn.api_base import APIBase
from cudnn.flex_attention._compat import sm90_utils
from cudnn.flex_attention.autograd import FlexAttnFunc
from cudnn.flex_attention.execution import FlexAttentionBwd, FlexAttentionFwd
from cudnn.flex_attention.plan.mask_plan import ArbitraryPlanRuntimeBinding, MaskPlan, validate_arbitrary_plan_runtime_binding
from cudnn.flex_attention.plan.validation import is_supported_head_dims, validate_call_options
from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES
from cudnn.flex_attention.runtime.dsl_utils import _cute_dsl_bulk_copy_self_elects

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ((4, 5, 2), False),
        ((4, 6, 0), True),
        ((4, 6, 1), True),
        ((4, 6, 2), False),
        ((4, 6, 3), False),
        ((4, 7, 0), False),
    ),
)
def test_bulk_copy_internal_election_version_window(version, expected):
    assert _cute_dsl_bulk_copy_self_elects(version) is expected


def test_public_exports_are_lazy_top_level_aliases():
    assert cudnn.flex_attention is flex_attention
    for name in (
        "FlexAttentionBwd",
        "FlexAttentionFwd",
        "create_mask_plan",
        "flex_attn_func",
    ):
        assert getattr(cudnn, name) is getattr(flex_attention, name)
        assert getattr(flex_attention, name) is getattr(flex_attention_api, name)
    assert flex_attention.MaskPlan is flex_attention_api.MaskPlan
    assert not hasattr(cudnn, "MaskPlan")
    assert issubclass(FlexAttentionFwd, APIBase)
    assert issubclass(FlexAttentionBwd, APIBase)
    for internal_name in ("flex_attention_forward", "flex_attention_backward"):
        assert not hasattr(cudnn, internal_name)
        assert not hasattr(flex_attention, internal_name)
        assert not hasattr(flex_attention_api, internal_name)


def test_clean_top_level_import_stays_optional_and_quack_free():
    repository_root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    python_path = str(repository_root / "python")
    if environment.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = python_path
    script = """
import sys
import cudnn
assert "torch" not in sys.modules
assert "cutlass" not in sys.modules
from cudnn import (
    FlexAttentionBwd,
    FlexAttentionFwd,
    create_mask_plan,
    flex_attn_func,
)
module = cudnn.flex_attention
assert create_mask_plan is module.create_mask_plan
assert module.flex_attn_func is cudnn.flex_attn_func
assert flex_attn_func is module.flex_attn_func
assert FlexAttentionFwd is module.FlexAttentionFwd
assert FlexAttentionBwd is module.FlexAttentionBwd
assert not hasattr(cudnn, "flex_attention_forward")
assert not hasattr(cudnn, "flex_attention_backward")
assert not hasattr(module, "flex_attention_forward")
assert not hasattr(module, "flex_attention_backward")
assert not hasattr(cudnn, "MaskPlan")
assert not any(name == "quack" or name.startswith("quack.") for name in sys.modules)
"""
    subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_kernel_sources_have_no_quack_imports():
    package_root = Path(flex_attention.__file__).resolve().parent
    sources = "\n".join(path.read_text(encoding="utf-8") for path in package_root.rglob("*.py"))
    assert re.search(r"^\s*(?:from|import)\s+quack\b", sources, flags=re.MULTILINE) is None
    assert "quack." not in sources


def test_supported_arch_and_head_dimension_contracts():
    assert SUPPORTED_ARCHES == (90, 100, 103)
    for dims in ((8, 8), (64, 128), (128, 8), (192, 128), (256, 256)):
        assert is_supported_head_dims(*dims)
    for dims in ((7, 8), (136, 128), (192, 192), (256, 128)):
        assert not is_supported_head_dims(*dims)


def test_sm90_gemm_zero_init_is_runtime_boolean():
    assert sm90_utils.gemm.__annotations__["zero_init"] is Boolean


def test_call_option_validation():
    validate_call_options(softmax_scale=None, deterministic=False, return_lse=False)
    validate_call_options(softmax_scale=0.125, deterministic=True, return_lse=True)
    with pytest.raises(ValueError, match="softmax_scale"):
        validate_call_options(softmax_scale=float("nan"), deterministic=False, return_lse=False)
    with pytest.raises(TypeError, match="deterministic"):
        validate_call_options(softmax_scale=None, deterministic=1, return_lse=False)


def test_mask_plan_mode_is_inferred_from_cumulative_sequence_lengths():
    plan = object.__new__(MaskPlan)
    plan._cu_seqlens_q = None
    plan._cu_seqlens_k = None
    assert not plan._is_varlen

    prefix = object()
    plan._cu_seqlens_q = prefix
    plan._cu_seqlens_k = prefix
    assert plan._is_varlen

    plan._cu_seqlens_k = None
    with pytest.raises(RuntimeError, match="both cu_seqlens_q and cu_seqlens_k"):
        _ = plan._is_varlen


@pytest.mark.parametrize("mutated_name", ("cu_seqlens_q", "cu_seqlens_k"))
def test_arbitrary_plan_runtime_binding_rejects_in_place_prefix_mutation(mutated_name):
    prefixes = {
        "cu_seqlens_q": torch.tensor((0, 96, 160), dtype=torch.int32),
        "cu_seqlens_k": torch.tensor((0, 80, 128), dtype=torch.int32),
    }
    runtime_args = {
        "is_varlen": True,
        "batch_size": 2,
        "seqlen_q": None,
        "seqlen_k": None,
        "total_q": 160,
        "total_k": 128,
        "max_seqlen_q": 96,
        "max_seqlen_k": 80,
        **prefixes,
    }
    binding = ArbitraryPlanRuntimeBinding.capture(**runtime_args)

    prefixes[mutated_name].add_(1)

    with pytest.raises(ValueError, match=rf"{mutated_name} was modified in-place"):
        validate_arbitrary_plan_runtime_binding(binding, context="Flex Attention plan", **runtime_args)


def test_functional_return_contract(monkeypatch):
    plan = object.__new__(MaskPlan)
    runtime_calls = []
    apply_calls = []

    def validate_runtime(self, q, k, v):
        runtime_calls.append((self, q, k, v))

    def apply(*args):
        apply_calls.append(args)
        return "out", "lse"

    monkeypatch.setattr(MaskPlan, "_validate_runtime", validate_runtime)
    monkeypatch.setattr(FlexAttnFunc, "apply", staticmethod(apply))
    q, k, v = object(), object(), object()
    assert flex_attention.flex_attn_func(q, k, v, mask_plan=plan) == "out"
    assert flex_attention.flex_attn_func(q, k, v, mask_plan=plan, return_lse=True) == ("out", "lse")
    assert runtime_calls == [(plan, q, k, v), (plan, q, k, v)]
    assert [call[-1] for call in apply_calls] == [False, True]


def test_allocating_wrappers_stay_internal():
    from cudnn.flex_attention import execution

    assert callable(execution._flex_attention_forward)
    assert callable(execution._flex_attention_backward)
    assert "_flex_attention_forward" not in execution.__all__
    assert "_flex_attention_backward" not in execution.__all__
