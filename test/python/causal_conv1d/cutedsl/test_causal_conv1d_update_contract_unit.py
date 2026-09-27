# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable host contracts for the public decode-update operation."""

import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.L0


def _meta_inputs(
    *,
    n_rows: int = 2,
    n_channels: int = 8,
    n_slots: int = 3,
    state_len: int = 4,
    width: int = 4,
):
    x = torch.empty(n_rows, n_channels, device="meta", dtype=torch.bfloat16)
    state = torch.empty(n_slots, n_channels, state_len, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(n_channels, width, device="meta", dtype=torch.bfloat16)
    return x, state, weight


def _channel_fast_state(n_slots: int, n_channels: int, state_len: int):
    return torch.empty(
        n_slots,
        state_len,
        n_channels,
        device="meta",
        dtype=torch.bfloat16,
    ).transpose(1, 2)


def test_only_cudnn_ops_exports_the_semantic_api(tmp_path):
    """Check the source package in a fresh interpreter, not conftest's overlay."""

    import cudnn

    source = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    probe = tmp_path / "cudnn"
    shutil.copytree(source, probe)
    compiled_modules = list(Path(cudnn.__file__).resolve().parent.glob("_compiled_module*.so"))
    if len(compiled_modules) != 1:
        pytest.skip(f"expected one compiled module next to {cudnn.__file__}, " f"found {len(compiled_modules)}")
    try:
        (probe / compiled_modules[0].name).symlink_to(compiled_modules[0])
    except OSError as error:
        pytest.skip(f"cannot link the compiled module into the probe package: {error}")

    script = """
import types
import cudnn
import cudnn.ops
from cudnn.ops import causal_conv1d_update

assert callable(causal_conv1d_update)
assert not isinstance(causal_conv1d_update, types.ModuleType)
assert cudnn.ops.causal_conv1d_update is causal_conv1d_update
assert not hasattr(cudnn, "causal_conv1d_update")
assert not hasattr(cudnn, "causal_conv1d_update_wrapper_sm100")
assert not hasattr(cudnn, "CausalConv1dUpdateSm100")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), environment.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_public_semantic_signature():
    from cudnn.ops import causal_conv1d_update

    parameters = inspect.signature(causal_conv1d_update).parameters
    assert tuple(parameters) == (
        "x",
        "conv_state",
        "weight",
        "bias",
        "activation",
        "cache_seqlens",
        "conv_state_indices",
    )
    assert parameters["bias"].default is None
    assert parameters["activation"].default is None
    for name in ("cache_seqlens", "conv_state_indices"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


@pytest.mark.parametrize("state_len", [3, 4], ids=["w-minus-one", "legacy-four"])
@pytest.mark.parametrize("activation", [None, "identity", "silu", "swish"])
def test_public_meta_contract_preserves_output_metadata(state_len, activation):
    from cudnn.ops import causal_conv1d_update

    x, state, weight = _meta_inputs(state_len=state_len)
    indices = torch.empty(x.shape[0], device="meta", dtype=torch.int32)
    output = causal_conv1d_update(
        x,
        state,
        weight,
        activation=activation,
        conv_state_indices=indices,
    )

    assert type(output) is torch.Tensor
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


@pytest.mark.parametrize("channel_fast", [False, True])
def test_public_accepts_both_prefill_handoff_and_compact_wminus1_state(
    channel_fast,
):
    from cudnn.ops import causal_conv1d_update

    x, compact_state, weight = _meta_inputs(state_len=3)
    state = _channel_fast_state(compact_state.shape[0], x.shape[1], 3) if channel_fast else compact_state

    output = causal_conv1d_update(x, state, weight, activation="silu")
    assert output.shape == x.shape


@pytest.mark.parametrize("n_channels,row_stride", [(10, 10), (10, 16)])
def test_public_accepts_supported_input_row_strides(n_channels, row_stride):
    from cudnn.ops import causal_conv1d_update

    x = torch.empty_strided(
        (2, n_channels),
        (row_stride, 1),
        device="meta",
        dtype=torch.bfloat16,
    )
    state = torch.empty(2, n_channels, 3, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(n_channels, 4, device="meta", dtype=torch.bfloat16)

    assert causal_conv1d_update(x, state, weight).shape == x.shape


@pytest.mark.parametrize(
    "stride,match",
    [
        ((20, 2), "row-major strides"),
        ((7, 1), "at least D"),
        ((12, 1), "16-byte-aligned"),
    ],
)
def test_public_rejects_unsupported_input_row_strides(stride, match):
    from cudnn.ops import causal_conv1d_update

    x = torch.empty_strided((2, 10), stride, device="meta", dtype=torch.bfloat16)
    state = torch.empty(2, 10, 3, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(10, 4, device="meta", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=match):
        causal_conv1d_update(x, state, weight)


def test_public_rejects_channel_fast_legacy_four_state():
    from cudnn.ops import causal_conv1d_update

    x, _, weight = _meta_inputs(state_len=4)
    state = _channel_fast_state(3, x.shape[1], 4)
    with pytest.raises(ValueError, match=r"contiguous.*or for L=3"):
        causal_conv1d_update(x, state, weight)


@pytest.mark.parametrize(
    "state_len,width,has_cache",
    [(2, 3, False), (5, 4, False), (4, 4, True)],
    ids=["width-three", "state-length-five", "circular-buffer"],
)
def test_unimplemented_compatibility_modes_decline_explicitly(state_len, width, has_cache):
    from cudnn.ops import causal_conv1d_update

    x, state, weight = _meta_inputs(state_len=state_len, width=width)
    cache_seqlens = torch.empty(x.shape[0], device="meta", dtype=torch.int32) if has_cache else None
    with pytest.raises(NotImplementedError, match=r"current native.*supports only"):
        causal_conv1d_update(x, state, weight, cache_seqlens=cache_seqlens)


def test_multi_token_input_is_not_silently_interpreted():
    from cudnn.ops import causal_conv1d_update

    x = torch.empty(2, 8, 2, device="meta", dtype=torch.bfloat16)
    state = torch.empty(2, 8, 4, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(8, 4, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"x must have shape \[N, D\]"):
        causal_conv1d_update(x, state, weight)


def test_public_rejects_invalid_activation_and_autograd():
    from cudnn.ops import causal_conv1d_update

    x, state, weight = _meta_inputs()
    with pytest.raises(ValueError, match="activation"):
        causal_conv1d_update(x, state, weight, activation="relu")

    grad_x = x.requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        causal_conv1d_update(grad_x, state, weight)
