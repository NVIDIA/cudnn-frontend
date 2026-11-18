# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Torch's public selector contract separate from the FE operator's domain."""

import inspect

import pytest
import torch

from cudnn.torch import sdpa_provider


def cudnn_rejection_reasons(query, key, value, cu_seq_q, cu_seq_k, max_q, window_size, enable_gqa=False, seqused_k=None):
    if not sdpa_provider._has_backend_arg(torch.ops.torch_attn._varlen_attn.default):
        return []
    from torch.nn.attention import varlen

    # The selector's capabilities evolve independently of FE (e.g. GQA, SM80,
    # causal KV caches). Use the installed selector rather than a Torch version
    # threshold or a second, inevitably stale copy of its capability rules.
    arguments = dict(locals())
    check = varlen._cudnn_rejection_reasons
    return check(**{name: arguments[name] for name in inspect.signature(check).parameters if name in arguments})


def assert_public_rejection(reasons, call):
    assert reasons
    before = dict(sdpa_provider.calls)
    with pytest.raises(RuntimeError, match=r"SDPBackend\.CUDNN_ATTENTION was requested for varlen_attn") as error:
        call()
    for reason in reasons:
        assert reason in str(error.value)
    assert sdpa_provider.calls == before, "a rejected public call must not reach the provider"


def provider_varlen(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, window_size=(-1, 0), enable_gqa=False):
    """Exercise the registered FE operator even when Torch's selector declines it."""
    kwargs = dict(is_causal=window_size[1] == 0, window_size=list(window_size), enable_gqa=enable_gqa)
    if sdpa_provider._has_backend_arg(torch.ops.torch_attn._varlen_attn.default):
        kwargs["backend"] = torch.nn.attention.SDPBackend.CUDNN_ATTENTION.value
    out, lse, _ = torch.ops.torch_attn._varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, **kwargs)
    return out, lse
