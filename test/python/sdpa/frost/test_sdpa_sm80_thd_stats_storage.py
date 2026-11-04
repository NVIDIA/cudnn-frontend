# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph Stats carriers obey declared packing, with capacity checked before launch."""

from types import SimpleNamespace

import pytest
import torch

from cudnn.sdpa.fwd import api_dsl
from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


class _ReachedWorkspace(Exception):
    pass


def _bind(monkeypatch, tensor, *, token_major, graph_storage=True, tokens=5):
    # Exercise the real runtime validation without compiling or launching a GPU
    # kernel. Stop only after Stats has passed its shape/stride/capacity checks.
    api = object.__new__(api_dsl.SdpaFwdDslSm80)
    api.paged_page_size = 0
    device = torch.device("cpu")
    for name in ("q", "k", "v", "o"):
        setattr(api, name + "_desc", SimpleNamespace(name=name, shape=(2, 2, 4, 8), stride=(64, 8, 16, 1), dtype=torch.float16, device=device))
    api.lse_desc = SimpleNamespace(stride=(8, 1, 2) if token_major else (16, 8, 1))
    api.h_q = 2
    api._thd_lse_token_major = token_major
    api._thd_lse_head_stride = 1 if token_major else 8
    api._thd_lse_raw_storage = graph_storage
    api.cu_seq_q_lens = api.cu_seq_kv_lens = False
    api.max_total_seq_len_q = api._t_q_cap = tokens
    api.max_total_seq_len_kv = api._t_kv_cap = 8
    api._checked_seq_lens = lambda value, name: value
    api.scratch_workspace_bytes = lambda: 32
    inputs = [torch.zeros((1, 8, 2, 8), dtype=torch.float16) for _ in range(4)]
    lengths = torch.tensor([3, 2], dtype=torch.int32)

    def workspace(*args):
        raise _ReachedWorkspace

    monkeypatch.setattr(api_dsl, "WorkspaceCarver", workspace)
    api._execute_thd(*inputs, tensor, lengths, lengths, None, None, None)


@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("carrier", ["flat", "capacity", "packed"])
def test_graph_stats_carriers_reach_launch(monkeypatch, token_major, carrier):
    owner = torch.full((32,), 19.0)
    tensor = owner[4:20]  # A nonzero storage offset and unused backing suffix.
    if carrier == "capacity":
        tensor = tensor.view(2, 4, 2, 1).transpose(1, 2)
    elif carrier == "packed":
        tensor = tensor.view(8, 2) if token_major else tensor.view(1, 2, 8)
    with pytest.raises(_ReachedWorkspace):
        _bind(monkeypatch, tensor, token_major=token_major)
    assert torch.equal(owner, torch.full_like(owner, 19.0))


@pytest.mark.parametrize("token_major", [False, True])
def test_graph_stats_capacity_is_the_exposed_view(monkeypatch, token_major):
    required = 10 if token_major else 13
    tensor = torch.empty(64)[4 : 4 + required - 1]
    with pytest.raises(ValueError, match="Stats.*capacity"):
        _bind(monkeypatch, tensor, token_major=token_major)


@pytest.mark.parametrize("token_major", [False, True])
def test_graph_stats_zero_tokens(monkeypatch, token_major):
    with pytest.raises(_ReachedWorkspace):
        _bind(monkeypatch, torch.empty(0), token_major=token_major, tokens=0)


@pytest.mark.parametrize("token_major", [False, True])
def test_standalone_stats_keeps_layout_validation(monkeypatch, token_major):
    tensor = torch.empty(2, 4, 2, 1).transpose(1, 2)
    with pytest.raises(ValueError, match="SM80 THD.*Stats must have shape"):
        _bind(monkeypatch, tensor, token_major=token_major, graph_storage=False)
