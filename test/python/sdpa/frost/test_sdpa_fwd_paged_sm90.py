# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native Hopper page-pool loads, independent tables and captured rebinding."""

import pytest
import torch

import test_sdpa_fwd_paged_sm120 as paging
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native SM90")]


@pytest.fixture(autouse=True)
def native(monkeypatch):
    from cudnn.sdpa.fwd.engines import engine_name

    monkeypatch.setattr(paging, "engine_name", lambda **kw: engine_name(arch="sm90"))
    with torch.cuda.stream(torch.cuda.Stream()):
        yield


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "page,dims,mask,packed,splits,sq,sched",
    [
        (8, (128, 128), "none", False, 1, 67, 0),
        (16, (192, 128), "br_swa", True, 3, 5, 1),
        (32, (256, 256), "band", False, 1, 65, 2),
        (64, (512, 512), "br_swa", True, 3, 1, 1),
        (128, (120, 72), "none", False, 1, 67, 0),
        (192, (248, 136), "band", True, 1, 67, 1),
        (256, (384, 320), "none", False, 3, 67, 2),
        (8, (64, 64), "br_swa", False, 1, 1, 0),
    ],
)
def test_paged_replay(dtype, page, dims, mask, packed, splits, sq, sched):
    g, nodes, bufs, ws, storage, stats_storage = paging._case(
        dtype,
        2,
        page,
        True,
        False,
        dims=dims,
        mask=mask,
        sink=True,
        packed=packed,
        split_kv=splits,
        sq=sq,
        scheduler=sched,
        stats_log2=True,
        explicit_plan=True,
        independent_tables=True,
    )
    variant = {nodes[n]: t for n, t in bufs.items()}

    def execute():
        g.execute(variant, ws)

    execute()
    paging._check(bufs, page, mask=mask, stats_log2=True)
    # The native geometry cache must bind new storage, including each table's
    # independent pitches, without recompilation or a Python binder fallback.
    for name, tensor in tuple(bufs.items()):
        if name in ("o", "stats"):
            continue
        fresh = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device)
        fresh.copy_(tensor)
        bufs[name] = fresh
        variant[nodes[name]] = fresh
    execute()
    paging._check(bufs, page, mask=mask, stats_log2=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            torch.cuda.set_sync_debug_mode("error")
            try:
                execute()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        bufs["kt"][0, 0, 0, 0] = 3
        bufs["vt"][0, 0, 0, 0] = 7
        bufs["kl"][0] = page - 1
        bufs["o"].fill_(float("nan"))
        bufs["stats"].fill_(float("nan"))
        graph.replay()
        paging._check(bufs, page, mask=mask, stats_log2=True)
        assert (storage[..., dims[1] :] == 123).all()
        assert (stats_storage[..., 1::2] == 777).all()
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("dims", [(128, 128), (192, 128), (512, 512)])
def test_batch_replay_and_golden(dtype, dims):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    packed = dtype == torch.bfloat16
    paging.test_sm120_frost_paged_width_determinism(dtype, dims, packed, split_kv=3 if packed else 1, api_type=SdpaFwdDslSm90, arch="sm90")


def test_paged_admission_and_optional_stats():
    from dataclasses import replace
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs, _sm90_spec, mismatch
    from cudnn.sdpa.graph_analyzer import analyze

    graph, nodes, bufs, ws, _, _ = paging._case(torch.float16, 2, 16, False, True)
    graph.execute({nodes[name]: tensor for name, tensor in bufs.items()}, ws)
    facts, caps = analyze(graph), _sm90_spec().capabilities
    for page in (8, 16, 32, 64, 128, 192, 256):
        assert mismatch(caps, replace(facts, page_size=page), SdpaFwdKnobs(tile_m=64, tile_n=64)) is None
    assert "dense Q" in mismatch(caps, replace(facts, thd=True))
    assert "page_size" in mismatch(caps, replace(facts, page_size=24))
    options = dict(paged_page_size=16, paged_max_seq_len_kv=48, seq_kv_lens_present=True, seq_q_lens_present=True)

    def make(**changes):
        return SdpaFwdDslSm90(bufs["q"], bufs["k"], bufs["v"], bufs["o"], **(options | changes))

    for changes, kind, message in (
        ({"thd": True}, NotImplementedError, "dense Q"),
        ({"paged_page_size": 24}, NotImplementedError, "page size"),
        ({"seq_kv_lens_present": False}, NotImplementedError, "KV lengths"),
        ({"paged_max_seq_len_kv": None}, ValueError, "paged_max_seq_len_kv"),
    ):
        with pytest.raises(kind, match=message):
            make(**changes).check_support()
    api = make()
    api.check_support()
    api.compile()
    assert api.scratch_workspace_bytes() == 0
    tensors = {name + "_tensor": bufs[name] for name in ("q", "k", "v", "o")}
    metadata = dict(seq_q_lens=bufs["ql"].flatten(), seq_kv_lens=bufs["kl"].flatten())
    with pytest.raises(ValueError, match="both K and V page tables"):
        api.execute(**tensors, **metadata)
    bufs["o"].fill_(float("nan"))
    api.execute(**tensors, **metadata, block_table=bufs["kt"], block_table_v=bufs["vt"])
    # The graph's reference helper checks O independently; its saved Stats are
    # populated once by the graph, while the standalone call omits that output.
    paging._check(bufs, 16)


@pytest.mark.parametrize("dims", [(8, 8), (8, 512), (512, 8)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_standalone_page_head_boundaries(dims, dtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    d, dv = dims
    b, h, hk, sq, page = 2, 4, 2, 5, 8
    rng = torch.Generator().manual_seed(9016)
    bufs = {}
    for name, shape in (("q", (b, h, sq, d)), ("k", (7, hk, page, d)), ("v", (7, hk, page, dv))):
        bufs[name] = (torch.randn(shape, generator=rng) * 0.2).to(dtype).cuda()
    bufs.update(
        o=torch.empty((b, h, sq, dv), dtype=dtype, device="cuda"),
        stats=torch.empty((b, h, sq), device="cuda"),
        kt=torch.tensor([[2, 3, 0], [1, -1, -1]], dtype=torch.int32, device="cuda").view(b, 1, 3, 1),
        vt=torch.tensor([[4, 1, 5], [2, -1, -1]], dtype=torch.int32, device="cuda").view(b, 1, 3, 1),
        ql=torch.tensor([sq, 3], dtype=torch.int32, device="cuda"),
        kl=torch.tensor([17, 1], dtype=torch.int32, device="cuda"),
    )
    api = SdpaFwdDslSm90(
        *(bufs[name] for name in ("q", "k", "v", "o", "stats")),
        paged_page_size=page,
        paged_max_seq_len_kv=17,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
    )
    api.compile()
    api.execute(
        *(bufs[name] for name in ("q", "k", "v", "o", "stats")),
        seq_q_lens=bufs["ql"],
        seq_kv_lens=bufs["kl"],
        block_table=bufs["kt"],
        block_table_v=bufs["vt"],
    )
    paging._check(bufs, page)


@pytest.mark.parametrize("h,kh", [(24, 2), (128, 1)])
def test_paged_partial_pack_groups(h, kh):
    graph, nodes, bufs, ws, _, _ = paging._case(
        torch.bfloat16,
        kh,
        16,
        False,
        True,
        h=h,
        dims=(512, 512),
        sq=1,
        packed=True,
        split_kv=3,
        sink=True,
        explicit_plan=True,
        independent_tables=True,
    )
    graph.execute({nodes[name]: tensor for name, tensor in bufs.items()}, ws)
    paging._check(bufs, 16)
