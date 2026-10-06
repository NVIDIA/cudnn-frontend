# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Hopper blk128 correctness, mutable metadata, and execution contracts."""

import pytest
import torch
from cudnn import BSA

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def hopper_only():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native SM90 blk128 requires Hopper")


def case(
    sq=256, sk=384, d=128, dv=128, ratio=1, dtype=torch.bfloat16, layout="bhsd", variable=False, sizes_rank=0, active_count=None, valid_size=None, b=2, h=4
):
    torch.manual_seed(109)
    q = torch.randn((b, h, sq, d), dtype=dtype, device="cuda")
    k = torch.randn((b, h // ratio, sk, d), dtype=dtype, device="cuda")
    v = torch.randn((b, h // ratio, sk, dv), dtype=dtype, device="cuda")
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    c = min(nk, 3 if active_count is None else active_count)
    indices = torch.rand(b, h, nq, nk, device="cuda").argsort(-1)[..., :c].int().contiguous()
    counts = torch.randint(0, c + 1, (b, h, nq), dtype=torch.int32, device="cuda") if variable else None
    if counts is not None:
        counts[0, 0, 0] = 0
        indices.masked_fill_(torch.arange(c, device="cuda") >= counts[..., None], 999999)
    sizes = None
    if sizes_rank:
        shape = {1: (nk,), 2: (b, nk), 3: (b, h, nk)}[sizes_rank]
        sizes = torch.randint(1, 129, shape, device="cuda", dtype=torch.int32)
        if valid_size is not None:
            sizes.fill_(valid_size)
        sizes[..., -1].clamp_(max=sk - 128 * (nk - 1))
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return q, k, v, indices, counts, sizes, c


def reference(q, k, v, indices, counts, sizes, count, scale, layout, do=None):
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        do = do.transpose(1, 2) if do is not None else None
    q, k, v = (t.double().detach().requires_grad_(do is not None) for t in (q, k, v))
    b, h, sq, _ = q.shape
    sk = k.shape[2]
    selected = torch.zeros((b, h, sq, sk), dtype=torch.bool, device=q.device)
    ids = indices.cpu()
    nums = counts.cpu() if counts is not None else None
    bs = sizes.cpu() if sizes is not None else None
    for bi in range(b):
        for hi in range(h):
            for m in range(ids.shape[2]):
                n = count if nums is None else int(nums[bi, hi, m])
                for pos in range(n):
                    block = int(ids[bi, hi, m, pos])
                    size = 128 if bs is None else int(bs[block] if bs.ndim == 1 else bs[bi, block] if bs.ndim == 2 else bs[bi, hi, block])
                    selected[bi, hi, m * 128 : min(sq, (m + 1) * 128), block * 128 : min(sk, block * 128 + size)] = True
    kr = k.repeat_interleave(h // k.shape[1], dim=1)
    vr = v.repeat_interleave(h // v.shape[1], dim=1)
    scores = (q @ kr.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~selected, -torch.inf)
    lse = scores.logsumexp(-1)
    empty = ~selected.any(-1, keepdim=True)
    probs = torch.softmax(scores.masked_fill(empty, 0), -1).masked_fill(~selected, 0)
    out = probs @ vr
    grads = None
    if do is not None:
        out.backward(do.double())
        grads = tuple(t.grad for t in (q, k, v))
    if layout == "bshd":
        out = out.transpose(1, 2)
        grads = tuple(t.transpose(1, 2) for t in grads) if grads is not None else None
    return out.detach(), lse.detach(), grads


def check(
    sq=256,
    sk=384,
    d=128,
    dv=128,
    ratio=1,
    dtype=torch.bfloat16,
    layout="bhsd",
    variable=False,
    sizes_rank=0,
    splits=1,
    backward=True,
    bucket=1,
    scale=0.137,
    active_count=None,
    valid_size=None,
    strided=False,
    b=2,
    h=4,
):
    q, k, v, ids, counts, sizes, count = case(sq, sk, d, dv, ratio, dtype, layout, variable, sizes_rank, active_count, valid_size, b, h)
    if strided:

        def stride_view(t, step):
            storage = torch.empty((*t.shape[:-1], t.shape[-1] * step), dtype=t.dtype, device=t.device)
            view = storage[..., ::step]
            view.copy_(t)
            return view

        ids = stride_view(ids, 2)
        counts = stride_view(counts, 2) if counts is not None else None
        sizes = stride_view(sizes, 2) if sizes is not None else None

        def row_view(t):
            storage = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype, device=t.device)
            view = storage[..., : t.shape[-1]]
            view.copy_(t)
            return view

        q, k, v = (row_view(t) for t in (q, k, v))
    opts = dict(block_sparse_num=count, q2k_block_nums=counts, block_sizes=sizes, sparse_block_size=128, softmax_scale=scale, layout=layout)
    result = BSA.block_sparse_attention_forward(q, k, v, ids, allow_empty_block_nums=variable, kv_splits=splits, **opts)
    assert result["o_tensor"].is_contiguous()
    assert result["lse_tensor"].is_contiguous()
    do = torch.randn_like(result["o_tensor"]) if backward else None
    ro, rl, rg = reference(q, k, v, ids, counts, sizes, count, scale, layout, do)
    torch.testing.assert_close(result["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(result["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)
    if backward:
        # Noncompact output rows exercise native writes, not temporary copies.
        backing = tuple(torch.full((*t.shape[:-1], t.shape[-1] * 2), 17.0, dtype=t.dtype, device=t.device) for t in (q, k, v))
        dest = tuple(storage[..., : t.shape[-1]] for storage, t in zip(backing, (q, k, v)))
        gradients = BSA.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            result["o_tensor"],
            result["lse_tensor"],
            ids,
            dq_tensor=dest[0],
            dk_tensor=dest[1],
            dv_tensor=dest[2],
            bucket_size_blocks=bucket,
            **opts,
        )
        for got, expected, buffer, storage in zip(gradients, rg, dest, backing):
            assert got.data_ptr() == buffer.data_ptr()
            # TMA output descriptors must preserve row strides and never touch
            # the allocation's padding, including partial KV tiles.
            torch.testing.assert_close(storage[..., buffer.shape[-1] :], torch.full_like(buffer, 17.0), atol=0, rtol=0)
            # With one token/block, dK accumulates BF16 rounding across many rows.
            # Same-input blk64 reproduces the identical 0.0512 maximum dK error;
            # bound both elementwise and relative L2 error for this extreme case.
            torch.testing.assert_close(got.double(), expected, atol=6e-2 if valid_size == 1 else 3e-2, rtol=3e-2)
            error = torch.linalg.vector_norm(got.double() - expected)
            assert error <= 0.01 * torch.linalg.vector_norm(expected) + 1e-5


@pytest.mark.L0
@pytest.mark.parametrize("layout,variable,sq,sk", [("bhsd", False, 256, 384), ("bshd", True, 129, 257)])
def test_forward_backward(layout, variable, sq, sk):
    check(layout=layout, variable=variable, sq=sq, sk=sk, sizes_rank=2 if variable else 0)


@pytest.mark.L1
@pytest.mark.parametrize("d,dv", [(d, dv) for d in (64, 96, 128) for dv in (64, 96, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dimensions(d, dv, dtype):
    check(d=d, dv=dv, dtype=dtype, backward=False, ratio=4, layout="bshd")


@pytest.mark.L1
@pytest.mark.parametrize("sq,sk", [(1, 1), (63, 127), (64, 129), (127, 257), (129, 513), (257, 385)])
@pytest.mark.parametrize("variable", [False, True])
def test_tails(sq, sk, variable):
    check(sq=sq, sk=sk, variable=variable, sizes_rank=1, bucket=2)


@pytest.mark.L1
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("splits", [1, 2, 7])
def test_split_partial_gqa(rank, splits):
    check(sq=129, sk=513, variable=True, sizes_rank=rank, splits=splits, ratio=2, backward=False)


@pytest.mark.L0
def test_graph_mutable_metadata_no_recompile():
    q, k, v, ids, counts, sizes, count = case(variable=True)
    from cudnn.block_sparse_attention import _sm90_blk128 as native

    do = torch.randn_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        f = BSA.block_sparse_attention_forward(q, k, v, ids, count, q2k_block_nums=counts, allow_empty_block_nums=True, sparse_block_size=128)
        g = BSA.block_sparse_attention_backward(do, q, k, v, f["o_tensor"], f["lse_tensor"], ids, count, q2k_block_nums=counts, sparse_block_size=128)
        return f, g

    with torch.cuda.stream(stream):
        run()
        run()
        sizes_before = len(native._FWD_CACHE), len(native._BWD_CACHE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            f, g = run()
        # Same pointers now describe an entirely empty graph; all output gradients must be reset.
        counts.zero_()
        torch.cuda.set_sync_debug_mode("error")
        try:
            graph.replay()
            run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    assert sizes_before == (len(native._FWD_CACHE), len(native._BWD_CACHE))
    assert torch.count_nonzero(f["o_tensor"]) == 0
    assert torch.isneginf(f["lse_tensor"]).all()
    for t in g:
        assert torch.count_nonzero(t) == 0

    # Change both the graph and counts at the same addresses, then replay again.
    ids.copy_(torch.arange(count - 1, -1, -1, device=ids.device, dtype=ids.dtype))
    counts.fill_(count)
    graph.replay()
    ro, rl, rg = reference(q, k, v, ids, counts, None, count, 128**-0.5, "bhsd", do)
    torch.testing.assert_close(f["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(f["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)
    for got, expected in zip(g, rg):
        torch.testing.assert_close(got.double(), expected, atol=3e-2, rtol=3e-2)
    assert sizes_before == (len(native._FWD_CACHE), len(native._BWD_CACHE))


@pytest.mark.L1
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
@pytest.mark.parametrize("size", [1, 63, 64, 65, 127, 128])
@pytest.mark.parametrize("bucket", [1, 192])
def test_partial_boundary_scale(scale, size, bucket):
    check(sq=129, sk=257, sizes_rank=2, valid_size=size, scale=scale, splits=2, layout="bshd", bucket=bucket)


@pytest.mark.L1
@pytest.mark.parametrize("variable,bucket", [(False, 1), (False, 192), (True, 3)])
def test_pipeline_ring(variable, bucket):
    check(sq=1025, sk=2049, active_count=16, variable=variable, sizes_rank=2, bucket=bucket)


@pytest.mark.L1
@pytest.mark.parametrize("sq", [65, 191, 193])
@pytest.mark.parametrize("bucket", [1, 256])
def test_backward_q64_subtiles(sq, bucket):
    # One-token second Q64 halves and entirely absent second halves, without
    # expanding the Q128 CSR. Exercise both dQ head partitions and KV buckets.
    check(sq=sq, sk=385, active_count=4, variable=True, sizes_rank=2, bucket=bucket, strided=True)


@pytest.mark.L1
@pytest.mark.parametrize("active_count", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("variable", [False, True])
def test_forward_pipeline_lengths(active_count, variable):
    # Exercise prologue/drain, both ring phases, and empty rows alongside active
    # rows. Variable counts use two stages even when the maximum count is one.
    check(sq=129, sk=1153, active_count=active_count, variable=variable, sizes_rank=2, backward=False)


@pytest.mark.L1
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
@pytest.mark.parametrize("variable", [False, True])
def test_forward_partial_prologue(scale, variable):
    # Begin with the one-token physical KV tail, then visit full blocks. Variable
    # counts include empty rows and exercise the scheduler's short-list paths.
    q, k, v, ids, counts, sizes, count = case(sq=129, sk=1153, dv=96, ratio=2, dtype=torch.float16, active_count=8, sizes_rank=2, variable=variable)
    order = torch.cat((torch.tensor([9], device=ids.device, dtype=ids.dtype), torch.arange(count - 1, device=ids.device, dtype=ids.dtype)))
    ids.copy_(order)
    sizes.fill_(128)
    sizes[:, -1] = 1
    result = BSA.block_sparse_attention_forward(
        q, k, v, ids, count, q2k_block_nums=counts, allow_empty_block_nums=variable, block_sizes=sizes, sparse_block_size=128, softmax_scale=scale
    )
    ro, rl, _ = reference(q, k, v, ids, counts, sizes, count, scale, "bhsd")
    torch.testing.assert_close(result["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(result["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["fp16_backward", "gqa_backward", "head_specific_backward", "unaligned_rows"])
def test_unsupported_contracts(kind):
    q, k, v, ids, counts, sizes, count = case(
        dtype=torch.float16 if kind == "fp16_backward" else torch.bfloat16,
        ratio=2 if kind == "gqa_backward" else 1,
        sizes_rank=3 if kind == "head_specific_backward" else 0,
    )
    if kind == "unaligned_rows":
        q = torch.empty((*q.shape[:-1], 129), device=q.device, dtype=q.dtype)[..., :128]
        with pytest.raises(NotImplementedError, match="aligned"):
            BSA.block_sparse_attention_forward(q, k, v, ids, count, sparse_block_size=128)
    else:
        with pytest.raises((NotImplementedError, ValueError)):
            BSA.block_sparse_attention_backward(
                torch.empty_like(q),
                q,
                k,
                v,
                torch.empty_like(q),
                torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
                ids,
                count,
                block_sizes=sizes,
                sparse_block_size=128,
            )


@pytest.mark.L1
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("bucket", [1, 192])
def test_native_strided_inputs_and_metadata(layout, bucket):
    check(sq=129, sk=257, variable=True, sizes_rank=2, strided=True, layout=layout, bucket=bucket)


@pytest.mark.L0
def test_dsl_floor_before_kernel_import(monkeypatch):
    import builtins
    from cudnn.block_sparse_attention import _interface

    monkeypatch.setattr(_interface, "_cutlass_dsl_version", lambda: (4, 6, 1))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert "sm90_blk128.bsa_" not in name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    q, k, v, ids, _, _, count = case()
    with pytest.raises(RuntimeError, match="requires nvidia-cutlass-dsl"):
        BSA.block_sparse_attention_forward(q, k, v, ids, count, sparse_block_size=128)


@pytest.mark.L1
@pytest.mark.parametrize("d,dv,dtype", [(64, 96, torch.float16), (96, 128, torch.float16), (128, 64, torch.bfloat16), (96, 96, torch.bfloat16)])
def test_split_mixed_dimensions(d, dv, dtype):
    check(sq=129, sk=257, d=d, dv=dv, dtype=dtype, layout="bshd", ratio=4, variable=True, sizes_rank=3, splits=7, backward=False)


@pytest.mark.L1
def test_maximum_splits():
    check(sq=1, sk=1, d=64, dv=96, dtype=torch.float16, layout="bshd", ratio=4, splits=256, backward=False)


@pytest.mark.L1
@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("bucket", [1, 256])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_dynamic_shape_compile_reuse(monkeypatch, variable, bucket, layout):
    from cudnn.block_sparse_attention import _sm90_blk128 as native

    monkeypatch.setattr(native, "_FWD_CACHE", {})
    monkeypatch.setattr(native, "_BWD_CACHE", {})
    for b, h, sq, sk, count, strided in [(2, 4, 193, 385, 2, False), (3, 6, 321, 641, 3, False), (1, 2, 449, 769, 4, True), (2, 4, 577, 897, 5, True)]:
        check(b=b, h=h, sq=sq, sk=sk, active_count=count, variable=variable, bucket=bucket, strided=strided, layout=layout)
        # A strided metadata view has no unit stride and needs a distinct tensor
        # type; subsequent sizes within each layout must reuse the compiled code.
        assert len(native._FWD_CACHE) == (2 if strided else 1)
        assert len(native._BWD_CACHE) == 1
