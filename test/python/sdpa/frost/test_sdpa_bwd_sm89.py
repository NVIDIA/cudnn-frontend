# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM89 (Ada) native half backward: ``sdpa_bwd_sm89``, the SM80 mma.sync chain under the 99 KiB tile rule.

The cuDNN backend serves Ada backward only up to D128, so above it this row is the only provider. The GPU cases
check every gradient against an FP64 reference; the declaration cases need no GPU.
"""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

ENGINE, ENGINE_ID = "sdpa_bwd_sm89", 20635
D64_ENGINE, D64_ENGINE_ID = "sdpa_bwd_sm89_f16_d64", 20636
_ON_SM89 = pytest.mark.skipif(_SM != 89, reason="requires SM89")


def _spec(name):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    return next(s for s in ENGINE_SPECS if s.name == name)


@pytest.mark.L0
@pytest.mark.parametrize("name,engine_id", [(ENGINE, ENGINE_ID), (D64_ENGINE, D64_ENGINE_ID)])
def test_rows_preserve_the_sm80_domain_with_fixed_ada_owners(name, engine_id):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd.heuristics import TRAIL_ONLY_ROWS

    sm80, sm89 = _spec("sdpa_bwd_sm80").capabilities, _spec(name).capabilities
    assert (sm89.sm_lo, sm89.sm_hi) == (89, 89)
    ownership = {"sm_lo", "sm_hi", "sm_extra", "implementation", "select_implementation", "delegated"}
    assert sm89.sm_extra == frozenset()
    assert {f: getattr(sm89, f) for f in sm89.__dataclass_fields__ if f not in ownership} == {
        f: getattr(sm80, f) for f in sm80.__dataclass_fields__ if f not in ownership
    }
    assert sm89.implementation == name and not sm89.delegated
    family = next(f for f in MANIFEST if name in f.slots)
    assert family.offered_ids()[name] == engine_id
    assert _spec(name).policy in TRAIL_ONLY_ROWS


@pytest.mark.L0
@_ON_SM89
@requires_dsl
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,skv", [(64, 128), (128, 256)])
def test_fixed_d64_owner_rebind_replay(dtype, sq, skv):
    from test_sdpa_bwd_prepared_sm80 import _case, _check

    _check(_case(64, hk=4, dtype=dtype, causal=False, layout="bshd", sq=sq, skv=skv, engine=D64_ENGINE))


@pytest.mark.L0
@pytest.mark.parametrize("dims,tile", [((64, 64), 64), ((128, 128), 64), ((192, 128), 32), ((256, 256), 32)])
def test_tiles_fit_the_ada_shared_memory_budget(dims, tile):
    from cudnn.sdpa.bwd.sm120_thd_contract import native_tiles

    m, n = native_tiles(*dims)
    assert (m, n) == (tile, tile)
    # One Q/dO stage, resident K/V, FP32 P and half transposed dS: inside the 99 KiB opt-in limit of cc 8.6 / 8.9.
    assert 2 * (m + n) * sum(dims) + 6 * m * n <= 99 * 1024


_DENSE_L0 = {((256, 256), "causal"), ((192, 128), "br_swa"), ((160, 160), "full")}


def _dense_params():
    for dims in [(256, 256), (192, 128), (160, 160), (200, 136), (128, 128), (64, 64)]:
        for mask in ("full", "causal", "swa", "br_swa"):
            for dtype in (torch.float16, torch.bfloat16):
                for deterministic in (False, True):
                    for padded in (False, True):
                        level = pytest.mark.L0 if (dims, mask) in _DENSE_L0 and not padded else pytest.mark.L1
                        yield pytest.param(
                            dims,
                            mask,
                            dtype,
                            deterministic,
                            padded,
                            marks=level,
                            id=f"{dims[0]}x{dims[1]}-{mask}-{str(dtype)[6:]}-det{int(deterministic)}-pad{int(padded)}",
                        )


@pytest.mark.parametrize("dims,mask,dtype,deterministic,padded", list(_dense_params()))
@_ON_SM89
@requires_dsl
def test_graph_dense(dims, mask, dtype, deterministic, padded):
    import cudnn
    from d128_thd_test_utils import GRADS, make_workspace, reference_sequences
    from frost_test_utils import select_engine

    d_qk, d_v = dims
    h, hk, sq, sk = 6, 2, 33, 49
    lens_q = (sq - 7, 0) if padded else (sq, sq)
    lens_k = (sk - 3, 0) if padded else (sk, sk)
    seqs = reference_sequences(lens_q, lens_k, h, dtype, mask != "full", 317, d_qk, d_v, hk, mask == "br_swa", 7 if "swa" in mask else None)
    bufs, bases = {}, {}
    for role in ("q", "k", "v", "o", "do", *GRADS):
        is_q = role in ("q", "o", "do", "dq")
        heads, n = (h if is_q else hk), (sq if is_q else sk)
        d = d_v if role in ("v", "o", "do", "dv") else d_qk
        base = torch.full((2, n, heads, d + 8), -7.0, device="cuda", dtype=dtype)
        buf = base[..., :d].transpose(1, 2)
        if role not in GRADS:
            buf.fill_(float("nan"))
            for bi, seq in enumerate(seqs):
                buf[bi, :, : seq[role].shape[0]].copy_(seq[role].to(dtype).transpose(0, 1))
        bufs[role], bases[role] = buf, base
    stats_base = torch.full((2, h, sq * 2, 1), float("nan"), device="cuda")
    bufs["stats"] = stats_base[:, :, ::2]
    for bi, seq in enumerate(seqs):
        bufs["stats"][bi, :, : seq["stats"].shape[0], 0].copy_(seq["stats"].T)
    if padded:
        bufs["seq_q"] = torch.tensor(lens_q, dtype=torch.int32, device="cuda").view(2, 1, 1, 1)
        bufs["seq_kv"] = torch.tensor(lens_k, dtype=torch.int32, device="cuda").view(2, 1, 1, 1)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    refs = {r: graph.tensor_like(t, name=r) for r, t in bufs.items() if r not in GRADS}
    outs = graph.sdpa_backward(
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=d_qk**-0.5,
        use_causal_mask=mask in ("causal", "swa"),
        use_causal_mask_bottom_right=mask == "br_swa",
        diagonal_band_left_bound=8 if "swa" in mask else None,
        use_deterministic_algorithm=deterministic,
        use_padding_mask=padded,
        seq_len_q=refs.get("seq_q"),
        seq_len_kv=refs.get("seq_kv"),
    )
    for r, out in zip(GRADS, outs):
        out.set_output(True).set_dim(bufs[r].shape).set_stride(bufs[r].stride()).set_data_type(io)
        refs[r] = out
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, ENGINE)
    graph.check_support()
    graph.build_plans()
    workspace = make_workspace(graph.get_workspace_size())
    pack = {refs[r]: t for r, t in bufs.items()}
    previous = None
    for _ in range(3 if deterministic else 1):
        workspace.tensor.fill_(0xFF)
        for r in GRADS:
            bufs[r].fill_(float("nan"))
        graph.execute(pack, workspace.tensor)
        torch.cuda.synchronize()
        for r in GRADS:
            expected = torch.zeros(bufs[r].shape, dtype=torch.float64)
            for bi, seq in enumerate(seqs):
                expected[bi, :, : seq[r].shape[0]] = seq[r].transpose(0, 1)
            torch.testing.assert_close(bufs[r].cpu().double(), expected, atol=0.006 if dtype == torch.float16 else 0.035, rtol=0.03)
            width = d_v if r == "dv" else d_qk
            assert torch.all(bases[r][..., width:] == -7), f"{r}: row padding overwritten"
        if deterministic:
            from determinism_utils import assert_bitwise, snapshot

            if previous is not None:
                assert_bitwise({r: bufs[r] for r in GRADS}, previous, "deterministic SM89 dense")
            previous = snapshot({r: bufs[r] for r in GRADS})


def _thd_params():
    for dims in [(256, 256), (192, 128), (128, 128)]:
        for mask in ("full", "causal", "swa", "br_swa"):
            for dtype in (torch.float16, torch.bfloat16):
                for deterministic in (False, True):
                    level = pytest.mark.L0 if dims == (256, 256) and mask in ("causal", "br_swa") and not deterministic else pytest.mark.L1
                    yield pytest.param(dims, mask, dtype, deterministic, marks=level, id=f"{dims[0]}x{dims[1]}-{mask}-{str(dtype)[6:]}-det{int(deterministic)}")


@pytest.mark.parametrize("dims,mask,dtype,deterministic", list(_thd_params()))
@_ON_SM89
@requires_dsl
def test_graph_thd_with_physical_origins(dims, mask, dtype, deterministic):
    from d128_thd_test_utils import check_case, make_case, make_graph, make_workspace, poison, variant_pack

    case = make_case(
        (33, 0, 17),
        (19, 49, 0),
        heads=6,
        kv_heads=2,
        d_qk=dims[0],
        d_v=dims[1],
        dtype=dtype,
        causal=mask != "full",
        bottom_right=mask == "br_swa",
        deterministic=deterministic,
        window_left=7 if "swa" in mask else None,
        stats_layout="HN",
    )
    graph, refs = make_graph(case, engine_name=ENGINE, engine_id=ENGINE_ID)
    pack, workspace = variant_pack(case, refs), make_workspace(graph.get_workspace_size())
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,dv", [(256, 256), (192, 128), (128, 128)])
@pytest.mark.parametrize("bias_batch", [1, 2])
@pytest.mark.parametrize("deterministic", [False, True])
@_ON_SM89
@requires_dsl
def test_bias_dbias_sink_and_padding(dtype, d, dv, bias_batch, deterministic):
    import cudnn
    from test_sdpa_bwd_prepared_sm80 import _case, _check

    if deterministic and bias_batch == 1:
        # A broadcast bias reduces dBias over B through unordered atomics, so the deterministic claim declines it.
        # Backend availability also depends on version and Stats strides, not just D128. If every engine
        # declines, planning reports Ada's specific reason; otherwise the strict selector reports its missing plan.
        declined = f"{ENGINE}: deterministic dBias requires a per-batch bias|no plan for engine '{ENGINE}'"
        with pytest.raises((cudnn.cudnnGraphNotSupportedError, AssertionError), match=declined):
            _case(d, dv, dtype=dtype, features=True, bias_batch=bias_batch, deterministic=deterministic, engine=ENGINE)
        return
    _check(_case(d, dv, dtype=dtype, features=True, bias_batch=bias_batch, deterministic=deterministic, engine=ENGINE))


@pytest.mark.L0
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [256, 128])
@_ON_SM89
@requires_dsl
def test_zero_scale(causal, d):
    from test_sdpa_bwd_prepared_sm80 import _case, _check

    _check(_case(d, causal=causal, features=True, scale=0.0, engine=ENGINE))


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@_ON_SM89
@requires_dsl
def test_deterministic_replay_and_capture(dtype):
    from d128_thd_test_utils import GRADS, check_case, copy_case_storage, make_case, make_graph, make_workspace, poison, variant_pack
    from determinism_utils import assert_bitwise, snapshot

    options = dict(heads=6, kv_heads=2, d_qk=256, d_v=256, dtype=dtype, causal=True, bottom_right=True, window_left=7, deterministic=True)
    case = make_case((33, 0, 17), (19, 49, 0), **options)
    graph, refs = make_graph(case, engine_name=ENGINE, engine_id=ENGINE_ID)
    pack, workspace = variant_pack(case, refs), make_workspace(graph.get_workspace_size())
    first = None
    for _ in range(3):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)
        got = snapshot({r: case.tensors[r] for r in GRADS})
        if first is not None:
            assert_bitwise(got, first, "SM89 deterministic repeat")
        first = got
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(pack, workspace.tensor)
    changed = make_case((33, 0, 17), (19, 49, 0), seed=151, **options)
    copy_case_storage(case, changed)
    for _ in range(2):
        poison(case, workspace)
        capture.replay()
        check_case(case, workspace)
    capture.reset()


@pytest.mark.L0
@pytest.mark.parametrize("d,sq,skv,expected_first", [(256, 128, 128, ENGINE), (128, 128, 128, None), (64, 128, 128, D64_ENGINE), (64, 128, 256, None)])
@_ON_SM89
@requires_dsl
def test_default_plan_order(d, sq, skv, expected_first, monkeypatch):
    """The backend declines D256; the fixed D64 owner leads only for qualifying squares."""
    import cudnn

    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)

    b, h = 2, 4
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def dims(length):
        return [b, h, length, d], [length * h * d, d, h * d, 1]

    tensors = {}
    for name, length in (("q", sq), ("k", skv), ("v", skv), ("o", sq), ("do", sq)):
        shape, stride = dims(length)
        tensors[name] = graph.tensor(name=name, dim=shape, stride=stride)
    stats = graph.tensor(name="stats", dim=[b, h, sq, 1], stride=[h * sq, sq, 1, 1], data_type=cudnn.data_type.FLOAT)
    outs = graph.sdpa_backward(
        q=tensors["q"], k=tensors["k"], v=tensors["v"], o=tensors["o"], dO=tensors["do"], stats=stats, attn_scale=d**-0.5, use_causal_mask=d != 64
    )
    for out, length in zip(outs, (sq, skv, skv)):
        shape, stride = dims(length)
        out.set_output(True).set_dim(shape).set_stride(stride).set_data_type(cudnn.data_type.BFLOAT16)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
    owner = D64_ENGINE if d == 64 else ENGINE
    assert owner in names, names
    if expected_first is None:
        assert names.index(owner) > 0, names
    else:
        assert names[0] == expected_first, names
    if d == 256:
        assert names == [ENGINE], names
