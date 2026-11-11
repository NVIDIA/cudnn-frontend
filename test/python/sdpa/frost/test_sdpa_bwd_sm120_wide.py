# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM120 wide backward qualification."""

import pytest
import torch
from d128_thd_test_utils import make_case, make_adapter, make_workspace, poison, check_case
from test_sdpa_bwd_thd_sm120 import _adapter_arguments
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM not in (120, 121), reason="requires SM120 or SM121")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(512, 512), (504, 248), (264, 8)])
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_wide_thd(dtype, dims, deterministic):
    from cudnn.sdpa.bwd.api_dsl_sm120_wide import SdpaBwdDslSm120Wide

    case = make_case(
        (33, 0, 17), (19, 49, 0), heads=6, kv_heads=2, d_qk=dims[0], d_v=dims[1], dtype=dtype, causal=True, bottom_right=True, deterministic=deterministic
    )
    api = make_adapter(case, api_type=SdpaBwdDslSm120Wide)
    workspace = make_workspace(api.scratch_workspace_bytes())
    for _ in range(2):
        poison(case, workspace)
        api.execute(**_adapter_arguments(case), workspace=workspace.tensor)
        check_case(case, workspace)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("mask", ["full", "causal", "swa", "br_swa"])
def test_graph_wide_thd(dtype, deterministic, mask):
    from test_sdpa_bwd_thd_sm120 import _graph_case

    case = make_case(
        (33, 0, 17),
        (19, 49, 0),
        heads=6,
        kv_heads=2,
        d_qk=512,
        d_v=512,
        dtype=dtype,
        causal=mask != "full",
        bottom_right=mask == "br_swa",
        deterministic=deterministic,
        window_left=7 if "swa" in mask else None,
        stats_layout="HN",
    )
    graph, _, pack, workspace = _graph_case(case)
    for _ in range(2):
        poison(case, workspace)
        graph.execute(pack, workspace.tensor)
        check_case(case, workspace)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("mask", ["full", "causal", "swa", "br_swa"])
@pytest.mark.parametrize("padded", [False, True])
def test_graph_wide_dense(dtype, deterministic, mask, padded):
    import cudnn
    from d128_thd_test_utils import reference_sequences, GRADS
    from frost_test_utils import select_engine

    h, hk, sq, sk = 6, 2, 33, 49
    lens_q = (sq - 7, 0) if padded else (sq, sq)
    lens_k = (sk - 3, 0) if padded else (sk, sk)
    seqs = reference_sequences(lens_q, lens_k, h, dtype, mask != "full", 317, 512, 512, hk, mask == "br_swa", 7 if "swa" in mask else None)
    bufs = {}
    bases = {}
    for role in ("q", "k", "v", "o", "do", *GRADS):
        is_q = role in ("q", "o", "do", "dq")
        heads, n, d = (h if is_q else hk), (sq if is_q else sk), 512
        base = torch.full((2, n, heads, d + 8), -7.0, device="cuda", dtype=dtype)
        buf = base[..., :d].transpose(1, 2)
        if role not in GRADS:
            buf.fill_(float("nan"))
            for bi, seq in enumerate(seqs):
                buf[bi, :, : seq[role].shape[0]].copy_(seq[role].to(dtype).transpose(0, 1))
        bufs[role], bases[role] = buf, base
    sb = torch.full((2, h, sq * 2, 1), float("nan"), device="cuda")
    bufs["stats"] = sb[:, :, ::2]
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
        attn_scale=512**-0.5,
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
    select_engine(graph, "sdpa_bwd_sm120")
    graph.check_support()
    graph.build_plans()
    workspace = make_workspace(graph.get_workspace_size())
    pack = {refs[r]: t for r, t in bufs.items()}
    previous = None
    for _ in range(3):
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
            assert torch.all(bases[r][..., 512:] == -7)
        if deterministic:
            from determinism_utils import snapshot, assert_bitwise

            if previous is not None:
                assert_bitwise({r: bufs[r] for r in GRADS}, previous, "deterministic wide dense")
            previous = snapshot({r: bufs[r] for r in GRADS})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deterministic_replay_batch_and_capture(dtype):
    from d128_thd_test_utils import GRADS, copy_case_storage
    from determinism_utils import assert_bitwise, assert_golden, assert_replays
    from test_sdpa_bwd_thd_sm120 import _graph_case

    options = dict(heads=6, kv_heads=2, d_qk=512, d_v=512, dtype=dtype, causal=True, bottom_right=True, window_left=7, deterministic=True)
    target = make_case((17,), (19,), **options).sequences[0]
    neighbor = make_case((17,), (19,), seed=151, **options).sequences[0]
    changed_neighbor = make_case((17,), (19,), seed=153, **options).sequences[0]
    expected = None
    for order in ([0], [0, 1, 0], [1, 0, 1], [0] * 129):
        sequences = [target if i == 0 else neighbor for i in order]
        case = make_case((17,) * len(order), (19,) * len(order), sequences=sequences, **options)
        graph, _, pack, workspace = _graph_case(case)
        target_index = order.index(0)

        def outputs(index):
            return {r: case.tensors[r][0, case.origins[r][index] : case.origins[r][index] + (17 if r == "dq" else 19)] for r in GRADS}

        def run():
            poison(case, workspace)
            graph.execute(pack, workspace.tensor)
            check_case(case, workspace)
            return outputs(target_index)

        got = assert_replays(run, repeats=3)
        if expected is None:
            expected = got
        for i, source in enumerate(order):
            if source == 0:
                assert_bitwise(outputs(i), expected, f"wide batch {len(order)} sequence {i}")
        if order == [0, 1, 0]:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                graph.execute(pack, workspace.tensor)
            changed = make_case((17,) * 3, (19,) * 3, sequences=[target, changed_neighbor, target], **options)
            copy_case_storage(case, changed)
            poison(case, workspace)
            capture.replay()
            check_case(case, workspace)
            assert_bitwise(outputs(0), expected, "wide changed-neighbor replay")
            assert_bitwise(outputs(2), expected, "wide duplicate replay")
            capture.reset()
    assert_golden(f"sdpa-sm120-wide-bwd-d512-gqa-swa-{dtype}", {r: target[r] for r in ("q", "k", "v", "o", "do", "stats")}, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
def test_wide_rebind_and_changed_lengths_capture(dtype, deterministic):
    from d128_thd_test_utils import GRADS, copy_case_storage, variant_pack
    from determinism_utils import snapshot, assert_bitwise
    from test_sdpa_bwd_thd_sm120 import _graph_case

    options = dict(
        heads=6,
        kv_heads=2,
        d_qk=504,
        d_v=248,
        dtype=dtype,
        stats_layout="HN",
        causal=True,
        bottom_right=True,
        window_left=17,
        deterministic=deterministic,
        envelope=(129, 129),
        capacities=(512, 512),
    )
    old = make_case((33, 0, 17), (19, 65, 0), **options)
    fresh = make_case((1, 65, 0), (49, 0, 33), seed=151, layout_shift=7, **options)
    changed = make_case((0, 97, 1), (19, 0, 65), seed=153, layout_shift=13, **options)
    graph, refs, pack, workspace = _graph_case(old)
    fresh_workspace = make_workspace(graph.get_workspace_size())
    graph.execute(pack, workspace.tensor)
    check_case(old, workspace)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        graph.execute(pack, workspace.tensor)
    poison(fresh, fresh_workspace)
    graph.execute(variant_pack(fresh, refs), fresh_workspace.tensor)
    check_case(fresh, fresh_workspace)
    untouched = snapshot({r: fresh.tensors[r] for r in GRADS})
    copy_case_storage(old, changed)
    for _ in range(3):
        poison(old, workspace)
        capture.replay()
        check_case(old, workspace)
    assert_bitwise({r: fresh.tensors[r] for r in GRADS}, untouched, "wide retained replay changed fresh bindings")
    capture.reset()
