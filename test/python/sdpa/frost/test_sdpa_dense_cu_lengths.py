# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense cumulative lengths preserve the ordinary per-batch length contract."""

import cudnn
import pytest
import torch

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines

pytestmark = pytest.mark.L0
_ARCHES = [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (10, 7), (12, 0)]


def _graph(dtype, forms, *, ambiguous=False, padding=True):
    b, h, hk, sq, skv, d = 3, 4, 2, 128, 256, 128
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    def port(name, heads, rows):
        return g.tensor(name=name, dim=(b, heads, rows, d), stride=(rows * heads * d, d, heads * d, 1), data_type=dtype)

    q, k, v = port("q", h, sq), port("k", hk, skv), port("v", hk, skv)
    lengths = {}
    for side, cumulative in zip(("q", "kv"), forms, strict=True):
        if cumulative is None:
            continue
        name = ("cu_seq_len_" if cumulative else "seq_len_") + side
        lengths[name] = g.tensor(name=name, dim=(b + int(cumulative), 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        if ambiguous and cumulative:
            name = "seq_len_" + side
            lengths[name] = g.tensor(name=name, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=d**-0.5, generate_stats=True, use_padding_mask=padding, **lengths)
    o.set_output(True).set_data_type(dtype).set_dim((b, h, sq, d)).set_stride((sq * h * d, d, h * d, 1))
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((b, h, sq, 1)).set_stride((h * sq, sq, 1, 1))
    return g


@pytest.fixture(params=_ARCHES, ids=lambda cc: f"sm{cc[0]}{cc[1]}")
def architecture(request, monkeypatch):
    cc = request.param
    from cudnn.frost import buffers

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: cc)
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(buffers, "cutedsl_arch_requirement_error", lambda *args: None)
    return cc


def _offered(graph):
    facts = ga.analyze(graph)
    assert facts.invalid is None, facts.invalid
    assert not facts.thd
    return {spec.name for spec in engines.ENGINE_SPECS if engines.mismatch(spec.capabilities, facts) is None}


@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("forms", [(False, False), (True, False), (False, True), (True, True)], ids=["ordinary", "cu-q", "cu-kv", "cu-both"])
def test_dense_length_forms_keep_existing_owners(architecture, dtype, forms):
    ordinary = _offered(_graph(dtype, (False, False)))
    assert ordinary, f"ordinary dense padded control has no owner on {architecture}"
    candidate = _offered(_graph(dtype, forms))
    if architecture == (8, 0) or not any(forms):
        assert ordinary <= candidate, f"cumulative lengths lost existing dense owners: {sorted(ordinary - candidate)}"
    else:
        assert not candidate, "unimplemented dense CU readers must still decline"


@pytest.mark.parametrize("forms", [(True, False), (False, True), (True, True)], ids=["cu-q", "cu-kv", "cu-both"])
def test_dense_length_forms_still_decline_ambiguous_side(architecture, forms):
    assert not _offered(_graph(cudnn.data_type.HALF, forms, ambiguous=True))


@pytest.mark.parametrize("forms", [(None, True), (True, None), (True, True)], ids=["cu-kv", "cu-q", "cu-both"])
def test_dense_cumulative_requires_padding_mask(architecture, forms):
    assert not _offered(_graph(cudnn.data_type.HALF, forms, padding=False))


@pytest.mark.skipif(torch.cuda.get_device_capability() != (8, 0), reason="requires native SM80")
class TestSm80DenseCumulativeLengths:
    @pytest.mark.parametrize("dims,splits,pack_gqa", [((64, 64), 1, False), ((256, 256), 1, False), ((576, 512), 1, False), ((128, 128), 3, True)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("forms", [(True, False), (False, True), (True, True)], ids=["cu-q", "cu-kv", "cu-both"])
    def test_native_lengths_rebind_and_replay(self, dims, splits, pack_gqa, dtype, forms, monkeypatch):
        from dense_cu_lengths import check_reference, execute, make_case, same_bytes, set_lengths, snapshot

        candidate = make_case(dims, dtype, forms, splits=splits, pack_gqa=pack_gqa)
        ordinary = make_case(dims, dtype, (False, False), splits=splits, pack_gqa=pack_gqa)
        qlens, klens = [17, 0, 7], [65, 11, 0]
        for case in (candidate, ordinary):
            set_lengths(case, qlens, klens)
            execute(case)
            check_reference(case, qlens, klens)
        same_bytes(snapshot(candidate), snapshot(ordinary))
        # Plan-time compilation must be complete before capture or rebinding.
        import cutlass.cute as cute

        def no_compile(*args, **kwargs):
            raise AssertionError("dense length values must not trigger compilation")

        monkeypatch.setattr(cute, "compile", no_compile)
        stream, capture, handle = torch.cuda.Stream(), torch.cuda.CUDAGraph(), cudnn.create_handle()
        cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
        try:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture, stream=stream):
                execute(candidate, handle)
            torch.cuda.current_stream().wait_stream(stream)
            for qlens, klens in (([3, 17, 0], [9, 0, 65]), ([17, 7, 1], [65, 13, 1])):
                for case in (candidate, ordinary):
                    set_lengths(case, qlens, klens)
                    case.bufs["o"].fill_(float("nan"))
                    case.bufs["stats"].fill_(float("nan"))
                capture.replay()
                execute(ordinary)
                check_reference(candidate, qlens, klens)
                same_bytes(snapshot(candidate), snapshot(ordinary))
                fixed = snapshot(candidate)
                capture.replay()
                same_bytes(snapshot(candidate), fixed)
            # Fresh metadata storage must be rebound by ordinary execution.
            for side in ("q", "kv"):
                name = side + "_len"
                candidate.bufs[name] = candidate.bufs[name].clone()
                candidate.pack[candidate.refs[name]] = candidate.bufs[name]
            set_lengths(candidate, [0, 17, 3], [65, 1, 7])
            execute(candidate)
            check_reference(candidate, [0, 17, 3], [65, 1, 7])
        finally:
            capture.reset()
            cudnn.destroy_handle(handle)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_standalone_prefix_binding_validates_b_plus_one(self, dtype):
        from dense_cu_lengths import check_reference, make_case, set_lengths
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

        case = make_case((64, 64), dtype, (True, True))
        b = case.bufs
        api = SdpaFwdDslSm80(*(b[n] for n in ("q", "k", "v", "o")), sample_lse=b["stats"].squeeze(-1), cu_seq_q_lens=True, cu_seq_kv_lens=True)
        assert api.check_support()
        api.compile()
        set_lengths(case, [17, 0, 3], [65, 1, 7])

        def launch(q=b["q_len"], kv=b["kv_len"]):
            api.execute(*(b[n] for n in ("q", "k", "v", "o")), lse_tensor=b["stats"].squeeze(-1), seq_q_lens=q, seq_kv_lens=kv)

        launch()
        check_reference(case, [17, 0, 3], [65, 1, 7])
        for side in ("q", "kv"):
            for invalid in (None, b[side + "_len"][:-1], torch.zeros(5, dtype=torch.int32, device="cuda"), b[side + "_len"].to(torch.int64)):
                b["o"].fill_(123)
                b["stats"].fill_(456)
                with pytest.raises(ValueError):
                    launch(**{side: invalid})
                assert torch.all(b["o"] == 123) and torch.all(b["stats"] == 456)
