# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Per-tensor FP8 page pools preserve independent THD Q/O/Stats origins."""

import math

import pytest

from frost_test_utils import requires_dsl

pytestmark = [requires_dsl]


@pytest.mark.parametrize(
    "dim,dtype_name,out_name,pool_layout,stats_layout,page",
    [
        pytest.param(128, "float8_e4m3fn", "bfloat16", "hnd", "token", 16, marks=pytest.mark.L0),
        pytest.param(256, "float8_e4m3fn", "float8_e4m3fn", "nhd", "head", 128, marks=pytest.mark.L0),
        pytest.param(128, "float8_e5m2", "float8_e4m3fn", "mixed", "head", 8, marks=pytest.mark.L1),
        pytest.param(256, "float8_e5m2", "bfloat16", "mixed", "token", 256, marks=pytest.mark.L1),
        pytest.param(128, "float8_e4m3fn", "float8_e4m3fn", "hnd", "padded", 16, marks=pytest.mark.L0),
        pytest.param(256, "float8_e5m2", "float16", "nhd", "padded", 8, marks=pytest.mark.L1),
        pytest.param(128, "float8_e5m2", "float8_e5m2", "nhd", "token", 256, marks=pytest.mark.L1),
        pytest.param(256, "float8_e4m3fn", "bfloat16", "hnd", "token", 16, marks=pytest.mark.L1),
    ],
)
def test_paged_thd_fp8_origins_rebind_and_capture(
    dim, dtype_name, out_name, pool_layout, stats_layout, page, heads=8, has_amax=False, mask="bottom_right", sink=False
):
    import cudnn
    import torch
    from cudnn.engines.manifest import family_for

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100/SM103 physical THD origin path")
    torch.manual_seed(261026737)
    batch, kv_heads, maximum, kv_max = 2, 2, 320, 512
    pages_per_sequence, page_count = kv_max // page, batch * kv_max // page
    dtype = getattr(torch, dtype_name)
    io_type = cudnn.data_type.FP8_E4M3 if dtype_name == "float8_e4m3fn" else cudnn.data_type.FP8_E5M2
    out_dtype = getattr(torch, out_name)
    out_type = {
        "bfloat16": cudnn.data_type.BFLOAT16,
        "float16": cudnn.data_type.HALF,
        "float8_e4m3fn": cudnn.data_type.FP8_E4M3,
        "float8_e5m2": cudnn.data_type.FP8_E5M2,
    }[out_name]
    scalar_names = ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
    widths = {"q": torch.int64, "o": torch.int32, "stats": torch.int64}
    if stats_layout == "padded":
        del widths["stats"]
    multipliers = {"q": heads * dim, "o": dim, "stats": 1}
    graph_ports = ("q", "k", "v", "o", "stats", "ql", "kl", "kt", "vt", "ro_q", "ro_o", "ro_stats") + scalar_names

    if stats_layout == "padded":
        graph_ports = tuple(name for name in graph_ports if name != "ro_stats")
    if has_amax:
        graph_ports += ("amax_o",)
    if sink:
        graph_ports += ("sinks",)

    def buffers(wide=False):
        q_capacity, o_capacity, stats_capacity = (640 if wide else 512), 640, 672
        pitch = dim + (16 if wide else 0)
        result = {
            "q": torch.empty(q_capacity, heads, pitch, device="cuda", dtype=dtype)[..., :dim],
            "backing_o": torch.empty(o_capacity, heads, pitch, device="cuda", dtype=out_dtype),
            "stats": torch.empty((stats_capacity, heads) if stats_layout == "token" else (heads, stats_capacity), device="cuda", dtype=torch.float32),
            "ql": torch.empty(batch, 1, 1, 1, device="cuda", dtype=torch.int32),
            "kl": torch.empty(batch, 1, 1, 1, device="cuda", dtype=torch.int32),
        }
        result["o"] = result["backing_o"][..., :dim]
        if stats_layout == "padded":
            result["stats"] = torch.empty(batch, heads, maximum, device="cuda", dtype=torch.float32)
        if has_amax:
            result["amax_o"] = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
        if pool_layout != "mixed":
            shape = (page_count + 1, kv_heads, page, 2 * dim) if pool_layout == "hnd" else (page_count + 1, page, kv_heads, 2 * dim)
            interleaved = torch.empty(shape, device="cuda", dtype=dtype)
            if pool_layout == "nhd":
                interleaved = interleaved.transpose(1, 2)
            result["k"], result["v"] = interleaved.split(dim, dim=-1)
        else:
            result["k"] = torch.empty(page_count + 1, kv_heads, page, dim, device="cuda", dtype=dtype)
            result["v"] = torch.empty(page_count + 1, page, kv_heads, dim, device="cuda", dtype=dtype).transpose(1, 2)
        for name in ("kt", "vt"):
            backing = torch.empty(batch, 1, pages_per_sequence * (2 if name == "vt" else 1), 1, device="cuda", dtype=torch.int32)
            result[name] = backing[:, :, ::2] if name == "vt" else backing
        for name, width in widths.items():
            result["ro_" + name] = torch.empty(batch + 1, 1, 1, 1, device="cuda", dtype=width)
        if sink:
            result["sinks"] = torch.empty(1, heads, 1, 1, device="cuda", dtype=torch.float32)
        for name in scalar_names:
            result[name] = torch.empty((1, 1, 1, 1), device="cuda", dtype=torch.float32)
        return result

    def fill(values, shift, replay=False, keyless=False, empty=False):
        ql = ([257, 49] if replay else [300, 177]) if not empty else [0, 0]
        kl = [kv_max - 31, 263] if replay else [kv_max, kv_max]
        if keyless:
            kl[1] = 0
        # shift=0 is the integration reproducer's shared [0,320,512] Q slots.
        # Subsequent bindings/replays also separate the three physical origins.
        starts = {
            "q": [shift, 320 + shift],
            "o": [shift + (16 if shift else 0), 320 + shift + (32 if shift else 0)],
            "stats": [shift + (32 if shift else 0), 320 + shift + (64 if shift else 0)],
        }
        for name, lengths in (("ql", ql), ("kl", kl)):
            values[name].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32).view(batch, 1, 1, 1))
        for name in widths:
            tensor = values[name]
            token_stride = tensor.stride(0) if name != "stats" or stats_layout == "token" else 1
            capacity = tensor.shape[0] if name != "stats" or stats_layout == "token" else tensor.shape[1]
            offsets = [(x * token_stride) // multipliers[name] for x in starts[name] + [capacity]]
            values["ro_" + name].copy_(torch.tensor(offsets, device="cuda", dtype=widths[name]).view(batch + 1, 1, 1, 1))
        values["q"].fill_(float("nan"))
        for start, length in zip(starts["q"], ql):
            values["q"][start : start + length].copy_(torch.randn(length, heads, dim, device="cuda").to(dtype))
        for name in ("k", "v"):
            values[name].copy_(torch.randn(values[name].shape, device="cuda").to(dtype))
            values[name][-1].fill_(float("nan"))
        for name in ("kt", "vt"):
            values[name].copy_(torch.randperm(page_count, device="cuda", dtype=torch.int32).view(batch, 1, pages_per_sequence, 1))
        for name, value in zip(scalar_names, (0.8, 0.9, 0.4 if replay else 0.7, 1.0, 1.0, 1.9 if replay else 1.3)):
            values[name].fill_(value)
        if sink:
            values["sinks"].fill_(-120.0 if replay else 0.3)
        return ql, kl, starts

    first = buffers()
    graph = cudnn.pygraph(
        io_data_type=io_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    nodes = {name: graph.tensor_like(first[name]) for name in graph_ports if name not in ("q", "o", "stats", "amax_o")}
    nodes["q"] = graph.tensor(dim=(batch, heads, maximum, dim), stride=(maximum * heads * dim, dim, heads * dim, 1), data_type=io_type)
    nodes["q"].set_ragged_offset(nodes["ro_q"]).set_ragged_offset_multiplier(multipliers["q"])
    nodes["o"], nodes["stats"], _, amax_o = graph.sdpa_fp8(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        generate_stats=True,
        **{name: nodes[name] for name in scalar_names},
        attn_scale=dim**-0.5,
        right_bound=0,
        diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT if mask == "top_left" else cudnn.diagonal_alignment.BOTTOM_RIGHT,
        **({"left_bound": 65} if mask == "window" else {}),
        **({"sink_token": nodes["sinks"]} if sink else {}),
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        paged_attention_k_table=nodes["kt"],
        paged_attention_v_table=nodes["vt"],
        paged_attention_max_seq_len_kv=kv_max,
        max_total_seq_len_q=477,
        stats_use_log2=stats_layout == "head",
    )
    nodes["o"].set_data_type(out_type).set_output(True).set_dim((batch, heads, maximum, dim)).set_stride(
        (maximum * heads * dim, dim, heads * dim, 1)
    ).set_ragged_offset(nodes["ro_o"]).set_ragged_offset_multiplier(multipliers["o"])
    stats_stride = (maximum * heads, 1, heads, 1) if stats_layout == "token" else (maximum, first["stats"].shape[1], 1, 1)
    if stats_layout == "padded":
        stats_stride = (heads * maximum, maximum, 1, 1)
    nodes["stats"].set_output(True).set_dim((batch, heads, maximum, 1)).set_stride(stats_stride).set_data_type(cudnn.data_type.FLOAT)
    if stats_layout != "padded":
        nodes["stats"].set_ragged_offset(nodes["ro_stats"])
    if has_amax:
        nodes["amax_o"] = amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(
        family_for(graph).offered_ids()["sdpa_fwd_prefill_sm100_fp8"],
        {cudnn.knob_type.TILE_CGA_M: 2 if dim == 128 else 1, cudnn.knob_type.SPLIT_KV: 1, cudnn.knob_type.PACK_GQA: 0},
    )
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_plan_name_at_index(graph._plan_index).startswith("sdpa_fwd_prefill_sm100_fp8")
    prepared = graph._compiled_plans[graph._plan_index]._prepared
    assert prepared.spec.origin_specs is not None
    assert prepared.spec.quant is not None
    print("ORIGIN_PLAN", graph.get_plan_name_at_index(graph._plan_index), flush=True)
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    controls = []
    if sink:
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        # Compare identical inputs with the existing dense-Q arithmetic and
        # the public standalone compact-prefix contract. These controls own
        # their storage and captures; no graph-lowered adapter is repurposed.
        for compact in (False, True):
            buffers_control = {name: torch.empty_like(first[name]) for name in ("k", "v", "kt", "vt", "ql", "kl", "sinks") + scalar_names}
            buffers_control["q"] = torch.empty(batch, maximum, heads, dim, device="cuda", dtype=dtype)
            buffers_control["o"] = torch.empty(batch, maximum, heads, dim, device="cuda", dtype=out_dtype)
            buffers_control["stats"] = torch.empty(batch, heads, maximum, device="cuda", dtype=torch.float32)
            api = SdpaFwdDslSm100(
                sample_q=buffers_control["q"].transpose(1, 2),
                sample_k=buffers_control["k"],
                sample_v=buffers_control["v"],
                sample_o=buffers_control["o"].transpose(1, 2),
                sample_lse=buffers_control["stats"],
                is_causal=True,
                causal_bottom_right=mask != "top_left",
                window_size_left=64 if mask == "window" else None,
                seq_kv_lens_present=True,
                seq_q_lens_present=not compact,
                has_sink=True,
                thd=compact,
                max_total_seq_len_q=477 if compact else None,
                pertensor_fp8=True,
                cga=2 if dim == 128 else 1,
                split_kv=1,
                pack_gqa=False,
                paged_page_size=page,
                paged_max_seq_len_kv=kv_max,
                thd_stats_padded=compact,
                has_amax_o=False,
            )
            assert api.check_support()
            api.compile()
            control_ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
            controls.append((compact, api, buffers_control, control_ws, torch.cuda.CUDAGraph()))

    def check_controls(values, state):
        ql, _, starts = state
        for compact, api, control, control_ws, capture in controls:
            for name in ("k", "v", "kt", "vt", "ql", "kl", "sinks") + scalar_names:
                control[name].copy_(values[name])
            control["q"].fill_(float("nan"))
            control["o"].fill_(-16)
            cursor = 0
            for b, length in enumerate(ql):
                target = control["q"].view(-1, heads, dim)[cursor : cursor + length] if compact else control["q"][b, :length]
                target.copy_(values["q"][starts["q"][b] : starts["q"][b] + length])
                cursor += length

            def launch():
                api.execute(
                    control["q"].view(-1, heads, dim) if compact else control["q"].transpose(1, 2),
                    control["k"],
                    control["v"],
                    control["o"].view(-1, heads, dim) if compact else control["o"].transpose(1, 2),
                    lse_tensor=control["stats"],
                    sinks=control["sinks"],
                    seq_q_lens=control["ql"].view(-1),
                    seq_kv_lens=control["kl"].view(-1),
                    block_table=control["kt"][:, 0, :, 0],
                    block_table_v=control["vt"][:, 0, :, 0],
                    workspace=control_ws,
                    descale_q=control["descale_q"],
                    descale_k=control["descale_k"],
                    descale_v=control["descale_v"],
                    scale_o=control["scale_o"],
                )

            if not control.get("captured"):
                launch()
                with torch.cuda.graph(capture):
                    launch()
                control["captured"] = True
            capture.replay()
            torch.cuda.synchronize()
            cursor = 0
            for b, length in enumerate(ql):
                expected = control["o"].view(-1, heads, dim)[cursor : cursor + length] if compact else control["o"][b, :length]
                actual = values["o"][starts["o"][b] : starts["o"][b] + length]
                torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
                torch.testing.assert_close(values["stats"][b, :, :length], control["stats"][b, :, :length], atol=0, rtol=0)
                cursor += length
            print("IDENTICAL_INPUT_CONTROL", "compact" if compact else "dense", dim, mask, ql, flush=True)

    def poison(values):
        values["backing_o"].fill_(-16)
        values["stats"].fill_(-29)
        workspace.fill_(255)
        if has_amax:
            values["amax_o"].fill_(999)

    def check(values, state):
        ql, kl, starts = state
        check_controls(values, state)
        if stats_layout == "padded":
            stats = values["stats"].transpose(1, 2).reshape(batch * maximum, heads)
        else:
            stats = values["stats"] if stats_layout == "token" else values["stats"].T
        from sdpa.fp8_ref import compute_ref

        expected_amax = 0.0
        live_o = torch.zeros(values["o"].shape[0], device="cuda", dtype=torch.bool)
        live_stats = torch.zeros(stats.shape[0], device="cuda", dtype=torch.bool)
        for b, (nq, nk) in enumerate(zip(ql, kl)):
            if not nq:
                continue
            k8 = values["k"][values["kt"][b].flatten().long()].transpose(0, 1).reshape(kv_heads, kv_max, dim)
            v8 = values["v"][values["vt"][b].flatten().long()].transpose(0, 1).reshape(kv_heads, kv_max, dim)
            # Match the established FP8 reference's P cast and online rescaling.
            # A real-valued softmax omits this rounding, especially at short
            # top-left rows. D128 biases P by 2**4; D256 casts it directly.
            p_scale = 2.0 ** (4 + (4 if dim == 128 else 0))
            quantized_o, _, _ = compute_ref(
                values["q"][starts["q"][b] : starts["q"][b] + nq].unsqueeze(0),
                k8.transpose(0, 1).unsqueeze(0),
                v8.transpose(0, 1).unsqueeze(0),
                attn_scale=dim**-0.5,
                q_descale=values["descale_q"],
                k_descale=values["descale_k"],
                v_descale=values["descale_v"],
                s_scale=p_scale,
                s_descale=1.0 / p_scale,
                torch_itype=dtype,
                torch_otype=out_dtype,
                padding=(values["ql"][b].reshape(1), values["kl"][b].reshape(1)),
                right_bound=0,
                left_bound=65 if mask == "window" else None,
                diag_align=cudnn.diagonal_alignment.TOP_LEFT if mask == "top_left" else cudnn.diagonal_alignment.BOTTOM_RIGHT,
                sink_token=values["sinks"] if sink else None,
                rescale_threshold=4.0,
                dtype=torch.float64,
                quantize_o=False,
                sink_in_max=False,
            )
            quantized_o = quantized_o[0].double() * values["scale_o"].item()
            k = k8[:, :nk].double().repeat_interleave(heads // kv_heads, 0)
            for lo in range(0, nq, 32):
                hi = min(lo + 32, nq)
                q = values["q"][starts["q"][b] + lo : starts["q"][b] + hi].double().transpose(0, 1)
                scores = q @ k.transpose(-1, -2) * dim**-0.5 * values["descale_q"].item() * values["descale_k"].item()
                diagonal = torch.arange(lo, hi, device="cuda")[:, None] + (0 if mask == "top_left" else nk - nq)
                keys = torch.arange(nk, device="cuda")[None, :]
                scores.masked_fill_(keys > diagonal, -torch.inf)
                if mask == "window":
                    scores.masked_fill_(keys < diagonal - 64, -torch.inf)
                if sink:
                    scores = torch.cat((scores, values["sinks"].double().view(heads, 1, 1).expand(-1, hi - lo, 1)), dim=-1)
                expected = quantized_o[lo:hi]
                expected_stats = scores.logsumexp(-1).T * (math.log2(math.e) if stats_layout == "head" else 1)
                os, ss = starts["o"][b], (b * maximum if stats_layout == "padded" else starts["stats"][b])
                expected_amax = max(expected_amax, expected.abs().max().item() / values["scale_o"].item())
                # Existing paged FP8 bound is in unscaled O units. Runtime
                # scale_o scales this error too; retain zero relative slack.
                # Dense/compact controls above also pin O/Stats bit for bit.
                atol = 0.04 * abs(values["scale_o"].item())
                if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    atol = max(atol, 3 * (expected - expected.to(out_dtype).double()).abs().max().item())
                torch.testing.assert_close(values["o"][os + lo : os + hi].double(), expected, atol=atol, rtol=0)
                torch.testing.assert_close(
                    stats[ss + lo : ss + hi].double(),
                    expected_stats,
                    atol=(0.01 if dim == 256 else 0.005) * (math.log2(math.e) if stats_layout == "head" else 1),
                    # D256 uses the existing prepared FP8 Stats budget.
                    # Its dense/compact controls still require exact equality.
                    rtol=0.01 if dim == 256 else 0,
                )
            live_o[starts["o"][b] : starts["o"][b] + nq] = True
            stats_start = b * maximum if stats_layout == "padded" else starts["stats"][b]
            live_stats[stats_start : stats_start + nq] = True
        assert torch.all(values["o"].float()[~live_o] == -16)
        if stats_layout == "padded":
            assert torch.isneginf(stats[~live_stats]).all()
        else:
            assert torch.all(stats[~live_stats] == -29)
        if has_amax:
            assert abs(values["amax_o"].item() - expected_amax) <= 0.03 + 0.03 * expected_amax
        assert torch.all(values["backing_o"].float()[..., dim:] == -16)
        assert torch.isnan(values["k"][-1]).all() and torch.isnan(values["v"][-1]).all()

    captures = []
    try:
        for values, shift in ((first, 0), (buffers(wide=True), 32)):
            state = fill(values, shift)
            pack = {nodes[name]: values[name] for name in graph_ports}
            overrides = dict(
                override_uids=[nodes[name].get_uid() for name in ("q", "o")],
                override_shapes=[nodes[name].get_dim() for name in ("q", "o")],
                override_strides=[(maximum * values[name].stride(0), values[name].stride(1), values[name].stride(0), 1) for name in ("q", "o")],
            )
            poison(values)
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(pack, workspace, **overrides)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            check(values, state)
            captured = torch.cuda.CUDAGraph()
            captures.append((captured, values, shift))
            with torch.cuda.graph(captured):
                graph.execute(pack, workspace, **overrides)
        for captured, values, shift in reversed(captures):
            state = fill(values, shift + 16, replay=True, keyless=shift == 0)
            poison(values)
            captured.replay()
            torch.cuda.synchronize()
            check(values, state)
        for captured, values, shift in captures:
            state = fill(values, shift, empty=True)
            poison(values)
            captured.replay()
            torch.cuda.synchronize()
            check(values, state)
    finally:
        for captured, _, _ in captures:
            captured.reset()
        for _, _, _, _, capture in controls:
            capture.reset()


@pytest.mark.L1
@pytest.mark.parametrize("dim", [128, 256])
def test_paged_thd_fp8_persistent_multiwave(dim):
    test_paged_thd_fp8_origins_rebind_and_capture(
        dim,
        "float8_e4m3fn",
        "bfloat16",
        "hnd",
        "token",
        16,
        heads=48 if dim == 128 else 96,
        has_amax=True,
    )


@pytest.mark.L1
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("mask", ["top_left", "window"])
def test_paged_thd_fp8_masked_sink_replay(dim, mask):
    test_paged_thd_fp8_origins_rebind_and_capture(
        dim,
        "float8_e4m3fn",
        "bfloat16",
        "hnd",
        "padded",
        16,
        mask=mask,
        sink=True,
    )
