# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned compressed-KV RoPE-to-group16-E4M3-scale-FP4-QDQ stage.

This times the complete GPU stage, including GPU unpack/copy where required.
It does not time projection/norm, ring-cache writes, attention, full model, or bprop.
Synthetic activations and exact source frequency/configuration are explicit.
Every sample starts from restored input; no repeated quantization drift.
"""

# Source, numerical and admission checks below rely on assertions.
if not __debug__:
    raise RuntimeError("This benchmark/auditor requires Python assertions; run without -O/-OO or PYTHONOPTIMIZE.")

import argparse
import ast
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SHARED = HERE
ROLES = (
    "source_wrapper",
    "source_quant_alias",
    "flashinfer_quant_wrapper",
    "flashinfer_quant_alias",
    "flashinfer_noftz_quant_alias",
    "torch_eager",
    "torch_fused",
    "torch_fused_tuned",
    "flashinfer_torch_quant",
    "flashinfer_nvfp4_cuda",
    "flashinfer_nvfp4_cute",
    "flashinfer_nvfp4_cuda_native_dequant",
    "flashinfer_nvfp4_cute_native_dequant",
    "torch_fma_eager",
    "torch_fma_fused",
    "torch_fma_fused_tuned",
    "torch_fma_unsnapped_fused",
    "torch_fma_unsnapped_tuned",
    "frost",
)
GENERATIONS = ("random", "negative_roll", "signed_zero", "small", "subnormal", "large", "midpoints")
REQUIRED = (
    "source_wrapper",
    "source_quant_alias",
    "flashinfer_noftz_quant_alias",
    "torch_fma_eager",
    "torch_fma_fused",
    "torch_fma_fused_tuned",
    "frost",
)
TIMED_GENERATIONS = ("random", "negative_roll")
TIMED_REQUIRED = REQUIRED + ("flashinfer_quant_alias",)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


from compkv_contract import model_cases


def trace_contract(events, role, calls):
    work = [event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    assert work and all(event.get("args", {}).get("graph id", 0) > 0 for event in work)
    graph_ids = {event["args"]["graph id"] for event in work}
    correlations = {event["args"].get("correlation") for event in work}
    streams = {event["args"].get("stream") for event in work}
    assert len(graph_ids) == len(correlations) == len(streams) == 1
    assert None not in correlations and None not in streams
    names = [event["name"] for event in work]
    assert not any(re.search(r"DtoH|HtoD|device.*host|host.*device", name, re.IGNORECASE) for name in names)
    if role.startswith("frost"):
        assert len(work) == calls and all("cudnn_frost_compkv_rope_qdq" in name for name in names)
    if role.startswith("flashinfer"):
        assert sum("BatchQKApplyRotaryPosIdsCosSinCache" in name for name in names) == calls
    if role in ("source_wrapper", "source_quant_alias", "flashinfer_quant_wrapper", "flashinfer_quant_alias", "flashinfer_noftz_quant_alias"):
        assert sum("fp4_quant_kernel" in name for name in names) == calls
    assert len(work) >= calls
    return dict(
        graph_id=next(iter(graph_ids)),
        correlation=next(iter(correlations)),
        stream=next(iter(streams)),
        work=[dict(category=e["cat"], name=e["name"]) for e in work],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deepseek-source", type=Path, required=True, help="Pinned inference directory containing model.py, kernel.py and config.json")
    args = parser.parse_args()
    SOURCE = args.deepseek_source.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    from benchmark_environment import configure_caches

    configure_caches(args.output)
    import numpy as np
    import torch
    import triton
    import flashinfer
    import cudnn
    from cudnn.rope import qdq as rope_api
    from flashinfer.quantization.fp4_quantization import get_fp4_kv_dequantization_module
    from references_compkv import fused_inplace, quant_inplace, omitted_rounding_control, unpack_nvfp4_inplace
    from references_compkv import fused_snap_inplace, fused_fma_snap_inplace, quant_snap_inplace
    from native_baselines import prepare as prepare_noftz

    result = dict(
        status="running",
        created_at=datetime.now(timezone.utc).isoformat(),
        scope=__doc__,
        performance_admitted=False,
        timing=None,
        cases={},
        checks={},
        negative_controls={},
        roles=list(ROLES),
        required_roles=list(REQUIRED),
        timed_generations=list(TIMED_GENERATIONS),
        timed_required_roles=list(TIMED_REQUIRED),
        generations=list(GENERATIONS),
        correctness_contract="Bit-exact finite BF16 source output, including zero sign; no tolerance-based rejection concealment",
        input_contract="Synthetic finite BF16 inputs; source RoPE outputs must remain finite; subnormals are included",
        timing_contract="Random/negative-roll only; every timed provider must be bit-exact on both; candidates and source/no-FTZ controls must also pass all seven stress generations. Three disjoint inputs, restored before every replay; no projection or scores",
    )

    def save():
        temporary = args.output.with_suffix(".writing.json")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    def equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def compare(actual, expected):
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        different = actual.view(torch.int16) != expected.view(torch.int16)
        count = int(different.sum())
        if count == 0:
            relative, maximum = 0.0, 0.0
        elif finite:
            a, b = actual.double(), expected.double()
            relative = float((a - b).norm() / b.norm().clamp_min(1e-300))
            maximum = float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))
            # Finite inputs can still overflow diagnostics against a zero reference.
            relative = relative if math.isfinite(relative) else None
            maximum = maximum if math.isfinite(maximum) else None
        else:
            relative, maximum = None, None
        return dict(passed=finite and count == 0, finite=finite, differing_bf16_bits=count, elements=actual.numel(), relative_l2=relative, max_scaled=maximum)

    save()
    try:
        assert torch.cuda.get_device_capability() == (10, 0)
        from compkv_contract import verify_source

        result["consumer_contract"] = verify_source(SOURCE)
        noftz_rotate, result["noftz_control"] = prepare_noftz()
        result["environment"] = dict(
            device=torch.cuda.get_device_name(),
            worker_pid=os.getpid(),
            uuid=str(torch.cuda.get_device_properties(0).uuid),
            torch=torch.__version__,
            triton=triton.__version__,
            flashinfer=flashinfer.__file__,
            script_sha256=sha(__file__),
            cudnn=cudnn.__file__,
            cudnn_version=cudnn.__version__,
            kernel_sha256=sha(Path(rope_api.__file__).parent / "frost/compressed_kv.py"),
            shared_kernel_sha256=sha(Path(rope_api.__file__).parent / "frost/kernels.py"),
            candidate_api="cudnn.RopeQDQInplace",
            api_path=rope_api.__file__,
            api_sha256=sha(rope_api.__file__),
            references_sha256=sha(HERE / "references_compkv.py"),
        )
        spec = importlib.util.spec_from_file_location("deepseek_rope_qdq_reference", SOURCE / "kernel.py")
        source = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = source
        spec.loader.exec_module(source)
        tree = ast.parse((SOURCE / "model.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("apply_rotary_emb", "precompute_freqs_cis")]
        assert len(functions) == 2
        namespace = dict(torch=torch, math=math, lru_cache=lru_cache)
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(SOURCE / "model.py"), "exec"), namespace)
        rotate_source, frequencies = namespace["apply_rotary_emb"], namespace["precompute_freqs_cis"]
        config = json.loads((SOURCE / "config.json").read_text())
        cases = model_cases(config)
        result["model_cases"] = cases
        result["source"] = dict(
            revision="dba1be0a40aa45a94ad051997016db3960a90277",
            files={name: sha(SOURCE / name) for name in ("model.py", "kernel.py", "config.json")},
            function_ast_sha256=hashlib.sha256("\n".join(ast.dump(node) for node in functions).encode()).hexdigest(),
        )
        freq_tables, caches = {}, {}
        with torch.device("cuda"):
            for rope in ("base", "compressed"):
                table = frequencies(
                    64,
                    16384,
                    config["original_seq_len"] if rope == "compressed" else 0,
                    config["compress_rope_theta"] if rope == "compressed" else config["rope_theta"],
                    config["rope_factor"],
                    config["beta_fast"],
                    config["beta_slow"],
                )
                freq_tables[rope] = table
                caches[rope] = torch.cat((table.real, table.imag), dim=-1).contiguous()
        readonly_tables = {name: value.clone() for name, value in freq_tables.items()}
        readonly_caches = {name: value.clone() for name, value in caches.items()}
        # Twelve source-shaped cases, three input buffers, two compile options.
        previous_limit = torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit = 128
        result["dynamo_specialization_budget"] = dict(previous=previous_limit, current=128, shapes=len(cases), buffers=3, option_sets=2)
        from references_compkv import fused_fma_inplace

        options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
        tuned = dict(options, max_autotune=True, coordinate_descent_tuning=True)
        compiled = dict(
            fused=torch.compile(fused_snap_inplace, fullgraph=True, dynamic=False, options=options),
            fma=torch.compile(fused_fma_snap_inplace, fullgraph=True, dynamic=False, options=options),
            fma_tuned=torch.compile(fused_fma_snap_inplace, fullgraph=True, dynamic=False, options=tuned),
            tuned=torch.compile(fused_snap_inplace, fullgraph=True, dynamic=False, options=tuned),
            quant=torch.compile(quant_snap_inplace, fullgraph=True, dynamic=False, options=tuned),
            unsnapped=torch.compile(fused_fma_inplace, fullgraph=True, dynamic=False, options=options),
            unsnapped_tuned=torch.compile(fused_fma_inplace, fullgraph=True, dynamic=False, options=tuned),
            unpack=torch.compile(unpack_nvfp4_inplace, fullgraph=True, dynamic=False, options=tuned),
        )
        result["torch_options"] = dict(default=options, tuned=tuned)
        result["torch_normalization"] = (
            "Compiled QDQ uses an FP16 normalization cast to preserve FP4 ties; original unsnapped compiled FMA controls are retained. Eager math and all FlashInfer controls are unchanged."
        )

        # A typical B1/S4096/ratio1 activation matrix isolates the required BF16 boundary.
        fx = torch.randn((4096, 1, 512), device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(497))
        fcache = caches["compressed"]
        fpos = torch.arange(4096, device="cuda", dtype=torch.int32)
        expected_fixture = fx.clone()
        rotate_source(expected_fixture[None, ..., -64:], freq_tables["compressed"][:4096])
        source.fp4_act_quant(expected_fixture, 16, True, scale_dtype=torch.float8_e4m3fn)
        wrong_fixture = omitted_rounding_control(fx, fcache, fpos)
        control = compare(wrong_fixture, expected_fixture)
        fixture_path = args.output.with_suffix(".bf16_boundary_control.pt")
        torch.save(dict(input=fx.cpu(), cache=fcache.cpu(), positions=fpos.cpu(), expected=expected_fixture.cpu(), wrong=wrong_fixture.cpu()), fixture_path)
        result["negative_controls"]["omitted_bf16_rounding"] = dict(
            rejected=not control["passed"], metrics=control, fixture_path=str(fixture_path), fixture_sha256=sha(fixture_path), actual_model_shape=[1, 4096, 512]
        )
        assert not control["passed"]
        global_sf = torch.ones((1,), device="cuda", dtype=torch.float32)
        readonly_global_sf = global_sf.clone()
        result["native_nvfp4_global_scale"] = 1.0

        eviction = torch.empty(4 * torch.cuda.get_device_properties(0).L2_cache_size, device="cuda", dtype=torch.uint8)
        all_timings = {}
        for case_spec in cases:
            tag, n, heads = case_spec["tag"], case_spec["tokens"], case_spec["heads"]
            batch, sequence = case_spec["batch"], case_spec["sequence"]
            freqs, cache = freq_tables[case_spec["rope"]], caches[case_spec["rope"]]
            readonly_freqs, readonly_cache = readonly_tables[case_spec["rope"]], readonly_caches[case_spec["rope"]]
            result["active_stage"] = dict(case=tag, phase="construct")
            print("Begin", tag, flush=True)
            case = result["cases"][tag] = dict(spec=case_spec, failures={}, routes={}, eligible=[], timing=None)
            generator = torch.Generator(device="cuda").manual_seed(419366 + n + heads)
            original = torch.randn((n, heads, 512), device="cuda", dtype=torch.bfloat16, generator=generator)
            positions = (torch.arange(sequence, device="cuda", dtype=torch.int32) * case_spec["position_stride"] + case_spec["start_pos"]).repeat(batch)
            readonly_positions = positions.clone()
            selected_freqs = freqs[positions[:sequence].long()]

            def source_rotate_kv(value):
                # Use actual [B,S,D] broadcasting rather than making a longer
                # synthetic sequence by flattening the batch for this baseline.
                rotate_source(value.view(batch, sequence, 512)[..., -64:], selected_freqs)

            elements = original.numel()
            storage = [torch.full((elements + 128,), 123.0, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            buffers = [value[64:-64].view_as(original) for value in storage]
            scale_buffers = [torch.empty((n * heads, 32), device="cuda", dtype=torch.float8_e4m3fn) for _ in buffers]
            empty_key = torch.empty((n, 0, 64), device="cuda", dtype=torch.bfloat16)
            native_quant = source.fp4_quant_kernel(512, 16, scale_dtype=source.FP8, inplace=True)
            pointers = [value.data_ptr() for value in buffers]
            intervals = sorted((ptr, ptr + elements * 2) for ptr in pointers)
            assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
            case["buffers"] = dict(pointers=pointers, bytes_each=elements * 2, independent=True, calls_per_graph=3)

            def generation(name):
                if name == "random":
                    return original
                if name == "negative_roll":
                    return -original.roll(7, dims=0)
                if name == "signed_zero":
                    value = torch.zeros_like(original)
                    value[..., 1::2] = -0.0
                    return value
                if name == "midpoints":
                    levels = torch.tensor(
                        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6],
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    values = levels.repeat(32).view(1, 1, 512).expand_as(original).clone()
                    values[..., 32:64] = -values[..., 32:64]
                    return values
                factor = {"small": 1e-4, "subnormal": 1e-38, "large": 1e4}[name]
                return (original.float() * factor).to(torch.bfloat16)

            def reset(value):
                for output, scales in zip(buffers, scale_buffers, strict=True):
                    output.copy_(value)
                    scales.view(torch.uint8).fill_(255)

            def fi_rotate(x):
                flashinfer.apply_rope_with_cos_sin_cache_inplace(positions, x[..., -64:], empty_key, 64, cache, is_neox=False)

            def invoke(role, count=3):
                result["active_role"] = role
                for index, x in enumerate(buffers[:count]):
                    if role in ("source_wrapper", "source_quant_alias"):
                        source_rotate_kv(x)
                    elif role in ("flashinfer_noftz_quant_alias",):
                        noftz_rotate(positions, x[..., -64:], empty_key, cache)
                    elif role.startswith("flashinfer"):
                        fi_rotate(x)
                    if role in ("source_wrapper", "flashinfer_quant_wrapper"):
                        source.fp4_act_quant(x, 16, True, scale_dtype=torch.float8_e4m3fn)
                    elif role in ("source_quant_alias", "flashinfer_quant_alias", "flashinfer_noftz_quant_alias"):
                        native_quant(x.view(-1, 512), x.view(-1, 512), scale_buffers[index])
                    elif role == "torch_fma_eager":
                        fused_fma_inplace(x, cache, positions)
                    elif role == "torch_fma_fused":
                        compiled["fma"](x, cache, positions)
                    elif role == "torch_fma_fused_tuned":
                        compiled["fma_tuned"](x, cache, positions)
                    elif role == "torch_fma_unsnapped_fused":
                        compiled["unsnapped"](x, cache, positions)
                    elif role == "torch_fma_unsnapped_tuned":
                        compiled["unsnapped_tuned"](x, cache, positions)
                    elif role == "torch_eager":
                        fused_inplace(x, cache, positions)
                    elif role == "torch_fused":
                        compiled["fused"](x, cache, positions)
                    elif role == "torch_fused_tuned":
                        compiled["tuned"](x, cache, positions)
                    elif role == "flashinfer_torch_quant":
                        compiled["quant"](x)
                    elif role.startswith("flashinfer_nvfp4"):
                        # Existing provider selector names are retained only for baselines.
                        backend = "cuda" if role.startswith("flashinfer_nvfp4_cuda") else "cute-dsl"
                        packed, scales = flashinfer.fp4_quantize(
                            x.view(-1, 512),
                            global_scale=global_sf,
                            sf_vec_size=16,
                            sf_use_ue8m0=False,
                            is_sf_swizzled_layout=False,
                            backend=backend,
                            enable_pdl=False,
                        )
                        if role.endswith("native_dequant"):
                            native_dequant.nvfp4_kv_dequant(packed, scales, global_sf, x.view(-1, 512))
                        else:
                            compiled["unpack"](x, packed, scales)
                    elif role.startswith("frost"):
                        plans[role].execute(x, cache, positions)
                    else:
                        raise AssertionError(role)

            plan = cudnn.RopeQDQInplace(buffers[0], cache, positions, quantization="fp4", backend="frost", group_size=16, scale_format="e4m3")
            plan.compile()
            plans = dict(frost=plan)
            native_dequant = get_fp4_kv_dequantization_module()

            stream = torch.cuda.Stream()
            graphs, timers = {}, {}
            for role in ROLES:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        reset(original)
                        invoke(role)
                stream.synchronize()
                reset(original)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                begin, end = torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                    begin.record()
                    invoke(role)
                    end.record()
                graphs[role], timers[role] = graph, (begin, end)

            passed = {role: True for role in ROLES}
            result["active_stage"] = dict(case=tag, phase="numerical_checks")
            for name in GENERATIONS:
                values = generation(name)
                expected = values.clone()
                source_rotate_kv(expected)
                assert bool(torch.isfinite(expected).all())
                source.fp4_act_quant(expected, 16, True, scale_dtype=torch.float8_e4m3fn)
                assert bool(torch.isfinite(expected).all())
                if name in ("random", "negative_roll"):
                    for negative, invalid in (("no_op", values), ("zeros", torch.zeros_like(values)), ("nan", torch.full_like(values, float("nan")))):
                        metrics = compare(invalid, expected)
                        result["negative_controls"][f"{tag}/{name}/{negative}"] = dict(rejected=not metrics["passed"], metrics=metrics)
                        assert not metrics["passed"]
                for role in ROLES:
                    for mode in ("eager", "graph"):
                        reset(values)
                        invoke(role) if mode == "eager" else graphs[role].replay()
                        for index, actual in enumerate(buffers):
                            key = f"{tag}/{name}/{role}/{mode}/{index}"
                            metrics = compare(actual, expected)
                            metrics["guards_unchanged"] = bool((storage[index][:64] == 123).all() and (storage[index][-64:] == 123).all())
                            result["checks"][key] = metrics
                            if not metrics["guards_unchanged"]:
                                raise RuntimeError(f"output guards changed: {key}")
                            passed[role] &= metrics["passed"]
                            if not metrics["passed"] and role not in case["failures"]:
                                path = args.output.with_name(args.output.stem + "." + tag + "." + role + ".failure.pt")
                                torch.save(
                                    dict(
                                        input=values.cpu(),
                                        actual=actual.cpu(),
                                        expected=expected.cpu(),
                                        positions=positions.cpu(),
                                        frequencies=selected_freqs.cpu(),
                                        generation=name,
                                        role=role,
                                        mode=mode,
                                        metrics=metrics,
                                    ),
                                    path,
                                )
                                case["failures"][role] = dict(path=str(path), sha256=sha(path), first_check=key)
                        assert (
                            equal(cache, readonly_cache)
                            and equal(freqs, readonly_freqs)
                            and equal(positions, readonly_positions)
                            and equal(global_sf, readonly_global_sf)
                        )
                save()
            result["active_stage"] = dict(case=tag, phase="unaligned_fallback")
            case["alignment_checks"], case["alignment_routes"] = {}, {}
            for alignment in (2, 4, 16):
                padding = 64 + alignment // 2
                unaligned_storage = torch.full((elements + 2 * padding,), 123.0, device="cuda", dtype=torch.bfloat16)
                unaligned = unaligned_storage[padding:-padding].view_as(original)
                assert unaligned.data_ptr() % 16 == (0 if alignment == 16 else alignment) and unaligned.is_contiguous()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    unaligned.copy_(original)
                    plan.execute(unaligned, cache, positions)
                stream.synchronize()
                fallback_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(fallback_graph, stream=stream):
                    plan.execute(unaligned, cache, positions)
                for alignment_generation in ("random", "signed_zero", "subnormal"):
                    alignment_input = generation(alignment_generation)
                    alignment_expected = alignment_input.clone()
                    source_rotate_kv(alignment_expected)
                    source.fp4_act_quant(alignment_expected, 16, True, scale_dtype=torch.float8_e4m3fn)
                    for alignment_mode in ("eager", "graph"):
                        unaligned.copy_(alignment_input)
                        if alignment_mode == "eager":
                            plan.execute(unaligned, cache, positions)
                        else:
                            fallback_graph.replay()
                        metric = compare(unaligned, alignment_expected)
                        metric["guards_unchanged"] = bool((unaligned_storage[:padding] == 123).all() and (unaligned_storage[-padding:] == 123).all())
                        key = f"align{alignment}/{alignment_generation}/{alignment_mode}"
                        case["alignment_checks"][key] = metric
                        if not metric["passed"] or not metric["guards_unchanged"]:
                            failure_path = args.output.with_name(args.output.stem + "." + tag + ".alignment.failure.pt")
                            torch.save(
                                dict(
                                    input=alignment_input.cpu(),
                                    actual=unaligned.cpu(),
                                    expected=alignment_expected.cpu(),
                                    positions=positions.cpu(),
                                    frequencies=selected_freqs.cpu(),
                                    key=key,
                                ),
                                failure_path,
                            )
                            case["alignment_failure"] = dict(path=str(failure_path), sha256=sha(failure_path))
                            raise AssertionError((tag, key, metric))
                trace_path = args.output.with_name(args.output.stem + "." + tag + f".unaligned_align{alignment}.trace.json")
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    fallback_graph.replay()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(trace_path))
                events = json.loads(trace_path.read_text())["traceEvents"]
                work = [event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
                assert len(work) == 1 and "cudnn_frost_compkv_rope_qdq" in work[0]["name"]
                assert work[0]["args"].get("graph id", 0) > 0
                case["alignment_routes"][f"align{alignment}"] = dict(path=str(trace_path), sha256=sha(trace_path))
                del fallback_graph
            assert (
                equal(cache, readonly_cache) and equal(freqs, readonly_freqs) and equal(positions, readonly_positions) and equal(global_sf, readonly_global_sf)
            )
            del unaligned, unaligned_storage, alignment_input, alignment_expected
            case["eligible"] = [role for role in ROLES if passed[role]]
            case["rejected_roles"] = [role for role in ROLES if not passed[role]]
            # Stress failures remain in full. Baselines that are correct on the
            # exact timed distributions still compete for the fastest baseline.
            case["timing_eligible"] = [
                role
                for role in ROLES
                if all(
                    result["checks"][f"{tag}/{generation}/{role}/{mode}/{index}"]["passed"]
                    for generation in TIMED_GENERATIONS
                    for mode in ("eager", "graph")
                    for index in range(3)
                )
            ]
            case["timed_with_stress_failures"] = [role for role in case["timing_eligible"] if role not in case["eligible"]]
            save()
            assert all(passed[role] for role in REQUIRED), case["rejected_roles"]
            assert all(role in case["timing_eligible"] for role in TIMED_REQUIRED), case["timing_eligible"]

            for role in case["timing_eligible"]:
                reset(original)
                path = args.output.with_name(args.output.stem + "." + tag + "." + role + ".trace.json")
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    graphs[role].replay()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(path))
                case["routes"][role] = dict(path=str(path), sha256=sha(path), **trace_contract(json.loads(path.read_text())["traceEvents"], role, 3))
            save()

            result["active_stage"] = dict(case=tag, phase="timing")
            timing = {}
            for regime in ("hot", "evicted"):
                blocks = []
                for block in range(8):
                    values = generation(("random", "negative_roll")[(block // 2) % 2])
                    order = case["timing_eligible"] if block % 2 == 0 else list(reversed(case["timing_eligible"]))
                    entry = dict(index=block, input_generation=("random", "negative_roll")[(block // 2) % 2], order=order, arms={})
                    for role in order:
                        samples = dict(gpu_us=[], wall_us=[])
                        begin, end = timers[role]
                        for sample in range(7):
                            reset(values)
                            if regime == "evicted":
                                eviction.fill_(block * 8 + sample + 1)
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            graphs[role].replay()
                            end.synchronize()
                            wall = (time.perf_counter() - started) * 1e6 / 3
                            gpu = begin.elapsed_time(end) * 1000 / 3
                            if sample >= 2:
                                assert gpu > 0 and wall > 0
                                samples["gpu_us"].append(gpu)
                                samples["wall_us"].append(wall)
                        entry["arms"][role] = samples
                    blocks.append(entry)
                summary = {}
                for metric in ("gpu_us", "wall_us"):
                    by_block = {role: [statistics.median(block["arms"][role][metric]) for block in blocks] for role in case["timing_eligible"]}
                    medians = {role: statistics.median(values) for role, values in by_block.items()}
                    best = min((role for role in case["timing_eligible"] if not role.startswith("frost")), key=medians.get)
                    candidates = {}
                    for role in ("frost",):
                        ratios = [a / b for a, b in zip(by_block[best], by_block[role], strict=True)]
                        ratio = statistics.median(ratios)
                        candidates[role] = dict(
                            paired_ratios=ratios, paired_speedup=ratio, latency_reduction_pct=100 * (1 - 1 / ratio), wins=sum(r > 1 for r in ratios)
                        )
                    summary[metric] = dict(medians=medians, fastest_valid_native=best, candidates=candidates)
                timing[regime] = dict(blocks=blocks, summary=summary)
            # Post-timing outputs are recomputed from restored inputs and checked.
            for name in ("random", "negative_roll"):
                values = generation(name)
                expected = values.clone()
                source_rotate_kv(expected)
                source.fp4_act_quant(expected, 16, True, scale_dtype=torch.float8_e4m3fn)
                for role in case["timing_eligible"]:
                    reset(values)
                    graphs[role].replay()
                    for index, actual in enumerate(buffers):
                        metrics = compare(actual, expected)
                        result["checks"][f"{tag}/after/{name}/{role}/{index}"] = metrics
                        assert metrics["passed"]
                    assert (
                        equal(cache, readonly_cache)
                        and equal(freqs, readonly_freqs)
                        and equal(positions, readonly_positions)
                        and equal(global_sf, readonly_global_sf)
                    )
            case["timing"] = timing
            all_timings[tag] = True
            save()
            print("Validated and timed", tag, "eligible", case["timing_eligible"], "rejected", case["rejected_roles"], flush=True)
            graphs.clear()
            timers.clear()
            del graph, buffers, storage, scale_buffers, original, expected, actual, values, invoke, reset, generation, fi_rotate
            gc.collect()
            torch.cuda.synchronize()
        assert set(all_timings) == {case["tag"] for case in cases}
        result["status"] = "compkv_fp4_complete_matrix_passed_pending_audit"
        result["timing"] = (
            "Complete stage on random/negative-roll distributions; fastest baseline includes stress-failing providers if exact on both timed inputs; candidates pass all seven generations; awaiting independent audit"
        )
    except BaseException:
        result["status"], result["timing"] = "fail", None
        result["traceback"] = traceback.format_exc()
        # This is exception context, not an additional numerical reference.
        # Per-provider numerical mismatches already retain exact paired tensors.
        frame = locals().copy()
        failure_path = args.output.with_suffix(".exception_context.pt")
        try:
            context = {
                name: frame[name].detach().cpu()
                for name in ("original", "values", "positions", "selected_freqs", "fx", "fcache")
                if name in frame and isinstance(frame[name], torch.Tensor)
            }
            if "buffers" in frame:
                context["outputs"] = [value.detach().cpu() for value in frame["buffers"]]
            context["active_stage"] = result.get("active_stage")
            context["active_role"] = result.get("active_role")
            torch.save(context, failure_path)
            result["exception_context"] = dict(path=str(failure_path), sha256=sha(failure_path))
        except Exception:
            result["exception_context_error"] = traceback.format_exc()
        for case in result["cases"].values():
            if case["timing"] is not None:
                case["rejected_timing_diagnostic_only"] = case["timing"]
                case["timing"] = None
        save()
        raise
    finally:
        imports, artifacts = {}, {}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                roots = (HERE, SHARED, SOURCE, Path(cudnn.__file__).parent, Path(flashinfer.__file__).parent)
                if path.is_file() and any(path.is_relative_to(root) for root in roots):
                    imports[str(path)] = sha(path)
        cache_root = Path(os.environ["ROPE_QDQ_CACHE_ROOT"])
        destination = args.output.with_suffix(".generated_artifacts")
        if cache_root.is_dir():
            for path in cache_root.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".cu", ".h", ".so", ".cubin", ".ptx", ".json", ".best_config"):
                    target = destination / path.relative_to(cache_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    assert sha(target) == sha(path)
                    artifacts[str(target)] = dict(sha256=sha(target), original_path=str(path))
        result.update(imported_sources=imports, generated_artifacts=artifacts)
        save()
    print(result["status"], len(result["checks"]), flush=True)


if __name__ == "__main__":
    main()
