# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 tail RoPE public-API correctness and B200 stage benchmark. No model throughput claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rope_qdq"))
from native_baselines import prepare
from source_contract import source_table
import cudnn
import flashinfer
from cudnn.rope import tail as public_api


def fused_math(x, cosine, sine, output):
    shape = x.shape
    n, d = shape[0], shape[-1]
    tail = x.reshape(n, -1, d)[..., -64:].float()
    a, b = tail[..., 0::2], tail[..., 1::2]
    c, s = cosine[:, None, :], sine[:, None, :]
    real = torch.addcmul(-(b * s), a, c).to(torch.bfloat16)
    imag = torch.addcmul(a * s, b, c).to(torch.bfloat16)
    output.copy_(x)
    target = output.reshape(n, -1, d)[..., -64:]
    target[..., 0::2].copy_(real)
    target[..., 1::2].copy_(imag)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--deepseek-source", type=Path, required=True, help="Pinned DeepSeek inference directory containing model.py and config.json")
args = parser.parse_args()
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
assert not output.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
roles = ("frost", "flashinfer", "flashinfer_noftz", "torch_fma", "torch_fma_tuned", "source")
record = dict(
    status="running",
    scope=__doc__,
    performance_admitted=False,
    cases={},
    roles=roles,
    timing_scope="Three-buffer Graph GPU and replay wall, 8 paired blocks, 5 samples, no_evict/evicted. Prepared tables supplied to all providers; table gathering is excluded.",
)


def save():
    temp = output.with_suffix(".writing.json")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(output)


def equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


save()
try:
    assert torch.cuda.get_device_capability() == (10, 0) and torch.cuda.get_device_name() == "NVIDIA B200"
    source = args.deepseek_source.resolve()
    config = json.loads((source / "config.json").read_text())
    assert (config["head_dim"], config["rope_head_dim"], config["n_heads"], config["index_head_dim"], config["index_n_heads"]) == (512, 64, 64, 128, 32)
    record["source_reference"] = dict(
        revision="dba1be0a40aa45a94ad051997016db3960a90277",
        files={str(source / name): sha(source / name) for name in ("model.py", "config.json")},
        attribution="DeepSeek precompute_freqs_cis and apply_rotary_emb, loaded without modification",
        activation_scope="Seeded synthetic BF16; model head geometry and source frequency tables; no checkpoint weights",
    )
    record["environment"] = dict(
        device=torch.cuda.get_device_name(),
        uuid=str(torch.cuda.get_device_properties(0).uuid),
        pid=os.getpid(),
        worker_pid=os.getpid(),
        script_sha256=sha(__file__),
        torch=torch.__version__,
        triton=triton.__version__,
    )
    record["files"] = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            HERE / "source_contract.py",
            HERE.parent / "rope_qdq/native_baselines.py",
            Path(public_api.__file__),
            Path(public_api.__file__).parent / "frost/tail.py",
        )
    }
    noftz, record["native_noftz"] = prepare()
    torch._dynamo.config.recompile_limit = 128
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    record["torch_options"] = dict(base=options, tuned=dict(options, max_autotune=True, coordinate_descent_tuning=True))
    compiled = {
        "torch_fma": torch.compile(fused_math, fullgraph=True, dynamic=False, options=options),
        "torch_fma_tuned": torch.compile(fused_math, fullgraph=True, dynamic=False, options=record["torch_options"]["tuned"]),
    }
    eviction = torch.empty(4 * torch.cuda.get_device_properties(0).L2_cache_size, device="cuda", dtype=torch.uint8)
    record["eviction_bytes"] = eviction.numel()
    for rope_mode in ("base", "compressed"):
        table, apply_source = source_table(source, 98304, compressed=rope_mode == "compressed", device="cuda")
        original_table = table.clone()
        for n in (1, 4, 4096, 16384):
            for h, d, inverse, geometry in (
                (64, 512, False, "q_tp1"),
                (64, 512, True, "inverse_o_tp1"),
                (32, 128, False, "index_q_tp1"),
                (1, 512, False, "kv"),
                (1, 128, False, "index_k"),
                (config["n_heads"] // 8, 512, False, "q_tp8"),
                (config["n_heads"] // 8, 512, True, "inverse_o_tp8"),
                (config["index_n_heads"] // 8, 128, False, "index_q_tp8"),
            ):
                tag = f"{'decode' if n < 4096 else 'prefill'}_n{n}_{geometry}_{rope_mode}"
                record["active_case"] = tag
                print("Begin", tag, flush=True)
                shape = (n, h, d) if h > 1 else (n, d)
                case = record["cases"][tag] = dict(shape=shape, inverse=inverse, checks={}, failures={}, routes={}, samples=[], timing={}, rejected={})
                original = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(596 + n + h + d))
                c = torch.empty(n, 32, device="cuda", dtype=torch.float32)
                s = torch.empty_like(c)
                cache = torch.empty(n, 64, device="cuda", dtype=torch.float32)
                complex_rows = torch.empty(n, 32, device="cuda", dtype=torch.complex64)
                local_positions = torch.arange(n, device="cuda", dtype=torch.int32)
                empty = torch.empty(n, 0, 64, device="cuda", dtype=torch.bfloat16)
                buffers = [dict(x=original.clone(), out=torch.empty_like(original)) for _ in range(3)]
                plan = cudnn.TailRoPEForward(buffers[0]["x"], c, s, buffers[0]["out"], backend="frost")
                plan.compile()
                case["buffers"] = [{key: tensor.data_ptr() for key, tensor in b.items()} for b in buffers]
                assert len({ptr for b in case["buffers"] for ptr in b.values()}) == 6

                def bank_values(bank):
                    if bank == 0:
                        return original
                    if bank == 1:
                        return -original.roll(7, -1)
                    if bank == 2:
                        values = torch.zeros_like(original)
                        values[..., 1::2] = -0.0
                        return values
                    if bank in (3, 4, 5):
                        return (original.float() * {3: 1e-4, 4: 1e-38, 5: 1e4}[bank]).to(torch.bfloat16)
                    values = torch.arange(d, device="cuda", dtype=torch.float32).remainder(32).sub(16).div(16).to(torch.bfloat16)
                    return values.expand(shape).contiguous()

                def reset(bank):
                    values = bank_values(bank)
                    start = (0, 32768, 128, 127, 65592, 32767, 65536)[bank]
                    positions = torch.arange(n, device="cuda") + start
                    selected = table.index_select(0, positions)
                    c.copy_(selected.real)
                    s.copy_(-selected.imag if inverse else selected.imag)
                    cache[:, :32].copy_(c)
                    cache[:, 32:].copy_(s)
                    complex_rows.copy_(selected.conj() if inverse else selected)
                    for b in buffers:
                        b["x"].copy_(values)
                        b["out"].fill_(float("nan"))
                    expected = values.clone()
                    apply_source(expected[None, ..., -64:], selected, inverse=inverse)
                    return values, expected, selected

                values, expected, selected = reset(0)

                def invoke(role):
                    for b in buffers:
                        x, out = b["x"], b["out"]
                        if role == "frost":
                            plan.execute(x, c, s, out)
                        elif role in compiled:
                            compiled[role](x, c, s, out)
                        elif role in ("flashinfer", "flashinfer_noftz"):
                            out.copy_(x)
                            tail = out.reshape(n, h, d)[..., -64:]
                            if role == "flashinfer_noftz":
                                noftz(local_positions, tail, empty, cache)
                            else:
                                flashinfer.apply_rope_with_cos_sin_cache_inplace(local_positions, tail, empty, 64, cache, is_neox=False)
                        else:
                            out.copy_(x)
                            apply_source(out[None, ..., -64:], complex_rows)

                def check(role, phase, bank, values, expected):
                    counts = [int((b["out"].view(torch.int16) != expected.view(torch.int16)).sum()) for b in buffers]
                    readonly = all(equal(b["x"], values) for b in buffers)
                    tables_readonly = equal(c, selected.real) and equal(s, -selected.imag if inverse else selected.imag)
                    tables_readonly = tables_readonly and equal(complex_rows, (selected.conj() if inverse else selected).resolve_conj())
                    tables_readonly = tables_readonly and equal(cache[:, :32], c) and equal(cache[:, 32:], s)
                    tables_readonly = tables_readonly and equal(local_positions, torch.arange(n, device="cuda", dtype=torch.int32))
                    result = dict(
                        differing_elements=counts,
                        input_readonly=readonly,
                        prepared_tables_readonly=tables_readonly,
                        passed=not any(counts) and readonly and tables_readonly,
                    )
                    case["checks"][f"{role}/{phase}/bank{bank}"] = result
                    if not result["passed"]:
                        actual = buffers[0]["out"]
                        indices = (actual.view(torch.int16).reshape(-1) != expected.view(torch.int16).reshape(-1)).nonzero().flatten()[:256]
                        flat_x = values.reshape(-1)
                        pair = indices // 2 * 2
                        rows = indices // (h * d)
                        witness = dict(
                            shape=shape,
                            bank=bank,
                            inverse=inverse,
                            full_difference_count=counts,
                            indices=indices.cpu(),
                            input_pairs=torch.stack((flat_x[pair], flat_x[pair + 1]), -1).cpu(),
                            cosine=c[rows].cpu(),
                            sine=s[rows].cpu(),
                            actual=actual.reshape(-1)[indices].cpu(),
                            expected=expected.reshape(-1)[indices].cpu(),
                            note="First 256 differing elements and exact input pairs/table rows; full input is generated by the retained seed and bank code.",
                        )
                        path = output.with_suffix(f".{tag}.{role}.{phase}.bank{bank}.failure.pt")
                        assert not path.exists()
                        torch.save(witness, path)
                        result["artifact"] = dict(path=str(path), sha256=sha(path))
                    return result["passed"]

                graphs = {}
                for role in roles:
                    try:
                        values, expected, selected = reset(0)
                        invoke(role)
                        start, end = torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            start.record()
                            invoke(role)
                            end.record()
                        flags = []
                        for bank in range(7):
                            values, expected, selected = reset(bank)
                            graph.replay()
                            flags.append(check(role, "before", bank, values, expected))
                        if role in ("frost", "source"):
                            assert all(flags), (tag, role, "seven-generation correctness")
                        case.setdefault("full_contract_passed", {})[role] = all(flags)
                        if all(flags[:2]):
                            graphs[role] = (graph, start, end)
                        else:
                            case["rejected"][role] = "Numerical failure on timed distribution"
                    except BaseException:
                        case["rejected"][role] = traceback.format_exc()
                        save()
                        if role in ("frost", "source"):
                            raise
                    save()
                case["strong_control_available"] = any(role in graphs for role in ("flashinfer", "flashinfer_noftz", "torch_fma", "torch_fma_tuned"))
                assert "frost" in graphs and "source" in graphs
                for role, (graph, start, end) in graphs.items():
                    reset(0)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        graph.replay()
                        torch.cuda.synchronize()
                    path = output.with_suffix(f".{tag}.{role}.trace.json")
                    profiler.export_chrome_trace(str(path))
                    work = [e for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
                    assert len(work) >= 3 and all(e["args"].get("graph id", 0) > 0 for e in work)
                    if role == "frost":
                        assert len(work) == 3 and all("cudnn" in e["name"] and "frost_tail_rope" in e["name"] for e in work)
                    case["routes"][role] = dict(path=str(path), sha256=sha(path), work=[e["name"] for e in work])
                order = list(graphs)
                for regime in ("no_evict", "evicted"):
                    for block in range(8):
                        bank = (block // 2) % 2
                        for role in (order if block % 2 == 0 else order[::-1]):
                            graph, start, end = graphs[role]
                            for sample in range(6):
                                reset(bank)
                                if regime == "evicted":
                                    eviction.fill_(block * 8 + sample + 1)
                                torch.cuda.synchronize()
                                begin = time.perf_counter_ns()
                                graph.replay()
                                end.synchronize()
                                wall_us = (time.perf_counter_ns() - begin) / 3000
                                gpu_us = start.elapsed_time(end) * 1000 / 3
                                if sample:
                                    assert gpu_us > 0 and wall_us > 0
                                    case["samples"].append(
                                        dict(regime=regime, block=block, bank=bank, role=role, sample=sample, gpu_us=gpu_us, wall_us=wall_us)
                                    )
                        save()
                    case["timing"][regime] = {}
                    for metric in ("gpu_us", "wall_us"):
                        medians = {
                            role: [
                                statistics.median(v[metric] for v in case["samples"] if v["role"] == role and v["regime"] == regime and v["block"] == block)
                                for block in range(8)
                            ]
                            for role in graphs
                        }
                        best = min((role for role in graphs if role != "frost"), key=lambda role: statistics.median(medians[role]))
                        ratios = [a / b for a, b in zip(medians[best], medians["frost"])]
                        case["timing"][regime][metric] = dict(
                            fastest_control=best, block_medians=medians, speedup=statistics.median(ratios), wins=sum(x > 1 for x in ratios)
                        )
                for role, (graph, start, end) in graphs.items():
                    for bank in (0, 1):
                        values, expected, selected = reset(bank)
                        graph.replay()
                        assert check(role, "after", bank, values, expected), (tag, role)
                assert equal(table, original_table)
                case["source_table_readonly"] = True
                print(tag, {r: case["timing"][r]["gpu_us"]["speedup"] for r in case["timing"]}, "strong_control", case["strong_control_available"], flush=True)
                del graphs, buffers, original, values, expected, selected
                save()
    assert len(record["cases"]) == 64
    record["all_cases_have_native_or_compiled_control"] = all(c["strong_control_available"] for c in record["cases"].values())
    assert all(sha(path) == digest for path, digest in record["files"].items())
    assert all(sha(path) == digest for path, digest in record["source_reference"]["files"].items())
    record["status"] = "tail_rope_public_api_matrix_pending_independent_review"
except BaseException:
    record.update(status="fail", error=traceback.format_exc())
    save()
    raise
finally:
    save()
