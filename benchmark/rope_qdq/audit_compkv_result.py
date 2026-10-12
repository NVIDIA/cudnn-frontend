# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify complete matrix coverage, hashes, routes and timing arithmetic.

This read-only audit does not establish GPU isolation or admit performance.
It requires the recorded source, traces and generated artifacts to be present.
"""

# Source, numerical and admission checks below rely on assertions.
if not __debug__:
    raise RuntimeError("This benchmark/auditor requires Python assertions; run without -O/-OO or PYTHONOPTIMIZE.")

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_metric(metric):
    assert metric["passed"] == (metric["finite"] and metric["differing_bf16_bits"] == 0)
    assert 0 <= metric["differing_bf16_bits"] <= metric["elements"]
    if metric["passed"]:
        assert metric["relative_l2"] == metric["max_scaled"] == 0.0
    elif metric["finite"]:
        assert all(metric[key] is None or (math.isfinite(metric[key]) and metric[key] >= 0) for key in ("relative_l2", "max_scaled"))


def validate_trace(item, role, calls, quantization):
    assert sha(item["path"]) == item["sha256"]
    work = [event for event in json.loads(Path(item["path"]).read_text())["traceEvents"] if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    assert len(work) >= calls
    for key, reported in (("graph id", "graph_id"), ("correlation", "correlation"), ("stream", "stream")):
        values = {event["args"].get(key) for event in work}
        assert len(values) == 1 and all(isinstance(value, int) and value > 0 for value in values)
        if reported in item:
            assert values == {item[reported]}
    names = [event["name"] for event in work]
    assert not any(re.search(r"DtoH|HtoD|device.*host|host.*device", name, re.I) for name in names)
    if role.startswith("frost"):
        assert len(work) == calls and all(event["cat"] == "kernel" and "cudnn_frost_compkv_rope_qdq" in event["name"] for event in work)
    if role.startswith("flashinfer"):
        assert sum("BatchQKApplyRotaryPosIdsCosSinCache" in name for name in names) == calls
    if role in ("source_wrapper", "source_quant_alias", "flashinfer_quant_wrapper", "flashinfer_quant_alias", "flashinfer_noftz_quant_alias"):
        symbol = "fp4_quant_kernel" if quantization == "fp4" else "act_quant_kernel"
        assert sum(symbol in name for name in names) == calls
    return len(work)


def audit(path):
    path = Path(path)
    raw = json.loads(path.read_text())
    quantization = "fp4"
    assert raw["status"] == "compkv_fp4_complete_matrix_passed_pending_audit"
    assert not raw["performance_admitted"]
    environment = raw["environment"]
    assert environment["candidate_api"] == "cudnn.RopeQDQInplace"
    api = Path(environment["api_path"])
    assert sha(api) == environment["api_sha256"]
    assert sha(api.parent / "frost/compressed_kv.py") == environment["kernel_sha256"]
    assert sha(api.parent / "frost/kernels.py") == environment["shared_kernel_sha256"]
    native = (
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
    )
    candidates = ("frost",)
    roles = list(native) + list(candidates)
    required = {
        "source_wrapper",
        "source_quant_alias",
        "flashinfer_noftz_quant_alias",
        "torch_fma_eager",
        "torch_fma_fused",
        "torch_fma_fused_tuned",
        "frost",
    }
    generations = {"random", "negative_roll", "signed_zero", "small", "subnormal", "large", "midpoints"}
    assert raw["roles"] == roles and set(raw["required_roles"]) == required
    assert set(raw["generations"]) == generations
    assert raw["timed_generations"] == ["random", "negative_roll"]
    assert set(raw["timed_required_roles"]) == required | {"flashinfer_quant_alias"}
    specs = {f"prefill_b{b}_s{s}_ratio{r}": (b, s // r, "compressed", 0, r, s, 0) for b in (1, 4) for s in (4096, 16384) for r in (1, 2)}
    specs.update({f"decode_b{b}_ratio{r}": (b, 1, "compressed", 4096, r, 1, 4096 if r == 1 else 4097) for b in (1, 4) for r in (1, 2)})
    assert raw["consumer_contract"]["quantization"] == dict(
        value="E2M1", group=16, scale="E4M3", minimum_scale=2**-9, maximum_scale=448.0, global_scale=None, output="BF16 QDQ"
    )
    assert raw["native_nvfp4_global_scale"] == 1.0
    assert set(raw["cases"]) == set(specs)
    for filename, digest in raw["imported_sources"].items():
        assert sha(filename) == digest, filename
    for filename, artifact in raw["generated_artifacts"].items():
        assert sha(filename) == artifact["sha256"], filename
    controlled = raw["noftz_control"]
    assert controlled["controlled_cuda_flags"] == controlled["original_cuda_flags"] + ["--ftz=false"]
    for filename, digest in controlled["sources"].items():
        assert sha(filename) == digest
    for metric in raw["checks"].values():
        validate_metric(metric)
    expected_checks, negatives, samples, routes, alignment_checks = set(), {"omitted_bf16_rounding"}, 0, 0, 0
    summaries = []
    for tag, spec in specs.items():
        case = raw["cases"][tag]
        fields = ("batch", "sequence", "rope", "start_pos", "position_stride", "source_sequence", "source_start_pos")
        assert tuple(case["spec"][key] for key in fields) == spec
        assert case["spec"]["head_dim"] == 512 and case["spec"]["heads"] == 1
        assert case["spec"]["tokens"] == spec[0] * spec[1]
        strict = {}
        for role in roles:
            keys = {f"{tag}/{generation}/{role}/{mode}/{index}" for generation in generations for mode in ("eager", "graph") for index in range(3)}
            expected_checks.update(keys)
            strict[role] = all(raw["checks"][key]["passed"] for key in keys)
            assert all(raw["checks"][key]["guards_unchanged"] and raw["checks"][key]["elements"] == spec[0] * spec[1] * 512 for key in keys)
        assert all(strict[role] for role in required)
        assert case["eligible"] == [role for role in roles if strict[role]]
        assert case["rejected_roles"] == [role for role in roles if not strict[role]]
        assert set(case["failures"]) == set(case["rejected_roles"])
        for role, failure in case["failures"].items():
            assert sha(failure["path"]) == failure["sha256"]
            assert not raw["checks"][failure["first_check"]]["passed"]
        valid = [
            role
            for role in roles
            if all(
                raw["checks"][f"{tag}/{generation}/{role}/{mode}/{index}"]["passed"]
                for generation in ("random", "negative_roll")
                for mode in ("eager", "graph")
                for index in range(3)
            )
        ]
        assert case["timing_eligible"] == valid and required | {"flashinfer_quant_alias"} <= set(valid)
        assert case["timed_with_stress_failures"] == [role for role in valid if not strict[role]]
        assert set(case["routes"]) == set(valid)
        for role, item in case["routes"].items():
            validate_trace(item, role, 3, quantization)
            routes += 1
        alignment_keys = {
            f"align{alignment}/{generation}/{mode}"
            for alignment in (2, 4, 16)
            for generation in ("random", "signed_zero", "subnormal")
            for mode in ("eager", "graph")
        }
        assert set(case["alignment_checks"]) == alignment_keys
        for metric in case["alignment_checks"].values():
            validate_metric(metric)
            assert metric["passed"] and metric["guards_unchanged"]
            alignment_checks += 1
        assert set(case["alignment_routes"]) == {"align2", "align4", "align16"}
        for item in case["alignment_routes"].values():
            validate_trace(item, "frost", 1, quantization)
            routes += 1
        buffers = case["buffers"]
        assert buffers["bytes_each"] == spec[0] * spec[1] * 512 * 2
        assert buffers["independent"] and buffers["calls_per_graph"] == 3
        ptrs = sorted(buffers["pointers"])
        assert len(ptrs) == 3 and all(a + buffers["bytes_each"] <= b for a, b in zip(ptrs, ptrs[1:]))
        negatives.update(f"{tag}/{generation}/{negative}" for generation in ("random", "negative_roll") for negative in ("no_op", "zeros", "nan"))
        for generation in ("random", "negative_roll"):
            for role in valid:
                for index in range(3):
                    key = f"{tag}/after/{generation}/{role}/{index}"
                    expected_checks.add(key)
                    assert raw["checks"][key]["passed"]
        assert set(case["timing"]) == {"hot", "evicted"}
        for regime, timing in case["timing"].items():
            blocks = timing["blocks"]
            assert len(blocks) == 8
            for index, block in enumerate(blocks):
                assert block["index"] == index and block["order"] == (valid if index % 2 == 0 else list(reversed(valid)))
                assert block["input_generation"] == ("random", "negative_roll")[(index // 2) % 2]
                assert set(block["arms"]) == set(valid)
                for values in block["arms"].values():
                    assert set(values) == {"gpu_us", "wall_us"}
                    for values in values.values():
                        assert len(values) == 5 and all(math.isfinite(v) and v > 0 for v in values)
                        samples += len(values)
            for metric in ("gpu_us", "wall_us"):
                by_block = {role: [statistics.median(block["arms"][role][metric]) for block in blocks] for role in valid}
                medians = {role: statistics.median(values) for role, values in by_block.items()}
                baseline = min((role for role in valid if not role.startswith("frost")), key=medians.get)
                summary = timing["summary"][metric]
                assert summary["medians"] == medians and summary["fastest_valid_native"] == baseline
                expected_candidates = {}
                for role in candidates:
                    ratios = [a / b for a, b in zip(by_block[baseline], by_block[role])]
                    ratio = statistics.median(ratios)
                    expected_candidates[role] = dict(
                        paired_ratios=ratios, paired_speedup=ratio, latency_reduction_pct=100 * (1 - 1 / ratio), wins=sum(r > 1 for r in ratios)
                    )
                    summaries.append(
                        dict(
                            case=tag,
                            phase=case["spec"]["phase"],
                            candidate=role,
                            regime=regime,
                            metric=metric,
                            baseline=baseline,
                            baseline_us=medians[baseline],
                            frost_us=medians[role],
                            paired_speedup=ratio,
                            wins=sum(r > 1 for r in ratios),
                        )
                    )
                assert summary["candidates"] == expected_candidates
    assert set(raw["checks"]) == expected_checks and set(raw["negative_controls"]) == negatives
    assert all(control["rejected"] and not control["metrics"]["passed"] for control in raw["negative_controls"].values())
    return dict(
        status="complete_numerics_routes_summaries_audited_environment_pending",
        quantization=quantization,
        checks=len(expected_checks),
        alignment_checks=alignment_checks,
        routes=routes,
        samples=samples,
        raw_sha256=sha(path),
        performance_admitted=False,
        summaries=summaries,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.result), indent=2))
