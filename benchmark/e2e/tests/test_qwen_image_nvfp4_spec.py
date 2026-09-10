# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
QWEN_DIR = E2E_DIR / "Qwen-Image"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOWP = load("qwen_image_nvfp4_spec", QWEN_DIR / "modelopt_nvfp4.py")
RUNNER = load("qwen_image_nvfp4_runner_spec", QWEN_DIR / "run_nvfp4.py")


FORMAL_SHAPE = {
    "layers": 4,
    "bs": 1,
    "image_tokens": 4096,
    "text_tokens": 512,
    "hidden": 3072,
    "ffn": 12288,
}


class QwenImageNvfp4SpecTest(unittest.TestCase):
    def test_recipe_is_exact_modelopt_046_fp4_max_without_mha_quantization(self):
        recipe = LOWP.MODELOPT_RECIPE
        self.assertEqual(recipe["release"], "0.46.0")
        self.assertEqual(recipe["commit"], "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a")
        self.assertEqual(
            recipe["upstream_anchor_args"],
            "--model qwen-image --format fp4 --quant-algo max",
        )
        self.assertEqual(recipe["repo_url"], "https://github.com/NVIDIA/Model-Optimizer")
        for key in (
            "selection",
            "qwen_defaults",
            "preset",
            "numerics",
            "mha_policy",
            "real_backend",
        ):
            self.assertIn(key, recipe["sources"])
            self.assertIn(recipe["commit"], recipe["source_permalinks"][key])
            self.assertTrue(recipe["source_permalinks"][key].endswith(recipe["sources"][key]))
        self.assertIn("BFloat16", recipe["proxy_overrides"]["model_dtype"])
        self.assertIn("prepacking", recipe["proxy_overrides"]["weights"])
        self.assertIn("placement", recipe["alignment_scope"])
        self.assertFalse(recipe["quantize_mha"])
        self.assertFalse(recipe["numerical_claim_eligible"])
        self.assertIn("synthetic", recipe["calibration"])

    def test_exact_fourteen_roles_include_both_m1_modulations(self):
        self.assertEqual(len(LOWP.ROLE_ORDER), 14)
        self.assertEqual(LOWP.ROLE_ORDER[:2], ("img_mod.1", "txt_mod.1"))
        self.assertEqual(len(LOWP.MLP_ROLES), 4)
        inputs = LOWP.expected_input_shapes(FORMAL_SHAPE)
        weights = LOWP.expected_weight_shapes(FORMAL_SHAPE)
        self.assertEqual(set(inputs), set(LOWP.ROLE_ORDER))
        self.assertEqual(set(weights), set(LOWP.ROLE_ORDER))
        for role in ("img_mod.1", "txt_mod.1"):
            self.assertEqual(inputs[role], (1, 3072))
            self.assertEqual(weights[role], (18432, 3072))

    def test_formal_proxy_maps_to_reviewed_interior_blocks(self):
        self.assertEqual(LOWP.representative_middle_blocks(4), [2, 20, 39, 57])
        self.assertEqual(LOWP.representative_middle_blocks(1), [30])
        self.assertEqual(LOWP.representative_middle_blocks(56), list(range(2, 58)))
        for invalid in (0, 57, None):
            with self.assertRaises(ValueError):
                LOWP.representative_middle_blocks(invalid)

    def test_exact_seven_low_precision_plan_contracts(self):
        contracts = LOWP.expected_plan_contracts(FORMAL_SHAPE)
        self.assertEqual(len(contracts), 7)
        self.assertEqual(len(set(contracts)), 7)
        self.assertIn((1, 18432, 3072, "linear_bias"), contracts)
        self.assertIn((512, 12288, 3072, "linear_bias_gelu_nvfp4"), contracts)
        self.assertIn((4096, 12288, 3072, "linear_bias_gelu_nvfp4"), contracts)

    def test_three_arm_design_balances_positions_and_ordered_carryover(self):
        orders = LOWP.three_arm_orders()
        self.assertEqual(len(orders), 6)
        positions = Counter((position, treatment) for order in orders for position, treatment in enumerate(order))
        carryover = Counter(pair for order in orders for pair in zip(order, order[1:]))
        self.assertEqual(set(positions.values()), {2})
        self.assertEqual(set(carryover.values()), {2})
        self.assertEqual([arm["id"] for arm in RUNNER.ARMS], ["A", "B", "C"])
        self.assertEqual(RUNNER.ARMS[2]["attention_route"], "cudnn")
        self.assertIn("quantize_mha=False", RUNNER.ARMS[2]["attention"])

    def test_protocol_requires_complete_six_order_cycles(self):
        args = SimpleNamespace(mode="formal", warmup=None, rounds=None, repeats=None)
        self.assertEqual(RUNNER._resolve_protocol(args), {"warmup": 3, "rounds": 42, "repeats": 3})
        args.rounds = 40
        with self.assertRaisesRegex(ValueError, "multiple of 6"):
            RUNNER._resolve_protocol(args)

    def test_c_route_counts_all_roles_and_shared_quantization(self):
        route = LOWP.expected_route_delta("C", 4)
        self.assertEqual(route["nvfp4_linear_calls"], 56)
        self.assertEqual(route["activation_quant_logical"], 56)
        self.assertEqual(route["activation_quant_physical"], 33)
        self.assertEqual(route["activation_quant_standalone"], 25)
        self.assertEqual(route["activation_quant_fused"], 8)
        self.assertEqual(route["activation_cache_hits"], 23)
        self.assertEqual(len(route["nvfp4_linear_by_role"]), 56)
        self.assertEqual(set(route["nvfp4_linear_by_role"].values()), {1})
        self.assertEqual(route["fallback_calls"], 0)
        self.assertEqual(route["weight_pack_calls"], 0)
        self.assertEqual(route["plan_build_calls"], 0)

    def test_a_and_b_keep_all_fourteen_logical_linears_bf16(self):
        a = LOWP.expected_route_delta("A", 4)
        b = LOWP.expected_route_delta("B", 4)
        self.assertEqual(a["bf16_linear_calls"], 56)
        self.assertEqual(b["bf16_linear_calls"], 56)
        self.assertEqual(a["mlp_calls"], {"torch": 8, "cudnn_bf16": 0, "nvfp4": 0})
        self.assertEqual(b["mlp_calls"], {"torch": 0, "cudnn_bf16": 8, "nvfp4": 0})
        self.assertEqual(a["nvfp4_linear_calls"], 0)
        self.assertEqual(b["nvfp4_linear_calls"], 0)

    def test_counter_tree_helpers_preserve_strict_nested_shape(self):
        one = LOWP.expected_route_delta("C", 1)
        two = RUNNER._scale_counter_tree(one, 2)
        self.assertEqual(two["nvfp4_linear_calls"], 28)
        self.assertEqual(set(two["nvfp4_linear_by_role"].values()), {2})
        total = RUNNER._add_counter_trees(one, two)
        self.assertEqual(total["nvfp4_linear_calls"], 42)
        self.assertEqual(LOWP.counter_delta(total, two), one)

    def test_pre_resolved_dynamic_output_is_never_retained(self):
        output_id = 123
        output = object()
        resolved = {456: object()}

        class Compiled:
            def __init__(self, fail=False):
                self.fail = fail
                self.seen = None

            def run_resolved(self, actual, *, stream):
                self.seen = (actual[output_id], stream)
                if self.fail:
                    raise ValueError("launch failed")
                return "ok"

        compiled = Compiled()
        self.assertEqual(
            LOWP._run_resolved_with_temporary_output(compiled, resolved, output_id, output, stream=789),
            "ok",
        )
        self.assertEqual(compiled.seen, (output, 789))
        self.assertNotIn(output_id, resolved)

        compiled = Compiled(fail=True)
        with self.assertRaisesRegex(ValueError, "launch failed"):
            LOWP._run_resolved_with_temporary_output(compiled, resolved, output_id, output, stream=987)
        self.assertEqual(compiled.seen, (output, 987))
        self.assertNotIn(output_id, resolved)

        resolved[output_id] = object()
        with self.assertRaisesRegex(RuntimeError, "already occupied"):
            LOWP._run_resolved_with_temporary_output(Compiled(), resolved, output_id, output, stream=0)

    def test_timed_plan_paths_use_public_pre_resolved_entrypoint(self):
        linear = inspect.getsource(LOWP._Nvfp4LinearPlan.__call__)
        fused = inspect.getsource(LOWP._Nvfp4FusedFc1Plan.__call__)
        self.assertIn("_run_resolved_with_temporary_output", linear)
        self.assertIn("compiled.run_resolved", fused)
        self.assertNotIn(".lowered", linear)
        self.assertNotIn(".lowered", fused)

    def test_prepared_binding_tracks_every_stable_runtime_buffer(self):
        fields = set(LOWP._PreparedNvfp4Binding.__dataclass_fields__)
        self.assertTrue(
            {
                "activation_packed_signature",
                "activation_scale_signature",
                "activation_global_scale_signature",
                "packed_weight_signature",
                "weight_scale_signature",
                "alpha_signature",
                "bias_signature",
            }
            <= fields
        )
        linear_guard = inspect.getsource(LOWP._Nvfp4LinearPlan.validate_prepared)
        fused_guard = inspect.getsource(LOWP._Nvfp4FusedFc1Plan.validate_prepared)
        for signature in (
            "packed_weight_signature",
            "weight_scale_signature",
            "alpha_signature",
            "bias_signature",
        ):
            self.assertIn(signature, linear_guard)
            self.assertIn(signature, fused_guard)
        self.assertNotIn("_tensor_signature", inspect.getsource(LOWP._Nvfp4LinearPlan._binding))
        self.assertNotIn("_tensor_signature", inspect.getsource(LOWP._Nvfp4FusedFc1Plan._binding))

    def test_prepared_binding_rejects_every_replaced_stable_buffer(self):
        class FakeTensor:
            next_pointer = 1000

            def __init__(self, shape=(1,)):
                self.pointer = FakeTensor.next_pointer
                FakeTensor.next_pointer += 1
                self.shape = shape
                self.dtype = "fake_dtype"
                self.device = "cuda:0"

            def data_ptr(self):
                return self.pointer

            def stride(self):
                return tuple(1 for _ in self.shape)

        activation = SimpleNamespace(
            packed=FakeTensor((2, 4)),
            scale_factors=FakeTensor((128, 4)),
            global_scale=FakeTensor(),
        )
        entry = SimpleNamespace(
            qualified_name="transformer_blocks.0.attn.to_q",
            role="attn.to_q",
            m=2,
            n=8,
            k=8,
            activation_global_scale=activation.global_scale,
            packed_weight=FakeTensor((8, 4)),
            weight_scale_factors=FakeTensor((128, 4)),
            alpha=FakeTensor((1, 1, 1)),
            module=SimpleNamespace(bias=FakeTensor((8,))),
        )
        binding = LOWP._PreparedNvfp4Binding(
            entry=entry,
            resolved={111: FakeTensor()},
            resolved_refs={},
            resolved_signatures={},
            activation_packed=activation.packed,
            activation_scale_factors=activation.scale_factors,
            activation_global_scale=activation.global_scale,
            packed_weight=entry.packed_weight,
            weight_scale_factors=entry.weight_scale_factors,
            alpha=entry.alpha,
            bias=entry.module.bias,
            activation_packed_signature=LOWP._tensor_signature(activation.packed),
            activation_scale_signature=LOWP._tensor_signature(activation.scale_factors),
            activation_global_scale_signature=LOWP._tensor_signature(activation.global_scale),
            packed_weight_signature=LOWP._tensor_signature(entry.packed_weight),
            weight_scale_signature=LOWP._tensor_signature(entry.weight_scale_factors),
            alpha_signature=LOWP._tensor_signature(entry.alpha),
            bias_signature=LOWP._tensor_signature(entry.module.bias),
            output_id=222,
        )
        binding.resolved_refs = dict(binding.resolved)
        binding.resolved_signatures = {key: LOWP._tensor_signature(value) for key, value in binding.resolved_refs.items()}
        plan = object.__new__(LOWP._Nvfp4LinearPlan)
        plan._prepared = {id(entry): binding}
        self.assertIs(plan._binding(activation, entry, entry.alpha, entry.module.bias), binding)

        adapter = object.__new__(LOWP.QwenImageModelOptNvfp4Adapter)
        adapter._active = False
        adapter._device = SimpleNamespace(index=0)
        adapter._stream = 333
        adapter._validate_stream = lambda device: None
        adapter.entries = [entry]
        adapter.by_name = {entry.qualified_name: entry}
        adapter._activation_buffers = {(entry.m, entry.k): (activation.packed, activation.scale_factors)}
        adapter._linear_plans = {(0, 333, entry.m, entry.n, entry.k, "linear_bias"): plan}
        adapter._fused_fc1_plans = {}
        adapter._installed_generic = {}
        adapter._installed_mod = {}
        adapter._installed_mlp = {}
        adapter.select("C")

        for owner, attribute in (
            (entry, "activation_global_scale"),
            (entry, "packed_weight"),
            (entry, "weight_scale_factors"),
            (entry, "alpha"),
            (entry.module, "bias"),
        ):
            original = getattr(owner, attribute)
            replacement = FakeTensor(original.shape)
            setattr(owner, attribute, replacement)
            with self.assertRaisesRegex(RuntimeError, "changed after NVFP4 preparation"):
                adapter.select("C")
            setattr(owner, attribute, original)

        original_buffers = adapter._activation_buffers[(entry.m, entry.k)]
        for index in (0, 1):
            changed = list(original_buffers)
            changed[index] = FakeTensor(changed[index].shape)
            adapter._activation_buffers[(entry.m, entry.k)] = tuple(changed)
            with self.assertRaisesRegex(RuntimeError, "changed after NVFP4 preparation"):
                adapter.select("C")
            adapter._activation_buffers[(entry.m, entry.k)] = original_buffers

        original_resolved = binding.resolved[111]
        binding.resolved[111] = FakeTensor(original_resolved.shape)
        with self.assertRaisesRegex(RuntimeError, "resolved binding changed"):
            adapter.select("C")
        binding.resolved[111] = original_resolved

    def test_fused_prepared_binding_rejects_hgs_and_fixed_output_replacement(self):
        class FakeTensor:
            next_pointer = 2000

            def __init__(self, shape=(1,)):
                self.pointer = FakeTensor.next_pointer
                FakeTensor.next_pointer += 1
                self.shape = shape
                self.dtype = "fake_dtype"
                self.device = "cuda:0"

            def data_ptr(self):
                return self.pointer

            def stride(self):
                return tuple(1 for _ in self.shape)

        activation = SimpleNamespace(
            packed=FakeTensor((2, 4)),
            scale_factors=FakeTensor((128, 4)),
            global_scale=FakeTensor(),
        )
        first = SimpleNamespace(
            qualified_name="transformer_blocks.0.img_mlp.net.0.proj",
            role="img_mlp.net.0.proj",
            m=2,
            n=32,
            k=8,
            activation_global_scale=activation.global_scale,
            packed_weight=FakeTensor((32, 4)),
            weight_scale_factors=FakeTensor((128, 4)),
            alpha=FakeTensor((1, 1, 1)),
            module=SimpleNamespace(bias=FakeTensor((32,))),
        )
        second = SimpleNamespace(
            qualified_name="transformer_blocks.0.img_mlp.net.2",
            role="img_mlp.net.2",
            m=2,
            n=8,
            k=32,
            activation_global_scale=FakeTensor(),
            packed_weight=FakeTensor((8, 16)),
            weight_scale_factors=FakeTensor((128, 4)),
            alpha=FakeTensor((1, 1, 1)),
            module=SimpleNamespace(bias=FakeTensor((8,))),
        )
        qh, sh = FakeTensor((2, 16)), FakeTensor((128, 4))
        hidden_view = FakeTensor((1, 1, 1))
        resolved_tensor = FakeTensor()
        binding = LOWP._PreparedNvfp4Binding(
            entry=first,
            resolved={111: resolved_tensor},
            resolved_refs={111: resolved_tensor},
            resolved_signatures={111: LOWP._tensor_signature(resolved_tensor)},
            activation_packed=activation.packed,
            activation_scale_factors=activation.scale_factors,
            activation_global_scale=activation.global_scale,
            packed_weight=first.packed_weight,
            weight_scale_factors=first.weight_scale_factors,
            alpha=first.alpha,
            bias=first.module.bias,
            activation_packed_signature=LOWP._tensor_signature(activation.packed),
            activation_scale_signature=LOWP._tensor_signature(activation.scale_factors),
            activation_global_scale_signature=LOWP._tensor_signature(activation.global_scale),
            packed_weight_signature=LOWP._tensor_signature(first.packed_weight),
            weight_scale_signature=LOWP._tensor_signature(first.weight_scale_factors),
            alpha_signature=LOWP._tensor_signature(first.alpha),
            bias_signature=LOWP._tensor_signature(first.module.bias),
            hidden_global_scale=hidden_view,
            hidden_global_scale_source=second.activation_global_scale,
            hidden_global_scale_signature=LOWP._tensor_signature(hidden_view),
            hidden_global_scale_source_signature=LOWP._tensor_signature(second.activation_global_scale),
            output_packed=qh,
            output_scale_factors=sh,
            output_packed_signature=LOWP._tensor_signature(qh),
            output_scale_signature=LOWP._tensor_signature(sh),
        )
        plan = object.__new__(LOWP._Nvfp4FusedFc1Plan)
        plan._prepared = {id(first): binding}
        plan.qh, plan.sh = qh, sh
        plan.validate_prepared(
            (activation.packed, activation.scale_factors),
            first,
            second.activation_global_scale,
        )
        self.assertIs(plan._binding(activation, first, first.alpha, first.module.bias), binding)

        original_hgs = second.activation_global_scale
        second.activation_global_scale = FakeTensor()
        with self.assertRaisesRegex(RuntimeError, "changed after NVFP4 preparation"):
            plan.validate_prepared(
                (activation.packed, activation.scale_factors),
                first,
                second.activation_global_scale,
            )
        second.activation_global_scale = original_hgs

        for attribute in ("qh", "sh"):
            original = getattr(plan, attribute)
            setattr(plan, attribute, FakeTensor(original.shape))
            with self.assertRaisesRegex(RuntimeError, "changed after NVFP4 preparation"):
                plan.validate_prepared(
                    (activation.packed, activation.scale_factors),
                    first,
                    second.activation_global_scale,
                )
            with self.assertRaisesRegex(RuntimeError, "changed after NVFP4 preparation"):
                plan._binding(activation, first, first.alpha, first.module.bias)
            setattr(plan, attribute, original)

    def test_report_separates_all_three_claims(self):
        batches = {"A": [10.0, 10.2], "B": [8.0, 8.1], "C": [6.0, 6.1]}
        summary = {}
        for arm, values in batches.items():
            summary[arm] = {
                "p10_ms": min(values),
                "p50_ms": sum(values) / len(values),
                "p90_ms": max(values),
                **RUNNER.paired_stats(values, batches["A"]),
            }
        comparisons = {
            "B_vs_A": RUNNER.paired_stats(batches["B"], batches["A"]),
            "C_vs_B": RUNNER.paired_stats(batches["C"], batches["B"]),
            "C_vs_A": RUNNER.paired_stats(batches["C"], batches["A"]),
        }
        metadata = {
            "completed_utc": "2026-08-21T00:00:00Z",
            "config": {
                "mode": "formal",
                "comparability_fingerprint": {"sha256": "comparable"},
                "build_fingerprint": {"sha256": "build"},
                "shape": {
                    **FORMAL_SHAPE,
                    "joint_tokens": 4608,
                    "heads": 24,
                    "head_dim": 128,
                },
                "representative_full_blocks": [2, 20, 39, 57],
                "numerical_recipe": dict(LOWP.MODELOPT_RECIPE),
            },
            "summary": summary,
            "comparisons": comparisons,
            "correctness": {"model_output_rel_l2_vs_A": {"A": 0.0, "B": 0.01, "C": 0.1}},
            "route": {
                "torch_probe": {
                    "natural_choice_name": "CUDNN_ATTENTION",
                    "forced_choice_name": "FLASH_ATTENTION",
                },
                "expected_C_per_forward": LOWP.expected_route_delta("C", 4),
            },
            "provenance": {"sources": {"adapter": {"path": "adapter.py", "sha256": "face"}}},
        }
        report = RUNNER._render_markdown(metadata, "result.json", "cafe")
        self.assertIn("BF16 cuDNN effect (B/A)", report)
        self.assertIn("ModelOpt NVFP4 increment (C/B)", report)
        self.assertIn("Total cuDNN platform impact (C/A)", report)
        self.assertIn("quantize_mha` defaults to false", report)
        self.assertIn("all fourteen Linear roles", report)
        self.assertIn("quality-ineligible", report)

    def test_report_labels_elapsed_time_regressions_as_slower(self):
        self.assertEqual(
            RUNNER._format_elapsed_effect(0.8),
            "1.250x speedup (20.00% lower elapsed time)",
        )
        self.assertEqual(
            RUNNER._format_elapsed_effect(1.0898134),
            "1.090x slower (8.98% higher elapsed time)",
        )
        self.assertEqual(RUNNER._format_elapsed_effect(1.0), "1.000x (no elapsed-time change)")
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RUNNER._format_elapsed_effect(invalid)


if __name__ == "__main__":
    unittest.main()
