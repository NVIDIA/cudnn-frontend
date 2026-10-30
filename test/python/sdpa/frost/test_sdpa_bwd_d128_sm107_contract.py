# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts for the native SM107 dense D128 backward plan."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cudnn
import pytest
import torch

pytestmark = [pytest.mark.L0]
ENGINE = "sdpa_bwd_sm107_d128"
VERSION = ("nvidia-cutlass-dsl", "4.8.1")
ROOT = Path(__file__).resolve().parents[4]


class TestDeclaration(unittest.TestCase):
    def validate(self, **changed):
        from cudnn.sdpa.bwd.d128_sm107_contract import validate_declaration

        args = dict(cc=(10, 7), version=VERSION, batch=2, heads=3, sq=256, skv=128, scale=128**-0.5)
        validate_declaration(**(args | changed))

    def test_supported_dense_extents(self):
        for sq, sk in ((128, 128), (256, 128), (128, 384), (65, 129), (257, 193), (2048, 2048), (8192, 8192)):
            with self.subTest(sq=sq, sk=sk):
                self.validate(sq=sq, skv=sk)

    def test_exact_architecture(self):
        for cc in ((9, 0), (10, 0), (10, 3), (10, 6), (10, 8), (11, 0), (12, 0)):
            with self.subTest(cc=cc), self.assertRaisesRegex(ValueError, "SM107"):
                self.validate(cc=cc)

    def test_dsl_version_declines(self):
        for version in (None, ("nvidia-cutlass-dsl", "4.7.9"), ("nvidia-cutlass-dsl", "4.8.0"), ("nvidia-cutlass-dsl", "unknown"), ("other", "4.8.1")):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, r"requires nvidia-cutlass-dsl >= 4\.8\.1; found") as error:
                    self.validate(version=version)
                self.assertIn(str(version), str(error.exception))
        self.validate(version=("nvidia-cutlass-dsl", "4.8.1"))

    def test_invalid_dimensions(self):
        for name in ("batch", "heads", "sq", "skv"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive integers"):
                    self.validate(**{name: value})

    def test_grid_stats_and_tma_bounds(self):
        for changed in ({"batch": 65536}, {"heads": 65536}, {"sq": 2**31}, {"skv": 2**31}, {"batch": 65535, "heads": 257}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.validate(**changed)
        # The scalar scheduler divides by swizzle * ceil(S_kv / tile).
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.validate(batch=1, heads=1, sq=128, skv=2**29)
        # The rounded Stats bound also bounds FP32 dQ byte strides below 2**40.
        self.validate(batch=1, heads=1, sq=2**31 - 128, skv=128)

    def test_scale_requires_positive_finite_fp32(self):
        for scale in (0.0, -0.125, float("nan"), float("inf"), 1e-50, 3e38):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "scale"):
                self.validate(scale=scale)
        self.validate(scale=0.125)

    def test_scale_never_coerces_a_tensor(self):
        class TensorScalar:
            def __float__(self):
                raise AssertionError("host preflight tried to read a tensor")

        for value in (TensorScalar(), True, "0.125"):
            with self.subTest(value=type(value)), self.assertRaisesRegex(ValueError, "host real scalar"):
                self.validate(scale=value)

    def test_single_query_and_single_key_decline(self):
        for changed in ({"sq": 1}, {"skv": 1}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "single-query/key"):
                self.validate(**changed)

    def test_workspace_is_caller_owned_and_covers_alignment(self):
        from cudnn.sdpa.bwd.d128_contract import workspace_layout

        for sq in (65, 128, 129, 8192):
            regions, total = workspace_layout(2, 3, sq)
            rounded = ((sq + 127) // 128) * 128
            dq_bytes, delta_bytes = 4 * 2 * 3 * rounded * 128, 4 * 2 * 3 * rounded
            self.assertEqual(regions[0][0], 0)
            self.assertEqual(regions[1][0], dq_bytes)
            self.assertEqual(total, dq_bytes + delta_bytes + 240)
            for residue in range(0, 256, 16):
                self.assertLessEqual((-residue) % 256 + dq_bytes + delta_bytes, total)


class TestEngineRow(unittest.TestCase):
    def spec(self):
        from cudnn.sdpa.bwd.engines import ENGINE_SPECS

        rows = [row for row in ENGINE_SPECS if row.name == ENGINE]
        self.assertEqual(len(rows), 1, "one native SM107 D128 engine row must exist")
        return rows[0]

    def facts(self, **changes):
        from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

        return SdpaGraphFacts(
            **(
                dict(
                    device_cc=(10, 7),
                    is_backward=True,
                    b=2,
                    h_q=3,
                    h_kv=3,
                    s_q=128,
                    s_kv=256,
                    d_qk=128,
                    d_v=128,
                    dtype=cudnn.data_type.BFLOAT16,
                    scale=128**-0.5,
                )
                | changes
            )
        )

    def reason(self, **changes):
        from cudnn.sdpa.bwd import engines

        with patch.object(engines, "cutedsl_state", return_value=(True, VERSION)), patch.object(engines, "cutedsl_arch_requirement_error", return_value=None):
            return engines.mismatch(self.spec().capabilities, self.facts(**changes))

    def test_row_accepts_both_dtypes_and_masks(self):
        for dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    self.assertIsNone(self.reason(dtype=dtype, causal=causal))

    def test_row_dsl_floor_boundary_before_lowering(self):
        from cudnn.sdpa.bwd import engines

        capabilities, facts = self.spec().capabilities, self.facts()
        for version in ("4.8.0", "4.8.1"):
            metadata = ("nvidia-cutlass-dsl", version)
            with (
                self.subTest(version=version),
                patch.object(engines, "cutedsl_state", return_value=(True, metadata)),
                patch.object(engines, "cutedsl_arch_requirement_error", return_value=None),
            ):
                reason = engines.mismatch(capabilities, facts)
                if version == "4.8.0":
                    self.assertIsNotNone(reason, "DSL 4.8.0 must decline before adapter lowering")
                    self.assertRegex(reason, r"requires nvidia-cutlass-dsl >= 4\.8\.1; found")
                    self.assertIn(str(metadata), reason)
                else:
                    self.assertIsNone(reason)

    def test_manifest_slot_and_unmeasured_placement(self):
        tree = ast.parse((ROOT / "python/cudnn/engines/manifest.py").read_text())
        slots = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == ENGINE
        ]
        self.assertEqual(len(slots), 1)
        self.assertEqual(ast.literal_eval(slots[0].args[0]), 11)
        self.assertEqual(slots[0].keywords, [])
        tree = ast.parse((ROOT / "python/cudnn/sdpa/bwd/heuristics.py").read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TRAIL_ONLY_ROWS" for target in node.targets)
        )
        self.assertIn(ENGINE, ast.literal_eval(assignment.value.args[0]))

    def test_row_declines_unserved_features(self):
        cases = (
            ({"deterministic": True}, "deterministic"),
            ({"thd": True}, "THD"),
            ({"h_kv": 1}, "GQA"),
            ({"padded": True}, "padding"),
            ({"bottom_right": True, "causal": True}, "bottom-right"),
            ({"window_left": 16}, "window"),
            ({"has_score_mod": True}, "score_mod"),
            ({"has_dropout": True}, "dropout"),
            ({"has_bias": True}, "bias"),
            ({"has_sink": True}, "sink"),
            ({"d_qk": 64, "d_v": 64}, "D"),
            ({"device_cc": (10, 0)}, "SM107"),
            ({"scale": -0.125}, "scale"),
            ({"scale": 0.0}, "scale"),
        )
        for changed, token in cases:
            with self.subTest(changed=changed):
                self.assertIn(token.lower(), self.reason(**changed).lower())

    def test_row_rejects_padded_payloads_and_knobs(self):
        from cudnn.sdpa.bwd import engines

        for port in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
            reason = self.reason(port_layouts=((port, (2, 3, 128, 128), (49280, 128, 384, 1)),))
            self.assertIn("compact BSHD", reason)
        with patch.object(engines, "cutedsl_state", return_value=(True, VERSION)), patch.object(engines, "cutedsl_arch_requirement_error", return_value=None):
            for knobs in (engines.SdpaBwdKnobs(tile_m=128), engines.SdpaBwdKnobs(tile_n=128)):
                self.assertIn("domain", engines.mismatch(self.spec().capabilities, self.facts(), knobs))


class _Desc:
    def __init__(self, shape, stride, dtype, device):
        self.shape, self.stride, self.dtype, self.device, self.ndim = shape, stride, dtype, device, len(shape)

    def is_contiguous(self):
        step = 1
        for n, st in zip(reversed(self.shape), reversed(self.stride)):
            if n > 1 and st != step:
                return False
            step *= n
        return True


class TestAdapter(unittest.TestCase):
    def api(self, dtype=None, **changed):
        from cudnn.sdpa.bwd.api_dsl_sm107_d128 import SdpaBwdDslSm107D128

        api = object.__new__(SdpaBwdDslSm107D128)
        device = torch.device("cuda", 0)
        dtype = torch.bfloat16 if dtype is None else dtype
        attrs = dict(
            thd=False,
            seq_q_lens_present=False,
            seq_kv_lens_present=False,
            causal_bottom_right=False,
            is_causal=False,
            deterministic=False,
            window_size_left=None,
            window_size_right=None,
            tile_m=None,
            tile_n=None,
            sink_desc=None,
            dsink_desc=None,
            bias_desc=None,
            dbias_desc=None,
            scale_softmax=None,
            max_total_seq_len_q=None,
            max_total_seq_len_kv=None,
            thd_stats_token_major=False,
            thd_stats_head_stride=0,
        )
        for name in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
            s = 256 if name in ("k", "v", "dk", "dv") else 128
            attrs[name + "_desc"] = _Desc((2, 3, s, 128), (s * 3 * 128, 128, 3 * 128, 1), dtype, device)
        attrs["stats_desc"] = _Desc((2, 3, 128, 1), (384, 128, 1, 1), torch.float32, device)
        for name, value in (attrs | changed).items():
            setattr(api, name, value)
        api._initialize_implementation()
        return api

    def support(self, api, cc=(10, 7), version=VERSION):
        from cudnn import _pybind_module
        from cudnn.frost import buffers, device

        with (
            patch.object(buffers, "cutedsl_state", return_value=(True, version)),
            patch.object(buffers, "cutedsl_arch_requirement_error", return_value=None),
            patch.object(device, "compute_capability", return_value=cc),
            patch.object(device, "resolve_device", return_value=0),
            patch.object(_pybind_module, "_SdpaBwdBinder", SimpleNamespace(supports_zero_workspace=True)),
        ):
            return api.check_support()

    def test_adapter_accepts_same_dtypes_and_masks(self):
        for dtype in (torch.bfloat16, torch.float16):
            for causal in (False, True):
                api = self.api(dtype=dtype, is_causal=causal)
                self.assertTrue(self.support(api))
                self.assertEqual(api._implementation, "accumulation")
                self.assertGreater(api.scratch_workspace_bytes(), 0)

    def test_adapter_declines_feature_cross_products(self):
        for changed, token in (
            ({"thd": True}, "THD"),
            ({"deterministic": True}, "deterministic"),
            ({"causal_bottom_right": True}, "bottom-right"),
            ({"window_size_left": 0}, "window"),
            ({"window_size_right": 1}, "right-band"),
            ({"seq_q_lens_present": True}, "padding"),
            ({"seq_kv_lens_present": True}, "padding"),
            ({"max_total_seq_len_q": 256}, "THD"),
            ({"tile_m": 128}, "fixed tile"),
            ({"sink_desc": object()}, "sink"),
            ({"scale_softmax": -0.125}, "scale"),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(NotImplementedError, token):
                self.support(self.api(**changed))

    def test_adapter_declines_grouped_layout_dtype_and_stats(self):
        for port in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
            api = self.api()
            desc = getattr(api, port + "_desc")
            desc.stride = (desc.stride[0] + 128, *desc.stride[1:])
            with self.subTest(port=port), self.assertRaisesRegex(NotImplementedError, "compact BSHD"):
                self.support(api)
        api = self.api()
        api.k_desc.shape = (2, 1, 256, 128)
        with self.assertRaisesRegex(NotImplementedError, "MHA"):
            self.support(api)
        api = self.api()
        api.dv_desc.dtype = torch.float16
        with self.assertRaisesRegex(NotImplementedError, "dtype"):
            self.support(api)
        api = self.api()
        api.stats_desc.stride = (768, 256, 2, 1)
        with self.assertRaisesRegex(NotImplementedError, "Stats"):
            self.support(api)

    def test_adapter_declines_before_compile_on_other_arch_or_dsl(self):
        for cc, version, token in (((10, 3), VERSION, "SM107"), ((10, 7), ("nvidia-cutlass-dsl", "4.7.0"), "nvidia-cutlass-dsl")):
            with self.subTest(cc=cc, version=version), self.assertRaisesRegex(NotImplementedError, token):
                self.support(self.api(), cc=cc, version=version)

    def test_adapter_dsl_floor_boundary_resets_support(self):
        api = self.api()
        self.assertTrue(self.support(api, version=("nvidia-cutlass-dsl", "4.8.1")))
        metadata = ("nvidia-cutlass-dsl", "4.8.0")
        with self.assertRaisesRegex(NotImplementedError, r"requires nvidia-cutlass-dsl >= 4\.8\.1; found") as error:
            self.support(api, version=metadata)
        self.assertIn(str(metadata), str(error.exception))
        self.assertFalse(api._is_supported)
        self.assertIsNone(api._prepared)
        self.assertTrue(self.support(api, version=("nvidia-cutlass-dsl", "4.8.1")))

    def test_launch_guard_rejects_aliases_and_invalid_scale_before_entry(self):
        from cudnn.sdpa.bwd.prepared import Operand
        from cudnn.sdpa.bwd.prepared_sm100_d128 import _GuardedAccumulationEntry

        calls = []
        operands = [Operand("bfloat16", (128,), (1,), 128, 16, 2)] * 9
        guard = _GuardedAccumulationEntry(lambda *args: calls.append(args), operands, 4096)
        addresses = [0x100000 + i * 0x10000 for i in range(9)]
        guard(*addresses, 0x400000, 0.125, 17)
        self.assertEqual(len(calls), 1)
        for changed, workspace, scale in (
            (addresses[:6] + [addresses[0]] + addresses[7:], 0x400000, 0.125),
            (addresses, addresses[0], 0.125),
            (addresses, 0x400000, 0.0),
        ):
            with self.assertRaises(ValueError):
                guard(*changed, workspace, scale, 17)
        self.assertEqual(len(calls), 1)


class TestNativeRuntimeGate(unittest.TestCase):
    def require_native(self, version, target_available=True):
        from cudnn import frost
        from cudnn.frost import buffers

        # Evaluate the real GPU-suite gate without importing its GPU oracles.
        path = Path(__file__).with_name("test_sdpa_bwd_d128_sm107.py")
        gate = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "_require_native")

        def skip(reason):
            raise unittest.SkipTest(reason)

        namespace = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (10, 7))),
            "pytest": SimpleNamespace(skip=skip),
        }
        exec(compile(ast.Module(body=[gate], type_ignores=[]), str(path), "exec"), namespace)
        with (
            patch.object(buffers, "cutedsl_state", return_value=(True, ("nvidia-cutlass-dsl", version))),
            patch.object(frost, "cutedsl_runtime_available", return_value=target_available, create=True),
        ):
            return namespace["_require_native"]()

    def test_row_floor_skips_gpu_positives_only_below_qualified_version(self):
        with self.assertRaisesRegex(unittest.SkipTest, r"requires nvidia-cutlass-dsl >= 4\.8\.1; found.*4\.8\.0"):
            self.require_native("4.8.0")
        self.require_native("4.8.1")

    def test_qualified_version_still_needs_target_support(self):
        with self.assertRaisesRegex(unittest.SkipTest, "target"):
            self.require_native("4.8.1", target_available=False)


class TestPreparedPlan(unittest.TestCase):
    def prepared(self, dtype=None, causal=False, scale=None, positional=True, workspace_delta=0):
        import cutlass
        from cudnn.frost import device
        from cudnn.sdpa.bwd import prepared_sm107_d128 as prepared

        helper = TestAdapter()
        api = helper.api(dtype=dtype, is_causal=causal, scale_softmax=scale)
        helper.support(api)
        frames = []
        entry = lambda *args: frames.append(args)
        compiler = Mock(return_value=entry)
        host_name = "cudnn.sdpa.bwd.kernels.sm107.prepared_d128_accum_host"
        required = api.scratch_workspace_bytes()
        with (
            patch.dict(sys.modules, {host_name: SimpleNamespace(compile_host=compiler)}),
            patch.object(device, "resolve_device", return_value=0),
            patch.object(device, "compute_capability", return_value=(10, 7)),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(cudnn._pybind_module, "_SdpaBwdBinder", Mock()),
            patch.object(prepared, "positional_entry", return_value=entry if positional else None),
            patch.object(api, "scratch_workspace_bytes", return_value=required + workspace_delta),
        ):
            spec = prepared.compile_plan(api)
        return spec, compiler.call_args.args, frames

    def test_native_host_gets_exact_plan_geometry(self):
        import cutlass

        for dtype, dsl_dtype in ((torch.bfloat16, cutlass.BFloat16), (torch.float16, cutlass.Float16)):
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    spec, args, _ = self.prepared(dtype, causal)
                    geometry, regions, actual_dtype, actual_causal, sm = args
                    self.assertEqual((actual_dtype, actual_causal, sm), (dsl_dtype, causal, 107))
                    self.assertEqual(geometry[0], ((2, 128, 3, 128), (49152, 384, 128, 1)))
                    self.assertEqual(geometry[5], ((2, 3, 128), (384, 128, 1)))
                    self.assertEqual(regions, spec.artifact.workspace_regions)
                    self.assertEqual(spec.workspace_bytes, 4 * 2 * 3 * 128 * 129 + 240)
                    self.assertEqual(len(spec.operands), 9)
                    self.assertTrue(spec.native_binding)
                    self.assertFalse(spec.scale_log2)
                    self.assertEqual(spec.name, ENGINE)
                    self.assertEqual(spec.artifact.implementation, "accumulation")

    def test_frame_matches_actual_frozen_host_signature(self):
        spec, _, frames = self.prepared()
        path = ROOT / "python/cudnn/sdpa/bwd/kernels/sm107/prepared_d128_accum_host.py"
        host = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "host")
        runtime = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
        expected = ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "do_ptr", "lse_ptr", "dq_ptr", "dk_ptr", "dv_ptr", "workspace", "scale", "stream"]
        self.assertEqual(runtime, expected)
        pointers = tuple(0x1000000 + i * 0x1000000 for i in range(9))
        frame = (*pointers, 0x40000000, 0.125, 17)
        spec.fn(*frame)
        self.assertEqual(frames, [frame])

    def test_scale_is_a_runtime_slot_not_a_compile_argument(self):
        first, first_args, _ = self.prepared(scale=0.125)
        second, second_args, _ = self.prepared(scale=0.25)
        self.assertEqual(first_args, second_args)
        self.assertEqual((first.scale, second.scale), (0.125, 0.25))

    def test_missing_positional_entry_declines(self):
        with self.assertRaisesRegex(NotImplementedError, "positional tvm-ffi"):
            self.prepared(positional=False)

    def test_workspace_contract_disagreement_is_loud(self):
        with self.assertRaisesRegex(RuntimeError, "workspace disagrees"):
            self.prepared(workspace_delta=1)
