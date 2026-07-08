# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for the JAX API base."""

from enum import Enum, auto
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_PACKAGE = "cudnn_jax_api_base_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()


class _ArrayMetadata:
    def __init__(self, shape, dtype, label=None):
        self.shape = shape
        self.dtype = dtype
        self.label = label


class _TensorSpec:
    def __init__(
        self,
        *,
        layout=None,
        mode=None,
        divisibility=None,
        ptr_assumed_align=None,
    ):
        self.layout = layout
        self.mode = mode
        self.divisibility = divisibility
        self.ptr_assumed_align = ptr_assumed_align


class _ShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _Device:
    platform = "gpu"

    def __init__(self, device_id, compute_capability):
        self.id = device_id
        self.compute_capability = compute_capability

    def __str__(self):
        return f"CudaDevice(id={self.id})"


class JaxApiBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        internal_name = f"{_PACKAGE}._jax"
        internal = types.ModuleType(internal_name)
        internal.__path__ = [str(_CUDNN_ROOT / "_jax")]
        internal.__package__ = internal_name
        internal.__spec__ = ModuleSpec(internal_name, loader=None, is_package=True)
        sys.modules[internal_name] = internal

        datatypes_name = f"{internal_name}.datatypes"
        datatypes = types.ModuleType(datatypes_name)
        datatypes.jax_to_cudnn_dtype = lambda dtype: {
            "bfloat16": _DataType.BFLOAT16,
            "float32": _DataType.FLOAT,
        }.get(dtype, _DataType.NOT_SET)
        datatypes.cudnn_to_jax_dtype = lambda dtype: {
            _DataType.BFLOAT16: "bfloat16",
            _DataType.FLOAT: "float32",
        }[dtype]
        sys.modules[datatypes_name] = datatypes

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": None,
                    "jax.numpy": None,
                    "cutlass": None,
                    "cutlass.jax": None,
                },
            ):
                cls.tensor_module = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")
                cls.module = importlib.import_module(f"{internal_name}.api_base")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def test_checks_all_local_jax_gpu_compute_capabilities(self):
        jax = types.ModuleType("jax")

        def local_devices(*, backend):
            self.assertEqual(backend, "gpu")
            return (_Device(0, "10.0"), _Device(1, "12.0"))

        jax.local_devices = local_devices
        with mock.patch.dict(sys.modules, {"jax": jax}):
            self.module.JaxApiBase._check_device_compatibility(
                minimum_compute_capability=100,
                operation_name="TestOp",
            )

    def test_rejects_an_incompatible_local_jax_gpu(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: (_Device(0, "10.0"), _Device(1, "9.0"))

        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"TestOp requires SM100\+, found CudaDevice\(id=1\) \(SM90\)"):
                self.module.JaxApiBase._check_device_compatibility(
                    minimum_compute_capability=100,
                    operation_name="TestOp",
                )

    def test_requires_a_local_jax_gpu(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: ()

        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"no local JAX GPU"):
                self.module.JaxApiBase._check_device_compatibility(
                    minimum_compute_capability=100,
                    operation_name="TestOp",
                )

    def test_reports_device_discovery_and_capability_failures(self):
        jax = types.ModuleType("jax")

        def fail_discovery(*, backend):
            raise RuntimeError("GPU backend unavailable")

        jax.local_devices = fail_discovery
        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"could not discover a local GPU"):
                self.module.JaxApiBase._check_device_compatibility(
                    minimum_compute_capability=100,
                    operation_name="TestOp",
                )

        jax.local_devices = lambda *, backend: (_Device(0, "unknown"),)
        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"invalid compute capability 'unknown'"):
                self.module.JaxApiBase._check_device_compatibility(
                    minimum_compute_capability=100,
                    operation_name="TestOp",
                )

    def test_resolves_exact_local_target_for_multi_arch_operation(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: (_Device(0, "10.3"),)

        with mock.patch.dict(sys.modules, {"jax": jax}):
            target = self.module.JaxApiBase._resolve_compute_capability(
                None,
                (90, 100, 103, 107),
                "TestOp",
            )

        self.assertEqual(target, 103)

    def test_rejects_heterogeneous_implicit_targets(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: (_Device(0, "10.0"), _Device(1, "10.3"))

        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"heterogeneous targets"):
                self.module.JaxApiBase._resolve_compute_capability(None, (90, 100, 103, 107), "TestOp")

    def test_explicit_target_must_match_local_devices(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: (_Device(0, "10.0"),)

        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"targets SM90.*found CudaDevice\(id=0\) \(SM100\)"):
                self.module.JaxApiBase._resolve_compute_capability(90, (90, 100, 103, 107), "TestOp")
            self.assertEqual(
                self.module.JaxApiBase._resolve_compute_capability(100, (90, 100, 103, 107), "TestOp"),
                100,
            )

    def test_rejects_unknown_explicit_target_before_lowering(self):
        with self.assertRaisesRegex(ValueError, r"no kernel for SM101.*supported targets"):
            self.module.JaxApiBase._resolve_compute_capability(
                101,
                (90, 100, 103, 107),
                "TestOp",
            )

    def test_implicit_target_requires_a_local_device(self):
        jax = types.ModuleType("jax")
        jax.local_devices = lambda *, backend: ()

        with mock.patch.dict(sys.modules, {"jax": jax}):
            with self.assertRaisesRegex(RuntimeError, r"no local JAX GPU"):
                self.module.JaxApiBase._resolve_compute_capability(None, (90, 100), "TestOp")

    def test_caches_raw_max_active_clusters_per_instance_and_cluster_size(self):
        query_calls = []

        class HardwareInfo:
            def get_max_active_clusters(self, cluster_size):
                query_calls.append(cluster_size)
                return {2: 8, 4: 12}[cluster_size]

        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_utils = types.ModuleType("cutlass.utils")
        cutlass_utils.HardwareInfo = HardwareInfo
        cutlass.utils = cutlass_utils

        class Adapter(self.module.JaxApiBase):
            def check_support(self):
                return True

            def __call__(self):
                return self.check_support()

        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                "cutlass.utils": cutlass_utils,
            },
        ):
            api = Adapter()
            self.assertEqual(api._get_max_active_clusters(4, overlap_margin=2), 10)
            self.assertEqual(api._get_max_active_clusters(4, overlap_margin=3), 9)
            self.assertEqual(api._get_max_active_clusters(2, overlap_margin=1), 7)
            with self.assertRaisesRegex(ValueError, "max_active_clusters must be positive"):
                api._get_max_active_clusters(4, overlap_margin=12)

            other_api = Adapter()
            self.assertEqual(other_api._get_max_active_clusters(4), 12)

        self.assertEqual(query_calls, [4, 2, 4])

    def test_converts_array_metadata_to_shared_tensor_desc(self):
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
        )

        self.assertIsInstance(desc, self.tensor_module.TensorDesc)
        self.assertEqual(desc.dtype, "bfloat16")
        self.assertEqual(desc.shape, (2, 3, 4))
        self.assertEqual(desc.stride, (12, 4, 1))
        self.assertEqual(desc.stride_order, (2, 1, 0))
        self.assertEqual(desc.name, "sample")
        self.assertIsNone(desc.init_value)
        self.assertEqual(desc.cudnn_dtype, _DataType.BFLOAT16)

    def test_converts_public_array_axes_to_canonical_descriptor_axes(self):
        mode = (2, 0, 1)
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
            mode=mode,
        )

        self.assertEqual(desc.shape, (4, 2, 3))
        self.assertEqual(desc.stride, (1, 12, 4))
        self.assertEqual(desc.stride_order, (0, 2, 1))
        self.assertEqual(desc.mode, mode)
        self.assertEqual(desc.array_shape, (2, 3, 4))

    def test_constructs_descriptor_directly_from_shape(self):
        desc = self.module.JaxTensorDesc.from_shape(
            (2, 3, 4),
            "bfloat16",
            name="sample",
            mode=(2, 0, 1),
        )

        self.assertEqual(desc.shape, (4, 2, 3))
        self.assertEqual(desc.array_shape, (2, 3, 4))
        self.assertEqual(desc.mode, (2, 0, 1))
        self.assertEqual(desc.cudnn_dtype, _DataType.BFLOAT16)

    def test_converts_public_physical_layout_to_canonical_descriptor_axes(self):
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
            mode=(1, 2, 0),
            public_stride_order=(0, 2, 1),
        )

        self.assertEqual(desc.shape, (3, 4, 2))
        self.assertEqual(desc.stride, (8, 2, 1))
        self.assertEqual(desc.stride_order, (2, 1, 0))

    def test_rejects_invalid_public_stride_orders(self):
        value = _ArrayMetadata((2, 3, 4), "bfloat16")

        with self.assertRaisesRegex(ValueError, "public_stride_order rank mismatch"):
            self.module.JaxApiBase._to_tensor_desc(value, "sample", public_stride_order=(1, 0))
        with self.assertRaisesRegex(ValueError, "public_stride_order must be a permutation"):
            self.module.JaxApiBase._to_tensor_desc(value, "sample", public_stride_order=(2, 2, 0))
        with self.assertRaisesRegex(TypeError, "public_stride_order entries must be integers"):
            self.module.JaxApiBase._to_tensor_desc(value, "sample", public_stride_order=(2, True, 0))

    def test_jax_descriptor_derives_compact_output_metadata(self):
        source = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3), "bfloat16"),
            "input",
        )

        output = source.compact_like(
            cudnn_dtype=_DataType.FLOAT,
            shape=(5, 7),
            stride_order=(0, 1),
            name="output",
            init_value=-2.0,
        )

        self.assertIsInstance(output, self.module.JaxTensorDesc)
        self.assertEqual(output.dtype, "float32")
        self.assertEqual(output.cudnn_dtype, _DataType.FLOAT)
        self.assertEqual(output.shape, (5, 7))
        self.assertEqual(output.stride, (1, 5))
        self.assertEqual(output.stride_order, (0, 1))
        self.assertEqual(output.name, "output")
        self.assertEqual(output.init_value, -2.0)
        self.assertEqual(output.mode, (0, 1))

    def test_checks_invocation_signature(self):
        desc = self.module.JaxApiBase._to_tensor_desc(_ArrayMetadata((2, 3), "bfloat16"), "sample")
        self.module.JaxApiBase._check_tensor_signature(_ArrayMetadata((2, 3), "bfloat16"), desc)

        with self.assertRaisesRegex(ValueError, "sample tensor shape mismatch"):
            self.module.JaxApiBase._check_tensor_signature(_ArrayMetadata((1, 3), "bfloat16"), desc)
        with self.assertRaisesRegex(ValueError, "sample tensor dtype mismatch"):
            self.module.JaxApiBase._check_tensor_signature(_ArrayMetadata((2, 3), "float32"), desc)

    def test_checks_public_invocation_against_canonical_descriptor(self):
        mode = (2, 0, 1)
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
            mode=mode,
        )

        self.module.JaxApiBase._check_tensor_signature(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            desc,
        )
        with self.assertRaisesRegex(ValueError, "sample tensor shape mismatch"):
            self.module.JaxApiBase._check_tensor_signature(
                _ArrayMetadata((1, 3, 4), "bfloat16"),
                desc,
            )

    def test_builds_public_tensor_spec_from_canonical_metadata(self):
        mode = (2, 0, 1)
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
            mode=mode,
            divisibility=(8, 2, 4),
            ptr_assumed_align=64,
        )
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.TensorSpec = _TensorSpec
        cutlass.jax = cutlass_jax

        with mock.patch.dict(sys.modules, {"cutlass": cutlass, "cutlass.jax": cutlass_jax}):
            spec = self.module.JaxApiBase._to_tensor_spec(desc)

        self.assertEqual(spec.layout, (2, 1, 0))
        self.assertEqual(spec.mode, mode)
        self.assertEqual(spec.divisibility, (2, 4, 8))
        self.assertEqual(spec.ptr_assumed_align, 64)

    def test_tensor_spec_preserves_explicit_public_physical_layout(self):
        mode = (1, 2, 0)
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3, 4), "bfloat16"),
            "sample",
            mode=mode,
            public_stride_order=(0, 2, 1),
        )
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.TensorSpec = _TensorSpec
        cutlass.jax = cutlass_jax

        with mock.patch.dict(sys.modules, {"cutlass": cutlass, "cutlass.jax": cutlass_jax}):
            spec = self.module.JaxApiBase._to_tensor_spec(desc)

        self.assertEqual(spec.layout, (0, 2, 1))
        self.assertEqual(spec.mode, mode)

    def test_default_descriptor_uses_cutlass_default_tensor_spec(self):
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3), "bfloat16"),
            "sample",
        )
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.TensorSpec = _TensorSpec
        cutlass.jax = cutlass_jax

        with mock.patch.dict(sys.modules, {"cutlass": cutlass, "cutlass.jax": cutlass_jax}):
            self.assertIsNone(self.module.JaxApiBase._to_tensor_spec(desc))

    def test_tensor_spec_rejects_divisibility_with_the_wrong_rank(self):
        desc = self.module.JaxApiBase._to_tensor_desc(
            _ArrayMetadata((2, 3), "bfloat16"),
            "sample",
        )

        with self.assertRaisesRegex(ValueError, "divisibility rank mismatch"):
            self.module.JaxTensorDesc.from_array(
                _ArrayMetadata((2, 3), "bfloat16"),
                name="sample",
                divisibility=(2,),
            )

    def test_builds_shape_dtype_struct_in_array_axis_order(self):
        desc = self.module.JaxTensorDesc.from_shape(
            (2, 3, 4),
            "bfloat16",
            mode=(2, 0, 1),
        )
        jax = types.ModuleType("jax")
        jax.ShapeDtypeStruct = _ShapeDtypeStruct

        with mock.patch.dict(sys.modules, {"jax": jax}):
            output = self.module.JaxApiBase._to_shape_dtype_struct(desc)

        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(output.dtype, "bfloat16")

    def test_calls_explicit_launcher_with_initialized_and_fresh_outputs(self):
        seen = {}

        output_descs = (
            self.module.JaxTensorDesc.from_shape(
                (2,),
                "float32",
                name="output",
            ),
            self.module.JaxTensorDesc.from_shape(
                (2, 3, 4),
                "float32",
                name="amax",
                init_value=float("-inf"),
                mode=(2, 0, 1),
            ),
        )
        workspace_descs = (
            self.module.JaxTensorDesc.from_shape(
                (3,),
                "float32",
                name="counter",
                init_value=0,
            ),
            self.module.JaxTensorDesc.from_shape(
                (5,),
                "bfloat16",
                name="scratch",
            ),
        )

        def launch(stream, *buffers):
            seen["launch_args"] = (stream, *buffers)

        class Adapter(self.module.JaxApiBase):
            def check_support(self):
                return True

            def __call__(self, value):
                return self._call_kernel(
                    (value,),
                    launch=launch,
                    input_descs=(input_desc,),
                    output_descs=output_descs,
                    workspace_descs=workspace_descs,
                )

        full_calls = []
        jax = types.ModuleType("jax")
        jax.__path__ = []
        jax.ShapeDtypeStruct = _ShapeDtypeStruct
        jnp = types.ModuleType("jax.numpy")

        def full(shape, value, dtype):
            result = _ArrayMetadata(tuple(shape), dtype, label=f"full({value})")
            full_calls.append((tuple(shape), value, dtype, result))
            return result

        jnp.full = full
        jax.numpy = jnp

        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_cute = types.ModuleType("cutlass.cute")

        def cute_jit(*, preprocess):
            self.assertFalse(preprocess)

            def decorate(function):
                seen["cute_launcher"] = function
                return function

            return decorate

        cutlass_cute.jit = cute_jit
        cutlass_jax = types.ModuleType("cutlass.jax")
        cutlass_jax.TensorSpec = _TensorSpec

        def cutlass_call(launcher, **options):
            seen["cutlass_launcher"] = launcher
            seen["call_options"] = options

            def invoke(*args):
                aliases = options["input_output_aliases"]
                result_metadata = options["output_shape_dtype"]
                results = [None] * len(result_metadata)
                for input_index, result_index in aliases.items():
                    results[result_index] = args[input_index]
                for result_index, metadata in enumerate(result_metadata):
                    if results[result_index] is None:
                        results[result_index] = _ArrayMetadata(
                            metadata.shape,
                            metadata.dtype,
                            label=f"allocated({result_index})",
                        )

                unaliased_results = [results[index] for index in range(len(results)) if index not in aliases.values()]
                launcher("stream", *args, *unaliased_results)
                return tuple(results)

            return invoke

        cutlass_jax.cutlass_call = cutlass_call
        cutlass.cute = cutlass_cute
        cutlass.jax = cutlass_jax

        input_value = _ArrayMetadata((2,), "float32", label="input")
        input_desc = self.module.JaxTensorDesc.from_array(
            input_value,
            name="input",
        )
        api = Adapter()
        with mock.patch.dict(
            sys.modules,
            {
                "jax": jax,
                "jax.numpy": jnp,
                "cutlass": cutlass,
                "cutlass.cute": cutlass_cute,
                "cutlass.jax": cutlass_jax,
            },
        ):
            results = api._call_kernel(
                (input_value,),
                launch=launch,
                input_descs=(input_desc,),
                output_descs=output_descs,
                workspace_descs=workspace_descs,
            )
            with self.assertRaisesRegex(TypeError, "input #0 must have shape and dtype metadata"):
                api._call_kernel(
                    ([input_value],),
                    launch=launch,
                    input_descs=(input_desc,),
                    output_descs=output_descs,
                    workspace_descs=workspace_descs,
                )
            with self.assertRaisesRegex(ValueError, "Expected 1 input descriptors, got 0"):
                api._call_kernel(
                    (input_value,),
                    launch=launch,
                    input_descs=(),
                    output_descs=output_descs,
                )
            with self.assertRaisesRegex(TypeError, "input_descs must contain JaxTensorDesc"):
                api._call_kernel(
                    (input_value,),
                    launch=launch,
                    input_descs=(
                        self.tensor_module.make_compact_tensor_desc(
                            dtype=_DataType.FLOAT,
                            shape=(2,),
                        ),
                    ),
                    output_descs=output_descs,
                )
            with self.assertRaisesRegex(TypeError, "output_descs must contain JaxTensorDesc"):
                api._call_kernel(
                    (input_value,),
                    launch=launch,
                    input_descs=(input_desc,),
                    output_descs=(
                        self.tensor_module.make_compact_tensor_desc(
                            dtype=_DataType.FLOAT,
                            shape=(2,),
                        ),
                    ),
                    workspace_descs=workspace_descs,
                )
            with self.assertRaisesRegex(TypeError, "launch must be callable"):
                api._call_kernel(
                    (input_value,),
                    launch=None,
                    input_descs=(input_desc,),
                    output_descs=output_descs,
                )

        self.assertEqual(
            [(shape, value, dtype) for shape, value, dtype, _ in full_calls],
            [
                ((2, 3, 4), float("-inf"), "float32"),
                ((3,), 0, "float32"),
            ],
        )
        self.assertEqual(
            seen["call_options"]["input_output_aliases"],
            {1: 1, 2: 2},
        )
        self.assertEqual(len(seen["call_options"]["input_spec"]), 3)
        self.assertIsNone(seen["call_options"]["input_spec"][0])
        self.assertIs(
            seen["call_options"]["input_spec"][1],
            seen["call_options"]["output_spec"][1],
        )
        self.assertIs(
            seen["call_options"]["input_spec"][2],
            seen["call_options"]["output_spec"][2],
        )

        stream, launch_input, output, amax, counter, scratch = seen["launch_args"]
        self.assertEqual(stream, "stream")
        self.assertIs(seen["cute_launcher"], seen["cutlass_launcher"])
        self.assertIs(launch_input, input_value)
        self.assertEqual(output.label, "allocated(0)")
        self.assertIs(amax, full_calls[0][3])
        self.assertIs(counter, full_calls[1][3])
        self.assertEqual(scratch.label, "allocated(3)")
        self.assertEqual([result.label for result in results], ["allocated(0)", "full(-inf)"])
        self.assertNotIn(full_calls[0][3], vars(api).values())
        self.assertNotIn(full_calls[1][3], vars(api).values())

    def test_returns_callable_without_owning_jit_policy(self):
        base = self.module.JaxApiBase

        class Adapter(base):
            def check_support(self):
                return True

            def __call__(self):
                return self.check_support()

        api = Adapter()
        api.option = 1
        self.assertIs(api.get_jax_callable(), api)
        self.assertTrue(api())
        api.option = 2
        self.assertEqual(api.option, 2)


if __name__ == "__main__":
    unittest.main()
