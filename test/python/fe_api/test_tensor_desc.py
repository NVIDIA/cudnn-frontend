# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compatibility tests for shared and Torch tensor descriptors."""

import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


try:
    import torch

    from cudnn import TensorDesc as SharedTensorDesc, data_type
    from cudnn.api_base import APIBase, TensorDesc as TorchTensorDesc
except ImportError as error:
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"Optional FE dependencies unavailable: {_IMPORT_ERROR}")
class TensorDescTest(unittest.TestCase):
    def _assert_torch_metadata(self, desc, *, name="sample", init_value=None):
        self.assertIsInstance(desc, SharedTensorDesc)
        self.assertEqual(desc.dtype, torch.bfloat16)
        self.assertEqual(desc.device, torch.device("cpu"))
        self.assertFalse(desc.interpret_uint8_as_fp4x2)
        self.assertEqual(desc.name, name)
        self.assertEqual(desc.init_value, init_value)
        self.assertEqual(desc.ndim, len(desc.shape))
        self.assertEqual(desc.cudnn_dtype, data_type.BFLOAT16)

    def test_torch_descriptor_preserves_constructor_and_layout_operations(self):
        desc = TorchTensorDesc(
            torch.bfloat16,
            (2, 3, 4),
            (12, 4, 1),
            (2, 1, 0),
            "cpu",
            False,
            "sample",
        )
        self._assert_torch_metadata(desc)
        self.assertEqual(desc.size(), torch.Size((2, 3, 4)))

        transformed = (
            desc.transpose(0, 1),
            desc.unsqueeze(-1),
            desc.unsqueeze(-1).squeeze(-1),
            desc.view(6, 4),
            desc.as_strided((2, 3, 4), (12, 4, 1)),
            desc.transpose(0, 1).contiguous(),
        )
        for result in transformed:
            with self.subTest(shape=result.shape, stride=result.stride):
                self._assert_torch_metadata(result)

    def test_packed_uint8_maps_to_canonical_fp4(self):
        desc = TorchTensorDesc(
            dtype=torch.uint8,
            shape=(16,),
            stride=(1,),
            stride_order=(0,),
            device="cpu",
            interpret_uint8_as_fp4x2=True,
            name="packed_fp4",
        )
        self.assertEqual(desc.cudnn_dtype, data_type.FP4_E2M1)

    def test_layout_operations_preserve_initial_value(self):
        desc = TorchTensorDesc(
            dtype=torch.float32,
            shape=(2, 3, 4),
            stride=(12, 4, 1),
            stride_order=(2, 1, 0),
            device="cpu",
            name="workspace",
            init_value=float("-inf"),
        )

        transformed = (
            desc.permute(1, 0, 2),
            desc.transpose(0, 1),
            desc.unsqueeze(-1),
            desc.unsqueeze(-1).squeeze(-1),
            desc.view(6, 4),
            desc.as_strided((2, 3, 4), (12, 4, 1)),
            desc.transpose(0, 1).contiguous(),
        )
        for result in transformed:
            with self.subTest(shape=result.shape, stride=result.stride):
                self.assertEqual(result.init_value, float("-inf"))

    def test_api_base_converts_torch_tensor_metadata(self):
        tensor = torch.empty_strided((2, 3, 4), (12, 1, 3), dtype=torch.bfloat16)
        desc = APIBase._to_tensor_desc(tensor, "sample")

        self._assert_torch_metadata(desc)
        self.assertEqual(desc.shape, (2, 3, 4))
        self.assertEqual(desc.stride, (12, 1, 3))
        self.assertEqual(desc.stride_order, (1, 2, 0))

    def test_api_base_checks_torch_tensor_signature(self):
        class Adapter(APIBase):
            def check_support(self):
                return True

            def compile(self):
                pass

            def execute(self, *args, **kwargs):
                pass

        adapter = Adapter()
        sample = torch.empty_strided((2, 3), (3, 1), dtype=torch.bfloat16)
        expected = TorchTensorDesc.from_tensor(sample, "sample")
        adapter._check_tensor_signature(sample, expected)

        cases = (
            (torch.empty((1, 3), dtype=torch.bfloat16), "sample tensor shape mismatch"),
            (torch.empty_strided((2, 3), (1, 2), dtype=torch.bfloat16), "sample tensor stride mismatch"),
            (torch.empty((2, 3), dtype=torch.float32), "sample dtype mismatch"),
        )
        for tensor, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter._check_tensor_signature(tensor, expected)

    def test_torch_descriptor_constructs_from_tensor_metadata(self):
        tensor = torch.empty_strided((2, 3, 4), (12, 1, 3), dtype=torch.uint8)
        desc = TorchTensorDesc.from_tensor(
            tensor,
            "logical_fp4",
            shape=(2, 3, 8),
            stride=(12, 1, 3),
            interpret_uint8_as_fp4x2=True,
        )

        self.assertEqual(desc.dtype, torch.uint8)
        self.assertEqual(desc.shape, (2, 3, 8))
        self.assertEqual(desc.stride, (12, 1, 3))
        self.assertEqual(desc.stride_order, (1, 2, 0))
        self.assertEqual(desc.device, tensor.device)
        self.assertEqual(desc.name, "logical_fp4")
        self.assertTrue(desc.interpret_uint8_as_fp4x2)
        self.assertEqual(desc.cudnn_dtype, data_type.FP4_E2M1)
        self.assertEqual(
            APIBase._to_tensor_desc(
                tensor,
                "logical_fp4",
                shape=(2, 3, 8),
                stride=(12, 1, 3),
                interpret_uint8_as_fp4x2=True,
            ),
            desc,
        )

    def test_torch_descriptor_derives_and_materializes_compact_output(self):
        source = TorchTensorDesc.from_tensor(
            torch.empty((4, 8), dtype=torch.bfloat16),
            "input",
        )
        output = source.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=(3, 5),
            stride_order=(0, 1),
            name="output",
            init_value=float("-inf"),
        )

        self.assertIsInstance(output, TorchTensorDesc)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device, source.device)
        self.assertEqual(output.shape, (3, 5))
        self.assertEqual(output.stride, (1, 3))
        self.assertEqual(output.stride_order, (0, 1))
        self.assertEqual(output.name, "output")
        self.assertEqual(output.init_value, float("-inf"))
        self.assertFalse(output.interpret_uint8_as_fp4x2)

        tensor = output.materialize()
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), output.shape)
        self.assertEqual(tuple(tensor.stride()), output.stride)
        self.assertTrue(torch.isneginf(tensor).all())

    def test_torch_descriptor_materializes_false_and_zero_initializers(self):
        source = TorchTensorDesc.from_tensor(torch.empty((1,), dtype=torch.float32))
        for init_value in (False, 0):
            with self.subTest(init_value=init_value):
                output = source.compact_like(
                    cudnn_dtype=data_type.FLOAT,
                    shape=(4,),
                    init_value=init_value,
                )
                self.assertTrue(torch.equal(output.materialize(), torch.zeros(4)))

    def test_torch_descriptor_rejects_materializing_logical_packed_storage(self):
        packed = TorchTensorDesc.from_tensor(
            torch.empty((4,), dtype=torch.uint8),
            interpret_uint8_as_fp4x2=True,
        )

        with self.assertRaisesRegex(ValueError, "logical FP4"):
            packed.materialize()

        unpacked = TorchTensorDesc(
            dtype=torch.float32,
            shape=(4,),
            stride=(1,),
            stride_order=(0,),
            device="cpu",
            interpret_uint8_as_fp4x2=True,
            init_value=1,
        )
        self.assertTrue(torch.equal(unpacked.materialize(), torch.ones(4)))

    def test_api_base_materializes_a_canonical_descriptor(self):
        desc = SharedTensorDesc(
            dtype=data_type.FLOAT,
            shape=(2, 3),
            stride=(1, 2),
            stride_order=(0, 1),
            name="output",
            init_value=float("-inf"),
        )

        tensor = APIBase._materialize_tensor_desc(desc, device=torch.device("cpu"))

        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tuple(tensor.shape), desc.shape)
        self.assertEqual(tuple(tensor.stride()), desc.stride)
        self.assertTrue(torch.isneginf(tensor).all())

    def test_api_base_initializes_on_the_requested_cuda_stream(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        from cuda.bindings import driver as cuda

        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        desc = SharedTensorDesc(
            dtype=data_type.FLOAT,
            shape=(8,),
            stride=(1,),
            stride_order=(0,),
            init_value=7,
        )

        tensor = APIBase._materialize_tensor_desc(
            desc,
            device=device,
            stream=cuda.CUstream(stream.cuda_stream),
        )
        stream.synchronize()

        self.assertTrue(torch.equal(tensor.cpu(), torch.full((8,), 7.0)))

        torch_desc = TorchTensorDesc.from_tensor(torch.empty((1,), dtype=torch.float32, device=device)).compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=(8,),
            init_value=9,
        )
        tensor = torch_desc.materialize(stream=cuda.CUstream(stream.cuda_stream))
        stream.synchronize()

        self.assertTrue(torch.equal(tensor.cpu(), torch.full((8,), 9.0)))


if __name__ == "__main__":
    unittest.main()
