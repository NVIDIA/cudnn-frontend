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


if __name__ == "__main__":
    unittest.main()
