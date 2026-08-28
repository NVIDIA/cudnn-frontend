# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU semantic tests for the bit shifts used by HSTU R2P masking."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import pytest
import torch

try:
    import cutlass.cute as cute
    from cutlass import Int32, Uint32
    from cutlass.cute.runtime import from_dlpack
except Exception as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_attention._kernels import utils

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]


class _ShiftHelpersKernel:
    @cute.jit
    def __call__(
        self,
        mShifts: cute.Tensor,
        mOutput: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mShifts, mOutput).launch(
            grid=(1, 1, 1),
            block=(mShifts.shape[0], 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mShifts: cute.Tensor, mOutput: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]
        shift = Uint32(mShifts[tidx])
        value = Uint32(0x80000001)
        mOutput[tidx, 0] = Int32(utils.shr_u32(value, shift))
        mOutput[tidx, 1] = Int32(utils.shl_b32(value, shift))


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_shift_helpers_have_ptx_clamp_at_word_width_semantics():
    shifts = torch.tensor((0, 1, 31, 32, 33), dtype=torch.int32, device="cuda")
    output = torch.empty((shifts.numel(), 2), dtype=torch.int32, device="cuda")
    cute_shifts = from_dlpack(shifts, assumed_align=16)
    cute_output = from_dlpack(output, assumed_align=16)

    compiled = cute.compile(
        _ShiftHelpersKernel(),
        cute_shifts,
        cute_output,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
    compiled(
        shifts,
        output,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    expected = torch.tensor(
        (
            (-2147483647, -2147483647),
            (1073741824, 2),
            (1, -2147483648),
            (0, 0),
            (0, 0),
        ),
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
