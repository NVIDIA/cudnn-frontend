# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THD fold host bounds and cross-compilation; not native GPU evidence."""

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def test_impossible_flat_work_declines_before_launch():
    from cudnn.sdpa.bwd.kernels.bprop_thd_reduce import _reduce_launch_grid

    assert _reduce_launch_grid(3, 65535, 2) == ((3, 65535, 2), False)
    assert _reduce_launch_grid(3, 65536, 2) == ((393216, 1, 1), True)
    assert _reduce_launch_grid(3, 2, 65536) == ((393216, 1, 1), True)
    with pytest.raises(ValueError, match=r"grid.x limit"):
        _reduce_launch_grid(1, 65536, 65536)


@pytest.mark.parametrize("arch", ["sm_80", "sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_thd_fold_flat_grid_keeps_runtime_length_abi(arch, dtype):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_thd_reduce import dkv_reduce_thd_origins_host

    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"Installed DSL cannot compile {arch}")
    batch, heads = (65536, 1) if dtype == "fp16" else (2, 65536)
    scalar = cutlass.Float16 if dtype == "fp16" else cutlass.BFloat16
    partial_type = cutlass.Float32 if dtype == "fp16" else scalar

    def tensor(dim, h, dt):
        return make_fake_tensor(dt, (1, 35, h, dim), (35 * h * dim, h * dim, dim, 1), assumed_align=16)

    @cute.jit
    def entry(pk, pv, dk, dv, meta, origins, max_skv, stream):
        dkv_reduce_thd_origins_host(pk, pv, dk, dv, meta, origins, batch, max_skv, 0, 1, 64, 3, scalar, stream)

    cute.compile(
        entry,
        tensor(64, heads * 3, partial_type),
        tensor(128, heads * 3, partial_type),
        tensor(64, heads, scalar),
        tensor(128, heads, scalar),
        make_fake_tensor(cutlass.Int32, (3 * batch + 2,), (1,), assumed_align=16),
        make_fake_tensor(cutlass.Int64, (2, batch), (batch, 1), assumed_align=16),
        cutlass.Int32(35),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )
