# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host grid bounds and cross-compilation, distinct from native GPU evidence."""

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


def test_flat_grid_limit_is_checked_before_launch():
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import _dot_launch_grid

    assert _dot_launch_grid(3, 65535, 2) == ((3, 65535, 2), 0)
    grid, q_blocks = _dot_launch_grid(3, 65536, 2)
    assert grid == (393216, 1, 1) and q_blocks == 3
    assert _dot_launch_grid(1, 1, 2**31 - 1) == ((2**31 - 1, 1, 1), 1)
    with pytest.raises(ValueError, match=r"grid.x limit"):
        _dot_launch_grid(1, 65536, 65536)
    assert _dot_launch_grid(1023, 32, 65536) == ((2145386496, 1, 1), 1023)
    for q_blocks in (1024, 2049):
        # The sibling THD fold missed this complete product. Delta already
        # checks all three factors, including work beyond uint32.
        with pytest.raises(ValueError, match=r"grid.x limit"):
            _dot_launch_grid(q_blocks, 32, 65536)


@pytest.mark.parametrize("arch", ["sm_80", "sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a"])
@pytest.mark.parametrize("scaled", [False, True], ids=["half", "fp8"])
def test_flat_delta_compiles_for_other_architectures(arch, scaled):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_host, dot_do_o_scaled_host

    if scaled and arch == "sm_80":
        pytest.skip("SM80 has no native FP8 conversion")
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"Installed DSL cannot compile {arch}")
    dtype = cutlass.Float8E4M3FN if scaled else cutlass.BFloat16
    shape = (2, 129, 65536, 64)
    stride = (129 * 65536 * 64, 65536 * 64, 64, 1)
    o = make_fake_tensor(dtype, shape, stride, assumed_align=16)
    delta = make_fake_tensor(cutlass.Float32, (2, 65536, 256), (65536 * 256, 256, 1), assumed_align=16)
    scale = make_fake_tensor(cutlass.Float32, (1,), (1,), assumed_align=16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    options = f"--enable-tvm-ffi --gpu-arch {arch}"
    try:
        if scaled:
            cute.compile(dot_do_o_scaled_host, o, o, delta, scale, scale, 128, 64, 64, stream, options=options)
        else:
            cute.compile(dot_do_o_host, o, o, delta, None, None, 128, 64, 64, 64, False, False, stream, options=options)
    except Exception as exc:
        # An architecture enum can exist before the bundled compiler supports
        # the target. Only its explicit capability diagnostic is a skip.
        message = str(exc)
        if "Unsupported GPU architecture" in message and arch in message and "CUDA toolkit version" in message:
            pytest.skip(f"Installed DSL compiler cannot target {arch}: {message}")
        raise
