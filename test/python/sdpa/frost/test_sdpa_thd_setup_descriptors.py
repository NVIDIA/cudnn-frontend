# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""THD setup must preserve descriptor bytes when cumulative inputs bypass metadata."""

import pytest
import torch

from frost_test_utils import _dsl_installed, requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_blackwell, requires_dsl]

if _dsl_installed():
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.experimental.cuda import tensor_map as tmap

    from cudnn.frost.tile_dsl.thd import THD_WORK_META_WORDS, TENSOR_MAP_QWORDS
    from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel

    @cute.jit
    def _setup_host(
        o,
        k,
        v,
        descs,
        meta,
        qlens,
        klens,
        batch,
        flags,
        stream,
        clamp: cutlass.Constexpr[bool],
        small: cutlass.Constexpr[bool],
        threads: cutlass.Constexpr[int],
        rank_kv: cutlass.Constexpr[bool],
    ):
        od = tmap.create_tensor_map_tiled_from_view(o, box_dims=(1, 16, 1, 64), stride_order=(3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.s128b)
        kd = tmap.create_tensor_map_tiled_from_view(k, box_dims=(1, 16, 1, 64), stride_order=(3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.s128b)
        vd = tmap.create_tensor_map_tiled_from_view(v, box_dims=(1, 16, 1, 64), stride_order=(3, 2, 1, 0), swizzle=tmap.TensorMapSwizzle.s128b)
        build_thd_meta_o_descs_kernel(
            o,
            od,
            kd,
            vd,
            descs,
            meta,
            qlens,
            klens,
            flags,
            cutlass.Int32(3),
            batch,
            cutlass.Int64(o.stride[1]),
            cutlass.Int32(16),
            cutlass.Int32(7),
            clamp_kv=clamp,
            splits=3,
            rank_by_kv=rank_kv,
            small_batch_meta=small,
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)


@pytest.fixture(scope="module")
def setup_runner():
    compiled = {}
    o = torch.empty_strided((1, 16384, 3, 64), (16384 * 256, 256, 64, 1), dtype=torch.bfloat16, device="cuda")
    k = torch.empty((1, 16384, 2, 64), dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    tensors = tuple(from_dlpack(x, assumed_align=16) for x in (o, k, v))

    def run(meta, desc, q, k, batch, flags, clamp, small, threads, rank_kv):
        dynamic = lambda x: from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=0)
        args = (
            *tensors,
            dynamic(desc),
            dynamic(meta),
            dynamic(q),
            dynamic(k),
            cutlass.Int32(batch),
            cutlass.Int32(flags),
            cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        key = (clamp, small, threads, rank_kv)
        if key not in compiled:
            compiled[key] = cute.compile(_setup_host, *args, clamp, small, threads, rank_kv)
        compiled[key](*args)

    return run


_CASES = [(b, f, c, 256, False) for b in (1, 2, 8, 32, 33) for f in range(4) for c in (False, True)]
_CASES += [(b, 3, c, t, False) for b in (2, 32) for c in (False, True) for t in (64, 128)]
_CASES += [(b, 3, c, 256, True) for b in (2, 32) for c in (False, True)]


@pytest.mark.parametrize("batch,flags,clamp,threads,rank_kv", _CASES)
def test_setup_descriptors_match_normalized_metadata(batch, flags, clamp, threads, rank_kv, setup_runner):
    # Nonzero cumulative origins must normalize before descriptor base-address
    # arithmetic. Empty requests and B=33 exercise both fast and fallback paths.
    q = (torch.arange(batch, dtype=torch.int32) * 17 + 13) % 257
    k = (torch.arange(batch, dtype=torch.int32) * 31 + 7) % 513
    if batch > 1:
        q[0], k[-1] = 0, 0

    def inputs(q, k, origin):
        cumulative = lambda x, start: torch.cat((torch.zeros(1, dtype=torch.int32), x.cumsum(0, dtype=torch.int32))) + start
        return (cumulative(q, origin) if flags & 1 else q).cuda(), (cumulative(k, origin + 14) if flags & 2 else k).cuda()

    qin, kin = inputs(q, k, 17)
    reference_meta = torch.full((THD_WORK_META_WORDS(batch),), -559038737, dtype=torch.int32, device="cuda")
    actual_meta = torch.empty_like(reference_meta)
    reference_desc = torch.full(((batch + 3) * TENSOR_MAP_QWORDS,), 0x123456789ABCDEF, dtype=torch.int64, device="cuda")
    actual_desc = torch.empty_like(reference_desc)

    def reference():
        setup_runner(reference_meta, reference_desc, qin, kin, batch, flags, clamp, False, threads, rank_kv)

    def candidate():
        setup_runner(actual_meta, actual_desc, qin, kin, batch, flags, clamp, True, threads, rank_kv)

    # The legacy B=1 path intentionally omits its unused work-prefix table.
    # Seed those two oracle words from CPU lengths so all words still compare.
    if batch == 1:
        reference_meta[-2:].copy_(torch.tensor([0, ((int(q[0]) + 15) // 16) * 3], device="cuda", dtype=torch.int32))
    reference()
    actual_meta.fill_(-559038737)
    actual_desc.fill_(0x123456789ABCDEF)
    candidate()
    torch.testing.assert_close(actual_meta, reference_meta, rtol=0, atol=0)
    torch.testing.assert_close(actual_desc, reference_desc, rtol=0, atol=0)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        candidate()
    try:
        # Reuse the captured setup with changed device lengths and a different
        # cumulative origin; no compile-time prefix or descriptor is reusable.
        q2 = q.roll(1) if batch > 1 else q + 1
        k2 = k.roll(1) if batch > 1 else k + 3
        qnew, knew = inputs(q2, k2, 41)
        qin.copy_(qnew)
        kin.copy_(knew)
        reference_meta.fill_(-559038737)
        reference_desc.fill_(0x123456789ABCDEF)
        if batch == 1:
            reference_meta[-2:].copy_(torch.tensor([0, ((int(q2[0]) + 15) // 16) * 3], device="cuda", dtype=torch.int32))
        reference()
        actual_meta.fill_(-559038737)
        actual_desc.fill_(0x123456789ABCDEF)
        captured.replay()
        torch.testing.assert_close(actual_meta, reference_meta, rtol=0, atol=0)
        torch.testing.assert_close(actual_desc, reference_desc, rtol=0, atol=0)
    finally:
        captured.reset()
