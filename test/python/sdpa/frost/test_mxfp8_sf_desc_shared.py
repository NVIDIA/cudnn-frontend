# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.sdpa.kernels._mxfp8_sf`` -- the ONE spelling of the MXFP8 scale-factor TMA descriptors (rowwise per-tile,
columnwise D-plane-major) and of the cga2 peer split of an SF slab, lifted from the Rubin d=256 MXFP8 forward
(``sm107/prefill_d256_mxfp8.py``) where the d192x128 forward carried a second and third copy.

Pins: (1) the host-int geometry helpers (``sf_peer_split``, ``sf_tma_rows``) on the real slab sizes of the two forwards
and their refusal of a non-divisible split; (2) both forwards import the module and carry NO private copy of the closure
or of the ``_PER_PEER`` arithmetic any more (a source pin: a fourth copy would pass every numerics test); (3) the builders
TRACE -- an sm_107a trace-compile (no device) of a host that builds a rowwise and a columnwise descriptor through the
module for a dense and a THD geometry.  The byte-identity of the refactor itself is proved outside this module: the
sm_107a cubin, PTX and clean-MLIR md5s of both forward kernels before / after (4 builds, identical; md5 records retained
internally -- re-check the cubin md5s before and after any edit to the builders), plus the forwards' own Rubin accept
tests (``test_sdpa_fwd_dsl_sm107.py -k mxfp8``,
``test_sdpa_fwd_mxfp8_sm100.py::test_mxfp8_d192_d128``).
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_dsl

pytestmark = pytest.mark.L0


def _kernel_source(rel):
    import cudnn

    path = os.path.join(os.path.dirname(os.path.abspath(cudnn.__file__)), "sdpa", "fwd", "kernels", rel)
    with open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# (1) host-int geometry
# ---------------------------------------------------------------------------


def test_peer_split_of_the_forwards_slabs():
    from cudnn.sdpa.kernels._mxfp8_sf import SF_TMA_ROW_BYTES, sf_peer_split, sf_tma_rows

    assert SF_TMA_ROW_BYTES == 128
    # d256 forward: SF_SMEM_SIZE_K = TILE_N(128) * 256 // 32 = 1024 B, cga2 -> 512 B / 4 rows per peer; cga1 -> the whole slab
    assert sf_peer_split(1024, 2) == (512, 4)
    assert sf_peer_split(1024, 1) == (1024, 8)
    # d192x128 forward: K slab 128 * 256 // 32 = 1024 (K rounds 192 up to 256), V slab 128 * 128 // 32 = 512 -> 256 B / 2 rows per peer
    assert sf_peer_split(512, 2) == (256, 2)
    assert sf_tma_rows(1024) == 8 and sf_tma_rows(512) == 4
    split = sf_peer_split(1024, 2)
    assert split.bytes_per_peer == 512 and split.rows_per_peer == 4  # named fields, not just a tuple


@pytest.mark.parametrize("bad", [(1024, 3), (100, 2), (128, 4)])
def test_peer_split_refuses_a_split_that_is_not_whole_tma_rows(bad):
    from cudnn.sdpa.kernels._mxfp8_sf import sf_peer_split

    with pytest.raises(ValueError, match=r"whole .*TMA rows|does not split evenly"):
        sf_peer_split(*bad)


# ---------------------------------------------------------------------------
# (2) the forwards call the module -- no private copy left
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", ["sm107/prefill_d256_mxfp8.py", "sm107/prefill_d192_d128_mxfp8.py"])
def test_forward_kernels_take_the_sf_descriptors_and_the_peer_split_from_the_shared_module(rel):
    src = _kernel_source(rel)
    assert "from cudnn.sdpa.kernels._mxfp8_sf import" in src, f"{rel}: must import the shared SF module"
    assert "def _build_sf_desc(" not in src, f"{rel}: a private copy of the rowwise SF descriptor closure survives"
    assert "build_rowwise_sf_desc(" in src, rel
    assert not re.search(r"^\s*SF_TMA_ROW_BYTES\s*=\s*128", src, re.M), f"{rel}: a private SF_TMA_ROW_BYTES literal survives"
    assert not re.search(r"_SF_BYTES_PER_PEER\s*=\s*SF_SMEM_SIZE_\w\s*//\s*CFG\.CTA_MMA", src), f"{rel}: a hand-rolled peer split survives"
    if "d256" in rel:
        assert "build_columnwise_sf_desc(" in src, "the d256 forward's V SF is the columnwise (D-plane-major) descriptor"
        assert "_v_plane_stride_16" not in src, "the columnwise stride arithmetic must live in the shared module"


# ---------------------------------------------------------------------------
# (3) the builders trace for sm_107a (no device)
# ---------------------------------------------------------------------------

_TRACE_PROBE = textwrap.dedent("""
    import os, sys
    arch = sys.argv[1]
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.experimental import primitives as nvvm
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.experimental.cuda import tensor_map as tmap
    from cudnn.sdpa.kernels._mxfp8_sf import SF_TMA_ROW_BYTES, build_columnwise_sf_desc, build_rowwise_sf_desc, sf_tma_rows

    SF_SMEM_K = 1024  # TILE_N 128 x (256 // 32) bytes
    SF_BYTES_PER_BLOCK = 512

    @cute.kernel
    def sink(tma_q_sf_desc: cutlass.GridConstant[tmap.TensorMap], tma_v_sf_desc: cutlass.GridConstant[tmap.TensorMap]):
        # the descriptors are grid constants, as in the forwards; the kernel prefetches them and nothing more
        nvvm.prefetch_tensormap(tma_q_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_sf_desc.get_ptr())

    @cute.jit
    def host(sf_q: cute.Tensor, sf_v: cute.Tensor, sq_tiles: cutlass.Int32, stream: cuda.CUstream, thd: cutlass.Constexpr[bool]):
        B, KH, QH = 2, 4, 8
        b_sf = 1 if thd else B
        q_desc = build_rowwise_sf_desc(sf_q, num_tiles=sq_tiles, sf_smem_size=SF_SMEM_K, num_rows_box=sf_tma_rows(SF_SMEM_K), num_heads=QH, num_batches=b_sf)
        v_desc = build_columnwise_sf_desc(sf_v, num_tiles=sq_tiles, num_heads=KH, num_batches=b_sf, num_planes=2, planes_per_box=1, sf_bytes_per_block=SF_BYTES_PER_BLOCK, sf_smem_size=SF_SMEM_K, thd_varlen=thd)
        sink(q_desc, v_desc).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    sf_q = make_fake_tensor(dtype=cutlass.Uint8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    sf_v = make_fake_tensor(dtype=cutlass.Uint8, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
    for thd in (False, True):
        cute.compile(host, sf_q, sf_v, cutlass.Int32(4), make_fake_stream(use_tvm_ffi_env_stream=False), thd, options=f"--enable-tvm-ffi --gpu-arch {arch}")
        print("COMPILED", arch, "thd" if thd else "dense")
    """)


@requires_dsl
def test_builders_trace_for_sm107a(tmp_path):
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    probe = tmp_path / "sf_desc_trace_probe.py"
    probe.write_text(_TRACE_PROBE)
    proc = subprocess.run([sys.executable, str(probe), "sm_107a"], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile of the SF descriptor builders failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "COMPILED sm_107a dense" in proc.stdout and "COMPILED sm_107a thd" in proc.stdout, proc.stdout[-2000:]
