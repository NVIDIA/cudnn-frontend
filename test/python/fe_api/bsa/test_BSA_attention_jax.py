# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX coverage for block-sparse attention (SM100/SM110 blk128 paths).

JAX contract: forward and backward on the SM100/SM110 blk128 kernels only
(SM90/SM120 and the blk64 paths reject JAX arrays with clear errors). The
backward's internal transposed tensor views travel as zero-copy permuted
DLPack wrappers (`cudnn.tensor_adapter.permuted_view`). Outputs are checked
bit-identical against the torch wrapper run on identical input bytes (both
paths share one compiled kernel per stride pattern).
"""

import importlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_utils import make_fixed_metadata, supported_block_size

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


def _skip_unless_blk128():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major not in {10, 11}:
        pytest.skip("the BSA JAX contract covers the SM100/SM110 blk128 paths only")


def _device_sync():
    # The eager JAX path runs on the CUDA legacy default stream, which XLA does
    # not track; synchronize before reading outputs.
    torch.cuda.synchronize()


def _to_jax(t: torch.Tensor):
    np_dtype = {torch.bfloat16: ml_dtypes.bfloat16, torch.float32: np.float32, torch.int32: np.int32}[t.dtype]
    return jnp.asarray(t.view(torch.uint8).cpu().numpy().view(np_dtype).reshape(t.shape) if t.dtype == torch.bfloat16 else t.cpu().numpy())


def _bits(x) -> np.ndarray:
    return np.asarray(x).view(np.uint8)


def _make_problem(seed: int):
    block_size = supported_block_size(backward=True)
    assert block_size == 128
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    do = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    return block_size, q, k, v, do, q2k, block_sparse_num, block_sizes


@pytest.mark.L0
def test_bsa_attention_forward_jax_matches_torch():
    _skip_unless_blk128()
    BSA = _import_bsa()
    block_size, q, k, v, _, q2k, block_sparse_num, block_sizes = _make_problem(seed=0)

    result_t = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size)
    torch.cuda.synchronize()

    q_j, k_j, v_j, q2k_j, sizes_j = (_to_jax(t) for t in (q, k, v, q2k, block_sizes))
    jax.block_until_ready((q_j, k_j, v_j, q2k_j, sizes_j))
    result_j = BSA.block_sparse_attention_forward(q_j, k_j, v_j, q2k_j, block_sparse_num, sizes_j, sparse_block_size=block_size)
    _device_sync()

    np.testing.assert_array_equal(
        _bits(result_j["o_tensor"]),
        result_t["o_tensor"].view(torch.uint8).cpu().numpy(),
        err_msg="BSA forward o: JAX output differs from torch output on identical input bytes",
    )
    np.testing.assert_array_equal(
        _bits(result_j["lse_tensor"]),
        result_t["lse_tensor"].view(torch.uint8).cpu().numpy(),
        err_msg="BSA forward lse: JAX output differs from torch output on identical input bytes",
    )


@pytest.mark.L0
def test_bsa_attention_forward_jax_bshd_layout():
    _skip_unless_blk128()
    BSA = _import_bsa()
    block_size, q, k, v, _, q2k, block_sparse_num, block_sizes = _make_problem(seed=1)
    # bshd-contiguous buffers for both frameworks (same bytes, same strides)
    q_s, k_s, v_s = (t.transpose(1, 2).contiguous() for t in (q, k, v))

    result_t = BSA.block_sparse_attention_forward(q_s, k_s, v_s, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size, layout="bshd")
    torch.cuda.synchronize()

    q_j, k_j, v_j, q2k_j, sizes_j = (_to_jax(t) for t in (q_s, k_s, v_s, q2k, block_sizes))
    jax.block_until_ready((q_j, k_j, v_j, q2k_j, sizes_j))
    result_j = BSA.block_sparse_attention_forward(q_j, k_j, v_j, q2k_j, block_sparse_num, sizes_j, sparse_block_size=block_size, layout="bshd")
    _device_sync()

    for key in ("o_tensor", "lse_tensor"):
        np.testing.assert_array_equal(
            _bits(result_j[key]),
            result_t[key].view(torch.uint8).cpu().numpy(),
            err_msg=f"BSA forward bshd {key}: JAX output differs from torch output on identical input bytes",
        )


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_bsa_attention_backward_jax_matches_torch(layout):
    _skip_unless_blk128()
    BSA = _import_bsa()
    block_size, q, k, v, do, q2k, block_sparse_num, _ = _make_problem(seed=2)
    if layout == "bshd":
        q, k, v, do = (t.transpose(1, 2).contiguous() for t in (q, k, v, do))

    forward_t = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, sparse_block_size=block_size, layout=layout)
    backward_t = BSA.block_sparse_attention_backward(
        do,
        q,
        k,
        v,
        forward_t["o_tensor"],
        forward_t["lse_tensor"],
        q2k,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
        layout=layout,
    )
    torch.cuda.synchronize()

    do_j, q_j, k_j, v_j, q2k_j = (_to_jax(t) for t in (do, q, k, v, q2k))
    o_j, lse_j = _to_jax(forward_t["o_tensor"]), _to_jax(forward_t["lse_tensor"])
    jax.block_until_ready((do_j, q_j, k_j, v_j, q2k_j, o_j, lse_j))
    backward_j = BSA.block_sparse_attention_backward(
        do_j,
        q_j,
        k_j,
        v_j,
        o_j,
        lse_j,
        q2k_j,
        block_sparse_num,
        None,
        sparse_block_size=block_size,
        layout=layout,
    )
    _device_sync()

    # dq is bit-deterministic; dk/dv accumulate across Q blocks in a
    # scheduling-dependent order (bit-nondeterministic run to run even within
    # one framework), so they compare at a 1-ulp-scale bf16 tolerance.
    for key, bitwise in (("dq_tensor", True), ("dk_tensor", False), ("dv_tensor", False)):
        got = backward_j[key]
        assert not hasattr(got, "_array"), f"{key} must be a real JAX array, not an internal view wrapper"
        expected = backward_t[key].contiguous()
        if bitwise:
            np.testing.assert_array_equal(
                _bits(got),
                expected.view(torch.uint8).cpu().numpy(),
                err_msg=f"BSA backward {layout} {key}: JAX output differs from torch output on identical input bytes",
            )
        else:
            np.testing.assert_allclose(
                np.asarray(got).astype(np.float32),
                expected.float().cpu().numpy(),
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"BSA backward {layout} {key}: JAX output differs from torch output beyond accumulation-order noise",
            )


@pytest.mark.L0
def test_bsa_attention_jax_errors():
    _skip_unless_blk128()
    BSA = _import_bsa()
    block_size, q, k, v, do, q2k, block_sparse_num, block_sizes = _make_problem(seed=3)
    q_j, k_j, v_j, q2k_j = (_to_jax(t) for t in (q, k, v, q2k))

    with pytest.raises(ValueError, match="blk128"):
        BSA.block_sparse_attention_forward(q_j, k_j, v_j, q2k_j, block_sparse_num, sparse_block_size=64)

    with pytest.raises(ValueError, match="blk128"):
        BSA.block_sparse_attention_backward(
            _to_jax(do),
            q_j,
            k_j,
            v_j,
            q_j,
            jnp.zeros(q.shape[:3], dtype=jnp.float32),
            q2k_j,
            block_sparse_num,
            None,
            sparse_block_size=64,
        )

    with pytest.raises(ValueError, match="Unsupported tensor framework"):
        BSA.block_sparse_attention_forward(np.asarray(q_j), np.asarray(k_j), np.asarray(v_j), np.asarray(q2k_j), block_sparse_num)
