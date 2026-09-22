# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import subprocess
import sys
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_bsa_jax_torch_parity(layout):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("JAX BSA requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("JAX BSA parity tests")
    if requirement_error:
        pytest.skip(requirement_error)
    jax = pytest.importorskip("jax", minversion="0.9.1")
    import jax.numpy as jnp
    from cudnn import BSA
    from cudnn.jax import block_sparse_attention_forward as jax_forward, block_sparse_attention_backward as jax_backward

    rng = np.random.default_rng(19)
    arrays = [rng.normal(size=shape).astype(np.float32) for shape in ((1, 2, 256, 64), (1, 2, 512, 64), (1, 2, 512, 64))]
    if layout == "bshd":
        arrays = [np.ascontiguousarray(x.transpose(0, 2, 1, 3)) for x in arrays]
    tq, tk, tv = [torch.tensor(x, device="cuda", dtype=torch.bfloat16) for x in arrays]
    jq, jk, jv = [jnp.asarray(x, jnp.bfloat16) for x in arrays]
    index = np.broadcast_to(np.array([0, 3], np.int32), (1, 2, 2, 2)).copy()
    ti, ji = torch.tensor(index, device="cuda"), jnp.asarray(index)
    tf = BSA.block_sparse_attention_forward(tq, tk, tv, ti, 2, sparse_block_size=128, layout=layout)
    jf = jax.jit(lambda q, k, v, i: jax_forward(q, k, v, i, 2, layout=layout))(jq, jk, jv, ji)
    tb = BSA.block_sparse_attention_backward(torch.ones_like(tq), tq, tk, tv, tf[0], tf[1], ti, 2, sparse_block_size=128, layout=layout)
    jb = jax_backward(jnp.ones_like(jq), jq, jk, jv, jf[0], jf[1], ji, 2, layout=layout, bucket_size_blocks=1)
    for jt, tt in zip((*jf, *jb), (*tf, *tb)):
        np.testing.assert_allclose(np.asarray(jt.astype(jnp.float32)), tt.float().cpu().numpy(), atol=3e-2, rtol=3e-2)

    for q, k, v, index in ((tq, jk, jv, ji), (jq, tk, jv, ji), (jq, jk, jv, ti)):
        with pytest.raises(TypeError, match="must be a JAX array"):
            jax_forward(q, k, v, index, 2, layout=layout)
    with pytest.raises(TypeError, match="must be a JAX array"):
        jax_backward(tq, jq, jk, jv, jf[0], jf[1], ji, 2, layout=layout)


def test_torch_runtime_without_jax():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("This isolation case uses SM100 blk128")
    script = """
import sys
class NoJax:
    def find_spec(self, fullname, *args):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ModuleNotFoundError("JAX is unavailable", name=fullname)
sys.meta_path.insert(0, NoJax())
import torch
import cudnn
import cudnn.torch as ct
assert "cudnn.block_sparse_attention.api" not in sys.modules
for name in ("block_sparse_attention_forward", "block_sparse_attention_backward", "block_sparse_attention_fp8_forward"):
    assert name in ct.__all__
    assert getattr(ct, name) is getattr(cudnn, name) is getattr(cudnn.BSA, name)
q = torch.ones((1, 1, 256, 64), device="cuda", dtype=torch.bfloat16)
i = torch.arange(2, device="cuda", dtype=torch.int32).expand(1, 1, 2, 2).contiguous()
o, lse = ct.block_sparse_attention_forward(q, q, q, i, 2)
dq, dk, dv = ct.block_sparse_attention_backward(q, q, q, q, o, lse, i, 2)
torch.testing.assert_close(o, q)
torch.testing.assert_close(dq, torch.zeros_like(q), atol=1e-3, rtol=0)
torch.testing.assert_close(dk, torch.zeros_like(q), atol=1e-3, rtol=0)
torch.testing.assert_close(dv, q)
assert "jax" not in sys.modules
assert "cudnn.block_sparse_attention.jax_api" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
