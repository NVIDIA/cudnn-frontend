# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_bsa_jax_torch_parity(layout):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("JAX BSA requires SM100")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from cudnn import BSA

    rng = np.random.default_rng(19)
    arrays = [rng.normal(size=shape).astype(np.float32) for shape in ((1, 2, 256, 64), (1, 2, 512, 64), (1, 2, 512, 64))]
    if layout == "bshd":
        arrays = [np.ascontiguousarray(x.transpose(0, 2, 1, 3)) for x in arrays]
    tq, tk, tv = [torch.tensor(x, device="cuda", dtype=torch.bfloat16) for x in arrays]
    jq, jk, jv = [jnp.asarray(x, jnp.bfloat16) for x in arrays]
    index = np.broadcast_to(np.array([0, 3], np.int32), (1, 2, 2, 2)).copy()
    ti, ji = torch.tensor(index, device="cuda"), jnp.asarray(index)
    tf = BSA.block_sparse_attention_forward(tq, tk, tv, ti, 2, sparse_block_size=128, layout=layout)
    jf = jax.jit(lambda q, k, v, i: BSA.block_sparse_attention_forward_jax(q, k, v, i, 2, layout=layout))(jq, jk, jv, ji)
    tb = BSA.block_sparse_attention_backward(torch.ones_like(tq), tq, tk, tv, tf[0], tf[1], ti, 2, sparse_block_size=128, layout=layout)
    jb = BSA.block_sparse_attention_backward_jax(jnp.ones_like(jq), jq, jk, jv, jf[0], jf[1], ji, 2, layout=layout, bucket_size_blocks=1)
    for jt, tt in zip((*jf, *jb), (*tf, *tb)):
        np.testing.assert_allclose(np.asarray(jt.astype(jnp.float32)), tt.float().cpu().numpy(), atol=3e-2, rtol=3e-2)
