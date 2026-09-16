# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework dispatch for the public BSA forward and backward APIs."""

from importlib import import_module

from cudnn.tensor_adapter import detect_framework


def tensor_framework(q_tensor, *tensors):
    framework = detect_framework(q_tensor)
    if framework not in ("torch", "jax"):
        raise TypeError("BSA requires PyTorch tensors or JAX arrays")
    if any(tensor is not None and detect_framework(tensor) != framework for tensor in tensors):
        raise TypeError("All BSA tensors must use the same framework")
    return framework


def block_sparse_attention_forward(
    q_tensor,
    k_tensor,
    v_tensor,
    q2k_block_index,
    block_sparse_num=None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    sparse_block_size=None,
    allow_empty_block_nums=False,
    softmax_scale=None,
    pack_gqa=None,
    layout="bhsd",
    kv_splits=1,
    use_clc=None,
):
    """Return output and LSE using the input framework's BSA implementation."""
    framework = tensor_framework(q_tensor, k_tensor, v_tensor, q2k_block_index, block_sizes, q2k_block_nums)
    options = dict(sparse_block_size=sparse_block_size, allow_empty_block_nums=allow_empty_block_nums, softmax_scale=softmax_scale, layout=layout)
    if framework == "jax":
        if pack_gqa not in (None, False) or kv_splits != 1 or use_clc is not None:
            raise NotImplementedError("JAX BSA requires pack_gqa=None or False, kv_splits=1, and use_clc=None")
        options["sparse_block_size"] = 128 if sparse_block_size is None else sparse_block_size
    else:
        options.update(pack_gqa=pack_gqa, kv_splits=kv_splits, use_clc=use_clc)
    backend = import_module(".jax_api" if framework == "jax" else ".api", __package__)
    return backend.block_sparse_attention_forward(q_tensor, k_tensor, v_tensor, q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums, **options)


def block_sparse_attention_backward(
    do_tensor,
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    lse_tensor,
    q2k_block_index,
    block_sparse_num=None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    softmax_scale=None,
    dq_tensor=None,
    dk_tensor=None,
    dv_tensor=None,
    bucket_size_blocks=None,
    sparse_block_size=None,
    layout="bhsd",
    allow_empty_block_nums=False,
):
    """Return explicit Q/K/V gradients; JAX always allocates fresh outputs."""
    framework = tensor_framework(
        q_tensor, do_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, q2k_block_index, block_sizes, q2k_block_nums, dq_tensor, dk_tensor, dv_tensor
    )
    options = dict(softmax_scale=softmax_scale, bucket_size_blocks=bucket_size_blocks, sparse_block_size=sparse_block_size, layout=layout)
    if framework == "jax":
        if any(output is not None for output in (dq_tensor, dk_tensor, dv_tensor)):
            raise NotImplementedError("JAX BSA does not accept caller-provided gradient outputs")
        options.update(sparse_block_size=128 if sparse_block_size is None else sparse_block_size, allow_empty_block_nums=allow_empty_block_nums)
    else:
        if allow_empty_block_nums:
            raise NotImplementedError("allow_empty_block_nums is only exposed by the JAX backward API")
        options.update(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)
    backend = import_module(".jax_api" if framework == "jax" else ".api", __package__)
    return backend.block_sparse_attention_backward(
        do_tensor, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums, **options
    )
