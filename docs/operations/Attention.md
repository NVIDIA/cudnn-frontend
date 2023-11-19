## Table of Contents
1. [Scaled Dot Product Flash Attention](#scaled-dot-product-flash-attention)
2. [Scaled Dot Product Flash Attention Backward](#scaled-dot-product-flash-attention-backward)

### Scaled Dot Product Flash Attention
Computes the scaled dot product attention for given Query, Key and Value tensors. Setting `is_inference` to false configures the operation to output `softmax_stats` to be used for backwards computation.

The user can also optionally configure attention scale, bias mask, alibi mask, padding mask, causal mask, and dropout for this operation.

The dimensions for

- Query tensor should be $(B, H, S_{q}, D)$
- Key tensor should be $(B, H, S_{kv}, D)$
- Value tensor should be $(B, H, S_{kv}, D)$
- Output tensor should be $(B, H, S_{q}, D)$
- Stats tensor should be $(B, H, S_{q}, 1)$

Where $B$ is the batch size, $H$ is the number of heads, $S_{q}$ is the sequence length of the query, $S_{kv}$ is the sequence length
of the key and value, and $D$ is the embedding dimension per head.

Additionally, the stride for the last dimension $D$ corresponding to the embedding dimension per head for each of these tensors must be 1.

**API:**

```cpp
std::array<std::shared_ptr<Tensor_attributes>, 2> 
scaled_dot_product_flash_attention
    (std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     Scaled_dot_product_flash_attention_attributes options);
```

where the output array has tensors in order of: `[output, softmax_stats]` and `Scaled_dot_product_flash_attention_attributes` controls the sub-graph in the operation

```cpp
Scaled_dot_product_flash_attention_attributes &
set_is_inference(bool const value);

Scaled_dot_product_flash_attention_attributes &
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_attributes &
set_bias(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_attributes&
set_alibi_mask(bool const value)

Scaled_dot_product_flash_attention_attributes&
set_padding_mask(bool const value);

Scaled_dot_product_flash_attention_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_attributes &
set_causal_mask(bool const value);

Scaled_dot_product_flash_attention_attributes &
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

Scaled_dot_product_flash_attention_attributes &
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale);

Scaled_dot_product_flash_attention_attributes &
set_compute_data_type(DataType_t value)
```

**Python API:**

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    is_inference (bool): Whether it is an inference step or training step.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
    use_padding_mask (Optional[bool]): Whether to use padding mask. Default is False.
    seq_len_q (Optional[cudnn_tensor]): The sequence length of the query.
    seq_len_kv (Optional[cudnn_tensor]): The sequence length of the key.
    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)], Tuple[mask: cudnn_tensor, scale: cudnn_tensor]]]): Whether to do dropout. Default is None.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    o (cudnn_tensor): The result of scaled dot-product flash attention.
    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
```

### Scaled Dot Product Flash Attention Backward
Computes the query, key and value gradient tensors for scaled dot product flash attention.

The user can also optionally configure attention scale, bias mask, alibi mask, padding mask, causal mask, and dropout for this operation.

The dimensions for

- Query tensor should be $(B, H, S_{q}, D)$
- Key tensor should be $(B, H, S_{kv}, D)$
- Value tensor should be $(B, H, S_{kv}, D)$
- Output tensor should be $(B, H, S_{q}, D)$
- Stats tensor should be $(B, H, S_{q}, 1)$
- Gradient tensors for query, key, value, and output should follow the same convention

Where $B$ is the batch size, $H$ is the number of heads, $S_{q}$ is the sequence length of the query, $S_{kv}$ is the sequence length
of the key and value, and $D$ is the embedding size per head.

Additionally, the stride for the last dimension corresponding to the embedding size per head for each of these tensors
must be 1.

API:
```cpp
std::array<std::shared_ptr<Tensor_attributes>, 3>
scaled_dot_product_flash_attention_backward
    (std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     std::shared_ptr<Tensor_attributes> o,
     std::shared_ptr<Tensor_attributes> dO,
     std::shared_ptr<Tensor_attributes> stats,
     Scaled_dot_product_flash_attention_backward_attributes);
```

where the output array has tensors in order of: `[dQ, dK, dV]`
where, `Scaled_dot_product_flash_attention_backward_attributes` controls the sub-graph in the operation


```cpp
Scaled_dot_product_flash_attention_backward_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value)

Scaled_dot_product_flash_attention_backward_attributes&
set_bias(std::shared_ptr<Tensor_attributes> value)

Scaled_dot_product_flash_attention_backward_attributes&
set_dbias(std::shared_ptr<Tensor_attributes> value)

Scaled_dot_product_flash_attention_backward_attributes&
set_alibi_mask(bool const value)

Scaled_dot_product_flash_attention_backward_attributes&
set_padding_mask(bool const value);

Scaled_dot_product_flash_attention_backward_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_backward_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

Scaled_dot_product_flash_attention_backward_attributes&
set_causal_mask(bool const value)

Scaled_dot_product_flash_attention_backward_attributes&
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset)

Scaled_dot_product_flash_attention_backward_attributes&
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale,
            std::shared_ptr<Tensor_attributes> scale_inv)

Scaled_dot_product_flash_attention_backward_attributes&
set_compute_data_type(DataType_t const value)
```

Python API: 

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    o (cudnn_tensor): The output data.
    dO (cudnn_tensor): The output loss gradient.
    stats (cudnn_tensor): The softmax statistics from the forward pass.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    bias (Optional[cudnn_tensor]): The bias data for attention. Default is None.
    dBias (Optional[cudnn_tensor]): The dBias output for attention. Default is None.
    use_alibi_mask (Optional[bool]): Whether to use alibi mask. Default is False.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    dropout (Optional[Union[Tuple[(probability: float, seed: cudnn_tensor, offset: cudnn_tensor)],
                            Tuple[mask: cudnn_tensor, scale: cudnn_tensor, scale_inv: cudnn_tensor]]]):
        Whether to do dropout. Default is None.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    dQ (cudnn_tensor): The query gradient tensor of scaled dot-product flash attention.
    dK (cudnn_tensor): The key gradient tensor of scaled dot-product flash attention.
    dV (cudnn_tensor): The value gradient tensor of scaled dot-product flash attention.
```

## Miscellaneous
- FE provides shadow enums which help avoid users to workaround having different enums for different cudnn versions.
- The cudnn backend enums are changed as follows:
    - `cudnnBackend<enum_name>` -> `cudnn_frontend::<enum_name>`
    - `cudnn<enum_name>` -> `cudnn_frontend::<enum_name>`
- Scaled Dot Product Flash Attention Backward improves performance by through the use of an optional dP workspace tensor. This tensor's memory consumption increases quadratically with the sequence length. The following describes the behavior of the `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT` environment variable, which allows the user to change the GPU memory limit for this workspace tensor:
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = unset`  
    The optimization will utilize workspace memory until reaching the default limit of 256MB.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = -1`  
    Workspace optimization is always enabled, regardless of memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = 0`  
    Workspace optimization is always disabled, avoiding the additional memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = n`  
    Allows workspace optimization up to a user-defined limit of n bytes, accommodating systems with varying GPU memory capacities.
