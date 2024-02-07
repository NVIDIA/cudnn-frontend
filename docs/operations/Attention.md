## Table of Contents
1. [Scaled Dot Product Attention](#scaled-dot-product-attention)
2. [Scaled Dot Product Attention Backward](#scaled-dot-product-attention-backward)
3. [Miscellaneous](#miscellaneous)

### Scaled Dot Product Attention

This operation computes the scaled dot product attention, as

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$

using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). It is applicable for both training and inference phases, with an option to generate a stats tensor to be used for backwards training computation.

#### Configurable Options:

- Attention scale (`attn_scale`): Applies a scaling factor to attention scores before the softmax, such as $\frac{1}{\sqrt{\text{d}}}$. Set to 1.0 by default.
- Bias mask: Applies an additive bias mask to attention scores. Users must pass a bias tensor as specified in the tensors section below.
- Alibi mask: Attention with Linear Biases (ALiBi) is an additive mask applied to the attention scores as described in the paper [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).
- Padding mask: Also called variable sequence length, this option masks out padded time steps to ignore them in computation. Users must pass a per-batch sequence length as specified in the tensors section below.
- Causal mask: Fills the upper triangular matrix of attention scores with negative infinity.
- Dropout: Randomly zeros some of the attention weights after the softmax as a form of regularization.
  Users can configure dropout in two ways:
  - To use the more performant Philox RNG dropout implementation, users must provide:
    - An RNG seed, passed as a cudnn tensor.
    - An RNG offset, passed as a cudnn tensor.
    - A float representing the dropout probability, which is the probability that any given weight is set to zero.
  - To use an user-provided dropout mask, users must provide:
    - `dropout mask` that matches the attention weights' dimensions, indicating which weights to drop.
    - `dropout scale` used to adjust the scale of the remaining weights accordingly, such as $1 / (1 - \text{dropout probability})$.
- Ragged tensor: allows the query, key, value, and output tensor to be [ragged tensors](https://www.tensorflow.org/guide/ragged_tensor), which are tensors with nested variable length lists as inner dimensions. Users must pass another tensor called ragged offset tensor using the `Tensor_attributes.set_ragged_offset()` method as specified in the tensors section below.

When multiple masking options are enabled, they are applied in the listed order above.

#### Tensors:

- Query tensor should have dimensions $(B, H_{q}, S_{q}, D_{qk})$ with input/output datatype.
- Key tensor should have dimensions $(B, H_{k}, S_{kv}, D_{qk})$ with input/output datatype.
- Value tensor should have dimensions $(B, H_{v}, S_{kv}, D_{v})$ with input/output datatype.
- Output tensor should have dimensions $(B, H_{q}, S_{q}, D_{v})$ with input/output datatype.
- (Optional) When `is_inference` is false, the stats tensor should have dimensions $(B, H_{q}, S_{q}, 1)$ with float32 datatype.
- (Optional) When bias mask is enabled, the bias tensor has dimensions $(1, 1, S_{q}, S_{kv})$, $(1, H_{q}, S_{q}, S_{kv})$, $(B, 1, S_{q}, S_{kv})$, or $(B, H_{q}, S_{q}, S_{kv})$ with input/output datatype.  
The dimensions that are passed as 1 will apply a broadcasted mask over attention scores.
- (Optional) When padding mask is enabled, the sequence length q, and sequence length kv tensors should have shape $(B, 1, 1, 1)$ with int32 datatype.
- (Optional) When philox RNG dropout mask is enabled, the RNG seed and offset tensors should have size $(1, 1, 1, 1)$ with int32 or int64 datatype as either a CPU or GPU tensor.
- (Optional) When a user provided dropout mask is enabled, a dropout mask tensor should have shape $(1, 1, S_{q}, S_{kv})$, $(1, H_{q}, S_{q}, S_{kv})$, $(B, 1, S_{q}, S_{kv})$, or $(B, H_{q}, S_{q}, S_{kv})$ with input/output datatype.  
The dimensions that are passed as 1 will apply a broadcasted mask over attention weights.
- (Optional) When query, key, value, and output tensors are ragged tensors, the ragged offset tensor must be a tensor of size $(B + 1, 1, 1, 1)$ that contains the nested tensor's offset in terms of number of elements (not bytes). The last value of the offset tensor specifies the offset of the past-the-end element of the ragged tensor.

Where,

- $B$ is the batch size
- $H_{q}$ is the number of query heads
- $H_{k}$ is the number of key heads
- $H_{v}$ is the number of value heads
- $S_{q}$ is the sequence length of the query
- $S_{kv}$ is the sequence length of the key and value
- $D_{qk}$ is the embedding dimension per head of query and key
- $D_{v}$ is the embedding dimension per head of value

#### Group-query attention (GQA) and Multi-query attention (MQA)

- As described in the paper [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245),
- When $H_{k}$ and $H_{v}$ is less than $H_{q}$ and factors of $H_{q}$, this operation will perform group-query attention (GQA) computation.
- When $H_{k}$ and $H_{v}$ are both set to 1, this operation perform multi-query attention (MQA) computation.

#### Limitations:

- All input and output tensor datatypes must be float16 or bfloat16 datatype except the softmax stats output tensor, which must be float32.
- The dimension of the embedding dimension per head $D_{qk}$ and $D_{v}$ must be a multiple of 8 with maximum value 128.
- the stride of the embedding dimension per head $D_{qk}$ and $D_{v}$ for all the tensors above must be 1.
- this operation is only supported on GPUs with NVIDIA Ampere architecture (SM80) or newer.

**API:**

```cpp
std::array<std::shared_ptr<Tensor_attributes>, 2> 
sdpa(std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     SDPA_attributes options);
```

The function returns an array of two tensors: `[output, softmax_stats]`.

The `options` parameter of type `SDPA_attributes` is used to control the attributes of the forward operation, as detailed below:

```cpp
SDPA_attributes &
set_is_inference(bool const value);

SDPA_attributes &
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_attn_scale(float const value);

SDPA_attributes &
set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_alibi_mask(bool const value);

SDPA_attributes&
set_padding_mask(bool const value);

SDPA_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes &
set_causal_mask(bool const value);

SDPA_attributes &
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

SDPA_attributes &
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale);

SDPA_attributes &
set_compute_data_type(DataType_t value);
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
    o (cudnn_tensor): The result of scaled dot-product attention.
    stats (Optional[cudnn_tensor]): The softmax statistics in case the operation is in a training step.
```

### Scaled Dot Product Attention Backward

This operation computes gradient tensors for scaled dot product attention using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). The user is required to pass the stats tensor from the forward operation to the backward operation as input.

#### Configurable Options:

All the options mentioned in the forward operation, including ragged tensors and GQA/MQA, are applicable in the backward operation as well.

#### Tensors:

All the tensor requirements described in the forward operation are applicable in the backward operation as well. The gradient tensors for query, key, value, output, and bias should have the same properites as their non-gradient counterparts.

#### Limitations:

All the limitations mentioned in the forward operation are applicable in the backward operation as well.

#### API:
```cpp
std::array<std::shared_ptr<Tensor_attributes>, 3>
sdpa_backward(std::shared_ptr<Tensor_attributes> q,
              std::shared_ptr<Tensor_attributes> k,
              std::shared_ptr<Tensor_attributes> v,
              std::shared_ptr<Tensor_attributes> o,
              std::shared_ptr<Tensor_attributes> dO,
              std::shared_ptr<Tensor_attributes> stats,
              SDPA_backward_attributes);
```

The function returns an array of three tensors: `[dQ, dK, dV]`.

The `options` parameter of type `SDPA_backward_attributes` is used to control the attributes of backward operation, as detailed below:

```cpp
SDPA_backward_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_attn_scale(float const value);

SDPA_backward_attributes&
set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_dbias(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_alibi_mask(bool const value);

SDPA_backward_attributes&
set_padding_mask(bool const value);

SDPA_backward_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes&
set_causal_mask(bool const value);

SDPA_backward_attributes&
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

SDPA_backward_attributes&
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale,
            std::shared_ptr<Tensor_attributes> scale_inv);

SDPA_backward_attributes&
set_compute_data_type(DataType_t const value);
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
    dQ (cudnn_tensor): The query gradient tensor of scaled dot-product attention.
    dK (cudnn_tensor): The key gradient tensor of scaled dot-product attention.
    dV (cudnn_tensor): The value gradient tensor of scaled dot-product attention.
```

### Miscellaneous
- FE provides shadow enums which help avoid users to workaround having different enums for different cudnn versions.
- The cudnn backend enums are changed as follows:
    - `cudnnBackend<enum_name>` -> `cudnn_frontend::<enum_name>`
    - `cudnn<enum_name>` -> `cudnn_frontend::<enum_name>`
- To dump the dropout mask generated by the Philox RNG dropout implementation for debugging purposes, users can use the `rng_dump` option. This option requires users to pass a tensor of dimensions $(B, H_{q}, S_{q}, S_{kv})$ 
- Scaled Dot Product Attention Backward improves performance by using an optional dP workspace tensor. This tensor's memory consumption increases quadratically with the sequence length. The following describes the behavior of the `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT` environment variable, which allows the user to change the GPU memory limit for this workspace tensor:
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = unset`  
    The optimization will utilize workspace memory until reaching the default limit of 256MB.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = -1`  
    Workspace optimization is always enabled, regardless of memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = 0`  
    Workspace optimization is always disabled, avoiding the additional memory usage.
  - `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT = n`  
    Allows workspace optimization up to a user-defined limit of n bytes, accommodating systems with varying GPU memory capacities.
