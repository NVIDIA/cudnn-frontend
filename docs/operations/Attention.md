## Table of Contents
1. [Scaled Dot Product Flash Attention](#Scaled Dot Product Flash Attention)


### Scaled Dot Product Flash Attention
Computes the scaled dot product attention for given Query, Key and Value tensors. Optionally, can set dropout probability, causal mask. Can optionally dump stats to be used for the bprop computation.

API:
```
std::array<std::shared_ptr<Tensor_attributes>, 2> 
scaled_dot_product_flash_attention
    (std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     Scaled_dot_product_flash_attention_attributes options);
```

where the output array has tensors in order of: `[output, softmax_stats]`
where, `Scaled_dot_product_flash_attention_attributes` controls the sub-graph in the operation


```
    Scaled_dot_product_flash_attention_attributes &
    set_is_inference(bool const value);

    Scaled_dot_product_flash_attention_attributes &
    set_causal_mask(bool const value);
    
    Scaled_dot_product_flash_attention_attributes &
    set_bias(std::shared_ptr<Tensor_attributes> value);
    
    Scaled_dot_product_flash_attention_attributes &
    set_attn_scale(std::shared_ptr<Tensor_attributes> value);
    
    Scaled_dot_product_flash_attention_attributes &
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset);
    
    Scaled_dot_product_flash_attention_attributes &
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale);
    
    Scaled_dot_product_flash_attention_attributes &
    set_compute_data_type(DataType_t value)
```

Python API: 
    - q
    - k
    - v
    - seq_q
    - seq_k
    - is_inference
    - attn_scale
    - bias
    - use_padding_mask
    - use_alibi_mask
    - use_causal_mask
    - dropout
    - compute_data_type
    - name

## Miscellaneous
- FE provides shadow enums which help avoid users to workaround having different enums for different cudnn versions.
- The cudnn backend enums are changed as follows:
    - `cudnnBackend<enum_name>` -> `cudnn_frontend::<enum_name>`
    - `cudnn<enum_name>` -> `cudnn_frontend::<enum_name>`
