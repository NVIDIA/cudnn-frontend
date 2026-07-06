# Grouped GEMM + Wgrad

`GroupedGemmWgradSm100` and `grouped_gemm_wgrad_wrapper_sm100` expose the grouped GEMM weight-gradient kernel integrated from the Cute DSL kernel library.

## Operation

The API computes grouped weight gradients in 2Dx2D form:

```text
A(hidden, tokens_sum) x B(tokens_sum, intermediate) -> Wgrad(experts, hidden, intermediate)
```

The grouped dimension is the token axis, segmented by `offsets_tensor`.

## Public APIs

- Class API: `cudnn.GroupedGemmWgradSm100`
- Wrapper API: `cudnn.grouped_gemm_wgrad_wrapper_sm100`

## Inputs

- `a_tensor`: input tensor with logical shape `(hidden, tokens_sum)`
- `b_tensor`: input tensor with logical shape `(tokens_sum, intermediate)`
- `sfa_tensor`: assembled scale-factor tensor for `a_tensor`
- `sfb_tensor`: assembled scale-factor tensor for `b_tensor`
- `offsets_tensor`: cumulative end offsets per expert, shape `(num_experts,)`, dtype `torch.int32`
- `global_scale_a` / `global_scale_b`: optional per-expert global scales. These are required for NVFP4 (`sf_vec_size == 16` with FP4 inputs).

`input_order` selects how `a_tensor` and `b_tensor` are interpreted by the kernel:

- `"tensor2d"` (default): the token axis is one contiguous 2-D tensor.
- `"tensor_ragged"`: each expert's token block is laid out independently in memory and concatenated expert-major; the kernel builds per-expert TMA descriptors for A and B.

## Output Modes

The API supports two output modes through one public surface:

- Dense:
  - output tensor is a contiguous stacked tensor with shape `(num_experts, hidden, intermediate)`
- Discrete:
  - the kernel uses per-expert output pointers internally
  - the convenience wrapper still returns a stacked tensor while exercising the discrete-output path

## Wrapper Example

```python
import cudnn
import torch

result = cudnn.grouped_gemm_wgrad_wrapper_sm100(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    offsets_tensor=offsets_tensor,
    output_mode="dense",
    wgrad_dtype=torch.bfloat16,
    input_order="tensor_ragged",
)

wgrad_tensor = result["wgrad_tensor"]
```

## Class API Example

```python
import cudnn
import torch

op = cudnn.GroupedGemmWgradSm100(
    sample_a=a_tensor,
    sample_b=b_tensor,
    sample_sfa=sfa_tensor,
    sample_sfb=sfb_tensor,
    sample_offsets=offsets_tensor,
    sample_wgrad=sample_wgrad_tensor,
    acc_dtype=torch.float32,
)
op.check_support()
op.compile()
op.execute(
    a_tensor=a_tensor,
    b_tensor=b_tensor,
    sfa_tensor=sfa_tensor,
    sfb_tensor=sfb_tensor,
    offsets_tensor=offsets_tensor,
    wgrad_tensor=wgrad_tensor,
)
```

## Notes

- Requires SM100+ GPUs.
- `output_mode="discrete"` is available in the wrapper for parity with the underlying kernel mode.
- `accumulate_on_output=True` expects the output tensor to be initialized by the caller; the wrapper zero-initializes it automatically.
