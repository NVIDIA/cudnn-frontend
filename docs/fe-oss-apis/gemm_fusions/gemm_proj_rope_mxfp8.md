# GEMM + RoPE + MXFP8 Projection (SM100)

**This is an experimental API and subject to change.**

## Overview

**Fused projection GEMM + per-head YARN RoPE + dual-direction MXFP8 quantize**: a persistent dense GEMM on NVIDIA Blackwell GPUs (SM100+) that projects activations, applies the Megatron MLA-YARN rotary embedding to each attention head's trailing rotary features, and MXFP8 (E4M3, block=32) quantizes the result in **both** the rowwise (D-direction) and columnwise (S-direction) layouts. Implemented with CUTLASS/CUTE.

Two sibling kernels implement the same operation, differing only in the **GEMM input precision**, and are selected by the dtype of `x`/`w`:

- **BF16 input** (`GemmProjRopeMxfp8Bf16InSm100`): `x` and `w` are `bfloat16`; the GEMM runs in bf16.
- **MXFP8 input** (`GemmProjRopeMxfp8Mxfp8InSm100`): `x` and `w` are pre-quantized MXFP8 (E4M3 codes + E8M0 rowwise block scales); the GEMM runs in MXFP8.

The **output is dual-direction MXFP8 in both cases**. Emitting both scale directions makes the output directly consumable by block-scaled matmuls that need either operand orientation. For example, in DeepSeek-V3 the rowwise output feeds the forward `QK^T` and the columnwise output feeds the backward `dK = dS^T · Q` on the cuDNN `is_input_fp8` attention path.

- **Inputs**: activations `x`, projection weight `w`, and bf16 rotary tables `cos`/`sin`; for the MXFP8-input path, the E8M0 block scales `x_scale`/`w_scale` as well.
- **Outputs**: rowwise and columnwise MXFP8 data (`out_fp8_row`, `out_fp8_col`) and their E8M0 scale factors (`out_scales_row`, `out_scales_col`).

The kernel is tuned for the DeepSeek-V3 Q up-projection shapes: `NUM_HEADS=128`, `HEAD_DIM=192` (`QK_NOPE=128` + `QK_ROPE=64`), MXFP8 `BLOCK=32`, tile `TILE_M=128` (one head per CTA).

### Shapes

- **Inputs**
  - `x`: `(tokens, Q_LORA)` — `tokens % TILE_M == 0`. Dtype `bfloat16` (bf16 path) or `float8_e4m3fn` (MXFP8 path).
  - `w`: `(Q_LORA, NUM_HEADS·HEAD_DIM)` when `w_out_in=False`, or the transformer-engine-native transposed `(NUM_HEADS·HEAD_DIM, Q_LORA)` when `w_out_in=True`. Same dtype as `x`.
  - `x_scale`, `w_scale` (MXFP8 path only): E8M0 rowwise block scales, `uint8`. `x_scale` is `(tokens, Q_LORA // BLOCK)`. `w_scale` follows `w`'s layout (the wrapper transposes it alongside `w`): `(NUM_HEADS·HEAD_DIM, Q_LORA // BLOCK)` when `w_out_in=True`, or `(Q_LORA // BLOCK, NUM_HEADS·HEAD_DIM)` when `w_out_in=False`.
  - `cos`, `sin`: `(tokens, QK_ROPE)`, `bfloat16`.

- **Outputs**
  - `out_fp8_row`, `out_fp8_col`: `(tokens, NUM_HEADS, HEAD_DIM)`
  - `out_scales_row`: `(tokens, NUM_HEADS, HEAD_DIM // BLOCK)`
  - `out_scales_col`: `(tokens // BLOCK, NUM_HEADS, HEAD_DIM)`

### Equations

Project and reshape per head, then apply the interleaved-in / halves-out YARN RoPE to the trailing `QK_ROPE` features of each head:

$$
Y[t, h, :] = (x \, W)\;\text{reshaped to}\;[\text{tokens}, \text{NUM\_HEADS}, \text{HEAD\_DIM}]
$$

$$
Y_{\text{pe}} = \operatorname{RoPE}(Y[\ldots, \text{QK\_NOPE}:],\; \cos, \sin)
$$

MXFP8 quantize with block size `BLOCK=32`, independently for each direction (E8M0 per-block scale, E4M3 data):

$$
(\text{out\_fp8\_row}, \text{out\_scales\_row}) = \operatorname{MXFP8}_{\text{D}}(Y)\quad\text{(blocks along HEAD\_DIM)}
$$

$$
(\text{out\_fp8\_col}, \text{out\_scales\_col}) = \operatorname{MXFP8}_{\text{S}}(Y)\quad\text{(blocks along tokens)}
$$

### Diagram

```text
x (tokens x Q_LORA), w (Q_LORA x NUM_HEADS*HEAD_DIM)     [bf16, or MXFP8 codes+scales]
     |  GEMM (bf16 or MXFP8, per input dtype)
     v
   Y (tokens x NUM_HEADS x HEAD_DIM)  ---- per-head YARN RoPE on trailing QK_ROPE
     |
     +--> MXFP8 rowwise (D)  -> out_fp8_row, out_scales_row
     +--> MXFP8 columnwise(S)-> out_fp8_col, out_scales_col
```

## API Usage

### High-level wrapper (dtype-dispatch)

Selects the bf16- or mxfp8-input kernel by `x`/`w` dtype (which must match); pass `x_scale`/`w_scale` only for the MXFP8 path.

```python
# BF16 inputs
result = gemm_proj_rope_mxfp8_wrapper_sm100(x, w, cos, sin, w_out_in=True, stream=None)

# MXFP8 inputs (E4M3 codes + E8M0 scales)
result = gemm_proj_rope_mxfp8_wrapper_sm100(
    x_code, w_code, cos, sin, x_scale=x_scale, w_scale=w_scale, w_out_in=True, stream=None,
)

out_fp8_row, out_scales_row, out_fp8_col, out_scales_col = result
# Key access: result["out_fp8_row"], result["out_scales_col"], ...
```

### Class API — BF16 input

The class constructor defaults to `w_out_in=False` (`w` stored `[in, out]`); pass `w_out_in=True` for TE-native `[out, in]` weights. (The high-level wrapper defaults the other way, to `w_out_in=True`.)

```python
from cudnn import GemmProjRopeMxfp8Bf16InSm100

op = GemmProjRopeMxfp8Bf16InSm100(
    sample_x=x, sample_w=w, sample_cos=cos, sample_sin=sin,
    sample_out_fp8_row=out_fp8_row, sample_out_scales_row=out_scales_row,
    sample_out_fp8_col=out_fp8_col, sample_out_scales_col=out_scales_col,
    w_out_in=False,  # w is [in, out]; use w_out_in=True for TE-native [out, in]
)
assert op.check_support()
op.compile()
op.execute(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, current_stream=None)
```

### Class API — MXFP8 input

Unlike the bf16 class, this class has **no `w_out_in` parameter**: `w_code`/`w_scale` must already be TE-native `[out, in] = [N, K]`. Use the high-level wrapper if your weight is `[in, out]` — it transposes the code and scale before constructing this API.

```python
from cudnn import GemmProjRopeMxfp8Mxfp8InSm100

op = GemmProjRopeMxfp8Mxfp8InSm100(
    sample_x_code=x_code, sample_x_scale=x_scale, sample_w_code=w_code, sample_w_scale=w_scale,
    sample_cos=cos, sample_sin=sin,
    sample_out_fp8_row=out_fp8_row, sample_out_scales_row=out_scales_row,
    sample_out_fp8_col=out_fp8_col, sample_out_scales_col=out_scales_col,
)   # w_code/w_scale are TE-native [out, in] = [N, K] (no w_out_in on this class)
assert op.check_support()
op.compile()
op.execute(x_code, x_scale, w_code, w_scale, cos, sin,
           out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, current_stream=None)
```

---

## Parameters

### Input/Output tensors
- Input **x**: `(tokens, Q_LORA)`; Dtype `bfloat16` (bf16 path) or `float8_e4m3fn` (MXFP8 path).
- Input **w**: `(Q_LORA, NUM_HEADS·HEAD_DIM)` (`w_out_in=False`) or `(NUM_HEADS·HEAD_DIM, Q_LORA)` (`w_out_in=True`); same dtype as **x**.
- Input **x_scale**, **w_scale** (MXFP8 path only): Dtype `uint8` (E8M0 rowwise block scales). `x_scale` is `(tokens, Q_LORA // BLOCK)`. `w_scale`'s shape depends on `w_out_in` (transposed with `w`): `(NUM_HEADS·HEAD_DIM, Q_LORA // BLOCK)` for `w_out_in=True`, `(Q_LORA // BLOCK, NUM_HEADS·HEAD_DIM)` for `w_out_in=False`.
- Input **cos**, **sin**: `(tokens, QK_ROPE)`; Dtype `bfloat16`.
- Output **out_fp8_row** / **out_fp8_col**: `(tokens, NUM_HEADS, HEAD_DIM)`; Dtype `float8_e4m3fn`.
- Output **out_scales_row**: `(tokens, NUM_HEADS, HEAD_DIM // BLOCK)`; Dtype `uint8` (E8M0).
- Output **out_scales_col**: `(tokens // BLOCK, NUM_HEADS, HEAD_DIM)`; Dtype `uint8` (E8M0).

### Common parameters
- `w_out_in: bool` — whether `w` is stored `[out, in]` (`True`) or `[in, out]` (`False`). Wrapper default: `True`. On the bf16 path the kernel consumes both via the cutlass major mode (no transposed copy); on the MXFP8 path the wrapper transposes the code + scale for `[in, out]`.
- `x_scale`, `w_scale: Optional[Tensor]` — required for `float8_e4m3fn` inputs; must be `None` for `bfloat16`. `x.dtype == w.dtype` is asserted.
- CUDA stream (`current_stream` in class API, `stream` in wrapper). Defaults to the current torch stream (required for CUDA-graph capture).

### Wrapper return values

Returns a `TupleDict` with keys `out_fp8_row`, `out_scales_row`, `out_fp8_col`, `out_scales_col`. Tuple unpacking order is `(out_fp8_row, out_scales_row, out_fp8_col, out_scales_col)`.

---

## Support surface and constraints

### Dtypes
- BF16 path: `x`, `w`, `cos`, `sin` are `bfloat16`.
- MXFP8 path: `x`, `w` are `float8_e4m3fn` codes; `x_scale`, `w_scale` are `uint8` (E8M0); `cos`, `sin` are `bfloat16`.
- Outputs (both paths): `out_fp8_row`, `out_fp8_col` are `float8_e4m3fn`; `out_scales_row`, `out_scales_col` are `uint8` (E8M0).

### Shapes and divisibility
- `tokens % TILE_M == 0` (`TILE_M = 128`); no tail handling.
- The projected weight dimension must equal `NUM_HEADS·HEAD_DIM`. On the MXFP8 path, `Q_LORA % BLOCK == 0`.

### Environment
- Requires CUDA with SM100+ compute capability.

---

## Source provenance

Integrated from the DeepSeek-V3 MLA fused Q up-projection kernel developed for Megatron-LM MXFP8 training (Blackwell / customte CUTLASS 4.4.1); originally added in the GEMM+RoPE+MXFP8 fusion commit. The BF16-GEMM and MXFP8-GEMM variants were consolidated into two input-precision kernel modules selected by input dtype:

- `python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py` — BF16-input kernel; also hosts the pure-PyTorch oracle `gemm_proj_rope_mxfp8_reference(...)`.
- `python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py` — MXFP8-input kernel.

The compiled-kernel lifecycle (`check_support`/`compile`/`execute`) lives in the APIBase classes in `api.py`; the wrapper caches the compiled objects (matching the sibling GEMM-fusion packages).

## Installation

Requires the optional CuTeDSL dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## Usage examples

For usage examples, see test cases in `test/python/fe_api/gemm/test_gemm_proj_rope_mxfp8.py`.
