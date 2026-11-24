# Matmul

The Matmul operation computes:

$$ C[M, N] = A[M, K] * B[K, N] $$

Last two dimensions of input dimensions are interpreted as M, N, K. All other preceding dimensions are interpreted as batch dimensions. The operation also has broadcasting capabilities which are described in [cudnn backend's matmul operation](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnn-backend-operation-matmul-descriptor).

### Tensor rank

Which ranks build depends on the engine that serves the graph:

- **cuDNN backend engines** take exactly rank-3 tensors: A `[B, M, K]`, B `[B, K, N]` and C `[B, M, N]`. A batch extent of 1 on A or B broadcasts across the other operand's batch. Rank-2 and rank-4 or higher graphs are declined when the plans are built with `cudnnGraphNotSupportedError`. Write the plain product `C[M, N] = A[M, K] * B[K, N]` as `[1, M, K] x [1, K, N] -> [1, M, N]`, and fold several batch dimensions into one before building the graph. C must be rank 3 as well: on SM90, rank-3 inputs with a rank-2 C were declined.
- **The FROST GEMM engine** (opt-in with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` on the GPUs it serves) supports rank-2 graphs and folds leading batch dimensions for rank-4 or higher graphs when they can be represented by one affine batch stride. After padding shorter batch prefixes with leading ones, each input's batch prefix must either match the output batch prefix or consist entirely of ones. Partial or crossed broadcasting, such as A `[2, 1, M, K]` and B `[1, 3, K, N]`, is not supported. Non-affine batch layouts are also declined rather than copied.

Measured with cuDNN 9.27 and fp16 inputs. On SM90 and SM103 the backend engines served only rank 3. On SM103 with FROST enabled, ranks 2 through 5 were served.

## C++ API

```
std::shared_ptr<Tensor_attributes>
Matmul(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Matmul_attributes);
```

Matmul attributes is a lightweight structure with setters:  
```
Matmul_attributes&
set_name(std::string const&)

Matmul_attributes&
set_compute_data_type(DataType_t value)
```
## Python API

- matmul
    - A
    - B
    - name
    - compute_data_type

### Batch-invariant BF16 forward for reinforcement learning

The Python graph API offers the `rl_gemm` engine on SM100, SM103 and SM107.
Request it with the `RL` numerical note on both the rollout and training forward graphs.

```python
g = cudnn.pygraph(
    io_data_type=cudnn.data_type.BFLOAT16,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)
a = g.tensor(dim=[1, m, k], stride=[m * k, k, 1], data_type=cudnn.data_type.BFLOAT16)
b = g.tensor(dim=[1, k, n], stride=[n * k, 1, k], data_type=cudnn.data_type.BFLOAT16)
y = g.matmul(A=a, B=b)
y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
g.select_numeric_notes([cudnn.numerical_note.RL])
g.build()
```

On the same architecture and software stack, a row with identical inputs, weights
and bias follows the same K16 FP32 accumulation order at any batch size or row
position. The weight shape fixes the number and order of K chunks. To include
a BF16 column bias, keep the matmul intermediate in FP32 and use `g.add` with
FP32 compute before the BF16 output conversion.

The engine accepts rank-2 matrices or rank-3 matrices with a singleton batch
dimension, positive M, and positive K and N divisible by eight. A has row-major
layout with a leading dimension divisible by eight. B is compact row-major or
column-major. Output is contiguous row-major and the operand pointers require
16-byte alignment. Build a graph for each distinct shape. Allocate
`g.get_workspace_size()` bytes of scratch for execution, with separate scratch
for concurrent launches. Requesting the note on an unsupported graph raises
`cudnnGraphNotSupportedError`. For a linear layer `Y = X @ W.T + bias`, training
can use native cuDNN matmul plans to compute `dX = dY @ W` and `dW = dY.T @ X`,
and an FP32 sum over the rows of `dY` for the bias gradient. The backward graphs
use the saved input and weight and do not require the RL note.
See the [determinism support matrix](https://docs.nvidia.com/cudnn/operations/determinism).

### Scaled FP8 and FP4 forward for reinforcement learning

Select `cudnn.numerical_note.RL` on the training graph and pass
`batch_invariant=True` to the corresponding FlashInfer rollout operation.
Use FP32 compute and intermediate tensors, then set the final output to
`BFLOAT16` or `HALF`. The input dtype, scale format and graph operations
identify the numerical recipe.

| FlashInfer operation | Inputs | Scale operations in the training graph |
| --- | --- | --- |
| `bmm_fp8` | E4M3 × E4M3 or E5M2 × E5M2 | Multiply two FP32 device scalars, then multiply the FP32 matmul result by their product. Supports multiple independent batch elements. |
| `gemm_fp8_nt_groupwise` | E4M3 × E4M3 | Dequantize A in [1, 128] blocks and B in [128, 128] blocks with FP32 scales. The engine scales each K-block product before adding it to the running sum. |
| `mm_mxfp8` | E4M3 × E4M3 | E8M0 block scales along K, 32 elements per scale. |
| `mm_fp4` (NVFP4) | E2M1 × E2M1 | E4M3 block scales along K, 16 elements per scale. Optional FP32 output multiplier. |
| `mm_fp4` (MXFP4) | E2M1 × E2M1 | E8M0 block scales along K, 32 elements per scale. Optional FP32 output multiplier. |
| `mm_mxfp8_mxfp4` | E4M3 × E2M1 | E8M0 block scales along K, 32 elements per scale. Optional FP32 output multiplier. |

All these paths require N divisible by 8. Per-tensor FP8 requires K divisible
by 16 (and N divisible by 16 for row-major B). Groupwise FP8 requires a whole
number of 128-element K blocks. MX/NV paths take K-contiguous operands and K
divisible by 32. Mixed MXFP8 × MXFP4 additionally requires K divisible by
128 on SM100/103. To request programmatic dependent launch for that mixed
projection, pass `enable_pdl=True` to `g.matmul`.

For example, given descriptors for MXFP4 A, B and their two scale buffers,
build the training forward as follows.

```python
x = g.block_scale_dequantize(input=a, descale=sa, block_size=[1, 32])
w = g.block_scale_dequantize(input=b, descale=sb, block_size=[32, 1])
p = g.matmul(A=x, B=w)
y = g.mul(a=p, b=alpha)  # Omit the multiply when alpha is absent.
y.set_output(True).set_data_type(cudnn.data_type.HALF)
g.select_numeric_notes([cudnn.numerical_note.RL])
g.build()
```

Describe A and B with logical dimensions `[M, K]` and `[K, N]`. For FP4,
each byte stores two adjacent K elements. For MX and NV block scales, use
`tensor_reordering.F8_128x4`. Round the number of rows up to 128 and the
number of K blocks up to 4. For A, describe this padded scale buffer as
`[padded_M, padded_blocks]` with strides `[padded_blocks, 1]`. For B, use
`[padded_blocks, padded_N]` with strides `[1, padded_blocks]`.
When extracting or permuting rows for rollout, move their logical activation
scales together with the quantized payload and repack the scale buffer.

The bitwise forward contract applies on the same GPU architecture and software
stack with fixed weights, scales, output dtype and logical input rows. Training
can compute gradients for the effective floating-point operands using native
cuDNN matmuls. A scalar output multiplier also multiplies the upstream gradient.
See the [RL tests](../../test/python/rl/README.md) for executable graphs.
