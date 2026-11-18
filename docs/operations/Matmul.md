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
