# Matmul

The Matmul operation computes:

$$ C[M, N] = A[M, K] * B[K, N] $$

Last two dimensions of input dimensions are interpreted as M, N, K. All other preceding dimensions are interpreted as batch dimensions. The operation also has broadcasting capabilities which are described in [cudnn backend's matmul operation](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnn-backend-operation-matmul-descriptor).

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

### FROST operand fusion

On supported FROST GEMM mainloop templates, a `leaky_relu` feeding either
matmul operand can be fused into the operand transform before MMA:

```python
activated = graph.leaky_relu(input=A, negative_slope=0.1)
activated.set_data_type(cudnn.data_type.BFLOAT16)
C = graph.matmul(A=activated, B=B)
C.set_output(True)
```

The finite `negative_slope` is preserved in the fusion IR, as are each
pointwise node's compute and output dtypes. Consecutive activations retain
their intermediate rounding boundaries; they are not algebraically combined.
This does not enable other parameterized mainloop operations, `gen_index`, or
per-row/per-column/per-element operand auxiliaries. Existing template and
architecture restrictions still apply.

Set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` to offer the FROST GEMM engine as a
plan candidate. This is an engine capability, not a guarantee that every graph
or chosen plan will use it. The
[self-checking sample](https://github.com/NVIDIA/cudnn-frontend/blob/develop/samples/frost/gemm/07_matmul_leaky_relu_mainloop.py)
explicitly selects FROST and optionally compares CUDA-graph replay against
separate PyTorch activations followed by FROST GEMM. Timings exclude graph
construction and JIT compilation; performance depends on the shape and plan.
