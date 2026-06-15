# Matmul

The Matmul operation computes:

$$ C[M, N] = A[M, K] * B[K, N] $$

Last two dimensions of input dimensions are interpreted as M, N, K. All other preceding dimensions are interpreted as batch dimensions. The operation also has broadcasting capabilities which are described in {ref}`cudnn backend's matmul operation <CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR>`.

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
