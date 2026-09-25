# Compile-time constant tensors

Some scalars should be **fixed when the graph is built** so the compiler and backend can fold them (constants in the plan), rather than supplied on every `execute()` like ordinary pass-by-value tensors.

The graph API distinguishes this with `ScalarType`:

| `ScalarType` | Role |
|--------------|------|
| `RUNTIME_PARAM` | Pass-by-value scalar whose **value can change** each launch (pointer in variant pack). |
| `COMPILE_TIME_CONST` | Scalar **baked into the graph**; not a separate runtime buffer for that value. |

## C++ API

Create tensors from a scalar and `ScalarType`:

```cpp
std::shared_ptr<Tensor_attributes> tensor(float const& scalar, ScalarType scalar_type);
std::shared_ptr<Tensor_attributes> tensor(half const& scalar, ScalarType scalar_type);
// ... int32, int64, double, bfloat16 overloads
```

Under the hood, `COMPILE_TIME_CONST` sets the tensor’s compile-time constant payload (`Tensor_attributes::set_compile_time_constant` / `get_has_compile_time_constant()`). Such tensors must be pass-by-value, non-virtual, and must not also carry a runtime `pass_by_value` payload.

## Python API

- `pygraph.tensor_scalar(value, scalar_type)`
  - Overloads for floating and integer scalars; **`scalar_type`** is `cudnn.scalar_type.RUNTIME_PARAM` or `cudnn.scalar_type.COMPILE_TIME_CONST`.

You can still query flags on `cudnn.tensor`:

- `get_has_compile_time_constant()` / `set_compile_time_constant` (advanced; prefer `tensor_scalar` for scalars).

## Example

See `samples/cpp/misc/compile_time_constant_example.cpp` for a fused multiply-add using `ScalarType::COMPILE_TIME_CONST`.
