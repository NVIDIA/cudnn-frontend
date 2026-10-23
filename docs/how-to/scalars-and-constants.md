# Use scalar, pass-by-value, and compile-time constant tensors

Choose a scalar representation from when the value is known and where its storage lives.

## Before you begin

Use the C++ Graph API for the complete compile-time-constant example. Decide
whether the value belongs to graph identity, changes between executions, or is
ordinary device data before choosing its representation.

## Pass by value

`tensor_scalar` and scalar `Tensor_attributes` create scalar-shaped graph values. Pass-by-value values are bound from host-visible storage according to the backend variant-pack contract; keep that storage valid until the asynchronous operation has consumed it.

## Compile-time constants

C++ scalar constructors with `ScalarType::COMPILE_TIME_CONST` embed supported values into graph construction. Changing the value changes the graph and can require a new plan. Use only supported scalar types and include the value in serialization and cache identity.

The C++ Graph has compile-time scalar overloads for `float`, `double`, `half`, `bfloat16`, `int32_t`, and `int64_t`. The Python binding exposes `float`, `double`, `int32_t`, and `int64_t` overloads through `tensor_scalar(value, cudnn.scalar_type.COMPILE_TIME_CONST)`; Python numeric conversion and pybind overload resolution determine the selected C++ type.

Under the hood, `Tensor_attributes::set_compile_time_constant(...)` stores the payload and `get_has_compile_time_constant()` reports it. A compile-time constant must be pass-by-value and non-virtual. It must not also carry a runtime `pass_by_value` payload. Prefer `tensor_scalar` over manipulating those flags directly because the scalar constructor establishes the invariant as one operation.

Compile-time constants are not present in the runtime variant pack. Replacing a constant requires a new graph identity and may require new plan compilation. Runtime parameters, by contrast, preserve graph structure and receive a host scalar binding at execution.

## Runtime scalar tensors

Use a regular device tensor when the value changes per execution or participates in broadcasting as data. It requires a runtime binding and follows normal tensor lifetime and layout rules.

Canonical references: [`tensor_scalar`](/api/python/tensor-scalar) and [C++ `tensor_scalar`](/api/cpp/tensor-scalar).

## Construct a constant FMA graph

The published C++ sample constructs and validates a multiply-add graph whose scale and bias are embedded in graph identity. It is compiled against the current headers; kernel execution requires the newer backend recorded below.

<Code
  src="../samples/cpp/compile_time_constants.cc"
  language="cpp"
  maxLines={0}
/>

## Verification limits

The sample compiles against the current headers. Its runtime contract requires
cuDNN 9.22 and CUDA 13.1, so it could not execute with the recorded cuDNN 9.15.1
backend. Direct tensor-attribute setters can also create contradictory runtime
and compile-time scalar state when called in the wrong order; prefer the scalar
constructors, which establish the invariant in one operation.

## Expected result

The C++ sample compiles a fused multiply-add graph with scale and bias embedded
in graph identity. On cuDNN 9.22 or newer it can build and execute without
runtime bindings for those constants; the recorded 9.15.1 backend establishes
compilation only, not execution.
