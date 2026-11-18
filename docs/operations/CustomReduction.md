# Custom Reduction

For a conceptual walkthrough, see How custom reductions work ([docs](/operations/CustomReduction/overview), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/custom_reduction/overview.md)).

A custom reduction reduces one or more input tensors along the same dimensions. Each input is a stream. A reducer callback that you write sets how two values of a stream combine. The callback can read the values of all streams, so a reduction such as argmax (a value stream and an index stream) is one operation.

The frontend calls the reducer when it builds the operation graph. After an expansion that succeeds, the frontend keeps the child graph and does not call the reducer again. If the reducer throws an exception, the build returns `INVALID_VALUE` with the exception message (Python: a `RuntimeError` that contains the type and the message of the Python exception). If the reducer throws an exception or the frontend refuses the child graph, the frontend does not keep the child graph, and the next build calls the reducer again. The reducer adds pointwise operations to a child graph. cuDNN runs that child graph for each combination step.

## Requirements

The operation needs cuDNN 9.29.0 or later. The API is always present. With an older cuDNN, building the operation graph returns `GRAPH_NOT_SUPPORTED` with the message "Custom reduction requires cuDNN v9.29.0" (Python: `cudnn.cudnnGraphNotSupportedError`). The frontend reads the version of the loaded cuDNN library before it calls the reducer, so with an older library the reducer does not run. This is also true for a program that you compile with cuDNN 9.29 headers and run with an older library.

## Semantics

- Stream `i` has the input `X[i]` and the output `Y[i]`. All inputs have the same shape.
- `Y[i]` has the shape of `X[i]`, with extent 1 in each reduced dimension. `Y[i]` has the data type of `X[i]`.
- The reducer gets the child graph and two lists, `lhs` and `rhs`, with one scalar tensor for each stream. It returns one scalar tensor for each stream.
- The reducer body contains only pointwise operations. A build of the body gives `GRAPH_NOT_SUPPORTED` for any other operation before an operation in the body expands, also if the reducer builds the child graph itself, on any thread. Thus a custom reduction or an SDPA `score_mod` in the body never runs. A reducer can build a different graph that contains a custom reduction. Do not assign to or copy the child graph object. An operand is `lhs`, `rhs`, the result of a body operation, or a compile-time constant.
- The combination must be associative and commutative.
- The identity of stream `i` must be a two-sided identity of the combination. For example, use 0 for a sum and -inf for a maximum.
- The identity has exactly the data type of `Y[i]`. The frontend does not convert identities.
- The compute type of a stream is the data type of the body values of that stream. If you do not set it, the stream computes in its own data type. A compute type must be an integer type for an integer or boolean stream, and a floating-point type for a floating-point stream.
- The semantic key identifies the reducer in the graph key. Change the key when the behavior of the reducer changes.
- The graph JSON shows a custom reduction for inspection only. A graph with a custom reduction cannot be deserialized; deserialize the execution plan instead.

## C++ API

```cpp
std::vector<std::shared_ptr<Tensor_attributes>>
custom_reduction(std::vector<std::shared_ptr<Tensor_attributes>> inputs, Custom_reduction_attributes attributes);
```

The function returns one output tensor for each input, in input order.

`Custom_reduction_attributes` setters:

```cpp
using Reducer_t = std::function<std::vector<Tensor_t>(Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs)>;

Custom_reduction_attributes&
set_reducer(Reducer_t reducer, std::string semantic_key);

Custom_reduction_attributes&
set_reduction_dims(std::vector<int64_t> dims);  // strictly ascending

Custom_reduction_attributes&
set_identity_values(std::vector<Custom_reduction_identity> identities);  // one for each stream

Custom_reduction_attributes&
set_compute_data_types(std::vector<DataType_t> types);  // one for each stream

Custom_reduction_attributes&
set_compute_data_type(DataType_t type);  // the same type for each stream

Custom_reduction_attributes&
set_name(std::string const&);
```

`set_compute_data_types` and `set_compute_data_type` replace each other. The last call sets the compute types.

`Custom_reduction_identity` holds one identity value and its exact data type. Make it from a value of the stream type:

| Data type | Constructor argument |
|---|---|
| `INT8`, `UINT8`, `INT32`, `UINT32`, `INT64` | `int8_t`, `uint8_t`, `int32_t`, `uint32_t`, `int64_t` |
| `FLOAT`, `DOUBLE` | `float`, `double` |
| `HALF`, `BFLOAT16` | `half`, `nv_bfloat16` |
| `FP8_E4M3`, `FP8_E5M2` | `Custom_reduction_fp8_e4m3{bits}`, `Custom_reduction_fp8_e5m2{bits}` |
| `BYTE_BOOLEAN` | `Custom_reduction_byte_boolean{byte}` (0 is false; each other value is true) |

`cudnn_frontend::graph::reduction_modifiers::sum`, `max` and `min` take a `Custom_reduction_attributes` and return it with a single-stream reducer and its semantic key. They do not set an identity or a compute type.

`reduction_modifiers` also has argmax and argmin. These set the reducer, the semantic key and the identities. They do not set a compute type.

```cpp
enum class Arg_nan_rule { TORCH, XLA, IGNORE_NAN };

Custom_reduction_attributes
argmax(Custom_reduction_attributes attributes,
       DataType_t value_type,                      // the data type of the value stream
       DataType_t index_type = DataType_t::INT32,  // the data type of the index stream
       Arg_nan_rule rule     = Arg_nan_rule::TORCH);

Custom_reduction_attributes
argmin(Custom_reduction_attributes attributes,
       DataType_t value_type,
       DataType_t index_type = DataType_t::INT32,
       Arg_nan_rule rule     = Arg_nan_rule::TORCH);
```

- `argmax` and `argmin` reduce two streams: stream 0 holds the values and stream 1 holds the index of each element along the reduced dimensions (for example 0 to N−1), in a tensor of the full input shape. Among equal values, the lowest index value wins.
- `value_type` and `index_type` are the data types of the two streams; the identities have exactly these types. A value stream that is not `FLOAT`, `HALF`, `BFLOAT16` or `DOUBLE`, or an index stream that is not `INT32` or `INT64`, gives `INVALID_VALUE` from `validate`.
- The identities are `(-inf, largest index)` for argmax and `(+inf, largest index)` for argmin, and `(NaN, largest index)` for `IGNORE_NAN`, where the largest index is the maximum of the index type. Each is a two-sided identity of its body for any index values smaller than the largest index.
- The frontend builds the graph for each stream type; the cuDNN library decides which stream types and body forms it runs, and this can change with the library version. If the library does not support the graph, the build returns its status unchanged (`GRAPH_NOT_SUPPORTED` for a not-supported status).

The rule sets the result for NaN values. Each rule gives the lowest index among equal values, and `-0` equals `+0`.

| Rule | Row with NaN values | Commutative | Matches |
|---|---|---|---|
| `TORCH` (default) | the first NaN | yes | `torch.argmax`, `torch.argmin` |
| `XLA` | one of the NaN positions; with more than one NaN, which one is not specified | no, for two NaN values | the XLA argmax and argmin comparators |
| `IGNORE_NAN` | the best number; a row of NaN only gives its first position | yes | a NaN-ignoring argmax or argmin |

The backend contract requires a commutative combination. The `XLA` body is not commutative for two NaN values, so for those rows its result depends on the reduction order.

Example, an argmax over dimension 1 of BF16 values:

```cpp
auto outputs = graph.custom_reduction(
    {values, indices},
    reduction_modifiers::argmax(Custom_reduction_attributes().set_name("argmax").set_reduction_dims({1}),
                                DataType_t::BFLOAT16));
outputs[1]->set_output(true);  // the index
```

Example, a sum over dimension 1:

```cpp
auto Y = graph.custom_reduction({X},
                                reduction_modifiers::sum(Custom_reduction_attributes()
                                                             .set_name("row_sum")
                                                             .set_reduction_dims({1})
                                                             .set_identity_values({Custom_reduction_identity(0.0f)})))[0];
Y->set_output(true);
```

Example, a reducer that writes the body:

```cpp
auto reducer = [](Custom_reduction_attributes::Graph_t body,
                  std::vector<Custom_reduction_attributes::Tensor_t> lhs,
                  std::vector<Custom_reduction_attributes::Tensor_t> rhs) {
    auto larger = body->pointwise(
        lhs[0], rhs[0], Pointwise_attributes().set_mode(PointwiseMode_t::MAX).set_compute_data_type(DataType_t::FLOAT));
    return std::vector<Custom_reduction_attributes::Tensor_t>{larger};
};
auto Y = graph.custom_reduction({X},
                                Custom_reduction_attributes()
                                    .set_reducer(reducer, "my_project.row_max.v1")
                                    .set_reduction_dims({1})
                                    .set_identity_values({Custom_reduction_identity(-INFINITY)}))[0];
```

## Python API

- `pygraph.custom_reduction(inputs, reducer, reducer_semantic_key, reduction_dims, identity_values, *, compute_data_types=None, compute_data_type=None, name="")`
  - **inputs**: list of input tensors, one for each stream.
  - **reducer**: callable `reducer(graph, lhs, rhs)` that returns a list with one tensor for each stream. The reducer can capture parent tensors, use them as operands and return them. It can read captured parent graph state; add reducer operations to the provided child graph.
  - **reducer_semantic_key**: string that identifies the reducer in the graph key.
  - **reduction_dims**: list of reduced dimensions, strictly ascending.
  - **identity_values**: list of `cudnn.custom_reduction_identity`, one for each stream.
  - **compute_data_types**: optional list with one `cudnn.data_type` for each stream.
  - **compute_data_type**: optional `cudnn.data_type` for each stream. Give `compute_data_types` or `compute_data_type`, not both.
  - Returns the list of output tensors.
- `cudnn.custom_reduction_identity(dtype, value)` makes an identity from a Python value. Use a real number (`numbers.Real`, for example a `float` or a NumPy floating-point scalar) for `FLOAT` and `DOUBLE`, and an integer (`numbers.Integral`) for the integer types. A `bool` is not a number for these types. The value must have an exact representation in `dtype`. Use a `bool` for `BYTE_BOOLEAN`. Use `float("inf")` and `float("-inf")` for infinite identities.
- `cudnn.custom_reduction_identity.from_bits(dtype, bits)` makes an identity from its exact storage bits. Use it for `HALF`, `BFLOAT16`, FP8 and NaN encodings.
- `cudnn.reduction_modifiers.sum()`, `max()` and `min()` return a dictionary with the `reducer` and `reducer_semantic_key` arguments. These are the C++ `reduction_modifiers` reducers and keys.
- `cudnn.reduction_modifiers.argmax(value_type, index_type=cudnn.data_type.INT32, rule=cudnn.reduction_modifiers.ArgNanRule.TORCH)` and `argmin(...)` return a dictionary with the `reducer`, `reducer_semantic_key` and `identity_values` arguments. `rule` is an `ArgNanRule` (`TORCH`, `XLA`, `IGNORE_NAN`) or its value (`"torch"`, `"xla"`, `"ignore_nan"`). These are the C++ helpers: the same bodies, keys and identities, and the same rules and limits (see the C++ API).

Example, an argmax over dimension 1. `x` holds the values and `idx` holds the index of each element along dimension 1, from 0 to N−1 for a dimension of extent N.

```python
value, index = graph.custom_reduction(
    [x, idx],
    reduction_dims=[1],
    **cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT),
)
index.set_output(True)
```

A pointwise operation of the graph can produce an input stream. Examples, with `B` rows, a graph with the intermediate data type `FLOAT`, and `idx` the index tensor:

```python
argmax = cudnn.reduction_modifiers.argmax(cudnn.data_type.FLOAT)

# Gumbel-max sampling: the noise g is an input tensor [B, V]; the temperature t is [B, 1].
y = graph.add(graph.div(x, t), g)

# Masked argmax: positions where mask (BYTE_BOOLEAN) is false take -inf.
y = graph.binary_select(x, graph.tensor_scalar(float("-inf"), cudnn.scalar_type.COMPILE_TIME_CONST), mask)

# Recovered token of speculative decoding: max(p_t - p_d, 0) * inv_q.
y = graph.mul(graph.max(graph.sub(p_t, p_d), graph.tensor_scalar(0.0, cudnn.scalar_type.COMPILE_TIME_CONST)), inv_q)

_, index = graph.custom_reduction([y, idx], reduction_dims=[1], **argmax)
index.set_output(True)
```

Example, a sum with a modifier:

```python
y = graph.custom_reduction(
    [x],
    reduction_dims=[1],
    identity_values=[cudnn.custom_reduction_identity(cudnn.data_type.FLOAT, 0.0)],
    **cudnn.reduction_modifiers.sum(),
)[0]
```

## XLA HLO examples

`samples/cpp/membound/custom_reduction_xla.cpp` runs the variadic argmax that XLA emits (value and `s32` index, only the index read) at every distinct argmax operand shape of a corpus of XLA HLO dumps of LLM and vision models: 31 shapes. Each shape runs with `Arg_nan_rule::XLA` (the XLA comparator) and with `Arg_nan_rule::TORCH`, on rows with ties, one NaN, two NaN, signed zeros and only `-inf`, against a host reference. The sample also runs the second stage of an XLA split argmax.

| Types (value, index) | Operand shape | Reduced dimension | Source models |
|---|---|---|---|
| (bf16, s32) | `[1, 262144]` | 1 | Gemma 3 1B / 4B / 12B (Flax sample loop) |
| (f32, s32) | `[1, 1, 32000]` | 2 | DeepSeek-V2 16B / 236B, Llama 2 13B, Llama 3.1 70B, Mistral 7B, Mixtral 8x7B, Qwen3 (0.6B–32B) |
| (f32, s32) | `[1, 1, 32768]` | 2 | Mixtral 8x22B |
| (f32, s32) | `[1, 1, 128256]` | 2 | Llama 3.1 8B / 70B |
| (f32, s32) | `[1, 1, 151936]` | 2 | Qwen3 0.6B / 4B / 8B / 14B |
| (f32, s32) | `[1, 1, 201088]` | 2 | gpt-oss 20B |
| (f32, s32) | `[1, 1, 202048]` | 2 | Llama 4 17B-16E |
| (f32, s32) | `[1, 1, 256128]` | 2 | Gemma 2 9B / 27B |
| (f32, s32) | `[1, 1, 262144]` | 2 | Gemma 3 27B |
| (f32, s32) | `[1, 256000]` | 1 | Gemma 2 2B (Keras / JAX) |
| (f32, s32) | `[8, 1, 32000]` | 2 | Llama 3.1 70B |
| (f32, s32) | `[16, 1, 32000]` | 2 | Llama 2 13B, Mistral 7B |
| (f32, s32) | `[16, 1, 128256]` | 2 | Llama 3.1 8B |
| (f32, s32) | `[16, 1, 151936]` | 2 | Qwen3 0.6B / 4B / 8B / 14B |
| (f32, s32) | `[16, 1, 256128]` | 2 | Gemma 2 9B |
| (f32, s32) | `[32, 1, 32000]` | 2 | Llama 2 13B |
| (f32, s32) | `[32, 128, 32128]` | 2 | T5 (T5X training step) |
| (f32, s32) | `[128, 1, 32000]` | 2 | DeepSeek-V2 16B, Mixtral 8x7B |
| (f32, s32) | `[200, 1, 256128]` | 2 | Gemma 2 9B |
| (f32, s32) | `[256, 1, 32000]` | 2 | DeepSeek-V2 236B |
| (f32, s32) | `[256, 1, 32768]` | 2 | Mixtral 8x22B |
| (f32, s32) | `[256, 1, 202048]` | 2 | Llama 4 17B-16E |
| (f32, s32) | `[400, 1, 128256]` | 2 | Llama 3.1 70B |
| (f32, s32) | `[400, 1, 151936]` | 2 | Qwen3 8B / 14B / 32B |
| (f32, s32) | `[400, 1, 256128]` | 2 | Gemma 2 27B |
| (f32, s32) | `[400, 1, 262144]` | 2 | Gemma 3 27B |
| (f32, s32) | `[512, 1000]` | 1 | ViT (training step, batch 512) |
| (f32, s32) | `[600, 1, 32000]` | 2 | Mistral 7B |
| (f32, s32) | `[600, 1, 128256]` | 2 | Llama 3.1 8B |
| (f32, s32) | `[600, 1, 151936]` | 2 | Qwen3 0.6B / 4B / 30B-A3B |
| (f32, s32) | `[800, 1, 201088]` | 2 | gpt-oss 20B |

## Samples

- C++: `samples/cpp/membound/custom_reduction.cpp`, `custom_reduction_argmax.cpp` (the argmax and argmin rules), `custom_reduction_sampling.cpp` (greedy decoding, speculative-decoding verify, masked argmax, Gumbel-max) and `custom_reduction_xla.cpp` (XLA argmax shapes).
- Python: `samples/python/71_custom_reduction_argmax.ipynb`.
