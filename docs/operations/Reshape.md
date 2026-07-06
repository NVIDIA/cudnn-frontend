# Reshape

The reshape operation connects an input tensor to an output tensor with a **different logical shape and stride**, without changing the underlying element type. The output rank and layout are usually set on the output tensor (or via `Reshape_attributes::set_dim` / `set_stride`); the node wires the reshape in the graph.

## Reshape modes (`ReshapeMode_t`)

| Mode | Meaning |
|------|--------|
| `VIEW_ONLY` | Default. View-style reshape when the layout is compatible with the backend. |
| `LOGICAL` | Lexicographic **logical** reshape: use when the mapping is expressed in logical (row-major lexicographic) terms as required by the backend. Prefer this when a plain view is not sufficient for the target shape. |

Mode is ignored on older cuDNN builds that do not expose reshape mode; on **cuDNN 9.22+**, it is passed to the backend reshape operation.

## C++ API

```cpp
std::shared_ptr<Tensor_attributes>
reshape(std::shared_ptr<Tensor_attributes> input, Reshape_attributes attributes);
```

Relevant `Reshape_attributes` setters:

```cpp
Reshape_attributes&
set_dim(std::vector<int64_t> const& value);

Reshape_attributes&
set_stride(std::vector<int64_t> const& value);

Reshape_attributes&
set_reshape_mode(ReshapeMode_t const& value);  // VIEW_ONLY or LOGICAL

Reshape_attributes&
set_name(std::string const& value);
```

## Python API

- `pygraph.reshape(input, name="", reshape_mode=cudnn.reshape_mode.VIEW_ONLY)`
  - **reshape_mode**: `cudnn.reshape_mode.VIEW_ONLY` or `cudnn.reshape_mode.LOGICAL`.

## Example (C++)

```cpp
using cudnn_frontend::ReshapeMode_t;
auto r = fe::graph::Reshape_attributes()
             .set_name("logical_flatten")
             .set_reshape_mode(ReshapeMode_t::LOGICAL)
             .set_dim({24})
             .set_stride({1});
auto Y = graph.reshape(X, r);
```
