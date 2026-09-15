# Custom Execution Plan

Here is an example in C++ of creating a custom execution plan with a hard-coded engine and knobs. Refer to the corresponding C++ sample in [samples/cpp/misc/custom_plan.cpp](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/misc/custom_plan.cpp).

## Get the Engine Count
```
inline error_t
get_engine_count(int64_t &count);
```
### Parameters

- `count`: number of engines [out parameter]

### Return Value
- An `error_t` object indicating the success or failure of the function.

## Get the Knobs Supported by an Engine
```
inline error_t
get_knobs_for_engine(int64_t const engine, std::vector<Knob> &);
```
### Parameters

- `engine`: engine index
- `knobs`: list of knobs [out parameter]

### Return Value
- An `error_t` object indicating the success or failure of the function.

## Create Plan with a Particular Engine and Knobs
```
error_t
create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &knobs);
```
### Parameters

- `engine_id`: engine index
- `knobs`: knobs

### Return Value
- An `error_t` object indicating the success or failure of the function.