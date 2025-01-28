Here is an example of creating a custom execution plan with hardcoded engine and knobs. Please see coresponding C++ sample in `samples/cpp/misc/custom_plan.cpp`.

### Get engine count
```
inline error_t
get_engine_count(int64_t &count);
```
#### Parameters

- `count`: number of engines [out parameter]

#### Return Value
- An `error_t` object indicating the success or failure of the function.

### Get knobs supported by an engine
```
inline error_t
get_knobs_for_engine(int64_t const engine, std::vector<Knob> &);
```
#### Parameters

- `engine`: engine index
- `knobs`: list of knobs [out parameter]

#### Return Value
- An `error_t` object indicating the success or failure of the function.

### Create a plan with particular engine and knobs
```
error_t
create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &knobs);
```
#### Parameters

- `engine_id`: engine index
- `knobs`: knobs

#### Return Value
- An `error_t` object indicating the success or failure of the function.