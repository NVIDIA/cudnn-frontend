# include/cudnn_frontend — Agent Guide

The header-only C++ library (CMake INTERFACE target `cudnn_frontend`, C++17). Umbrella header: `include/cudnn_frontend.h`. Build/test commands: [../../AGENTS.md](../../AGENTS.md).

## Hard rules

- **Header-only**: no `.cpp` files, no link-time dependencies beyond cuDNN/CUDA. New third-party code must be vendored under `thirdparty/` (currently only `nlohmann/json.hpp`, excludable via `CUDNN_FRONTEND_SKIP_JSON_LIB`).
- Builds with `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang) and `/W4 /WX` (MSVC) — code must be warning-clean on both.
- C++17 only in `include/` (the pybind11 layer under `python/` is C++20).
- Guard anything needing a newer cuDNN with runtime `detail::get_backend_version()` checks (compare against `CUDNN_FRONTEND_VERSION`-style integers, e.g. 9.12.0 → 91200); the same headers must compile against older cuDNN 9.x.
- clang-format (Google-based, indent 4, 120 cols, `SortIncludes: false`) via `pre-commit run`.

## Layering

```
graph_interface.h   Graph class (namespace cudnn_frontend::graph); includes every node/*.h
  node/*.h          one header per op node (matmul, conv_fprop, sdpa, rmsnorm, ...): <Op>Node + <Op>_attributes
  node_interface.h  INode base; graph_properties.h holds attribute classes
  cudnn_interface.h ICudnn: lowering to cuDNN backend ops/plans
  backend/          C-backend descriptor wrappers (execution plans, kernel cache, device properties)
plans.h, knobs.h, graph_helpers.h, context.h   plan management, autotuning knobs, error handling
```

Legacy flat API (`include/cudnn_frontend_*.h`: Tensor, Operation, ExecutionPlan, Heuristics...) is maintained but frozen — new features target the graph API under this directory.

## Adding a graph operation (typical shape)

1. `node/<op>.h`: `<Op>_attributes` (+ `NLOHMANN_DEFINE_TYPE_INTRUSIVE`-style serialization where applicable) and the node class implementing infer/expand/lowering.
2. Wire into `graph_interface.h` (include + `Graph::<op>(...)` builder method returning tensor attributes).
3. Serialization support in `utils/serialize.h` if the op is cacheable.
4. Doc page in `docs/operations/`, sample under `samples/cpp/`, tests (C++ and/or `test/python/`).
5. Python surface: pybind wrapper in `python/pygraph/` when the op should be scriptable.

## experimental/ and generated/

- `generated/` holds **open-sourced kernel source embedded as raw C++ string literals** (`inline constexpr const char <name>_source[]`, namespace `cudnn_frontend::experimental::generated`) — SDPA prefill (sm90/sm100) and RMSNorm+SiLU. These files are large and machine-produced; don't hand-edit kernel bodies casually, and don't "clean them up".
- `experimental/` is the NVRTC glue that compiles those strings at runtime (`IOssSdpaEngine`: `check_support`/`build`/`execute`, per-arch engines, `nvrtc_shim.h`).

## Versioning

`cudnn_frontend_version.h` defines `CUDNN_FRONTEND_{MAJOR,MINOR,PATCH}_VERSION`. Keep in sync with `CMakeLists.txt` `project(VERSION ...)` and `python/cudnn/__init__.py.__version__` when bumping.
