# Pin a C++ engine and knobs

Query backend engines for an operation graph, inspect supported knobs, build a chosen configuration, and fall back cleanly when it is unsupported.

## Before you begin

Use a validated C++ Graph and read [engines and plans](/concepts/execution-plans).
Treat every discovered engine ID and knob value as local to the graph and
backend version that produced it.

## Discover engines

Create the graph representation, call `get_engine_count()`, and query knobs for each candidate. Engine indices and knobs are backend-version- and graph-specific; do not hard-code values discovered on another graph or release.

## Build a custom plan

Call `create_execution_plan(engine_id, knob_choices)` or the matching current overload, check its `error_t`, build it, query workspace, and execute with compatible bindings. Use heuristic planning as a fallback when a pinned configuration becomes invalid.

## Record evidence

Log graph key, engine ID, knob values, plan name, numerical and behavior notes, workspace, backend version, GPU, and build result. Treat this record as a cache entry with explicit compatibility, not as a portable API constant.

References: [`create_execution_plan`](/api/cpp/create-execution-plan), [`get_knobs_for_engine`](/api/cpp/get-knobs-for-engine), and [`get_plan_name`](/api/cpp/get-plan-name).

## Expected result

A supported engine and knob assignment builds and executes with compatible
bindings. If that assignment is no longer supported, the application records
the rejection and uses its checked heuristic fallback instead of relying on a
stale numeric ID.
