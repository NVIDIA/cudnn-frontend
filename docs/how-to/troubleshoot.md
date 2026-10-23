# Diagnose graph failures

Separate description errors, support failures, compilation failures, and runtime failures instead of retrying the complete build blindly.

## Before you begin

Reproduce the failure with one graph, fixed inputs, and
[`cudnn.pygraph` lifecycle stages](/how-to/explicit-lifecycle) exposed separately.
Record the frontend, backend, CUDA, GPU, graph metadata, and first failing
stage.

## Validate the description

Call `graph.validate()` first. Inspect dimensions, strides, data types, names, UIDs, virtual state, pass-by-value values, and inferred outputs. `ValueError` normally identifies an invalid Python IR relationship; C++ returns an `error_t` whose message names the failed frontend check.

## Create the graph representation

Call `graph.build_operation_graph()` separately. Failure here can indicate invalid backend descriptors, unavailable operation attributes, or a backend version that cannot represent the graph.

## Inspect and select plans

Call `graph.create_execution_plans()` and inspect `graph.plans`. A Python engine should decline only with a documented decline exception; another exception is an engine defect. For backend candidates, use engine and plan filters before compilation.

## Check support and build

`graph.check_support()` distinguishes a well-formed but unsupported graph. `graph.build_plans()` can still fail during kernel or plan compilation. Record the selected engine, knobs, behavior notes, numerical notes, workspace, backend version, GPU, and complete error.

## Validate runtime bindings

Provide every required non-virtual UID exactly once, enough workspace, compatible override arrays, and pointers valid until asynchronous work completes. Confirm the cuDNN handle uses the intended CUDA stream.

## Enable logs

```bash
CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=stdout <your command>
```

For numerical issues, compare one operation at a time in the same compute and output types. Synchronize before reading output or attributing a later asynchronous error to the wrong call.

## Expected result

The first failing lifecycle stage is isolated with its original status and
complete graph context. Description errors are corrected before planning;
unsupported graphs take only an explicit fallback; runtime failures are not
misreported as plan-selection problems.
