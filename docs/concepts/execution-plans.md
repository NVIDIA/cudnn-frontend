# Engines, heuristics, plans, routing, and autotuning

An engine is an implementation family. A configuration chooses its knobs. An execution plan is a graph-specific built configuration. Heuristics propose candidates; filters apply application policy; autotuning measures eligible plans.

## Python routing

`cudnn.pygraph` discovers eligible Python engines from the library's static manifest. Family heuristics return ranked `PlanConfig(engine_id, knobs)` entries and place the cuDNN backend's heuristic block among them. There is no runtime engine-registration or router-replacement API.

Backend entries retain their own backend-plan indices when those candidates are exposed; a delegating backend entry represents candidates that remain internal. The combined `graph.plans` index must never be passed directly as a backend index.

This routed Python plan model is unreleased. See
[Python IR routing availability](/concepts/python-ir-routing#availability) before
depending on its representation or index-space contract.

## C++ planning

The C++ Graph API builds an operation graph, queries heuristic modes, filters engines or configurations, checks support, builds one or more plans, and executes a selected plan. Engine IDs and knob choices are backend- and graph-specific.

## Notes and policy

Behavior and numerical notes describe properties such as determinism, down-conversion, or runtime compilation. Workspace and shared-memory filters enforce resource budgets. A policy filter is not an accuracy test; validate representative outputs independently.

## Autotuning

Warm up, time on the intended stream, synchronize at measurement boundaries, and include representative addresses and shapes. Cache a choice with frontend/backend/driver/GPU and graph metadata. Re-tune when a compatibility field changes.
