---
name: cutedsl-kernel-integration
description: Use when integrating a CuTeDSL/CUTE DSL kernel into cuDNN Frontend as a frontend-only Python API, including APIBase wrappers, lazy cudnn exports, optional cutedsl dependencies, FE OSS documentation, and pytest coverage.
---

# CuTeDSL Kernel Integration

Use this skill to add or update a CuTeDSL frontend-only API in cuDNN Frontend. The goal is a complete integration: Python API, wrapper, exports, docs, and tests.

## Before Editing

1. Inspect the current repo state and avoid overwriting unrelated changes.
2. Confirm every original source file needed for the integration is available. If a source file is missing, report that gap instead of inferring its contract from a related kernel.
3. Record source provenance when it is available: upstream URL, local source path, commit, and which files map to public API modules versus private helpers.
4. Classify the kernel before choosing a template:
   - Kernel family: dense GEMM, GEMM fusion, grouped GEMM, discrete grouped GEMM, MoE, attention, sparse attention, or another frontend-only API family.
   - Execution topology: single kernel, paired forward/backward APIs, multi-kernel orchestrator, helper-kernel setup, distributed/runtime-coordinated execution, or internal scheduler.
   - Public surface: class API, high-level wrapper, returned tensors, optional outputs, workspace ownership, and import/export namespace.
   - Internal support: source helper modules, schedulers, metadata utilities, and generated descriptors that must stay private to the package.
   - Architecture variant: whether the public API needs transparent dispatch to an alternate CuTeDSL module for a newer GPU (for example Rubin `sm107` vs the default SM100 kernel). Keep the public class and wrapper unchanged when dispatch is internal.
5. Read `references/integration-pattern.md` for the detailed repo conventions before implementing.

## Integration Workflow

1. Add or update the operation package under the closest existing family, such as `python/cudnn/<operation>/`, `python/cudnn/gemm/cutedsl/dense/<operation>/`, `python/cudnn/gemm/cutedsl/grouped/<operation>/`, `python/cudnn/gemm/cutedsl/discrete_grouped/<operation>/`, or `python/cudnn/sdpa/<direction>/`.
2. Implement the class API by extending `APIBase`; keep constructor descriptors, `check_support()`, `compile()`, and `execute()` consistent with the closest template.
3. Add a high-level wrapper that allocates outputs, caches/reuses compiled kernels where the template does, and returns a `TupleDict`.
4. Export the public class and wrapper through the operation/family `__init__.py` files and `_LAZY_OPTIONAL_IMPORTS` in `python/cudnn/__init__.py`.
5. Reuse the existing `cutedsl` optional dependency unless the new kernel truly needs an additional package.
6. Add FE OSS documentation and update the relevant overview or operation index links.
7. Add tests under `test/python/fe_api/`, including support validation and numerical/reference coverage when executable.
8. For grouped/discrete/MoE/SDPA kernels, preserve the source helper and scheduler topology; shared helper modules should be internal package files, not public `cudnn` exports.
9. When an existing SM100 kernel needs a Rubin (`sm107`) variant, follow the architecture-dispatch pattern in `references/integration-pattern.md` instead of exposing a new public API. Current examples: `grouped_gemm_quant`, `grouped_gemm_glu`, and `grouped_gemm_dglu`.

## Verification

- Run focused formatting or tests for the files changed.
- At minimum for skill-only edits, verify this `SKILL.md` has valid frontmatter and all referenced paths exist.
- For kernel integrations, run the relevant `pytest test/python/fe_api/test_<operation>.py` target when the environment has the required GPU and optional dependencies; otherwise report the skipped verification explicitly.
- For architecture-dispatch work, also run `pytest test/python/fe_api/test_rubin_kernel_dispatch.py`. On Rubin hardware, the existing FE API e2e tests for the affected operation should still pass without API changes.
