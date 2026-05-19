# CuTeDSL Frontend-Only API Integration Pattern

This reference captures the local cuDNN Frontend pattern for integrating a CuTeDSL/CUTE DSL kernel as a Python frontend-only API.

## Package Layout

Create or update an operation package under `python/cudnn/`.

Typical layout:

```text
python/cudnn/<operation>/
|-- __init__.py
|-- api.py
`-- <cutedsl_kernel_module>.py
```

Nested API families are also valid when matching existing structure, for example:

- `python/cudnn/grouped_gemm/<operation>/`
- `python/cudnn/discrete_grouped_gemm/<operation>/`
- `python/cudnn/sdpa/fwd/` or `python/cudnn/sdpa/bwd/`
- `python/cudnn/native_sparse_attention/<component>/`

Choose the closest existing family before creating a new top-level package.

Use this routing table before choosing the package namespace:

| Kernel shape | Preferred family |
| --- | --- |
| Plain dense GEMM or alpha/beta GEMM | Dense frontend-only package under `python/cudnn/<operation>/` |
| Dense GEMM with a fused pointwise epilogue, scale, amax, or auxiliary output | GEMM-fusion package under `python/cudnn/<operation>/` |
| Dense GEMM with distributed all-reduce | Dense package only after documenting distributed runtime requirements |
| Grouped GEMM with packed or contiguous grouped inputs | `python/cudnn/grouped_gemm/<operation>/` |
| Grouped GEMM with `padded_offsets`, `expert_cnt`, or scheduler workspace | `python/cudnn/grouped_gemm/<operation>/`, regardless of "dense" wording in source comments |
| Grouped GEMM with `group_count`, `problem_shape_mnkl`, `strides_abc`, or `tensor_address_*` arrays | `python/cudnn/grouped_gemm/<operation>/` with per-group metadata |
| Grouped GEMM with per-expert discrete weight pointers | `python/cudnn/discrete_grouped_gemm/<operation>/` |
| SDPA/FMHA-style attention kernel | `python/cudnn/sdpa/<direction>/` |
| SDPA/FMHA two-kernel backward with DQ plus DK/DV subkernels | `python/cudnn/sdpa/bwd/`, preserving orchestrator and helper modules |
| Multi-component sparse attention API | `python/cudnn/native_sparse_attention/<component>/` |

## API Conventions

Follow the closest template instead of inventing a new lifecycle.

- Class API:
  - Extend `cudnn.api_base.APIBase`.
  - Capture sample tensor metadata with `_make_tensor_desc(...)`.
  - Implement `check_support()` for dtype, shape, stride, tiling, architecture, and unsupported combinations.
  - For fused kernels, validate epilogue-specific constraints: activation mode, auxiliary inputs/outputs, output shape/stride contract, dtype compatibility, and unsupported activation/config combinations.
  - For dense blockscaled epilogue kernels, model every public output explicitly. Common contracts include `C` plus `AB12`, split `D1/D2`, `amax`, `SFC`, `SFD1/SFD2`, or probability inputs such as `prob_tensor`.
  - Reject or document stubbed auxiliary-output paths; do not silently advertise an output that the source kernel only prints as unimplemented.
  - For grouped kernels, distinguish `padded_offsets` contiguous-grouped metadata from per-group metadata arrays. Validate fixed padding, monotonic offset progression, total valid token count, and any scheduler workspace requirements.
  - For discrete grouped kernels, preserve the pointer-array contract. Validate B/SFB pointer arrays as int64 device metadata, keep descriptor workspace initialization explicit, and do not normalize them into a dense B tensor.
  - For MoE kernels, keep shared helper, scheduler, extension, and metadata modules factored as internal package files. Treat `padded_offsets`, `MoESchedulerParams`, `MoEPersistentTileScheduler`, `helper_kernel`, and weight mode as API/codegen-sensitive state.
  - For SDPA/FMHA kernels, preserve orchestrator/helper topology. For example, a two-kernel backward source may have one public orchestrator that launches DQ and DK/DV subkernels plus internal `fmha_utils.py` and `utils.py` helpers. Validate the head-dimension constraint, Q/K/V/O/dO/LSE/dQ/dK/dV tensor contracts, varlen `cum_seqlen_q/k` and max-sequence arguments, `scale_softmax`, causal/window mask semantics, split-head mode, CLC dynamic scheduler mode, workspace shape/layout, and two-kernel execution order.
  - For distributed kernels, validate and document `torch.distributed` state, world size, barrier flags, all-reduce mode, and symmetric-memory requirements separately from `cutedsl`.
  - Implement `compile()` for the CuTeDSL kernel compile path.
  - Implement `execute(...)` for preallocated runtime inputs/outputs and stream handling.
- Wrapper API:
  - Use a Pythonic function named like `<operation>_wrapper...`.
  - Allocate output tensors for common use.
  - Reuse the existing template's cache strategy when applicable.
  - Return `cudnn.api_base.TupleDict` so callers can use both key access and tuple unpacking.
  - Include auxiliary outputs in `TupleDict` with stable names and tuple order. Validate coupled optional tensors as a set, for example `bias`/`dbias`, `d_col`, `dprob`, `amax`, `sfd_row`, `sfd_col`, `linear_offset`, and workspace-backed outputs.
  - For SDPA backward wrappers, return gradients with stable keys such as `dq_tensor`, `dk_tensor`, and `dv_tensor`; make wrapper-owned workspace allocation, zeroing, and reuse explicit.
  - For paired forward/backward kernels, expose sibling APIs with shared internal helpers instead of merging incompatible public contracts.
- `__init__.py`:
  - Export the public class and wrapper.
  - Keep `__all__` complete and explicit.

Use existing helpers from `api_base.py`, `datatypes.py`, and family utility modules before adding new helpers.

## Public Exports

Add lazy top-level exports in `python/cudnn/__init__.py` for public APIs intended to be imported as `from cudnn import ...`.

Expose both the class API and the high-level wrapper by default when both are implemented and documented. If an API is intentionally wrapper-only or class-only, document that choice in the FE OSS API page and keep `__all__` aligned with the public surface. For nested families, re-export through the operation and family `__init__.py` files whenever callers are expected to import from that family namespace.

Use the existing `_LAZY_OPTIONAL_IMPORTS` table rather than eager optional imports. Add entries that route public names through the family module when one exists:

```python
_LAZY_OPTIONAL_IMPORTS = {
    "PublicApiClass": (".family_module", "PublicApiClass"),
    "public_wrapper": (".family_module", "public_wrapper"),
}
```

The loader already formats optional dependency failures as:

```python
raise ImportError(f"{name} requires optional dependencies. {_OPTIONAL_DEPENDENCY_INSTALL_HINT}: {e}") from e
```

For family modules such as `grouped_gemm`, `discrete_grouped_gemm`, or `sdpa`, also update the family `__init__.py` if the class or wrapper should be available from that namespace.

## Dependencies

`pyproject.toml` already defines `[project.optional-dependencies].cutedsl` for CuTeDSL integrations. Reuse it by default.

Only change dependencies when the new kernel requires a package not already covered by `cutedsl`. If adding one, document why it is needed and keep the dependency scoped to the optional extra.

Distributed kernels may require runtime setup beyond package dependencies, such as initialized `torch.distributed`, world-size constraints, or symmetric-memory support. Document those as environment requirements and test skip reasons, not as unconditional import requirements.

## Documentation

Add an FE OSS API doc page for user-facing APIs.

Common locations:

- GEMM fusions: `docs/fe-oss-apis/gemm_fusions/<operation>.md`
- Attention-specific pages: `docs/fe-oss-apis/attention/<operation>.md`
- Other frontend-only APIs: `docs/fe-oss-apis/<operation>.md`

Update `docs/fe-oss-apis/overview.md` with a link to the new page or to the broader operation anchor used by nearby docs. If the operation is summarized from a broader operation page such as `docs/operations/Attention.md`, update that operation page and keep the FE OSS page and operation summary in sync.

The doc page should cover:

- A compact operator definition or equation for the fused computation.
- Source provenance when it is useful for maintainers: upstream URL, source commit, and which original files were integrated.
- Experimental status if matching existing FE OSS APIs.
- Install command when optional CuTeDSL dependencies are required.
- High-level wrapper usage.
- Class API lifecycle when exposed.
- Epilogue semantics, activation behavior, and any auxiliary outputs.
- Discrete pointer-array arguments, descriptor workspace layout, and how the API differs from dense grouped GEMM when applicable.
- Grouped metadata semantics such as `padded_offsets`, per-group shape/stride/address arrays, scheduler mode, and helper-kernel requirements.
- Distributed runtime requirements and unsupported local/single-process paths when applicable.
- SDPA/FMHA mask mode, varlen/cumulative-sequence semantics, split-head mode, scheduler mode, workspace ownership, and D=256 or other head-dimension constraints when applicable.
- Input/output tensor names, shapes, strides, dtypes, and stream argument.
- Supported architecture and known constraints.

## Tests

Add focused pytest coverage under `test/python/fe_api/`.

Typical files:

- `test/python/fe_api/test_<operation>.py`
- `test/python/fe_api/test_<operation>_utils.py` for reusable test helpers or shape/reference utilities.
- A family subdirectory when matching existing structure, such as `test/python/fe_api/nsa/`.

Coverage should include:

- Import and wrapper smoke coverage.
- `check_support()` success and failure cases for key dtype, shape, stride, and config constraints.
- For fused kernels, a matrix that separates supported epilogue paths, unsupported epilogue/config paths, output-shape invariants, reference comparisons, and environment-based skip reasons.
- For blockscaled epilogue kernels, one reference check per public output, including `amax` and auxiliary scale-factor tensors; include nontrivial `alpha`/`beta`, `prob_tensor`, and epilogue cases that affect auxiliary values.
- For grouped kernels, cover `padded_offsets`, per-group metadata, helper/scheduler modes, fixed padding, and workspace requirements.
- For discrete grouped kernels, cover B/SFB pointer-array creation, descriptor workspace initialization, CUDA-graph padding or `permuted_m`, discrete-vs-contiguous column SFD, and bias/dbias variants.
- For MoE kernels, cover forward quant, GLU/bias, dGLU/dbias, and wgrad-style output contracts separately when source files exist.
- For SDPA/FMHA kernels, cover forward and backward outputs separately, including LSE/scaled-LSE behavior, dQ/dK/dV reference comparisons, varlen and non-varlen modes, causal/no-mask/window masks, split-head mode, CLC dynamic scheduler mode, workspace zeroing/reuse, and unsupported head dimensions.
- For distributed kernels, cover world-size gating, initialized-distributed preconditions, all-reduce modes, barrier flags, and symmetric-memory skip reasons.
- Numerical comparison against PyTorch or an existing reference implementation when executable.
- Skip behavior when optional CuTeDSL dependencies, CUDA, GPU architecture, or dtype support are unavailable.

Use existing test markers and skip patterns from nearby FE API tests.

## Completion Checklist

- Operation package and public exports are present.
- Optional dependencies are reused or intentionally updated.
- Missing original source files are reported explicitly.
- Source helper/scheduler modules are preserved as internal files when needed.
- FE OSS docs and overview links are updated.
- Tests cover support checks and executable correctness paths.
- Focused tests or explicit environment-limited verification evidence are reported.
