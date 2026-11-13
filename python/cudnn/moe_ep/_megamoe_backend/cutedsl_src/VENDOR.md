# Vendoring record: cutedsl_megamoe

This file records provenance and synchronization state for the CuTeDSL
MegaMoE source snapshot. Runtime behavior and integration details are
documented in the parent backend `README.md`.

## Upstream

- **Project**: `cutedsl_megamoe` (NVIDIA-internal repository; URL omitted).
- **Source tree**: `sources` at the repository root (formerly
  `next/sources`).
- **Current synchronized commit**:
  `72796e96cfdd4690b6af583302a3258bada40f27`.
- **Discrete-weight implementation ancestor**:
  `7cc8d2eb2fb2fc9643ccd6244d6c810ed9f4341b`.
- **Last synced**: 2026-11-11.
- **Vendored subset**: the exact export closure produced for
  `RubinTrainingFwdGluMegaMoE` and `RubinTrainingBwdDgluMegaMoE`, including
  the materialized source copy
  `kernel_src/rubin/custom_mix_cga_helpers.py` (from
  `kernel_src/blackwell/custom_mix_cga_helpers.py`).
- **Raw exported source files**: 38 (plus 12 generated package markers).
- **Expected vendored Python files**: 50 after the local overlay.
- **Expected vendored Python tree SHA256**:
  `eb883900803f459a3fa55c620d9235396d1d03d532f73a759957b767dfba7146`.
  This uses the same relative-path-and-content algorithm as
  `_megamoe_backend.mxfp8._fingerprint.source_tree_sha256`.

The export command, run from the upstream repository root, is:

```bash
python export_src.py \
  --kernels RubinTrainingFwdGluMegaMoE RubinTrainingBwdDgluMegaMoE \
  --dst_dir <empty-directory>
```

## Policy

- Vendored Python source bodies track the corresponding upstream paths at the
  synchronized commit.
- Repository-required copyright and BSD-3-Clause SPDX headers may be added
  where the upstream snapshot did not carry them.
- Local kernel fixes should go upstream first and then be synchronized here.
  Any unavoidable local source difference must be listed below.

The synchronized Python sources use BSD-3-Clause SPDX identifiers.
`LICENSE.Apache-2.0` is retained as historical snapshot metadata.

## Local differences from upstream

- The root `cutedsl_src/__init__.py` is a package marker rather than the
  exporter's eager kernel re-export. The integration imports concrete kernel
  implementation modules.
- The forward MegaMoE column-requantization output keeps MoeEP's established
  row-major WGrad operand ABI: fake stride `(1, 0)`, `dst_k_major=False`, and
  row-major fixed-matrix validation. This preserves `fc1_a` as
  `(pool_rows, hidden)` with stride `(hidden, 1)`.
- MoeEP supplies an optional `data_token_capacity` physical-row count
  independently from the upstream `max_recv_size_per_rank` logical route
  limit. The deterministic communication component defaults the new field to
  upstream's padded logical capacity for non-MoeEP consumers, while MoeEP
  always supplies its exact caller-prescribed pool rows. Upstream topology
  clamps remain intact, and the component rejects physical capacity below the
  padded requirement for the clamped logical limit.
- The forward epilogue module docstring that referenced unavailable external
  design material is replaced with a self-contained description.
- Code blocks that the vendored training closure never reaches are removed
  from `helpers/dsl_helpers.py`, `helpers/ptx_helpers.py`, and
  `kernel_src/rubin/custom_mix_cga_helpers.py`.
- One comment in `kernel_src/schedulers/non_clc_mixed_cga.py` that named a
  class outside the export closure is shortened.
- Repository-required copyright and BSD-3-Clause SPDX headers are added to
  generated empty package markers.

No other vendored Python source-body differences are expected.

The group-consistent overflow policy, previously a local difference, is
upstream as of this snapshot: every rank derives the same group-wide overflow
bit from the replicated pre-truncation destination totals, while only the
overflowing destination truncates its local expert sizes. The export closure
no longer reaches Blackwell swap-AB extension types, so the former `Any`
annotation overlay and the omitted `kernel_src/blackwell` package tree are no
longer needed.

Public `nvidia-cutlass-dsl` 4.8.0 and 4.8.1 do not provide
`PipelineUmmaAsync.create(defer_init=...)`. Upstream's forward fused FC1/FC2
kernel feature-detects that keyword at import: without it, the kernel keeps
the per-create mbarrier initialization that mixed-CGA launches already use
instead of the folded initialization. No local overlay is needed for this.

## Integration boundary

Public API validation, symmetric-workspace ownership, overflow reporting,
input and weight staging, CUDA Graph handling, dprob materialization, and
grouped-WGrad layout conversion live in the parent `_megamoe_backend`
package.

Discrete mode passes four caller-owned CUDA `int64[E_local]` pointer arrays
directly to the vendored forward/backward kernels. MoeEP owns neither the
per-expert buffers nor the pointer tables.

The vendored routing metadata record stores the source token index in 16 bits,
so the communication component rejects `max_tokens_per_rank > 65536`. The
MegaMoE capability check declines such configurations before compilation.

The vendored Rubin sources require a CUTLASS DSL distribution that provides
`cutlass.utils.rubin_helpers`. The executable backend enforces
`nvidia-cutlass-dsl>=4.8.0` before importing these kernels.

## Consumers

- `_megamoe_backend/mxfp8/_compile.py`: Rubin MXFP8 forward preparation and
  compilation.
- `_megamoe_backend/mxfp8/_backward_compile.py`: Rubin MXFP8 backward dGLU
  preparation and compilation.
- `_megamoe_backend/_comm.py`: symmetric-buffer peer mapping.
- `_megamoe_backend/mxfp8/_training_stage_kernel.py`: MXFP8 staging constants
  and PTX conversion helpers.
- `_megamoe_backend/mxfp8/_fingerprint.py` and `_training_resources.py`:
  vendored source-tree fingerprint.
