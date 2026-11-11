# Vendoring record: cutedsl_megamoe

This file records provenance and synchronization state for the CuTeDSL
MegaMoE source snapshot. Runtime behavior and integration details are
documented in the parent backend `README.md`.

## Upstream

- **Project**: `cutedsl_megamoe` (NVIDIA-internal repository; URL omitted).
- **Current synchronized commit**:
  `e8df888670a44b099e9d7009d5338a4ab45bf848`.
- **Discrete-weight implementation ancestor**:
  `7cc8d2eb2fb2fc9643ccd6244d6c810ed9f4341b`.
- **Last synced**: 2026-09-22.
- **Vendored subset**: the exact export closure produced for
  `RubinTrainingFwdGluMegaMoE` and `RubinTrainingBwdDgluMegaMoE`, including
  materialized source-copy modules.
- **Raw exported source files**: 37 (plus 15 generated package markers).
- **Expected vendored Python files**: 48 after the local overlay.
- **Expected vendored Python tree SHA256**:
  `5c000a4c2824b71ea669415a35c27f0ac678a076b6b3d4f586595d6b2d1796d4`.
  This uses the same relative-path-and-content algorithm as
  `_megamoe_backend.mxfp8._fingerprint.source_tree_sha256`.

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
- Overflow policy is group-consistent without a post-kernel collective:
  every rank derives the same group-wide overflow bit from the replicated
  pre-truncation destination totals, while only the overflowing destination
  truncates its local expert sizes.
- The materialized Rubin training TMEM helper replaces two unused Blackwell
  swap-AB extension annotations with `Any`; the extension and its now-empty
  local `kernel_src/blackwell` package tree are omitted from the vendored
  closure.
- Forward epilogue comments that referenced unavailable external design
  material are replaced with self-contained dataflow descriptions.
- Repository-required copyright and BSD-3-Clause SPDX headers are added to
  generated empty package markers.

No other vendored Python source-body differences are expected.

## Integration boundary

Public API validation, symmetric-workspace ownership, overflow reporting,
input and weight staging, CUDA Graph handling, dprob materialization, and
grouped-WGrad layout conversion live in the parent `_megamoe_backend`
package.

Discrete mode passes four caller-owned CUDA `int64[E_local]` pointer arrays
directly to the vendored forward/backward kernels. MoeEP owns neither the
per-expert buffers nor the pointer tables.

The vendored Rubin sources require a CUTLASS DSL distribution that provides
`cutlass.utils.rubin_helpers`. The executable backend enforces
`nvidia-cutlass-dsl>=4.8.0` before importing these kernels.

## Consumers

- `_megamoe_backend/mxfp8/_compile.py`: Rubin MXFP8 forward preparation and
  compilation.
- `_megamoe_backend/mxfp8/_backward_compile.py`: Rubin MXFP8 backward dGLU
  preparation and compilation.
