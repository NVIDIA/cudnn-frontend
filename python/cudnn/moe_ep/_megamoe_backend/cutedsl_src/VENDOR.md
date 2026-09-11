# Vendoring record: cutedsl_megamoe

This file records provenance and synchronization state for the CuTeDSL
MegaMoE source snapshot. Runtime behavior and integration details are
documented in the parent backend `README.md`.

## Upstream

- **Project**: `cutedsl_megamoe` (NVIDIA-internal repository; URL omitted).
- **Source tree**: `cutedsl_megamoe/next/sources`.
- **Current synchronized commit**:
  `aa173b4af2dc859e86de61e51d34a69c7f8cafe6`.
- **Discrete-weight implementation**:
  `7cc8d2eb2fb2fc9643ccd6244d6c810ed9f4341b`.
- **Last synced**: 2026-09-11.
- **Vendored subset**: the exact export closure produced for
  `RubinTrainingFwdGluMegaMoE` and `RubinTrainingBwdDgluMegaMoE`, including
  materialized source-copy modules.

The export command is:

```bash
python next/export_src.py \
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
