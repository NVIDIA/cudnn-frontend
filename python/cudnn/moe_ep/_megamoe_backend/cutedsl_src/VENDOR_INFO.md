# Vendored CuTeDSL MegaMoE sources

## Provenance

- Source project: `cutedsl_megamoe`
- Source tree: `cutedsl_megamoe/next/sources`
- Base forward upstream revision:
  `882c83e2ce4086c3cd4211fc5a2296143c5e2aea`
- Selected forward updates and backward dGLU upstream revision:
  `92dd334af2eeedb36087834354b58ace08e880c6`
- Latest synchronized upstream revision:
  `5a43c8523ea5215923c2fc8d0abae75bd6762011` (merge of source revision
  `dc05bbdf38350a0eb67e9d9440e3c7c0e21e99fc`)
- Vendoring dates: 2026-08-11 (base), 2026-08-17 (selected updates), and
  2026-08-20 and 2026-08-24 (latest synchronizations).
- On 2026-08-24 every vendored Python source except the intentionally minimal
  `kernel_src/rubin/training/__init__.py` was synchronized byte-for-byte with
  the revision above. Other integration-specific behavior lives outside this
  directory.
- License: synchronized Python sources retain their upstream BSD-3-Clause
  SPDX identifiers. `LICENSE.Apache-2.0` remains as historical snapshot
  metadata.

The source repository URL is intentionally omitted because it is an internal
development location. The revisions above identify the upstream baselines;
the manifest and local-modification notes below describe the packaged snapshot.

## Scope

This directory contains the recursive Python import closures for these Rubin
SM107 products:

- training MegaMoE forward GLU;
- optional forward MXFP8 column requantization (disabled by default);
- training MegaMoE backward dGLU.

It preserves the `next/sources` package hierarchy and includes the shared API,
quantization, workspace, synchronization, NVLink token communication,
schedulers, TopK reduction, and Rubin helper modules required by those roots.
The Rubin training initializer remains a minimal package marker so importing
the MegaMoE products does not pull in the unused traditional-wgrad product.
The public backend exposes the backward dGLU product through a restricted Rubin
MXFP8 dgrad/dprob path. Unsupported formats and semantics retain explicit
capability gates; see the backend README.

Complete Blackwell kernel products, runners, tests, and repository-only tooling
are excluded. Three architecture-neutral Blackwell donor modules remain because
the upstream Rubin source-copy shims import them. Imports of external CUTLASS
utility modules remain because they are CUTLASS helpers, not vendored kernel
support.

## Manifest

```text
LICENSE.Apache-2.0
VENDOR_INFO.md
__init__.py
api.py
communication/__init__.py
communication/nvlink_domain/__init__.py
communication/nvlink_domain/symmetric_buffer.py
communication/nvlink_domain/token_comm.py
communication/nvlink_domain/token_comm_deterministic.py
communication/token_protocol.py
helpers/__init__.py
helpers/constants.py
helpers/cute_py_helpers.py
helpers/device_workspace.py
helpers/dsl_helpers.py
helpers/flag_batch.py
helpers/iket_compat.py
helpers/ptx_helpers.py
helpers/software_sync.py
helpers/smem_workspace.py
helpers/utils.py
kernel_src/__init__.py
kernel_src/function_mapping.py
kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py
kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_extension.py
kernel_src/blackwell/inference/mega/topk_reduce.py
kernel_src/rubin/__init__.py
kernel_src/rubin/training/__init__.py
kernel_src/rubin/training/mega/__init__.py
kernel_src/rubin/training/mega/bwd_dglu/__init__.py
kernel_src/rubin/training/mega/bwd_dglu/dglu_mxfp8_fc12_epilogue.py
kernel_src/rubin/training/mega/bwd_dglu/dglu_mxfp8_fc12_extension.py
kernel_src/rubin/training/mega/bwd_dglu/dglu_mxfp8_fc12_kernel.py
kernel_src/rubin/training/mega/bwd_dglu/dglu_mxfp8_mega_moe_kernel.py
kernel_src/rubin/training/mega/fwd_glu/__init__.py
kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_col_requant.py
kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_fc12_epilogue.py
kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_fc12_extension.py
kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_fc12_kernel.py
kernel_src/rubin/training/mega/fwd_glu/glu_mxfp8_mega_moe_kernel.py
kernel_src/rubin/training/mega/helpers/__init__.py
kernel_src/rubin/training/mega/helpers/constants.py
kernel_src/rubin/training/mega/helpers/utils.py
kernel_src/rubin/training/mega/tmem_transpose.py
kernel_src/rubin/training/mega/topk_reduce.py
kernel_src/schedulers/__init__.py
kernel_src/schedulers/base.py
kernel_src/schedulers/fc12_mapping.py
kernel_src/schedulers/fc12_scheduler.py
kernel_src/schedulers/non_clc_mixed_cga.py
kernel_src/schedulers/work_id_claim.py
quant_def.py
```

## Integration boundary

- Every `.py` file listed in the manifest except
  `kernel_src/rubin/training/__init__.py` is a byte-for-byte copy of the same
  relative path at revision `5a43c8523ea5215923c2fc8d0abae75bd6762011`.
- `kernel_src/rubin/training/__init__.py` is intentionally reduced to a
  package marker. This avoids vendoring and eagerly importing the unused Rubin
  traditional-wgrad product.
- `helpers/software_sync.py` replaces the former integration-only
  `communication/nvlink_domain/software_sync.py` path.
- The upstream Rubin `topk_reduce.py` and `tmem_transpose.py` source-copy shims
  require the three architecture-neutral Blackwell donor modules in the
  manifest. Unrelated Blackwell and Rubin inference products remain excluded.
- Public API validation, symmetric-workspace ownership, overflow reporting,
  staging, CUDA Graph handling, dprob materialization, and grouped-wgrad layout
  conversion live in the parent `_megamoe_backend` package.
- The synchronized Rubin sources require a CUTLASS DSL distribution that
  provides `cutlass.utils.rubin_helpers`.

## Updating the snapshot

1. Review source changes from the revision above.
2. Recompute recursive relative imports from all Rubin SM107 product roots,
   exact package initializers, and shared product entry points.
3. Include only the exact donor and eager-import dependencies required by that
   closure; do not add unrelated products.
4. Copy every selected Python source without modification and remove paths no
   longer present in the selected upstream closure.
5. Update the revision, date, manifest, and integration-boundary notes.
6. Assert byte equality for every vendored Python source, then run compileall,
   relative-import closure validation, package import smoke tests, and the
   MoeEP regression suite.
