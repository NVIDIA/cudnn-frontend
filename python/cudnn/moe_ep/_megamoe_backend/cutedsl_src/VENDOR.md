# Vendoring record: cutedsl_megamoe

This file records provenance and synchronization state for the CuTeDSL
MegaMoE source snapshot. Runtime behavior and integration details are
documented in the parent backend `README.md`.

## Upstream

- **Project**: `cutedsl_megamoe` (NVIDIA-internal repository; URL omitted).
- **Source tree**: `cutedsl_megamoe/next/sources`.
- **Current synchronized commit**:
  `5b89819cb16069dfe20a1a0ba0778d35cb428352`.
- **Earlier import points**:
  - base forward:
    `882c83e2ce4086c3cd4211fc5a2296143c5e2aea`;
  - selected forward updates and backward dGLU:
    `92dd334af2eeedb36087834354b58ace08e880c6`.
- **Last synced**: 2026-08-28. Earlier imports occurred on 2026-08-11,
  2026-08-17, 2026-08-20, and 2026-08-24.
- **Vendored subset**: the recursive Python import closure required by Rubin
  SM107 training MegaMoE forward GLU, optional forward MXFP8 column
  requantization, and backward dGLU.

Complete kernel products, runners, tests, repository scaffolding, and
unrelated Blackwell and Rubin inference sources are excluded. Three
architecture-neutral Blackwell donor modules are retained because the Rubin
`topk_reduce.py` and `tmem_transpose.py` source-copy shims import them.

## Policy

- Vendored Python source bodies track the corresponding upstream paths at the
  synchronized commit.
- Repository-required copyright and BSD-3-Clause SPDX headers may be added
  where the upstream snapshot did not carry them.
- Integration behavior belongs in the parent `_megamoe_backend` package, not
  in the vendored source bodies.
- Local kernel fixes should go upstream first and then be synchronized here.
  Any unavoidable local source difference must be listed below.
- Snapshot updates must preserve the minimal recursive import closure and
  verify source-body equality while ignoring repository-added header lines.

The synchronized Python sources use BSD-3-Clause SPDX identifiers.
`LICENSE.Apache-2.0` is retained as historical snapshot metadata.

## Local differences from upstream

- `kernel_src/rubin/training/__init__.py` is reduced to a package marker. This
  avoids importing the unused traditional-wgrad product.
- Repository-required copyright and BSD-3-Clause SPDX headers are added to
  source files that lacked explicit headers.

No other vendored Python source-body differences are expected.

## Integration boundary

Public API validation, symmetric-workspace ownership, overflow reporting,
input and weight staging, CUDA Graph handling, dprob materialization, and
grouped-WGrad layout conversion live in the parent `_megamoe_backend`
package.

The vendored Rubin sources require a CUTLASS DSL distribution that provides
`cutlass.utils.rubin_helpers`. The executable backend enforces
`nvidia-cutlass-dsl>=4.8.0` before importing these kernels.

## Consumers

- `_megamoe_backend/mxfp8/_compile.py`: Rubin MXFP8 forward preparation and
  compilation.
- `_megamoe_backend/mxfp8/_backward_compile.py`: Rubin MXFP8 backward dGLU
  preparation and compilation.
