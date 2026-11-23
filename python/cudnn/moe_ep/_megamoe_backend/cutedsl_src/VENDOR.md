# Vendoring record: cutedsl_megamoe

This file records provenance and synchronization state for the CuTeDSL
MegaMoE source snapshot. Runtime behavior and integration details are
documented in the parent backend `README.md`.

## Upstream

- **Project**: `cutedsl_megamoe` (NVIDIA-internal repository; URL omitted).
- **Source tree**: `sources` at the repository root (formerly
  `next/sources`).
- **Current synchronized commit**:
  `1fd8c51fd64640da21c8cc45e95c19d0de35da19`.
- **Discrete-weight implementation ancestor**:
  `7cc8d2eb2fb2fc9643ccd6244d6c810ed9f4341b`.
- **Last synced**: 2026-11-18; the WGrad closure was added on 2026-11-19 and
  the Blackwell training kernels on 2026-11-22, both at the same commit.
- **Vendored subset**: the exact export closure produced for
  `RubinTrainingFwdGluMegaMoE`, `RubinTrainingBwdDgluMegaMoE`,
  `RubinTrainingWgrad`, `BlackwellTrainingFwdGluMegaMoE`, and
  `BlackwellTrainingBwdDgluMegaMoE`, including the materialized source copy
  `kernel_src/rubin/custom_mix_cga_helpers.py` (from
  `kernel_src/blackwell/custom_mix_cga_helpers.py`). Adding the Blackwell
  kernels to the export leaves every Rubin and shared file byte-identical to
  the Rubin-only export.
- **Raw exported source files**: 58 (plus 20 generated package markers).
- **Expected vendored Python files**: 78 after the local overlay.
- **Expected vendored Python tree SHA256**:
  `cf22bf4f46515d848f52b55d87627a59326e2b331f351b0df8d81a266e0f264e`.
  This uses the same relative-path-and-content algorithm as
  `_megamoe_backend.mxfp8._fingerprint.source_tree_sha256`.

The export command, run from the upstream repository root, is:

```bash
python export_src.py \
  --kernels RubinTrainingFwdGluMegaMoE RubinTrainingBwdDgluMegaMoE \
    RubinTrainingWgrad \
    BlackwellTrainingFwdGluMegaMoE BlackwellTrainingBwdDgluMegaMoE \
  --dst_dir <empty-directory>
```

The WGrad closure adds `helpers/cute_py_helpers.py`,
`kernel_src/schedulers/wgrad_mapping.py`,
`kernel_src/schedulers/wgrad_scheduler.py`, and the four
`kernel_src/rubin/training/traditional/wgrad/block_scaled_moe_wgrad_*.py`
modules.

## Policy

- Vendored Python source bodies track the corresponding upstream paths at the
  synchronized commit.
- Repository-required copyright and BSD-3-Clause SPDX headers may be added
  where the upstream snapshot did not carry them.
- Local kernel fixes should go upstream first and then be synchronized here.
  Any unavoidable local code difference must be listed below.
- Comments, docstrings and error-message text may be worded differently from
  upstream, for example to keep them self-contained. Such wording-only
  differences are not listed below.
- Temporary exception: the MoeEP `data_token_capacity` and row-major
  column-requantization overlays below are carried in both the Rubin and the
  Blackwell kernel copies instead of going upstream first. Once upstream
  provides equivalent options, remove both copies of these overlays and this
  exception.

The synchronized Python sources use BSD-3-Clause SPDX identifiers.
`LICENSE.Apache-2.0` is retained as historical snapshot metadata.

## Local differences from upstream

- The root `cutedsl_src/__init__.py` is a package marker rather than the
  exporter's eager kernel re-export. The integration imports concrete kernel
  implementation modules.
- The Rubin and Blackwell forward MegaMoE column-requantization outputs keep
  MoeEP's established row-major WGrad operand ABI: fake stride `(1, 0)`,
  `dst_k_major=False`, and row-major fixed-matrix validation. This preserves
  `fc1_a` as `(pool_rows, hidden)` with stride `(hidden, 1)`.
- The Blackwell forward MegaMoE kernel additionally does not pass
  `previous_expert_token_sizes` to the column requantizer. Upstream's
  retired-row clearing zeroes the capacity tail left by a larger previous
  launch and supports only K-major output; MoeEP leaves the row-major capacity
  tail unspecified, and the Rubin forward kernel has no such clearing. The
  previous-size snapshot region is still allocated and maintained, but unused.
- MoeEP supplies an optional `data_token_capacity` physical-row count
  independently from the upstream `max_recv_size_per_rank` logical route
  limit. The Rubin and Blackwell forward and backward MegaMoE kernels pass it
  to the deterministic communication component. The component defaults the
  new field to upstream's padded logical capacity for non-MoeEP consumers,
  while MoeEP always supplies its exact caller-prescribed pool rows. Upstream
  topology clamps remain intact, and the component rejects physical capacity
  below the padded requirement for the clamped logical limit.
- Repository-required copyright and BSD-3-Clause SPDX headers are added to
  generated empty package markers and to the Blackwell
  `kernel_src/blackwell/training/mega/bwd_dglu/dgrad_config.py`, which
  upstream ships without one.

No other vendored Python code differences are expected.

Except for the Blackwell retired-row clearing removal, the overlay hunks in
each Rubin and Blackwell MegaMoE kernel pair are textually identical.

The group-consistent overflow policy, previously a local difference, is
upstream as of this snapshot: every rank derives the same group-wide overflow
bit from the replicated pre-truncation destination totals, while only the
overflowing destination truncates its local expert sizes. The export closure
no longer reaches Blackwell swap-AB extension types, so the former `Any`
annotation overlay is no longer needed. `kernel_src/blackwell` holds only the
Blackwell training MegaMoE closure.

## Synchronization procedure

1. Export the five kernels listed above at the target upstream commit into an
   empty directory.
2. Reapply every local difference listed above to the Rubin and the Blackwell
   copies, and carry over the wording-only changes to comments, docstrings
   and error messages from the previously vendored tree.
3. For each forward and backward MegaMoE kernel, diff the Rubin and Blackwell
   overlay hunks against the export. They must match, except for the
   Blackwell retired-row clearing removal.
4. Run `test_vendored_mega_kernels_keep_data_token_capacity_overlay` in
   `test/python/moe_ep/test_moe_ep_forward.py`.
5. Update the upstream commit, file counts and tree SHA256 in this file.

Public `nvidia-cutlass-dsl` 4.8.0 and 4.8.1 do not provide
`PipelineUmmaAsync.create(defer_init=...)`. Upstream's forward fused FC1/FC2
kernel feature-detects that keyword at import: without it, the kernel keeps
the per-create mbarrier initialization that mixed-CGA launches already use
instead of the folded initialization. No local overlay is needed for this.

The Rubin WGrad kernel builds its TMA descriptor overrides through
`helpers/dsl_helpers.py` (`make_override_tma_atom`, `make_tma_residue_tensor`)
from the DSL's override copy ops and traits and its residue-tensor MLIR op.
Public `nvidia-cutlass-dsl` 4.8.0 and 4.8.1 provide all of them, so WGrad
compiles and runs on the public wheels.
`_cutedsl.rubin_wgrad_cutedsl_error` probes these capabilities before the WGrad
module is imported.

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
- `_megamoe_backend/mxfp8/_wgrad_compile.py`: Rubin MXFP8 WGrad preparation,
  compilation, and launch.
- `_megamoe_backend/mxfp8/_fingerprint.py` and `_training_resources.py`:
  vendored source-tree fingerprint.

The backend does not consume `kernel_src/blackwell` yet; it still selects only
the Rubin kernels.
