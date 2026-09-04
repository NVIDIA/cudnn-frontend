# Provenance of the frozen CAKE kernel bodies

These files are byte-for-byte copies of the CAKE-generated KDA training kernels
as exported into FlashInfer (`flashinfer-ai/flashinfer`, `csrc/kda/`), taken at
commit `dbc4741` (2026-09-04), the last commit before FlashInfer removed its
training family. They were introduced by FlashInfer PRs #4636 and #4726 and are
licensed Apache-2.0 by their upstream headers.

| file | upstream path | kernels |
|---|---|---|
| `flashkda_training_c16.cu` | `csrc/kda/flashkda_training_c16.cu` | `kernel_flashkda_forward_checkpoint_c16`, `kernel_flashkda_backward_persistent_c16` (partial-tail variants) |
| `cake_flashkda_training_c16_aligned_forward_36075669f2.cu` | `csrc/kda/cake_aligned_training_export/...` | `kernel_cake_flashkda_forward_checkpoint_c16_aligned` |
| `cake_flashkda_training_c16_aligned_backward_0f8187e742.cu` | `csrc/kda/cake_aligned_training_export/...` | `kernel_cake_flashkda_backward_persistent_c16_aligned` |
| `cake_flashkda_training_c16_aligned_param_reduce_be120a1e72.cu` | `csrc/kda/cake_aligned_training_export/...` | `kernel_cake_flashkda_backward_param_reduce_c16_aligned` |
| `flashkda_training_aux.cu` | `csrc/kda/flashkda_training_aux.cu` | `kernel_flashkda_refine_forgetting_horizons`, `kernel_flashkda_backward_param_reduce_c16_partial`, `kernel_flashkda_grouped_qk_expand`, `kernel_flashkda_grouped_qk_reduce` |

`fe_cake_kda_helpers.cu` is first-party (not generated).

Do not edit the generated bodies; `SHA256SUMS` pins them and the engine's cubin
cache is keyed by their digest. Regeneration flows through the CAKE pipeline.
The C32 tensor-tape and row-split routes of the same export are not vendored;
the engine serves the C16 route and declines the rest to `kda_frost`.
