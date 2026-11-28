# Batch-invariant RMSNorm tests

Run the suite from `test/python` on a supported GPU (SM100, SM103 or SM107).

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/norm/batch_invariant/goldens" \
    python -m pytest -n0 -q -rs norm/batch_invariant/
```

- [Plan selection](test_bi_rmsnorm_dispatch.py) checks that the BATCH_INVARIANT note selects the engine and that unsupported graphs raise.
- [Determinism](test_bi_rmsnorm_determinism.py) covers:
  - Byte comparisons against the exact CPU model of the class, for FP32, FP16 and BF16 inputs, mixed-dtype weights, uneven lane loads, sub-warp rows, and hidden sizes whose granules are 8, 4 and 2 bytes. The model is `reference` in [bi_rmsnorm_utils.py](bi_rmsnorm_utils.py).
  - Replays, batch invariance under reordering, duplication, truncation and changed neighbours, and training-versus-inference equality.
  - Every placement of the class, including staged prefetch, at its scheduled grid and at grids small enough that each CTA walks many row groups.
  - Row groups across waves, row counts run through execute-time shape overrides (one per placement the plan can
    pick, for inference and training graphs), rejected overrides, mismatched bindings (short or wrong-dtype
    buffers), misaligned operand pointers, rank-3 inputs, kernel row strides, and known output bytes.

The [goldens](goldens/) are grouped by the test runtime hash. Each JSON file records its GPU and software versions.
