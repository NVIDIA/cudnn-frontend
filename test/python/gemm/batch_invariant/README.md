# Batch-invariant GEMM tests

Run the suite from `test/python` on a supported GPU. Install a FlashInfer build
with cuDNN batch-invariant support to include the rollout integration tests.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/gemm/batch_invariant/goldens" \
    python -m pytest -n0 -q -rs gemm/batch_invariant/
```

- [Plan selection](test_dispatch.py) covers the BATCH_INVARIANT numerical note, engine discovery, replay and unsupported graphs.
- [Quantized plan selection](test_quantized_dispatch.py) covers scale multiplication order, PDL eligibility and byte-buffer validation for swizzled scales.
- [BF16 matmul](test_matmul_determinism.py) covers batch invariance, pipeline stress, known output bytes and batch-invariant forward paired with native cuDNN matmul and reduction plans for the gradients.
- [Quantized matmul](test_quantized_matmul.py) exercises both FP16 and BF16 output for tensor-scaled E4M3/E5M2, groupwise E4M3, MXFP8, NVFP4, MXFP4 and mixed MXFP8 × MXFP4. It compares rollout, training and the original FlashInfer kernel byte for byte, checks independently decoded FP64 references, and runs native cuDNN gradients for the effective floating-point operands.
- [FlashInfer integration](test_flashinfer_integration.py) compares a rollout row bitwise with the corresponding training row and the original FlashInfer implementation.

The [goldens](goldens/) are grouped by the test runtime hash. Each JSON file records its GPU and software versions.
