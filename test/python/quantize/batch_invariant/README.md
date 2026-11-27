# Batch-invariant quantization tests

Run from `test/python` on SM100, SM103 or SM107. Install FlashInfer with its batch-invariant quantizers to include the
parity cases; without it those cases skip.

```sh
CUDNN_DETERMINISM_GOLDENS="$PWD/quantize/batch_invariant/goldens" \
    python -m pytest -n0 -q -rs -m L0 quantize/batch_invariant/
```

[`test_mx_quantize.py`](test_mx_quantize.py) covers `mxfp8_quantize_forward` and `mxfp4_quantize_forward`, and
[`test_nvfp4_quantize.py`](test_nvfp4_quantize.py) covers `nvfp4_quantize_forward` per tensor and per token (the global scale
forms, the per-token scale vector and the per-tensor scale input as well). Both share the cases below:

- rows are independent of the batch (replay with concurrent GPU work, reorder and duplicate, a single row, changed neighbours) for each input dtype and scale layout;
- codes and scale bytes against an independent torch implementation of the recipe ([`quant_reference.py`](quant_reference.py));
- boundary inputs (zero and signed-zero blocks, subnormal and near-max values, rounding midpoints, a wide exponent range, outliers, and inf and NaN blocks where no independent reference exists) through the same row-independence, reference and FlashInfer checks;
- byte equality with FlashInfer's `batch_invariant=True` quantizers;
- the current stream, strided and misaligned inputs, K padding, empty batches, leading dimensions and rejected inputs;
- small [goldens](goldens/) grouped by the runtime hash. Each JSON file records its GPU and software versions.
