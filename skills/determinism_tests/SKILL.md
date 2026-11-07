---
name: determinism-tests
description: Add and run L0 bitwise determinism and batch-invariance tests for new or changed cuDNN Frontend numerical APIs, kernel paths, datatypes, or GPU targets, then update docs/operations/determinism.md from measured evidence.
---

# Determinism tests

Use this workflow with each new numerical API or kernel change. Finish with
forward/backward coverage, measured results, and an updated API support row in
the same PR. If execution is unavailable, add the tests and mark that GPU
unverified. A skill guides the agent; CI enforcement is separate.

Read the [root determinism rules](../../AGENTS.md#kernel-determinism-and-batch-invariance)
and [test guide](../../test/AGENTS.md). Use the
[coverage report](../../test/python/determinism_goldens/README.md) for shapes,
test methods, golden handling, and existing gaps.

## Identify the contract

- List the public API, forward/backward entry points, supported datatypes and
  GPUs, deterministic mode, and selected backend. State unsupported directions.
- List every output, saved statistic, input/parameter/state gradient, and
  mutable buffer. Include FP32 intermediates even when output uses FP16/BF16.
- Identify the batch axis, packed offsets, sparse indices, shared reductions,
  quantization scales, and upstream gradients that must stay fixed.
- Confirm the checkout under test with `import cudnn; print(cudnn.__file__)`.
  An editable install can resolve to another checkout despite `PYTHONPATH`.

## Extend the closest suite

Place `test_<op>_determinism.py` beside the operation's tests. Mark every case
L0, including boundary and stress cases. Keep parameter IDs explicit about
direction, dtype, and shape. Reuse
[determinism_utils.py](../../test/python/determinism_utils.py) for Torch byte
comparisons, snapshots, replay, and goldens. Keep operation-specific invocation
and batch packing in the suite.

Choose the nearest example; do not copy its temporary skips into a new API:

| Operation contract | Example |
| --- | --- |
| Recurrent output/state and autograd | [Linear attention](../../test/python/linear_attention/determinism.py) |
| Packed sparse attention and saved statistics | [QSA](../../test/python/QSA/cutedsl/test_qsa_determinism.py) |
| Sparse indices, shared sink gradient, empty-row sentinels | [DSA attention](../../test/python/deepseek_sparse_attention/cutedsl/test_DSA_determinism.py) |
| Mutable scores and sparse/dense gradient reductions | [DSA indexer](../../test/python/deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py) |
| Fixed quantized payloads and repacked scales | [FP8/MXFP8 indexer](../../test/python/deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py) |

Cover these checks in each supported direction:

- Replay fixed inputs with fresh buffers and concurrent GPU work. Snapshot
  results before the next launch. Check names, shapes, dtypes, and raw bytes.
- Compare a sequence alone, packed with neighbors, reordered, and duplicated.
  Preserve its exact data, upstream gradients, and loss scale. Change neighbor
  data separately to check isolation. Remap offsets and sparse IDs correctly.
- Include tile/dispatch boundaries, ragged lengths, empty inputs where valid,
  grouped reductions, and enough work to cross scheduling boundaries. Exercise
  each supported precision, including its scale format and gradient dtype.
- Keep an independent numerical reference separate from byte comparisons.
  Reject NaNs; allow infinity sentinels only where the API declares them and
  check their sign and location. Check unintended input mutation.
- Add a small golden case per direction and distinct numerical path. Require
  replay and the reference to pass before recording its hashes. A small golden
  does not establish determinism for larger reductions.

Use the API's deterministic mode when provided. Pin the intended backend;
do not let a missing plan silently select another provider. Quantize each
sequence once and preserve its payload and logical scales across batch changes.

## Run and record

Run the new or changed suites on each supported GPU available to the task.
Record unavailable GPUs as unverified. Confirm the runtime and import path on
each host; do not copy outcomes between devices or software versions.

From the repo root, replace the example suite path below with the actual path:

```bash
cd test/python
determinism_suite=path/to/test_new_op_determinism.py
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest "$determinism_suite" -m L0 --collect-only -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest "$determinism_suite" -m L0 -q -rs
```

Use the [golden workflow](../../test/python/determinism_goldens/README.md#golden-checks)
to prepare a missing baseline before the strict check above. Record into a new
directory, inspect input/output hash changes and reference results, then commit
the reviewed JSON files. Normal checks must leave `CUDNN_DETERMINISM_RECORD`
unset. A missing baseline must not become a CI pass.

Record GPU, runtime, source revision, exact command, expanded case counts, and
failures/skips in the coverage report. Preserve earlier measurements as dated
records. A failed case stops at its first mismatch; later comparisons have no
result. Do not replace raw-byte equality with a tolerance.

Retain existing `skip until fix` cases and their diagnostics. Add a temporary
skip only when explicitly requested, scoped to the known failing parameters,
with a defect reason and removal condition. Remove it with the fix and rerun
the affected cases. Never infer a fix from an intermittent pass.

## Update the support page

Update [docs/operations/determinism.md](../../docs/operations/determinism.md)
in the same change, even when a new API can only be marked unverified. Add or
revise rows by API, direction, and datatype; state required modes and limits.
Keep the current GPU headers, including the combined `SM100/103` column.

| Status | Required evidence for the documented configurations |
| --- | --- |
| Supported | Repeatability and batch invariance verified |
| Replay only | Repeatability verified; batch invariance has a known failure |
| Not guaranteed | Repeatability has a known failure |
| Unverified | Insufficient execution evidence on this GPU |
| Unsupported | The API precision path is unavailable by contract |

Require evidence for both GPUs before marking `SM100/103` supported. If their
evidence differs, state the GPU-specific limit within the existing column;
do not add a separate SM103 status column. Missing hardware or dependencies
are verification gaps, not proof that an API path is unsupported.

Do not upgrade a status because failing cases were skipped or a golden passed.
Keep test counts, commands, runtime logs, and failure diagnostics in the test
report, outside the support page. Run `fern check` from `docs/fern` after a
status-page edit, and run pre-commit on changed files.

Before finishing, report the added coverage, actual pass/fail/skip counts,
unmeasured GPUs, and changed support rows in the PR description. Identify
which results are new and which are retained from earlier runs.
