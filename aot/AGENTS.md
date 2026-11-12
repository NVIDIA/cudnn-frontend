# Ahead-of-time kernels

The optional AOT library exports kernels built from the current FE checkout. Keep its default build opt-in.

- Reusing an exported object/header pair requires a matching fingerprint of transitive source and compiler inputs,
  plus both artifact hashes. File existence alone can silently ship a stale kernel after a source edit.
  Detector: `python -m unittest discover -s aot/attention/test`.
- Bound export-worker memory with fresh batches. Do not rely on `ProcessPoolExecutor.max_tasks_per_child`:
  it is absent on Python 3.10 and worker replacement can hang on affected later interpreters. Exercise more than
  one batch in offline build validation.
- Plan creation must load every launchable function, including reductions. Capture the **first** execute in a fresh
  process (`test_aot_attention --only graph` with lazy loading enabled); a warmup hides deferred loading.
- Workspace queries must work before tensor pointers are bound. Exercise valid misaligned views with exactly the
  queried capacity, and reject misaligned workspace before any launch (`test_aot_attention --only contracts`).
- The optional `cudnn_frontend_aot_loader` target carries platform loader link requirements. Keep those off the
  original `cudnn_frontend` header-only target. A plan must survive the lifetime of a supplied `Library` wrapper.
