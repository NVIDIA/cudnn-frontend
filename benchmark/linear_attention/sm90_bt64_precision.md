# SM90 BT64 KDA local-derivative precision

This change targets the BF16 KDA backward with K=V=128, FP32 log gates,
beta and state. It is independent of the FP32 state-gradient-P change in
[NVIDIA/cudnn-frontend#2482](https://github.com/NVIDIA/cudnn-frontend/pull/2482).
The baseline is upstream `7ab3ffd75681651ad5b5feb36a3e7e63341f1ef8`, including
its existing BF16 P buffer. No experimental BT16 code is included.

## Why the gate gradient loses precision

The FP32 gate gradient contains signed local contributions that can largely
cancel. Storing their coefficient matrices in BF16 discards small corrections
before that cancellation; keeping only the final sum in FP32 cannot recover
them. The original inverse-gradient evaluation also rounds two intermediate
matrix products to BF16. A longer chunk increases exposure to these effects.

For the local solve `Z = A B`, with `A = (I + M)^-1`, define the adjoint
`Y = A.T dZ`. The derivative is `dM = -strict_lower(Y Z.T)`. Reuse the
adjoint already computed for dV instead of forming the two-sided derivative
through the stored inverse. This removes redundant products and intermediate
rounding sites. The direct solve derivative is also the strategy used by the
SM100 backward; the kernels otherwise retain different chunk sizes and
factorizations.

The strict dC and dM coefficients additionally keep a BF16 residual:
`hi = BF16(x)` and `lo = BF16(x - FP32(hi))`. Local row/column contractions
accumulate the low then high BF16 Tensor Core products into FP32. This is a
two-term approximation, not full FP32 multiplication. Y, Z, the stored
inverse, other MMA operands, and final dQ/dK/dV still use BF16. No full
SM100 precision parity or universal gradient improvement is claimed.

## Memory and synchronization

Two 8 KiB residual tiles reuse the upper half of existing shared-memory region
Y after its phase-A state operands are consumed. The dC residual waits in
registers until that region is free. The original V tile is reused for saved Z
after the dBeta reduction consumes V. The dM high/residual stores precede the
barrier that releases Y/Z operand storage for the final contractions.

Caller workspace size, kernel count, host ABI, dispatch, compile-cache axes,
and dynamic shared-memory allocation are unchanged. Both local-backward
specializations retain 128 registers/thread and 230656 bytes of dynamic
shared memory. Full-tile stack usage stays at 32 bytes/thread; the tail
specialization increases from 32 to 56 bytes/thread. This small local-stack
change is separate from caller workspace.

## Reproduction

Use the same H20 and software environment for both checkouts. The benchmark
loads the baseline backward host from its checkout while importing the
candidate package. Confirm `cudnn.__file__` before running; an editable
installation pointing elsewhere invalidates the comparison.

```bash
PYTHONPATH="$PWD/python" python benchmark/linear_attention/benchmark_sm90_bt64_precision.py \
  --baseline /path/to/upstream-checkout --mode precision --output precision.json
PYTHONPATH="$PWD/python" python benchmark/linear_attention/benchmark_sm90_bt64_precision.py \
  --baseline /path/to/upstream-checkout --mode performance --output performance.json
```

Precision uses CPU-generated inputs from four seeds, thirteen layouts per
seed, and independent FP64 recurrent autograd. It reports relative-L2 and
maximum-absolute error for all six gradients, plus byte replay checks.
Performance measures preparation, reverse state scan and local backward in
one CUDA Graph, excluding forward, allocation, compilation and host planning.
Nine warmups precede nine alternating groups of nine timed replays; results
are medians of group means. This is a single-layer backward measurement,
not a model-training or context-parallel communication benchmark.

## H20 measurements (2026-11-12)

Torch 2.9.1+cu128, CUDA toolkit 12.8, cuDNN 9.10.2, CuTe DSL 4.8.1;
the CUDA kernel is compiled with NVRTC. Baseline and candidate use the same
inputs, clock conditions, launch geometry and software.

| Layout (K=V=128) | Heads | Baseline backward (ms) | Fixed backward (ms) | Cost | Workspace bytes, either version |
|---|---:|---:|---:|---:|---:|
| [0, 1, 15, 16, 17, 65] | 2 | 0.052299 | 0.052953 | +1.25% | 1,849,344 |
| 1,024 dense | 8 | 0.122770 | 0.123755 | +0.80% | 26,280,960 |
| 16,384 dense | 8 | 1.570404 | 1.585223 | +0.94% | 420,481,536 |
| [0, 1, 65, 257, 4097] | 8 | 0.461863 | 0.470830 | +1.94% | 117,663,232 |
| 131,072 dense | 8 | 12.186116 | 12.294155 | +0.89% | 3,363,848,704 |
| 131,072 dense | 32 | 43.711904 | 44.126105 | +0.95% | 13,455,344,128 |
| 128K THD: 20 short + 2 long | 8 | 11.638019 | 11.873422 | +2.02% | 3,381,826,560 |

The 22-document THD case is 20 × 256 + 2 × 62,976 = 131,072 tokens.
The extra local contractions cost approximately 1% of dense backward and
2% of these ragged/THD cases. Removing inverse-gradient products offsets
most of the additional residual work.

The following are **maximum dG relative-L2 errors within each family**, in
percent. A family's two maxima need not come from the same seed/length.
Every one of the 52 paired cases improves dG.

| Family | Cases | Baseline dG error (%) | Fixed dG error (%) |
|---|---:|---:|---:|
| correlated | 24 | 8.3028 | 1.7076 |
| dense_correlated | 4 | 1.4728 | 1.1443 |
| random | 12 | 0.4725 | 0.4579 |
| weak | 4 | 0.4361 | 0.4316 |
| steep | 4 | 0.9184 | 0.9167 |
| mixed_gate | 4 | 0.4354 | 0.4231 |

For the paired seed-713 correlated cases, T=65 improves from 4.3598% to
1.1740%; T=257 improves from 5.2566% to 1.5294%. These are ordinary
single-sequence backward cases with no context parallelism.

All six gradients were measured. dQ improves in 50/52 cases, dK in 49/52,
and dBeta in 28/52. The largest relative increases in their error metrics
are 0.048%, 0.020%, and 5.40%, respectively. The last corresponds to dBeta
relative-L2 error increasing from 0.3701% to 0.3900%, not a 5.40 percentage
point increase. dV and dInitialState remain byte-identical in all 52 cases.
Steep-decay accuracy changes little: this fix addresses local coefficient
rounding, not every BT64 error source.

## Validation

The public suite covers O, final state and all six gradients against a
separate recurrent reference, plus raw-byte replay, batch splitting,
reordering, duplication, changed neighbors and concurrent work. Six
correlated regressions (three seeds × two lengths) fail on upstream and
pass with this change, with an independent FP64 dG error bound of 2%.
The benchmark extends the FP64 comparison to all six gradients in 52 cases.

Warm stress has 0/3000 warm and 0/3000 poisoned failures; its unsafe
control fails 200/200 times. Memcheck and synccheck report zero errors;
racecheck reports zero errors and zero warnings. Raw-launch sanitizer
cases cover full/tail tiles, both scan tile sizes, ragged sequences and
an entirely empty batch, with scratch/output poisoning between replays.

The final compiled full/tail local-backward bodies are instruction-identical
to the previously inspected candidate. The changed lifetimes are fenced
by stores consuming the old operands before reuse: phase-A state products
before residual publication, original V before saved-Z reload, Y/Z before
B6 operand staging, and residual contractions before output staging.
No new broad fence or synchronization primitive is introduced.

The committed golden was recorded into a new directory after replay and
reference checks. Relative to an independently recorded upstream golden,
inputs/runtime are identical; only dQ, dK, dG and dBeta hashes change.
O, final state, dV and dInitialState hashes are unchanged. The test coverage
report records the strict golden check and public-suite command.

Sanitizer reproduction (repeat with `racecheck` and `synccheck`):

```bash
PYTHONPATH="$PWD/python" compute-sanitizer --tool memcheck --error-exitcode 99 \
  python benchmark/linear_attention/benchmark_sm90_bt64_precision.py \
  --mode smoke --output smoke.json
```
