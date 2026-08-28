# Flex Attention benchmark

This benchmark measures the training performance of the CuTe DSL Flex
Attention implementation exposed as `cudnn.flex_attention`. It covers mask
plan construction and forward and backward execution for the supported static
mask patterns.

## Requirements

- NVIDIA Hopper SM90, Blackwell SM100, or Blackwell SM103 GPU
- CUDA-enabled PyTorch
- the cuDNN Frontend `cutedsl` optional dependencies

From a source checkout:

```bash
pip install -e ".[cutedsl]"
pip install --group torch
```

## Run

Run the standard BF16 workload (`B=1`, `S=128K`, `Hq=Hkv=4`,
`Dqk=Dv=128`) from the repository root:

```bash
PYTHONPATH=python python benchmark/flex_attention/benchmark_flex_attention.py
```

Select masks, phases, or a different supported head configuration:

```bash
PYTHONPATH=python python benchmark/flex_attention/benchmark_flex_attention.py \
  --mask causal,local,longformer \
  --phase forward,backward \
  --head-dim 192 \
  --seqlen 131072
```

`--head-dim 128`, `192`, and `256` select `(Dqk,Dv)` equal to `(128,128)`,
`(192,128)`, and `(256,256)`, respectively. Use `--dry-run` to construct and
summarize masks without initializing CUDA kernels.

The eight mask workloads are causal, document causal, causal local, sink plus
local, tree DFS, tree BFS, Longformer, and packed HSTU context/target. Each is
encoded as an odd number of endpoints describing the interval union
`[0,F0) U [F1,F2) U ...` for every query row.

## Measurements and output

The benchmark reports median latency for:

- mask-plan construction (`metadata`);
- forward, backward, and combined forward-plus-backward kernel execution;
- plan construction plus forward, and plan construction plus training.

Kernel samples use CUDA events. Measurements containing mask-plan construction
use synchronized wall-clock time. L2 is flushed before every sample, first-use
JIT compilation is reported separately, and active TFLOP/s uses only visible
query/key pairs.

Results are written beneath `benchmark/flex_attention/results/<timestamp>/`
by default, or to `--output-dir`. Each run produces `results.json`,
`results.csv`, and a compact `results.md` table. Provenance records the current
repository revision, GPU, driver, PyTorch, and CUTLASS DSL versions.
