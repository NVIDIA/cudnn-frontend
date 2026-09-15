# Isolated JAX tests

Run with one visible SM100 GPU and the edited frontend installed:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest test/jax -q
```

Requires CuTeDSL >=4.7 and JAX >=0.9.1 with CUDA support. The suite does not load
`test/python/conftest.py`. Fresh-process tests block every torch import, including
when torch and TVM-FFI are installed. Keep torch parity tests under
`test/python/fe_api/bsa/test_BSA_jax_parity.py`.

The suite covers output/gradient numerics, both physical layouts, variable and
empty rows, multiple backward buckets, asynchronous metadata changes, input
immutability, result pytrees, unsupported inputs, and initialized-output argument
ordering in the shared bridge. Compiled forward HLO checks apply to compact
standalone inputs; they do not promise arbitrary surrounding JAX graphs avoid
layout conversions.

Run this directory separately from torch tests in GPU CI. The repository's public
GitHub workflows currently provide no GPU test job; adding a qualified runner is
still needed for automatic JAX qualification.

## Container qualification

The [Dockerfile](Dockerfile) pins the NGC JAX 26.07 image and CuTeDSL 4.7.0,
builds this checkout's frontend wheel from source, and runs the isolated suite
without torch installed. Build from committed sources to exclude local venvs,
build artifacts and planning files:

```bash
git archive --format=tar.gz HEAD > /tmp/cudnn-bsa-jax.tar.gz
docker build -t cudnn-bsa-jax -f test/jax/Dockerfile - < /tmp/cudnn-bsa-jax.tar.gz
docker run --rm --gpus device=0 cudnn-bsa-jax
```

The default command is `python -m pytest test/jax -q --require-sm100`.
`--require-sm100` makes a missing GPU, unsupported architecture or multiple
visible GPUs an error instead of a skipped qualification run. Select the idle
SM100 device appropriate to the host with Docker's `--gpus` option.

A container CI job must invoke this suite explicitly, in its own pytest process.
A job collecting only `test/python/fe_api` does not collect `test/jax`, and a
CuTeDSL 4.6.x job cannot qualify BSA JAX. Installing JAX or running the torch
parity test alone does not provide this coverage.

Validated on SM100 on 2026-09-10: **28 passed** in the torch-free image. A
CPU-only run with `JAX_PLATFORMS=cpu` and `--require-sm100` exited with status 4.
The frontend 1.29.0 extension was built from this checkout against cuDNN 9.24;
installed JAX adapter sources were checked against the build inputs. Versions:
Python 3.12.3, JAX 0.10.2.dev20260630+3757395a28,
jaxlib 0.10.2.dev20260725, CuTeDSL 4.7.0, TVM-FFI 0.1.13.post3.
NGC enabled CUDA 13.3 forward compatibility (driver 610.43.02 over host kernel
driver 580.159.03).

In a separate instance of the same image, installing torch 2.14.0+cu130 and
looseversion and running the following produced **33 passed, 2 xfailed**:

```bash
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip install looseversion
cd /opt/cudnn-frontend/test/python
python -m pytest -s -q \
  fe_api/bsa/test_BSA_attention_forward.py \
  fe_api/bsa/test_BSA_attention_backward.py \
  fe_api/bsa/test_BSA_jax_parity.py \
  fe_api/gemm/test_gemm_amax_jax.py \
  fe_api/gemm/test_gemm_srelu_dsrelu_jax.py
```

For JAX/torch/GPU timing comparisons and known overhead findings, see the
[BSA benchmark](../../benchmark/bsa/README.md). Run it separately: it imports torch.
