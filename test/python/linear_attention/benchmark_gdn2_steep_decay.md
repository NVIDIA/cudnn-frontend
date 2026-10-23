# Standalone GDN-2 steep-decay reproduction

This benchmark needs one supported Blackwell GPU, PyTorch, and public cuDNN Frontend dependencies.
It generates every input from seed 7082. It needs no model, checkpoint, dataset, application package, or distributed runtime.

## Workload and checks

Cases use 16 or 256 tokens, one head, key/value dimensions 64, and 0/1/64 steep channels.
Q/K are normalized random vectors. Q/K/V and erase/write gates use BF16; log gates use FP32.
Mild log gates are -0.02, steep gates are -6, and erase/write gates are 0.5.
The output and all six gradients must match a CPU FP64 recurrence within 1% relative Frobenius error.

An untimed diagnostic checks the backward exact-path scheduler counter: zero for mild cases and above one for steep cases.
It temporarily wraps internal plan methods and restores them before timing. Use this script in a single process.
Five warmups precede 30 eager CUDA-event samples and five profiler samples of ten calls each.
Eager intervals include host enqueue gaps. GPU busy time merges profiler intervals, excludes idle gaps, and counts overlaps once.
Profiler overhead remains. These measurements do not establish end-to-end throughput gains.

## Public environment and ABBA procedure

The measured environment used `nvcr.io/nvidia/pytorch:25.10-py3`, PyTorch `2.9.0a0+145a3a7bda.nv25.10`, CUDA 13.0,
CuTeDSL 4.7.0, cuDNN 9.27.0.42, and one B300 (SM 10.3, driver 580.173.02).
The engine requires CuTeDSL 4.7.0 even though the package metadata permits a lower version.
Build requirements include nanobind and Ninja; install them explicitly when disabling build isolation.

Start from a clean checkout of this PR. For example:

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend
git fetch origin pull/1816/head
git checkout --detach FETCH_HEAD
docker run --rm -it --gpus device=0 --ipc=host \
  -v "$PWD:/work" -w /work nvcr.io/nvidia/pytorch:25.10-py3 bash
```

Inside the public container, build one wheel and create two runtime copies.
The base copy restores the helper from the exact upstream base commit.
The compiled extension and all other runtime files therefore remain identical.
Use a fresh output directory for each experiment:

```bash
set -euo pipefail
unset PYTHONPATH PIP_CONSTRAINT
REPRO_DIR=$(mktemp -d /tmp/gdn2-steep.XXXXXX)
export REPRO_DIR
python -m venv --system-site-packages "$REPRO_DIR/venv"
export PATH="$REPRO_DIR/venv/bin:$PATH"
python -m pip install \
  'nvidia-cutlass-dsl[cu13]==4.7.0' 'apache-tvm-ffi==0.1.11' \
  'nvidia-cudnn-cu13==9.27.0.42' 'nanobind==3.1.0' \
  'ninja==1.11.1.1' 'cmake>=3.26' 'setuptools>=64' wheel
export CUDNN_PATH=$(python -c 'import nvidia.cudnn; print(next(iter(nvidia.cudnn.__path__)))')
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL=4
python -m pip wheel --no-deps --no-build-isolation . -w "$REPRO_DIR/wheels"
python -m pip install --no-deps --target "$REPRO_DIR/patch" "$REPRO_DIR"/wheels/*.whl
cp -a "$REPRO_DIR/patch" "$REPRO_DIR/base"
curl -fL --retry 3 \
  https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/f2aa360e5a08436651d158cfad26b05b7db5ab92/python/cudnn/linear_attention/frost/common/steep_decay.py \
  -o "$REPRO_DIR/base/cudnn/linear_attention/frost/common/steep_decay.py"
find "$REPRO_DIR/base" "$REPRO_DIR/patch" -name '*.pyc' -delete
mkdir "$REPRO_DIR/results"
python -m pip freeze > "$REPRO_DIR/results/packages.txt"
nvidia-smi -q > "$REPRO_DIR/results/nvidia-smi.txt"
for arm in base-A1 patch-B1 patch-B2 base-A2; do
  case "$arm" in base-*) variant=base ;; *) variant=patch ;; esac
  PYTHONPATH="$REPRO_DIR/$variant" python \
    test/python/linear_attention/benchmark_gdn2_steep_decay.py \
    --label "$arm" --output "$REPRO_DIR/results/$arm.json"
done
python test/python/linear_attention/summarize_gdn2_steep_decay.py "$REPRO_DIR/results"
```

Copy the result directory out before removing the container.
If running against another kernel revision, update the pinned base and compare runtime manifests again.
Do not use an editable installation that resolves to another checkout. Each JSON records the actual import path and helper hash.

The summarizer checks identical script/environment identities, repeated helper hashes, all six cases, and the FP64 pass flags.
It also checks finite errors for the output and all six gradients against the 1% tolerance.
Timing medians must be finite and positive. Derived measurements must be finite.
These checks remain active under optimized Python (`-O`). Both scripts write strict JSON.
It averages the two per-arm medians for each variant. Keep all samples, including the mild controls.
Synthetic output/gradient snapshots are saved alongside the JSON for paired numerical comparisons.

## Results and evidence

Measured results and qualification counts are in [PR #1816](https://github.com/NVIDIA/cudnn-frontend/pull/1816).
The [evidence archive](https://github.com/alfredclwong/cudnn-frontend/releases/tag/evidence-pr-1816-20261023) contains raw samples, logs, JUnit reports, environment records, and the measured scripts.
The release includes a SHA256 checksum. Generated results do not belong in this checkout.
The archive retains the measured scripts. Later validation fixes change the benchmark script hash and require fresh ABBA records for new comparisons.

## CPU summary tests

The summary command needs only the Python standard library. Its regression tests need pytest and no GPU.
Run them without the GPU test harness:

```bash
python -m pytest --noconftest -c test/python/pytest.ini -o addopts= \
  test/python/linear_attention/test_gdn2_steep_decay_summary.py
```

## Upstream correctness tests

Use the public environment above and the patched runtime. Install the public test dependencies:

```bash
python -m pip install pytest pytest-xdist looseversion
export PYTHONPATH="$REPRO_DIR/patch"
cd test/python
python -m pytest -n 0 -m '' \
  linear_attention/test_gdn2_frost_steep_decay.py \
  linear_attention/test_gdn2_frost_steep_scratch.py -ra
python -m pytest -n 0 -m '' linear_attention/test_la.py \
  -k 'gdn2 and frost and (overwrite or state_indices)' -ra
```

The empty marker expression disables the default smoke-level filter.
The linear-attention test configuration overlays this checkout's Python source onto the installed package.
Use the exact PR checkout and verify its helper hash matches the patched runtime before testing.
Run these commands sequentially on one GPU. Read pytest's full summary and exit status, including skips and errors.
