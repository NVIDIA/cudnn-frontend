# Runbook: running and testing the `cudnn.MoeEp` PR

How to validate this PR end to end: the pure-Python API/reference tests and
the MegaMoE (CuTe DSL) device backend that ships inside cuDNN under
`python/cudnn/moe_ep/_megamoe_backend/` (megamoe + pt + the CuTe DSL kernel
sources — nothing is loaded from an external checkout).

## 1. What is under test

| Piece | Where | Needs GPU? |
|---|---|---|
| Public API (`cudnn.MoeEp`, `MoeFormat`, `BlockScaledTensor`) | `python/cudnn/moe_ep/api.py` | no (stub path) |
| PyTorch FP32 oracle | `test/python/fe_api/moe_ep/moe_ep_reference.py` | CPU ok |
| pytest suite (API + reference semantics) | `test/python/fe_api/moe_ep/test_moe_ep.py` | 1 GPU for two tests |
| Device backend shim | `python/cudnn/moe_ep/_megamoe.py` | — |
| Bundled kernels (MXFP8 fwd, NVFP4 fwd, FP8 mega bwd — bprop from the Flashinfer team, not FastKernel) | `python/cudnn/moe_ep/_megamoe_backend/` | SM100 (GB200) |
| Kernel-vs-oracle parity driver | `test/python/fe_api/moe_ep/megamoe_backend_parity.py` | SM100 (GB200) |
| One-shot wrapper for all of the above | `test/python/fe_api/moe_ep/run_pr_tests.sh` | SM100 (GB200) |

## 2. Environment

Requirements for the device backend: SM100+ GPU (GB200), PyTorch with CUDA
13.x, `nvidia-cutlass-dsl` (CuTe DSL), `nvshmem4py`, and a `cudnn` Python
wheel to graft the `moe_ep` package onto (see §4).  A known-good container:

```
/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh
```

Backend policy knob (the only env variable):

| `CUDNN_MOE_EP_BACKEND` | Meaning |
|---|---|
| `auto` (default) | use the kernel when the config is supported, else warn once and fall back to allocate-only stubs |
| `megamoe` | required — any backend failure raises (use for validation/CI) |
| `none` | never use the device backend |

`MEGA_NO_DIST=1` must be exported for single-GPU runs **before** any megamoe
import (the shim sets it for `ep_group=None`, but exporting it in the shell
is the safe habit).

## 3. One-shot validation (recommended)

Runs the pytest suite, a bundled-kernel import check (FP4 forward + FP8 mega
backward), and the single-rank kernel parity, in that order:

```bash
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik   # container/mount root
srun -A coreai_libraries_cudnn -p batch -N1 --ntasks=1 -t 60 \
  --container-image=$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh \
  --container-mounts=$ROOT:$ROOT \
  bash <clone>/test/python/fe_api/moe_ep/run_pr_tests.sh
```

Expected tail: `ALL PR TESTS PASSED` (exit 0).  Budget ~10 min: the first
parity call pays the one-time `cute.compile` (~3–5 min per kernel).

## 4. Step-by-step (what the wrapper does)

### 4a. pytest suite

If the installed `cudnn` wheel predates this PR, graft the pure-Python
package onto it first (container overlay, nothing persists):

```bash
SITE=$(python -c "import cudnn,os;print(os.path.dirname(cudnn.__file__))")
cp -r <clone>/python/cudnn/moe_ep $SITE/
grep -q moe_ep $SITE/__init__.py || \
  echo "from .moe_ep import BlockScaledTensor, MoeEp, MoeFormat, MoeTensor" >> $SITE/__init__.py
cd <clone>/test/python
NVIDIA_TF32_OVERRIDE=0 python -m pytest fe_api/moe_ep/test_moe_ep.py -q
```

Expected: **`18 passed, 7 xfailed`**.

- The 7 xfails are *intended*: strict `xfail` gates asserting bit-exact
  API-vs-oracle equality, which an MXFP8-compute kernel can never satisfy
  against an FP32 oracle.  Strict means an unexpected pass (`XPASS`) errors,
  so they double as tripwires.  Any `failed`/`error` count > 0 is a real
  regression.
- `NVIDIA_TF32_OVERRIDE=0` is required: NGC containers default TF32 on and
  `test_four_rank_expert_parallel` compares GPU fp32 GEMMs against CPU at
  fp32 tolerance.

### 4b. Kernel parity vs the PyTorch oracle (single rank)

```bash
cd <clone>/test/python
CUDNN_MOE_EP_BACKEND=megamoe MEGA_NO_DIST=1 \
  python fe_api/moe_ep/megamoe_backend_parity.py
```

Expected (baseline 2026-08-05, T=128 H=1024 I=512 E=8 K=4; gate is 0.10):

```
fwd rel_err ≈ 6.5e-02    fc1_c rel_err ≈ 3.8e-02    metadata equal=True
bwd rel_err ≈ 6.5e-02 for grad_activation / grad_fc1_weight /
             grad_fc2_weight / grad_topk_weights    dtw[-1 slot]==0: True
mxfp8 output quantizer bit-exact vs reference: True
PASS
```

How to read it: the ~6.5e-2 relative errors are MXFP8-compute vs FP32-oracle
noise — drift well beyond that (or past the 0.10 gate) is a numerical
regression.  `metadata equal=True` and the quantizer bit-exactness are hard
invariants — any regression there is a wiring bug, not noise.

### 4c. Kernel parity, 4-rank expert parallel (one GB200 node)

```bash
cd <clone>/test/python
CUDNN_MOE_EP_BACKEND=megamoe \
  torchrun --nproc_per_node=4 --standalone fe_api/moe_ep/megamoe_backend_parity.py
```

Same expected numbers, printed per rank; NVSHMEM is bootstrapped over the
torchrun world.

## 5. Gotchas

1. **Stub fallback returns uninitialized memory silently** — that is the
   PR's documented contract for unsupported configs under `auto`.  During
   validation always set `CUDNN_MOE_EP_BACKEND=megamoe` so misconfigurations
   raise instead.
2. **Supported envelope** (outside it `auto` falls back): SM100+,
   `apply_topk_in_fc1=True`, `hidden_size`/`intermediate_size` % 32 == 0
   (% 128 for the `generate_c`/backward path), `combine_format='bf16'` for
   training (`mxfp8` is forward-only, `nvfp4` combine unsupported).
3. **One backward per forward** on the same `MoeEp` instance — backward
   consumes the kernel pool stash of the immediately preceding forward.
4. **Process mode is frozen at first megamoe import**: one process serves
   either single-rank (`MEGA_NO_DIST=1`) or EP/NVSHMEM instances, not both.
5. **`max_tokens_per_rank`**: if omitted, kernel buffers are sized to the
   first call's token count and larger later calls raise.  Pass it
   explicitly for variable batch sizes.
6. **Compile cache**: every new process pays `cute.compile` once per kernel;
   shim/API edits don't recompile, kernel-source edits do.
7. The vendored `_megamoe_backend/` tree is a verbatim copy (provenance in
   its README) and is excluded from black — don't reformat it.
