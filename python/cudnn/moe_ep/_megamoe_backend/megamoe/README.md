# megamoe — hackable CuTe DSL MegaMoE forward for the torch pipeline

A small, flat wrapper around the `cutedsl_megamoe` MXFP8 GLU mega-kernel
(dispatch → grouped FC1 SwiGLU → FC2 → combine, all in ONE kernel launch over
NVSHMEM) so it can be dropped into a PyTorch expert-parallel pipeline (e.g. as
a drop-in replacement for `pt/layer.py::MoEEpTrainingLayer.forward`).

Source of truth: the sibling clone `../cutedsl_megamoe` (latest `main`,
override with `MEGAMOE_REPO=`). Nothing in the clone is modified; this package
only re-hosts the launch path so it is easy to hack on.

```
repo_path.py   sys.path shim for the cutedsl_megamoe clone
weights.py     bf16 w13/w2 (pt/ convention) -> kernel MXFP8 layout:
               gate/up 32-block interleave, per-32 E8M0 block scales,
               32x4x4 atom-swizzled weight SFs (to_blocked)
forward.py     MegaMoeMxfp8Forward: persistent sym-heap buffers, one-time
               cute.compile of (a) DataPreprocess (fused bf16->mxfp8 quant +
               routing repack, src/inputs_process.py — the framework
               integration path added in `ag_dev/fi_input_quant`) and
               (b) Sm100MegaMoEMxfp8Kernel; forward() = copy-in + 2 launches
turboquant.py  MegaMoeTurboQuantForward: randomized block-Hadamard (b=128)
               incoherence on the hidden dim — activation rotated in forward,
               counter-rotation folded into fc1 weights (exact math)
tests/         torchrun parity test vs compute_megamoe_reference_mxfp8
               + turboquant numerics validation vs an exact fp32 oracle
```

## Usage

```python
from megamoe import MegaMoeForwardConfig, MegaMoeMxfp8Forward
from src.bootstrap import init_dist_and_nvshmem   # via megamoe.repo_path

_, rank, world_size, _ = init_dist_and_nvshmem()
cfg = MegaMoeForwardConfig(
    max_tokens_per_rank=1024, hidden=2048, intermediate=512,  # I per branch
    num_total_experts=32, num_topk=4,
)
fwd = MegaMoeMxfp8Forward(cfg, rank=rank, world_size=world_size)
fwd.load_weights(w13, w2)      # [E_local, 2I, H], [E_local, H, I] bf16
out = fwd(x, topk_idx, topk_weights)   # (T, hidden) bf16, T <= max_tokens
```

Contracts:
- `forward` is collective — every EP rank must call it (in-kernel NVLink
  barriers). Stream-ordered, no host sync inside.
- `topk_idx` holds **global** expert ids; expert `e` lives on rank
  `e // (num_total_experts // world_size)` (contiguous sharding, same as pt/).
- Math convention matches pt/: `silu(gate) * linear` with
  `w13[:, :I]` = linear/up, `w13[:, I:]` = gate; topk weights are folded into
  the fc1 epilogue (`apply_topk_in_fc1=True`, kernel-exact).
- `load_weights` re-quantizes in place — call per optimizer step, no
  recompile.
- Output is a view of a persistent buffer; clone if kept across steps.

## Test

Inside the GPU container (needs `nvidia-cutlass-dsl`, `nvshmem4py`, GB200):

```bash
cd moe_ep_training
torchrun --nproc_per_node=4 --standalone -m megamoe.tests.test_forward_parity
MEGA_NO_DIST=1 python -m megamoe.tests.test_forward_parity   # single-rank
```

Reference tolerance is the repo's own (atol=rtol=1e-2 vs
`compute_megamoe_reference_mxfp8`, which consumes the device-quantized
activations read back from the sym heap).

## Tuning the kernel knobs

`MegaMoeForwardConfig.impl` (a `TrainingImplDesc`) carries the tunables:
`mma_tiler_mnk`, `cluster_shape_mnk`, `load_balance_mode`
(`static`/`atomic_counter`), `group_hint`, `flag_batch`, `epi_flag_batch`,
`token_back_mode`, `in_kernel_fc2_reduce`, `use_stg_fc1`. Sweep them with the
repo's tester and paste the winner:

```bash
cd ../cutedsl_megamoe
torchrun --nproc_per_node=4 -m tester.tester --problems problems.jsonl \
    --mode Perf --sweep --use_knob mma_tiler_mnk=256,256,128 --results out.jsonl
```

(Problem jsonls come from `tester/problem_gen`; `tester.tester --help` for the
knob grammar. Note the tester's Perf solver covers the NVFP4 kernel; for MXFP8
knob sweeps use `moe_mxfp8_glu.mega_runner --perf_run --use_torch_profiler`.)

## Next steps / optimization hooks

- **turboquant / network-specific quant**: replace `DataPreprocess` (activation
  side) and/or `weights.py` (weight side) — everything downstream only sees
  fp8 bytes + E8M0 SFs in the fixed layout.
- `impl.generate_c=True` stashes the pre-SwiGLU fc1 gate+up (bf16) the future
  backward needs.
- `combine_format="32e4m3xe8m0"` halves combine NVLink bytes.
- Baseline to beat: `bench/bench_training_forward.py` (pure-torch pt/ path).
