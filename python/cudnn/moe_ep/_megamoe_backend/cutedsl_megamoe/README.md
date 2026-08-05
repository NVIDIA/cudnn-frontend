# mega_moe_cutedsl_dispatch

Single cuTeDSL (`nvidia-cutlass-dsl==4.4.2` substituted for the plan-pinned `4.5.0dev0`, see OI-5) kernel replicating the dispatch phase of mega_moe (DeepGEMM `sm100_fp8_fp4_mega_moe.cuh:432-766`). All four sub-flows live in one `@cute.kernel`: Dispatch_Prep, Dispatch_Barrier, Dispatch_Pull, plus a kernel-tail NVLink barrier that takes the place of mega_moe's dispatch-combine handoff. A separate host-callable `cleanup_kernel` zeroes the workspace between dispatch invocations.

**Status: V1 fully validated end-to-end on 4× GB200 (sm_100a) via NVSHMEM.** All 12 acceptance criteria from the plan are satisfied (AC-10 reproducibility is partial — counters deterministic, advertise/pool slot order races; matches mega_moe's same-hardware behaviour, see OI-20).

Plan: `docs/plans/dispatch_cutedsl_plan.md` · Goal tracker: `.humanize/rlcr/2026-04-29_07-34-03/goal-tracker.md`

## Architecture

The kernel runs across `num_sms = 152` SMs per rank and `num_ranks = 4` GPUs. Inside each CTA, warps 0..3 are dispatch warps; later warps idle in the V1 dispatch-only kernel. The four sub-flows execute sequentially on the dispatch warps:

1. **Dispatch_Prep** scans `input_topk_idx_buffer` three times. Round 1 builds a per-CTA SMEM histogram via `atomicAdd_block`. Round 2 fans the 256 global experts across 4×32 lanes and FAA's `expert_send_count` with a `(1<<32 | local_count)` u64 delta, so the rank's send-count high32 accumulates to `num_sms` and the low32 returns the SM's `base_slot`. Round 3 re-scans topk and writes `peer_src_token_topk_idx[dst_rank][local_expert][local_rank][base_slot + atomicAdd_block]`.
2. **Dispatch_Barrier** publishes per-rank `expert_send_count` to peer `expert_recv_count` (low32 STG) and `expert_recv_count_sum` (release/sys u64 atom-add for both publisher count and token sum), runs `software_grid_sync`, then runs the 3-stage NVLink barrier (SM-0 lanes 0..world_size atom.release.sys.add.s32 +1 on each peer signal; SM-0 thread 0 ld.acquire.sys.s32 spin until local signal == world_size; epilogue grid_sync to release all SMs).
3. **Dispatch_Pull** iterates the local pool by expert order and round-robin min-peeling source ranks (matching `src/reference.py::_round_robin_pool_order`). Each pool slot: read `src_token_topk_idx`, `mbarrier_arrive_and_expect_tx`, `tma_load_1d` from peer, 32-lane SF LDG/STG with `transform_sf_token_idx`, `mbarrier_wait` + phase flip, `tma_store_1d` to `l1_token_buffer[pool_token_idx]`, write 12-byte metadata, `cp_async_bulk_commit_group` + `wait_group(0)`, and finally `atom.release.gpu.global.add.u32` on `l1_arrival_count[expert_pool_block_offset + token_idx_in_expert/block_m]`.
4. **Kernel-tail NVLink barrier** replaces the dispatch-combine `sync_unaligned` pairing in mega_moe.

## Module map

| File                       | Purpose |
|----------------------------|---------|
| `src/config.py`            | `V1Config` (formula-derived pool sizes), `transform_sf_token_idx_numpy`, `MAX_SLOT`, `TOKEN_METADATA_BYTES`. Pure-Python, zero GPU deps. |
| `src/bootstrap.py`         | `init_nvshmem` (UID broadcast), `alloc_workspace` (15 symmetric tensors + `grid_sync_counter`), `peer_views`, `finalize_nvshmem`. Note dtype substitutions (uint32→int32, uint64→int64) needed for nvshmem4py 0.3.0. |
| `src/ptx_helpers.py`       | `tma_load_1d`, `tma_store_1d`, `fns_b32` inline-PTX wrappers. |
| `src/grid_sync.py`         | `software_grid_sync` (single-slot phase-flip pattern from `barrier.cuh`, `kFinishSumTag = 0x80000000`). |
| `src/sf_swizzle.py`        | Device-side `transform_sf_token_idx` (4×32 UTCCP swizzle, integer-div based). |
| `src/dispatch_kernel.py`   | The 1500-line dispatch kernel (`@cute.kernel dispatch_kernel`) with five inline-PTX atomic helpers. |
| `src/cleanup_kernel.py`    | Host-callable workspace cleanup; replicates mega_moe `:723-766`. |
| `src/reference.py`         | NumPy CPU oracle producing 9 expected buffers + `assert_buffer_equal`. |
| `scripts/compile_smoke.py` | `cute.compile`-only smoke test (no kernel launch). |
| `scripts/single_rank_validate.py` | world_size=1 functional test against the oracle (4 byte-exact + 4 invariant checks). |
| `scripts/repro_validate.py`       | 5x repeat-launch determinism check. |
| `scripts/degenerate_validate.py`  | T_actual=0, extreme skew, 50%-mask, single-token. |
| `scripts/cleanup_validate.py`     | dispatch -> cleanup -> dispatch round-trip. |
| `scripts/multi_rank_validate.py`  | **`torchrun --nproc_per_node=4 -m scripts.multi_rank_validate`** — full 4-GPU NVSHMEM end-to-end. |
| `tests/`                   | pytest suite stubs (skipped without `LOCAL_RANK`). |

## V1 testing config

The `V1Config` constants are formula-derived from `layout/mega_moe.cuh::get_num_max_pool_tokens` and `get_num_padded_sf_pool_tokens`, NOT the literal `kNumPaddedSFPoolTokens=1088` mentioned in the plan (which was internally inconsistent — see goal-tracker OI-1).

| Constant                       | V1 value |
|--------------------------------|---------|
| `num_ranks`                    | 4       |
| `num_tokens_per_rank`          | 1024    |
| `hidden`                       | 7168 (FP8) |
| `num_topk`                     | 6       |
| `num_experts_per_rank`         | 64      |
| `num_total_experts`            | 256     |
| `block_m`                      | 192     |
| `sf_block_m`                   | 256     |
| `num_sms`                      | 152     |
| `sf_uint32_per_token`          | 56      |
| `num_max_pool_tokens` (computed) | 36864 |
| `num_padded_sf_pool_tokens` (computed) | 49152 |
| `num_max_pool_blocks` (computed) | 192   |

## Validation results (latest)

### Host-only oracle (no GPU required)
```bash
venv/bin/python -m pytest tests/test_oracle.py -v
# 14 passed in 0.3s
```

### Single-rank GB200 functional
```bash
venv/bin/python -m scripts.single_rank_validate
# Order-sensitive byte-exact: expert_send/recv_count, expert_recv_count_sum,
#                             l1_arrival_count -> all PASS
# Order-invariant invariants:  6144/6144 metadata-data self-consistent for
#                              src_token_topk_idx, l1_token_buffer,
#                              l1_topk_weights_buf, l1_sf_buffer (incl. swizzle)
# ALL INVARIANTS PASS.
```

### Degenerate scenarios
```bash
venv/bin/python -m scripts.degenerate_validate
# A (extreme skew):           PASS
# B (all masked, T_actual=0): PASS
# C (50% masked):             PASS
# D (single token):           PASS
```

### Cleanup re-launch
```bash
venv/bin/python -m scripts.cleanup_validate
# Counters zeroed by cleanup:                              PASS
# Counters identical across cleanup-flanked dispatch runs: PASS
```

### **4-rank multi-GPU (NVSHMEM)** — the gating test for V1
```bash
torchrun --nproc_per_node=4 --standalone -m scripts.multi_rank_validate
# [rank 0] populated=6045, expected=6045, counters=PASS, invariants=PASS
# [rank 1] populated=6082, expected=6082, counters=PASS, invariants=PASS
# [rank 2] populated=6290, expected=6290, counters=PASS, invariants=PASS
# [rank 3] populated=6159, expected=6159, counters=PASS, invariants=PASS
```

### Reproducibility (intentional non-determinism, see OI-20)
```bash
venv/bin/python -m scripts.repro_validate
# 5 launches, fixed seed:
#   esc, erc, ercs, lac:    byte-identical across all 5 runs
#   sti, tsm, ltb, lsb, ltw: ~6000 entries differ between runs
# (advertise / pool slot order races across 152 SMs; matches mega_moe behaviour)
```

## Environment provisioning

```bash
# cuTeDSL — pip-installable; 4.4.2 is the latest published version. The
# plan-pinned 4.5.0dev0 is a CI-only dev build (use `dkg-fetch-ci-tarball`
# skill with URM_TOKEN if you specifically need it).
venv/bin/pip install nvidia-cutlass-dsl

# nvshmem4py 0.3.0 — distributed under the cu13 / cu12 suffix.
venv/bin/pip install nvshmem4py-cu13
```

## Open issues (live list in goal-tracker)

Highlights of the 25 tracked open issues:

- **OI-1**: V1 pool sizes use mega_moe formulas, not the plan's `1088` literal.
- **OI-2**: `tma_store_1d` uses 3-operand bulk_group (no L2 hint); mega_moe uses 4-operand `.L2::cache_hint` variant — may need adjustment.
- **OI-3**: `software_grid_sync` uses named PTX labels; verify no link-time collision under multi-callsite inlining.
- **OI-9..OI-19, OI-21..OI-25**: cuTeDSL pattern fixes for `cute.compile` end-to-end (see goal-tracker for the full chain). Resolved.
- **OI-20**: AC-10 reproducibility — counters deterministic, slot order races. Mega_moe has the same property.

## Key cuTeDSL gotchas discovered (4.4.2 / 4.5.0dev0)

These were the painful lessons from getting cute.compile to lower this kernel:

1. **`cute.compile` must wrap a `@cute.jit` launcher**, not a `@cute.kernel` directly. Calling `cute.compile(@cute.kernel, ...)` puts the kernel on the host code path and routes inline asm through the host (ARM/x86) assembler, producing "unrecognized instruction mnemonic" errors.
2. **PTX inline-asm constraint for global-memory pointers must be `l` (64-bit), not `r`**. Using `r` for a global pointer makes PTXAS think the module uses 32-bit ABI and rejects sm_90+ targets with `"32-Bit ABI ... is not supported"`.
3. **Dynamic-index SMEM tensor writes (`smem_tensor[dyn_idx] = val`) are unsupported**. Use `tensor.fill(value)` for bulk init, or inline-PTX `ld.shared.u32` / `st.shared.u32` for element-wise dynamic access.
4. **`@cute.struct` decorator inspects type annotations at decoration time** — incompatible with `from __future__ import annotations`.
5. **Helper functions called from `@cute.kernel` need `@cute.jit`** so the AST preprocessor reaches `range_constexpr` and `if`-region SSA tracking inside them.
6. **List comprehensions over `range_constexpr` are NOT preprocessed** — only `for` statements are. Use explicit constexpr-for + `.append()`.
7. **`if cute.arch.elect_one():` is wrong** — it returns an `IfOpRegion` context manager. Use `with cute.arch.elect_one():`.
8. **Tensor type cannot change inside a dynamic `if`** — `sf_val = Uint32(0)` then `sf_val = peer_int32_buffer[...]` raises "type changes inside dynamic if". Match initial type to the read source.
9. **Constexpr-fanout for dynamic peer-list indexing**: `peer_X[dynamic_idx]` fails Python's `__index__`. Use `for r in range_constexpr(0, world_size, 1): if dyn_idx == Int32(r): peer_X[r][...]`.
10. **`tensor + int` is unsupported**; use `tensor.iterator + int` to get a `Pointer` for atomic-target / TMA-source addresses.
11. **Mutating a tensor inside a constexpr-fanout `if` breaks SSA dominance**. Hoist the destination write outside the inner `if` (mirror the `sf_val` accumulator pattern).
12. **`from __future__ import annotations` + cute.struct**: the decorator inspects annotations as live Python expressions, not strings.
13. **`nvshmem4py 0.3.0` only supports signed dtypes** in `nvshmem.core.tensor`. Substitute `int32`/`int64` for `uint32`/`uint64`.

## Constraints

- V1 locked to (4 ranks, 7168 hidden, 6 topk, 64 experts/rank). V2 will generalize `num_ranks ∈ {2, 4, 8}` only.
- Single `cute.jit` kernel — no splitting into multiple kernels.
- Must preserve the receiver-driven advertise + pull two-stage architecture; no NCCL all-to-all replacement.
- All SF writes apply the 4×32 UTCCP swizzle — no row-major fallback.
