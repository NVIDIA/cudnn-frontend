# Fused GDN Decode / Verify Steps as a FlashInfer Backend

**Experimental.** `cudnn.linear_attention.frost.gdn_decode_fused` and
`cudnn.linear_attention.frost.gdn_verify_fused` are the fused Gated DeltaNet (GDN) decode step
and MTP verify step of a Qwen 3.5-architecture dense model's GDN sub-layer, one CuTe DSL launch
each, for SM100-class (compute capability 10.x) and SM120-class (12.x) parts -- validated on
cc 10.0 (GB200) and cc 12.0 (RTX 5090). `cudnn.linear_attention.frost.flashinfer_gdn_backend` registers
them as the backend `"frost_fe"` of FlashInfer's experimental Qwen 3.5 GDN block
(`flashinfer.experimental.qwen_3_5_gdn_block`), so the block's plans, plan cache, CUDA-graph
rules, reference tests and benchmark run the cuDNN Frontend kernels unchanged.

Each step fuses, after the FP8 QKVZ projection of a serving step: the BA projection, the causal
conv + SiLU over the slot's history, the Q / K RMS scales, the gated delta-rule state update
(one token per request for decode, the request's `T` draft tokens for verify, every token's
state written to its own slot), the gated RMSNorm, the static per-tensor FP8 output quantization
and the conv-history shift. One CTA per (request, V head) with 256 threads; the 128 x 128 fp32
state rows are register-resident and travel through padded shared-memory boxes by
`cp.async.bulk`. The kernels serve the GDN sub-layer of one tensor-parallel rank as a geometry
`Geometry(K_HEADS, V_HEADS, HIDDEN)` (`cudnn.linear_attention.frost.kernel.gdn_geometry`): the
head counts and the hidden width move, head dim 128 and conv width 4 are pinned, and every
geometry is its own compiled form of the same kernel body (the default geometry -- 16 K heads,
48 V heads, hidden 5120 -- is the one every number on this page is written for; an entry takes
another with `geometry=`, a FlashInfer plan through its `Qwen35GdnGeometry`). The default form
follows three plan-time facts, the request count, the device's SM count and the device's
shared-memory readings (per SM and the per-CTA opt-in): a cluster of two CTAs per head (half the
state rows each) while the `requests x 2 V` cluster grid has an SM per CTA (`requests x 96` at 48
V heads: one request on 148- / 152- / 170-SM parts), one CTA per SM while the `requests x V` grid
has one (`requests x 48`), and past that a register-capped two-CTAs-per-SM form so the grid
runs in one wave -- where two of its CTAs can share an SM, i.e. `2 x (footprint + 1 KB)` fits
the SM's shared memory (228 KB on cc 10.0 parts). On a 100 KB-per-SM part (cc 12.0: 99 KB
per-CTA opt-in) every whole-slab form is one CTA per SM by shared memory alone, so the cap is
never picked there (it would only spill); the decode step runs the one-CTA form in two waves
past the one-CTA grid, and the verify step -- whose cc 10.0 forms carry a 3-deep ring of 64 KB
state slabs that a 99 KB opt-in cannot hold -- serves the same kernel body with shallower rings:
the two-CTAs-per-head pair with a 2-deep ring at one request and again past the one-CTA grid
(four and more requests on 170- / 188-SM parts: its one-per-SM cluster CTAs fill the waves the
one-CTA form's nearly empty second wave would leave -- measured at four requests on one 170-SM
RTX 5090 at boost clocks (the board allows no clock lock; three interleaved A/B/A rounds of 100
event-timed iterations, the L2 swept before every launch so code and data are cold, every arm
bitwise with the pick): 65.54 vs 71.68 us at T 6 and 50.90 vs 53.25 us at T 4, +4 to +9 % faster
than the one-CTA form across the cold, warm and in-situ readings; the other beyond-the-grid cells
take the order by the budget rule, unmeasured -- see Performance), the one-CTA landing-slab
(`NBUF` 1) form at two and three requests, each up to 8 tokens per request, and the landing-slab
pair for 9 to 17 tokens per request at every requests count. The verify step traces its token
loop once (`ROLLED_TOKENS`) in the forms that keep the state in registers. Every knob is a compile-time
axis of one kernel body and bitwise with the default pick (no knob moves an fp32 addition), and
the outputs reproduce FlashInfer's built-in `frost` backend byte for byte.

## Register

Importing the module registers the backend; FlashInfer's benchmark loads a backend
module with `--register`:

```python
import cudnn.linear_attention.frost.flashinfer_gdn_backend  # registers "frost_fe"

from flashinfer.qwen_3_5_gdn_block import (
    qwen_3_5_gdn_block_decode,
    qwen_3_5_gdn_block_support,
    qwen_3_5_gdn_block_verify,
)

sup = qwen_3_5_gdn_block_support(device, geometry, "decode", "frost_fe")
out = qwen_3_5_gdn_block_decode(..., geometry=geometry, backend="frost_fe")

sup = qwen_3_5_gdn_block_support(device, geometry, "verify", "frost_fe", requests=4, tokens_per_request=6)
out = qwen_3_5_gdn_block_verify(..., geometry=geometry, backend="frost_fe")
```

```bash
python benchmarks/bench_qwen_3_5_gdn_block.py \
    --register cudnn.linear_attention.frost.flashinfer_gdn_backend --backends frost frost_fe
```

`register_frost_fe(replace=False)` is the explicit entry: idempotent for this
module's own registration, `replace=True` re-registers (and drops the backend's
cached plans). Nothing here is imported at `import cudnn` time; the two entries are also
reachable lazily as `cudnn.GdnDecodeFused` / `cudnn.GdnVerifyFused` (and under
`cudnn.linear_attention.frost`): direct-call kernels with the block's argument contract, not
graph engines. Each entry resolves its device to a concrete CUDA ordinal at construction and
validates the binding (every operand on exactly that device, dtype, shape, stride, 16-byte
alignment, the step shape) before every compile and every launch, compile-cache hits included
-- the compile key pins the plan-time facts only, and a compiled CuTe DSL entry accepts a tensor
of another dtype, footprint or device silently -- and declines a step of more than 65535
requests (one grid row each). CPU operands or operands on another CUDA device are a
`ValueError` naming the operand and both devices.

## Served and declined

| question | answer |
|---|---|
| form | `decode` and `verify` (kind `form` otherwise) |
| geometry | head dims 128 and conv kernel 4 (pinned), with the head counts and `d_model` of a tensor-parallel rank inside the kernels' envelope: at most 16 K heads, at most 128 V heads (a whole number of V heads per K head), `d_model` a multiple of 1024 -- the block's dense-27B geometry at rank 1 (`d_model` 5120, 48 V heads, 16 K heads; the default every figure on this page is measured at) and the per-rank shards of larger dense models (e.g. `d_model` 8192 with 1 K head and 8 V heads at 16-way tensor parallelism), each its own compiled form (kind `geometry`, naming the fields otherwise); any `norm_eps` |
| device | an SM100-class part (compute capability 10.x) or an SM120-class part (12.x) (kind `device` otherwise, naming both served families; cc 11.x has never run these kernels and is declined); validated on cc 10.0 (GB200: 152 SMs, 228 KB of shared memory per SM, 227 KB per-CTA opt-in) and cc 12.0 (RTX 5090: 170 SMs, 100 KB per SM, 99 KB opt-in); cc 12.1 is admitted on the same budget-derived picks without a run of its own; a device whose shared memory holds no verify form is kind `device` too, naming its opt-in |
| CuTe DSL | `nvidia-cutlass-dsl >= 4.8.0` (kind `dsl`, naming the installed version) |
| FlashInfer | the experimental Qwen 3.5 GDN block with its backend registry (`flashinfer.experimental.qwen_3_5_gdn_block.api.register_backend`): FlashInfer main at or after PR #8131 (2026-10-31); no tagged release carried it at the time of writing, and FlashInfer marks the block experimental (it may change or be removed without deprecation). Importing the backend module without it raises `ImportError` naming the requirement |
| decode rows | up to 65535 requests per call, one grid row each (kind `shape`) |
| verify step shape | up to 65535 requests per call, one grid row each; `tokens_per_request` from 2 up to the shared-memory budget of the forms the entry picks for that requests count on the device (`verify_max_tokens(device, requests)`, the bound of the pick's whole ladder): the cluster pair while the `requests x 96` CTAs have an SM each, the one-CTA-per-SM form while the `requests x 48` CTAs have one, the two-CTAs-per-SM form past that -- on a cc 10.0 part 35, 8 and 45 tokens per request at the default geometry (at a per-rank geometry the one-CTA form serves past the grid too, in two waves, so 35, 8 and 8: the capped form is measured at the default geometry only, and it spills at hidden 8192); on a cc 12.0 part (99 KB per-CTA opt-in) the regime's shallower-ring forms of the same kernel: 8 tokens per request with the picked form at every requests count (the 2-deep-ring pair at one request and past the one-CTA grid, the one-CTA landing-slab form at two and three) and up to 17 through the landing-slab pair, the pick moving with the step's token count (kind `shape`, naming the deepest form and its bound) |
| parameter dtypes | decode: `A_log` fp32, `dt_bias` and `norm_weight` bf16 (FlashInfer's default `param_dtypes`; the kernel-native dtypes). Not part of FlashInfer's support question, so a decode plan built for other dtypes is declined at `prepare_qwen_3_5_gdn_block` time with a `NotImplementedError` naming them. Verify: each of the three in fp32 or bf16, compiled per dtype set. |
| conv pool, decode | the `[num_slots, C, K - 1]` view in either layout: channel-innermost storage (`[num_slots, K - 1, C]` transposed) or dim-first (`[num_slots, C, K - 1]` contiguous) |
| conv pool, verify | the `[num_slots, C, T + 2]` view of a channel-innermost pool (`[num_slots, T + 2, C]` transposed) only; the dim-first pool FlashInfer admits is a typed decline at plan time (`NotImplementedError` naming the layout), as with FlashInfer's built-in backend |
| state layout, verify | the block's per-token layout: request `n` with `a = num_accepted_tokens[n]` starts from slot `state_indices[n, a - 1]` and writes the state after token `t` to slot `state_indices[n, t]`; a slot `<= 0` skips that write; a request whose source or conv slot is `<= 0` is padding (zero outputs, nothing touched); a request may carry fewer than `T` tokens (a ragged step), in which case the direct entry takes the step's row count as `rows` |
| padding, decode | a row whose slot is `<= 0` writes zero outputs and touches no state (the same convention as the block) |

## Plans and CUDA graphs

A plan's `prepare` compiles the kernel for the binding's strides once (plan-time
facts only; `T` is a compile-time constant of the verify kernel) and is cheap
afterwards; `run` launches on the current stream, writes the caller's FP8 `out` (and
bf16 `out_bf16` when given) and advances `conv_state` / `ssm_state` in place. A
binding that was never run eagerly is refused under capture with a `RuntimeError`,
per the block's contract. Compiled kernels are shared by every plan of one form,
device and norm epsilon.

Both entries take the launch knobs `pdl` / `pdl_release` (programmatic dependent launch:
the kernel waits for its predecessor only right before it reads the QKVZ projection's rows and
the normed hidden, and releases its successor at entry). The caller then owes the obligation in
the config docstrings: the launch's immediate predecessor must not write this layer's state,
conv state or slot indices. The default forms keep PDL off.

## Validation

`test/python/linear_attention/frost/test_gdn_decode_fused.py` and `test_gdn_verify_fused.py`
cover the host entries (typed declines, the plan-time compile key, the knob validation, the
shared-memory budget per tokens count on the cc 10.0 and cc 12.0 readings, the budget-derived
pick tables of both parts, the geometry envelope and the per-rank geometries' derived constants,
operand shapes and BA-partition coverage, the DSL and architecture gates) and, on an SM100-class or SM120-class
device, the null-slot and padding contracts, determinism, CUDA-graph replay over distinct
layers, a torch reference of the fused chain and the bitwise agreement of every knob arm the part's
shared memory holds with the default pick (an arm the part does not hold is its typed decline,
reported as a skip). When `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR` names a directory of saved verify
operand sets with the reference kernel's outputs (`set_<tag>.pt` / `ref_<tag>.pt`), every set is
replayed and compared byte for byte.

`test/python/linear_attention/frost/test_flashinfer_gdn_backend.py` runs
FlashInfer's own decode and verify tests (`tests/experimental/qwen_3_5_gdn_block/`)
against `frost_fe` when a FlashInfer source tree is next to the installed package --
the verify reference also under the 148-SM launch policy and with the SM count held
below one request's grid (the beyond-the-grid regime's budget-fitting form at every shape: the
two-CTAs-per-SM form on a cc 10.0 part, the 2-deep-ring pair on a cc 12.0 part) -- and checks the backend bitwise
against the built-in `frost` backend where that one is served: decode on its
reference cases, verify on the benchmark's own input sets at requests 1 / 2 / 4 x 6
tokens (every token accepted, and random acceptance with a padding request).

## Performance

One GPU class so far, GB200 (1200 W, 152 SMs, SM clock locked at 1830 MHz, one exclusive GPU) -- the cc 12.0
admission ships with its functional validation only; a GeForce board allows no clock lock, so an RTX 5090 table
will be a separate table with its own (boost-clock, interleaved) protocol stated beside it (the one cc 12.0 figure on
this page, the verify pick order past the one-CTA grid, is such a boost-clock A/B/A reading with its protocol stated
inline, not a table entry). CUPTI
kernel-duration medians inside a serving step at batch 1 (cold code and data, as every layer's
launch finds them), speed-ups positive = faster (`baseline / new - 1`):

| step | FlashInfer `frost` backend | `frost_fe` (this page) | speed-up |
|---|---|---|---|
| GDN decode step, 1 request | 8.58 us | 4.90 us | +75.1 % |
| GDN MTP verify step, T = 6, 1 request | 15.01 us | 12.77 us | +17.5 % |

Outputs were bitwise identical to the `frost` backend on every saved reference set and in
serving (identical tokens and acceptance counters). End to end on the same path (three
interleaved rounds, acceptance-length gate), the time per output token improved by +6.4 % with
MTP (five drafts, batch 1) and +6.9 % without; those figures include the PDL launch mode of the
fused steps and the GEMM tactic changes made in the same path, which are not part of this page's
kernels (the kernels alone account for +2.0 % and +4.6 %, read against a same-snapshot control).
