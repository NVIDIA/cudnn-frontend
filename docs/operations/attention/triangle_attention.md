# Triangle Attention (SM107)

**This is an experimental API and subject to change.**

## Overview

`cudnn.triangle_attention` is the AlphaFold-class triangle attention (AlphaFold 3 Algorithms 14 and 15: attention over one axis
of the pair representation with a pair bias on every score and a key mask) behind the signature of
`cuequivariance_ops_torch.triangle_attention` (pinned to cuequivariance-ops-torch **0.12.0**: the same parameter names, order and
defaults, so a model switches providers by changing one import).

```text
q, k, v    (B, N, H, S, D)      bf16 / fp16     o[b, n, h, i] = sum_j softmax_j( scale * <q_i, k_j> + bias[b, h, i, j] ) v_j     over the live keys j
bias       (B, 1, H, S_q, S_kv) the same dtype  the pair bias of outer entry b, shared by its N rows
mask       (B, N, 1, 1, S_kv)   bool            a dense key mask (False = the key is masked for every query of that row)      -- or --
kv_lengths (B, N, 1, 1, 1)      int32           right-padded lengths: keys j < kv_lengths[b, n] are live (never both)
mask, mask_is_prefix=True       bool            the same mask ASSERTED prefix-shaped (True on [0, len), False after): its row lengths are
                                                derived on the device -- no host synchronisation -- and the call rides the kv_lengths route
-> o (B, N, H, S_q, D) in q's dtype
```

The attention core is **not a new kernel**: it is the shipped `sdpa` graph node served by the FROST engine row
`sdpa_fwd_prefill_sm107` on its 32- / 64-wide head-dim tiles -- the additive bias held as a resident SMEM row-block (or streamed
per kv tile above the resident budget), the boolean key mask folded into the kernel's bit-word mask, the right-padded lengths through
the `seq_len_kv` padding path at zero kernel cost. `cudnn.triangle_attention` is the API layer: it flattens `(B, N)` onto the graph's
batch axis (the `N` rows of an outer entry share one `[1, H, S_q, S_kv]` bias -- the graph's existing bias broadcast; one launch per
outer entry), chooses the mask path from the arguments, caches one compiled plan per declaration and launches it on the caller's
stream. The engine row's `Capabilities` and its `mismatch()` are the source of every typed decline below; the same row serves the graph
form (`graph.sdpa(q, k, v, bias=..., kv_mask=... / seq_len_kv=...)`) with no wrapper.

**Target: NVIDIA Rubin (SM107, compute capability 10.7) only.** Every other architecture is declined with a typed error
(`NotImplementedError`), never served slowly. The engine row is opt-in: `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` must be set before
`import cudnn` (the call declines with that instruction otherwise).

## Mathematics and numerics

For every outer entry `b`, row `n`, head `h` and query `i`, over the keys `j` the mask keeps (all keys without a mask):

```text
s_j = scale * <q[b, n, h, i], k[b, n, h, j]> + bias[b, 0, h, i, j]       scale = 1 / sqrt(D) unless given
o   = sum_j softmax_j(s) v[b, n, h, j]
lse = max_j s_j + log( sum_j exp(s_j - max_j s_j) )                       (fp32, natural log; the class API's optional output)
```

`k` and `v` may carry fewer heads than `q` (`H_kv` dividing `H`, the GQA fold); the bias may be declared with a head extent of 1
(broadcast over the heads) and a batch extent of 1 (shared by every outer entry).

Rounding points: the scores accumulate in fp32 (the MMA), the bias enters the fp32 score by one FFMA of `bias * (1 / scale)` before the
scale multiply (one more fp32 rounding than the eager `s * scale + bias`), the softmax statistics are fp32, `P` is rounded once to the
io dtype as the second MMA's operand, `o` accumulates in fp32 and is rounded once to the io dtype at the store. **The bias is read in the
activations' dtype** (bf16 with bf16, fp16 with fp16): the resident block loads the io dtype, so an fp32 bias is **cast once per call by
the function** (counted, see the layouts section) -- never an fp32 arm of the kernel. `cudnn.triangle_attention.triangle_attention_reference`
is the fp64 reference with this contract (pass it the bf16 bias the kernel reads). Measured on the test grid (`S` in 64..512 incl. a
217 tail, `D` in {24, 32, 48, 64}, `B` in {1, 2}, the three mask forms, bf16 and fp16) against that reference: `o` within the engine
row's own tolerance `atol 5e-2 / rtol 3e-2` (an fp32 accumulation, one 16-bit rounding at the store), the LSE within `1e-4 / 1e-4` --
the SDPA row's tolerances, not widened for this layer.

### Masked rows: three conventions, documented here, one chosen

A row with **no live key** (a zero `kv_lengths`, a mask row that is all False) is read differently by the three implementations a
model may have been validated against. This API follows the FROST SDPA invariant for an empty key range; the others are stated so a
parity test knows what it compares (compare rows **with** keys only):

| implementation | `o` on a row without a live key | `lse` | `max` |
|---|---|---|---|
| **`cudnn.triangle_attention`** (both mask forms) | **exactly 0** | **-inf** | **-inf** |
| cuEquivariance, `kv_lengths` route | 0 (the row is zeroed by a device epilogue) | 0 | 0 |
| cuEquivariance, dense-mask route | the uniform average of `v` (a finite `-1e9` is folded into the scores) | -1e9 | -1e9 |

So `mask_to_kv_lengths(mask)` -- the conversion of a prefix-shaped mask into lengths -- changes what a fully masked row reads **on
cuEquivariance** (uniform average -> zeros) and nothing here (both forms read 0 / -inf). Wherever such rows are padding, which the
models mask out of the loss and of the next layer, the difference is invisible; a model that consumes fully masked rows should be
checked against the table. The same contract holds in the forward's statistics, so a backward recomputing `P = exp(s - lse)` from the
published `lse` yields no gradient into masked keys.

## Served today, and the typed declines

| what | served | otherwise |
|---|---|---|
| architecture | compute capability 10.7 (Rubin), the FROST engine row `sdpa_fwd_prefill_sm107` with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` | `NotImplementedError` (the instruction / the device found) |
| dtype | bf16 or fp16 activations, the bias in the same dtype; the function casts an fp32 bias once per call (counted) | `NotImplementedError` |
| head dim `D` | `D % 8 == 0` and `D <= 64` (the row's 32- and 64-wide tiles; 8 / 16 / 24 -> the 32-wide tile, 40 / 48 / 56 -> the 64-wide one, by TMA zero fill) | `NotImplementedError` quoting the engine (`D = 20`: the 16-byte stride rule; `D = 128`: no bias form on the d128 tile) |
| `S_q`, `S_kv` | any; a compact bias needs `S_kv % 8 == 0` (the TMA 16-byte row-stride rule) -- the function pads the row otherwise (counted); the bias is resident in SMEM up to the engine's budget per tile (768 kv columns at `D <= 32` for a bias shared by two or more rows of an entry, 384 at `D <= 64`; 384 / 128..256 for a per-entry bias on a single row) and streamed per kv tile above it | `NotImplementedError` quoting the engine where neither bias form fits |
| `B`, `N`, `H` | any (`B > 1` runs one launch per outer entry; `H_kv` dividing `H`) | `ValueError` on a mismatch |
| key mask | `mask` (bool, dense -- the in-kernel bit-word path) OR `kv_lengths` (int32, right-padded -- the padding path) OR `mask` with `mask_is_prefix=True` (the mask asserted prefix-shaped: lengths derived on the device, the padding path, no host synchronisation); `mask_to_kv_lengths` converts a prefix-shaped mask with one synchronisation, `prefix_mask_to_kv_lengths` without one | both: `ValueError` (cuEquivariance's rule); a non-bool mask / non-int32 lengths: `ValueError`; `mask_is_prefix` without a mask: `ValueError`; a non-prefix mask under the assertion: `ValueError` naming the rows ONLY with the debug check on (below) |
| `scale` | any positive finite float (default `1 / sqrt(D)`) | `ValueError` |
| `return_aux` | `False` only: the row max is a side output the engine's bias arm does not store yet (the d128 tile stores it, the small tiles' bias arm declines it) | `NotImplementedError` quoting the engine ("without the score side outputs") |
| layout | zero-copy on the token-major form (below); every other layout is repacked by the function with a counted copy | the class API: `NotImplementedError` naming the tensor (never a hidden copy) |
| device | every `execute` tensor on the device the plan was declared on | `ValueError` |
| autograd | the forward is served under grad mode (a `torch.autograd.Function` that saves `q, k, v, bias, o, lse`); the backward is a later item of this API | `NotImplementedError` from `.backward()` (never a silent zero gradient) |

## Drop-in function

```python
import cudnn
from cudnn import triangle_attention, mask_to_kv_lengths   # or: from cuequivariance_torch import triangle_attention

o = triangle_attention(q, k, v, bias, mask=mask, scale=scale)                 # the Boltz-2 / Protenix / OpenFold-3 call, unchanged
o = triangle_attention(q, k, v, bias, scale=scale, kv_lengths=mask_to_kv_lengths(mask))   # a prefix mask as lengths (one host sync per mask)
o = triangle_attention(q, k, v, bias, mask=mask, scale=scale, mask_is_prefix=True)         # the same, derived on the device: no host sync
```

Signature (cuEquivariance 0.12.0, verbatim -- `kv_lengths` is its eighth parameter, positional-or-keyword there as here; the four
keyword-only additions are appended, in the order they were added):

```python
triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False, kv_lengths=None,
                   *, out=None, stream=None, mask_is_prefix=False, check_prefix_mask=None) -> torch.Tensor
```

`out` writes the result into a caller-provided tensor (stable addresses for CUDA-graph users; copied into when it is not in the
kernel's layout, counted); `stream` is a raw CUDA stream handle the allocations, copies and launches are ordered on (None = torch's
current stream). The function caches one compiled plan per `(the q / k / v / bias shapes and strides, dtype, mask form, outputs, scale)` in the process
(the `k` / `v` shapes carry `H_kv` and `S_kv`: a KV head slice of the same buffer is its own plan),
allocates the output per call (and the workspace `get_workspace_size()` names: the plan's own bytes plus, on the `kv_lengths` form, the
Q-side lengths below -- the torch allocator recycles it), and launches. Warm a shape once before capturing a
`torch.cuda.graph` around the call; the captured graph replays the kernel and follows the inputs' contents.

`mask_to_kv_lengths(mask)` returns `(B, N, 1, 1, 1)` int32 lengths when every row of the boolean mask is a prefix (True on `[0, len)`,
False after), else `None` so the caller falls back to `mask=`. It performs one host synchronisation (the prefix test): call it once per
mask, not once per layer.

### The prefix-mask form: `mask_is_prefix=True`

The models' key masks are padding masks -- every row True on `[0, len)` and False after -- and the padding (`kv_lengths`) route of the
kernel trims the kv loop to the live tiles where the dense-mask route visits every tile. `mask_to_kv_lengths` buys that route at the
price of one host synchronisation per mask (the prefix test is a device-wide `all`); `mask_is_prefix=True` buys it without one:

- **The contract.** The CALLER asserts the mask is prefix-shaped. The API computes each row's True count on the device -- ONE launch of a
  small row-popcount kernel (a warp per row, byte loads, `popc(ballot)`, one int32 store into a lengths chunk of the workspace: no cast
  pass, no staging, no allocation, no host synchronisation, captured by a CUDA graph) -- and binds the counts as the `kv_lengths` form.
  The kernel is arch-generic (it runs on every CUDA device the DSL targets) and serves any mask layout; the mask is never read on the
  host, so a captured graph re-derives the lengths from the mask's contents at every replay.
- **What a violation does.** The count of a non-prefix row is still a length: such a row is served as the prefix of its True count --
  the WRONG keys, no error. A mask that may not be prefix-shaped belongs on `mask=` (the in-kernel dense mask) or on
  `mask_to_kv_lengths`, which tests and falls back.
- **The debug check.** `check_prefix_mask=True` (per call) or `CUDNN_FRONTEND_TRIANGLE_ATTENTION_CHECK_PREFIX_MASK=1` (the environment,
  for every call that does not decide it itself) runs the prefix test after the derivation -- the device-wide test with its one host
  synchronisation -- and raises a typed `ValueError` naming the violating `(b, n)` rows. Off by default; never on inside a CUDA-graph
  capture (`make_cuda_graph` runs it on the warm-up calls only, as the environment decides, and never on the captured call).
- **Masked rows.** Unchanged: a row whose count is 0 reads `o = 0`, `lse = max = -inf` -- the `kv_lengths` route's convention, which on
  this provider is also the dense mask's (the table above).
- **The device-side helper.** A caller that holds a mask and a plan of its own (a `kv_lengths=True` plan, the graph's `seq_len_kv`, or
  cuEquivariance's `kv_lengths`) gets the same derivation from
  `prefix_mask_to_kv_lengths(mask, out=None, *, workspace=None, check=None, stream=None) -> (B, N, 1, 1, 1) int32`: with `out` (an int32
  `(B, N, 1, 1, 1)` contiguous tensor) the call allocates nothing and a CUDA graph captures it; without it the output is allocated (the
  convenience form); `workspace` is accepted for compatibility with the two-kernel form this function shipped with first and unused
  (`prefix_mask_workspace_bytes(B, N, S_kv)` is 0: the fused kernel stages nothing); `check` is the debug check (None = the
  environment); `stream` the raw handle or torch stream the launch is ordered on. Any mask layout is served (a batch slice, a padded
  row stride, a strided key view); a CPU mask takes torch's reduction; a DSL below the floor is a typed `NotImplementedError` naming the
  version. It is the sync-free twin of `mask_to_kv_lengths`: that one TESTS the shape and returns `None` when it fails, this one TRUSTS
  the assertion.

### Layouts: the zero-copy forms and the counted fallbacks

The kernel's TMA descriptors are built from the declared strides, so the layout is part of the plan. Zero-copy:

- **activations** in the **`(B, N, S, H, D)` memory form** presented as `(B, N, H, S, D)` -- the head dim contiguous, the token
  stride `>= H * head stride`, every stride a 16-byte multiple (a permuted view of a `(B, N, S, H, D)` buffer, compact or padded with
  gaps; the "layout_left" form the models produce by projecting `z` and permuting);
- the **bias** head-major with a **contiguous key dim** and 16-byte row strides (`S_kv % 8 == 0`, or a padded row stride);
- a **contiguous mask / lengths** tensor -- at any base address for the byte mask (the kernel reads it one byte per lane) and
  word-aligned for the int32 lengths, so a per-entry slice `mask[b:b+1]` / `kv_lengths[b:b+1]` binds as it is.

Anything else is repacked by the function with ONE copy per tensor per call, **counted** and announced once per layout
(`RuntimeWarning` naming the tensor and its strides): `cudnn.triangle_attention.copy_counters()` reports `fallback_copies` and the
per-kind counts (`q_repack` / `k_repack` / `v_repack`, `o_repack` -- the output copied back into `q`'s own layout when `q` was
head-major, `bias_repack` -- a non-contiguous key dim or the `S_kv % 8 != 0` pad, `bias_cast` -- an fp32 bias, `mask_repack`,
`kv_lengths_repack`, `out_copy`); `reset_copy_counters()` clears them. The output follows `q`: a zero-copy `q` yields `o` in the same
memory form (which is the caller's), a repacked `q` yields `o` in `q`'s own layout. A model on its native layout should read 0
everywhere except the documented bias pad at `S_kv % 8 != 0`. The prepared class API never copies: a layout it cannot bind is a typed
decline from `check_support`.

## Prepared class API

```python
from cudnn.triangle_attention import TriangleAttentionFwd

op = TriangleAttentionFwd(sample_q, sample_k, sample_v, sample_bias, kv_lengths=True, lse=True)   # tensors or TensorDesc declarations
op.check_support()                       # typed declines (malformed -> ValueError; unserved -> NotImplementedError quoting the engine row)
op.compile()                             # the FROST row's plan, pinned by name; op.plan_name / op.knobs record what was built
ws = torch.empty(max(op.get_workspace_size(), 1), dtype=torch.uint8, device=q.device)
op.execute(q, k, v, bias, kv_lengths, o, lse, None, ws if op.get_workspace_size() else None, stream)   # no allocation, no conversion, no sync
graph = op.make_cuda_graph(q, k, v, bias, kv_lengths, o, lse, ws)                                  # one graph launch per replay

op_p = TriangleAttentionFwd(sample_q, sample_k, sample_v, sample_bias, mask=True, mask_is_prefix=True)   # the prefix-mask form
op_p.check_support(); op_p.compile()
ws_p = torch.empty(op_p.get_workspace_size(), dtype=torch.uint8, device=q.device)                 # + the derived KV lengths
op_p.execute(q, k, v, bias, None, o, None, None, ws_p, stream, mask=mask)                         # lengths derived on the device, no sync
op_p.execute(q, k, v, bias, None, o, None, None, ws_p, stream, mask=mask, check_prefix_mask=True)  # + the debug check (one sync, typed error)
```

Constructor flags: `kv_lengths=True` / `mask=True` (the key-mask form; the dense form is passed to `execute(..., mask=mask)`),
`mask_is_prefix=True` (with `mask=True`: the boolean mask asserted prefix-shaped, its row lengths derived on the device at `execute`
and served through the `kv_lengths` graph form -- the same compiled plan as `kv_lengths=True`), `lse=True` (the fp32 `(B, N, H, S_q)`
log-sum-exp), `score_max=True` (declined today, see the table), `scale`, `sample_o` (the output's layout; None = the compact
`(B, N, S, H, D)` form). `execute` validates every tensor against the declaration -- device, dtype, shape,
strides, the base alignment each operand's reader needs (16 bytes for the TMA-read `q` / `k` / `v` / `bias` / `o`, 4 bytes for the
int32 lengths and the fp32 `lse`, none for the byte mask), the mask form, and that `o`, the `lse` and the workspace intersect, as byte
ranges, no tensor the kernel reads and not each other (`ValueError` naming the two operands; inputs may overlap one another) -- and
launches once per outer batch entry
(`launches_per_call`); its keyword `check_prefix_mask` (the prefix-mask form only; True / False / None = the environment decides) is the
debug check above. A plan is bound to the device it was declared on; declare and compile one per device. The prepared API owns
no device memory. The graph's padding contract takes a length tensor on both sides and the kernel bounds the live Q rows by `seq_len_q`;
every Q row of a triangle-attention call is live, so on the `kv_lengths` and prefix-mask forms `get_workspace_size()` adds `N` int32 words
that `execute` fills with `S_q` on its stream before the launches -- one driver memset, stream-ordered, a memset node in a captured CUDA
graph; the prefix-mask form adds `B * N` int32 words for the KV lengths it derives (the fused kernel reads the boolean mask directly and
stages nothing). The workspace is sized and carved by the SDPA adapters' shared recipe (`ws_align` 128-byte chunks dealt in order by
`WorkspaceCarver`: the plan's own bytes first, the Q-side lengths next, then the derived KV lengths); a missing, undersized or misaligned
buffer raises the carver's own error -- its base rule is
16-byte alignment, which a torch allocation satisfies (the lengths' reader itself needs 4) -- and the buffer must be contiguous and on the
plan's device. Nothing is allocated at build or at execute, and no per-plan tensor exists to be released at garbage-collection time inside
someone else's capture; the drop-in function allocates that workspace per call.

## Performance

The kernel-side numbers of the bias arm (the resident block, the pair work item, the streamed ring; the `D = 32` tile against the
zero-filled d128 tile) are the SDPA engine row's, recorded in `python/cudnn/sdpa/frost/SUPPORT_MATRIX_TRACKER.md` and
`python/cudnn/sdpa/AGENTS.md`. The API layer adds the host overhead of one eager call. Its host path: the engine's prepared launch binds
its argument frame ONCE per plan (the engine's own binder runs on the first call and applies every dtype / layout / alignment / span
rule of the row); every later call re-checks the tensors against the declaration and refreshes the frame's pointer slots (the geometry
is the declaration's), then issues the kernel -- so an eager class-API call costs the API's checks plus the launch itself, and the
drop-in function adds the layout decisions, the plan lookup and the output allocation. The bias / per-key-mask arms of the engine
have no native (C) binder today, which is why the frame is bound in Python once rather than per call. A benchmark package with the
host floor as a first-class column (eager CPU wall without a synchronisation vs the captured graph's replay, min and median over
hundreds of pinned calls) and the comparison against cuEquivariance on Rubin follows this API.

**The prefix-mask form's price, measured** (the 212-SM Rubin perf-node class at the 2376 MHz lock, sampled 2364 MHz; D 32, H 4, bf16, N = S
rows; CUDA-graph replay of the class API's call, 100 x 3 with a control pair; every route bitwise equal on every cell): the derivation is ONE
kernel of 2.2-3.6 us (a warp per row, byte loads, `popc(ballot)`; its replay alone 4.6 / 5.2 / 5.2 / 6.0 us at S = 256 / 384 / 512 / 768,
where the two torch kernels it replaced -- the mask's cast and the row count -- read 9.6 / 9.8 / 10.4 / 10.9 us), so the prefix form costs
+2.3 / +2.9 / +2.9 / +4.0 us over the explicit `kv_lengths` route at 75 % live keys: -9.0 / -4.8 / -2.7 / -1.3 % (23.2 -> 25.5, 57.3 -> 60.3,
103.4 -> 106.3, 308.0 -> 311.9 us; at 50 % live keys +3.0 / +3.9 us = -3.8 / -1.8 % at S = 512 / 768), where the two-kernel form cost
+8.1 / +8.5 / +9.0 / +9.6 us (-25.8 / -12.9 / -8.0 / -3.0 %).  Against the in-kernel dense mask fed the SAME mask the prefix form wins at every
crop: +7.8 / +16.7 / +33.0 / +24.2 % at 75 % live keys (27.5 / 70.3 / 141.4 / 387.3 us: the dense route visits every kv tile, the padded route
the live ones) and +77.8 / +73.5 % at 50 % (S = 512 / 768).  Host floor of the class API's `execute` (CPU-pinned, 500 calls, min / median):
`kv_lengths` 19-20 / 20-23 us, the prefix form 26 / 27 us (the two-kernel form read 43-45 / 45-48 us), the dense mask 11-12 / 12 us;
the derivation alone 3.8-4.1 / 3.9-5.9 us (the two torch kernels 8.5-8.9 / 8.8-9.4 us); the drop-in function adds ~9 us to each (the output
and workspace allocations).  So: a caller who holds lengths passes `kv_lengths`; one who holds a prefix mask uses `mask_is_prefix=True` --
within one launch (~3 us) of the lengths route and ahead of the dense mask at every training crop; `mask_to_kv_lengths` remains for a caller
who prefers the tested conversion with its one host synchronisation per mask.

## Tests

`test/python/triangle_attention/cutedsl/test_api.py` (L0): the fp64 reference over the acceptance grid (`S` x `D` x the three mask
forms at `B = 1`, empty rows on both forms), the head dims served by zero fill, fp16 / GQA / a head-broadcast bias / `S_q != S_kv`, the prefix mask vs
`kv_lengths` (bitwise), the prefix-mask form (`mask_is_prefix=True`) bitwise against both lengths routes on two shapes (the 217 tail at
D 32, 512 at D 64) with rows of every length incl. empty (o and lse, the derived lengths read back from the caller's workspace), its debug check on a holed mask (the
kwarg and the environment variable; the check off = the documented count behaviour, stated by the test), its allocation-free and
synchronisation-free execute (the allocator's allocation count unchanged), its CUDA-graph capture following the mask's contents, the
device-side helper on CPU tensors, the class API's LSE and the masked-row contract, `B = 2` against the single-entry calls (bitwise), the
single-entry slices of a `B = 2` batch at `S = 217` on both surfaces (`mask[1:]`, `kv_lengths[1:]`, `lse[1:]` bind as they are;
`kv_lengths` passed positionally), the layout
fallbacks (head-major activations: 4 counted copies, the output following `q`; the permuted / fp32 / tail bias: 1 counted copy each),
`out=` / `stream=`, the execute-path contract (no allocation, no synchronisation; on both key-mask forms), CUDA-graph capture and
replay (the `kv_lengths` form's workspace fill replaying and following new lengths), `return_aux` and the autograd backward as typed
declines, the live rejects (a `kv_lengths` plan without its workspace, a misaligned and a strided one), and the GPU-free contract tests (the
workspace sizing and carve by the shared SDPA recipe, the lazy
export in every import order, no torch at `import cudnn.triangle_attention`, the signature pin, `mask_to_kv_lengths`, the reference, the
declarations declined before any device work, the plan-device predicate).
