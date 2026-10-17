# SDPA backward, d = 128 (SM100)

The opt-in `sdpa_bwd_sm100_d128` engine serves dense multi-head attention with
BF16 or FP16 inputs and gradients, compact BSHD physical storage, and
contiguous FP32 natural-log Stats. It supports unmasked and top-left causal
attention, including rectangular query and key/value lengths. Packed THD MHA
supports both dtypes. Dense and THD GQA/MQA initially support BF16 only. Packed
and grouped paths require exact SM100 (cc 10.0), CuTeDSL 4.8 or newer and
positive FP32 attention scale. Grouped dense lengths must be multiples of 128;
packed grouped lengths may have tails or empty actual sequences. Other masks,
bias, sinks, dropout and deterministic requests are declined.

Use the ordinary cuDNN graph preparation and execution API. After building
and selecting a plan, allocate the size reported by that plan:

```python
workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=q.device)
graph.execute(variant_pack, workspace)
```

Plan construction selects the implementation from declarations and the build
device. On SM100 with public CuTeDSL 4.8 or newer, positive attention scale and
both sequence lengths divisible by 128, the MHA three-kernel chain computes the row
delta while initializing FP32 dQ scratch, accumulates dQ and writes dK/dV,
then converts dQ to the output type. The output conversion applies no extra
scale. Its scratch size is `4 * B * H * S_q * 129 + 240` bytes. The final 240
bytes cover internal alignment for any valid 16-byte-aligned caller workspace.

Other dense MHA declarations in the existing envelope, including sequence tails and
negative finite nonzero scales, retain the one-kernel implementation with
zero workspace. Always query the plan's size rather than assuming either
implementation. Zero and nonfinite scales are unsupported.

Dense MHA declarations are bounded before either implementation is selected:
`max(S_q,S_kv) <= 2**31-128`, `B*H*S_q <= 2**31-1`,
`B*H*(ceil(S_q/128)+ceil(S_kv/128)) <= 2**31-1`, and
`max(S_q,S_kv)*H*128*2 < 2**40`. These conservative limits keep tile rounding,
Stats indexing, the recompute launch grid and tensor-map byte strides
representable. Exceeding a limit declines the engine before compilation.

The accumulation implementation requires both attention scale and its product
with log2(e) to remain finite and positive in FP32. A scalar override on an
already compiled accumulation plan must satisfy that requirement. Invalid
overrides raise before any kernel launches; they do not change the plan.

Inputs, outputs and workspace must remain on the declared CUDA device.
Gradient outputs must not overlap each other or any input, and workspace
must not overlap any operand. Execution binds the current buffers and stream,
initializes its scratch each time, and performs no allocation, layout copy or
device-to-host read. The plan supports CUDA-graph capture and replay. FP32 dQ
atomic accumulation does not promise bitwise repeatability.

The three-kernel implementation is selected only on SM100. Existing support
on other pre-Rubin Blackwell devices retains the previous implementation.
The engine remains opt-in while performance qualification proceeds.

For packed THD, declare positive `max_total_seq_len_q` and
`max_total_seq_len_kv` capacities and device sequence lengths. Each port's
ragged offsets retain their declared origins, including nonzero initial
offsets and gaps. Packed rows require compact head/token strides; the logical
batch stride is unused. FP32 Stats may be packed token-major `(T,H)` or
head-major `(1,H,head_stride)`, with the latter covering the physical query
capacity. Actual sequences may be empty, and lengths need not divide 128.

The packed MHA plan launches metadata/tensor-map setup, Delta plus dQ
initialization, the backward main kernel, then dQ conversion. Its caller
workspace holds FP32 dQ and Delta, sequence metadata, per-port origins and
seven tensor maps per batch; it has no dK/dV workspace. Query the plan's
workspace size. Execution reads lengths and offsets on the device and
performs no allocation, tensor conversion or host read of metadata. Gradient
outputs and workspace must also be disjoint from all metadata operands.
Declared capacities must be positive even when every actual sequence is
empty. This path requires exact cc 10.0 and both scale and scale*log2(e)
finite and positive in FP32.

GQA/MQA requires `Hq > Hkv > 0` and an integral `Hq/Hkv`; ratios need not
be powers of two. Each query head uses KV head `q_head // (Hq/Hkv)` and writes
its own BF16 dK/dV partials. A final kernel sums them in fixed query-head order using FP32
and converts to the native gradient dtype. Thus partial storage rounds each
query head's contribution before the group fold. Regular correctness tests
retain the same numerical tolerances as MHA.

Grouped plans append two caller-workspace regions: dense
`[B,Skv,Hq,128]` or packed `[1,round128(Tkv_compact),Hq,128]`, each at two
bytes per element. Packed partials use compact KV prefixes; the final fold
writes each dK/dV port at its independent declared origin. Dense grouped
execution has four launches; packed grouped execution has five. Their dQ
scratch, device metadata, alias checks, rebinding and capture contract match
the corresponding MHA plan. Always query workspace size from the plan.

Grouped support and packed support do not imply performance parity with
another backend; that qualification is a separate step.

FP16 grouped attention is declined because per-query-head partials can overflow
FP16 before cancellation in the final group sum. Wider partial storage is
required before enabling that path. FP16 dense and packed MHA remain supported.
