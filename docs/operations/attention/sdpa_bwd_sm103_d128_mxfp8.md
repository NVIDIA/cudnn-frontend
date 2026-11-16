# SDPA MXFP8 backward, D = 128 on SM103 (`sdpa_bwd_sm103_d128_mxfp8`)

This engine addresses the backward-performance portion of
[issue #659](https://github.com/NVIDIA/cudnn-frontend/issues/659); it does not add variable-length attention.

The opt-in `sdpa_bwd_sm103_d128_mxfp8` engine serves the block-scale MXFP8 backward graph
(`sdpa_mxfp8_backward()`) at head dimension 128 on SM103 (B300, cc 10.3):

* E4M3 payloads (`q`, `q_T`, `k`, `k_T`, `v`, `dO`, `dO_T`), BF16 `o_f16` / `dO_f16` / dQ / dK / dV, and the seven
  F8_128x4 E8M0 scale tensors, read in place;
* `S_q == S_kv` of any length, any batch, MHA / GQA / MQA, unmasked or causal (`use_causal_mask=True`, or
  `use_causal_mask_bottom_right=True`, which is the same mask here because `S_q == S_kv`);
* BHSD-compact storage for every payload, half-precision tensor and gradient (strides `(H*S*D, S*D, D, 1)`);
* a finite attention scale in (0, 2^12];
* `amax_dQ` / `amax_dK` / `amax_dV` when the graph requests them (the maximum |x| of the FP32 values before the BF16
  cast).

Windowed and band-widened masks, padding, THD, bias / dBias, sink / dSink, dropout, E5M2, an FP16 half side,
unequal lengths, other layouts and decode-shaped graphs are declined at plan time, so the graph falls back to another
engine.

## Enabling and selecting it

The row is opt-in while it matures:

```bash
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
```

Use the ordinary graph preparation and execution API. Heuristic mode A offers the row when the graph qualifies; to pin
it, select its plan by name:

```python
graph.create_execution_plans([cudnn.heur_mode.A])
names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
graph.select_plan(names.index("sdpa_bwd_sm103_d128_mxfp8"))
graph.check_support()
graph.build_plans()
workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
graph.execute(variant_pack, workspace)
```

Building the plan compiles the CuTeDSL chain for the problem (S, B, H_q, H_kv, attention scale, amax, causal) and the
plan's walk split; FROST's compiled-object cache keeps it on disk, so later builds of the same problem load it.

## How it computes

Three kernels run on the caller's stream, chained by programmatic dependent launch:

1. **prep**: delta = rowsum(O * dO_f16), the -LSE and -delta terms the dK/dV kernel adds, a copy of the LSE for the dQ
   kernel, and resets of the dQ work counter and the amax outputs;
2. **dK/dV**: one CTA per 128-key block walks every query tile of its GQA group with dK / dV accumulated in tensor
   memory; 2- or 4-CTA clusters share each Q / dO tile by TMA multicast. At small S two CTAs split each key block's
   query walk and add their FP32 partials in a fixed order;
3. **dQ**: a persistent grid of 2-CTA clusters pulls (head, 256 query rows) units from the counter.

Numerics follow the cuDNN MXFP8 backward contract: P with a fixed 2^-8 scale for dV; dS quantized to E4M3 with one E8M0
scale per 32-element block for the dK and dQ products.

Launch grids use the device's resident cluster capacities queried at plan time. The walk split
is selected when its tasks fit one wave of four-CTA clusters; deterministic mode makes this
decision per batch element.

Per-head tensor maps clip loads and stores at S. Padded query rows use -inf LSE;
dQ masks keys past S. Causal variants skip fully masked tiles and mask diagonal tiles.
No operand staging copies are needed.

Workspace: `16 * B * H_q * round_up(S, 256) + 272` bytes.

## Determinism

Every output tile has one writer and the amax reduction is order-free, so every plan is bitwise repeatable, including
under CUDA-graph replay, concurrent work on other streams and fresh allocations. The only plan choice that changes
rounding is the dK/dV walk split:

* with `use_deterministic_algorithm=True` it is decided from one batch element, so results are also **batch invariant**
  (each batch element's gradients are bitwise those of the B = 1 problem on its data);
* otherwise it is decided over the whole batch, which keeps a large batch off the split's extra partial exchange
  (B = 8, S = 1024, 88 / 8 heads: 0.600 ms vs 0.641 ms per call; see the B = 8 table below).

## amax

`amax_dQ` / `amax_dK` / `amax_dV` are produced when the graph makes them real outputs (any subset): the maximum |x| of
the FP32 values before the BF16 cast, reduced with order-free atomics. A graph without them compiles a body without the
reduction. Requesting them costs at most 1.6 % (S = 1K dense; at most 0.7 % at the other sizes and masks; see the amax
table below).

## Performance

B300 (148 SMs), H_q = 88, H_kv = 8, cuDNN 9.28, CuTe DSL 4.8.1: speedup over the cuDNN backend engine on the same graph
(`benchmark/attention_training/benchmark_mxfp8_bwd_sm103_d128.py`). Times are cold, the convention of the repo's
attention benchmarks: each call is a single CUDA-graph launch after an L2 flush and an idle gap, median of 7 calls,
timed end to end with CUDA events. The row's three kernels overlap under programmatic dependent launch, so a sum of
per-kernel profiler times, the timer of `benchmark_single_sdpa.py`, overstates it (B = 1, S = 8192: 6.3 ms summed for a
call that takes 3.8 ms); that script also builds BSHD tensors, which this row declines. The `sdpa_bwd_sm100_mxfp8` row
reads BSHD storage, so it runs on a BSHD copy of the operands.

#### B = 1, Hq = 88, Hkv = 8, no amax, default determinism

| S | mask | backend ms | **new row** ms | **new row speedup** | FROST sm100_mxfp8 speedup |
|---:|---|---:|---:|---:|---:|
| 1000 | dense | 0.2084 | 0.1149 | **1.81** | 1.01 |
| 1000 | causal | 0.2006 | 0.0976 | **2.05** | 1.09 |
| 1024 | dense | 0.1970 | 0.1140 | **1.73** | 0.96 |
| 1024 | causal | 0.1716 | 0.0979 | **1.75** | 0.92 |
| 2000 | dense | 0.5399 | 0.2928 | **1.84** | 1.26 |
| 2000 | causal | 0.4579 | 0.2124 | **2.16** | 1.29 |
| 2048 | dense | 0.4951 | 0.2899 | **1.71** | 1.16 |
| 2048 | causal | 0.3672 | 0.2159 | **1.70** | 1.03 |
| 4000 | dense | 1.671 | 1.000 | **1.67** | 1.14 |
| 4000 | causal | 1.258 | 0.6161 | **2.04** | 1.62 |
| 4096 | dense | 1.499 | 1.003 | **1.49** | 1.02 |
| 4096 | causal | 0.9511 | 0.6161 | **1.54** | 1.22 |
| 8192 | dense | 5.147 | 3.850 | **1.34** | 0.94 |
| 8192 | causal | 2.950 | 2.187 | **1.35** | 1.10 |
| 16384 | dense | 19.269 | 15.168 | **1.27** | 0.97 |
| 16384 | causal | 10.339 | 8.366 | **1.24** | 1.03 |

#### amax requested (B = 1; FROST sm100_mxfp8 declines amax)

| S | mask | new row ms, no amax | with amax | amax cost | new row speedup with amax |
|---:|---|---:|---:|---:|---:|
| 1024 | dense | 0.1140 | 0.1158 | +1.6 % | 1.71 |
| 1024 | causal | 0.0979 | 0.0985 | +0.7 % | 1.75 |
| 4096 | dense | 1.003 | 1.004 | +0.1 % | 1.49 |
| 4096 | causal | 0.6161 | 0.6190 | +0.5 % | 1.55 |
| 16384 | dense | 15.168 | 15.182 | +0.1 % | 1.26 |
| 16384 | causal | 8.366 | 8.415 | +0.6 % | 1.25 |

#### B = 8 (Hq = 88, Hkv = 8): default (throughput) plan vs use_deterministic_algorithm (batch-invariant plan)

| S | mask | request | backend ms | new row ms | **new row speedup** | FROST speedup |
|---:|---|---|---:|---:|---:|---:|
| 1024 | dense | default | 1.149 | 0.5997 | **1.92** | 1.33 |
| 1024 | dense | deterministic | 1.149 | 0.6410 | **1.79** | 1.33 |
| 1024 | causal | default | 0.9587 | 0.4974 | **1.93** | 1.65 |
| 1024 | causal | deterministic | 0.9604 | 0.5142 | **1.87** | 1.65 |
| 4096 | dense | default | 11.015 | 7.964 | **1.38** | 1.07 |
| 4096 | dense | deterministic | 11.040 | 7.975 | **1.38** | 1.07 |
| 4096 | causal | default | 6.991 | 4.959 | **1.41** | 1.22 |
| 4096 | causal | deterministic | 6.989 | 4.963 | **1.41** | 1.22 |

The error against an FP32 reference of the unquantized operands is the backend's to within 1 % (dense and causal;
`test_sdpa_bwd_mxfp8_sm103_d128.py::test_error_matches_backend`); input quantization dominates both.

## Known limits

* SM103 only (built for `sm_103a`; not measured on other parts).
* Needs `nvidia-cutlass-dsl` 4.8.1 or newer: below it the row declines the graph at plan time. Validated with the CUDA 13
  (cu13) DSL wheels only (CuTe DSL 4.8.1 compiling with CUDA 13.4, Torch 2.11 + cu130, cuDNN 9.28).
* BHSD-compact storage only; BSHD graphs are declined, not staged.
* Each new problem shape costs one CuTeDSL compile at plan build (about 2 to 6 s; then cached on disk).
* MQA with few key blocks is slower than the backend engine. The dK/dV grid is one CTA per 128-key block per KV head
  (two with the walk split), so H_kv = 1 at S = 2K to 4K launches only 32 to 64 dK/dV CTAs on 148 SMs. Measured at
  B = 1 (speedup over the backend):
  * 88 / 1 at S = 2048: 0.81 dense, 0.57 causal;
  * 88 / 1 at S = 4096 causal: 0.72;
  * 8 / 1 at S = 2048: 0.91, dense and causal;
  * 8 / 1 at S = 4096 causal: 0.90.

  Dense is ahead from S = 4096 (88 / 1: 1.16; 8 / 1: 1.24), and every one of these head shapes is ahead at S = 8192
  (88 / 1: 1.33 dense, 1.05 causal; 8 / 1: 1.34 dense, 1.23 causal). The row still offers itself for these graphs;
  select another plan by name if they dominate a workload.
