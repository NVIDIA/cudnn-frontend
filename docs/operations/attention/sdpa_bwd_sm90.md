# SDPA Backward (SM90)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** for Hopper (`SM90`, cc 9.0), implemented with CuTe DSL primitives: TMA loads, a
producer warp, a Stats loader warp, a relay warp and two WGMMA consumer warpgroups per CTA. It
consumes the forward activations (`Q/K/V/O`), the loss gradient `dO` and the forward `Stats` (LSE) and
produces `dQ/dK/dV`.

The engine (`sdpa_bwd_sm90`, see `cudnn.sdpa.bwd.engines`) serves single-node `sdpa_backward()` graphs
built with `cudnn.pygraph`. It is a default candidate ranked behind the backend, so the default dispatch
uses it when the backend declines a graph; select it explicitly from the ranked plan list
(`graph.plans` / `graph.select_plan(i)`), or set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` to rank it
first. The standalone adapter is `cudnn.sdpa.bwd.api_dsl_sm90.SdpaBwdDslSm90`.

The kernel lives in `python/cudnn/sdpa/bwd/kernels/sm90/bprop_f16.py`; the pointer entry that chains
the launches is `prepared_host.py` beside it.

## The chain

Three launches serve every graph:

1. the row delta `sum_d O * dO`, which also zeroes the fp32 dQ accumulator, the turn counters and the
   key-tile tickets (shared with the SM100 chain);
2. the **key-owner kernel**: a CTA claims a 128-row key tile from its head's ticket and streams 64-row
   Q / dO tiles through a three-stage TMA ring while the loader warp stages the tiles' `Stats` and delta
   rows. Its two consumer warpgroups recompute `S^T` and `dP^T`, keep `P^T` and `dS^T` in registers as the
   A operands of `dV += P^T dO` and `dK += dS^T Q`, stage `dS^T` in shared memory and contract it against
   their own key rows of the resident key tile into the step's fp32 `dQ` partial; the second warpgroup
   folds its partial onto the first's staged tile. A relay warp adds the sum into the fp32 dQ accumulator
   with a TMA reduction once the query tile's turn counter reaches the key tile index, so every query tile
   accumulates its key tiles in index order. Tickets hand out key tiles in CTA start order, so a CTA only
   ever waits for CTAs that are already running; causal CTAs walk the query tiles from the sequence end
   down to their diagonal, unmasked CTAs walk them upward, and the query heads of a group interleave
   inside the walk. dK and dV accumulate in registers and are stored once;
3. the **dQ cast** from the fp32 accumulator to the output type.

Every reduction has a fixed summation order, so repeated executions return identical bits;
`use_deterministic_algorithm=True` and the default request run the same chain. The workspace holds the
fp32 dQ accumulator and the fp32 delta rows over the 128-row-rounded `S_q`, then one int32 turn counter
per (batch, query head, 64-row query tile) and one ticket per (batch, query head)
(`d128_contract.relay_workspace_layout`); always query the plan's size.

## Supported declarations

* cc 9.0 only, `nvidia-cutlass-dsl >= 4.8`;
* dense compact-BSHD `Q/K/V/O/dO/dQ/dK/dV`, contiguous FP32 `Stats` of shape `(B, H_q, S_q, 1)`;
* `D_QK = D_V` in {64, 128}; BF16 or FP16 throughout;
* MHA, or GQA / MQA with an integral head ratio (the kernel walks the group's query heads);
* unmasked or top-left causal attention, any sequence lengths (tail tiles zero-fill, tail stores clip);
* a positive finite `attn_scale`.

Declined by name: packed THD / ragged offsets, padding masks, bottom-right causal, sliding windows,
bias / dBias, sinks / dSink, dropout, other head dimensions and dtypes.

## Execution contract

Inputs, outputs and workspace must stay on the declared CUDA device; gradient outputs must not overlap
each other or any input, and the workspace must not overlap any operand. Execution binds the current
buffers and stream, initializes its scratch each time, performs no allocation, layout copy or
device-to-host read, and supports CUDA-graph capture and replay.
