# SDPA Backward (SM90)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** for Hopper (`SM90`, cc 9.0), implemented with CuTe DSL primitives: TMA loads, a
TMA producer warpgroup and two WGMMA consumer warpgroups per CTA. It consumes the forward activations
(`Q/K/V/O`), the loss gradient `dO` and the forward `Stats` (LSE) and produces `dQ/dK/dV`.

The engine (`sdpa_bwd_sm90`, see `cudnn.sdpa.bwd.engines`) serves single-node `sdpa_backward()` graphs
built with `cudnn.pygraph`. It is a default candidate ranked behind the backend, so the default dispatch
uses it when the backend declines a graph; select it explicitly from the ranked plan list
(`graph.plans` / `graph.select_plan(i)`), or set `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` to rank it
first. The standalone adapter is `cudnn.sdpa.bwd.api_dsl_sm90.SdpaBwdDslSm90`.

The kernels live in `python/cudnn/sdpa/bwd/kernels/sm90/bprop_f16.py`; the pointer entry that chains
them is `prepared_host.py` beside it.

## The kernel pair

Three launches serve every graph:

1. the row delta `sum_d O * dO` (shared with the SM100 chain);
2. the **key-owner dK / dV kernel**: a CTA owns a 128-row key tile, streams 64-row Q / dO tiles
   through a three-stage TMA ring, recomputes `S^T` and `dP^T`, forms `P^T` and `dS^T` in registers and
   feeds them as the register A operands of `dV += P^T dO` and `dK += dS^T Q`; dK and dV accumulate in
   registers and are stored once;
3. the **query-owner dQ kernel**: a CTA owns a 128-row query tile, streams 64-row K / V tiles,
   recomputes `S` and `dP`, forms `dS` and accumulates `dQ += dS K` in registers, stored once.

Every gradient tile has one owner and a fixed summation order, so repeated executions return identical
bits; `use_deterministic_algorithm=True` and the default request run the same chain. There is no
FP32 dQ workspace: the scratch holds the FP32 delta rows, `4 * B * H_q * round_up(S_q, 128) + 240`
bytes (`d128_contract.pair_workspace_layout`); always query the plan's size.

## Supported declarations

* cc 9.0 only, `nvidia-cutlass-dsl >= 4.8`;
* dense compact-BSHD `Q/K/V/O/dO/dQ/dK/dV`, contiguous FP32 `Stats` of shape `(B, H_q, S_q, 1)`;
* `D_QK = D_V` in {64, 128}; BF16 or FP16 throughout;
* MHA, or GQA / MQA with an integral head ratio (the dK / dV kernel walks the group's query heads);
* unmasked or top-left causal attention, any sequence lengths (tail tiles zero-fill, tail stores clip);
* a positive finite `attn_scale`.

Declined by name: packed THD / ragged offsets, padding masks, bottom-right causal, sliding windows,
bias / dBias, sinks / dSink, dropout, other head dimensions and dtypes.

## Execution contract

Inputs, outputs and workspace must stay on the declared CUDA device; gradient outputs must not overlap
each other or any input, and the workspace must not overlap any operand. Execution binds the current
buffers and stream, initializes its scratch each time, performs no allocation, layout copy or
device-to-host read, and supports CUDA-graph capture and replay.
