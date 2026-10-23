# OSS block-sparse attention contract

Use `cudnn.BSA` when the attention pattern is already expressed as fixed-size query-to-key block metadata and the workload satisfies one of the architecture-specific CUTLASS DSL paths.

## Before you begin

Install the `cutedsl` extra, use an architecture listed below, and prepare a
dense attention reference plus valid INT32 block metadata. Validate every
active block index and per-row count before invoking the wrapper.

## Install and import

Install the `cutedsl` extra from the same frontend checkout. `cudnn.BSA` is a lazy namespace; accessing it imports optional CUDA Python, PyTorch, and CUTLASS DSL dependencies. Import failure does not mean the cuDNN Graph API is unavailable.

## block_sparse_attention_forward

`cudnn.block_sparse_attention_forward(q_tensor, k_tensor, v_tensor, q2k_block_index, block_sparse_num=None, block_sizes=None, q2k_block_nums=None, *, sparse_block_size=None, allow_empty_block_nums=False, softmax_scale=None, pack_gqa=None, layout="bhsd", kv_splits=1, use_clc=None)` returns a tuple-like mapping with `o_tensor` and `lse_tensor`.

The inputs use BHSD by default. `layout="bshd"` selects BSHD. Q, K, and V must be CUDA FP16 or BF16 tensors with a contiguous head dimension and compatible batch, head, and sequence dimensions. `softmax_scale=None` uses `1 / sqrt(D_qk)`.

`q2k_block_index` is an INT32 tensor whose active prefix lists physical KV-block indices for each query block. Supply either one fixed `block_sparse_num` or an INT32 `q2k_block_nums` tensor with the per-row active counts. `block_sizes` optionally describes valid tokens in physical KV blocks. Active indices must be in range; duplicate indices are not combined for the caller.

The forward support matrix is:

| GPU | Sparse block | Q/K/V dtype | QK and V dimensions | Heads |
| --- | ---: | --- | --- | --- |
| SM90 | 64 | FP16 or BF16 | 64, 96, or 128 | MHA, GQA, MQA |
| SM100/SM110 | 128 | FP16 or BF16 | matching 64, 96, or 128 | MHA, GQA, MQA |
| SM100/SM110 | 64, explicitly selected | BF16 | QK=V=128 | MHA only |
| SM120 | 64 | FP16 or BF16 | QK=V=128 | MHA, GQA, MQA |

SM90 requires `S_q` to be a multiple of 64. The SM100/SM110 block-128 fixed count must be even and at least two. `pack_gqa` applies only to that block-128 path. Integer `kv_splits` is supported by SM90 and the explicit SM100/SM110 block-64 path; `kv_splits="auto"` and `use_clc` are specific to the latter.

## block_sparse_attention_fp8_forward

The Sage FP8 wrapper is forward-only. It accepts contiguous BF16 Q, K, and V in BHSD layout, quantizes them internally to E4M3, and returns one BF16 `o_tensor`; it does not return LSE and has no corresponding backward API. Inputs must use MHA with `D=128` and matching batch, head, and sequence metadata.

SM90, SM100/SM103, and SM120 support sparse block size 64 or 128 with architecture-specific metadata rules. SM90 requires both sequence lengths to be multiples of the chosen block size and accepts only fixed counts. SM100/SM103 block 64 also requires aligned sequence lengths and fixed counts; its split-KV choice is internal. The native SM100/SM103 block-128 path accepts physical tails, empty variable-count rows, rank-1 through rank-3 `block_sizes`, and explicit `kv_splits` from 1 through 256. SM120 accepts sequence tails, fixed or per-query-block counts, and rank-1 through rank-3 block sizes without split-KV.

Outputs are approximate and need not be bitwise identical across block sizes or architectures. The wrapper requires CuTe DSL 4.6.1 or later; the SM90 block-128 path requires 4.6.2 or later.

## block_sparse_attention_backward

`cudnn.block_sparse_attention_backward(do_tensor, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, q2k_block_index, block_sparse_num=None, block_sizes=None, q2k_block_nums=None, *, softmax_scale=None, dq_tensor=None, dk_tensor=None, dv_tensor=None, bucket_size_blocks=None, sparse_block_size=None, layout="bhsd")` returns `dq_tensor`, `dk_tensor`, and `dv_tensor`.

Backward must receive the forward output, natural-log LSE, sparse metadata, layout, scale, and block-size choice from the matching forward pass. Optional gradient outputs allow caller-owned storage. The current backward paths require BF16, equal QK/V dimensions, and MHA:

| GPU | Sparse block | Head dimension |
| --- | ---: | ---: |
| SM90 | 64 | 128 |
| SM100/SM110 | 64 | 128 |
| SM100/SM110 | 128 | 64 or 128 |

SM120 backward is not implemented. The SM100/SM110 block-128 path does not consume `block_sizes`; pass `None` and use full physical KV blocks.

## Reuse compiled kernels

The wrapper caches compiled specializations. Treat device, architecture, dtype, complete shape and stride metadata, sparse block size, tiling choices, and relevant flags as part of the specialization identity. Reuse input and output buffers only when their metadata still matches the compiled contract.

Current implementation: [public wrapper and validation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/block_sparse_attention/api.py#L148).

## Expected result

Forward returns output and natural-log LSE for the selected sparse pattern;
supported backward paths reproduce the corresponding dense masked gradients.
This page records the source contract but does not publish a documentation-owned
executable sample, so verify both results on the target architecture.
