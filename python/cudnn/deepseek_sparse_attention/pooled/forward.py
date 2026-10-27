# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared paged pooled DSA scoring, selection, and token expansion."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import torch

from cudnn._torch_stream import stream_context
from cudnn.api_base import APIBase, TupleDict, check_buffers_disjoint
from .indices import _require_dsl

_NAMES = ("q", "weights", "cache", "block_table", "cu_query", "positions", "output", "q_scale")
_BYTES = {"bf16": 256, "fp8": 132, "fp4": 68}


def _require_triton():
    import triton
    from packaging.version import Version

    if Version(triton.__version__) < Version("3.7.0"):
        raise RuntimeError(f"Pooled DSA scoring requires Triton >=3.7.0; installed {triton.__version__}")


@lru_cache(maxsize=128)
def _compile_scores(device, capability, dtype, position_dtype, scale_dtype, pool_size, page_size, mode, weight_dtype, q_dtype, fp8_scale_format):
    _require_triton()
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from ._score_sm100 import cudnn_pooled_scores

    constants = dict(
        POOL_SIZE=pool_size,
        PAGE_SIZE=page_size,
        DTYPE=dtype,
        QUERY_TILE=(2 if weight_dtype == torch.bfloat16 else 4) if mode == "prefill" else 1,
        POOL_TILE=256 if mode == "prefill" and dtype != "fp4" and weight_dtype == torch.float32 else 128,
        BF16_REDUCTION=weight_dtype == torch.bfloat16,
        FP8_SCALE_FORMAT=fp8_scale_format,
    )
    signature = {name: "i64" for name in cudnn_pooled_scores.arg_names}
    signature.update(
        Q="*i8" if q_dtype == torch.int8 else {"bf16": "*bf16", "fp8": "*fp8e4nv", "fp4": "*u8"}[dtype],
        QSCALE="*i32" if scale_dtype == torch.int32 else "*u8",
        WEIGHTS="*bf16" if weight_dtype == torch.bfloat16 else "*fp32",
        CACHE="*u8",
        BLOCK_TABLE="*i32",
        CU_QUERY="*i32",
        POSITIONS="*i64" if position_dtype == torch.int64 else "*i32",
        SCORES="*bf16" if weight_dtype == torch.bfloat16 else "*fp32",
    )
    if scale_dtype is None:
        constants["QSCALE"] = None
    signature.update({name: "constexpr" for name in constants})
    artifacts = {}
    item_size = 2 if dtype == "bf16" else 1
    with torch.cuda.device(device):
        for alignment in (1, 4, 16):
            divisors = dict(
                Q=max(alignment, item_size),
                Q_ROW=max(alignment // item_size, 1),
                Q_HEAD=max(alignment // item_size, 1),
                CACHE=max(alignment, 4),
                CACHE_PAGE=max(alignment, 4),
                SCORES=256,
                SCORE_ROW=page_size,
            )
            attrs = {(cudnn_pooled_scores.arg_names.index(name),): [["tt.divisibility", value]] for name, value in divisors.items()}
            compiled = triton.compile(
                ASTSource(cudnn_pooled_scores, signature, constexprs=constants, attrs=attrs),
                target=GPUTarget("cuda", capability[0] * 10 + capability[1], 32),
                options=dict(num_warps=4, enable_fp_fusion=False),
            )
            compiled[(1, 1, 1)]
            artifacts[alignment] = compiled
    return artifacts, constants["QUERY_TILE"], constants["POOL_TILE"]


@lru_cache(maxsize=64)
def _compile_topk(device, capability, max_pools, selected_pools, mode):
    _require_dsl()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
    from cudnn.deepseek_sparse_attention.indexer_top_k.indexer_top_k_decode_varlen import IndexerTopKKernelVarlenDecode
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options

    def tensor(dtype, shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=32)

    rows, columns = cute.sym_int(64), cute.sym_int(64)
    scores = tensor(cutlass.Float32, (rows, columns))
    scratch = tensor(cutlass.Int32, (rows, 2, columns))
    lengths = tensor(cutlass.Int32, (rows,))
    ids = tensor(cutlass.Int32, (rows, selected_pools))
    kernel = IndexerTopKKernelVarlenDecode(
        cutlass.Float32,
        1 << (max_pools - 1).bit_length(),
        selected_pools,
        return_val=False,
        large_occupancy=mode == "prefill",
    )
    with torch.cuda.device(device):
        return cute.compile(
            kernel,
            scores,
            None,
            scratch,
            None,
            lengths,
            ids,
            None,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=compile_options(),
        )


@lru_cache(maxsize=64)
def _compile_pooled_select(device, capability, max_pools, selected_pools, pool_size, position_dtype, mode, score_dtype=torch.float32):
    _require_dsl()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from ._select_sm100 import PooledSelectKernel

    rows, columns = cute.sym_int(64), cute.sym_int(64)
    if score_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("pooled selection requires FP32 or BF16 scores")
    score_type = cutlass.Float32 if score_dtype == torch.float32 else cutlass.BFloat16
    scores = make_fake_tensor(score_type, (rows, columns), (cute.sym_int(64), 1), assumed_align=32)
    scratch = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(64), 2, cute.sym_int(64)), stride_order=(2, 1, 0), assumed_align=32)
    pos_type = cutlass.Int32 if position_dtype == torch.int32 else cutlass.Int64
    positions = make_fake_tensor(pos_type, (rows,), (cute.sym_int(64),), assumed_align=4)
    output = make_fake_tensor(cutlass.Int32, (cute.sym_int(64), cute.sym_int(64)), (cute.sym_int(64), 1), assumed_align=4)
    kernel = PooledSelectKernel(1 << (max_pools - 1).bit_length(), selected_pools, pool_size, mode, score_type)
    with torch.cuda.device(device):
        return cute.compile(kernel, scores, scratch, positions, output, make_fake_stream(use_tvm_ffi_env_stream=False), options=compile_options())


class IndexerForward(APIBase):
    """Select pooled index keys and emit individual causal token IDs.

    Inputs use packed queries, per-request page tables, and absolute token
    positions. Query projection, rotation, and quantization happen upstream.
    BF16 weights select DeepGEMM's BF16 head reduction and logits.
    FP8/FP4 accept packed UE8M0 query scales without conversion.
    Execute launches scoring and fused selection/expansion into caller buffers.
    """

    def __init__(
        self,
        q,
        weights,
        cache,
        block_table,
        cu_query,
        positions,
        output,
        q_scale=None,
        *,
        pool_size=4,
        page_size=64,
        selected_pools=512,
        cache_dtype="fp8",
        mode="decode",
        max_pools=None,
        keys=None,
        gates=None,
        ape=None,
        tail=None,
        pool_slots=None,
        tail_slots=None,
        fp8_scale_format="fp32",
    ):
        super().__init__()
        self.pool_size, self.page_size, self.selected_pools = int(pool_size), int(page_size), int(selected_pools)
        self.cache_dtype, self.mode = cache_dtype, mode
        self.fp8_scale_format = fp8_scale_format
        self._weight_dtype, self._q_dtype = weights.dtype, q.dtype
        self.max_pools = int(max_pools) if max_pools is not None else block_table.shape[1] * self.page_size
        tensors = (q, weights, cache, block_table, cu_query, positions, output, q_scale)
        self._descs = tuple(self._make_tensor_desc(t) for t in tensors)
        self._device = torch.device(q.device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._position_dtype = positions.dtype
        self._scale_dtype = q_scale.dtype if q_scale is not None else None
        self._update = None
        updates = (keys, gates, ape, tail, pool_slots, tail_slots)
        if any(t is not None for t in updates):
            if any(t is None for t in updates):
                raise ValueError("cache updates require keys, gates, ape, tail, pool_slots, and tail_slots together")
            if keys.shape[0] != q.shape[0]:
                raise ValueError("cache-update keys must match the packed query count")
            from .cache import PoolCacheUpdate

            self._update = PoolCacheUpdate(
                keys,
                gates,
                ape,
                cache,
                tail,
                pool_slots,
                tail_slots,
                positions if mode == "decode" else None,
                cu_query if mode == "decode" else None,
                pool_size=pool_size,
                page_size=page_size,
                cache_dtype=cache_dtype,
                mode=mode,
                fp8_scale_format=fp8_scale_format,
            )

    def _validate(self, tensors, *, descriptors=False):
        q, weights, cache, table, cu, positions, output, scales = tensors
        if not descriptors and any(t is not None and not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("execute() requires torch tensors")
        if any(t is None or t.ndim != rank for t, rank in zip(tensors[:7], (3, 2, 2, 2, 1, 1, 2))):
            raise ValueError("expected Q [N,32,D], matrix weights/cache/table/output, and vector offsets/positions")
        rows, heads, dim = q.shape
        qdim = 64 if self.cache_dtype == "fp4" else 128
        qdtypes = {"bf16": (torch.bfloat16,), "fp8": (torch.float8_e4m3fn,), "fp4": (torch.uint8, torch.int8)}[self.cache_dtype]
        if (heads, dim) != (32, qdim) or q.dtype not in qdtypes or q.dtype != self._q_dtype or rows >= 2**31:
            raise ValueError(f"q must have shape [N,32,{qdim}] and planned dtype in {qdtypes}, with N < 2**31")
        if self._weight_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("weights must be BF16 or FP32")
        expected = (
            (weights, (rows, 32), self._weight_dtype),
            (cache, (cache.shape[0], self.page_size * _BYTES[self.cache_dtype]), torch.uint8),
            (cu, (table.shape[0] + 1,), torch.int32),
            (positions, (rows,), self._position_dtype),
        )
        for tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"expected shape {shape} and dtype {dtype}")
        if table.dtype != torch.int32 or not 0 < table.shape[0] <= 65535 or not 0 < table.shape[1] * self.page_size <= self.max_pools:
            raise ValueError("block_table must be INT32 with 1..65535 requests and a page capacity within max_pools")
        if self._position_dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be INT32 or INT64")
        if output.dtype != torch.int32 or output.shape[0] < rows or output.shape[1] < self.selected_pools * self.pool_size + self.pool_size - 1:
            raise ValueError("output must be INT32 and cover expanded pools, tail slots, and all query rows")
        if output.shape[0] >= 2**31 or output.shape[1] > 65535 * 256:
            raise ValueError("output exceeds the CUDA launch grid limits")
        if self.cache_dtype == "fp4" or self._scale_dtype is not None:
            shape = (rows, 32) if self._scale_dtype == torch.int32 else (rows, 32, 4)
            if scales is None or scales.dtype != self._scale_dtype or self._scale_dtype not in (torch.int32, torch.uint8) or scales.shape != shape:
                raise ValueError("expected planned INT32 [N,32] or UINT8 [N,32,4] UE8M0 query scales")
            if self.cache_dtype != "fp4" and not (self.cache_dtype == "fp8" and self.fp8_scale_format == "ue8m0"):
                raise ValueError("q_scale requires FP4 or FP8 with fp8_scale_format='ue8m0'")
        elif scales is not None:
            raise ValueError("q_scale presence must match the plan")
        for name, tensor in zip(_NAMES, tensors):
            if tensor is None:
                continue
            if tensor.device.type != "cuda" or tensor.device != self._device:
                raise ValueError(f"{name} must be on the planned CUDA device")
            strides = tensor.stride if descriptors else tensor.stride()
            if any(s <= 0 for s in strides):
                raise ValueError(f"{name} requires positive strides")
            if tensor.ndim >= 2 and strides[-1] != 1:
                raise ValueError(f"{name} requires a contiguous last axis")
            if any(strides[i] < strides[i + 1] * tensor.shape[i + 1] for i in range(tensor.ndim - 1)):
                raise ValueError(f"{name} requires non-overlapping row-major axes")
            if name == "cache" and strides[0] % 4:
                raise ValueError("cache page stride must be a multiple of four bytes")
            if not descriptors and (tensor.data_ptr() % (4 if name == "cache" else tensor.element_size()) or tensor.requires_grad):
                raise ValueError(f"{name} must have aligned storage and no autograd requirement")
        return rows, table.shape[1] * self.page_size

    def check_support(self):
        _require_dsl()
        _require_triton()
        if self.pool_size not in (4, 16) or self.page_size not in (64, 128, 256):
            raise ValueError("supported pool sizes are 4/16 and page sizes are 64/128/256")
        if self.mode not in ("prefill", "decode") or self.cache_dtype not in _BYTES:
            raise ValueError("expected prefill/decode mode and bf16/fp8/fp4 cache_dtype")
        if self.fp8_scale_format not in ("fp32", "ue8m0") or (self.cache_dtype != "fp8" and self.fp8_scale_format != "fp32"):
            raise ValueError("fp8_scale_format must be fp32, or ue8m0 with an FP8 cache")
        if not 1 <= self.selected_pools <= 2048 or self.selected_pools % 32 or not 1 <= self.max_pools <= 32768:
            raise ValueError("selected_pools must be a multiple of 32 up to 2048, and max_pools in 1..32768")
        rows, columns = self._validate(self._descs, descriptors=True)
        self._workspace_layout(rows, columns)
        if self._update is not None:
            self._update.check_support()
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability not in ((10, 0), (10, 3)):
            raise NotImplementedError("Pooled DSA requires Blackwell SM100 or SM103")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self._update is not None:
            self._update.compile()
        self._score = _compile_scores(
            self._device.index,
            self._capability,
            self.cache_dtype,
            self._position_dtype,
            self._scale_dtype,
            self.pool_size,
            self.page_size,
            self.mode,
            self._weight_dtype,
            self._q_dtype,
            self.fp8_scale_format,
        )
        self._select = _compile_pooled_select(
            self._device.index, self._capability, self.max_pools, self.selected_pools, self.pool_size, self._position_dtype, self.mode, self._weight_dtype
        )
        self._compiled_kernel = True

    def _workspace_layout(self, rows, columns):
        if rows < 0 or columns < 1 or columns > self.max_pools or rows * columns * 2 >= 2**31:
            raise ValueError("workspace dimensions must fit max_pools and the top-k int32 addressing limit")
        offset, layout = 0, []
        bucket = 1 << (self.max_pools - 1).bit_length()
        scratch_shape = (1, 2, 4) if self._weight_dtype == torch.bfloat16 and 8192 <= bucket <= 32768 else (rows, 2, columns)
        for dtype, shape, item_bytes in (
            (self._weight_dtype, (rows, columns), 2 if self._weight_dtype == torch.bfloat16 else 4),
            (torch.int32, scratch_shape, 4),
        ):
            count = 1
            for size in shape:
                count *= size
            layout.append((offset, count * item_bytes, dtype, shape))
            offset = (offset + count * item_bytes + 255) // 256 * 256
        return offset, layout

    def scratch_workspace_bytes(self, num_queries=None, num_pools=None):
        rows = self._descs[0].shape[0] if num_queries is None else num_queries
        columns = self._descs[3].shape[1] * self.page_size if num_pools is None else num_pools
        return self._workspace_layout(rows, columns)[0]

    def execute(
        self,
        q,
        weights,
        cache,
        block_table,
        cu_query,
        positions,
        output,
        workspace,
        q_scale=None,
        *,
        max_query_tokens,
        current_stream=None,
        keys=None,
        gates=None,
        ape=None,
        tail=None,
        pool_slots=None,
        tail_slots=None,
    ):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        tensors = (q, weights, cache, block_table, cu_query, positions, output, q_scale)
        rows, columns = self._validate(tensors)
        updates = (keys, gates, ape, tail, pool_slots, tail_slots)
        if (self._update is None and any(t is not None for t in updates)) or (self._update is not None and any(t is None for t in updates)):
            raise ValueError("cache-update tensor presence must match the plan")
        update_args = None
        if self._update is not None:
            if keys.shape[0] != rows:
                raise ValueError("cache-update keys must match the packed query count")
            update_args = (
                keys,
                gates,
                ape,
                cache,
                tail,
                pool_slots,
                tail_slots,
                positions if self.mode == "decode" else None,
                cu_query if self.mode == "decode" else None,
            )
        needed, layout = self._workspace_layout(rows, columns)
        if not isinstance(workspace, torch.Tensor) or workspace.device != self._device or workspace.dtype != torch.uint8 or workspace.ndim != 1:
            raise ValueError("workspace must be a UINT8 vector on the planned CUDA device")
        if not workspace.is_contiguous() or workspace.numel() < needed or workspace.data_ptr() % 256:
            raise ValueError("workspace must be contiguous, large enough, and 256-byte aligned")
        artifacts, query_tile, pool_tile = self._score
        alignment = 1
        for candidate in (16, 4):
            if all(
                value % candidate == 0
                for value in (q.data_ptr(), q.stride(0) * q.element_size(), q.stride(1) * q.element_size(), cache.data_ptr(), cache.stride(0))
            ):
                alignment = candidate
                break
        compiled = artifacts[alignment]
        if type(max_query_tokens) is not int or max_query_tokens < 1 or (max_query_tokens + query_tile - 1) // query_tile > 65535:
            raise ValueError("max_query_tokens must be positive and fit the CUDA y-grid")
        if max_query_tokens * block_table.shape[0] < rows:
            raise ValueError("max_query_tokens cannot cover the packed query count")
        bindings = dict(zip(_NAMES, tensors))
        bindings.update(workspace=workspace, keys=keys, gates=gates, ape=ape, tail=tail, pool_slots=pool_slots, tail_slots=tail_slots)
        writes = ("workspace", "output", "cache", "tail") if self._update is not None else ("workspace", "output")
        check_buffers_disjoint(
            [(name, bindings[name]) for name in writes],
            [(name, tensor) for name, tensor in bindings.items() if name not in writes],
            what="pooled DSA kernels",
        )
        scores, extra = [workspace[offset : offset + size].view(dtype).view(shape) for offset, size, dtype, shape in layout]
        with torch.cuda.device(self._device):
            stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(self._device).cuda_stream
            if update_args is not None:
                self._update.execute(*update_args, current_stream=cuda.CUstream(stream))
            if rows:
                qs_row, qs_head = (q_scale.stride(0) * q_scale.element_size(), q_scale.stride(1) * q_scale.element_size()) if q_scale is not None else (0, 0)
                compiled[((columns + pool_tile - 1) // pool_tile, (max_query_tokens + query_tile - 1) // query_tile, block_table.shape[0])](
                    q,
                    q_scale,
                    weights,
                    cache,
                    block_table,
                    cu_query,
                    positions,
                    scores,
                    rows,
                    cache.shape[0],
                    columns,
                    block_table.shape[1],
                    q.stride(0),
                    q.stride(1),
                    qs_row,
                    qs_head,
                    weights.stride(0),
                    cache.stride(0),
                    block_table.stride(0),
                    cu_query.stride(0),
                    positions.stride(0),
                    columns,
                    self.pool_size,
                    self.page_size,
                    self.cache_dtype,
                    query_tile,
                    pool_tile,
                    self._weight_dtype == torch.bfloat16,
                    self.fp8_scale_format,
                    stream=stream,
                )
            if output.shape[0]:
                self._select(scores, extra, positions, output, cuda.CUstream(stream))
        return TupleDict(indices=output)


def indexer_forward_wrapper(
    q,
    weights,
    cache,
    block_table,
    cu_query,
    positions,
    q_scale=None,
    *,
    output=None,
    workspace=None,
    pool_size=4,
    page_size=64,
    selected_pools=512,
    cache_dtype="fp8",
    mode="decode",
    max_pools=None,
    max_query_tokens,
    current_stream=None,
    keys=None,
    gates=None,
    ape=None,
    tail=None,
    pool_slots=None,
    tail_slots=None,
    fp8_scale_format="fp32",
):
    """Allocate optional output/workspace and run the prepared indexer."""
    with stream_context(current_stream, q.device):
        if output is None:
            output = torch.empty(q.shape[0], selected_pools * pool_size + pool_size - 1, device=q.device, dtype=torch.int32)
        plan = IndexerForward(
            q,
            weights,
            cache,
            block_table,
            cu_query,
            positions,
            output,
            q_scale,
            pool_size=pool_size,
            page_size=page_size,
            selected_pools=selected_pools,
            cache_dtype=cache_dtype,
            mode=mode,
            max_pools=max_pools,
            keys=keys,
            gates=gates,
            ape=ape,
            tail=tail,
            pool_slots=pool_slots,
            tail_slots=tail_slots,
            fp8_scale_format=fp8_scale_format,
        )
        plan.compile()
        if workspace is None:
            workspace = torch.empty(plan.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
        return plan.execute(
            q,
            weights,
            cache,
            block_table,
            cu_query,
            positions,
            output,
            workspace,
            q_scale,
            max_query_tokens=max_query_tokens,
            current_stream=current_stream,
            keys=keys,
            gates=gates,
            ape=ape,
            tail=tail,
            pool_slots=pool_slots,
            tail_slots=tail_slots,
        )
