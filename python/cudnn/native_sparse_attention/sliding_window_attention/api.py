# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import math
import cudnn
import torch

from cuda.bindings import driver as cuda
from cudnn.datatypes import _torch_to_cudnn_data_type
from cudnn.api_base import APIBase, TupleDict, WorkspaceCarver
from cudnn._torch_stream import contiguous_on_stream, stream_context
from typing import Optional

from ..utils import make_tensor_strided_like


def packed_thd_ragged_offsets(
    seq_len_q: torch.Tensor,
    seq_len_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    stats: Optional[torch.Tensor] = None,
):
    """Ragged offsets for a FULLY PACKED ``T,H,D`` batch: ``(q, k, v, o, stats)`` int64 ``(b+1, 1, 1, 1)`` tensors.

    ``offset[i] = sum(seq_len[:i]) * tensor.stride(0)`` (element offsets, exclusive prefix sum). ``stats`` is
    ``None`` -> the stats offset is ``None``. Runs a few small torch kernels on the CALLER's current stream;
    build the offsets once per batch and pass them to ``SlidingWindowAttention.execute`` -- the engine never
    derives them itself (per-execute torch.cat/cumsum on the execute path, Rule 1/8).
    """

    def exclusive_prefix_sum(seq_len: torch.Tensor) -> torch.Tensor:
        seq_len = seq_len.reshape(-1)
        zero = torch.zeros((1,), dtype=torch.int64, device=seq_len.device)
        return torch.cat((zero, torch.cumsum(seq_len, dim=0, dtype=torch.int64))).view(-1, 1, 1, 1)

    cum_q = exclusive_prefix_sum(seq_len_q)
    cum_kv = exclusive_prefix_sum(seq_len_kv)
    return (
        cum_q * q.stride(0),
        cum_kv * k.stride(0),
        cum_kv * v.stride(0),
        cum_q * o.stride(0),
        cum_q * stats.stride(0) if stats is not None else None,
    )


class SlidingWindowAttention(APIBase):
    """Sliding-window SDPA on the cuDNN backend graph API.

    Contract (Rule 8: the graph owns no device memory and never blocks the host):
        * ``compile()`` builds the plans; ``get_workspace_size()`` then reports the backend workspace in bytes.
          ``execute()`` takes that buffer as ``workspace=`` (uint8, on the graph's device) and never allocates
          one; when the size is 0 ``workspace`` may be ``None``.
        * ``T,H,D`` layout: ``q/k/v/o`` (and ``stats`` in training mode) ragged-offset tensors -- int64,
          ``(b+1, 1, 1, 1)`` -- are REQUIRED at ``execute()``; the engine does not derive them. For the fully
          packed layout build them once per batch with ``cudnn.NSA.packed_thd_ragged_offsets``. The
          ``sample_*_ragged_offset`` arguments are optional and only validated; the graph descriptors are built
          from the batch shape.
        * ``current_stream`` re-streams the cuDNN handle (``cudnn.set_stream``) before the launch.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_stats: Optional[torch.Tensor] = None,
        left_bound: int = 0,
        right_bound: int = 0,
        sample_seq_len_q: Optional[torch.Tensor] = None,
        sample_seq_len_kv: Optional[torch.Tensor] = None,
        sample_q_ragged_offset: Optional[torch.Tensor] = None,
        sample_k_ragged_offset: Optional[torch.Tensor] = None,
        sample_v_ragged_offset: Optional[torch.Tensor] = None,
        sample_o_ragged_offset: Optional[torch.Tensor] = None,
        sample_stats_ragged_offset: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_kv: Optional[int] = None,
        attn_scale: Optional[float] = None,
        intermediate_data_type: torch.dtype = torch.float32,
        compute_data_type: torch.dtype = torch.float32,
        cudnn_handle: Optional[cudnn.handle] = None,
    ):
        super().__init__()
        self._logger.debug("Entering __init__")

        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.is_infer = sample_stats is None
        self.sample_stats = self._pad_tensor_to_ndim(sample_stats, self.sample_o.ndim, "sample_stats") if sample_stats is not None else None
        self.left_bound = left_bound
        self.right_bound = right_bound

        self.sample_seq_len_q = self._pad_tensor_to_ndim(sample_seq_len_q, 4, "sample_seq_len_q")
        self.sample_seq_len_kv = self._pad_tensor_to_ndim(sample_seq_len_kv, 4, "sample_seq_len_kv")
        self.max_seq_len_q = max_seq_len_q
        self.max_seq_len_kv = max_seq_len_kv
        self.sample_q_ragged_offset = self._pad_tensor_to_ndim(sample_q_ragged_offset, 4, "sample_q_ragged_offset")
        self.sample_k_ragged_offset = self._pad_tensor_to_ndim(sample_k_ragged_offset, 4, "sample_k_ragged_offset")
        self.sample_v_ragged_offset = self._pad_tensor_to_ndim(sample_v_ragged_offset, 4, "sample_v_ragged_offset")
        self.sample_o_ragged_offset = self._pad_tensor_to_ndim(sample_o_ragged_offset, 4, "sample_o_ragged_offset")
        self.sample_stats_ragged_offset = (
            self._pad_tensor_to_ndim(sample_stats_ragged_offset, 4, "sample_stats_ragged_offset") if sample_stats_ragged_offset is not None else None
        )

        self.attn_scale = attn_scale if attn_scale is not None else 1.0 / math.sqrt(self.sample_q.shape[-1])
        self.intermediate_data_type = intermediate_data_type
        self.compute_data_type = compute_data_type

        self.dtype = None
        self.sm_version = None
        self.input_layout = None

        if cudnn_handle is None:
            self._logger.critical(
                "cudnn_handle not provided, creating new handle. This is not recommended as this is significant overhead and will occur for each SlidingWindowAttention object created."
            )
        self._cudnn_handle = cudnn_handle if cudnn_handle is not None else cudnn.create_handle()
        self._cudnn_swa_graph = None
        self._cudnn_compiled = False
        self._workspace_bytes = None
        self.batch_size = None
        self._logger.debug(
            f"__init__ completed with args: sample_q {tuple(sample_q.shape)}, sample_k {tuple(sample_k.shape)}, sample_v {tuple(sample_v.shape)}, sample_o {tuple(sample_o.shape)}, sample_stats {None if sample_stats is None else tuple(sample_stats.shape)}, left_bound {left_bound}, right_bound {right_bound}, sample_seq_len_q {None if sample_seq_len_q is None else tuple(sample_seq_len_q.shape)}, sample_seq_len_kv {None if sample_seq_len_kv is None else tuple(sample_seq_len_kv.shape)}, max_seq_len_q {max_seq_len_q}, max_seq_len_kv {max_seq_len_kv}, is_infer {self.is_infer}, attn_scale {attn_scale}, intermediate_data_type {intermediate_data_type}, compute_data_type {compute_data_type}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.dtype = self.sample_q.dtype
        self.sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
        if self.sample_q.ndim == 4:
            self._logger.debug("Inferred bshd layout")
            self.input_layout = "bshd"
        elif self.sample_q.ndim == 3:
            self._logger.debug("Inferred thd layout")
            self.input_layout = "thd"
        else:
            raise ValueError(f"Invalid input layout: {self.sample_q.ndim}")

        swa_graph = cudnn.pygraph(
            io_data_type=_torch_to_cudnn_data_type(self.dtype),
            intermediate_data_type=_torch_to_cudnn_data_type(self.intermediate_data_type),
            compute_data_type=_torch_to_cudnn_data_type(self.compute_data_type),
            handle=self._cudnn_handle,
            sm_version=self.sm_version,
        )

        (
            self.q_cudnn,
            self.k_cudnn,
            self.v_cudnn,
            self.seq_len_q_cudnn,
            self.seq_len_kv_cudnn,
            self.q_ragged_offset_cudnn,
            self.k_ragged_offset_cudnn,
            self.v_ragged_offset_cudnn,
            self.o_ragged_offset_cudnn,
            self.stats_ragged_offset_cudnn,
        ) = (None, None, None, None, None, None, None, None, None, None)
        if self.input_layout == "bshd":
            b, h_q, s_q, d_qk = self.sample_q.shape
            b, h_kv, s_kv, d_qk = self.sample_k.shape
            b, h_kv, s_kv, d_v = self.sample_v.shape
            b, h_q, s_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (b, h_q, s_q, d_qk):
                raise ValueError(f"Input shape mismatch: expected Q tensor shape {b, h_q, s_q, d_qk}, got {self.sample_q.shape}")
            if self.sample_k.shape != (b, h_kv, s_kv, d_qk):
                raise ValueError(f"Input shape mismatch: expected K tensor shape {b, h_kv, s_kv, d_qk}, got {self.sample_k.shape}")
            if self.sample_v.shape != (b, h_kv, s_kv, d_v):
                raise ValueError(f"Input shape mismatch: expected V tensor shape {b, h_kv, s_kv, d_v}, got {self.sample_v.shape}")
            if self.sample_o.shape != (b, h_q, s_q, d_v):
                raise ValueError(f"Output shape mismatch: expected O tensor shape {b, h_q, s_q, d_v}, got {self.sample_o.shape}")
            if not self.is_infer:
                self.sample_stats = self._pad_tensor_to_ndim(self.sample_stats, 4, "sample_stats")
                if self.sample_stats.shape != (b, h_q, s_q, 1):
                    raise ValueError(f"Output shape mismatch: expected Stats tensor shape {b, h_q, s_q, 1}, got {self.sample_stats.shape}")
            if self.sample_seq_len_q is not None or self.sample_seq_len_kv is not None:
                raise ValueError(
                    f"sample_seq_len_q and sample_seq_len_kv should be None for bshd layout, got {self.sample_seq_len_q} and {self.sample_seq_len_kv}"
                )
            if self.max_seq_len_q is not None or self.max_seq_len_kv is not None:
                raise ValueError(f"max_seq_len_q and max_seq_len_kv should be None for bshd layout, got {self.max_seq_len_q} and {self.max_seq_len_kv}")
            if (
                self.sample_q_ragged_offset is not None
                or self.sample_k_ragged_offset is not None
                or self.sample_v_ragged_offset is not None
                or self.sample_o_ragged_offset is not None
                or self.sample_stats_ragged_offset is not None
            ):
                raise ValueError(
                    f"sample_q_ragged_offset, sample_k_ragged_offset, sample_v_ragged_offset, sample_o_ragged_offset, and sample_stats_ragged_offset should be None for bshd layout, got {self.sample_q_ragged_offset}, {self.sample_k_ragged_offset}, {self.sample_v_ragged_offset}, {self.sample_o_ragged_offset}, and {self.sample_stats_ragged_offset}"
                )

            self.q_cudnn = swa_graph.tensor_like(self.sample_q)
            self.k_cudnn = swa_graph.tensor_like(self.sample_k)
            self.v_cudnn = swa_graph.tensor_like(self.sample_v)
        elif self.input_layout == "thd":
            t, h_q, d_qk = self.sample_q.shape
            t, h_kv, d_qk = self.sample_k.shape
            t, h_kv, d_v = self.sample_v.shape
            t, h_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (t, h_q, d_qk):
                raise ValueError(f"Input shape mismatch: expected Q tensor shape {t, h_q, d_qk}, got {self.sample_q.shape}")
            if self.sample_k.shape != (t, h_kv, d_qk):
                raise ValueError(f"Input shape mismatch: expected K tensor shape {t, h_kv, d_qk}, got {self.sample_k.shape}")
            if self.sample_v.shape != (t, h_kv, d_v):
                raise ValueError(f"Input shape mismatch: expected V tensor shape {t, h_kv, d_v}, got {self.sample_v.shape}")
            if self.sample_o.shape != (t, h_q, d_v):
                raise ValueError(f"Output shape mismatch: expected O tensor shape {t, h_q, d_v}, got {self.sample_o.shape}")
            if not self.is_infer:
                self.sample_stats = self._pad_tensor_to_ndim(self.sample_stats, 3, "sample_stats")
                if self.sample_stats.shape != (t, h_q, 1):
                    raise ValueError(f"Output shape mismatch: expected Stats tensor shape {t, h_q, 1}, got {self.sample_stats.shape}")

            if self.sample_seq_len_q is None or self.sample_seq_len_kv is None:
                raise ValueError(
                    f"sample_seq_len_q and sample_seq_len_kv must be provided for thd layout, got {self.sample_seq_len_q} and {self.sample_seq_len_kv}"
                )
            if self.max_seq_len_q is None or self.max_seq_len_kv is None:
                raise ValueError(f"max_seq_len_q and max_seq_len_kv must be provided for thd layout, got {self.max_seq_len_q} and {self.max_seq_len_kv}")

            b = len(self.sample_seq_len_q)
            self.batch_size = b
            # The sample ragged offsets are optional metadata: validated when given (all or none), never read, and
            # not what the descriptors are built from -- the graph only needs the (b+1, 1, 1, 1) int64 shape.
            sample_ragged = self._ragged_offset_set(
                self.sample_q_ragged_offset,
                self.sample_k_ragged_offset,
                self.sample_v_ragged_offset,
                self.sample_o_ragged_offset,
                self.sample_stats_ragged_offset,
                prefix="sample_",
                suffix="",
                required=False,
            )
            for name, tensor in sample_ragged.items():
                self._check_ragged_offset(tensor, name, b)
            self.q_cudnn = swa_graph.tensor(
                dim=(b, h_q, self.max_seq_len_q, d_qk),
                stride=(
                    self.sample_q.stride()[0] * self.max_seq_len_q,
                    self.sample_q.stride()[1],
                    self.sample_q.stride()[0],
                    self.sample_q.stride()[2],
                ),
            )
            self.k_cudnn = swa_graph.tensor(
                dim=(b, h_kv, self.max_seq_len_kv, d_qk),
                stride=(
                    self.sample_k.stride()[0] * self.max_seq_len_kv,
                    self.sample_k.stride()[1],
                    self.sample_k.stride()[0],
                    self.sample_k.stride()[2],
                ),
            )
            self.v_cudnn = swa_graph.tensor(
                dim=(b, h_kv, self.max_seq_len_kv, d_v),
                stride=(
                    self.sample_v.stride()[0] * self.max_seq_len_kv,
                    self.sample_v.stride()[1],
                    self.sample_v.stride()[0],
                    self.sample_v.stride()[2],
                ),
            )
            self.seq_len_q_cudnn = swa_graph.tensor_like(self.sample_seq_len_q)
            self.seq_len_kv_cudnn = swa_graph.tensor_like(self.sample_seq_len_kv)

            def ragged_offset_desc(name: str):
                return swa_graph.tensor(name=name, dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)

            self.q_ragged_offset_cudnn = ragged_offset_desc("q_ragged_offset")
            self.k_ragged_offset_cudnn = ragged_offset_desc("k_ragged_offset")
            self.v_ragged_offset_cudnn = ragged_offset_desc("v_ragged_offset")
            self.o_ragged_offset_cudnn = ragged_offset_desc("o_ragged_offset")
            if not self.is_infer:
                self.stats_ragged_offset_cudnn = ragged_offset_desc("stats_ragged_offset")

            self.q_cudnn.set_ragged_offset(self.q_ragged_offset_cudnn)
            self.k_cudnn.set_ragged_offset(self.k_ragged_offset_cudnn)
            self.v_cudnn.set_ragged_offset(self.v_ragged_offset_cudnn)

        self.o_cudnn, self.stats_cudnn = swa_graph.sdpa(
            name="sdpa",
            q=self.q_cudnn,
            k=self.k_cudnn,
            v=self.v_cudnn,
            generate_stats=not self.is_infer,
            attn_scale=self.attn_scale,
            bias=None,
            use_alibi_mask=False,
            use_padding_mask=(self.input_layout == "thd"),
            seq_len_q=self.seq_len_q_cudnn,
            seq_len_kv=self.seq_len_kv_cudnn,
            diagonal_band_left_bound=self.left_bound,
            diagonal_band_right_bound=self.right_bound,
            diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
            dropout=None,
            rng_dump=None,
            paged_attention_k_table=None,
            paged_attention_v_table=None,
            paged_attention_max_seq_len_kv=None,
        )
        self.o_cudnn.set_output(True)

        if self.input_layout == "bshd":
            self.o_cudnn.set_dim(self.sample_o.shape).set_stride(self.sample_o.stride())
            if not self.is_infer:
                self.stats_cudnn.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.stats_cudnn.set_dim(self.sample_stats.shape).set_stride(self.sample_stats.stride())
        elif self.input_layout == "thd":
            self.o_cudnn.set_dim((b, h_q, self.max_seq_len_q, d_v))
            self.o_cudnn.set_stride(
                (
                    self.sample_o.stride()[0] * self.max_seq_len_q,
                    self.sample_o.stride()[1],
                    self.sample_o.stride()[0],
                    self.sample_o.stride()[2],
                )
            )
            self.o_cudnn.set_ragged_offset(self.o_ragged_offset_cudnn)

            if not self.is_infer:
                self.stats_cudnn.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.stats_cudnn.set_dim((b, h_q, self.max_seq_len_q, 1))
                self.stats_cudnn.set_stride(
                    (
                        self.sample_stats.stride()[0] * self.max_seq_len_q,
                        self.sample_stats.stride()[1],
                        self.sample_stats.stride()[0],
                        self.sample_stats.stride()[2],
                    )
                )
                self.stats_cudnn.set_ragged_offset(self.stats_ragged_offset_cudnn)

        try:
            swa_graph.validate()
        except cudnn.cudnnGraphNotSupportedError as e:
            self._logger.error(f"Graph not supported (cudnnGraphNotSupportedError): {e}")
            return False
        except Exception as e:
            self._logger.error(f"Graph not supported: {e}")
            return False

        self._cudnn_swa_graph = swa_graph
        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        self._ensure_support_checked()

        self._cudnn_swa_graph.build_operation_graph()
        self._cudnn_swa_graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        self._cudnn_swa_graph.check_support()
        self._cudnn_swa_graph.build_plans()
        self._workspace_bytes = int(self._cudnn_swa_graph.get_workspace_size())

        self._cudnn_compiled = True
        self._logger.debug("SlidingWindowAttention kernel compiled successfully")

    def get_workspace_size(self) -> int:
        """Backend workspace bytes ``execute()`` needs (R2); allocate this much uint8 on the graph's device and pass it as ``workspace=``."""
        if not self._cudnn_compiled:
            raise RuntimeError("SlidingWindowAttention.get_workspace_size(): compile() first")
        return self._workspace_bytes

    def _ragged_offset_set(self, q, k, v, o, stats, *, prefix: str, suffix: str, required: bool) -> dict:
        """The ragged-offset tensors as a validated SET (Rule 1): all given, or -- when not ``required`` -- all None."""
        names = {
            f"{prefix}q_ragged_offset{suffix}": q,
            f"{prefix}k_ragged_offset{suffix}": k,
            f"{prefix}v_ragged_offset{suffix}": v,
            f"{prefix}o_ragged_offset{suffix}": o,
        }
        stats_name = f"{prefix}stats_ragged_offset{suffix}"
        if self.is_infer:
            if stats is not None:
                raise ValueError(f"SlidingWindowAttention: {stats_name} was given but the graph was built in inference mode (no stats); pass None")
        else:
            names[stats_name] = stats
        given = {n: t for n, t in names.items() if t is not None}
        if len(given) == len(names):
            return given
        if given or required:
            missing = [n for n, t in names.items() if t is None]
            raise ValueError(
                f"SlidingWindowAttention: {', '.join(names)} are required for the T,H,D layout as a set; missing {missing}. "
                "The engine does not derive them (per-execute torch.cat/cumsum, Rule 1/8) -- build them once per batch "
                "(see cudnn.NSA.packed_thd_ragged_offsets) and pass them in"
            )
        return {}

    def _check_ragged_offset(self, tensor: torch.Tensor, name: str, b: int) -> None:
        if tensor.dtype != torch.int64:
            raise ValueError(f"SlidingWindowAttention: {name} must be int64, got {tensor.dtype}")
        if not tensor.is_cuda or tensor.device != self.sample_q.device:
            raise ValueError(f"SlidingWindowAttention: {name} must be on the plan's device {self.sample_q.device}, got {tensor.device}")
        if tuple(tensor.shape) != (b + 1, 1, 1, 1):
            raise ValueError(f"SlidingWindowAttention: {name} must have shape (b+1, 1, 1, 1) = {(b + 1, 1, 1, 1)}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():  # the graph declares stride (1, 1, 1, 1); a strided view would be read as packed
            raise ValueError(f"SlidingWindowAttention: {name} must be contiguous, got strides {tuple(tensor.stride())}")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        stats_tensor: Optional[torch.Tensor] = None,
        seq_len_q_tensor: Optional[torch.Tensor] = None,
        seq_len_kv_tensor: Optional[torch.Tensor] = None,
        q_ragged_offset_tensor: Optional[torch.Tensor] = None,
        k_ragged_offset_tensor: Optional[torch.Tensor] = None,
        v_ragged_offset_tensor: Optional[torch.Tensor] = None,
        o_ragged_offset_tensor: Optional[torch.Tensor] = None,
        stats_ragged_offset_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        cudnn_handle: Optional[cudnn.handle] = None,
        skip_compile: bool = False,
        workspace: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the compiled graph. ``workspace``: ``get_workspace_size()`` bytes (uint8, graph's device); required
        when that size is non-zero. ``T,H,D``: the ragged-offset tensors are required (see the class docstring)."""
        self._logger.debug("Entering execute")
        cudnn_handle = self._cudnn_handle if cudnn_handle is None else cudnn_handle
        if current_stream is not None:
            self._logger.info("Overwriting cudnn_handle stream with provided cuda stream. Do not pass in current_stream if this is not intended.")
            cudnn.set_stream(cudnn_handle, current_stream)

        if skip_compile:
            raise NotImplementedError("cudnn sliding window attention kernel does not support skip_compile")
        if self._cudnn_swa_graph is None or not self._cudnn_compiled:
            raise ValueError("SlidingWindowAttention kernel not compiled")
        self._logger.debug("Executing with compiled kernel")

        self._logger.debug("Reshaping tensors to kernel expected format")
        stats_tensor = self._pad_tensor_to_ndim(stats_tensor, self.sample_o.ndim, "stats_tensor") if stats_tensor is not None else None
        seq_len_q_tensor = self._pad_tensor_to_ndim(seq_len_q_tensor, 4, "seq_len_q_tensor")
        seq_len_kv_tensor = self._pad_tensor_to_ndim(seq_len_kv_tensor, 4, "seq_len_kv_tensor")
        q_ragged_offset_tensor = self._pad_tensor_to_ndim(q_ragged_offset_tensor, 4, "q_ragged_offset_tensor")
        k_ragged_offset_tensor = self._pad_tensor_to_ndim(k_ragged_offset_tensor, 4, "k_ragged_offset_tensor")
        v_ragged_offset_tensor = self._pad_tensor_to_ndim(v_ragged_offset_tensor, 4, "v_ragged_offset_tensor")
        o_ragged_offset_tensor = self._pad_tensor_to_ndim(o_ragged_offset_tensor, 4, "o_ragged_offset_tensor")
        stats_ragged_offset_tensor = self._pad_tensor_to_ndim(stats_ragged_offset_tensor, 4, "stats_ragged_offset_tensor")

        if not self.is_infer and stats_tensor is None:
            raise ValueError(f"stats_tensor must be provided when compiled in non-inference mode, got {stats_tensor}")

        if self.input_layout == "thd":
            if seq_len_q_tensor is None or seq_len_kv_tensor is None:
                raise ValueError(f"seq_len_q_tensor and seq_len_kv_tensor must be provided for thd layout, got {seq_len_q_tensor} and {seq_len_kv_tensor}")
            ragged = self._ragged_offset_set(
                q_ragged_offset_tensor,
                k_ragged_offset_tensor,
                v_ragged_offset_tensor,
                o_ragged_offset_tensor,
                stats_ragged_offset_tensor,
                prefix="",
                suffix="_tensor",
                required=True,
            )
            for name, tensor in ragged.items():
                self._check_ragged_offset(tensor, name, self.batch_size)
        elif (
            q_ragged_offset_tensor is not None
            or k_ragged_offset_tensor is not None
            or v_ragged_offset_tensor is not None
            or o_ragged_offset_tensor is not None
            or stats_ragged_offset_tensor is not None
        ):
            raise ValueError("SlidingWindowAttention.execute: ragged offset tensors are only defined for the T,H,D layout; pass None for bshd")

        variant_pack = {
            self.q_cudnn: q_tensor,
            self.k_cudnn: k_tensor,
            self.v_cudnn: v_tensor,
            self.o_cudnn: o_tensor,
            self.seq_len_q_cudnn: seq_len_q_tensor,
            self.seq_len_kv_cudnn: seq_len_kv_tensor,
            self.q_ragged_offset_cudnn: q_ragged_offset_tensor,
            self.k_ragged_offset_cudnn: k_ragged_offset_tensor,
            self.v_ragged_offset_cudnn: v_ragged_offset_tensor,
            self.o_ragged_offset_cudnn: o_ragged_offset_tensor,
        }
        if not self.is_infer:
            variant_pack[self.stats_cudnn] = stats_tensor
            variant_pack[self.stats_ragged_offset_cudnn] = stats_ragged_offset_tensor

        required = self.get_workspace_size()
        if required > 0:
            WorkspaceCarver(workspace, required, "SlidingWindowAttention")  # validates None / undersized / unaligned (R2)
            if not workspace.is_cuda or workspace.device != self.sample_q.device:
                raise ValueError(f"SlidingWindowAttention: workspace must be on the plan's device {self.sample_q.device}, got {workspace.device}")
        # pygraph.execute lowers workspace=None to a null pointer, legal only when the plan needs 0 bytes.
        self._cudnn_swa_graph.execute(variant_pack, workspace if required > 0 else None, handle=cudnn_handle)
        self._logger.debug("Executed successfully")

    def __call__(self, *args, **kwargs) -> None:
        self.execute(*args, skip_compile=True, **kwargs)


import logging

_logger = logging.getLogger(__name__)
_cache_of_SlidingWindowAttentionObjects = {}


def sliding_window_attention_wrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    seq_len_q_tensor: Optional[torch.Tensor] = None,
    seq_len_kv_tensor: Optional[torch.Tensor] = None,
    q_ragged_offset_tensor: Optional[torch.Tensor] = None,
    k_ragged_offset_tensor: Optional[torch.Tensor] = None,
    v_ragged_offset_tensor: Optional[torch.Tensor] = None,
    o_ragged_offset_tensor: Optional[torch.Tensor] = None,
    stats_ragged_offset_tensor: Optional[torch.Tensor] = None,
    left_bound: int = 0,
    right_bound: int = 0,
    is_infer: bool = False,
    attn_scale: Optional[float] = None,
    o_dtype: Optional[torch.dtype] = None,
    intermediate_data_type: torch.dtype = torch.float32,
    compute_data_type: torch.dtype = torch.float32,
    cudnn_handle: Optional[cudnn.handle] = None,
    stream: Optional[cuda.CUstream] = None,
    max_seq_len_q: Optional[int] = None,
    max_seq_len_kv: Optional[int] = None,
) -> TupleDict:
    """Allocate ``o`` (and ``stats`` unless ``is_infer``) and run sliding-window attention; objects are memoized per shape.

    This is the eager torch-op layer over ``SlidingWindowAttention``. ``T,H,D`` (rank-3 q): ``max_seq_len_q`` /
    ``max_seq_len_kv`` (the plan's launch envelope, part of the memo key) are read back from ``seq_len_*_tensor``
    (``.max().item()``, a device sync) when not passed; pass them to avoid the sync. Caller-provided ragged-offset
    tensors are made contiguous; when none is given the batch is taken as fully packed and the offsets are derived
    with ``NSA.packed_thd_ragged_offsets``. The backend workspace (``get_workspace_size()`` bytes) is allocated per
    call. All of that torch work runs on the launch stream: ``stream`` if given, else the cuDNN handle's stream.
    Use the class API with a caller-owned workspace and precomputed offsets to avoid it.
    """
    o_tensor, stats_tensor = None, None
    o_dtype = o_dtype if o_dtype is not None else q_tensor.dtype
    if q_tensor.ndim == 3:  # thd
        if seq_len_q_tensor is None or seq_len_kv_tensor is None:
            raise ValueError("sliding_window_attention_wrapper: seq_len_q_tensor and seq_len_kv_tensor are required for the T,H,D layout")
        if max_seq_len_q is None or max_seq_len_kv is None:
            # The reductions read tensors the caller may have produced asynchronously on `stream`: run them on the
            # launch stream (`stream`, else the handle's) so they see the final lengths.
            envelope_stream = stream if stream is not None else cudnn.get_stream(cudnn_handle) if cudnn_handle is not None else None
            with torch.cuda.device(q_tensor.device), stream_context(envelope_stream, q_tensor.device):
                if max_seq_len_q is None:
                    max_seq_len_q = int(seq_len_q_tensor.max().item())
                if max_seq_len_kv is None:
                    max_seq_len_kv = int(seq_len_kv_tensor.max().item())
        _logger.debug("sliding_window_attention_wrapper: Creating empty output tensor o for thd layout")
        t, h_q, d = q_tensor.shape
        _, h_k, d_v = v_tensor.shape
        o_tensor = make_tensor_strided_like(q_tensor, (t, h_q, d_v), dtype=o_dtype, device=q_tensor.device)
        if not is_infer:
            _logger.debug("sliding_window_attention_wrapper: Creating empty output tensor stats for thd layout")
            stats_tensor = make_tensor_strided_like(q_tensor, (t, h_q, 1), dtype=torch.float32, device=q_tensor.device)
    else:  # bshd
        _logger.debug("sliding_window_attention_wrapper: Creating empty output tensor o for bshd layout")
        b, h_q, s_q, d = q_tensor.shape
        _, h_k, s_k, d_v = v_tensor.shape
        o_tensor = make_tensor_strided_like(q_tensor, (b, h_q, s_q, d_v), dtype=o_dtype, device=q_tensor.device)
        if not is_infer:
            _logger.debug("sliding_window_attention_wrapper: Creating empty output tensor stats for bshd layout")
            stats_tensor = make_tensor_strided_like(q_tensor, (b, h_q, s_q, 1), dtype=torch.float32, device=q_tensor.device)

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        seq_len_q_tensor.shape if seq_len_q_tensor is not None else None,
        seq_len_kv_tensor.shape if seq_len_kv_tensor is not None else None,
        q_ragged_offset_tensor.shape if q_ragged_offset_tensor is not None else None,
        k_ragged_offset_tensor.shape if k_ragged_offset_tensor is not None else None,
        v_ragged_offset_tensor.shape if v_ragged_offset_tensor is not None else None,
        o_ragged_offset_tensor.shape if o_ragged_offset_tensor is not None else None,
        (stats_ragged_offset_tensor.shape if stats_ragged_offset_tensor is not None else None),
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        seq_len_q_tensor.stride() if seq_len_q_tensor is not None else None,
        seq_len_kv_tensor.stride() if seq_len_kv_tensor is not None else None,
        q_ragged_offset_tensor.stride() if q_ragged_offset_tensor is not None else None,
        k_ragged_offset_tensor.stride() if k_ragged_offset_tensor is not None else None,
        v_ragged_offset_tensor.stride() if v_ragged_offset_tensor is not None else None,
        o_ragged_offset_tensor.stride() if o_ragged_offset_tensor is not None else None,
        (stats_ragged_offset_tensor.stride() if stats_ragged_offset_tensor is not None else None),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        left_bound,
        right_bound,
        is_infer,
        attn_scale,
        intermediate_data_type,
        compute_data_type,
        max_seq_len_q,
        max_seq_len_kv,
    )
    sliding_window_attention_object = _cache_of_SlidingWindowAttentionObjects.get(cache_key)
    if cudnn_handle is None and sliding_window_attention_object is None:
        cudnn_handle = cudnn.create_handle()
    handle = cudnn_handle if cudnn_handle is not None else sliding_window_attention_object._cudnn_handle
    # The graph launches on `stream` (re-streaming the handle) or on the handle's own stream; every torch op below
    # must run there (R1: the caching allocator orders a block's reuse only against its allocation stream).
    launch = stream if stream is not None else cudnn.get_stream(handle)

    if q_tensor.ndim == 3:
        ragged_given = [q_ragged_offset_tensor, k_ragged_offset_tensor, v_ragged_offset_tensor, o_ragged_offset_tensor, stats_ragged_offset_tensor]
        with torch.cuda.device(q_tensor.device), stream_context(launch, q_tensor.device):
            if all(t is None for t in ragged_given):
                (
                    q_ragged_offset_tensor,
                    k_ragged_offset_tensor,
                    v_ragged_offset_tensor,
                    o_ragged_offset_tensor,
                    stats_ragged_offset_tensor,
                ) = packed_thd_ragged_offsets(seq_len_q_tensor, seq_len_kv_tensor, q_tensor, k_tensor, v_tensor, o_tensor, stats_tensor)
            else:
                (
                    q_ragged_offset_tensor,
                    k_ragged_offset_tensor,
                    v_ragged_offset_tensor,
                    o_ragged_offset_tensor,
                    stats_ragged_offset_tensor,
                ) = (
                    contiguous_on_stream(t, launch, q_tensor.device) for t in ragged_given
                )  # R1 staging: originals recorded first

    def allocate_workspace(obj: SlidingWindowAttention) -> torch.Tensor:
        with torch.cuda.device(q_tensor.device), stream_context(launch, q_tensor.device):
            return torch.empty(max(obj.get_workspace_size(), 1), dtype=torch.uint8, device=q_tensor.device)

    if sliding_window_attention_object is not None:
        _logger.debug("sliding_window_attention_wrapper: Using previously cached SlidingWindowAttention object")

        sliding_window_attention_object.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            stats_tensor=stats_tensor,
            seq_len_q_tensor=seq_len_q_tensor,
            seq_len_kv_tensor=seq_len_kv_tensor,
            q_ragged_offset_tensor=q_ragged_offset_tensor,
            k_ragged_offset_tensor=k_ragged_offset_tensor,
            v_ragged_offset_tensor=v_ragged_offset_tensor,
            o_ragged_offset_tensor=o_ragged_offset_tensor,
            stats_ragged_offset_tensor=stats_ragged_offset_tensor,
            workspace=allocate_workspace(sliding_window_attention_object),
            current_stream=stream,
            cudnn_handle=cudnn_handle,
        )
    else:
        _logger.debug("sliding_window_attention_wrapper: No previously cached SlidingWindowAttention object found, creating new SlidingWindowAttention object")
        sliding_window_attention_object = SlidingWindowAttention(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_stats=stats_tensor,
            sample_seq_len_q=seq_len_q_tensor,
            sample_seq_len_kv=seq_len_kv_tensor,
            sample_q_ragged_offset=q_ragged_offset_tensor,
            sample_k_ragged_offset=k_ragged_offset_tensor,
            sample_v_ragged_offset=v_ragged_offset_tensor,
            sample_o_ragged_offset=o_ragged_offset_tensor,
            sample_stats_ragged_offset=stats_ragged_offset_tensor,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            left_bound=left_bound,
            right_bound=right_bound,
            attn_scale=attn_scale,
            intermediate_data_type=intermediate_data_type,
            compute_data_type=compute_data_type,
            cudnn_handle=cudnn_handle,
        )

        assert sliding_window_attention_object.check_support()
        sliding_window_attention_object.compile()
        sliding_window_attention_object.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            stats_tensor=stats_tensor,
            seq_len_q_tensor=seq_len_q_tensor,
            seq_len_kv_tensor=seq_len_kv_tensor,
            q_ragged_offset_tensor=q_ragged_offset_tensor,
            k_ragged_offset_tensor=k_ragged_offset_tensor,
            v_ragged_offset_tensor=v_ragged_offset_tensor,
            o_ragged_offset_tensor=o_ragged_offset_tensor,
            stats_ragged_offset_tensor=stats_ragged_offset_tensor,
            workspace=allocate_workspace(sliding_window_attention_object),
            current_stream=stream,
        )
        _cache_of_SlidingWindowAttentionObjects[cache_key] = sliding_window_attention_object

    return TupleDict(
        o_tensor=o_tensor,
        stats_tensor=stats_tensor,
    )
