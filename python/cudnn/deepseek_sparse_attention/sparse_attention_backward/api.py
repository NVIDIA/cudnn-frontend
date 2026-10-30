# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for DeepSeek Sparse Attention backward.

The wrapper dispatches to the Hopper (SM90) or Blackwell (SM100) CuTe DSL
implementation based on the active CUDA device. It consumes the ``out`` and
``lse`` tensors produced by the DSA sparse-attention forward path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn.deepseek_sparse_attention.utils.runtime import device_capability

from cudnn.api_base import APIBase, TupleDict
from cudnn._torch_stream import contiguous_on_stream, copy_into_on_stream
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context

from . import _interface_sm100 as _iface_sm100
from . import _interface_sm100_d576 as _iface_d576


def _checked_deterministic_relay_cluster_size(value: Optional[int]) -> Optional[int]:
    """Validate the public plan-time relay override without coercion."""
    if value is None:
        return None
    if type(value) is not int or value not in (4, 8, 16, 32, 64):
        raise ValueError(f"deterministic_relay_cluster_size must be one of {{4, 8, 16, 32, 64}} or None, got {value!r}")
    return value


class SparseAttentionBackward(APIBase):
    """Validated architecture-dispatch wrapper for DSA backward."""

    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D) FP16/BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D) FP16/BF16 (K=V)
        sample_out: torch.Tensor,  # (total_S_q, H, D_v)
        sample_dout: torch.Tensor,  # (total_S_q, H, D_v)
        sample_lse: torch.Tensor,  # (total_S_q, H) FP32, KV-only LSE
        sample_attn_sink: torch.Tensor,  # (H,) FP32
        sample_topk_idxs: torch.Tensor,  # (total_S_q, topk_max) INT32
        sample_dq: Optional[torch.Tensor] = None,
        sample_dkv: Optional[torch.Tensor] = None,
        sample_topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        block_tile: int = 64,
        deterministic: bool = False,
        q_cluster_mode: str = "off",
        deterministic_relay_cluster_size: Optional[int] = None,
    ):
        """Capture the sample tensor contract and execution policy.

        ``q_cluster_mode='adaptive_pair'`` opts H16/H32 Dqk576 into the
        graph-safe adjacent-query sharing plan; the default keeps the existing
        backend unchanged. H16 ``topk_max <= 256`` and H32 ``topk_max <= 128``
        accept the mode but retain the ordinary backend.

        ``deterministic_relay_cluster_size`` optionally requests an exact
        4-, 8-, 16-, 32-, or 64-CTA logical relay for a deterministic SM100
        generic-M64 large-context plan. The 32- and 64-CTA options use two and
        four ordered 16-CTA hardware-cluster launches, respectively. ``None`` retains automatic
        selection. The choice is compiled into the plan and changes its
        workspace requirement; it cannot be changed by ``execute()`` or
        during CUDA Graph replay.
        """
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.dout_desc = self._make_tensor_desc(sample_dout, name="sample_dout")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.block_tile = int(block_tile)
        self.softmax_scale = softmax_scale
        self.deterministic = bool(deterministic)
        self.q_cluster_mode = q_cluster_mode
        self.deterministic_relay_cluster_size = _checked_deterministic_relay_cluster_size(deterministic_relay_cluster_size)
        self._backend = None
        self._backend_block_tile = None
        self._device_major = None
        self._two_cta_split_count = 1
        self._qcluster_plan = None
        self._wide_addressing = False

    def _validate_compiled_inputs(self, q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length) -> None:
        """Require execution inputs to match the plan's compiled contract."""
        named_tensors = (
            ("q", q, self.q_desc),
            ("kv", kv, self.kv_desc),
            ("out", out, self.out_desc),
            ("dout", dout, self.dout_desc),
            ("lse", lse, self.lse_desc),
            ("attn_sink", attn_sink, self.attn_sink_desc),
            ("topk_idxs", topk_idxs, self.topk_idxs_desc),
            ("topk_length", topk_length, self.topk_length_desc),
        )
        for name, tensor, desc in named_tensors:
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled plan")
            if desc is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tuple(tensor.shape) != desc.shape or tensor.dtype != desc.dtype or tensor.device != desc.device:
                raise ValueError(f"{name} shape, dtype, and device must match the compiled plan")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if lse.data_ptr() % 8 != 0:
            raise ValueError(f"lse must be 8-byte aligned for the SM100 FP32 pair-copy path; got data_ptr=0x{lse.data_ptr():x}")

    def check_support(self) -> bool:
        """Validate the device, dtype, shape, and deterministic contracts."""
        self._value_error_if(self.q_desc.device.type != "cuda", f"Q must live on CUDA, got {self.q_desc.device}")
        capability = torch.cuda.get_device_capability(self.q_desc.device)
        major, _ = capability
        self._device_major = major
        self._runtime_error_if(
            major < 9,
            f"SparseAttentionBackward requires SM90+, found SM{major}",
        )
        self._value_error_if(
            self.q_desc.ndim != 3,
            f"Q must be 3-D (total_S_q, H, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.kv_desc.ndim != 2,
            f"KV must be 2-D (total_S_kv, D), got {self.kv_desc.shape}",
        )
        self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        self._check_dtype(
            self.kv_desc,
            self.q_desc.dtype,
            name="KV",
            extra_error_msg="KV must have same dtype as Q",
        )
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.attn_sink_desc, torch.float32, name="attn_sink")
        self._check_dtype(self.topk_idxs_desc, torch.int32, name="topk_idxs")
        self._check_dtype(self.out_desc, self.q_desc.dtype, name="out", extra_error_msg="out must have same dtype as Q")
        self._check_dtype(self.dout_desc, self.q_desc.dtype, name="dout", extra_error_msg="dout must have same dtype as Q")
        if self.topk_length_desc is not None:
            self._check_dtype(self.topk_length_desc, torch.int32, name="topk_length")

        # Device placement + cross-tensor device consistency. The SM90/SM100
        # kernels are CUDA-only and reject CPU or cross-device inputs at
        # execution time (see the is_cuda / same-device assert in
        # ``_interface_sm100.flash_attn_bwd_sm100``), so a placement mismatch
        # must fail the support gate here rather than compile/launch and crash.
        ref_device = self.q_desc.device
        named_descriptors = [
            ("q", self.q_desc),
            ("kv", self.kv_desc),
            ("out", self.out_desc),
            ("dout", self.dout_desc),
            ("lse", self.lse_desc),
            ("attn_sink", self.attn_sink_desc),
            ("topk_idxs", self.topk_idxs_desc),
        ]
        if self.topk_length_desc is not None:
            named_descriptors.append(("topk_length", self.topk_length_desc))
        descriptors = [desc for _, desc in named_descriptors]
        self._value_error_if(
            any(desc.device != ref_device for desc in descriptors),
            f"All inputs must share Q's device {ref_device}, got {[desc.device for desc in descriptors]}",
        )

        # Cross-tensor shape contract: every companion tensor is indexed with
        # coordinates derived from Q, so a mismatched shape silently reads or
        # writes out of place at execution time instead of failing.
        total_s_q, num_heads, head_dim = self.q_desc.shape
        self._value_error_if(
            self.q_cluster_mode not in ("off", "adaptive_pair"),
            f"q_cluster_mode must be 'off' or 'adaptive_pair', got {self.q_cluster_mode!r}",
        )
        if self.q_cluster_mode == "adaptive_pair":
            max_topk = self.topk_idxs_desc.shape[1] if self.topk_idxs_desc.ndim == 2 else 0
            self._value_error_if(
                capability not in _iface_sm100._BLACKWELL_CAPABILITIES
                or self.q_desc.dtype != torch.bfloat16
                or num_heads not in (16, 32)
                or head_dim != 576
                or self.deterministic
                or not 0 < max_topk <= 2048
                or not all(desc.is_contiguous() for desc in descriptors),
                "q_cluster_mode='adaptive_pair' requires contiguous SM100 BF16 H16/H32 Dqk576 inputs, " "non-deterministic execution, and 1 <= topk <= 2048",
            )
        self._value_error_if(
            self.deterministic and (major != 10 or num_heads not in _iface_sm100._DETERMINISTIC_HEAD_COUNTS),
            f"deterministic DSA backward requires SM100 and heads in {_iface_sm100._DETERMINISTIC_HEAD_COUNTS}, found SM{major} H{num_heads}",
        )
        self._value_error_if(
            self.deterministic_relay_cluster_size is not None and not self.deterministic,
            "deterministic_relay_cluster_size requires deterministic=True",
        )
        # The SM100 kernel is tiled only for head_dim in {512, 576} (the 576
        # MLA case splits QK=576 / V=512); any other head_dim compiles to a
        # layout that indexes shared memory out of bounds and crashes.
        self._value_error_if(
            head_dim not in (512, 576),
            f"head_dim must be 512 or 576, got {head_dim}",
        )
        head_dim_v = 512 if head_dim == 576 else head_dim
        expected_o_shape = (total_s_q, num_heads, head_dim_v)
        self._value_error_if(
            self.kv_desc.shape[1] != head_dim,
            f"KV must have shape (total_S_kv, {head_dim}), got {self.kv_desc.shape}",
        )
        self._value_error_if(
            self.out_desc.shape != expected_o_shape,
            f"out must have shape {expected_o_shape}, got {self.out_desc.shape}",
        )
        self._value_error_if(
            self.dout_desc.shape != expected_o_shape,
            f"dout must have shape {expected_o_shape}, got {self.dout_desc.shape}",
        )
        self._value_error_if(
            self.lse_desc.shape != (total_s_q, num_heads),
            f"LSE must have shape {(total_s_q, num_heads)}, got {self.lse_desc.shape}",
        )
        self._value_error_if(
            self.attn_sink_desc.shape != (num_heads,),
            f"attn_sink must have shape {(num_heads,)}, got {self.attn_sink_desc.shape}",
        )
        self._value_error_if(
            self.topk_idxs_desc.ndim != 2 or self.topk_idxs_desc.shape[0] != total_s_q,
            f"topk_idxs must have shape ({total_s_q}, topk_max), got {self.topk_idxs_desc.shape}",
        )
        if self.topk_length_desc is not None:
            self._value_error_if(
                self.topk_length_desc.shape != (total_s_q,),
                f"topk_length must have shape {(total_s_q,)}, got {self.topk_length_desc.shape}",
            )

        # Resolve the SM100 backend once so compile()/execute() and the wrapper
        # agree on the route; the interface repeats the selection for direct
        # callers of ``flash_attn_bwd_sm100``.
        self._backend = None
        self._backend_block_tile = None
        self._wide_addressing = False
        if major == 10:
            self._backend, self._backend_block_tile = _iface_sm100._select_sm100_backend(
                num_heads,
                head_dim,
                head_dim_v=head_dim_v,
                dtype=self.q_desc.dtype,
                max_topk=self.topk_idxs_desc.shape[1],
                device_capability=capability,
                deterministic=self.deterministic,
                is_contiguous=all(desc.is_contiguous() for desc in descriptors),
            )
            self._wide_addressing = _iface_sm100._requires_wide_data_addressing(
                total_s_q,
                self.kv_desc.shape[0],
                head_dim,
                head_dim_v,
                num_heads,
                self.topk_idxs_desc.shape[1],
                self.deterministic,
                self._backend,
                include_workspace=self._backend not in _iface_d576.BACKENDS,
                deterministic_relay_cluster_size=self.deterministic_relay_cluster_size,
            )
        if self.deterministic and self._backend == "generic_m64":
            # Resolve the shape-dependent policy during support checking so a
            # relay-only DSL requirement fails before compilation. Ordinary
            # shard and tuned deterministic routes retain the 4.6.2 floor.
            _iface_sm100._get_sm100_kernel_class(
                self._backend,
                True,
                total_s_q,
                self.kv_desc.shape[0],
                head_dim,
                num_heads,
                self.deterministic_relay_cluster_size,
            )
        elif self.deterministic_relay_cluster_size is not None:
            raise ValueError("deterministic_relay_cluster_size is supported only by the deterministic SM100 generic_m64 backend")
        if self._backend in _iface_d576.BACKENDS:
            self._value_error_if(total_s_q <= 0 or self.kv_desc.shape[0] <= 0, "Q and KV sequence extents must be positive")
            if self.deterministic:
                self._two_cta_split_count = 1
            else:
                self._two_cta_split_count = _iface_d576._split_count(
                    total_s_q, self.topk_idxs_desc.shape[1], torch.cuda.get_device_properties(self.q_desc.device).multi_processor_count
                )
        else:
            # The kernels read the declared layout natively and execute never
            # repacks (R5); the D576 route already rejects strides at execute.
            for name, desc in named_descriptors:
                self._not_implemented_error_if(
                    not desc.is_contiguous(),
                    f"SparseAttentionBackward addresses only contiguous {name}; got shape {desc.shape} strides {desc.stride}",
                )

        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile every artifact required by the selected prepared plan.

        SM100 compilation uses only stored tensor metadata and completes before
        ``execute()`` or CUDA Graph capture. The compiled artifact belongs to
        the sample tensors' CUDA device; keep this plan alive while a graph
        that launches it may replay. The legacy SM90 interface still compiles
        from concrete tensors on its first execution.
        """
        self._ensure_support_checked()
        if self.q_cluster_mode == "adaptive_pair":
            from ._qcluster_sm100 import compile_plan

            total_q, heads, dim = self.q_desc.shape
            tile = 128 if heads == 16 else 64
            if total_q >= 2 and self.topk_idxs_desc.shape[1] > 2 * tile:
                with torch.cuda.device(self.q_desc.device):
                    self._qcluster_plan = compile_plan(
                        torch.cuda.get_device_capability(self.q_desc.device),
                        total_q,
                        self.kv_desc.shape[0],
                        heads,
                        dim,
                        self.topk_idxs_desc.shape[1],
                        self.topk_length_desc is not None,
                    )
                self._compiled_kernel = self._qcluster_plan
                return
        if self._backend in _iface_d576.BACKENDS:
            with torch.cuda.device(self.q_desc.device):
                self._compiled_kernel = _iface_d576._compile_d576_2cta(
                    torch.cuda.get_device_capability(self.q_desc.device),
                    self.topk_idxs_desc.shape[1],
                    self.topk_length_desc is not None,
                    self.q_desc.shape[0] == 1,
                    self._two_cta_split_count > 1,
                    self.deterministic,
                    self.q_desc.shape[1],
                    (
                        _iface_d576._deterministic_compact_slots(torch.cuda.get_device_properties(self.q_desc.device).multi_processor_count)
                        if self.deterministic
                        else 74
                    ),
                    self._wide_addressing,
                )
            return
        if self.deterministic:
            total_s_q, num_heads, head_dim = self.q_desc.shape
            head_dim_v = self.out_desc.shape[2]
            kernel_cls = _iface_sm100._get_sm100_kernel_class(
                self._backend,
                True,
                total_s_q,
                self.kv_desc.shape[0],
                head_dim,
                num_heads,
                self.deterministic_relay_cluster_size,
            )
            shards = kernel_cls.num_dkv_shards
            relay_cluster_size = getattr(kernel_cls, "dkv_relay_cluster_size", 1)
            with torch.cuda.device(self.q_desc.device):
                self._compiled_kernel = _iface_sm100._compile_deterministic_sm100(
                    self.q_desc.device.index,
                    torch.cuda.get_device_capability(self.q_desc.device),
                    self._backend,
                    self.q_desc.dtype,
                    head_dim,
                    head_dim_v,
                    num_heads,
                    self.topk_idxs_desc.shape[1],
                    self.topk_length_desc is not None,
                    shards,
                    relay_cluster_size,
                    self._wide_addressing,
                )
            return
        if self._device_major == 10:
            _, num_heads, head_dim = self.q_desc.shape
            with torch.cuda.device(self.q_desc.device):
                self._compiled_kernel = _iface_sm100._compile_nondeterministic_sm100(
                    self.q_desc.device.index,
                    torch.cuda.get_device_capability(self.q_desc.device),
                    self._backend,
                    self.q_desc.dtype,
                    head_dim,
                    self.out_desc.shape[2],
                    num_heads,
                    self.topk_idxs_desc.shape[1],
                    self.topk_length_desc is not None,
                    self._backend_block_tile,
                    self._wide_addressing,
                )
            return
        # SM90 owns concrete-tensor caches and retains first-execute JIT.
        self._compiled_kernel = True

    def scratch_workspace_bytes(self) -> int:
        """Return the minimum reusable per-execution workspace in bytes.

        Generic deterministic SM100 plans include a query-dependent 32-, 64-,
        or 128-plane FP32 dKV accumulator. Large-context plans that would
        exceed 16 GiB instead use 8, 16, or 32 planes plus packed row-relay
        masks; the eight-plane memory-first tier is selected when the preferred
        relay would exceed 32 GiB. An explicit relay cluster size reports that
        exact specialization even when it exceeds the automatic budget.
        Deterministic H96/H128 D576 two-CTA plans use the compact 96-plane
        capacity. The returned size is authoritative for this plan;
        ``execute()`` accepts a larger contiguous uint8 buffer on the plan
        device but never allocates one.
        """
        self._ensure_support_checked()
        major, _ = device_capability(self.q_desc.device)
        total_s_q, num_heads, head_dim = self.q_desc.shape
        total_s_kv = self.kv_desc.shape[0]
        tile = 128 if num_heads == 16 else 64
        if self.q_cluster_mode == "adaptive_pair" and total_s_q >= 2 and self.topk_idxs_desc.shape[1] > 2 * tile:
            from ._qcluster_sm100 import workspace_size

            return workspace_size(total_s_q, total_s_kv, head_dim, num_heads, self.topk_idxs_desc.shape[1])
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.flash_attn_bwd_sm90_workspace_size(total_s_q, total_s_kv, head_dim, num_heads)
        return _iface_sm100.flash_attn_bwd_sm100_workspace_size(
            total_s_q,
            total_s_kv,
            head_dim,
            num_heads,
            self.deterministic,
            backend=self._backend,
            deterministic_relay_cluster_size=self.deterministic_relay_cluster_size,
        )

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        dq: torch.Tensor,
        dkv: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        workspace: Optional[torch.Tensor] = None,
        *,
        d_sink: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch one validated execution to the active GPU architecture.

        Every backend requires ``dq``, ``dkv``, ``d_sink`` and (when
        ``scratch_workspace_bytes()`` is non-zero) ``workspace`` to be
        caller-provided; execute never allocates. ``d_sink`` is keyword-only so
        the positional order of the pre-existing parameters is unchanged.
        Every prepared SM100 route additionally never compiles here. Adaptive
        pairing makes its overlap decision on device and launches only stages
        compiled by ``compile()``, so replay needs no host branch or D2H read.
        SM100 runtime inputs must match the sample contract used to compile the
        plan. SM90 retains its legacy first-execute compilation behavior.
        Inputs, outputs, and the live workspace must not overlap; concurrent
        executions need distinct output and workspace storage.
        """
        self._ensure_support_checked()
        for name, tensor in (("dq", dq), ("dkv", dkv), ("d_sink", d_sink)):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        if self._device_major == 10:
            if self._compiled_kernel is None:
                raise RuntimeError("call compile() before execute()")
            self._validate_compiled_inputs(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)
        if self._backend in _iface_d576.BACKENDS:
            return _iface_d576._execute_d576_2cta(
                self._compiled_kernel,
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                topk_length,
                dq,
                dkv,
                d_sink,
                workspace,
                scale,
                self._two_cta_split_count,
                current_stream,
                deterministic=self.deterministic,
            )
        if self._qcluster_plan is not None:
            scale = 1.0 / math.sqrt(self.q_desc.shape[2]) if scale is None else scale
            with torch.cuda.device(q.device):
                return self._qcluster_plan.execute(
                    q,
                    kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    topk_idxs,
                    topk_length,
                    dq,
                    dkv,
                    d_sink,
                    workspace,
                    scale,
                    current_stream,
                )
        # Resolve the architecture from Q's device rather than the ambient current
        # device, and launch under that device context, matching check_support().
        major, _ = device_capability(q.device)
        with torch.cuda.device(q.device):
            if major == 9:
                from . import _interface_sm90 as _iface_sm90

                return _iface_sm90.flash_attn_bwd_sm90(
                    q,
                    kv,
                    out,
                    dout,
                    lse,
                    attn_sink,
                    topk_idxs,
                    dq,
                    dkv,
                    d_sink,
                    workspace,
                    softmax_scale=scale,
                    topk_length=topk_length,
                    current_stream=current_stream,
                )
            return _iface_sm100.flash_attn_bwd_sm100(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                softmax_scale=scale,
                topk_length=topk_length,
                dq=dq,
                dkv=dkv,
                d_sink=d_sink,
                deterministic=self.deterministic,
                workspace=workspace,
                current_stream=current_stream,
                _compiled_kernel=self._compiled_kernel,
                deterministic_relay_cluster_size=self.deterministic_relay_cluster_size,
            )


_cache_of_SparseAttentionBackwardObjects: dict = {}


def sparse_attention_backward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    block_tile: int = 64,
    deterministic: bool = False,
    stream: Optional[cuda.CUstream] = None,
    workspace: Optional[torch.Tensor] = None,
    q_cluster_mode: str = "off",
    deterministic_relay_cluster_size: Optional[int] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'dq', 'dkv', 'd_sink'}``.

    Dispatches to SM90 or SM100 from the input device and tensor metadata. The
    returned ``d_sink`` is computed from ``attn_sink`` and ``dout``. Set
    ``deterministic=True`` for bitwise-reproducible
    H16/H32/H64/H96/H128 gradients on SM100. The optional reusable uint8
    ``workspace`` must hold at least
    ``SparseAttentionBackward.scratch_workspace_bytes()`` bytes.

    An SM100 ``SparseAttentionBackward.execute()`` is capture-safe after
    ``compile()``. Warm this convenience wrapper before capture so its cached
    plan has already been constructed and compiled. SM90 still requires one
    eager execution to populate its concrete-tensor compile caches.

    ``q_cluster_mode='adaptive_pair'`` partitions each adjacent query pair on
    the GPU and evaluates a profitable shared KV segment once. The opt-in
    SM100/SM103 BF16 H16/H32 Dqk576 path accepts contiguous raw top-k rows with
    ``topk_max <= 2048``, requires non-deterministic execution, and remains
    graph-capturable. Low-overlap pairs fall back on device; H16
    ``topk_max <= 256`` and H32 ``topk_max <= 128`` use the ordinary backend.
    The wrapper stages strided inputs and outputs and allocates omitted outputs
    or workspace on the launch stream; the plan's ``execute()`` does neither.

    ``deterministic_relay_cluster_size`` optionally fixes a deterministic
        SM100 generic-M64 large-context plan to a 4-, 8-, 16-, 32-, or 64-CTA
        logical relay. Cluster-32/64 use two/four ordered 16-CTA hardware-cluster launches. The
    default ``None`` uses the automatic policy. This plan-time choice changes
    the required workspace and remains fixed across every execution and CUDA
    Graph replay of the cached plan.
    """
    deterministic_relay_cluster_size = _checked_deterministic_relay_cluster_size(deterministic_relay_cluster_size)
    if deterministic_relay_cluster_size is not None and not deterministic:
        raise ValueError("deterministic_relay_cluster_size requires deterministic=True")
    if q_cluster_mode not in ("off", "adaptive_pair"):
        raise ValueError(f"q_cluster_mode must be 'off' or 'adaptive_pair', got {q_cluster_mode!r}")
    # The plan addresses contiguous tensors only (check_support declines the rest, R5).
    # The wrapper is the eager torch-op layer: it normalises strided inputs here, on the
    # launch stream, as it always did, and stages a strided caller output so the
    # caller's tensor keeps its identity.
    with torch.cuda.device(q.device):
        launch_stream = resolve_stream(stream)
    # R1 staging: each original is record_stream'ed on the launch stream before it is rebound.
    q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length = (
        contiguous_on_stream(t, launch_stream, q.device) for t in (q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)
    )
    key = (
        q.device,
        q.dtype,
        q.shape,
        kv.shape,
        out.shape,
        dout.shape,
        lse.shape,
        attn_sink.shape,
        topk_idxs.shape,
        topk_length is not None,
        int(block_tile),
        softmax_scale,
        bool(deterministic),
        q_cluster_mode,
        deterministic_relay_cluster_size,
    )
    obj = _cache_of_SparseAttentionBackwardObjects.get(key)
    if obj is None:
        obj = SparseAttentionBackward(
            sample_q=q,
            sample_kv=kv,
            sample_out=out,
            sample_dout=dout,
            sample_lse=lse,
            sample_attn_sink=attn_sink,
            sample_topk_idxs=topk_idxs,
            sample_topk_length=topk_length,
            softmax_scale=softmax_scale,
            block_tile=block_tile,
            deterministic=deterministic,
            q_cluster_mode=q_cluster_mode,
            deterministic_relay_cluster_size=deterministic_relay_cluster_size,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseAttentionBackwardObjects[key] = obj

    dq_user, dkv_user = dq, dkv
    with torch.cuda.device(q.device):
        # execute() never allocates: outputs and scratch are provided here, ordered with the launch stream (R2).
        with torch_stream_context(launch_stream):
            if workspace is None:
                workspace_bytes = obj.scratch_workspace_bytes()
                if workspace_bytes:
                    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
            if dq is None:
                dq = torch.empty_like(q)
            elif not dq.is_contiguous():
                dq = torch.empty(dq.shape, dtype=dq.dtype, device=dq.device)  # staged; copied back into the caller's tensor below
            if dkv is None:
                dkv = torch.empty_like(kv)
            elif not dkv.is_contiguous():
                dkv = torch.empty(dkv.shape, dtype=dkv.dtype, device=dkv.device)
            d_sink = torch.empty_like(attn_sink)

    dq_out, dkv_out, d_sink_out = obj.execute(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        dq,
        dkv,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        workspace=workspace,
        current_stream=launch_stream,
        d_sink=d_sink,
    )
    # Copy-back into the caller's strided outputs: the destination is recorded on the launch stream first (R1 staging).
    if dq_user is not None and dq_user is not dq_out:
        copy_into_on_stream(dq_user, dq_out, launch_stream, q.device)
        dq_out = dq_user
    if dkv_user is not None and dkv_user is not dkv_out:
        copy_into_on_stream(dkv_user, dkv_out, launch_stream, q.device)
        dkv_out = dkv_user
    return TupleDict(dq=dq_out, dkv=dkv_out, d_sink=d_sink_out)
