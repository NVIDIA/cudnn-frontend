# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MegaMoE (CuTe DSL) device backend for :class:`cudnn.MoeEp`.

This module wires the ``megamoe`` training package (the SM100 MegaMoE MXFP8
mega-kernel: NVSHMEM dispatch -> grouped FC1/SwiGLU/FC2 -> combine in one
launch, plus the ``bwd_impl="mega"`` backward kernel) behind the public
``MoeEp`` contract.  ``api.MoeEp`` calls :func:`maybe_create` lazily on the
first ``__call__``; when the environment or the operator configuration is not
supported the call returns ``None`` and ``MoeEp`` keeps its allocate-only
behavior.

Environment:

- ``CUDNN_MOE_EP_BACKEND``: ``auto`` (default; use megamoe when possible,
  warn once and fall back otherwise), ``megamoe`` (required; raise if the
  backend cannot be built), ``none`` (never use a backend).
- ``CUDNN_MEGAMOE_ROOT``: path to the ``moe_ep_training`` checkout that
  contains the ``megamoe`` package (with its sibling ``cutedsl_megamoe``
  kernel clone).  Not needed when ``megamoe`` is already importable.

Backend requirements and deviations from the pure reference:

- SM100 (GB200) GPU, ``nvidia-cutlass-dsl``, ``nvshmem4py``; the first call
  pays the one-time ``cute.compile`` (minutes per kernel).
- Internal compute is MXFP8: bf16 activations/weights are quantized on
  device, so outputs match the FP32 reference at MXFP8 tolerance (~1e-2
  relative), not bitwise.  Block-scaled inputs are dequantized and
  re-quantized (straight-through, matching the reference's gradient
  convention but adding one rounding).
- ``apply_topk_in_fc1=False`` is not supported by the kernel.
- ``combine_format="mxfp8"`` is forward-only (kernel wire format
  ``32e4m3xe8m0``); ``generate_c``/backward need bf16 combine staging.
  ``combine_format="nvfp4"`` is unsupported (the NVFP4 wire format belongs
  to the NVFP4 kernel, which has no training path).
- Quantized ``output_format`` is produced by quantizing the kernel's bf16
  result on the host side of the launch (reference-identical algorithm).
- ``backward`` consumes the kernel's persistent pools from the immediately
  preceding forward on the same operator (the passed ``fc1_c``/
  ``route_metadata`` are validated but the pool stash is the source of
  truth).  One forward per backward; a second forward overwrites the stash.
- With an ``ep_group`` every rank must call forward/backward collectively,
  the group must span the whole NVSHMEM/torch.distributed world, and all
  ranks must pass the same token count ``T``.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .api import BlockScaledTensor, MoeFormat, MoeTensor

_COMBINE_WIRE = {
    MoeFormat.BF16: "bf16",
    MoeFormat.MXFP8: "32e4m3xe8m0",
}

_state = {
    "modules": None,       # imported megamoe modules, once per process
    "mode": None,          # "no_dist" | "dist"
    "nvshmem_ready": False,
    "warned": False,
}


class BackendUnavailable(RuntimeError):
    """Raised when the megamoe backend cannot serve this MoeEp config."""


def _policy() -> str:
    value = os.environ.get("CUDNN_MOE_EP_BACKEND", "auto").lower()
    if value in ("", "auto"):
        return "auto"
    if value in ("none", "0", "off"):
        return "none"
    if value == "megamoe":
        return "megamoe"
    raise ValueError(f"CUDNN_MOE_EP_BACKEND must be auto|megamoe|none, got {value!r}")


def _import_megamoe(single_rank: bool):
    """Import the megamoe package once; the dist/no-dist mode is frozen at
    first import (``MEGA_NO_DIST`` is read at module import time)."""

    if _state["modules"] is not None:
        want = "no_dist" if single_rank else "dist"
        if _state["mode"] != want:
            raise BackendUnavailable(
                f"megamoe already imported in {_state['mode']!r} mode; cannot "
                f"serve a {want!r} MoeEp in the same process"
            )
        return _state["modules"]

    root = os.environ.get("CUDNN_MEGAMOE_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    if single_rank:
        os.environ.setdefault("MEGA_NO_DIST", "1")

    import megamoe.repo_path  # noqa: F401  (sys.path shim for the kernel clone)
    from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward  # noqa: F401
    from megamoe.training import MegaMoeHybridMxfp8Layer
    from megamoe.pools import (
        decode_token_src_metadata,
        local_pool_views,
    )
    from pt.config import EpConfig
    from pt.quant import QuantConfig

    _state["modules"] = {
        "MegaMoeForwardConfig": MegaMoeForwardConfig,
        "MegaMoeHybridMxfp8Layer": MegaMoeHybridMxfp8Layer,
        "decode_token_src_metadata": decode_token_src_metadata,
        "local_pool_views": local_pool_views,
        "EpConfig": EpConfig,
        "QuantConfig": QuantConfig,
    }
    _state["mode"] = "no_dist" if single_rank else "dist"
    return _state["modules"]


def _ensure_single_rank_dist():
    """The hybrid layer's comm needs an initialized process group even for
    ``ep_size=1``; bootstrap a world-1 NCCL group like the megamoe tests."""

    if dist.is_available() and dist.is_initialized():
        return
    import tempfile

    store_path = tempfile.mktemp(prefix="cudnn_moe_ep_pg_")
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, 1),
        world_size=1,
        rank=0,
    )


def _ensure_nvshmem():
    if _state["nvshmem_ready"]:
        return
    from src.bootstrap import init_dist_and_nvshmem  # via megamoe.repo_path

    init_dist_and_nvshmem()
    _state["nvshmem_ready"] = True


def _check_supported(op) -> None:
    """Raise :class:`BackendUnavailable` when this MoeEp configuration cannot
    run on the MegaMoE kernel."""

    if not torch.cuda.is_available():
        raise BackendUnavailable("CUDA is not available")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise BackendUnavailable(
            f"MegaMoE kernels need SM100+, found SM{major}x"
        )
    if not op.apply_topk_in_fc1:
        raise BackendUnavailable("kernel supports apply_topk_in_fc1=True only")
    if op.hidden_size % 32 or op.intermediate_size % 32:
        raise BackendUnavailable(
            "hidden_size and intermediate_size must be multiples of 32"
        )
    if op.generate_c and (op.hidden_size % 128 or op.intermediate_size % 128):
        raise BackendUnavailable(
            "generate_c/backward path needs hidden_size and intermediate_size "
            "to be multiples of 128"
        )
    if op.combine_format not in _COMBINE_WIRE:
        raise BackendUnavailable(
            f"combine_format={op.combine_format.value!r} has no MXFP8-kernel "
            "wire format"
        )
    if op.generate_c and op.combine_format is not MoeFormat.BF16:
        raise BackendUnavailable(
            "generate_c/backward needs combine_format='bf16' (dtw reads the "
            "bf16 combine staging)"
        )
    if op.ep_group is None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise BackendUnavailable(
                "ep_group=None (one-rank execution) inside an initialized "
                "multi-rank process is not supported"
            )
    else:
        if dist.get_world_size(op.ep_group) != dist.get_world_size():
            raise BackendUnavailable(
                "ep_group must span the entire torch.distributed world "
                "(NVSHMEM is bootstrapped over the global world)"
            )


def maybe_create(op, device: torch.device, token_count: int):
    """Return a live backend for ``op`` or ``None`` (allocate-only fallback)."""

    policy = _policy()
    if policy == "none":
        return None
    try:
        _check_supported(op)
        return _MegamoeBackend(op, device, token_count)
    except Exception as exc:  # noqa: BLE001 - fold every failure into policy
        if policy == "megamoe":
            raise
        if not _state["warned"]:
            _state["warned"] = True
            warnings.warn(
                f"cudnn.MoeEp: megamoe backend unavailable ({exc}); "
                "returning uninitialized allocations (set "
                "CUDNN_MOE_EP_BACKEND=megamoe to make this an error)",
                RuntimeWarning,
                stacklevel=3,
            )
        return None


def _dequant_to_bf16(tensor: MoeTensor) -> torch.Tensor:
    if isinstance(tensor, BlockScaledTensor):
        return tensor.dequantize(torch.bfloat16)
    return tensor.to(torch.bfloat16)


def _weight_signature(tensor: MoeTensor) -> Tuple:
    data = tensor.data if isinstance(tensor, BlockScaledTensor) else tensor
    return (data.data_ptr(), data._version)


def _quantize_output(values: torch.Tensor, fmt: MoeFormat) -> BlockScaledTensor:
    """Quantize the kernel's bf16 (T, H) result along axis 1, matching the
    reference ``quantize_blockwise`` bit for bit."""

    logical_shape = tuple(values.shape)
    block = 32 if fmt is MoeFormat.MXFP8 else 16
    limit = 448.0 if fmt is MoeFormat.MXFP8 else 6.0
    blocks = values.float().reshape(values.shape[0], -1, block)

    scale_float = blocks.abs().amax(dim=-1) / limit
    if fmt is MoeFormat.MXFP8:
        safe = torch.where(scale_float > 0, scale_float, torch.ones_like(scale_float))
        scale_float = torch.where(
            scale_float > 0,
            torch.pow(2.0, torch.ceil(torch.log2(safe))),
            torch.zeros_like(scale_float),
        )
        scale = scale_float.to(torch.float8_e8m0fnu)
    else:
        scale = scale_float.to(torch.float8_e4m3fn)

    scale_math = scale.float()
    reciprocal = torch.where(scale_math > 0, scale_math.reciprocal(), torch.zeros_like(scale_math))
    normalized = (blocks * reciprocal.unsqueeze(-1)).clamp(-limit, limit)

    if fmt is MoeFormat.MXFP8:
        data = normalized.to(torch.float8_e4m3fn).reshape(logical_shape)
    else:
        codes = _nearest_e2m1_codes(normalized).reshape(logical_shape)
        data = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)

    return BlockScaledTensor(
        data=data.contiguous(),
        scale=scale.contiguous(),
        format=fmt,
        logical_shape=logical_shape,
        axis=-1,
    )


def _nearest_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """E2M1 nibble codes, round-to-nearest ties-to-even (reference copy)."""

    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=values.device,
    )
    magnitudes = values.abs().unsqueeze(-1)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    codes = torch.arange(8, dtype=torch.int64, device=values.device)
    any_code = torch.where(candidates, codes, 8).amin(dim=-1)
    even_code = torch.where(candidates & ((codes & 1) == 0), codes, 8).amin(dim=-1)
    magnitude_code = torch.where(even_code < 8, even_code, any_code)
    sign_code = torch.signbit(values).to(torch.int64) << 3
    return magnitude_code | sign_code


class _MegamoeBackend:
    """Owns one ``MegaMoeHybridMxfp8Layer`` for one ``MoeEp`` instance."""

    def __init__(self, op, device: torch.device, token_count: int) -> None:
        self._op = op
        self._device = device
        single_rank = op.ep_group is None
        self._mods = _import_megamoe(single_rank)
        if single_rank:
            _ensure_single_rank_dist()
        else:
            _ensure_nvshmem()

        max_tokens = op.max_tokens_per_rank
        if max_tokens is None:
            max_tokens = token_count
        if max_tokens <= 0:
            raise BackendUnavailable("cannot size kernel buffers for 0 tokens")
        self._max_tokens = max_tokens

        self._layer = None            # built on first forward (needs weights)
        self._weight_sig = None
        self._stash_serial = 0        # forward serial whose pools are live
        self._done_serial = 0

    # -- construction ---------------------------------------------------

    def _build_layer(self, w13: torch.Tensor, w2: torch.Tensor):
        op = self._op
        mods = self._mods
        ep_cfg = mods["EpConfig"](
            num_experts=op.num_experts,
            top_k=op.top_k,
            hidden_size=op.hidden_size,
            intermediate_size=op.intermediate_size,
            ep_size=op.ep_size,
            ep_rank=op.ep_rank,
            process_group=op.ep_group,
        )
        mm_cfg = mods["MegaMoeForwardConfig"](
            max_tokens_per_rank=self._max_tokens,
            hidden=op.hidden_size,
            intermediate=op.intermediate_size,
            num_total_experts=op.num_experts,
            num_topk=op.top_k,
            gate_up_clamp=op.gate_up_clamp,
            combine_format=_COMBINE_WIRE[op.combine_format],
        )
        mm_cfg.impl.generate_c = bool(op.generate_c)
        qcfg = mods["QuantConfig"](fprop_fmt="mxfp8", quant_bprop=True)
        self._layer = mods["MegaMoeHybridMxfp8Layer"](
            ep_cfg,
            mm_cfg,
            w13,
            w2,
            qcfg=qcfg,
            comm="torch_dist",
            bwd_impl="mega" if op.generate_c else "replay",
        )

    # -- layout adapters --------------------------------------------------

    def _to_kernel_weights(self, fc1_weight: MoeTensor, fc2_weight: MoeTensor):
        """(E,H,2I) gate-first / (E,I,H)  ->  w13 (E,2I,H) up-first / w2 (E,H,I)."""

        I = self._op.intermediate_size
        fc1 = _dequant_to_bf16(fc1_weight)
        gate = fc1[..., :I].transpose(1, 2)
        up = fc1[..., I:].transpose(1, 2)
        w13 = torch.cat([up, gate], dim=1).contiguous()
        w2 = _dequant_to_bf16(fc2_weight).transpose(1, 2).contiguous()
        return w13, w2

    def _sync_weights(self, fc1_weight: MoeTensor, fc2_weight: MoeTensor) -> None:
        sig = (_weight_signature(fc1_weight), _weight_signature(fc2_weight))
        if sig == self._weight_sig:
            return
        w13, w2 = self._to_kernel_weights(fc1_weight, fc2_weight)
        if self._layer is None:
            self._build_layer(w13, w2)
        else:
            with torch.no_grad():
                self._layer.w13.data.copy_(w13)
                self._layer.w2.data.copy_(w2)
            self._layer.refresh_weights()
        self._weight_sig = sig

    # -- forward ----------------------------------------------------------

    def forward(self, activation, fc1_weight, fc2_weight, topk_idx, topk_weights):
        op = self._op
        x = _dequant_to_bf16(activation)
        T = x.shape[0]
        if T > self._max_tokens:
            raise RuntimeError(
                f"token count {T} exceeds the backend capacity {self._max_tokens} "
                "(pass max_tokens_per_rank to MoeEp to size the kernel buffers)"
            )
        self._sync_weights(fc1_weight, fc2_weight)
        idx = topk_idx.to(torch.int64)
        tw = topk_weights.to(torch.float32)

        out_view = self._layer._fwd(x, idx, tw)
        self._stash_serial += 1

        if op.output_format is MoeFormat.BF16:
            output = out_view[:T].clone()
        else:
            output = _quantize_output(out_view[:T], op.output_format)
        if not op.generate_c:
            return output
        fc1_c, route_metadata = self._extract_stash(idx, T)
        return output, fc1_c, route_metadata

    def _extract_stash(self, topk_idx: torch.Tensor, T: int):
        """Kernel pools -> the logical (fc1_c, route_metadata) contract:
        rows grouped by local expert, sorted by (src_rank, src_token,
        src_slot); fc1_c de-interleaved from the kernel's 32-block (gate, up)
        pairs into plain [gate | up] halves."""

        op = self._op
        mods = self._mods
        fwd = self._layer._fwd
        device = topk_idx.device
        E_local, I, K = op.experts_per_rank, op.intermediate_size, op.top_k
        world = op.ep_size

        if world > 1:
            t_all = torch.full((world,), T, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(
                t_all,
                torch.tensor([T], dtype=torch.int64, device=device),
                group=op.ep_group,
            )
            if not bool((t_all == T).all().item()):
                raise RuntimeError(
                    "megamoe backend requires the same token count on every "
                    f"EP rank, got {t_all.tolist()}"
                )
            ids_all = torch.empty((world, T, K), dtype=topk_idx.dtype, device=device)
            dist.all_gather_into_tensor(
                ids_all, topk_idx.contiguous(), group=op.ep_group
            )
        else:
            ids_all = topk_idx.view(1, T, K)

        local = ids_all.reshape(-1) - op.ep_rank * E_local
        counts = torch.bincount(
            local[(local >= 0) & (local < E_local)], minlength=E_local
        )
        counts_list = counts.tolist()
        padded = [-(-n // 128) * 128 for n in counts_list]
        Mp = max(sum(padded), 128)
        doffs = [sum(padded[:i]) for i in range(E_local)]
        if sum(counts_list):
            valid = torch.cat(
                [
                    torch.arange(o, o + n, device=device)
                    for o, n in zip(doffs, counts_list)
                ]
            )
        else:
            valid = torch.empty(0, dtype=torch.long, device=device)

        lv = mods["local_pool_views"](fwd)
        src_rank, src_token, src_topk, _, _ = mods["decode_token_src_metadata"](
            lv["token_src_metadata"][:Mp]
        )
        sr = src_rank[valid].long()
        st = src_token[valid].long()
        sk = src_topk[valid].long()
        expert = torch.repeat_interleave(
            torch.arange(E_local, dtype=torch.long, device=device), counts
        )

        # contract row order: expert asc, then (src_rank, src_token, src_slot)
        key = ((expert * world + sr) * T + st) * K + sk
        order = torch.argsort(key)

        rows = fwd.fc1_c[:Mp][valid][order]
        n = rows.shape[0]
        pair = rows.view(n, I // 32, 2, 32)
        fc1_c = torch.cat(
            [pair[:, :, 0].reshape(n, I), pair[:, :, 1].reshape(n, I)], dim=-1
        ).contiguous()

        route_metadata = torch.stack(
            [expert[order], sr[order], st[order], sk[order]], dim=-1
        ).to(torch.int32)
        return fc1_c, route_metadata

    # -- backward ---------------------------------------------------------

    def backward(
        self,
        grad_output,
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
    ):
        op = self._op
        if self._layer is None or self._stash_serial == 0:
            raise RuntimeError("backward called before any forward on this MoeEp")
        if self._done_serial == self._stash_serial:
            raise RuntimeError(
                "backward already consumed this forward's stash; run another "
                "forward first"
            )
        self._sync_weights(fc1_weight, fc2_weight)

        layer = self._layer
        if layer._mega_bwd is None:
            from megamoe.bwd_kernel.backward import MegaMoeMxfp8Backward

            layer._mega_bwd = MegaMoeMxfp8Backward(layer._fwd, layer.ep_cfg)
        from megamoe.bwd_kernel.backward import mega_backward

        idx = topk_idx.to(torch.int64)
        tw = topk_weights.to(torch.float32)
        T = idx.shape[0]
        dx, dtw, dw13, dw2 = mega_backward(layer, idx, tw, grad_output, T)
        self._done_serial = self._stash_serial

        I = op.intermediate_size
        dw13 = dw13.float()
        grad_fc1 = torch.cat(
            [dw13[:, I:, :].transpose(1, 2), dw13[:, :I, :].transpose(1, 2)],
            dim=-1,
        ).contiguous()
        grad_fc2 = dw2.float().transpose(1, 2).contiguous()
        grad_tw = (dtw.float() * (idx != -1)).contiguous()
        return dx.float(), grad_fc1, grad_fc2, grad_tw
