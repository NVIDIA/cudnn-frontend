# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid MoE-EP training layer: MegaMoE MXFP8 kernel fprop + torch bprop.

Step 1 of the bprop-megakernel plan: forward runs at (near-)inference speed
through :class:`megamoe.forward.MegaMoeMxfp8Forward`; backward is a pure
PyTorch *replay* of the quantized forward built from tensors REUSED FROM THE
KERNEL'S PERSISTENT POOLS — no stash copies:

- ``my_activation`` / ``my_activation_sf`` (the fused DataPreprocess quant
  output, sym-heap) are dequantized in backward and become the replay's
  activation. pt/quant.py's ``fake_quant_mxfp8`` was proven bit-exact vs the
  host quantizer (pt/tests/test_quant_vs_kernel.py), so re-quantizing the
  dequantized values inside the replay is idempotent — the replay consumes
  exactly the bytes the kernel multiplied.
- Weights are re-quantized in the replay from the bf16 masters with the same
  (conformance-tested) algorithm ``load_weights`` used, so no unswizzling of
  the kernel's weight buffers is needed.

The pool-reuse contract is enforced with a GENERATION COUNTER: backward
asserts no later ``forward`` has overwritten the pools (one fwd per bwd; for
gradient accumulation a stash ring is needed — deliberately out of scope
here). The kernel output view is cloned before autograd sees it (the buffer
is zeroed at the next forward).

Gradients (dX, d topk_weights, dW13, dW2) are the STE/quantized-bprop grads
of the replayed quantized function, produced by ``torch.autograd.grad``
through pt's differentiable dispatch/combine (NCCL all-to-all adjoints) and
``QuantGemmT``. This backward is the SLOW oracle baseline the future bprop
megakernel replaces stage by stage; profile it to see where the time goes.

Collective contract: every EP rank must call forward AND backward each step
(both contain all-to-alls). Not compatible with ``turboquant`` yet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import megamoe.repo_path  # noqa: F401

from dataclasses import replace as dc_replace

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward

from pt.comm import TokenComm, create_comm
from pt.config import EpConfig
from pt.dispatch_combine import combine, dispatch
from pt.experts_fp4 import grouped_expert_ffn_fp4
from pt.quant import MXFP8_BLOCK, QuantConfig, make_rotation, rotate_trailing
from pt.routing import build_routing_plan


def dequant_mxfp8_pool(
    vals_fp8: torch.Tensor, sf_e8m0: torch.Tensor, hidden: int
) -> torch.Tensor:
    """Dequantize a kernel pool tensor: fp8-e4m3 values + per-32 E8M0 scales
    (scale cols possibly padded). Returns fp32 ``[rows, hidden]``."""
    sf_cols = (hidden + MXFP8_BLOCK - 1) // MXFP8_BLOCK
    scale = torch.exp2(sf_e8m0[:, :sf_cols].view(torch.uint8).float() - 127.0)
    vals = vals_fp8[:, :hidden].float()
    return vals * scale.repeat_interleave(MXFP8_BLOCK, dim=-1)[:, :hidden]


class _HybridMoEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_weights, w13, w2, topk_ids, layer):
        out = layer._fwd(x.detach(), topk_ids, topk_weights.detach())
        layer._generation += 1
        ctx.layer = layer
        ctx.generation = layer._generation
        ctx.num_tokens = x.shape[0]
        # topk tensors are saved (tiny) — the pool copies may be repacked by
        # DataPreprocess, which would scramble the d(topk_weights) layout.
        ctx.save_for_backward(topk_ids, topk_weights)
        # The output is a view into the persistent buffer (zeroed next fwd).
        return out.clone()

    @staticmethod
    def backward(ctx, dout):
        layer = ctx.layer
        topk_ids, topk_weights = ctx.saved_tensors
        if layer._generation != ctx.generation:
            raise RuntimeError(
                "hybrid backward after a later forward overwrote the kernel "
                f"pools (saved generation {ctx.generation}, current "
                f"{layer._generation}); run backward before the next forward "
                "or add a stash ring for gradient accumulation."
            )
        T = ctx.num_tokens
        if layer.bwd_impl == "fp8":
            from megamoe.fp8_bwd import fp8_backward

            dx, dtw, dw13, dw2 = fp8_backward(layer, topk_ids, topk_weights, dout, T)
            return dx, dtw, dw13, dw2, None, None
        if layer.bwd_impl == "pool":
            from megamoe.bwd_v0 import pool_backward

            dx, dtw, dw13, dw2 = pool_backward(layer, topk_ids, topk_weights, dout, T)
            return dx, dtw, dw13, dw2, None, None
        if layer.bwd_impl == "mega":
            from megamoe.bwd_kernel.backward import (
                MegaMoeMxfp8Backward,
                mega_backward,
            )

            if layer._mega_bwd is None:
                layer._mega_bwd = MegaMoeMxfp8Backward(layer._fwd, layer.ep_cfg)
            dx, dtw, dw13, dw2 = mega_backward(layer, topk_ids, topk_weights, dout, T)
            return dx, dtw, dw13, dw2, None, None
        fwd = layer._fwd

        # Reuse the pool: the exact quantized activation the kernel consumed.
        x_q = dequant_mxfp8_pool(
            fwd.my_activation[:T], fwd.my_activation_sf[:T], layer.ep_cfg.hidden_size
        ).to(torch.bfloat16)

        tq = layer.qcfg.turboquant
        with torch.enable_grad():
            # With turboquant the pool holds the ROTATED quantized activation
            # (the mixin rotates at staging); replay in the rotated basis with
            # w13 rotated IN-GRAPH so autograd delivers dW13 in the master
            # basis, and rotate dX back through Q^T afterwards.
            xl = x_q.requires_grad_()
            twl = topk_weights.detach().clone().requires_grad_()
            w13l = layer.w13.detach().requires_grad_()
            w2l = layer.w2.detach().requires_grad_()
            plan = build_routing_plan(topk_ids, layer.ep_cfg, layer.comm)
            xg = dispatch(xl, plan, layer.comm)
            w13_eff = rotate_trailing(w13l, layer.q_rot) if tq else w13l
            yg = grouped_expert_ffn_fp4(
                xg, plan.tokens_per_expert, w13_eff, w2l,
                dc_replace(layer.qcfg, turboquant=False), None,
            )
            out = combine(yg, twl, plan, torch.bfloat16, layer.comm)
            dx, dtw, dw13, dw2 = torch.autograd.grad(
                out, (xl, twl, w13l, w2l), dout
            )
        if tq:
            dx = rotate_trailing(dx, layer.q_rot.t())
        return dx, dtw, dw13, dw2, None, None


class MegaMoeHybridMxfp8Layer(nn.Module):
    """Trainable EP MoE: MegaMoE MXFP8 mega-kernel forward, torch backward.

    ``w13`` ``[num_local_experts, 2*intermediate, hidden]`` bf16,
    ``w2`` ``[num_local_experts, hidden, intermediate]`` bf16 — this rank's
    expert shard, wrapped as master parameters. Call :meth:`refresh_weights`
    after each optimizer step to re-quantize the masters into the kernel's
    weight buffers (in-place, no recompile).
    """

    def __init__(
        self,
        ep_cfg: EpConfig,
        mm_cfg: MegaMoeForwardConfig,
        w13: torch.Tensor,
        w2: torch.Tensor,
        qcfg: QuantConfig | None = None,
        comm: str = "torch_dist",
        bwd_impl: str = "replay",
    ):
        """``bwd_impl``: "replay" (autograd fake-quant oracle), "fp8"
        (manual backward on real torch._scaled_grouped_mm fp8 GEMMs — same
        gradient semantics, ~bf16-rounding deviations, much faster), or
        "pool" (fp8 GEMMs fed straight from the kernel's persistent pools +
        generate_c stash via token_src_metadata — the megakernel dataflow;
        requires ``mm_cfg.impl.generate_c=True``)."""
        super().__init__()
        if bwd_impl not in ("replay", "fp8", "pool", "mega"):
            raise ValueError(
                f"bwd_impl must be 'replay', 'fp8', 'pool' or 'mega', got {bwd_impl}"
            )
        if bwd_impl in ("pool", "mega") and not mm_cfg.impl.generate_c:
            raise ValueError(
                f"bwd_impl={bwd_impl!r} needs mm_cfg.impl.generate_c=True"
            )
        if bwd_impl == "mega" and mm_cfg.combine_format != "bf16":
            raise ValueError("bwd_impl='mega' dtw path needs bf16 combine staging")
        self.bwd_impl = bwd_impl
        if (
            mm_cfg.hidden != ep_cfg.hidden_size
            or mm_cfg.intermediate != ep_cfg.intermediate_size
            or mm_cfg.num_total_experts != ep_cfg.num_experts
            or mm_cfg.num_topk != ep_cfg.top_k
        ):
            raise ValueError("ep_cfg and mm_cfg problem shapes disagree")
        self.ep_cfg = ep_cfg
        self.qcfg = qcfg or QuantConfig(fprop_fmt="mxfp8", quant_bprop=True)
        if self.qcfg.fprop_fmt != "mxfp8":
            raise ValueError("hybrid layer is mxfp8-fprop (kernel format)")
        self.w13 = w13 if isinstance(w13, nn.Parameter) else nn.Parameter(w13)
        self.w2 = w2 if isinstance(w2, nn.Parameter) else nn.Parameter(w2)
        self.comm: TokenComm = create_comm(comm, group=ep_cfg.process_group)
        if self.qcfg.turboquant:
            # TurboQuant training: masters stay UNROTATED. The kernel mixin
            # rotates activations at staging and folds Q into fc1 weights at
            # load_weights; backward rotates dX / dW13 back through Q^T.
            if ep_cfg.hidden_size % self.qcfg.rotation_block:
                raise ValueError(
                    f"hidden_size ({ep_cfg.hidden_size}) must be a multiple "
                    f"of rotation_block ({self.qcfg.rotation_block})"
                )
            from megamoe.turboquant import MegaMoeTurboQuantForward

            self._fwd = MegaMoeTurboQuantForward(
                mm_cfg, rank=ep_cfg.ep_rank, world_size=ep_cfg.ep_size,
                rotation_block=self.qcfg.rotation_block,
                rotation_seed=self.qcfg.rotation_seed,
            )
            self.register_buffer(
                "q_rot",
                make_rotation(
                    self.qcfg.rotation_block, self.qcfg.rotation_seed,
                    device=self.w13.device,
                ),
            )
        else:
            self._fwd = MegaMoeMxfp8Forward(
                mm_cfg, rank=ep_cfg.ep_rank, world_size=ep_cfg.ep_size
            )
            self.q_rot = None
        self._generation = 0
        self._bwd_wcache = None
        self._mega_bwd = None
        self.refresh_weights()

    def refresh_weights(self) -> None:
        """Quantize the bf16 masters into the kernel's persistent buffers
        (and invalidate the fp8 backward's cached weight quantizations)."""
        with torch.no_grad():
            self._fwd.load_weights(self.w13.data, self.w2.data)
        self._bwd_wcache = None
        self._pool_bwd_wcache = None
        self._mega_bwd_wdirty = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _HybridMoEFn.apply(
            hidden_states, topk_weights, self.w13, self.w2, topk_ids, self
        )

    def finalize(self) -> None:
        self._fwd.finalize()
