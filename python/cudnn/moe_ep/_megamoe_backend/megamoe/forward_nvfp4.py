# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hackable single-call NVFP4 (4-bit) MegaMoE forward.

NVFP4 sibling of ``megamoe.forward.MegaMoeMxfp8Forward``, wrapping
``moe_nvfp4_swapab.megamoe_kernel.Sm100MegaMoEKernel`` (swap-AB): E2M1 data,
per-16 FP8-E4M3 block scales, plus a per-tensor global scale ("norm_const")
per operand.  Half the dispatch wire bytes of MXFP8.

Global-scale bookkeeping (see moe_nvfp4_swapab/mega_reference.py):

  quantized(x) satisfies  data * sf = x * norm_x   (norm_x = 2688 / amax(x))
  GEMM(a_q, b_q)          = (x . w) * norm_x * norm_w
  =>  fc1_alpha[e]        = 1 / (norm_act * norm_w13[e])
      fc1_norm_const[e]   = norm chosen for the in-kernel fc1-out requant
      fc2_alpha[e]        = 1 / (fc1_norm_const[e] * norm_w2[e])

``DataPreprocess`` (online mode) derives ``norm_act`` from the activation
amax on device and leaves it in a (1,) buffer; ``forward`` folds it into the
persistent ``fc1_alpha`` tensor with one on-stream torch op — no host sync.
Weight norms are per-expert, computed at ``load_weights``.
"""

from typing import Optional

import torch

import megamoe.repo_path  # noqa: F401

from common.megamoe_constants import (
    Fp8E4M3FNMax,
    Nvfp4BlockSize,
    Nvfp4E2M1Max,
    SfPaddingBlock,
)
from moe_nvfp4_swapab.runner_common import (
    ceil_div,
    round_up,
    to_blocked,
    nvfp4_quantize_per_block_16,
    _stack_byte_reinterpretable_tensors,
)
from moe_nvfp4_swapab.mega_runner import (
    _NO_DIST,
    _sym_zeros,
    _sym_zeros_byte_view,
    _compute_peer_offsets,
)
from src.token_comm import CombineFormat

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward

Nvfp4DataDtype = torch.float4_e2m1fn_x2
Nvfp4ScaleDtype = torch.float8_e4m3fn
_NORM_NUMER = float(Fp8E4M3FNMax * Nvfp4E2M1Max)  # 448 * 6
_GATE_UP_INTERLEAVE = 16  # kernel's fc1 gate/up column interleave (NVFP4)


class MegaMoeNvfp4Forward(MegaMoeMxfp8Forward):
    """4-bit dispatch + swap-AB fc12.  Reuses the mxfp8 class's compile/launch
    skeleton; overrides the quant staging, weight layout, and epilogue scalars.

    ``cfg.kind`` is ignored (NVFP4 is fixed); ``fc1_out_amax_estimate``
    calibrates the in-kernel fc1-out requant's global scale.
    """

    def __init__(self, cfg: MegaMoeForwardConfig, *, rank: int, world_size: int,
                 fc1_out_amax_estimate: float = 8.0) -> None:
        # Deliberately NOT calling super().__init__ — buffer dtypes/shapes
        # differ; replicate the skeleton with NVFP4 legs.
        if cfg.hidden % 32:
            raise ValueError("hidden must be a multiple of 32 (fp4 pack + TMA).")
        if cfg.intermediate % (2 * _GATE_UP_INTERLEAVE):
            raise ValueError("intermediate must be a multiple of 32.")
        if cfg.num_total_experts % world_size:
            raise ValueError("num_total_experts must divide by world_size.")
        # The config's default impl carries the MXFP8-validated geometry
        # (256x256x128, 2-CTA); swap in the NVFP4 swap-AB kernel's own
        # defaults unless the caller explicitly set an nvfp4-legal tile.
        from moe_nvfp4_swapab.runner_fc12_common import ImplDesc
        if cfg.impl.mma_tiler_mnk == (256, 256, 128):
            cfg = MegaMoeForwardConfig(
                **{**cfg.__dict__, "impl": ImplDesc(
                    load_balance_mode=cfg.impl.load_balance_mode,
                    in_kernel_fc2_reduce=cfg.impl.in_kernel_fc2_reduce,
                )},
            )
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_total_experts // world_size
        self._combine_format = CombineFormat.parse(cfg.combine_format)
        self._apply_topk_in_fc1 = True   # deepgemm graph
        self._compiled_mega = None
        self._compiled_preproc = None
        self._weights = None
        self._padding_valid_tokens = -1
        self.torch_ab_dtype = Nvfp4DataDtype
        self.fc1_out_norm_const = _NORM_NUMER / float(fc1_out_amax_estimate)

        T, H, K, E = (cfg.max_tokens_per_rank, cfg.hidden, cfg.num_topk,
                      self.num_local_experts)
        sf_cols = ceil_div(H, Nvfp4BlockSize)
        self._sf_cols_padded = round_up(sf_cols, 4)

        self.x_staging = torch.zeros((T, H), dtype=torch.bfloat16, device="cuda")
        self.topk_idx_in = torch.full((T, K), -1, dtype=torch.int64, device="cuda")
        self.topk_weights_in = torch.zeros((T, K), dtype=torch.float32, device="cuda")
        self.token_padding_info = torch.ones((T,), dtype=torch.int32, device="cuda")
        self.act_norm_const = torch.ones((1,), dtype=torch.float32, device="cuda")

        # sym-heap kernel inputs; fp4 packs 2/byte (storage (T, H/2)).
        self.my_activation = _sym_zeros_byte_view((T, H), Nvfp4DataDtype)
        self.my_activation_sf = _sym_zeros_byte_view(
            (T, self._sf_cols_padded), Nvfp4ScaleDtype
        )
        self.my_topk_idx = _sym_zeros((T, K), torch.int64)
        self.my_topk_weights = _sym_zeros((T, K), torch.float32)

        if cfg.impl.in_kernel_fc2_reduce:
            self.output_activation = _sym_zeros((T, H), torch.bfloat16)
        else:
            self.output_activation = torch.zeros(
                (T, H), dtype=torch.bfloat16, device="cuda"
            )

        # per-expert epilogue scalars (fc1_alpha is act-norm-dependent ->
        # rewritten on-stream each forward; the others are static per weights).
        self.fc1_alpha = torch.ones((E,), dtype=torch.float32, device="cuda")
        self.fc2_alpha = torch.ones((E,), dtype=torch.float32, device="cuda")
        self.fc1_norm_const = torch.full(
            (E,), self.fc1_out_norm_const, dtype=torch.float32, device="cuda"
        )
        self._fc1_alpha_base = torch.ones((E,), dtype=torch.float32, device="cuda")

        self.fc1_weight = None
        self.fc1_weight_sf = None
        self.fc2_weight = None
        self.fc2_weight_sf = None

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    def load_weights(self, w13: torch.Tensor, w2: torch.Tensor) -> None:
        """Quantize [E,2I,H] / [E,H,I] bf16 weights to NVFP4 kernel layout."""
        E, cfg = self.num_local_experts, self.cfg
        I = cfg.intermediate
        H = cfg.hidden
        if w13.shape != (E, 2 * I, H) or w2.shape != (E, H, I):
            raise ValueError(f"bad weight shapes: {tuple(w13.shape)}, {tuple(w2.shape)}")

        # gate/up 16-column interleave: [gate16 | up16 | ...] (gate slot 0).
        up = w13[:, :I].reshape(E, I // _GATE_UP_INTERLEAVE, _GATE_UP_INTERLEAVE, H)
        gate = w13[:, I:].reshape(E, I // _GATE_UP_INTERLEAVE, _GATE_UP_INTERLEAVE, H)
        fc1 = torch.stack((gate, up), dim=2).reshape(E, 2 * I, H).float().cuda()
        fc2 = w2.float().cuda()  # (E, H, I): quantize along K=I

        fc1_q, fc1_sf, fc2_q, fc2_sf = [], [], [], []
        fc1_norm_w = torch.empty((E,), dtype=torch.float32, device="cuda")
        fc2_norm_w = torch.empty((E,), dtype=torch.float32, device="cuda")
        for e in range(E):
            n1 = _NORM_NUMER / fc1[e].abs().amax().clamp(min=1e-12)
            n2 = _NORM_NUMER / fc2[e].abs().amax().clamp(min=1e-12)
            fc1_norm_w[e], fc2_norm_w[e] = n1, n2
            q1, s1 = nvfp4_quantize_per_block_16(fc1[e], float(n1.item()))
            q2, s2 = nvfp4_quantize_per_block_16(fc2[e], float(n2.item()))
            fc1_q.append(q1)   # (2I, H/2) fp4x2
            fc1_sf.append(s1)  # (2I, H/16) fp8
            fc2_q.append(q2)   # (H, I/2)
            fc2_sf.append(s2)  # (H, I/16)

        fc1_weight = _stack_byte_reinterpretable_tensors(fc1_q, dim=0)  # (E,2I,H/2)
        fc2_weight = _stack_byte_reinterpretable_tensors(fc2_q, dim=0)  # (E,H,I/2)
        fc1_sf_plain = _stack_byte_reinterpretable_tensors(fc1_sf, dim=0)
        fc2_sf_plain = _stack_byte_reinterpretable_tensors(fc2_sf, dim=0)
        fc1_sf_sw = _stack_byte_reinterpretable_tensors(
            [to_blocked(s) for s in fc1_sf], dim=0
        )
        fc2_sf_sw = _stack_byte_reinterpretable_tensors(
            [to_blocked(s) for s in fc2_sf], dim=0
        )

        if self.fc1_weight is None:
            # kernel layouts: fc1 (E, H_packed, 2I) K stride-1; fc2 (E, I_packed, H).
            self.fc1_weight = fc1_weight.permute(0, 2, 1)
            self.fc1_weight_sf = fc1_sf_sw
            self.fc2_weight = fc2_weight.permute(0, 2, 1)
            self.fc2_weight_sf = fc2_sf_sw
        else:
            self.fc1_weight.view(torch.uint8).copy_(
                fc1_weight.permute(0, 2, 1).view(torch.uint8))
            self.fc1_weight_sf.view(torch.uint8).copy_(fc1_sf_sw.view(torch.uint8))
            self.fc2_weight.view(torch.uint8).copy_(
                fc2_weight.permute(0, 2, 1).view(torch.uint8))
            self.fc2_weight_sf.view(torch.uint8).copy_(fc2_sf_sw.view(torch.uint8))

        # static epilogue scalars (act norm folded in per-forward).
        self._fc1_alpha_base.copy_(1.0 / fc1_norm_w)
        self.fc2_alpha.copy_(1.0 / (self.fc1_out_norm_const * fc2_norm_w))

        # keep plain SFs + norms for host references / debugging
        self._weights = dict(
            fc1_sf_plain=fc1_sf_plain, fc2_sf_plain=fc2_sf_plain,
            fc1_norm_w=fc1_norm_w, fc2_norm_w=fc2_norm_w,
        )

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.typing import AddressSpace

        from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel
        from moe_nvfp4_swapab.epilogue_refactor import SwapABSwigluFp4Epilogue
        from src.sym_buffer import SymBufferHost
        from src.inputs_process import DataPreprocess

        cfg, impl = self.cfg, self.cfg.impl
        if self._weights is None:
            raise RuntimeError("load_weights must be called before the first forward.")

        cluster_size = impl.cluster_shape_mnk[0] * impl.cluster_shape_mnk[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        group_hint = impl.group_hint if impl.group_hint is not None else max_active_clusters

        self._kernel = Sm100MegaMoEKernel(
            mma_tiler_mnk=impl.mma_tiler_mnk,
            cluster_shape_mnk=impl.cluster_shape_mnk,
            use_2cta_instrs=impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=SwapABSwigluFp4Epilogue._EpilogueTokenTileSize,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=impl.load_balance_mode,
            static_expert_shape=(
                self.num_local_experts, 2 * cfg.intermediate, cfg.hidden,
            ),
            force_static_sched=impl.force_static_sched,
            clc_bundle_size=impl.clc_bundle_size,
            num_sched_stages=impl.num_sched_stages,
            world_size=self.world_size,
            num_topk=cfg.num_topk,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
            hidden=cfg.hidden,
            fc2_output_dtype=cutlass.BFloat16,
            combine_format=self._combine_format,
            non_ubulk_fc2_store=impl.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=impl.in_kernel_fc2_reduce,
            token_back_mode=impl.token_back_mode,
            apply_topk_in_fc1=True,
            gate_up_clamp=cfg.gate_up_clamp,
            flag_batch=impl.flag_batch,
            epi_flag_batch=impl.epi_flag_batch,
        )

        local_ws_bytes, shared_ws_bytes = self._kernel.get_workspace_sizes()
        self.local_workspace = torch.zeros((local_ws_bytes,), dtype=torch.uint8, device="cuda")
        self.shared_workspace = _sym_zeros((shared_ws_bytes,), torch.uint8)
        symmetric_base, peer_offsets = _compute_peer_offsets(
            self.shared_workspace, self.world_size
        )
        peer_mapper = SymBufferHost(
            base_addr=symmetric_base, offsets=tuple(peer_offsets),
            rank_idx=self.rank, num_max_ranks=self.world_size,
        )

        def _ptr(tensor, assumed_align=16):
            return cute.runtime.make_ptr(
                cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._mega_kwargs = dict(
            activation=self._to_cute(self.my_activation),
            activation_sf=self._to_cute(self.my_activation_sf),
            topk_idx=self._to_cute(self.my_topk_idx),
            topk_weights=self._to_cute(self.my_topk_weights),
            fc1_weight=self._to_cute(self.fc1_weight),
            fc1_weight_sf=self._to_cute(self.fc1_weight_sf),
            fc2_weight=self._to_cute(self.fc2_weight),
            fc2_weight_sf=self._to_cute(self.fc2_weight_sf),
            fc1_alpha=self._to_cute(self.fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(self.fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(self.fc1_norm_const, assumed_align=4),
            output_activation=self._to_cute(self.output_activation),
            local_workspace=_ptr(self.local_workspace),
            shared_workspace=_ptr(self.shared_workspace),
            peer_rank_ptr_mapper_host=peer_mapper,
            stream=stream,
        )
        compile_kwargs = dict(self._mega_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        self._compiled_mega = cute.compile(self._kernel, **compile_kwargs)

        # Preprocess is split at the amax/quant boundary: the kernel has ONE
        # fc1_alpha per local expert but receives tokens from EVERY rank, so
        # the activation norm_const must be globally consistent — a 4-byte
        # NCCL MIN-reduce of the per-rank online norms runs between the two
        # stages (forward()).  The stages reuse DataPreprocess's own device
        # kernels unchanged.

        class _SplitNvfp4Preprocess(DataPreprocess):
            @cute.jit
            def amax_stage(self, activation_bf16, token_padding_info,
                           online_norm_const, cuda_stream):
                num_tokens = activation_bf16.shape[0]
                self._init_online_scale_impl(online_norm_const).launch(
                    grid=[1, 1, 1], block=[1, 1, 1], stream=cuda_stream,
                )
                self.nvfp4_amax_impl(
                    activation_bf16, token_padding_info, online_norm_const,
                ).launch(
                    grid=[num_tokens, 1, 1],
                    block=[self._amax_threads_per_cta, 1, 1],
                    stream=cuda_stream,
                )

            @cute.jit
            def quant_stage(self, activation_bf16, topk_idx, topk_weights,
                            token_padding_info, activation_quant, activation_sf,
                            topk_idx_output, topk_weights_output, norm_const,
                            cuda_stream):
                num_tokens = activation_bf16.shape[0]
                num_sf_blocks = activation_sf.shape[1]
                # same broadcast SF view rebuild as DataPreprocess.__call__
                activation_sf = cute.make_tensor(
                    activation_sf.iterator,
                    cute.make_layout(
                        (activation_sf.shape[0], (self.sf_vec, num_sf_blocks)),
                        stride=(activation_sf.stride[0], (0, activation_sf.stride[1])),
                    ),
                )
                self.nvfp4_quant_and_process_impl(
                    activation_bf16, topk_idx, topk_weights, token_padding_info,
                    activation_quant, activation_sf, topk_idx_output,
                    topk_weights_output, norm_const,
                ).launch(
                    grid=[num_tokens, 1, 1],
                    block=[self._threads_per_cta, 1, 1],
                    stream=cuda_stream,
                )

        self._preproc = _SplitNvfp4Preprocess(
            topk=cfg.num_topk, hidden=cfg.hidden, quant_type="nvfp4",
        )
        sf_cols = ceil_div(cfg.hidden, Nvfp4BlockSize)
        norm_cute = self._to_cute(self.act_norm_const, assumed_align=4)
        self._amax_kwargs = dict(
            activation_bf16=self._to_cute(self.x_staging),
            token_padding_info=self._to_cute(self.token_padding_info),
            online_norm_const=norm_cute,
            cuda_stream=stream,
        )
        self._quant_kwargs = dict(
            activation_bf16=self._to_cute(self.x_staging),
            topk_idx=self._to_cute(self.topk_idx_in),
            topk_weights=self._to_cute(self.topk_weights_in),
            token_padding_info=self._to_cute(self.token_padding_info),
            activation_quant=self._to_cute(self.my_activation),
            activation_sf=self._to_cute(self.my_activation_sf[:, :sf_cols]),
            topk_idx_output=self._to_cute(self.my_topk_idx),
            topk_weights_output=self._to_cute(self.my_topk_weights),
            norm_const=norm_cute,
            cuda_stream=stream,
        )
        self._compiled_amax = cute.compile(self._preproc.amax_stage, **self._amax_kwargs)
        self._compiled_quant = cute.compile(self._preproc.quant_stage, **self._quant_kwargs)
        self._compiled_preproc = True  # sentinel for the base-class None check

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x, topk_idx, topk_weights):
        if self._compiled_mega is None:
            self._compile()

        T = x.shape[0]
        Tmax = self.cfg.max_tokens_per_rank
        if T > Tmax:
            raise ValueError(f"got {T} tokens > max_tokens_per_rank ({Tmax}).")

        self._stage_input(x, T)
        self.topk_idx_in[:T].copy_(topk_idx)
        self.topk_weights_in[:T].copy_(topk_weights)
        if T != self._padding_valid_tokens:
            self.token_padding_info[:T].zero_()
            self.token_padding_info[T:].fill_(1)
            if T < Tmax:
                self.topk_idx_in[T:].fill_(-1)
                self.topk_weights_in[T:].zero_()
            self._padding_valid_tokens = T

        self.output_activation.zero_()
        # amax -> global norm (4-byte NCCL MIN, on-stream, no host sync) ->
        # quant with the GLOBAL norm (so every rank's sf bytes share one
        # scale convention) -> fold norm into fc1_alpha -> mega kernel.
        self._compiled_amax(**self._amax_kwargs)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                self.act_norm_const, op=torch.distributed.ReduceOp.MIN
            )
        self._compiled_quant(**self._quant_kwargs)
        # fc1_alpha[e] = (1/norm_w13[e]) / norm_act
        torch.div(self._fc1_alpha_base, self.act_norm_const, out=self.fc1_alpha)
        self._compiled_mega(**self._mega_kwargs)
        return self.output_activation[:T]

    __call__ = forward
