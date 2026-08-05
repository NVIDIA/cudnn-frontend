# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hackable single-call MegaMoE forward for a PyTorch pipeline.

A thin, flat re-package of ``moe_mxfp8_glu/mega_runner.py::run_kernel`` with
the tester scaffolding (input generation / reference / validation) removed:

  1. ctor          -- persistent sym-heap staging buffers, workspaces,
                      kernel + preprocess ``cute.compile`` (all one-time).
  2. load_weights  -- bf16 -> MXFP8 quant into persistent weight buffers
                      (repeatable; no recompile on weight updates).
  3. forward(x, topk_idx, topk_weights) -> (T, hidden) bf16
                      copy-in -> fused device quant (``DataPreprocess``)
                      -> one mega-kernel launch (dispatch + grouped FC1 GLU
                      + FC2 + combine).  Stream-ordered; no host sync.

All ranks must call ``forward`` collectively (the kernel contains NVLink
barriers).  Everything is deliberately kept in this one file so it is easy
to hack on (swap quant schemes, insert turboquant, add hooks, ...).

Env: ``MEGA_NO_DIST=1`` runs single-rank without torch.distributed/NVSHMEM
(sym buffers degrade to plain cuda tensors) -- handy for kernel debugging.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

import megamoe.repo_path  # noqa: F401  (sys.path side effect)

from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock
from moe_nvfp4_swapab.runner_common import ceil_div, round_up, Mxfp8ScaleDtype
from moe_nvfp4_swapab.mega_runner import (
    _NO_DIST,
    _sym_zeros,
    _compute_peer_offsets,
)
from moe_mxfp8_glu.mega_runner import (
    _KIND_TO_TORCH_DTYPE,
    _kind_to_cutlass_dtype,
    _sym_zeros_byte_view_1b,
)
from moe_mxfp8_glu.runner_common import TrainingImplDesc
from src.token_comm import CombineFormat

from megamoe.weights import quantize_moe_weights_mxfp8, QuantizedExpertWeights


@dataclass
class MegaMoeForwardConfig:
    """Problem shape + kernel knobs.

    ``intermediate`` is the per-GLU-branch FFN width I (pt/ convention:
    w13 is [E, 2*I, H]).  The kernel-side gate+up width is 2*I.

    ``impl`` carries the tunable kernel knobs (mma_tiler_mnk,
    cluster_shape_mnk, load_balance_mode, flag batching, ...).  Tune them
    with the repo's tester (``torchrun ... -m tester.tester --mode Perf
    --sweep --use_knob ...``) and paste the winner here.
    """

    max_tokens_per_rank: int
    hidden: int
    intermediate: int              # per-branch I; kernel sees 2*I
    num_total_experts: int
    num_topk: int
    kind: str = "mxfp8_e4m3"
    gate_up_clamp: Optional[float] = None
    combine_format: str = "bf16"   # "bf16" | "32e4m3xe8m0" | "32e5m2xe8m0"
    impl: TrainingImplDesc = field(
        default_factory=lambda: TrainingImplDesc(
            # validated MXFP8 tile/cluster; generate_c off until bwd needs it
            generate_c=False,
            token_back_mode="standalone_warps",
            epi_flag_batch=(2, 4),
            flag_batch=1,
            # 2026-07-12 sweep (fwd_20260712_bucket_sweep.csv): group_hint=160
            # beats the max_active_clusters default at EVERY token bucket
            # (-8..-16%, balanced AND power-law routing) at 256E/top8/H7168.
            group_hint=160,
        )
    )

    @property
    def intermediate_gateup(self) -> int:
        return 2 * self.intermediate


class MegaMoeMxfp8Forward:
    def __init__(
        self,
        cfg: MegaMoeForwardConfig,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        if cfg.hidden % Mxfp8BlockSize or cfg.intermediate % Mxfp8BlockSize:
            raise ValueError("hidden and intermediate must be multiples of 32.")
        if cfg.num_total_experts % world_size:
            raise ValueError("num_total_experts must divide by world_size.")
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_total_experts // world_size
        self.torch_ab_dtype = _KIND_TO_TORCH_DTYPE[cfg.kind]
        self._combine_format = CombineFormat.parse(cfg.combine_format)
        self._apply_topk_in_fc1 = True   # matches MegaMoEMxfp8Tester
        self._compiled_mega = None
        self._compiled_preproc = None
        self._weights: Optional[QuantizedExpertWeights] = None
        self._padding_valid_tokens = -1  # cached T of token_padding_info

        T, H, K = cfg.max_tokens_per_rank, cfg.hidden, cfg.num_topk
        sf_cols = ceil_div(H, Mxfp8BlockSize)
        self._sf_cols_padded = round_up(sf_cols, 4)  # LDG.32 wire format

        # -- persistent local staging (raw bf16 inputs land here each step) --
        self.x_staging = torch.zeros((T, H), dtype=torch.bfloat16, device="cuda")
        self.topk_idx_in = torch.full((T, K), -1, dtype=torch.int64, device="cuda")
        self.topk_weights_in = torch.zeros((T, K), dtype=torch.float32, device="cuda")
        self.token_padding_info = torch.ones((T,), dtype=torch.int32, device="cuda")

        # -- persistent sym-heap kernel inputs (peer-pulled by dispatch) --
        self.my_activation = _sym_zeros_byte_view_1b((T, H), self.torch_ab_dtype)
        self.my_activation_sf = _sym_zeros_byte_view_1b(
            (T, self._sf_cols_padded), Mxfp8ScaleDtype
        )
        self.my_topk_idx = _sym_zeros((T, K), torch.int64)
        self.my_topk_weights = _sym_zeros((T, K), torch.float32)

        # -- output: sym-heap iff it is the in-kernel REDG target --
        if cfg.impl.in_kernel_fc2_reduce or getattr(
            cfg.impl, "combine_in_flight_reduce", False
        ):
            self.output_activation = _sym_zeros((T, H), torch.bfloat16)
        else:
            self.output_activation = torch.zeros(
                (T, H), dtype=torch.bfloat16, device="cuda"
            )

        # -- optional fc1 stash for training bwd (generate_c): the kernel
        # streams the pre-SwiGLU gate+up fc1 output here from the epilogue.
        # Sized for worst-case arrivals + per-expert 128-row padding.
        if cfg.impl.generate_c:
            c_rows = world_size * T * K + self.num_local_experts * 128
            self.fc1_c = torch.zeros(
                (c_rows, cfg.intermediate_gateup), dtype=torch.bfloat16, device="cuda"
            )
        else:
            self.fc1_c = None

        # weight buffers allocated on first load_weights (contiguous storage;
        # the kernel-facing permuted views are built once there).
        self.fc1_weight = None      # storage (E, 2I, H) fp8; view (E, H, 2I)
        self.fc1_weight_sf = None   # (E, flat) swizzled E8M0
        self.fc2_weight = None      # storage (E, H, I) fp8; view (E, I, H)
        self.fc2_weight_sf = None

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    def load_weights(self, w13: torch.Tensor, w2: torch.Tensor) -> None:
        """Quantize local-expert bf16 weights into the persistent buffers.

        w13 [E_local, 2*I, H] ([:I]=linear/up, [I:]=gate), w2 [E_local, H, I].
        Call again anytime the master weights change (e.g. every optimizer
        step) -- buffers are updated in place, no recompile.
        """
        E, cfg = self.num_local_experts, self.cfg
        if w13.shape != (E, cfg.intermediate_gateup, cfg.hidden):
            raise ValueError(
                f"w13 must be {(E, cfg.intermediate_gateup, cfg.hidden)}, "
                f"got {tuple(w13.shape)}."
            )
        q = quantize_moe_weights_mxfp8(w13, w2, kind=cfg.kind)
        if self.fc1_weight is None:
            self.fc1_weight = q.fc1_weight
            self.fc1_weight_sf = q.fc1_weight_sf
            self.fc2_weight = q.fc2_weight
            self.fc2_weight_sf = q.fc2_weight_sf
        else:  # in-place update, keep pointers baked into runtime kwargs valid
            self.fc1_weight.copy_(q.fc1_weight)
            self.fc1_weight_sf.view(torch.uint8).copy_(q.fc1_weight_sf.view(torch.uint8))
            self.fc2_weight.copy_(q.fc2_weight)
            self.fc2_weight_sf.view(torch.uint8).copy_(q.fc2_weight_sf.view(torch.uint8))
        self._weights = q

    # ------------------------------------------------------------------
    # compile (lazy, one-time)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cute(tensor: torch.Tensor, assumed_align: int = 16, force_static=False):
        import cutlass.torch as cutlass_torch

        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        if force_static:
            return cute_tensor
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    def _compile(self) -> None:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.utils as utils

        from moe_nvfp4_swapab.epilogue import EpilogueTokenTile
        from moe_mxfp8_glu.megamoe_kernel_mxfp8 import Sm100MegaMoEMxfp8Kernel
        from src.sym_buffer import SymBufferHost
        from src.inputs_process import DataPreprocess

        cfg, impl = self.cfg, self.cfg.impl
        if self._weights is None:
            raise RuntimeError("load_weights must be called before the first forward.")

        cluster_size = impl.cluster_shape_mnk[0] * impl.cluster_shape_mnk[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        group_hint = impl.group_hint if impl.group_hint is not None else max_active_clusters

        self._kernel = Sm100MegaMoEMxfp8Kernel(
            mma_tiler_mnk=impl.mma_tiler_mnk,
            cluster_shape_mnk=impl.cluster_shape_mnk,
            use_2cta_instrs=impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=128 if impl.generate_c else EpilogueTokenTile,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=impl.load_balance_mode,
            static_expert_shape=(
                self.num_local_experts, cfg.intermediate_gateup, cfg.hidden,
            ),
            force_static_sched=impl.force_static_sched,
            clc_bundle_size=impl.clc_bundle_size,
            num_sched_stages=impl.num_sched_stages,
            ab_dtype=_kind_to_cutlass_dtype(cfg.kind),
            sf_vec_size=Mxfp8BlockSize,
            world_size=self.world_size,
            local_rank=self.rank,
            num_topk=cfg.num_topk,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
            hidden=cfg.hidden,
            fc2_in_kernel_topk_reduce=impl.in_kernel_fc2_reduce,
            token_back_mode=impl.token_back_mode,
            epi_flag_batch=impl.epi_flag_batch,
            flag_batch=impl.flag_batch,
            gate_up_clamp=cfg.gate_up_clamp,
            apply_topk_in_fc1=self._apply_topk_in_fc1,
            generate_c=impl.generate_c,
            use_stg_fc1=impl.use_stg_fc1,
            combine_format=self._combine_format,
            dedup_dispatch=getattr(impl, "dedup_dispatch", False),
            combine_in_flight_reduce=getattr(
                impl, "combine_in_flight_reduce", False
            ),
            combine_pre_reduce=getattr(impl, "combine_pre_reduce", False),
        )

        # workspaces: local plain cuda, shared on the sym heap (its peer-delta
        # table covers every sym sub-allocation in the heap).
        local_ws_bytes, shared_ws_bytes = self._kernel.get_workspace_sizes()
        self.local_workspace = torch.zeros((local_ws_bytes,), dtype=torch.uint8, device="cuda")
        self.shared_workspace = _sym_zeros((shared_ws_bytes,), torch.uint8)
        symmetric_base, peer_offsets = _compute_peer_offsets(
            self.shared_workspace, self.world_size
        )
        peer_mapper = SymBufferHost(
            base_addr=symmetric_base,
            offsets=tuple(peer_offsets),
            rank_idx=self.rank,
            num_max_ranks=self.world_size,
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- mega kernel: runtime kwargs are baked to the persistent buffers --
        self._mega_kwargs = dict(
            activation=self._to_cute(self.my_activation),
            activation_sf=self._to_cute(self.my_activation_sf),
            topk_idx=self._to_cute(self.my_topk_idx),
            topk_weights=self._to_cute(self.my_topk_weights),
            fc1_weight=self._to_cute(self.fc1_weight),
            fc1_weight_sf=self._to_cute(self.fc1_weight_sf),
            fc2_weight=self._to_cute(self.fc2_weight),
            fc2_weight_sf=self._to_cute(self.fc2_weight_sf),
            output_activation=self._to_cute(self.output_activation),
            local_workspace=self._to_cute(self.local_workspace, force_static=True),
            shared_workspace=self._to_cute(self.shared_workspace),
            peer_rank_ptr_mapper_host=peer_mapper,
            fc1_c=self._to_cute(self.fc1_c) if self.fc1_c is not None else None,
            stream=stream,
        )
        compile_kwargs = dict(self._mega_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        self._compiled_mega = cute.compile(self._kernel, **compile_kwargs)

        # -- fused bf16 -> mxfp8 staging kernel (quant + routing repack) --
        self._preproc = DataPreprocess(
            topk=cfg.num_topk, hidden=cfg.hidden, quant_type=cfg.kind,
        )
        sf_cols = ceil_div(cfg.hidden, Mxfp8BlockSize)
        self._preproc_kwargs = dict(
            activation_bf16=self._to_cute(self.x_staging),
            topk_idx=self._to_cute(self.topk_idx_in),
            topk_weights=self._to_cute(self.topk_weights_in),
            token_padding_info=self._to_cute(self.token_padding_info),
            activation_quant=self._to_cute(self.my_activation),
            # unpadded (T, H//32) view; padded tail cols stay zero from alloc
            activation_sf=self._to_cute(self.my_activation_sf[:, :sf_cols]),
            topk_idx_output=self._to_cute(self.my_topk_idx),
            topk_weights_output=self._to_cute(self.my_topk_weights),
            cuda_stream=stream,
        )
        self._compiled_preproc = cute.compile(self._preproc, **self._preproc_kwargs)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _stage_input(self, x: torch.Tensor, T: int) -> None:
        """Land the raw bf16 activation in the staging buffer (overridable —
        the turboquant mixin fuses its rotation into this write)."""
        self.x_staging[:T].copy_(x)

    def forward(
        self,
        x: torch.Tensor,             # (T, hidden) bf16, T <= max_tokens_per_rank
        topk_idx: torch.Tensor,      # (T, num_topk) int, global expert ids
        topk_weights: torch.Tensor,  # (T, num_topk) float
    ) -> torch.Tensor:
        """One expert-parallel MoE forward.  Collective: every rank must call.

        Returns the (T, hidden) bf16 combined output (a view into the
        persistent output buffer -- clone it if you need it past the next
        forward).
        """
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
            if T < Tmax:  # belt & braces: mask staged routing of pad rows too
                self.topk_idx_in[T:].fill_(-1)
                self.topk_weights_in[T:].zero_()
            self._padding_valid_tokens = T

        # REDG accumulates into output (in_kernel_reduce) and padding rows are
        # never written -- start from zeros every step.
        self.output_activation.zero_()

        self._compiled_preproc(**self._preproc_kwargs)
        self._compiled_mega(**self._mega_kwargs)
        return self.output_activation[:T]

    __call__ = forward

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Free sym-heap tensors (call before nvshmem finalize)."""
        self._compiled_mega = None
        self._compiled_preproc = None
        self._kernel = None
        if _NO_DIST:
            return
        import nvshmem.core

        sym = [
            self.my_activation, self.my_activation_sf,
            self.my_topk_idx, self.my_topk_weights,
            getattr(self, "shared_workspace", None),
        ]
        if self.cfg.impl.in_kernel_fc2_reduce or getattr(
            self.cfg.impl, "combine_in_flight_reduce", False
        ):
            sym.append(self.output_activation)
        for t in sym:
            if t is not None:
                try:
                    nvshmem.core.free_tensor(t)
                except Exception:  # noqa: BLE001
                    pass
