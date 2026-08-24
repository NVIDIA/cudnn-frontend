# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Full MegaMoE (multi-rank) mxfp8 dGLU training-backward kernel."""

from types import SimpleNamespace
from typing import Any, Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from ......api import ImplDesc, KernelClass, ProblemDesc, StaticOrRuntimeIntegerType
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.smem_workspace import SmemWorkspace
from ......helpers.utils import ceil_div, round_up
from ......quant_def import CombineFormat, QuantKind
from ......communication.nvlink_domain.token_comm_deterministic import TokenCommDeterministic
from ..topk_reduce import TopkReduce
from ..fwd_glu.glu_mxfp8_col_requant import Mxfp8ColRequant
from .dglu_mxfp8_fc12_kernel import Sm107Mxfp8DgluDfc21Kernel


_AB_DTYPE_TO_QUANT_KIND = {cutlass.Float8E4M3FN: QuantKind.mxfp8_e4m3, cutlass.Float8E5M2: QuantKind.mxfp8_e5m2}
_QUANT_KIND_TO_AB_DTYPE = {str(k): d for d, k in _AB_DTYPE_TO_QUANT_KIND.items()}

# TVM-FFI export symbol for the AOT-compiled callable.
_aot_symbol_prefix = "rubin_mega_moe_dglu_mxfp8_aot"


class Sm107MegaMoEMxfp8DgluKernel(Sm107Mxfp8DgluDfc21Kernel, KernelClass):
    """Multi-rank MegaMoE wrapper around the lean mxfp8 dGLU kernel."""

    # grad_y1 (dfc2 output), its SF, cross-phase counter, and internal dGLU pools.
    fc1_output_region = "rubin.dglu_mxfp8.mega.fc1_output"
    fc1_output_sf_region = "rubin.dglu_mxfp8.mega.fc1_output_sf"
    fc1_done_counter_region = "rubin.dglu_mxfp8.mega.fc1_done_counter"
    load_balance_counter_region = "rubin.dglu_mxfp8.mega.load_balance_counter"
    sched_work_id_region = "rubin.dglu_mxfp8.mega.sched_work_id"
    # Host-side local mirror of the token_comm shared token_src_metadata, so the
    # legacy dfc2_recompute / dfc2_col_output validation (which reads it from the
    # LOCAL workspace) works with the next's shared-heap metadata layout.
    token_src_metadata_local_region = "rubin.dglu_mxfp8.mega.token_src_metadata_local"
    fc1_preact_region = "rubin.dglu_mxfp8.mega.fc1_preact"
    grad_y2_sizes_region = "rubin.dglu_mxfp8.mega.grad_y2_expert_token_sizes"

    # Reserved on top of the exact token_comm/sched SMEM to cover smem.allocate
    # inter-allocation alignment padding that _compute_stages does not model.
    _SMEM_ALLOC_MARGIN = 2048

    @classmethod
    def problem_desc_require(cls):
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "world_size": int,
            "local_rank": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "max_recv_size_per_rank": int,
            "gate_up_clamp": Optional[float],
        }

    @classmethod
    def impl_desc_require(cls):
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mnk": tuple,
            "use_2cta_instrs": bool,
            "group_hint": int,
            "token_padding_block": int,
            "sf_padding_block": int,
            "load_balance_mode": str,
            "force_static_sched": bool,
            "clc_bundle_size": Optional[int],
            "num_sched_stages": Optional[int],
            "acc_dtype": type,
            "sf_vec_size": int,
            "launch_cluster_count": int,
            "drop_on_overflow": bool,
            "fc2_in_kernel_topk_reduce": bool,
            "token_back_mode": str,
            "epi_flag_batch": tuple,
            "flag_batch": int,
            "act_func": str,
            "dfc2_recompute": bool,
            "dfc2_col_output": bool,
            "enable_grad_y2_col_quant": bool,
            "num_ctas_grad_y2_col_quant": int,
        }

    def name(self) -> str:
        return (
            f"sm107_megamoe_dglu_{self.quant_kind}_m{self.mma_tiler_mnk[0]}n{self.mma_tiler_mnk[1]}"
            f"k{self.mma_tiler_mnk[2]}_e{self.expert_count}_ep{self.world_size}_topk{self.topk}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_combine{self.combine_format}_"
            f"clamp{self.gate_up_clamp}_"
            f"tokenback{self.token_back_mode}_hint{self.group_hint}_"
            f"epi{self.epi_flag_batch[0]}x{self.epi_flag_batch[1]}_tif{self.flag_batch}_"
            f"deterministic_mtpr{self.max_tokens_per_rank}_mrpr{self.max_recv_size_per_rank}_"
            f"drop{int(self.drop_on_overflow)}_lc{self.launch_cluster_count}_"
            f"recompute{int(self.dfc2_recompute)}x{int(self.dfc2_col_output)}_"
            f"redtopk{int(self.reduce_topk_in_kernel)}_preactarg1"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable."""
        import math

        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64
        from cutlass.cutlass_dsl import Int32, Int64

        from ......communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost

        def fake_tensor(dtype, shape, stride_order, dynamic_axes, alignment):
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128)) if axis in dynamic_axes else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(dtype, extents, stride_order=stride_order, assumed_align=alignment)

        tokens = self.max_tokens_per_rank
        hidden = self.hidden
        inter_half = self.intermediate_downproj  # dfc2 weight N
        gate_up = self.intermediate_gateup  # grad_y1 width / dfc1 K = 2 * inter_half
        experts = self.num_experts_per_rank
        vec = self.sf_vec_size
        activation_dtype = self.token_comm.activation_dtype  # grad_out fp8
        sf_dtype = self.token_comm.activation_sf_dtype  # E8M0
        # Atom-swizzled weight SF extents (to_blocked pads rows->128, cols->4).
        fc1_weight_sf_columns = round_up(inter_half, 128) * round_up(hidden // vec, 4)
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(gate_up // vec, 4)
        aux_shapes = self.get_aux_output_shapes()

        fake_arguments = dict(
            grad_out=fake_tensor(activation_dtype, (tokens, hidden), (1, 0), {0}, 16),
            grad_out_sf=fake_tensor(sf_dtype, (tokens, self.token_comm.activation_sf_hidden_padded), (1, 0), {0}, 16),
            topk_idx=fake_tensor(cutlass.Int64, (tokens, self.num_topk), (1, 0), {0}, 16),
            topk_weights=fake_tensor(cutlass.Float32, (tokens, self.num_topk), (1, 0), {0}, 4),
            fc1_weight=fake_tensor(self.ab_dtype, (experts, hidden, inter_half), (2, 0, 1), {0, 2}, 16),
            fc1_weight_sf=fake_tensor(sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {0}, 16),
            fc2_weight=fake_tensor(self.ab_dtype, (experts, gate_up, hidden), (2, 0, 1), {0, 2}, 16),
            fc2_weight_sf=fake_tensor(sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {0}, 16),
            beta=fake_tensor(cutlass.Float32, (experts,), (0,), {0}, 4),
            fc1_preact=fake_tensor(cutlass.BFloat16, self.get_fc1_preact_shape(), (1, 0), set(), 128),
            output_activation=fake_tensor(cutlass.BFloat16, (tokens, hidden), (1, 0), {0}, 16),
            overflow_flag=fake_tensor(cutlass.Int32, (1,), (0,), set(), 4),
            dprob=fake_tensor(cutlass.Float32, aux_shapes["dprob"], (1, 0), {0}, 16),
            fc1_recompute=fake_tensor(self.ab_dtype, aux_shapes["fc1_recompute"], (1, 0), set(), 128),
            fc1_recompute_sf=fake_tensor(sf_dtype, aux_shapes["fc1_recompute_sf"], (1, 0), set(), 128),
            fc1_col_output=fake_tensor(self.ab_dtype, aux_shapes["fc1_col_output"], (1, 0), set(), 128),
            fc1_col_output_sf=fake_tensor(sf_dtype, aux_shapes["fc1_col_output_sf"], (1, 0), set(), 128),
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            stream=make_fake_stream(),
        )
        fake_arguments["grad_y2"] = fake_tensor(
            self.ab_dtype, aux_shapes["grad_y2"], (1, 0), {0}, 16
        )
        fake_arguments["grad_y2_sf"] = fake_tensor(
            cutlass.Uint8, aux_shapes["grad_y2_sf"], (0,), set(), 16
        )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True)
        return out_path

    @staticmethod
    def load_compiled(path: str):
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    @classmethod
    def from_kwargs(
        cls,
        # Base-class (lean dfc2+dfc1) kwargs.
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        sf_vec_size: int = 32,
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        max_recv_size_per_rank: int,
        hidden: int,
        launch_cluster_count: int,
        drop_on_overflow: bool,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_mode: Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] = "epi_warps",
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        flag_batch: int = 1,
        combine_format: Optional[CombineFormat] = None,
        act_func: str = "swiglu",
        gate_up_clamp: Optional[float] = None,
        dfc2_recompute: bool = False,
        dfc2_col_output: bool = False,
        enable_grad_y2_col_quant: bool = False,
        num_ctas_grad_y2_col_quant: int = 2368,
    ) -> "Sm107MegaMoEMxfp8DgluKernel":
        """Build the ``(ProblemDesc, ImplDesc)`` pair from the legacy flat signature."""
        if static_expert_shape is None:
            raise NotImplementedError("Sm107MegaMoEMxfp8DgluKernel requires a static_expert_shape.")
        if hidden != static_expert_shape[2]:
            raise ValueError(f"hidden ({hidden}) must equal static_expert_shape[2] ({static_expert_shape[2]}).")
        if ab_dtype not in _AB_DTYPE_TO_QUANT_KIND:
            raise ValueError(f"ab_dtype {ab_dtype} has no mxfp8 QuantKind.")
        num_experts_per_rank, intermediate_gateup, _hidden = static_expert_shape
        combine_format = CombineFormat.parse("bf16" if combine_format is None else str(combine_format))
        problem_desc = ProblemDesc(
            {
                "expert_count": world_size * num_experts_per_rank,
                "intermediate_gateup_size": intermediate_gateup,
                "hidden_size": hidden,
                "quant_kind": str(_AB_DTYPE_TO_QUANT_KIND[ab_dtype]),
                "combine_format": combine_format,
                "world_size": world_size,
                "local_rank": local_rank,
                "topk": num_topk,
                "max_tokens_per_rank": max_tokens_per_rank,
                "max_recv_size_per_rank": max_recv_size_per_rank,
                "gate_up_clamp": gate_up_clamp,
            }
        )
        impl_desc = ImplDesc(
            {
                "mma_tiler_mnk": tuple(mma_tiler_mnk),
                "cluster_shape_mnk": tuple(cluster_shape_mnk),
                "use_2cta_instrs": use_2cta_instrs,
                "group_hint": group_hint,
                "token_padding_block": token_padding_block,
                "sf_padding_block": sf_padding_block,
                "load_balance_mode": load_balance_mode,
                "force_static_sched": force_static_sched,
                "clc_bundle_size": clc_bundle_size,
                "num_sched_stages": num_sched_stages,
                "acc_dtype": acc_dtype,
                "sf_vec_size": sf_vec_size,
                "launch_cluster_count": launch_cluster_count,
                "drop_on_overflow": drop_on_overflow,
                "fc2_in_kernel_topk_reduce": fc2_in_kernel_topk_reduce,
                "token_back_mode": token_back_mode,
                "epi_flag_batch": tuple(epi_flag_batch) if epi_flag_batch is not None else (1, 1),
                "flag_batch": flag_batch,
                "act_func": act_func,
                "dfc2_recompute": dfc2_recompute,
                "dfc2_col_output": dfc2_col_output,
                "enable_grad_y2_col_quant": enable_grad_y2_col_quant,
                "num_ctas_grad_y2_col_quant": num_ctas_grad_y2_col_quant,
            }
        )
        return cls(problem_desc, impl_desc)

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        # -- Extract descriptors into locals matching the legacy param names. --
        world_size = problem_desc["world_size"]
        local_rank = problem_desc["local_rank"]
        num_topk = problem_desc["topk"]
        max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        max_recv_size_per_rank = min(
            problem_desc["max_recv_size_per_rank"], world_size * max_tokens_per_rank * num_topk
        )
        hidden = problem_desc["hidden_size"]
        combine_format = problem_desc["combine_format"]
        gate_up_clamp = problem_desc["gate_up_clamp"]
        _quant_kind = problem_desc["quant_kind"]
        ab_dtype = _QUANT_KIND_TO_AB_DTYPE[_quant_kind]
        static_expert_shape = (
            problem_desc["expert_count"] // world_size,
            problem_desc["intermediate_gateup_size"],
            hidden,
        )

        mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        cluster_shape_mnk = impl_desc["cluster_shape_mnk"]
        use_2cta_instrs = impl_desc["use_2cta_instrs"]
        group_hint = impl_desc["group_hint"]
        token_padding_block = impl_desc["token_padding_block"]
        sf_padding_block = impl_desc["sf_padding_block"]
        load_balance_mode = impl_desc["load_balance_mode"]
        force_static_sched = impl_desc["force_static_sched"]
        clc_bundle_size = impl_desc["clc_bundle_size"]
        num_sched_stages = impl_desc["num_sched_stages"]
        acc_dtype = impl_desc["acc_dtype"]
        sf_vec_size = impl_desc["sf_vec_size"]
        launch_cluster_count = impl_desc["launch_cluster_count"]
        drop_on_overflow = impl_desc["drop_on_overflow"]
        fc2_in_kernel_topk_reduce = impl_desc["fc2_in_kernel_topk_reduce"]
        token_back_mode = impl_desc["token_back_mode"]
        epi_flag_batch = impl_desc["epi_flag_batch"]
        flag_batch = impl_desc["flag_batch"]
        act_func = impl_desc["act_func"]
        dfc2_recompute = impl_desc["dfc2_recompute"]
        dfc2_col_output = impl_desc["dfc2_col_output"]
        self.enable_grad_y2_col_quant = impl_desc["enable_grad_y2_col_quant"]
        self.num_ctas_grad_y2_col_quant = impl_desc["num_ctas_grad_y2_col_quant"]

        if hidden != static_expert_shape[2]:
            raise ValueError(f"hidden ({hidden}) must equal static_expert_shape[2] ({static_expert_shape[2]}).")
        token_back_by_dispatch = token_back_mode != "epi_warps"
        combine_format = CombineFormat.parse("bf16" if combine_format is None else str(combine_format))
        # in-kernel topk reduce only conflicts with a QUANTIZED combine (no per-topk
        # reduced-plane accumulation for quantized).  It DOES work with the
        # standalone/reuse_dispatch token-back modes (the token_comm token_back path
        # honours token_back_reduce_topk), so those are allowed (mirrors the legacy).
        if fc2_in_kernel_topk_reduce and combine_format.is_quantized:
            raise ValueError("fc2_in_kernel_topk_reduce requires a non-quantized (bf16) combine.")
        if token_back_mode not in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
            raise ValueError(f"unsupported token_back_mode={token_back_mode!r}.")
        if ab_dtype not in _AB_DTYPE_TO_QUANT_KIND:
            raise ValueError(f"ab_dtype {ab_dtype} has no mxfp8 QuantKind.")

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            ab_dtype=ab_dtype,
            sf_vec_size=sf_vec_size,
            epi_flag_batch=epi_flag_batch,
            dfc2_recompute=dfc2_recompute,
            dfc2_col_output=dfc2_col_output,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            act_func=act_func,
            gate_up_clamp=gate_up_clamp,
        )

        # --- Warp topology (realigned for next's TokenCommDeterministic). ---
        # next derives each transfer warp's transfer index as ``thread_idx % 128``
        # (token_comm.py:1421-1424 / 1911), which HARD-REQUIRES the dispatch warps --
        # and, standalone, the token_back warps -- to begin on a 4-warp / 128-thread
        # boundary.  So dispatch sits at warps 8-11 (thread 256 -> 256%128=0) and
        # standalone token_back at 12-15 (thread 384 -> 384%128=0), mirroring the
        # forward GLU.  The dGLU c_load warp does NOT use the transfer index, so it
        # moves ABOVE the transfer block (warp 16 iff standalone else 12), overriding
        # the base kernel's default (warp 8, now occupied by dispatch).
        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        self.token_back_mode = token_back_mode
        # Thread token_back_by_dispatch to the FC12 base (it hardcodes False) so the
        # epilogue, built later in _setup_attributes(), fires the fc2_done counter for
        # the standalone / reuse_dispatch token-back warps.  Without this the dedicated
        # token-back warps spin forever on fc2_done < target and the block-wide
        # sync_threads() in kernel_tail deadlocks (M09/M10/M14/M15 hang).  epi_warps is
        # unaffected (it peer-writes grad_x directly and never reads fc2_done).
        self.token_back_by_dispatch = token_back_by_dispatch
        self.token_back_standalone = token_back_by_dispatch and token_back_mode == "standalone_warps"
        self.token_back_warp_id = (12, 13, 14, 15) if self.token_back_standalone else None
        num_token_back_warps = len(self.token_back_warp_id) if self.token_back_standalone else 0
        self.c_load_warp_id = 16 if self.token_back_standalone else 12

        # Register re-balance for the mega warp layout.  The base kernel sizes
        # ``epi_reg_cnt`` (256) for the lean 9-warp dGLU; mega adds the 4 dispatch
        # warps (+4 token-back if standalone) and the dedicated c_load warp, so the
        # per-CTA register file can no longer grant 256 regs to all 4 epilogue warps
        # -- the epilogue warpgroup then stalls forever inside
        # ``warpgroup_reg_alloc`` and the mma/tmem barrier deadlocks.  Mirror the
        # legacy mega dGLU (megamoe_kernel_mxfp8_dglu.py:181-184).
        self.epi_reg_cnt = 168 if self.token_back_standalone else 200
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id)  # 4  (warps 0-3)
            + 1  # mma      (warp 4)
            + 1  # tma_a    (warp 5)
            + 1  # tma_b    (warp 6)
            + 1  # sched    (warp 7)
            + len(self.dispatch_warp_id)  # 4  (warps 8-11)
            + num_token_back_warps  # 4 iff standalone_warps (warps 12-15)
            + 1  # c_load (warp 12 or 16, dGLU-specific)
        )

        # --- MegaMoE constants. ---
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.max_recv_size_per_rank = max_recv_size_per_rank
        self.hidden = hidden
        self.launch_cluster_count = launch_cluster_count
        self.drop_on_overflow = drop_on_overflow
        self.combine_format = combine_format
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_downproj = static_expert_shape[1]
        self.intermediate_gateup = self.intermediate_downproj * 2
        self.num_total_experts = world_size * self.num_experts_per_rank
        self.reduce_topk_in_kernel = fc2_in_kernel_topk_reduce
        self.token_back_schedule_mode = load_balance_mode if load_balance_mode == "atomic_counter" else "static"

        # --- next Router-push token communication component. ---
        # dGLU dispatches raw grad_out tokens + the per-token routing prob (topk score)
        # into the pool; the dfc2 epilogue folds the prob into d_gate/d_up.  So the router
        # ALWAYS pushes scores into the pool -> apply_topk_at_fc1=True.
        mma_cta_count = 2 if use_2cta_instrs else 1
        cta_tile_m = mma_tiler_mnk[0] // mma_cta_count
        cluster_m, cluster_n = self.cluster_shape_mn
        tokens_per_fc1_ready_slot = cta_tile_m * cluster_m
        hidden_per_fc2_cluster_tile = cta_tile_m * cluster_m
        fc2_done_signals_per_token_tile = ceil_div(hidden, hidden_per_fc2_cluster_tile) * cluster_m * cluster_n
        promised_launchable_sm_count = launch_cluster_count * cluster_m * cluster_n
        quant_kind = _AB_DTYPE_TO_QUANT_KIND[ab_dtype]
        tc_problem_desc = ProblemDesc(
            {
                "world_size": world_size,
                "expert_count": self.num_total_experts,
                "topk": num_topk,
                "max_tokens_per_rank": max_tokens_per_rank,
                "max_recv_size_per_rank": max_recv_size_per_rank,
                "hidden_size": hidden,
                "quant_kind": str(quant_kind),
                "combine_format": combine_format,
                "apply_topk_at_fc1": True,
            }
        )
        tc_impl_desc = ImplDesc(
            {
                "token_padding_block": token_padding_block,
                "sf_padding_block": sf_padding_block,
                "tokens_per_fc1_ready_slot": tokens_per_fc1_ready_slot,
                "fc2_done_signals_per_token_tile": fc2_done_signals_per_token_tile,
                "promised_launchable_sm_count": promised_launchable_sm_count,
                "drop_on_overflow": drop_on_overflow,
                "token_in_flag_batch": flag_batch,
                "token_back_mode": token_back_mode,
                "token_back_schedule_mode": self.token_back_schedule_mode,
                "reduce_topk_in_kernel": fc2_in_kernel_topk_reduce,
            }
        )
        self.token_comm = TokenCommDeterministic(tc_problem_desc, tc_impl_desc)
        self.pool_token_capacity = self.token_comm.worst_case_token_count

        # --- SMEM sub-buffer for the token_comm transport. ---
        tc_smem_ws = SmemWorkspace()
        self.token_comm.register_smem_regions(tc_smem_ws)
        tc_smem_ws.finalize(max_bytes=self.smem_capacity)
        self.tc_smem_ws = tc_smem_ws
        self._token_comm_smem_bytes = tc_smem_ws.total_bytes

        # Build the scheduler NOW (launch_cluster_count known at construction) so its
        # separately-allocated SMEM is reservable by ``_smem_misc_budget_bytes``.
        _ec, _ig, _hd = static_expert_shape
        self._build_scheduler(
            expert_cnt=_ec, intermediate_gateup=_ig, hidden_dim=_hd, launch_cluster_count=launch_cluster_count
        )
        self._sched_smem_bytes = self.sched_smem_ws.total_bytes

        # --- Post-kernel top-k reduction (skipped under in-kernel reduce). ---
        self._topk_reduce = None if fc2_in_kernel_topk_reduce else TopkReduce(hidden, num_topk, combine_format)

        # --- Device workspace (next model): dGLU pools + token_comm regions. ---
        self._mega_device_workspace = self._build_megamoe_device_workspace()

        # --- Bind every KernelClass schema field under its schema name. ---
        self.expert_count = self.num_total_experts
        self.intermediate_gateup_size = self.intermediate_downproj
        self.hidden_size = hidden
        self.quant_kind = _quant_kind
        self.topk = num_topk
        self.cluster_shape_mnk = tuple(cluster_shape_mnk)
        self.mma_tiler_mnk = tuple(mma_tiler_mnk)
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode
        self.force_static_sched = force_static_sched
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.epi_flag_batch = tuple(epi_flag_batch)
        self.flag_batch = flag_batch
        self.act_func = act_func
        self.dfc2_recompute = dfc2_recompute
        self.dfc2_col_output = dfc2_col_output
        self.use_2cta_instrs = use_2cta_instrs

        # Optional post-kernel token-axis MXFP8 requantization of the routed
        # grad_out pool consumed as the dfc2 input.
        if self.enable_grad_y2_col_quant:
            col_quant_type = "mxfp8_e4m3" if ab_dtype is cutlass.Float8E4M3FN else "mxfp8_e5m2"
            self.grad_y2_col_quant = Mxfp8ColRequant(
                hidden=self.hidden,
                num_experts=self.num_experts_per_rank,
                max_total_tokens=(
                    self.world_size
                    * self.max_tokens_per_rank
                    * min(self.num_topk, self.num_experts_per_rank)
                ),
                quant_type=col_quant_type,
                num_persistent_ctas=self.num_ctas_grad_y2_col_quant,
                token_padding_block=self.token_padding_block,
                sf_padding_block=self.sf_padding_block,
            )

    def _smem_misc_budget_bytes(self) -> int:
        """Reserve the token_comm transport + scheduler SMEM on top of the base misc budget."""
        _sched = getattr(self, "_sched_smem_bytes", 0)
        return super()._smem_misc_budget_bytes() + self._token_comm_smem_bytes + _sched + self._SMEM_ALLOC_MARGIN

    def get_aux_output_shapes(self) -> dict:
        """Shapes of the fixed-ABI dFC2 auxiliary outputs."""
        data_token_capacity = self.token_comm.worst_case_token_count
        sf_token_capacity = self.token_comm.worst_case_sf_token_count
        column_sf_row_count = sf_token_capacity // self.sf_vec_size
        return {
            "dprob": (self.max_tokens_per_rank, self.num_topk),
            "fc1_recompute": (data_token_capacity, self.intermediate_downproj),
            "fc1_recompute_sf": (column_sf_row_count, self.intermediate_downproj),
            "fc1_col_output": (data_token_capacity, self.intermediate_gateup),
            "fc1_col_output_sf": (column_sf_row_count, self.intermediate_gateup),
            "grad_y2": (data_token_capacity, self.hidden),
            "grad_y2_sf": (sf_token_capacity * (self.hidden // self.sf_vec_size),),
        }

    def get_fc1_preact_shape(self) -> Tuple[int, int]:
        """Shape of the externally supplied, pool-indexed gate||up pre-activations."""
        return (self.token_comm.worst_case_token_count, self.intermediate_gateup)

    @cute.jit
    def _validate_fixed_pool_tensor(self, tensor: cute.Tensor, dtype, expected_shape) -> None:
        if cutlass.const_expr(tensor.element_type is not dtype):
            raise TypeError("pool-domain tensor has an unexpected element type.")
        if cutlass.const_expr(cute.rank(tensor.layout) != 2):
            raise ValueError("pool-domain tensor must be rank 2.")
        if cutlass.const_expr(
            not isinstance(tensor.shape[0], int)
            or not isinstance(tensor.shape[1], int)
            or tensor.shape[0] != expected_shape[0]
            or tensor.shape[1] != expected_shape[1]
        ):
            raise ValueError(f"pool-domain tensor must have static shape {expected_shape}.")
        if cutlass.const_expr(tensor.stride[0] != expected_shape[1] or tensor.stride[1] != 1):
            raise ValueError("pool-domain tensor must be compact row-major.")

    def _build_megamoe_device_workspace(self) -> DeviceWorkspace:
        """Register internal dGLU pools, counters, and token-comm regions."""
        sf_dtype = cutlass.Float8E8M0FNU
        sf_vec_size = self.sf_vec_size
        data_token_capacity = self.token_comm.worst_case_token_count
        sf_token_capacity = self.token_comm.worst_case_sf_token_count
        inter_gateup = self.intermediate_gateup  # grad_y1 width (DOUBLED)

        # grad_y1 SF: row-quant, DOUBLED N columns.
        sf_block_cols_back = round_up(ceil_div(inter_gateup, sf_vec_size), 4)
        counter_slot_count = self.token_comm.max_fc1_ready_slot_count

        dw = DeviceWorkspace()
        # grad_y1 (dfc2 output), consumed as the dfc1 (fc2) GEMM-B.
        dw.register(
            self.fc1_output_region,
            self.ab_dtype,
            (data_token_capacity, inter_gateup),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        dw.register(
            self.fc1_output_sf_region,
            sf_dtype,
            (sf_token_capacity, sf_block_cols_back),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        # cross-phase fc1->fc2 done counter.
        dw.register(
            self.fc1_done_counter_region,
            cutlass.Int32,
            (counter_slot_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        # Dynamic load-balance atomic counter (scheduler claims work by atomic-inc).
        # Always registered (cheap 1-int); the base kernel only reads it in
        # atomic_counter mode, but it must be zeroed between back-to-back launches.
        dw.register(
            self.load_balance_counter_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        # Reserve a slot for the scheduler's atomic work-id counter (persistent-grid
        # dynamic work distribution in atomic_counter mode).
        dw.register(
            self.sched_work_id_region, cutlass.Int32, (4,), buffer_space="local", byte_alignment=16, reset="tail_reset"
        )
        # Local mirror of the shared token_src_metadata (Int64 per pool slot), filled
        # host-side after the launch so the recompute/col-output validation can read it.
        dw.register(
            self.token_src_metadata_local_region,
            cutlass.Int64,
            (data_token_capacity,),
            buffer_space="local",
            byte_alignment=16,
        )
        if self.enable_grad_y2_col_quant:
            dw.register(
                self.grad_y2_sizes_region,
                cutlass.Int32,
                (self.num_experts_per_rank,),
                buffer_space="local",
                byte_alignment=16,
            )
        self.token_comm.register_device_workspace(dw)
        dw.finalize()
        return dw

    @property
    def _local_offsets(self) -> dict:
        """Legacy-name -> byte-offset map for inherited tester pool reads."""
        dw = self._mega_device_workspace
        return {
            "fc1_output": dw.offset(self.fc1_output_region),
            "fc1_output_sf": dw.offset(self.fc1_output_sf_region),
            "fc1_done_counter": dw.offset(self.fc1_done_counter_region),
            # For the legacy dfc2_recompute / dfc2_col_output validation:
            "l1_token_buffer": dw.offset(self.token_comm.fc1_activation_region),
            "token_src_metadata": dw.offset(self.token_src_metadata_local_region),
        }

    @property
    def _shared_metadata_offset(self) -> int:
        """Byte offset of the token_comm shared token_src_metadata (for the host mirror copy)."""
        return self._mega_device_workspace.offset(self.token_comm._router.token_src_metadata_region)

    @property
    def _local_region_by_name(self) -> dict:
        """Legacy-name -> object exposing ``.nbytes`` for the inherited tester's pool reads."""
        dw = self._mega_device_workspace
        name_to_region = {
            "fc1_output": self.fc1_output_region,
            "fc1_output_sf": self.fc1_output_sf_region,
            "fc1_done_counter": self.fc1_done_counter_region,
            "l1_token_buffer": self.token_comm.fc1_activation_region,
            "token_src_metadata": self.token_src_metadata_local_region,
        }
        return {name: SimpleNamespace(nbytes=dw.nbytes(region)) for name, region in name_to_region.items()}

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return required (local, shared/symmetric) workspace bytes."""
        return self._mega_device_workspace.local_and_shared_bytes

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        return self._mega_device_workspace.require_zero_workspace_leading_bytes

    # =========================================================================
    # token_comm_hook_* -- filled with next's Router-push TokenCommDeterministic calls.
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.tc_smem_ws.storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_pointer(self._mega_device_workspace)

    def sched_ext_fc1_peek_threshold(self) -> int:  # noqa: D401
        return super().sched_ext_fc1_peek_threshold()

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """The scheduler warp must wait for the Router to publish per-expert sizes."""
        self.token_comm.wait_for_sizes_ready(self._mega_device_workspace)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        """No-op: FC1 input readiness is enforced by the scheduler extension's fc1_ready spin."""
        pass

    @cute.jit
    def token_comm_hook_dispatch_warp_body(self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx):
        """Transfer warps (8-11): pull grad_out from peers into the local FC1 pool."""
        self.token_comm.token_in(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())
        if cutlass.const_expr(self.token_comm.token_back_enabled and not self.token_back_standalone):
            self.token_comm.token_back(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())

    @cute.jit
    def token_comm_hook_token_back_warp_body(self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx):
        """Standalone token-back warps (12-15): push grad_x back to source ranks."""
        self.token_comm.token_back(self.tc_smem_ws, token_comm_storage.buffer.data_ptr())

    @cute.jit
    def token_comm_hook_kernel_tail(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """Cross-rank drain + workspace tail reset, performed by the transfer warps."""
        if cutlass.const_expr(self.enable_grad_y2_col_quant):
            self._snapshot_grad_y2_expert_sizes(tidx)
        cute.arch.sync_threads()
        if (warp_idx >= self.dispatch_warp_id[0]) & (warp_idx <= self.dispatch_warp_id[-1]):
            self.token_comm.reset_tail()
        self.token_comm.remove_device_members()

    @cute.jit
    def _snapshot_grad_y2_expert_sizes(self, tidx) -> None:
        """Preserve local expert counts before token_comm tail reset."""
        from cutlass.cutlass_dsl import Int32

        dw = self._mega_device_workspace
        if self.token_comm._linear_cta_idx == Int32(0):
            sizes = self.token_comm.local_expert_sizes(dw, self.token_comm._local_rank)
            snapshot = dw.tensor(self.grad_y2_sizes_region)
            block_dim_x, _, _ = cute.arch.block_dim()
            expert_idx = tidx
            while expert_idx < Int32(self.num_experts_per_rank):
                snapshot[expert_idx] = Int32(sizes[expert_idx])
                expert_idx = expert_idx + block_dim_x

    # =========================================================================
    # Host launch: Router kernel -> fused MegaMoE backward kernel -> top-k reduction.
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        grad_out: cute.Tensor,  # (max_tokens_per_rank, hidden) fp8
        grad_out_sf: cute.Tensor,  # (max_tokens_per_rank, hidden // sf_vec_size) E8M0
        topk_idx: cute.Tensor,  # (max_tokens_per_rank, num_topk)
        topk_weights: cute.Tensor,  # (max_tokens_per_rank, num_topk) Float32 (prob)
        fc1_weight: cute.Tensor,  # W2^T: (experts_per_rank, hidden, inter_downproj)
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,  # W1^T: (experts_per_rank, intermediate, hidden)
        fc2_weight_sf: cute.Tensor,
        beta: cute.Tensor,  # (experts_per_rank,) Float32
        fc1_preact: cute.Tensor,  # (pool_token_capacity, intermediate_gateup) BFloat16
        output_activation: cute.Tensor,  # (max_tokens_per_rank, topk, hidden) BF16
        overflow_flag: cute.Tensor,  # (1,) Int32, per-rank FC12 overflow output
        dprob: cute.Tensor,  # (max_tokens_per_rank, topk) Float32; symmetric, pre-zeroed
        fc1_recompute: cute.Tensor,  # (pool_token_capacity, inter_downproj)
        fc1_recompute_sf: cute.Tensor,  # (col_sf_rows, inter_downproj) E8M0
        fc1_col_output: cute.Tensor,  # (pool_token_capacity, intermediate_gateup)
        fc1_col_output_sf: cute.Tensor,  # (col_sf_rows, intermediate_gateup) E8M0
        grad_y2: cute.Tensor,  # (pool_token_capacity, hidden) token-axis MXFP8
        grad_y2_sf: cute.Tensor,  # flat MN-major E8M0 bytes
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,  # symmetric (NVLink) heap base
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
    ) -> None:
        """Launch the Router, then the fused backward main kernel, then (optionally) the top-k reduce."""

        dw = self._mega_device_workspace
        local_rank = peer_rank_ptr_mapper_host.rank
        aux_shapes = self.get_aux_output_shapes()
        self._validate_fixed_pool_tensor(
            fc1_preact, cutlass.BFloat16, self.get_fc1_preact_shape()
        )
        self._validate_fixed_pool_tensor(
            fc1_recompute, self.ab_dtype, aux_shapes["fc1_recompute"]
        )
        self._validate_fixed_pool_tensor(
            fc1_recompute_sf,
            self.token_comm.activation_sf_dtype,
            aux_shapes["fc1_recompute_sf"],
        )
        self._validate_fixed_pool_tensor(
            fc1_col_output, self.ab_dtype, aux_shapes["fc1_col_output"]
        )
        self._validate_fixed_pool_tensor(
            fc1_col_output_sf,
            self.token_comm.activation_sf_dtype,
            aux_shapes["fc1_col_output_sf"],
        )
        self.token_comm.launch_router(
            topk_indices=topk_idx,
            topk_scores=topk_weights,
            local_rank=local_rank,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            device_workspace=dw,
            overflow_flag=overflow_flag,
            stream=stream,
        )
        peer_mapper = peer_rank_ptr_mapper_host.make_device_object()
        dw.assign_device_members(local_workspace, shared_workspace)

        activation_pool = self.token_comm.fc1_activation_tensor(dw)
        _sf_pool_atom = self.token_comm.fc1_activation_sf_tensor(dw)
        activation_sf_pool = cute.make_tensor(
            _sf_pool_atom.iterator,
            cute.make_layout(
                (self.token_comm.worst_case_sf_token_count, self.hidden // self.sf_vec_size),
                stride=(self.token_comm.activation_sf_hidden_padded, 1),
            ),
        )
        fc1_output = dw.tensor(self.fc1_output_region)
        fc1_output_sf = dw.tensor(self.fc1_output_sf_region)
        fc1_done_counter = dw.tensor(self.fc1_done_counter_region)
        load_balance_counter = (
            dw.tensor(self.load_balance_counter_region) if self.token_back_schedule_mode == "atomic_counter" else None
        )
        pool_topk_scores = self.token_comm.fc1_topk_scores_tensor(dw)

        if cutlass.const_expr(self.reduce_topk_in_kernel):
            # In-kernel top-k reduce (epi_warps + bf16 combine): the epilogue red-adds each
            # topk grad_x contribution straight into the (pre-zeroed) 2D output.
            pre_reduced = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (output_activation.shape[0], 1, output_activation.shape[1]),
                    stride=(output_activation.stride[0], output_activation.stride[0], output_activation.stride[1]),
                ),
            )
            pre_reduced_sf = None
        else:
            pre_reduced = self.token_comm.pre_reduced_activation_tensor(dw)
            pre_reduced_sf = self.token_comm.pre_reduced_activation_sf_tensor(dw)

        if cutlass.const_expr(self.token_comm.token_back_push_data):
            # token_back-by-dispatch: the epilogue writes grad_x to the LOCAL fc2_activation
            # pool (same pool token_back reads), in its native (tokens, 1, hidden) shape.
            fc2_output = self.token_comm.fc2_activation_tensor(dw)
        else:
            # epi_warps: the epilogue peer-writes grad_x directly (combine_output = pre_reduced),
            _combine_hidden = pre_reduced.shape[2]
            fc2_output = cute.make_tensor(
                pre_reduced.iterator,
                cute.make_layout(
                    (pre_reduced.shape[0] * pre_reduced.shape[1], _combine_hidden), stride=(_combine_hidden, 1)
                ),
            )

        # dprob is a source-domain combine plane. Add a singleton value mode so
        # the epilogue can reuse Fc2OutputDest's (token, topk, value) resolver.
        dprob_combine = cute.make_tensor(
            dprob.iterator,
            cute.make_layout(
                (dprob.shape[0], dprob.shape[1], 1),
                stride=(dprob.stride[0], dprob.stride[1], 0),
            ),
        )

        super().__call__(
            activation_pool,
            fc1_weight,
            activation_sf_pool,
            fc1_weight_sf,
            fc1_output,
            fc1_output_sf,
            fc1_recompute,
            fc1_recompute_sf,
            fc1_col_output,
            fc1_col_output_sf,
            fc2_weight,
            fc2_weight_sf,
            fc2_output,
            fc1_preact,
            pool_topk_scores,
            beta,
            dprob_combine,
            fc1_done_counter,
            offs=None,
            load_balance_counter=load_balance_counter,
            max_active_clusters=self.launch_cluster_count,
            stream=stream,
            overflow_flag=overflow_flag,
            mega_peer_rank_ptr_mapper=peer_mapper,
            mega_local_rank=local_rank,
            mega_local_workspace=local_workspace,
            mega_shared_workspace=shared_workspace,
            mega_activation=grad_out,
            mega_activation_sf=grad_out_sf,
            mega_pre_reduced_activation=pre_reduced,
            mega_pre_reduced_activation_sf=pre_reduced_sf,
        )

        # Post-kernel top-k reduction: dequant + K-sum into the final output.
        if cutlass.const_expr(not self.reduce_topk_in_kernel):
            self._topk_reduce(pre_reduced, pre_reduced_sf, output_activation, None, stream)

        # Export the routed dfc2 input in token-axis MXFP8 form. The source
        # grad_out pool and its row-wise SF remain resident after reset_tail.
        if cutlass.const_expr(self.enable_grad_y2_col_quant):
            lw = local_workspace
            data_offset = dw.offset(self.token_comm.fc1_activation_region)
            sf_offset = dw.offset(self.token_comm.fc1_activation_sf_region)
            sizes_offset = dw.offset(self.grad_y2_sizes_region)
            sf_pool_bytes = self.token_comm.worst_case_sf_token_count * (self.hidden // self.sf_vec_size)
            src_data = cute.make_tensor(
                cute.make_ptr(
                    self.ab_dtype, lw.toint() + Int64(data_offset), AddressSpace.gmem, assumed_align=128
                ),
                cute.make_layout(
                    (self.token_comm.worst_case_token_count, self.hidden), stride=(self.hidden, 1)
                ),
            )
            src_sf_u8 = cute.make_tensor(
                cute.make_ptr(cutlass.Uint8, lw.toint() + Int64(sf_offset), AddressSpace.gmem, assumed_align=16),
                cute.make_layout((sf_pool_bytes,)),
            )
            expert_sizes = cute.make_tensor(
                cute.make_ptr(cutlass.Int32, lw.toint() + Int64(sizes_offset), AddressSpace.gmem, assumed_align=16),
                cute.make_layout((self.num_experts_per_rank,)),
            )
            self.grad_y2_col_quant(
                src_data,
                src_sf_u8,
                expert_sizes,
                grad_y2,
                grad_y2_sf,
                stream,
            )
