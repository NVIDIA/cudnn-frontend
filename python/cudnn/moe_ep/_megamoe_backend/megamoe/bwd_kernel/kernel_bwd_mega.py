# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M2.C part 2 (D1): in-kernel metadata-gather dout dispatch.

``BwdGatherComm`` is a trimmed ``TokenInPullTokenBackPush``: it drops the
forward's routing walk / count-exchange / dedup and instead drives the pull
straight off the FORWARD's persistent ``token_src_metadata``.  Per pool row it
reads ``(src_rank, src_token)`` from the metadata and peer-pulls the quantized
``dout`` row (+ SF) from a sym-heap ``my_dout`` buffer into the activation pool,
publishing ``fc1_ready_counter`` on the SAME slot the GEMM's TMA-B warp spins on
(``expert_task_tile_offset + token_idx_in_expert // cluster_tile_tokens``).

Everything else — the pull/store TMA ops, the SF ldg/store, the fc1-ready
release tracker, the SMEM ``pull_buffer`` storage, the TMA-B predispatch spin,
and the sched handshake — is REUSED verbatim from the forward comm (contracts
in ``BWD_DESIGN.md`` §D1).  Preconditions vs the forward: ``my_dout`` is staged
host-side before the collective launch and there are no cross-rank sends, so
``dispatch_prep`` / ``dispatch_barrier`` / the NVLink barrier are all skipped.

``Sm100MegaMoEMxfp8BwdKernel`` mirrors the forward's ``Sm100MegaMoEMxfp8Kernel``
delegation surface: it flips ``enable_token_comm`` on, builds the comm object,
and forwards the token-comm hooks.  It keeps ``fc2_in_kernel_topk_reduce=True``
so the dx token-back (M2.C part 1) still runs in the same launch.
"""

from __future__ import annotations

from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Int32, Int64, Float32
from cutlass.cute.typing import AddressSpace

from common.megamoe_constants import SfPaddingBlock

from src.token_comm import TokenInPullTokenBackPush, TokenSrcMetadata
from src.ptx_helpers import tma_load_1d_raw, tma_store_1d, ldg_b32_raw
from src.flag_batch import GpuReleaseFlagBatchTracker
from src.sf_swizzle import sf_atom_int32_offset

from megamoe.bwd_kernel.kernel_bwd_fc12 import Sm100SwigluMxfp8Fc12Kernel


class BwdGatherComm(TokenInPullTokenBackPush):
    """Metadata-driven dout gather (no routing walk / sends / dedup)."""

    @cute.jit
    def dispatch_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_linear_id = (
            Int32(bidx)
            + Int32(self.cluster_shape_mn[1]) * Int32(bidy)
            + Int32(self.cluster_shape_mn[1] * self.cluster_shape_mn[0])
            * Int32(bidz)
        )
        local_warp_idx = Int32(warp_idx) - Int32(self.dispatch_warp_start)

        # Sizes are host-supplied (offs); nothing to prep / exchange.  Just
        # release the sched warp (it arrive_and_waits on this same barrier).
        nb = pipeline.NamedBarrier(
            barrier_id=self.dispatch_to_sched_named_barrier_id,
            num_threads=self.dispatch_to_sched_threads,
        )
        nb.arrive()

        self.dispatch_pull_by_metadata(
            token_comm_storage,
            token_comm_args.input_token_buffer,
            token_comm_args.input_sf_buffer,
            token_comm_args.expert_recv_count_sum,
            token_comm_args.fc1_input_token_buffer,
            token_comm_args.fc1_input_sf_buffer,
            token_comm_args.fc1_ready_counter,
            token_comm_args.token_src_metadata,
            token_comm_args.peer_rank_ptr_mapper,
            cta_linear_id,
            local_warp_idx,
            lane_idx,
            num_sms=token_comm_args.sm_count,
        )

    @cute.jit
    def dispatch_pull_by_metadata(
        self,
        token_comm_storage,
        input_token_buffer,      # sym-heap my_dout (peer-pull SOURCE)
        input_sf_buffer,         # sym-heap my_dout_sf
        expert_recv_count_sum,   # host-staged per-expert token totals (i64)
        fc1_input_token_buffer,  # activation pool (peer-pull DEST = GEMM-A)
        fc1_input_sf_buffer,     # activation SF pool
        fc1_ready_counter,       # per-tile release counter (GEMM spins on it)
        token_src_metadata,      # forward's persistent record (READ)
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        *,
        num_sms,
    ):
        pull_mbar_ptr = token_comm_storage.pull_mbar.data_ptr()
        pull_buffer_ptr = token_comm_storage.pull_buffer.data_ptr()
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(pull_mbar_ptr + warp_idx, 1)
        cute.arch.sync_warp()

        phase_bit = Int32(0)
        current_expert_idx = Int32(-1)
        expert_start_idx = Int32(0)
        expert_end_idx = Int32(0)
        expert_pool_block_offset = Int32(0)
        expert_task_tile_offset = Int32(0)
        expert_sf_pool_block_offset = Int32(0)

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_addr=Int64(0),
            cumulated_flags=Int32(0),
            phase=Int32(0),
            tid=lane_idx,
        )

        NUM_EXPERTS_PER_LANE: cutlass.Constexpr[int] = (
            self.num_experts_per_rank + 31
        ) // 32
        stored_num_tokens_per_expert = []
        for _ in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            stored_num_tokens_per_expert.append(Int32(0))
        for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
            e_idx_for_lane = Int32(i * self.warp_threads) + lane_idx
            if e_idx_for_lane < Int32(self.num_experts_per_rank):
                sum_packed_init = expert_recv_count_sum[e_idx_for_lane]
                stored_num_tokens_per_expert[i] = Int32(
                    Int64(sum_packed_init) & Int64(0xFFFFFFFF)
                )
        cute.arch.sync_warp()

        num_global_warps: cutlass.Constexpr[int] = num_sms * self.num_dispatch_warps
        token_idx = sm_idx * Int32(self.num_dispatch_warps) + warp_idx

        while current_expert_idx < Int32(self.num_experts_per_rank):
            while (token_idx >= expert_end_idx) and (
                current_expert_idx < Int32(self.num_experts_per_rank)
            ):
                prev_valid_count = expert_end_idx - expert_start_idx
                prev_block_count = (
                    prev_valid_count + Int32(self.token_padding_block) - Int32(1)
                ) // Int32(self.token_padding_block)
                expert_pool_block_offset = expert_pool_block_offset + prev_block_count
                prev_task_tile_count = (
                    prev_valid_count + Int32(self.cluster_tile_tokens) - Int32(1)
                ) // Int32(self.cluster_tile_tokens)
                expert_task_tile_offset = (
                    expert_task_tile_offset + prev_task_tile_count
                )
                prev_sf_block_count = (
                    prev_valid_count + Int32(self.sf_padding_block) - Int32(1)
                ) // Int32(self.sf_padding_block)
                expert_sf_pool_block_offset = (
                    expert_sf_pool_block_offset + prev_sf_block_count
                )
                current_expert_idx = current_expert_idx + Int32(1)
                if current_expert_idx < Int32(self.num_experts_per_rank):
                    expert_start_idx = expert_end_idx
                    valid_value = Int32(0)
                    for i in cutlass.range_constexpr(0, NUM_EXPERTS_PER_LANE, 1):
                        if current_expert_idx == Int32(
                            i * self.warp_threads
                        ) + lane_idx:
                            valid_value = stored_num_tokens_per_expert[i]
                    total_for_expert = cute.arch.shuffle_sync(
                        valid_value, current_expert_idx % Int32(self.warp_threads)
                    )
                    expert_end_idx = expert_end_idx + total_for_expert

            if current_expert_idx < Int32(self.num_experts_per_rank):
                token_idx_in_expert = token_idx - expert_start_idx
                pool_token_idx = (
                    expert_pool_block_offset * Int32(self.token_padding_block)
                    + token_idx_in_expert
                )
                sf_token_in_pool_axis = (
                    expert_sf_pool_block_offset * Int32(self.sf_padding_block)
                    + token_idx_in_expert
                )

                md = TokenSrcMetadata.load(
                    token_src_metadata.iterator
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                src_rank = md.src_rank
                src_token = md.src_token

                cur_peer_offset = peer_rank_ptr_mapper.map(
                    Int64(0), src_rank, Int64(0)
                )
                inp_tok_local_base = input_token_buffer.iterator.toint()
                inp_sf_local_base = input_sf_buffer.iterator.toint()

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_src_addr = (
                        inp_tok_local_base
                        + cur_peer_offset
                        + Int64(src_token * Int32(self.hidden_bytes))
                    )
                    tma_load_1d_raw(
                        pull_buffer_warp_ptr,
                        tma_src_addr,
                        pull_mbar_ptr + warp_idx,
                        Int32(self.hidden_bytes),
                    )
                cute.arch.sync_warp()

                sf_passes: cutlass.Constexpr[int] = (
                    self.sf_uint32_per_token + 31
                ) // 32
                sf_vals = []
                for _ in cutlass.range_constexpr(0, sf_passes, 1):
                    sf_vals.append(Int32(0))
                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_addr = (
                            inp_sf_local_base
                            + cur_peer_offset
                            + Int64(
                                (src_token * Int32(self.sf_uint32_per_token) + j)
                                * Int32(4)
                            )
                        )
                        sf_vals[i] = ldg_b32_raw(sf_addr)

                for i in cutlass.range_constexpr(0, sf_passes, 1):
                    j = Int32(i * self.warp_threads) + lane_idx
                    if j < Int32(self.sf_uint32_per_token):
                        sf_int32_pos = sf_atom_int32_offset(
                            sf_token_in_pool_axis,
                            j,
                            num_k_atoms=self.sf_uint32_per_token,
                        )
                        fc1_input_sf_buffer[sf_int32_pos] = sf_vals[i]
                cute.arch.sync_warp()

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pull_mbar_ptr + warp_idx, Int32(self.hidden_bytes)
                    )
                    cute.arch.mbarrier_wait(pull_mbar_ptr + warp_idx, phase_bit)

                with cute.arch.elect_one():
                    pull_buffer_warp_ptr = pull_buffer_ptr + (
                        warp_idx * Int32(self.hidden_bytes)
                    )
                    tma_store_1d(
                        fc1_input_token_buffer.iterator
                        + (Int64(pool_token_idx) * Int64(self.hidden_bytes)),
                        pull_buffer_warp_ptr,
                        Int32(self.hidden_bytes),
                    )
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)

                task_tile_idx = expert_task_tile_offset + (
                    token_idx_in_expert // Int32(self.cluster_tile_tokens)
                )
                task_tile_addr = (fc1_ready_counter.iterator + task_tile_idx).toint()
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self._flag_batch, task_tile_addr,
                )
                cute.arch.sync_warp()
                phase_bit = phase_bit ^ Int32(1)

            token_idx = token_idx + Int32(num_global_warps)

        flag_tracker.fire()


class Sm100MegaMoEMxfp8BwdKernel(Sm100SwigluMxfp8Fc12Kernel):
    """Backward FC12 kernel with in-kernel dout gather + dx token-back."""

    def __init__(
        self,
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        **base_kwargs,
    ) -> None:
        super().__init__(fc2_in_kernel_topk_reduce=True, **base_kwargs)

        self.comm_world_size = world_size
        self.comm_local_rank = local_rank
        self.comm_num_topk = num_topk

        num_experts_per_rank = self.static_expert_shape[0]
        # MXFP8: 1 byte / elem; SF pulled in uint32 units (4 E8M0 per word).
        sf_atom_k_elements = 4 * self.sf_vec_size
        self.comm_sf_uint32_per_token = (
            (self.static_expert_shape[2] + sf_atom_k_elements - 1)
            // sf_atom_k_elements
        )

        # 12-warp topology (epilogue 0-3, mma/tma_a/tma_b/sched 4-7, dispatch 8-11).
        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        self.token_back_standalone = False
        self.token_back_warp_id = None

        num_other_warps = len(self.epilogue_warp_id) + 4  # + mma/tma_a/tma_b/sched
        self.token_comm = BwdGatherComm(
            world_size=world_size,
            num_topk=num_topk,
            num_experts_per_rank=num_experts_per_rank,
            num_total_experts=world_size * num_experts_per_rank,
            hidden=self.static_expert_shape[2],
            fc1_token_dtype=self.ab_dtype,
            sf_uint32_per_token=self.comm_sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.mma_tiler_mnk[0],
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            token_back_by_dispatch=False,
            token_back_standalone=False,
            combine_format=None,       # bf16 combine (dx via fc2_in_kernel_topk_reduce)
            flag_batch=1,
        )

    # -- token-comm delegation surface (mirrors Sm100MegaMoEMxfp8Kernel) ----

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    def sched_ext_fc1_peek_threshold(self) -> int:
        return self.cluster_tile_tokens

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        self.token_comm.fc1_tma_b_predispatch_spin(token_comm_args, work_tile_info)

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
    ):
        self.token_comm.dispatch_warp_body(
            token_comm_args, token_comm_storage,
            warp_idx=warp_idx, lane_idx=lane_idx, tidx=tidx,
        )
