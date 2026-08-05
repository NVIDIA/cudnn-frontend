# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M2.B backward wrapper: the backward FC12 kernel fed from the forward pools.

``MegaMoeMxfp8Backward`` mirrors ``megamoe/forward.py``: persistent
capacity-sized buffers, one-time ``cute.compile``, per-step launch with
baked kwargs.  The dgrad chain (gemm1 dA -> SwiGLU-bwd(fc1_c) -> gemm2 dxg
+ bf16 DFC1 stash) runs in ONE kernel; around it, thin torch glue does the
metadata-driven dout gather, the dx top-k sum, dtw from the forward's
per-copy combine staging, and the token-K wgrads (fp8_bwd's
``gmm_wgrad_2d2d``) on the kernel stashes.

v1 comm model: same as bwd_v0 (allgather for world>1; world=1 = pure local
gathers).  The in-kernel peer-pull dispatch / token-back push (BWD_DESIGN.md
D1) replaces the gathers in M2.C.

Collective contract: every rank calls backward each step (world>1 has
allgathers/all_reduce).  Requires the forward layer built with
``impl.generate_c=True`` and bf16 combine staging (default).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import torch.distributed as dist

import megamoe.repo_path  # noqa: F401

from pt.quant import rotate_trailing

from megamoe.fp8_bwd import (
    _col_atom_order,
    _pack_scales_rowgroups,
    _phase,
    _round_up,
    gmm_wgrad_2d2d,
)
from megamoe.pools import decode_token_src_metadata, local_pool_views, shared_pool_views
from megamoe.quant_kernels import mxfp8_rowquant
from megamoe.training import dequant_mxfp8_pool  # noqa: F401  (XG dequant)

from megamoe.bwd_kernel.weights_bwd import quantize_moe_weights_mxfp8_bwd


class MegaMoeMxfp8Backward:
    """Compile-once backward kernel around a live MegaMoeMxfp8Forward."""

    def __init__(self, fwd, ep_cfg):
        self.fwd = fwd
        self.ep = ep_cfg
        H, I = ep_cfg.hidden_size, ep_cfg.intermediate_size
        self.E_local = ep_cfg.num_experts // ep_cfg.ep_size
        cap = fwd.fc1_c.shape[0]           # world*T*K + E_local*128
        self.capacity = cap
        dev = "cuda"

        # persistent kernel-facing buffers (baked into compiled kwargs)
        self.act_pool = torch.zeros((cap, H), dtype=torch.float8_e4m3fn, device=dev)
        # SF dtype must be e8m0 — the kernel specializes activation_sf's
        # cute dtype from this tensor (uint8 breaks the SFA TMA path).
        self.act_sf_swz = torch.zeros(
            (cap, H // 32), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)
        self.fc2_output = torch.zeros((cap, H), dtype=torch.bfloat16, device=dev)
        self.dfc1_stash = torch.zeros((cap, 2 * I), dtype=torch.bfloat16, device=dev)
        self.offs_buf = torch.zeros((self.E_local,), dtype=torch.int32, device=dev)

        # M2.C in-kernel dx token-back: sym-heap per-rank (T, H) buffer the
        # epilogue REDG-adds each source token's summed-over-topk dx into
        # (fc2_in_kernel_topk_reduce).  world=1 degrades to a plain cuda tensor.
        # (T, 1, H) so a plain DLPack view gives the (H, H, 1) layout the
        # epilogue's Fc2OutputDest slices as [src_token, 0, :].
        import os
        from moe_nvfp4_swapab.mega_runner import _sym_zeros
        from moe_nvfp4_swapab.runner_common import round_up as _round_up_sf
        self.T = fwd.cfg.max_tokens_per_rank
        self.dx_pool = _sym_zeros((self.T, 1, H), torch.bfloat16)

        # M2.C part 2 (D1): in-kernel dout gather.  Default ON — verified
        # (world=1 + 4-rank parity) and faster than the torch dispatch at both
        # world=1 (4.95 vs 5.12 ms bwd) and 4-rank (3.46 vs 3.50 ms).  Set
        # MEGA_BWD_INKERNEL_DISPATCH=0 to fall back to the torch gather.
        self.in_kernel_dispatch = bool(
            int(os.environ.get("MEGA_BWD_INKERNEL_DISPATCH", "1"))
        )
        if self.in_kernel_dispatch:
            # sym-heap dout SOURCE (staged locally each step, no allgather).
            # NVSHMEM has no fp8/e8m0 dtype, so allocate byte buffers viewed as
            # fp8/e8m0 (same trick as the forward's my_activation).
            from moe_mxfp8_glu.mega_runner import _sym_zeros_byte_view_1b
            sf_cols_padded = _round_up_sf(H // 32, 4)
            self.my_dout = _sym_zeros_byte_view_1b(
                (self.T, H), torch.float8_e4m3fn
            )
            self.my_dout_sf = _sym_zeros_byte_view_1b(
                (self.T, sf_cols_padded), torch.float8_e8m0fnu
            )
            # per-expert token totals the gather walks (low32 = count).
            self.recv_count_sum = torch.zeros(
                (self.E_local,), dtype=torch.int64, device=dev
            )
            # fc1-ready release counter: one Int32 per cluster tile + slack.
            cluster_tile_tokens = 256  # == mma_tiler_mnk[0]
            self.fc1_ready_slots = cap // cluster_tile_tokens + self.E_local
            self.fc1_ready_counter = torch.zeros(
                (self.fc1_ready_slots,), dtype=torch.int32, device=dev
            )

        # workspace (mirror of the kernel's get_workspace_size layout, 2I wide)
        downproj = 2 * I
        from common.megamoe_constants import SfPaddingBlock

        sf_rows = cap + self.E_local * SfPaddingBlock
        sf_cols = ((downproj // 32) + 3) // 4 * 4
        self._counter_slots = (cap + 255) // 256 + self.E_local
        n0 = cap * downproj
        n1 = sf_rows * sf_cols
        n2 = self._counter_slots * 4
        self.workspace = torch.zeros((n0 + n1 + n2,), dtype=torch.uint8, device=dev)
        self.fc1_output = self.workspace[:n0].view(torch.float8_e4m3fn).view(cap, downproj)
        self.fc1_output_sf = self.workspace[n0 : n0 + n1].view(sf_rows, sf_cols)
        self.fc1_done_counter = self.workspace[n0 + n1 : n0 + n1 + n2].view(torch.int32)

        self._weights = None
        self._compiled = None

    def load_weights(self, w13: torch.Tensor, w2: torch.Tensor) -> None:
        q = quantize_moe_weights_mxfp8_bwd(w13, w2)
        if self._weights is None:
            self._weights = q
        else:
            self._weights.gemm1_weight.copy_(q.gemm1_weight)
            self._weights.gemm1_weight_sf.view(torch.uint8).copy_(
                q.gemm1_weight_sf.view(torch.uint8)
            )
            self._weights.gemm2_weight.copy_(q.gemm2_weight)
            self._weights.gemm2_weight_sf.view(torch.uint8).copy_(
                q.gemm2_weight_sf.view(torch.uint8)
            )

    def _compile(self):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from common.megamoe_constants import SfPaddingBlock
        from megamoe.bwd_kernel.kernel_bwd_fc12 import Sm100SwigluMxfp8Fc12Kernel

        ep = self.ep
        H, I = ep.hidden_size, ep.intermediate_size
        mma_tiler = (256, 256, 128)
        cluster = (2, 1, 1)
        max_active = utils.HardwareInfo().get_max_active_clusters(
            cluster[0] * cluster[1]
        )
        base_kw = dict(
            mma_tiler_mnk=mma_tiler,
            cluster_shape_mnk=cluster,
            use_2cta_instrs=True,
            group_hint=max_active,
            token_padding_block=128,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode="static",
            static_expert_shape=(self.E_local, I, H),   # gemm1 N = I
            ab_dtype=cutlass.Float8E4M3FN,
            sf_vec_size=32,
            apply_topk_in_fc1=True,
            generate_c=True,
            use_stg_fc1=True,
        )
        if self.in_kernel_dispatch:
            # D1: dispatch warps gather dout; epilogue still pushes dx
            # (fc2_in_kernel_topk_reduce is forced on inside the subclass).
            from megamoe.bwd_kernel.kernel_bwd_mega import Sm100MegaMoEMxfp8BwdKernel
            self._kernel = Sm100MegaMoEMxfp8BwdKernel(
                world_size=ep.ep_size,
                local_rank=ep.ep_rank,
                num_topk=ep.top_k,
                max_tokens_per_rank=self.T,
                **base_kw,
            )
            self._kernel.comm_sm_count = max_active * (cluster[0] * cluster[1])
        else:
            self._kernel = Sm100SwigluMxfp8Fc12Kernel(
                # M2.C part 1: epilogue REDG-adds dx across the top-k copies.
                fc2_in_kernel_topk_reduce=True, **base_kw,
            )
        self._kernel.comm_world_size = ep.ep_size
        self._kernel.comm_local_rank = ep.ep_rank
        self._kernel.comm_num_topk = ep.top_k

        def to_cute(t):
            ct = cutlass_torch.from_dlpack(t, assumed_align=16)
            return ct.mark_layout_dynamic(
                leading_dim=cutlass_torch.get_leading_dim(t)
            )

        # tw pool: the forward's persistent l1_topk_weights_buffer, zero-copy.
        # Its own capacity (pool_token_capacity) can differ from fc1_c's row
        # count; the kernel only ever indexes rows < Mp <= both.
        lv = local_pool_views(self.fwd)
        tw_pool = lv["l1_topk_weights_buffer"]

        # -- M2.C in-kernel dx push plumbing --------------------------------
        # peer mapper over the dx sym-heap buffer (identity delta for world=1);
        # combine_output is the (T, 1, H) view the epilogue's Fc2OutputDest
        # slices as [src_token, 0, :]; token_src_metadata is the forward's
        # persistent record (same pool order the backward GEMM walks).
        from moe_nvfp4_swapab.mega_runner import _compute_peer_offsets
        from src.sym_buffer import SymBufferHost

        sym_base, peer_offsets = _compute_peer_offsets(self.dx_pool, ep.ep_size)
        self._peer_mapper = SymBufferHost(
            base_addr=sym_base,
            offsets=tuple(peer_offsets),
            rank_idx=ep.ep_rank,
            num_max_ranks=ep.ep_size,
        )
        self._combine_out_cute = to_cute(self.dx_pool)
        self._md_cute = to_cute(lv["token_src_metadata"])

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._kwargs = dict(
            activation=to_cute(self.act_pool),
            activation_sf=to_cute(self.act_sf_swz),
            fc1_weight=to_cute(self._weights.gemm1_weight),
            fc1_weight_sf=to_cute(self._weights.gemm1_weight_sf),
            fc1_output=to_cute(self.fc1_output),
            fc1_output_sf=to_cute(self.fc1_output_sf),
            fc2_weight=to_cute(self._weights.gemm2_weight),
            fc2_weight_sf=to_cute(self._weights.gemm2_weight_sf),
            fc2_output=to_cute(self.fc2_output),
            topk_scores=to_cute(tw_pool),
            fc1_done_counter=to_cute(self.fc1_done_counter),
            offs=to_cute(self.offs_buf),
            fc1_c=to_cute(self.fwd.fc1_c),
            fc1_c_out=to_cute(self.dfc1_stash),
            peer_rank_ptr_mapper_host=self._peer_mapper,
            combine_output_bwd=self._combine_out_cute,
            token_src_metadata_bwd=self._md_cute,
            stream=stream,
        )
        if self.in_kernel_dispatch:
            # byte views for the peer-pulled fp8 source + activation pool;
            # SF pool as flat int32 (the gather writes swizzled Int32 words).
            act_sf_i32 = self.act_sf_swz.view(torch.uint8).view(torch.int32).reshape(-1)
            self._kwargs.update(
                input_token_buffer_bwd=to_cute(self.my_dout.view(torch.uint8)),
                input_sf_buffer_bwd=to_cute(self.my_dout_sf.view(torch.uint8)),
                fc1_input_token_buffer_bwd=to_cute(self.act_pool.view(torch.uint8)),
                fc1_input_sf_buffer_bwd=to_cute(act_sf_i32),
                fc1_ready_counter_bwd=to_cute(self.fc1_ready_counter),
                expert_recv_count_sum_bwd=to_cute(self.recv_count_sum),
            )
        compile_kwargs = dict(self._kwargs)
        compile_kwargs["max_active_clusters"] = max_active
        self._compiled = cute.compile(self._kernel, **compile_kwargs)

    def launch(self):
        if self._compiled is None:
            self._compile()
        self._compiled(**self._kwargs)


def mega_backward(layer, topk_ids, topk_weights, dout, T):
    """Kernel-dgrad mxfp8 backward; returns (dx, dtw, dw13, dw2)."""
    ep = layer.ep_cfg
    H, I, K = ep.hidden_size, ep.intermediate_size, ep.top_k
    E_local = ep.num_experts // ep.ep_size
    expert_start = ep.ep_rank * E_local
    world = ep.ep_size
    fwd = layer._fwd
    device = dout.device
    group = ep.process_group

    bwd: MegaMoeMxfp8Backward = layer._mega_bwd
    if getattr(layer, "_mega_bwd_wdirty", True):
        with _phase("weight-quant"):
            w13 = layer.w13.detach().to(torch.bfloat16)
            if layer.qcfg.turboquant:
                w13 = rotate_trailing(w13, layer.q_rot)
            bwd.load_weights(w13, layer.w2.detach().to(torch.bfloat16))
        layer._mega_bwd_wdirty = False

    with _phase("prep"):
        if world > 1:
            ids_all = torch.empty((world, T, K), dtype=topk_ids.dtype, device=device)
            dist.all_gather_into_tensor(ids_all, topk_ids.contiguous(), group=group)
        else:
            ids_all = topk_ids.view(1, T, K)
        local = ids_all.reshape(-1) - expert_start
        counts = torch.bincount(
            local[(local >= 0) & (local < E_local)], minlength=E_local
        ).tolist()
        padded = [_round_up(n, 128) for n in counts]
        Mp = max(sum(padded), 128)
        offs_valid = torch.tensor(
            [sum(counts[: i + 1]) for i in range(E_local)],
            device=device, dtype=torch.int32,
        )
        offs_padded = torch.tensor(
            [sum(padded[: i + 1]) for i in range(E_local)],
            device=device, dtype=torch.int32,
        )
        doffs = [sum(padded[:i]) for i in range(E_local)]
        valid = torch.cat(
            [torch.arange(o, o + n, device=device) for o, n in zip(doffs, counts)]
        ) if sum(counts) else torch.empty(0, dtype=torch.long, device=device)

        lv = local_pool_views(fwd)
        src_rank, src_token, src_topk, _, _ = decode_token_src_metadata(
            lv["token_src_metadata"][:Mp]
        )
        sr, st, sk = src_rank[valid].long(), src_token[valid].long(), src_topk[valid].long()
        flat = (sr * T + st) if world > 1 else st

        cbt = Mp // 128
        order_h = _col_atom_order(offs_padded, H // 128, cbt)
        order_2i = _col_atom_order(offs_padded, (2 * I) // 128, cbt)
        order_i = _col_atom_order(offs_padded, I // 128, cbt)

    with torch.no_grad():
        with _phase("dispatch"):
            if bwd.in_kernel_dispatch:
                # D1: stage dout locally into the sym source (no allgather);
                # the kernel's dispatch warps peer-pull it into act_pool by
                # metadata.  Pool + SF start zero so pad rows stay clean.
                dout_q, dout_sf = mxfp8_rowquant(dout.to(torch.bfloat16))
                bwd.my_dout.view(torch.uint8).copy_(dout_q.view(torch.uint8))
                bwd.my_dout_sf.view(torch.uint8)[:, : H // 32].copy_(
                    dout_sf.contiguous().view(torch.uint8)
                )
                bwd.recv_count_sum.copy_(
                    torch.tensor(counts, device=device, dtype=torch.int64)
                )
                bwd.act_pool.view(torch.uint8).zero_()
                bwd.act_sf_swz.view(torch.uint8).zero_()
                bwd.fc1_ready_counter.zero_()
                if world > 1:
                    # all ranks must finish staging before any peer pull.
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
            else:
                # quantize dout ONCE per source token, gather fp8 bytes + SFs.
                # All indexing/collectives run on uint8 views (fp8/e8m0 advanced
                # indexing and NCCL support are not implemented).
                dout_q, dout_sf = mxfp8_rowquant(dout.to(torch.bfloat16))
                q_local = dout_q.view(torch.uint8)
                sf_local = dout_sf.contiguous().view(torch.uint8)
                if world > 1:
                    q_all = torch.empty((world * T, H), dtype=torch.uint8, device=device)
                    dist.all_gather_into_tensor(q_all, q_local, group=group)
                    sf_all = torch.empty(
                        (world * T, H // 32), dtype=torch.uint8, device=device
                    )
                    dist.all_gather_into_tensor(sf_all, sf_local, group=group)
                else:
                    q_all, sf_all = q_local, sf_local
                bwd.act_pool[:Mp].view(torch.uint8).zero_()
                bwd.act_pool.view(torch.uint8)[valid] = q_all[flat]
                sf_pool = torch.zeros((Mp, H // 32), dtype=torch.uint8, device=device)
                sf_pool[valid] = sf_all[flat]
                packed = _pack_scales_rowgroups(sf_pool, offs_padded)
                bwd.act_sf_swz.view(torch.uint8).view(-1)[: packed.numel()].copy_(
                    packed.view(torch.uint8).view(-1)
                )
            bwd.offs_buf.copy_(offs_valid)
            bwd.fc1_done_counter.zero_()
            # dx REDG target must start at zero each step (epilogue accumulates).
            bwd.dx_pool.zero_()

        with _phase("gemm"):
            bwd.launch()   # dgrad chain + DFC1 stash + in-kernel dx push

        with _phase("elemwise"):
            # wgrad operands from the stashes
            pair = fwd.fc1_c[:Mp].float().view(Mp, I // 32, 2, 32)
            gate, up = pair[:, :, 0].reshape(Mp, I), pair[:, :, 1].reshape(Mp, I)
            tw_pool = local_pool_views(fwd)["l1_topk_weights_buffer"][:Mp]
            actw = (F.silu(gate) * up) * tw_pool[:, None]
            ACTW = torch.zeros((Mp, I), dtype=torch.bfloat16, device=device)
            ACTW[valid] = actw[valid].to(torch.bfloat16)
            # wgrad dY operand: gather the raw bf16 dout (the wgrad
            # transquant re-quantizes along tokens anyway — same semantics
            # as fp8_bwd/pool, and far cheaper than dequantizing the pool)
            if world > 1:
                dout_all = torch.empty(
                    (world * T, H), dtype=torch.bfloat16, device=device
                )
                dist.all_gather_into_tensor(
                    dout_all, dout.to(torch.bfloat16).contiguous(), group=group
                )
            else:
                dout_all = dout.to(torch.bfloat16)
            DOUTG = torch.zeros((Mp, H), dtype=torch.bfloat16, device=device)
            DOUTG[valid] = dout_all[flat]
            x_q = dequant_mxfp8_pool(
                fwd.my_activation[:T], fwd.my_activation_sf[:T], H
            ).to(torch.bfloat16)
            if world > 1:
                x_all = torch.empty((world * T, H), dtype=x_q.dtype, device=device)
                dist.all_gather_into_tensor(x_all, x_q, group=group)
            else:
                x_all = x_q
            XG = torch.zeros((Mp, H), dtype=torch.bfloat16, device=device)
            XG[valid] = x_all[flat]

        # wgrads (tokens on K); DFC1 stash is (dg, du)-interleaved
        dw13_int = gmm_wgrad_2d2d(
            bwd.dfc1_stash[:Mp], XG, offs_padded, order_2i, order_h
        )
        dw13_pair = dw13_int.view(E_local, I // 32, 2, 32, H)
        dw13 = torch.cat(
            [dw13_pair[:, :, 1], dw13_pair[:, :, 0]], dim=1
        ).reshape(E_local, 2 * I, H)                      # pt [lin | gate]
        if layer.qcfg.turboquant:
            dw13 = rotate_trailing(dw13.float(), layer.q_rot.t())
        dw13 = dw13.to(layer.w13.dtype)
        dw2 = gmm_wgrad_2d2d(DOUTG, ACTW, offs_padded, order_h, order_i).to(
            layer.w2.dtype
        )

        with _phase("comm-adj"):
            # dtw from the forward's per-copy combine staging (local, C4)
            staged = shared_pool_views(fwd)["combine_quant"][:T]   # (T, K, H) bf16
            dtw = torch.einsum(
                "th,tkh->tk", dout.float(), staged.float()
            ) / topk_weights.float().clamp_min(1e-12)

            # dx: the kernel already pushed each dispatched dx_copy row to its
            # source rank and REDG-added the top-k copies into dx_pool[src_token]
            # (in-epilogue, fc2_in_kernel_topk_reduce).  For world>1 the peer
            # red.adds are cross-rank, so fence the reads with a barrier.
            if world > 1:
                torch.cuda.synchronize()
                dist.barrier(group=group)
            dx = bwd.dx_pool[:T, 0, :].float()
            if layer.qcfg.turboquant:
                dx = rotate_trailing(dx, layer.q_rot.t())

    return dx.to(torch.bfloat16), dtw, dw13, dw2
