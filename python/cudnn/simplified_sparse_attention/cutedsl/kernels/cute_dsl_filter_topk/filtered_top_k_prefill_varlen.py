# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync

try:
    from cutlass import testing
except ImportError:  # public wheel: only needed by the __main__ benchmark path
    testing = None
from cutlass.torch import dtype as torch_dtype

from .filtered_top_k_varlen_util import (
    _RUBIN_TOPK_ARCHITECTURES,
    FilteredTopKKernelVarlen,
    auto_tma_load,
    auto_unroll_factor,
    compare_top_k_results,
    create_random_logits,
    get_topk_architecture_config,
    run_reference_top_k,
    tma_tuned_default,
)

"""
Prefill top-k kernel using the radix-based filter algorithm.

Differences from the decode variant:
- Row extents come from row_starts / row_ends tensors rather than seq_lens,
  so row_start may be non-zero.
- Always single-CTA per row (no multi-CTA / merge path).
- Always large_occupancy=True (num_rows >> num_sms in prefill).
- Outputs LOCAL indices relative to row_start, matching the CUDA kernel
  contract in IndexerTopKOp.cpp / indexerTopK.cu.
"""


class FilteredTopKKernelVarlenPrefill(FilteredTopKKernelVarlen):
    """Single-CTA large-occupancy top-k kernel for the prefill phase.

    Key differences vs FilteredTopKKernelVarlenDecode:
    - Takes row_starts / row_ends per-row tensors; row_start may be non-zero.
    - Always single-CTA (no multi-CTA / merge blocks).
    - Always 512 threads (large_occupancy path) with reduced SMEM for high
      occupancy.
    - Output indices are LOCAL (0-indexed within [row_start, row_end)), matching
      the CUDA indexer_topk_prefill convention.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        num_copy_bits: int = 256,
        return_val: bool = False,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
        target_blocks_per_sm: int = 4,
        architecture: str = "sm_100",
        unroll_factor: int = 4,
        enable_tma_load: bool = False,
        tma_num_stages: int = 4,
        enable_tma_load_p3: bool = False,
        tma_num_stages_p3: int = 2,
        in_kernel_sort: bool = False,
        ssa_epilogue: bool = False,
        ssa_tri_input: bool = False,
        num_threads: int = 512,
        reg_cache: bool = False,
        boundary_classify: bool = False,
        reg_select: bool = False,
        subtract_row_start: bool = True,
        use_external_hist: bool = False,
    ):
        self._target_blocks_per_sm = target_blocks_per_sm
        # async-TMA enable is a plain on/off flag here (default off). The tuned
        # auto-default (fp32 on Rubin at large N, num_cols/policy-gated) lives solely in
        # _prepare_topk_prefill, which passes a concrete flag -- constructing this class
        # directly requires an explicit opt-in.
        super().__init__(
            dtype,
            max_num_cols,
            top_k,
            num_copy_bits,
            return_val,
            enable_multi_cta=False,
            chunk_size_per_cta=16384,
            num_ctas_per_row=1,
            merge_blocks=False,
            overflow_policy=overflow_policy,
            # Large-occupancy launch is (512 threads/block, target_blocks_per_sm=4).
            # The SMEM candidate buffer S is sized ~128KB // blocks_per_sm, so at
            # large max_num_cols this S can be too small and the threshold bin
            # overflows it. Switching to (1024 threads, 2 blocks/SM) keeps
            # threads_per_sm (=2048) and bytes-in-flight unchanged but doubles S
            # (128KB // 2), reducing overflow -- a tuning lever for large-N shapes.
            # Default (512thr, 4 CTA/SM) is the tuned large-occupancy launch. The
            # (1024thr, 2 CTA/SM) large-N variant keeps threads/SM = 2048 but
            # halves concurrent rows (L2 working set) and doubles the candidate
            # buffer S -- pushes the L2-reuse cliff from 64K to 128K cols and the
            # REREAD onset out by 2x.
            num_threads_override=num_threads,
            cache_smem_values=cache_smem_values,
            architecture=architecture,
            unroll_factor=unroll_factor,
            enable_tma_load=enable_tma_load,
            tma_num_stages=tma_num_stages,
            enable_tma_load_p3=enable_tma_load_p3,
            tma_num_stages_p3=tma_num_stages_p3,
            in_kernel_sort=in_kernel_sort,
            ssa_epilogue=ssa_epilogue,
            ssa_tri_input=ssa_tri_input,
            reg_cache=reg_cache,
            boundary_classify=boundary_classify,
            reg_select=reg_select,
            use_external_hist=use_external_hist,
        )

        # Resolve the "auto" (0) unroll sentinel now that num_threads_per_cta
        # (=512) and vec_size are set, before the kernel body reads
        # self.unroll_factor. Prefill always runs the large-occupancy path, so
        # blocks/SM == target_blocks_per_sm.
        if self.unroll_factor == 0:
            self.unroll_factor = auto_unroll_factor(
                self.max_num_cols,
                self.num_threads_per_cta,
                target_blocks_per_sm,
                self.vec_size,
                self.dtype.width // 8,
            )

        # Output local indices (subtract row_start) by default; the SSA THD
        # contract wants GLOBAL key coordinates -> subtract_row_start=False.
        self.subtract_row_start_on_output = subtract_row_start

    def _compute_smem_input_size(self) -> int:
        return self._compute_smem_input_size_for_occupancy(target_blocks_per_sm=self._target_blocks_per_sm)

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        extra_buffer: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        output_probs=None,
        tma_atom=None,
        tma_tensor=None,
        hist_gmem=None,
    ):
        """CuTe DSL top-k kernel for the prefill phase."""
        cute.arch.griddepcontrol_wait()

        smem = cutlass.memory.SmemAllocator()
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.radix + 1), order=(0)),
            byte_alignment=128,
        )
        s_counter = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_threshold_bin_id = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0)),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
            g_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2), order=(0)),
                byte_alignment=128,
            )
        else:
            g_num_input = None
        s_indices = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout((self.filtered_topk_max_k,), order=(0)),
            byte_alignment=128,
        )
        s_prob_red = None
        if cutlass.const_expr(self.ssa_epilogue):
            s_prob_red = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_ordered_layout((32,), order=(0)),
                byte_alignment=128,
            )
        # Candidate buffer + async-TMA staging. Each TMA ring (p1 histogram, p3 filter)
        # needs SMEM; instead of a dedicated buffer (which would shrink the candidate
        # capacity S), it ALIASES a candidate region that is idle while that phase runs
        # -> zero extra SMEM. The ring must fit its aliased region
        # (n_stages * tile_bytes <= region bytes); that is guaranteed upstream (__init__
        # asserts capacity, adaptively caps n_stages, and reserves the mbarriers), so no
        # runtime check is needed here. Three cases:
        #   p1                : whole candidate buffer is unused during the histogram
        #                       (no candidate written yet) -> aliased below.
        #   p3 fp32 (nbuf==2) : buffer 1 is idle during the filter (candidates go to
        #                       buffer 0) -> alias buffer 1 (this branch).
        #   p3 fp16/bf16 (1)  : single buffer, nothing idle -> dedicated carve (else).
        if cutlass.const_expr(self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 2):
            # Buffer-major layout: one contiguous Int32 pool split into `n_buffers`
            # equal, 128B-padded slots (so buffer 1 is one contiguous region the p3
            # staging can alias). Each slot holds [idx (+ val when caching)]; idx/val
            # are (n_buffers, capacity) strided views into the slots, so kernel access
            # is unchanged.
            n_buffers = self.num_buffer_smem_input_idx
            capacity = self.filtered_topk_smem_input_size  # candidate slots / buffer
            idx_bytes = self.index_type.width // 8
            val_bytes = self.ordered_type.width // 8 if self.cache_smem_values else 0
            slot_bytes = ((capacity * (idx_bytes + val_bytes) + 127) // 128) * 128
            int32_bytes = 4  # pool element size; byte offsets below are / int32_bytes

            pool = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout(((n_buffers * slot_bytes) // int32_bytes,), order=(0,)),
                byte_alignment=128,
            )
            base = pool.iterator
            # idx double-buffer: buffer b at byte offset b * slot_bytes.
            s_input_idx = cute.make_tensor(
                cute.recast_ptr(base, dtype=self.index_type),
                cute.make_layout((n_buffers, capacity), stride=(slot_bytes // idx_bytes, 1)),
            )
            if cutlass.const_expr(self.cache_smem_values):
                # val double-buffer, placed right after idx inside each slot.
                s_input_val = cute.make_tensor(
                    cute.recast_ptr(
                        base + (capacity * idx_bytes) // int32_bytes,
                        dtype=self.ordered_type,
                    ),
                    cute.make_layout((n_buffers, capacity), stride=(slot_bytes // val_bytes, 1)),
                )
            else:
                s_input_val = None
            # p3 staging ring aliases buffer 1 (byte offset slot_bytes).
            tile_cols = self.num_threads_per_cta * self.vec_size
            s_tma_stage_p3 = cute.make_tensor(
                cute.recast_ptr(base + slot_bytes // int32_bytes, dtype=self.dtype),
                cute.make_ordered_layout((self.tma_num_stages_p3, 1, tile_cols), order=(2, 1, 0)),
            )
            s_tma_mbar_p3 = smem.allocate_array(cutlass.Int64, 2 * self.tma_num_stages_p3)
        else:
            # fp16/bf16 p3: dedicated carve (no idle buffer to alias, reserved in
            # sizing so S shrinks); or p3 off. Either way the candidate buffer keeps
            # its plain flat (nbuf,S) layout.
            if cutlass.const_expr(self.enable_tma_load_p3):
                tile_cols = self.num_threads_per_cta * self.vec_size
                s_tma_stage_p3 = smem.allocate_tensor(
                    element_type=self.dtype,
                    layout=cute.make_ordered_layout((self.tma_num_stages_p3, 1, tile_cols), order=(2, 1, 0)),
                    byte_alignment=128,
                )
                s_tma_mbar_p3 = smem.allocate_array(cutlass.Int64, 2 * self.tma_num_stages_p3)
            else:
                s_tma_stage_p3 = None
                s_tma_mbar_p3 = None
            if cutlass.const_expr(not self.enable_reread_always):
                s_input_idx = smem.allocate_tensor(
                    element_type=self.index_type,
                    layout=cute.make_ordered_layout(
                        (
                            self.num_buffer_smem_input_idx,
                            self.filtered_topk_smem_input_size,
                        ),
                        order=(1, 0),
                    ),
                    byte_alignment=128,
                )
            else:
                s_input_idx = None
            if cutlass.const_expr(self.cache_smem_values and not self.enable_reread_always):
                s_input_val = smem.allocate_tensor(
                    element_type=self.ordered_type,
                    layout=cute.make_ordered_layout(
                        (
                            self.num_buffer_smem_input_idx,
                            self.filtered_topk_smem_input_size,
                        ),
                        order=(1, 0),
                    ),
                    byte_alignment=128,
                )
            else:
                s_input_val = None
        if cutlass.const_expr(self.enable_reread or self.enable_bounded_spill):
            s_overflow_flag = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
        else:
            s_overflow_flag = None
        s_last_remain = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=128,
        )
        num_warps = cutlass.const_expr(min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32))
        s_warp_sums = smem.allocate_tensor(
            element_type=cute.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )

        # p1 staging ring aliases the (idle) candidate buffer s_input_idx -- see the
        # alias note above. Zero extra SMEM; S unchanged. Each stage holds one scan tile
        # (num_threads * vec_size cols); tma_num_stages = ring depth / bytes in flight.
        # p1 ends with a barrier before p3 first writes s_input_idx, so the aliased
        # lifetimes do not overlap.
        if cutlass.const_expr(self.enable_tma_load):
            tile_cols = self.num_threads_per_cta * self.vec_size
            _stage_ptr = cute.recast_ptr(s_input_idx.iterator, dtype=self.dtype)
            s_tma_stage = cute.make_tensor(
                _stage_ptr,
                cute.make_ordered_layout((self.tma_num_stages, 1, tile_cols), order=(2, 1, 0)),
            )
            # PipelineTmaAsync barrier storage: 2 mbarriers/stage (full+empty).
            s_tma_mbar = smem.allocate_array(cutlass.Int64, 2 * self.tma_num_stages)
        else:
            s_tma_stage = None
            s_tma_mbar = None

        bidx, _, _ = cute.arch.block_idx()
        row_start = cutlass.Int32(row_starts[bidx])
        row_end = cutlass.Int32(row_ends[bidx])
        length = row_end - row_start

        self.filtered_topk_kernel_per_row(
            input,
            output_indices,  # dummy: input_indices unused when merge_blocks=False
            extra_buffer,
            output_indices,
            output_values,
            row_start,
            length,
            bidx,
            s_histogram,
            s_counter,
            s_threshold_bin_id,
            s_num_input,
            g_num_input,
            s_indices,
            s_input_idx,
            s_input_val,
            s_last_remain,
            num_warps,
            s_warp_sums,
            s_overflow_flag,
            tma_atom=tma_atom,
            tma_tensor=tma_tensor,
            s_tma_stage=s_tma_stage,
            s_tma_mbar=s_tma_mbar,
            s_tma_stage_p3=s_tma_stage_p3,
            s_tma_mbar_p3=s_tma_mbar_p3,
            output_probs=output_probs,
            s_prob_red=s_prob_red,
            hist_gmem=hist_gmem,
        )

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        input_values,
        row_starts,
        row_ends,
        extra_buffer,
        output_indices,
        output_values,
        output_probs,
        external_hist,
        stream: cuda.CUstream,
        min_blocks_per_mp: cutlass.Constexpr[int] = 4,
    ):
        """Host function: launch one CTA per row."""
        if cutlass.const_expr(self.ssa_tri_input):
            # tri-layout input is flat 1D; the grid comes from the row tables
            num_rows = row_starts.shape[0]
        else:
            num_rows = input_values.shape[0]
        # Stage 0: build the async-TMA (cp.async.bulk / UTMALDG) load atom on the
        # 2D input tensor. The per-CTA tile is one coarse-scan iteration's worth of
        # columns (num_threads * vec_size), loaded [1, tile_cols] at runtime coord
        # (row, col_tile). Only the aligned column region is TMA-fed; the scalar
        # prologue/left tails keep the LDG path. Off by default (baseline untouched).
        if cutlass.const_expr(self.enable_tma_load or self.enable_tma_load_p3):
            tile_cols = self.num_threads_per_cta * self.vec_size
            tma_smem_layout = cute.make_layout((1, tile_cols), stride=(tile_cols, 1))
            tma_cta_tiler = cute.product_each(tma_smem_layout.shape)
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                input_values,
                tma_smem_layout,
                tma_cta_tiler,
            )
        else:
            tma_atom, tma_tensor = None, None
        self.filtered_topk_kernel(
            input_values,
            row_starts,
            row_ends,
            extra_buffer,
            output_indices,
            output_values,
            output_probs,
            tma_atom,
            tma_tensor,
            external_hist,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.num_threads_per_cta, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=min_blocks_per_mp,
        )


def _next_positive_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _bucket_num_cols(num_cols: int) -> int:
    return _next_positive_power_of_2(num_cols)


compiled_filter_topk_prefill_dict = {}


class _TopKPrefillLaunch:
    """A compiled prefill top-k launch and its runtime workspace."""

    def __init__(
        self,
        launcher,
        arguments: testing.JitArguments,
        output_indices: torch.Tensor,
        output_values: torch.Tensor | None,
        workspace_tensors: tuple[torch.Tensor | None, ...],
    ):
        self.launcher = launcher
        self.arguments = arguments
        self.output_indices = output_indices
        self.output_values = output_values
        self.workspace_bytes = sum(tensor.numel() * tensor.element_size() for tensor in workspace_tensors if tensor is not None)

    def run(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.launcher(*self.arguments.args, **self.arguments.kwargs)
        return self.output_indices, self.output_values


def _prepare_topk_prefill(
    input_values,
    row_starts,
    row_ends,
    top_k,
    return_val=False,
    num_copy_bits=256,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
    in_kernel_sort: bool = False,
    ssa_epilogue: bool = False,
    ssa_tri_input: bool = False,
    logical_num_cols: int | None = None,
    out_indices=None,
    out_probs=None,
    num_threads: int | None = None,
    blocks_per_sm: int | None = None,
    reg_cache: bool | None = None,
    boundary_classify: bool | None = None,
    reg_select: bool | None = None,
    subtract_row_start: bool = True,
    external_hist=None,
):
    torch_dtype_ = input_values.dtype
    dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype_]
    if ssa_tri_input:
        # input is SSA's flat triangular row-contiguous workspace; the
        # logical row width must be supplied explicitly.
        assert logical_num_cols is not None
        num_rows = row_starts.shape[0]
        num_cols = logical_num_cols
    else:
        num_rows, num_cols = input_values.shape
    bucketed_num_cols = _bucket_num_cols(num_cols)
    # BOUNDED_SPILL: per-row GMEM candidate capacity G (the extra_buffer's last
    # dim; the kernel reads it as buffer.shape[1] for the tier-2 bound). The host
    # caps G by an explicit per-row count and/or a total-byte budget (the stricter
    # wins), keeping the scratch bounded instead of GMEM_SPILL's O(num_cols).
    _bs_bufnum = 2 if dtype == cutlass.Float32 else 1
    _bs_slot_bytes = 4  # int32 index per candidate slot
    spill_g = num_cols
    if spill_capacity is not None:
        # Fail fast on a degenerate cap instead of silently clamping: a 0/negative
        # per-row capacity is meaningless (use overflow_policy=REREAD for no GMEM
        # tier), and a negative torch extent would crash the allocation anyway.
        assert spill_capacity >= 1, f"spill_capacity must be >= 1, got {spill_capacity}; use " "overflow_policy=REREAD for no GMEM spill tier."
        spill_g = min(spill_g, int(spill_capacity))
    if spill_budget_bytes is not None:
        _bytes_per_g = num_rows * _bs_bufnum * _bs_slot_bytes
        assert spill_budget_bytes >= _bytes_per_g, (
            f"spill_budget_bytes ({spill_budget_bytes}) too small to fit even one " f"candidate per row; need >= {_bytes_per_g} bytes."
        )
        spill_g = min(spill_g, int(spill_budget_bytes) // _bytes_per_g)
    architecture, min_blocks_per_mp = get_topk_architecture_config()

    # ---- tuned auto defaults (GB300-measured, fp32 Blackwell prefill) ----
    # Explicit arguments always win; None picks the measured-best config by
    # column count (8192 causal rows, K=512 reference numbers):
    #   cols <= 8192  : reg_cache + boundary_classify
    #                   (0.145/0.194 ms unsorted/sorted vs 0.151/0.200 base)
    #   cols <  65536 : boundary_classify only (rows exceed the register cap)
    #   cols >= 65536 : boundary_classify + 1024thr/2CTA -- halves concurrent
    #                   rows so the filter pass keeps L2 reuse (1.35x at 64K)
    #                   and doubles the candidate buffer (REREAD onset ~2x
    #                   later).
    # reg_select ties reg_cache+boundary_classify at <= 8K and stays an
    # explicit opt-in (newer code path). fp16/bf16 and Rubin keep upstream
    # defaults (the register/boundary paths are fp32 + sm_100 tuned).
    _auto_ok = dtype == cutlass.Float32 and (architecture not in _RUBIN_TOPK_ARCHITECTURES)
    if external_hist is not None:
        # p1 is replaced by the producer-built histogram: the scan-replacement
        # knobs (reg_cache fills its p3 fragments during p1; p1 TMA has no scan
        # to feed) are forced off. reg_select stays a hard error (asserted in
        # the kernel class).
        assert external_hist.dtype == torch.int32 and external_hist.is_contiguous()
        assert external_hist.shape[0] >= num_rows and external_hist.shape[1] == 256
        reg_cache = False
        enable_tma_load = False
    if _auto_ok and 49152 <= num_cols < 131072 and num_threads is None and blocks_per_sm is None:
        # the 1024thr/2CTA shape only pays inside the L2-cliff window
        # [48K, 128K) -- the 512thr cliff onsets between 40K and 48K
        # (probe 2026-09-01: 40K 512t wins 0.63 vs 0.70; 48K 1024t wins
        # 0.84 vs 0.91; 56K tie). halved concurrent rows keep the filter pass's
        # inter-pass working set L2-resident. From 128K nothing fits L2
        # anyway and the doubled residency of 512thr/4CTA wins again
        # (matrix sweep 2026-09-01: 128K 2.44 vs 2.62 ms, 256K 5.59 vs
        # 5.86; TMA p3 staging hurt everywhere, p1 helps only >= 256K).
        num_threads = 1024
        blocks_per_sm = 2
    if _auto_ok and num_cols >= 262144 and enable_tma_load is None and overflow_policy != "REREAD_ALWAYS":
        # async-TMA p1 histogram scan: +12% at 256K on the 512thr shape
        # (5.00 vs 5.59 ms), negative at 128K -- gate on 256K columns.
        enable_tma_load = True
    if reg_cache is None:
        reg_cache = _auto_ok and num_cols <= 8192 and num_threads in (None, 512)
    if boundary_classify is None:
        boundary_classify = _auto_ok
    if reg_select is None:
        reg_select = False
    if reg_select:
        reg_cache = False  # reg_select subsumes reg_cache (asserted in init)

    # Launch-shape overrides (None == arch-tuned default, i.e. 512thr/4 CTA on
    # Blackwell). threads * blocks must keep threads/SM within the SM limit.
    if blocks_per_sm is not None:
        min_blocks_per_mp = blocks_per_sm
    if num_threads is None:
        num_threads = 512

    # Arch-gated tuned defaults (None == "use the tuned default for this arch").
    # Rubin: SMEM value caching on + bytes-in-flight auto unroll (both validated
    # on sm_107). Blackwell: baseline behaviour (caching off, no unroll) since
    # these knobs are new and unprofiled on sm_100 -- keeps SM100 non-regressing.
    is_rubin = architecture in _RUBIN_TOPK_ARCHITECTURES
    if cache_smem_values is None:
        # Rubin tuned default, plus GB300 large columns: caching the ordered
        # values beside the indices spares the refine rounds their per-
        # candidate score[idx] gathers, which outweighs the halved candidate
        # buffer from 128K up (measured +3-4% at 128K/192K/256K; neutral-to-
        # negative below, where overflow would newly engage).
        cache_smem_values = is_rubin or (_auto_ok and num_cols >= 131072)
    if unroll_factor is None:
        unroll_factor = 0 if is_rubin else 1

    # async-TMA enable: an explicit enable_tma_load(_p3) True/False wins; None uses the
    # auto default. The auto default (tuned: fp32 on Rubin at large N) only fires when
    # the ACTUAL num_cols meets the TMA row-stride alignment -- divisible by
    # 16 // dtype_bytes (fp32 -> 4, fp16/bf16 -> 8) -- so non-conforming shapes fall back
    # to LDG instead of hitting the descriptor's divisibility ValueError; REREAD_ALWAYS
    # is excluded (no candidate buffer to alias). An explicit True bypasses those gates
    # and hits the hard contract if the shape does not conform. The resolved flags go
    # into the compile key so the same bucketed shape with different divisibility /
    # override does not collide.
    _tma_div = 16 // (dtype.width // 8)
    _auto_tma = tma_tuned_default(dtype, architecture, bucketed_num_cols) and (num_cols % _tma_div == 0) and overflow_policy != "REREAD_ALWAYS"
    enable_tma_load = auto_tma_load(enable_tma_load, _auto_tma)
    enable_tma_load_p3 = auto_tma_load(enable_tma_load_p3, _auto_tma)
    if reg_cache or reg_select:
        # reg_cache/reg_select replace the gmem scans; TMA staging is mutually
        # exclusive (kernel-level assert), so drop any auto-resolved TMA enables.
        enable_tma_load = False
        enable_tma_load_p3 = False

    key = (
        dtype,
        bucketed_num_cols,
        top_k,
        return_val,
        num_copy_bits,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
        enable_tma_load,
        enable_tma_load_p3,
        in_kernel_sort,
        ssa_epilogue,
        ssa_tri_input,
        num_threads,
        min_blocks_per_mp,
        reg_cache,
        boundary_classify,
        reg_select,
        subtract_row_start,
        external_hist is not None,
    )
    if key not in compiled_filter_topk_prefill_dict:
        n_rows = cute.sym_int()
        _tma_on = enable_tma_load or enable_tma_load_p3
        n_cols = cute.sym_int(divisibility=_tma_div) if _tma_on else cute.sym_int()
        if ssa_tri_input:
            input_fake = cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), stride_order=(0,), assumed_align=32)
        else:
            input_fake = cute.runtime.make_fake_compact_tensor(dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=32)
        row_starts_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_rows,), stride_order=(0,))
        row_ends_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_rows,), stride_order=(0,))
        if overflow_policy == "GMEM_SPILL":
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, cute.sym_int(), n_cols),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
        elif overflow_policy == "BOUNDED_SPILL":
            # Last dim is a fresh sym_int (dynamic G) so one compiled kernel
            # serves any spill_capacity; the runtime extent = buffer.shape[1].
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (n_rows, cute.sym_int(), cute.sym_int()),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
        else:
            buffer_fake = None
        output_indices_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_rows, top_k), stride_order=(1, 0))
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(dtype, (n_rows, top_k), stride_order=(1, 0))
        else:
            output_values_fake = None
        if ssa_epilogue:
            output_probs_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n_rows, top_k), stride_order=(1, 0))
        else:
            output_probs_fake = None
        if external_hist is not None:
            hist_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 256), stride_order=(1, 0))
        else:
            hist_fake = None
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        filtered_topk_func = FilteredTopKKernelVarlenPrefill(
            dtype,
            bucketed_num_cols,
            top_k,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            in_kernel_sort=in_kernel_sort,
            ssa_epilogue=ssa_epilogue,
            ssa_tri_input=ssa_tri_input,
            num_threads=num_threads,
            reg_cache=reg_cache,
            boundary_classify=boundary_classify,
            reg_select=reg_select,
            subtract_row_start=subtract_row_start,
            target_blocks_per_sm=min_blocks_per_mp,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            architecture=architecture,
            unroll_factor=unroll_factor,
            # num_cols-divisibility-gated resolution from above (explicit so the
            # kernel does not re-resolve the tuned default from bucketed_num_cols,
            # which cannot see the actual num_cols alignment).
            enable_tma_load=enable_tma_load,
            enable_tma_load_p3=enable_tma_load_p3,
            use_external_hist=external_hist is not None,
        )
        compiled_kernel = cute.compile(
            filtered_topk_func,
            input_fake,
            row_starts_fake,
            row_ends_fake,
            buffer_fake,
            output_indices_fake,
            output_values_fake,
            output_probs_fake,
            hist_fake,
            stream=fake_stream,
            min_blocks_per_mp=min_blocks_per_mp,
            options="--enable-tvm-ffi",
        )
        compiled_filter_topk_prefill_dict[key] = compiled_kernel
    else:
        compiled_kernel = compiled_filter_topk_prefill_dict[key]

    if out_indices is not None:
        output_indices_torch = out_indices
    else:
        output_indices_torch = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(num_rows, top_k, dtype=torch_dtype_, device="cuda")
    else:
        output_values_torch = None
    if ssa_epilogue:
        if out_probs is not None:
            output_probs_torch = out_probs
        else:
            output_probs_torch = torch.empty(num_rows, top_k, dtype=torch.float32, device="cuda")
    else:
        output_probs_torch = None

    if overflow_policy == "GMEM_SPILL":
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_torch = torch.empty(num_rows, buffer_numbers, num_cols, dtype=torch.int32, device="cuda")
    elif overflow_policy == "BOUNDED_SPILL":
        buffer_torch = torch.empty(num_rows, _bs_bufnum, spill_g, dtype=torch.int32, device="cuda")
    else:
        buffer_torch = None

    return _TopKPrefillLaunch(
        compiled_kernel,
        testing.JitArguments(
            input_values,
            row_starts,
            row_ends,
            buffer_torch,
            output_indices_torch,
            output_values_torch,
            output_probs_torch,
            external_hist,
        ),
        output_indices_torch,
        output_values_torch,
        (
            input_values,
            row_starts,
            row_ends,
            buffer_torch,
            output_indices_torch,
            output_values_torch,
            output_probs_torch,
            external_hist,
        ),
    )


# Prepared-launch memo: the python prepare path (arg validation, auto-knob
# resolution, JitArguments construction) costs ~23-36us per call (CUPTI-
# measured), ~14% of the whole 8K top-k. Replay the prepared launch when the
# caller passes the SAME tensors and knobs. Enabled only when the caller
# provides the output tensors (otherwise replay would alias the internally
# allocated outputs across calls). Holding the launch object keeps tensor
# references alive, so a data_ptr cannot be recycled while memoized; capped
# to 8 entries (production uses 1-2).
_launch_memo = {}


def cute_dsl_topk_prefill_wrapper(
    input_values,
    row_starts,
    row_ends,
    top_k,
    return_val=False,
    num_copy_bits=256,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
    in_kernel_sort: bool = False,
    ssa_epilogue: bool = False,
    ssa_tri_input: bool = False,
    logical_num_cols: int | None = None,
    out_indices=None,
    out_probs=None,
    num_threads: int | None = None,
    blocks_per_sm: int | None = None,
    reg_cache: bool | None = None,
    boundary_classify: bool | None = None,
    reg_select: bool | None = None,
    subtract_row_start: bool = True,
    external_hist=None,
):
    """Compile and launch the prefill top-k wrapper.

    ``cache_smem_values`` / ``unroll_factor`` / ``enable_tma_load`` /
    ``enable_tma_load_p3`` default to ``None``, which selects the tuned best
    configuration for the detected architecture (Rubin: SMEM value caching +
    bytes-in-flight auto unroll; async-TMA load for fp32 at N >= 131072). Pass
    explicit values to override for tuning.
    """
    memoizable = out_indices is not None and not return_val and (not ssa_epilogue or out_probs is not None)
    if memoizable:
        sig = (
            input_values.data_ptr(),
            tuple(input_values.shape),
            row_starts.data_ptr(),
            row_ends.data_ptr(),
            out_indices.data_ptr(),
            out_probs.data_ptr() if out_probs is not None else None,
            top_k,
            num_copy_bits,
            overflow_policy,
            cache_smem_values,
            unroll_factor,
            enable_tma_load,
            enable_tma_load_p3,
            spill_capacity,
            spill_budget_bytes,
            in_kernel_sort,
            ssa_epilogue,
            ssa_tri_input,
            logical_num_cols,
            num_threads,
            blocks_per_sm,
            reg_cache,
            boundary_classify,
            reg_select,
            subtract_row_start,
            external_hist.data_ptr() if external_hist is not None else None,
        )
        hit = _launch_memo.get(sig)
        if hit is not None:
            return hit.run()
    launch = _prepare_topk_prefill(
        input_values,
        row_starts,
        row_ends,
        top_k,
        return_val,
        num_copy_bits,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
        enable_tma_load,
        enable_tma_load_p3,
        spill_capacity,
        spill_budget_bytes,
        in_kernel_sort,
        ssa_epilogue,
        ssa_tri_input,
        logical_num_cols,
        out_indices,
        out_probs,
        num_threads,
        blocks_per_sm,
        reg_cache,
        boundary_classify,
        reg_select,
        subtract_row_start,
        external_hist,
    )
    if memoizable:
        if len(_launch_memo) >= 8:
            _launch_memo.pop(next(iter(_launch_memo)))
        _launch_memo[sig] = launch
    return launch.run()


def run_filtered_topk_prefill(
    dtype: type[cutlass.Numeric],
    num_rows: int,
    max_num_cols: int,
    top_k: int,
    num_copy_bits: int = 256,
    return_val: bool = False,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    print_verbose: bool = True,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool = False,
    variable_lengths: bool = True,
    causal_batch: int = 1,
    chunk_start: int = 0,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
):
    """Prepare input tensors, launch prefill top-k kernel, and check reference."""
    if print_verbose:
        print("=" * 60)
        print("Launching Blackwell Filtered TopK Prefill Test")
        print("-" * 60)
        print(f"dtype: {dtype}")
        print(f"num_rows: {num_rows}")
        print(f"max_num_cols: {max_num_cols}")
        print(f"top_k: {top_k}")
        print(f"num_copy_bits: {num_copy_bits}")
        print(f"return_val: {return_val}")
        print(f"variable_lengths: {variable_lengths}")
        if variable_lengths:
            print(f"causal_batch: {causal_batch}")
            print(f"chunk_start: {chunk_start}")
        print(f"overflow_policy: {overflow_policy}")
        print(f"Do reference checking: {do_ref_check}")
        print(f"Do benchmark: {do_benchmark}")
        print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if variable_lengths:
        # Causal extents over `causal_batch` independent sequences laid out
        # block diagonally. Each sequence owns ctx = max_num_cols //
        # causal_batch keys and contributes rows_per_seq = num_rows //
        # causal_batch query rows starting at chunk_start, so row (s, i) spans
        # keys [s * ctx, s * ctx + chunk_start + i + 1). Extents ramp
        # chunk_start + 1 .. chunk_start + rows_per_seq, and the block offsets
        # keep the nonzero-row-start local-index contract under test.
        #
        # chunk_start == 0 with max_num_cols == num_rows is a whole
        # lower-triangular prompt, whose footprint is quadratic in the context
        # length. Otherwise this models one chunk of a chunked prefill: the row
        # count and the key extent decouple, so the footprint is only
        # num_rows * max_num_cols and long contexts stay affordable. The first
        # chunk is the one that produces extents below top_k; the last chunk
        # (chunk_start == ctx - rows_per_seq) is the heaviest.
        if num_rows % causal_batch != 0:
            raise ValueError(f"num_rows ({num_rows}) must be divisible by " f"causal_batch ({causal_batch})")
        if max_num_cols % causal_batch != 0:
            raise ValueError(f"max_num_cols ({max_num_cols}) must be divisible by " f"causal_batch ({causal_batch})")
        if chunk_start < 0:
            raise ValueError(f"chunk_start ({chunk_start}) must not be negative")
        rows_per_seq = num_rows // causal_batch
        ctx = max_num_cols // causal_batch
        if chunk_start + rows_per_seq > ctx:
            raise ValueError(f"chunk_start ({chunk_start}) plus the per-sequence row count " f"({rows_per_seq}) exceeds the per-sequence key extent ({ctx})")
        seq_offsets = torch.arange(causal_batch, dtype=torch.int32, device="cuda") * ctx
        row_starts = seq_offsets.repeat_interleave(rows_per_seq)
        within = (torch.arange(1, rows_per_seq + 1, dtype=torch.int32, device="cuda") + chunk_start).tile(causal_batch)
        row_ends = row_starts + within
    else:
        row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
        row_ends = torch.full((num_rows,), max_num_cols, dtype=torch.int32, device="cuda")

    def create_input_tensor() -> torch.Tensor:
        input_tensor = create_random_logits(
            row_starts,
            row_ends,
            torch_dtype(dtype),
            seed,
        )
        if input_tensor.shape[1] == max_num_cols:
            return input_tensor

        padded_input = torch.full(
            (num_rows, max_num_cols),
            float("-inf"),
            dtype=torch_dtype(dtype),
            device="cuda",
        )
        padded_input[:, : input_tensor.shape[1]] = input_tensor
        return padded_input

    input_torch = create_input_tensor()
    output_indices_torch, output_values_torch = cute_dsl_topk_prefill_wrapper(
        input_torch,
        row_starts,
        row_ends,
        top_k,
        return_val,
        num_copy_bits,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
        enable_tma_load,
        enable_tma_load_p3,
        spill_capacity,
        spill_budget_bytes,
    )

    if do_ref_check:
        torch.cuda.synchronize()
        torch_indices = run_reference_top_k(input_torch, row_starts, row_ends, top_k)
        assert compare_top_k_results(
            input_torch,
            output_indices_torch,
            torch_indices,
            row_starts,
            row_ends,
            top_k,
        ), "prefill wrapper results don't match torch.topk"

        if return_val:
            valid_mask = output_indices_torch != -1
            output_rows = torch.arange(num_rows, device="cuda")[:, None]
            output_rows = output_rows.expand_as(output_indices_torch)
            absolute_indices = output_indices_torch + row_starts[:, None]
            expected_values = input_torch[
                output_rows[valid_mask],
                absolute_indices[valid_mask],
            ]
            assert torch.allclose(
                output_values_torch[valid_mask],
                expected_values,
                rtol=1e-5,
                atol=1e-5,
            ), "prefill wrapper returned values that do not match its indices"
            assert torch.isneginf(output_values_torch[~valid_mask]).all(), "prefill wrapper did not write -inf for invalid outputs"

        if print_verbose:
            print("Prefill wrapper: PASSED")

    if do_benchmark:
        torch.cuda.synchronize()
        benchmark_launch = _prepare_topk_prefill(
            input_torch,
            row_starts,
            row_ends,
            top_k,
            return_val,
            num_copy_bits,
            overflow_policy,
            cache_smem_values,
            unroll_factor,
            enable_tma_load,
            enable_tma_load_p3,
            spill_capacity,
            spill_budget_bytes,
        )
        use_prepared_workspace = True

        def generate_inputs() -> testing.JitArguments:
            nonlocal use_prepared_workspace
            if use_prepared_workspace:
                use_prepared_workspace = False
                return benchmark_launch.arguments

            return _prepare_topk_prefill(
                create_input_tensor(),
                row_starts.clone(),
                row_ends.clone(),
                top_k,
                return_val,
                num_copy_bits,
                overflow_policy,
                cache_smem_values,
                unroll_factor,
                enable_tma_load,
                enable_tma_load_p3,
                spill_capacity,
                spill_budget_bytes,
            ).arguments

        workspace_count = (
            testing.get_workspace_count(
                benchmark_launch.workspace_bytes,
                warmup_iterations,
                iterations,
            )
            if use_cold_l2
            else 1
        )
        if print_verbose:
            print(f"Workspace count: {workspace_count}")
        torch_stream = torch.cuda.Stream()
        benchmark_stream = cuda.CUstream(torch_stream.cuda_stream)
        with torch.cuda.stream(torch_stream):
            time = testing.benchmark(
                benchmark_launch.launcher,
                workspace_generator=generate_inputs,
                workspace_count=workspace_count,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                use_cuda_graphs=True,
                stream=benchmark_stream,
            )
        if print_verbose:
            print(f"Time: {time} us")
        print(f"{dtype}-{num_rows}-{max_num_cols}-{top_k} {time}")
    torch.cuda.synchronize()


def run_topk_prefill(
    dtype: type[cutlass.Numeric],
    num_rows: int,
    max_num_cols: int,
    top_k: int,
    num_copy_bits: int = 256,
    return_val: bool = False,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool = False,
    variable_lengths: bool = True,
    causal_batch: int = 1,
    chunk_start: int = 0,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
):
    run_filtered_topk_prefill(
        dtype,
        num_rows,
        max_num_cols,
        top_k,
        num_copy_bits,
        return_val,
        do_ref_check,
        do_benchmark,
        warmup_iterations,
        iterations,
        use_cold_l2,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
        variable_lengths=variable_lengths,
        causal_batch=causal_batch,
        chunk_start=chunk_start,
        unroll_factor=unroll_factor,
        enable_tma_load=enable_tma_load,
        enable_tma_load_p3=enable_tma_load_p3,
        spill_capacity=spill_capacity,
        spill_budget_bytes=spill_budget_bytes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blackwell CuTE DSL filtered top-k prefill benchmark.")
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16],
        help="Data type of the input matrix",
    )
    parser.add_argument("--num_rows", type=int, default=256, help="Number of rows (sequences)")
    parser.add_argument("--max_num_cols", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--top_k", type=int, default=1024, help="top_k")
    parser.add_argument(
        "--num_copy_bits",
        type=int,
        default=256,
        help="num_copy_bits, used for vectorization",
    )
    parser.add_argument("--return_val", action="store_true", default=False, help="Return values")
    parser.add_argument(
        "--do_ref_check",
        action="store_true",
        default=False,
        help="Do reference checking",
    )
    parser.add_argument(
        "--benchmark",
        "--do_benchmark",
        dest="do_benchmark",
        action="store_true",
        default=False,
        help="Run the benchmark",
    )
    parser.add_argument(
        "--warmup",
        "--warmup_iterations",
        dest="warmup_iterations",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument(
        "--no-cold-l2",
        dest="use_cold_l2",
        action="store_false",
        help="Disable cold L2 cache simulation",
    )
    parser.add_argument(
        "--use_cold_l2",
        dest="use_cold_l2",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_cold_l2=True)
    parser.add_argument(
        "--overflow_policy",
        type=str,
        default="REREAD",
        choices=["GMEM_SPILL", "TRUNCATE", "REREAD", "REREAD_ALWAYS", "BOUNDED_SPILL"],
        help="Overflow policy when candidates exceed SMEM capacity",
    )
    parser.add_argument(
        "--spill_capacity",
        type=int,
        default=None,
        help="BOUNDED_SPILL: per-row GMEM candidate capacity G (upper bound)",
    )
    parser.add_argument(
        "--spill_budget_bytes",
        type=int,
        default=None,
        help="BOUNDED_SPILL: total GMEM scratch byte budget; caps G to fit the budget",
    )
    parser.add_argument(
        "--cache_smem_values",
        action="store_true",
        default=False,
        help="Cache ordered values alongside indices in SMEM to avoid re-reading from GMEM in refinement rounds",
    )
    parser.add_argument(
        "--length_mode",
        "--length-mode",
        dest="length_mode",
        type=str,
        default="fixlen",
        choices=["fixlen", "varlen"],
        help=(
            "Row extent distribution: fixlen spans the full "
            "[0, max_num_cols) for every row; varlen builds causal "
            "chunked-prefill spans over --causal_batch sequences, whose "
            "extents ramp from --chunk_start and can fall below top_k"
        ),
    )
    parser.add_argument(
        "--causal_batch",
        "--causal-batch",
        dest="causal_batch",
        type=int,
        default=1,
        help=(
            "Number of independent sequences in causal varlen mode. Each owns "
            "max_num_cols // causal_batch keys and contributes "
            "num_rows // causal_batch rows; both must divide evenly"
        ),
    )
    parser.add_argument(
        "--chunk_start",
        "--chunk-start",
        dest="chunk_start",
        type=int,
        default=0,
        help=(
            "Query offset of this chunk within each sequence, for modelling "
            "chunked prefill. 0 is the first chunk (extents ramp from 1); the "
            "last chunk starts at max_num_cols//causal_batch - "
            "num_rows//causal_batch"
        ),
    )

    def _parse_unroll_factor(value):
        # "auto" -> None: pick the arch-tuned default (Rubin: bytes-in-flight
        # auto; Blackwell: baseline uf=1). Otherwise an explicit 1/2/4/8.
        if value == "auto":
            return None
        try:
            iv = int(value)
        except ValueError:
            iv = None
        if iv not in (1, 2, 4, 8):
            raise argparse.ArgumentTypeError(f"--unroll_factor must be 'auto' or one of 1,2,4,8; got {value!r}")
        return iv

    parser.add_argument(
        "--unroll_factor",
        "--unroll-factor",
        dest="unroll_factor",
        type=_parse_unroll_factor,
        default=None,
        metavar="{auto,1,2,4,8}",
        help=(
            "load-instruction unroll factor for the coarse/filter input scans. Default 'auto' "
            "picks the arch-tuned factor (bytes-in-flight on Rubin, baseline uf=1 "
            "on Blackwell). 1 = original baseline (1-in-flight while); 2/4/8 "
            "overlap that many loads for memory-level parallelism"
        ),
    )

    def _parse_tma(value):
        # auto -> None (tuned default: async-TMA for fp32 at N>=131072); on/off force.
        return {"auto": None, "on": True, "off": False}[value]

    parser.add_argument(
        "--enable_tma_load",
        "--enable-tma-load",
        dest="enable_tma_load",
        type=_parse_tma,
        default=None,
        metavar="{auto,on,off}",
        help="p1 async-TMA coarse-histogram load. auto = tuned default.",
    )
    parser.add_argument(
        "--enable_tma_load_p3",
        "--enable-tma-load-p3",
        dest="enable_tma_load_p3",
        type=_parse_tma,
        default=None,
        metavar="{auto,on,off}",
        help="p3 async-TMA coarse-filter load. auto = tuned default.",
    )

    args = parser.parse_args()
    run_topk_prefill(
        variable_lengths=(args.length_mode == "varlen"),
        causal_batch=args.causal_batch,
        chunk_start=args.chunk_start,
        dtype=args.dtype,
        num_rows=args.num_rows,
        max_num_cols=args.max_num_cols,
        top_k=args.top_k,
        num_copy_bits=args.num_copy_bits,
        return_val=args.return_val,
        do_ref_check=args.do_ref_check,
        do_benchmark=args.do_benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
        overflow_policy=args.overflow_policy,
        spill_capacity=args.spill_capacity,
        spill_budget_bytes=args.spill_budget_bytes,
        cache_smem_values=args.cache_smem_values,
        unroll_factor=args.unroll_factor,
        enable_tma_load=args.enable_tma_load,
        enable_tma_load_p3=args.enable_tma_load_p3,
    )
