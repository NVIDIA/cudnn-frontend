# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM100 CuTe DSL indexer top-K decode kernel."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.utils.distributed import atomicAdd

from .block_scan import block_prefix_sum_kernel
from .indexer_top_k_varlen_util import IndexerTopKKernelVarlen


class ComputeDynamicCTAOffsets:
    """CuTE DSL kernel to compute row_cta_offsets and row_output_offsets.

    Replaces ~20 small PyTorch tensor ops (arange, indexing, arithmetic,
    cumsum, etc.) with a single GPU kernel launch.

    Uses 512 threads: each thread computes the CTA count for one row,
    then a block-wide parallel prefix sum (via block_prefix_sum_kernel)
    produces the exclusive scan in a single pass.

    Inputs:
        seq_lens:  (batch_size,) int32
    Outputs (pre-allocated by caller):
        row_cta_offsets:    (num_rows + 1,) int32  — exclusive prefix sum of per-row CTAs
        row_output_offsets: (num_rows + 1,) int32  — exclusive prefix sum of per-row output elems
    """

    # Max supported num_rows (batch_size * next_n). Must equal NUM_THREADS
    # since each thread handles one row.
    MAX_NUM_ROWS = 512

    def __init__(self, next_n: int, chunk_size_per_cta: int, top_k: int):
        self.next_n = next_n
        self.chunk_size_per_cta = chunk_size_per_cta
        self.top_k = top_k
        self.NUM_THREADS = 512

    @cute.kernel
    def compute_offsets_kernel(
        self,
        seq_lens: cute.Tensor,
        row_cta_offsets: cute.Tensor,
        row_output_offsets: cute.Tensor,
    ):
        smem = utils.SmemAllocator()
        num_warps = cutlass.const_expr(self.NUM_THREADS // 32)
        s_warp_sums = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )

        tidx, _, _ = cute.arch.thread_idx()
        num_rows = seq_lens.shape[0] * self.next_n

        # Each thread computes CTA count for its row (0 if out of bounds)
        ctas = 0
        if tidx < num_rows:
            batch_idx = tidx // self.next_n
            next_n_off = tidx % self.next_n
            eff_len = seq_lens[batch_idx] - self.next_n + next_n_off + 1
            ctas = (eff_len + self.chunk_size_per_cta - 1) // self.chunk_size_per_cta
            if ctas < 1:
                ctas = 1

        # Block-wide inclusive prefix sum
        prefix_ctas, _ = block_prefix_sum_kernel(ctas, s_warp_sums, tidx, self.NUM_THREADS, num_warps, barrier_id=1)

        # Write exclusive prefix sum (shifted by 1)
        if tidx == 0:
            row_cta_offsets[0] = 0
            row_output_offsets[0] = 0
        if tidx < num_rows:
            row_cta_offsets[tidx + 1] = prefix_ctas
            row_output_offsets[tidx + 1] = prefix_ctas * self.top_k

    @cute.jit
    def __call__(
        self,
        seq_lens,
        row_cta_offsets,
        row_output_offsets,
        stream: cuda.CUstream,
    ):
        self.compute_offsets_kernel(
            seq_lens,
            row_cta_offsets,
            row_output_offsets,
        ).launch(
            grid=(1, 1, 1),
            block=(self.NUM_THREADS, 1, 1),
            stream=stream,
        )


class IndexerTopKKernelVarlenDecode(IndexerTopKKernelVarlen):
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        return_val: bool = True,
        large_occupancy: bool = False,
        # for multi-cta version.
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
        enable_dynamic_multi_cta: bool = False,
        varlen_merge_input: bool = False,
        num_sms: int = 148,
        debug: bool = False,
    ):
        super().__init__(
            dtype,
            max_num_cols,
            top_k,
            num_copy_bits,
            return_val,
            enable_multi_cta,
            chunk_size_per_cta,
            num_ctas_per_row,
            merge_blocks,
        )
        self.next_n = next_n
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.merge_blocks = merge_blocks
        self.num_ctas_per_row = num_ctas_per_row
        self.enable_dynamic_multi_cta = enable_dynamic_multi_cta
        self.varlen_merge_input = varlen_merge_input
        self.num_sms = num_sms

        if cutlass.const_expr(large_occupancy):
            # tuned value, could be tuned further.
            # reduce the smem usage and improve occupancy.
            if self.max_num_cols >= 262144:
                self.indexer_topk_smem_input_size = 4096
            elif self.max_num_cols >= 131072:
                self.indexer_topk_smem_input_size = 3072
            elif self.max_num_cols >= 65536:
                self.indexer_topk_smem_input_size = 2048
            elif self.max_num_cols >= 32768:
                self.indexer_topk_smem_input_size = 1024
            elif self.max_num_cols >= 16384:
                self.indexer_topk_smem_input_size = 1024
            elif self.max_num_cols >= 8192:
                self.indexer_topk_smem_input_size = 512
            else:
                self.indexer_topk_smem_input_size = 256

            if cutlass.const_expr(self.max_num_cols > self.indexer_topk_smem_input_size):
                self.enable_gmem_store = True
            else:
                self.enable_gmem_store = False

            # set the number of threads per cta to 512.
            if cutlass.const_expr(not self.merge_blocks):
                self.num_threads_per_cta = 512
            else:
                # For merge_blocks, cap num_threads_per_cta so that the tile
                # width (num_threads_per_cta * vec_size) does not exceed
                # max_num_cols. Otherwise, out-of-bounds padding elements
                # created by _fill_oob are counted in the radix histogram
                # and may be selected as top-k candidates with invalid
                # indices, causing incorrect results.
                self.num_threads_per_cta = min(self.max_num_cols // self.vec_size, 512)

        # only used for debug info
        if cutlass.const_expr(debug):
            print(f"dtype: {self.dtype}, vec_size: {self.vec_size}")
            print(f"max_num_cols: {self.max_num_cols}, num_threads_per_cta: {self.num_threads_per_cta}")
            print(f"indexer_topk_smem_input_size: {self.indexer_topk_smem_input_size}")
            print(f"enable_gmem_store: {self.enable_gmem_store}")
            print(f"return_val: {self.return_val}")
            print(f"large_occupancy: {large_occupancy}")
            print(f"indexer_topk_smem_input_size: {self.indexer_topk_smem_input_size}")
            print(f"first_refine_shift: {self.first_refine_shift}, num_refine_rounds: {self.num_refine_rounds}")

    @cute.jit
    def run_kernel(
        self,
        input,
        indices,
        extra_buffer,
        output_indices,
        output_values,
        tiler_mn,
        copy_atom,
        tiled_copy,
        seqlen,
        task_id,
        s_histogram,
        s_counter,
        s_threshold_bin_id,
        s_num_input,
        g_num_input,
        s_indices,
        s_input_idx,
        s_last_remain,
        num_warps,
        s_warp_sums,
    ):
        # TODO: update row_start to align with multi-cta version.
        row_start = 0
        seq_len = seqlen[task_id // self.next_n]
        row_end = seq_len - self.next_n + (task_id % self.next_n) + 1

        length = row_end - row_start

        self.indexer_topk_kernel_per_row(
            input,
            indices,
            extra_buffer,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
            row_start,
            length,
            task_id,
            s_histogram,
            s_counter,
            s_threshold_bin_id,
            s_num_input,
            g_num_input,
            s_indices,
            s_input_idx,
            s_last_remain,
            num_warps,
            s_warp_sums,
        )

    @cute.kernel
    def indexer_topk_kernel(
        self,
        input: cute.Tensor,
        indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        g_global_counter: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tiler_mn: cute.Shape,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        enable_persistent_dynamic_scheduling: cutlass.Constexpr[bool] = False,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        smem = utils.SmemAllocator()
        # TODO: how to simplify the smem allocate codes?
        s_histogram_buf_layout = cute.make_ordered_layout((self.radix + 1), order=(0))
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=s_histogram_buf_layout,
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
        if cutlass.const_expr(self.enable_gmem_store):
            g_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2), order=(0)),
                byte_alignment=128,
            )
        else:
            g_num_input = None
        s_indices = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout((self.indexer_topk_max_k,), order=(0)),
            byte_alignment=128,
        )
        s_input_idx = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout(
                (
                    self.num_buffer_smem_input_idx,
                    self.indexer_topk_smem_input_size,
                ),
                order=(1, 0),
            ),
            byte_alignment=128,
        )
        s_last_remain = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        num_warps = cutlass.const_expr(min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32))
        s_warp_sums = smem.allocate_tensor(
            element_type=cute.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )

        if cutlass.const_expr(not enable_persistent_dynamic_scheduling):
            # Thread and block indexing
            bidx, bidy, _ = cute.arch.block_idx()

            if cutlass.const_expr(self.enable_dynamic_multi_cta):
                # 2D grid with early exit: bidx = row_id, bidy = chunk_id.
                # Each CTA computes how many chunks its row actually needs
                # from seqlen and exits early if bidy >= num_needed_ctas.
                # This avoids prefix sum + binary search overhead entirely.
                num_rows_val = seqlen.shape[0] * self.next_n

            row_start = 0
            row_end = 0
            length = 0
            seq_len = 0

            if not cutlass.const_expr(self.merge_blocks):
                seq_len = seqlen[bidx // self.next_n]
                row_end = seq_len - self.next_n + (bidx % self.next_n) + 1
                length = row_end - row_start

            if cutlass.const_expr(self.enable_multi_cta):
                # update row_start and row_end.
                row_start = self.chunk_size_per_cta * bidy
                row_end = min(row_end, row_start + self.chunk_size_per_cta)
                length = row_end - row_start
                output_indices = cute.flat_divide(output_indices, (1, self.top_k))[0, None, bidx, bidy]
                output_values = cute.flat_divide(output_values, (1, self.top_k))[0, None, bidx, bidy]

            if cutlass.const_expr(self.merge_blocks):
                if cutlass.const_expr(self.varlen_merge_input):
                    # Varlen merge: compute per-row valid length from seqlen.
                    _batch = bidx // self.next_n
                    _off = bidx % self.next_n
                    _eff = seqlen[_batch] - self.next_n + _off + 1
                    _num_ctas = (_eff + self.chunk_size_per_cta - 1) // self.chunk_size_per_cta
                    if _num_ctas < 1:
                        _num_ctas = 1
                    merge_width = _num_ctas * self.top_k
                    row_end = merge_width
                    length = merge_width
                else:
                    # Existing fixed-length path
                    # Note, after 1st kernel, the output is fix-lenght.
                    # Note, for merge_block kernels, need to ensure max_num_cols is the same as bucketed_num_cols.
                    row_end = self.max_num_cols
                    length = self.max_num_cols

            # Skip CTAs that exceed this row's actual chunk count.
            _should_run = True
            if cutlass.const_expr(self.enable_dynamic_multi_cta):
                _batch_check = bidx // self.next_n
                _off_check = bidx % self.next_n
                _eff_check = seqlen[_batch_check] - self.next_n + _off_check + 1
                _needed_ctas = (_eff_check + self.chunk_size_per_cta - 1) // self.chunk_size_per_cta
                if _needed_ctas < 1:
                    _needed_ctas = 1
                _should_run = (bidx < num_rows_val) and (bidy < _needed_ctas)

            if _should_run:
                self.indexer_topk_kernel_per_row(
                    input,
                    indices,
                    extra_buffer,
                    output_indices,
                    output_values,
                    tiler_mn,
                    copy_atom,
                    tiled_copy,
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
                    s_last_remain,
                    num_warps,
                    s_warp_sums,
                )
        else:
            num_rows = input.shape[0]
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            row_start = cutlass.Int32(0)
            row_end = cutlass.Int32(0)
            length = cutlass.Int32(0)
            seq_len = cutlass.Int32(0)

            # Persistent dynamic scheduler.
            # First task: use bidx directly (no atomic needed).
            # Subsequent tasks: use atomicAdd (counter pre-initialized
            # to grid_size on host, so values start from grid_size).
            s_row_id = smem.allocate_tensor(
                element_type=cute.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )

            # First task: deterministic assignment by block index.
            task_id = bidx
            if task_id < num_rows:
                row_start = 0
                seq_len = seqlen[task_id // self.next_n]
                row_end = seq_len - self.next_n + (task_id % self.next_n) + 1
                length = row_end - row_start

                self.indexer_topk_kernel_per_row(
                    input,
                    indices,
                    extra_buffer,
                    output_indices,
                    output_values,
                    tiler_mn,
                    copy_atom,
                    tiled_copy,
                    row_start,
                    length,
                    task_id,
                    s_histogram,
                    s_counter,
                    s_threshold_bin_id,
                    s_num_input,
                    g_num_input,
                    s_indices,
                    s_input_idx,
                    s_last_remain,
                    num_warps,
                    s_warp_sums,
                )

            # Subsequent tasks: dynamic work stealing via atomic counter.
            # Counter starts at 0, so offset by grid_size to skip
            # the first-round tasks already handled by bidx.
            grid_size_x, _, _ = cute.arch.grid_dim()
            work_remaining = task_id < num_rows
            while work_remaining:
                if tidx == 0:
                    s_row_id[0] = atomicAdd(g_global_counter.iterator, cutlass.Int32(1)) + grid_size_x
                cute.arch.barrier()

                row_id = s_row_id[0]
                has_work = row_id < num_rows

                if has_work:
                    task_id = row_id
                    row_start = 0
                    seq_len = seqlen[task_id // self.next_n]
                    row_end = seq_len - self.next_n + (task_id % self.next_n) + 1
                    length = row_end - row_start

                    self.indexer_topk_kernel_per_row(
                        input,
                        indices,
                        extra_buffer,
                        output_indices,
                        output_values,
                        tiler_mn,
                        copy_atom,
                        tiled_copy,
                        row_start,
                        length,
                        task_id,
                        s_histogram,
                        s_counter,
                        s_threshold_bin_id,
                        s_num_input,
                        g_num_input,
                        s_indices,
                        s_input_idx,
                        s_last_remain,
                        num_warps,
                        s_warp_sums,
                    )
                work_remaining = has_work

    @cute.jit
    def __call__(
        self,
        input_values,
        indices,
        extra_buffer,
        g_global_counter,
        seqlen,
        output_indices,
        output_values,
        stream: cuda.CUstream,
        enable_persistent_dynamic_scheduling: cutlass.Constexpr[bool] = False,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        """Host function for the indexer topk kernel"""
        # now we don't support it.
        assert not (
            self.enable_multi_cta and enable_persistent_dynamic_scheduling
        ), "enable_multi_cta and enable_persistent_dynamic_scheduling cannot both be True"

        num_rows = input_values.shape[0]
        # each cta processes one row of input.
        if cutlass.const_expr(self.enable_dynamic_multi_cta):
            blocks = (num_rows, self.num_ctas_per_row, 1)
        elif cutlass.const_expr(not enable_persistent_dynamic_scheduling):
            blocks = (num_rows, self.num_ctas_per_row, 1)
        else:
            blocks = (min(self.num_sms * min_blocks_per_mp, num_rows), self.num_ctas_per_row, 1)

        (
            copy_atom,
            tiled_copy,
            tiler_mn,
        ) = self._get_tiled_copy()
        self.indexer_topk_kernel(
            input_values,
            indices,
            extra_buffer,
            g_global_counter,
            seqlen,
            output_indices,
            output_values,
            tiler_mn,
            copy_atom,
            tiled_copy,
            enable_persistent_dynamic_scheduling,
            min_blocks_per_mp,
        ).launch(
            grid=blocks,
            block=(tiled_copy.size, 1, 1),
            stream=stream,
        )
        return


def _next_positive_power_of_2(x: int) -> int:
    """Round up to the next power of 2 (returns x if already a power of 2)."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _bucket_num_cols(num_cols: int) -> int:
    """Bucket num_cols to the next power of 2 for compilation caching.

    This reduces recompilations when num_cols changes slightly (e.g.,
    KV cache length growing each decode step). Safe because num_cols
    only affects compile-time config; actual data access is bounded
    by seq_lens.
    """
    return _next_positive_power_of_2(num_cols)


_compile_cache: dict[tuple, object] = {}


def cute_dsl_topk_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    load_balance=False,
    num_copy_bits=256,
):
    torch_dtype = input_values.dtype
    dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
    num_rows, num_cols = input_values.shape
    bucketed_num_cols = _bucket_num_cols(num_cols)

    large_occupancy = num_rows > 148
    assert not load_balance

    key = (
        dtype,
        bucketed_num_cols,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        load_balance,
        large_occupancy,
    )
    if key not in _compile_cache:
        n_rows = cute.sym_int()
        n_cols = cute.sym_int()
        n_batch = cute.sym_int()
        input_fake = cute.runtime.make_fake_compact_tensor(dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=32)
        buffer_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (cute.sym_int(), cute.sym_int(), cute.sym_int()),
            stride_order=(2, 1, 0),
            assumed_align=32,
        )
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cute.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, top_k),
            stride_order=(1, 0),
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
        else:
            output_values_fake = None
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        indexer_topk_func = IndexerTopKKernelVarlenDecode(
            dtype,
            bucketed_num_cols,
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            large_occupancy=large_occupancy,
        )

        # Compile the kernel
        compiled_kernel = cute.compile(
            indexer_topk_func,
            input_fake,
            None,  # indices_fake,
            buffer_fake,
            None,  # g_global_counter_fake,
            seqlen_fake,
            output_indices_fake,
            output_values_fake,
            stream=fake_stream,
            enable_persistent_dynamic_scheduling=load_balance,
            min_blocks_per_mp=1,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled_kernel
    else:
        compiled_kernel = _compile_cache[key]

    output_indices_torch = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    if return_val:
        output_values_torch = torch.empty(num_rows, top_k, dtype=torch_dtype, device="cuda")
    else:
        output_values_torch = None

    if dtype == cutlass.Float32:
        buffer_numbers = 2
    else:
        buffer_numbers = 1
    # Note: zeros will trigger an elementwise_add kernel.
    buffer_torch = torch.empty(num_rows, buffer_numbers, num_cols, dtype=torch.int32, device="cuda")
    g_global_counter_torch = None

    # TVM FFI uses env stream automatically
    compiled_kernel(
        input_values,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_torch,
        g_global_counter_torch,
        seq_lens,
        output_indices_torch,
        output_values_torch,
    )
    return output_indices_torch, output_values_torch
