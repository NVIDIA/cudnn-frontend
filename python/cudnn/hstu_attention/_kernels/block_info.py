# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from .seqlen_info import SeqlenInfo
from . import utils
from .named_barrier import NamedBarrierFwd


@dataclass(frozen=True)
class BlockInfo:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    cta_tiler: cutlass.Constexpr[Tuple[int, int]]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    is_paged: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    sn_valid_block_max: cute.Pointer = None
    sValidBlockIds: cute.Tensor = None # (MaxValidBlock,)
    sBlockBound: cute.Pointer = None
    func_num: cutlass.Constexpr[int] = 0
    func: cute.Tensor = None # (n_func, L_func)
    arbitrary_barrier: NamedBarrierFwd = None
    arbitrary_barrier_threads: cutlass.Constexpr[int] = 0

    @cute.jit
    def get_n_block_info(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32, offset_dynamic: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        seqlen_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        row_begin = m_block * self.cta_tiler[0] - offset_dynamic
        row_end = row_begin + self.cta_tiler[0]

        n_block_min = max(0, (row_begin + seqlen_offset - self.window_size_left) // self.cta_tiler[1]) if self.is_local else 0
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.cta_tiler[1])

        if self.is_causal or self.is_local:
            n_block_max = min(n_block_max, cute.ceil_div(row_end + seqlen_offset + self.window_size_right, self.cta_tiler[1]))

        n_masking_steps = 0
        if const_expr(not self.is_paged):
            n_masking_block_max = cute.ceil_div(min(seqlen_info.seqlen_k, row_end + seqlen_offset), self.cta_tiler[1])
            n_masking_block_min = (row_begin + seqlen_offset) // self.cta_tiler[1]

            # 1: first tile should be masked for boundary check
            n_masking_steps = 1 if not self.is_causal else n_masking_block_max - n_masking_block_min
        else:
            n_masking_pages = 0
            if row_begin + seqlen_offset < seqlen_info.seqlen_k:
                n_masking_pages = cute.ceil_div(min(seqlen_info.seqlen_k, row_end + seqlen_offset), self.cta_tiler[1])
                n_masking_pages -= max((row_begin + seqlen_offset) // self.cta_tiler[1], 0)
            n_masking_steps = n_masking_pages

        return n_block_max, n_block_min, n_masking_steps


    @cute.jit
    def get_valid_block_ids(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32, n_block_max: cutlass.Int32, n_block_min: cutlass.Int32, is_calwarp: cutlass.Constexpr[bool]
    ):
        lane_id = cute.arch.lane_idx()
        sn_valid_block_max_tensor = cute.make_tensor(self.sn_valid_block_max, (1,))
        sBlockMin = cute.make_tensor(self.sBlockBound, (self.func_num // 2 + 1,))
        sBlockMax = cute.make_tensor(self.sBlockBound + (self.func_num // 2 + 1) * 4, (self.func_num // 2 + 1,))
        sValidBlockIds = self.sValidBlockIds
        int_max = (1 << 31) - 1 # INF_MAX
        int_min = -(1 << 31)    # INF_MIN
        if cutlass.const_expr(is_calwarp):
            sn_valid_block_max_tensor[0] = 0
            sBlockMin[0] = 0
            cute.arch.sync_warp()

            cur_func = cute.domain_offset((0, seqlen_info.offset_q), self.func)
            base_row = m_block * self.cta_tiler[0]
            f_min = int_max
            for i in cutlass.range(self.func_num // 2):
                for j in cutlass.range(lane_id, self.cta_tiler[0], 32):
                    row = base_row + j
                    if row < seqlen_info.seqlen_q:
                        f_min = min(f_min, cur_func[2 * i + 1, row])

                f_min = utils.warp_reduce(f_min, cutlass.min)
                if lane_id == 0:
                    sBlockMin[i + 1] = f_min
                f_min = int_max

            f_max = int_min
            for i in cutlass.range(self.func_num // 2 + 1):
                for j in cutlass.range(lane_id, self.cta_tiler[0], 32):
                    row = base_row + j
                    if row < seqlen_info.seqlen_q:
                        f_max = max(f_max, cur_func[2 * i, row])

                f_max = utils.warp_reduce(f_max, cutlass.max)
                if lane_id == 0:
                    sBlockMax[i] = f_max
                f_max = int_min

            if lane_id == 0:
                for n_block in cutlass.range(n_block_min, n_block_max):
                    b_max = (n_block + 1) * self.cta_tiler[1]
                    b_min = n_block * self.cta_tiler[1]
                    block_valid = False
                    for i in cutlass.range(self.func_num // 2 + 1):
                        f_min = sBlockMin[i]
                        f_max = sBlockMax[i]

                        case1 = (f_min <= b_min and f_max > b_min)
                        case2 = (f_min >= b_min and b_max > f_min)
                        case3 = (f_min >= b_min and f_max < b_max)

                        if case1 or case2 or case3:
                            block_valid = True

                    if block_valid:
                        sValidBlockIds[sn_valid_block_max_tensor[0]] = n_block
                        sn_valid_block_max_tensor[0] = sn_valid_block_max_tensor[0] + 1

        cute.arch.barrier(barrier_id=self.arbitrary_barrier, number_of_threads=self.arbitrary_barrier_threads)
        return sn_valid_block_max_tensor[0], 0


@dataclass(frozen=True)
class BWDBlockInfo:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    cta_tiler: cutlass.Constexpr[Tuple[int, int]]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    sm_valid_block_max: cute.Pointer = None
    sValidBlockIds: cute.Tensor = None # (MaxValidBlock,)
    func_num: cutlass.Constexpr[int] = 0
    func: cute.Tensor = None # (n_func, L_func)
    arbitrary_barrier: NamedBarrierFwd = None
    arbitrary_barrier_threads: cutlass.Constexpr[int] = 0

    @cute.jit
    def get_m_block_info(
        self,  seqlen_info: SeqlenInfo, n_block: cutlass.Int32, offset_dynamic: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        seqlen_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q

        m_masking_block_max = cute.ceil_div(max(0, (n_block + 1) * self.cta_tiler[1] - seqlen_offset), self.cta_tiler[0])

        m_masking_block_min = max(0, n_block * self.cta_tiler[1] - seqlen_offset) // self.cta_tiler[0]
        m_masking_steps = m_masking_block_max - m_masking_block_min if self.is_causal else 1

        m_block_min = 0 if (not self.is_causal and not self.is_local) else max(0, (n_block * self.cta_tiler[1] - seqlen_offset - self.window_size_right) // self.cta_tiler[0])
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.cta_tiler[0])

        if self.is_local:
            m_block_max = min(m_block_max, cute.ceil_div((n_block + 1) * self.cta_tiler[1] - seqlen_offset + self.window_size_left, self.cta_tiler[0]))

        return m_block_min, m_block_max, m_masking_steps


    @cute.jit
    def get_bwd_valid_block_ids(self, seqlen_info: SeqlenInfo, n_block: cutlass.Int32, m_block_min: cutlass.Int32, m_block_max: cutlass.Int32,
        is_calwarp: cutlass.Constexpr[bool]
        ):
        lane_id = cute.arch.lane_idx() #thread idx in a warp, 0~31
        bidx, bidy, bidz = cute.arch.block_idx()
        sm_valid_block_max_tensor = cute.make_tensor(self.sm_valid_block_max, (1,))
        actual_seqlen_q = seqlen_info.seqlen_q

        sValidBlockIds = self.sValidBlockIds
        cur_func = cute.domain_offset((0, seqlen_info.offset_q), self.func)

        INT_MAX = cutlass.Int32.max
        INT_MIN = cutlass.Int32.min
        if cutlass.const_expr(is_calwarp):
            sm_valid_block_max_tensor[0] = 0
            b_min = n_block * self.cta_tiler[1]
            b_max = (n_block + 1) * self.cta_tiler[1]
            for m_block in cutlass.range(m_block_min, m_block_max):
                base_row = m_block * self.cta_tiler[0]
                f_min = 0
                f_max = INT_MIN
                for j in cutlass.range(lane_id, self.cta_tiler[0], 32):
                    row = base_row + j
                    if row < actual_seqlen_q:
                        f_max = max(f_max, cur_func[0, row])
                f_max = utils.warp_reduce(f_max, cutlass.max)

                case1 = (f_min <= b_min and f_max > b_min)
                case2 = (f_min >= b_min and b_max > f_min)
                case3 = (f_min >= b_min and f_max < b_max)

                is_valid = cute.arch.shuffle_sync(mask=0xFFFFFFFF, value=((case1 or case2 or case3) and (f_max > f_min)), offset=0)

                if is_valid:
                    sValidBlockIds[sm_valid_block_max_tensor[0]] = m_block
                    if lane_id == 0:
                        sm_valid_block_max_tensor[0] = sm_valid_block_max_tensor[0] + 1
                else:
                    break_state = True
                    for i in cutlass.range(self.func_num // 2):
                        if break_state:
                            f_min = INT_MAX
                            f_max = INT_MIN
                            for j in cutlass.range(lane_id, self.cta_tiler[0], 32):
                                row = base_row + j
                                if row < actual_seqlen_q:
                                    f_min = min(f_min, cur_func[2 * i + 1, row])
                                    f_max = max(f_max, cur_func[2 * i + 2, row]) #TODO: is this right?

                            f_min = utils.warp_reduce(f_min, cutlass.min)
                            f_max = utils.warp_reduce(f_max, cutlass.max)

                            case1 = (f_min <= b_min and f_max > b_min)
                            case2 = (f_min >= b_min and b_max > f_min)
                            case3 = (f_min >= b_min and f_max < b_max)
                            is_valid = cute.arch.shuffle_sync(mask=0xFFFFFFFF, value=((case1 or case2 or case3) and (f_max > f_min)), offset=0)

                            if is_valid:
                                sValidBlockIds[sm_valid_block_max_tensor[0]] = m_block
                                if lane_id == 0:
                                    sm_valid_block_max_tensor[0] = sm_valid_block_max_tensor[0] + 1
                                break_state = False

            # sm_valid_block_max_tensor[0] = 0
            # # sValidBlockIds[sm_valid_block_max_tensor[0]] = 0
            # if lane_id == 0:
            #     for i in range(m_block_max):
            #         sValidBlockIds[sm_valid_block_max_tensor[0]] = i
            #         sm_valid_block_max_tensor[0] = sm_valid_block_max_tensor[0] + 1

        cute.arch.barrier(barrier_id=self.arbitrary_barrier, number_of_threads=self.arbitrary_barrier_threads)
        return sm_valid_block_max_tensor[0], 0
