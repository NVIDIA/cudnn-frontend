# Copyright (c) 2025-2026, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal CuTe DSL pipeline classes adapted from quack-kernels 0.5.0."""

from typing import Optional
from dataclasses import dataclass

from cutlass import Boolean, const_expr
from cutlass.cutlass_dsl import if_generate, and_, dsl_user_op
from cutlass.pipeline import PipelineState
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg


def _override_create(parent_cls, child_cls):
    """Create a static factory that constructs parent_cls then re-classes to child_cls."""

    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineTmaUmmaOg):
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if const_expr(extra_tx_count == 0):
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineUmmaAsyncOg):
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)
