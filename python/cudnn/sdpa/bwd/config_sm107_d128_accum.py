# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM107 d128 accumulation configuration.

No new stage, TMEM plane, allocation mode or issue order is enabled here.
The pure source Config is reused; the target owns its constants and validation.
BAR_LAYOUT retains the audited source offsets, kinds and named contributor counts.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

from .config_sm100_d128_accum import Config as SourceConfig

BASE_SHA = "850a7e99f01a879f218ce6e0059c272b685418d2"


@dataclass(frozen=True)
class Config(SourceConfig):
    TMEM_COLS: int = 512  # Explicit target budget; same value as the source field.
    ARCH_CC: ClassVar[tuple[int, int]] = (10, 7)
    CTA_MMA: ClassVar[int] = 1
    CTA_CLUSTER: ClassVar[tuple[int, int, int]] = (1, 1, 1)
    DO_STAGES: ClassVar[int] = 1
    SINGLE_STAGE: ClassVar[int] = 1
    DO_HALF_SLOTS: ClassVar[int] = SourceConfig.D // SourceConfig.HEAD_BOX
    DQ_SLICES: ClassVar[int] = SourceConfig.D // SourceConfig.DQ_BOX
    SCAFFOLD_BYTES: ClassVar[int] = 1024
    SCHED_TX_BYTES: ClassVar[int] = 16
    SCHED_PAYLOAD_STAGE_BYTES: ClassVar[int] = 32
    WARP_LANES: ClassVar[int] = 32
    SOFTMAX_WARPS: ClassVar[int] = SourceConfig.SOFTMAX_LANES // WARP_LANES
    EPILOGUE_WARPS: ClassVar[int] = SourceConfig.EPILOGUE_LANES // WARP_LANES
    SERVICE_WARPS: ClassVar[int] = 4
    MMA_WARP: ClassVar[int] = SOFTMAX_WARPS + EPILOGUE_WARPS
    LOAD_WARP: ClassVar[int] = MMA_WARP + 1
    STORE_WARP: ClassVar[int] = MMA_WARP + 2
    SCHED_WARP: ClassVar[int] = MMA_WARP + 3
    SMEM_ALIGNMENT: ClassVar[int] = 1024
    BARRIER_ALIGNMENT: ClassVar[int] = 16
    SCHED_PAYLOAD_WORDS: ClassVar[int] = SourceConfig.SCHED_STAGES * SCHED_PAYLOAD_STAGE_BYTES // SourceConfig.BPE_ACC
    SOURCE_SCHED_L2_BYTES: ClassVar[int] = 50 * 1024 * 1024  # Source scheduling heuristic, not a target capacity claim.
    MIN_BLOCKS_PER_MP: ClassVar[int] = 1
    ENTRY_REGS: ClassVar[int] = 128  # Register bound for the 512-thread entry.
    S_COL: ClassVar[int] = 0
    DP_DS_DQ_COL: ClassVar[int] = SourceConfig.D
    PACK_GROUP_COLS: ClassVar[int] = SourceConfig.D_HALF // SourceConfig.BPE
    FP32_PAIR_ELEMS: ClassVar[int] = 2
    STATS_VECTOR_ELEMS: ClassVar[int] = 4
    STATS_VECTOR_ALIGNMENT: ClassVar[int] = STATS_VECTOR_ELEMS * SourceConfig.BPE_ACC
    SOFTMAX_TMEM_LOAD_COLS: ClassVar[int] = SourceConfig.D_HALF // FP32_PAIR_ELEMS
    SOFTMAX_CHUNKS: ClassVar[int] = SourceConfig.D_HALF // STATS_VECTOR_ELEMS
    SOFTMAX_PAIRS_PER_CHUNK: ClassVar[int] = STATS_VECTOR_ELEMS // FP32_PAIR_ELEMS
    STORE_VECTOR_BYTES: ClassVar[int] = 16
    STORE_VECTOR_ELEMS: ClassVar[int] = STORE_VECTOR_BYTES // SourceConfig.BPE_ACC
    STORE_VECS_PER_ROW: ClassVar[int] = SourceConfig.SWIZZLE_BYTES // STORE_VECTOR_BYTES
    SWIZZLE_ROWS: ClassVar[int] = 8
    # Retain the source descriptor version for these operand layouts.
    # Allocate and deallocate the same 512-column TMEM extent.


CFG = Config()


def smem_layout():
    """Predictor rows in source declaration order; Stats/delta share one1-KiB region."""
    return (
        dict(name="Q", bytes=CFG.OPERAND_BYTES, stages=CFG.Q_STAGES, descriptor_fed=True),
        dict(name="dO", bytes=CFG.OPERAND_BYTES, stages=CFG.DO_STAGES, descriptor_fed=True),
        dict(name="K", bytes=CFG.OPERAND_BYTES, stages=CFG.SINGLE_STAGE, descriptor_fed=True),
        dict(name="V", bytes=CFG.OPERAND_BYTES, stages=CFG.SINGLE_STAGE, descriptor_fed=True),
        dict(name="ALIAS_dS_dQ_dV_dK", bytes=CFG.M * CFG.D * CFG.BPE_ACC, descriptor_fed=True),
        dict(name="STATS_DELTA", bytes=2 * CFG.N * CFG.BPE_ACC, descriptor_fed=False),
    )


SMEM_OFFSETS = (CFG.Q, CFG.DO, CFG.K, CFG.V, CFG.ALIAS, CFG.STATS)
TMEM_PLANES = (
    ("S_P", CFG.S_COL, CFG.D),
    ("dP_dS_dQ", CFG.DP_DS_DQ_COL, CFG.D),
    ("dV", CFG.DV_COL, CFG.D),
    ("dK", CFG.DK_COL, CFG.D),
)

# Values are references to named source contributor counts, not new literal init counts.
# The inspector consumes these names when filling the actual BAR_LAYOUT.
INIT_COUNT_NAMES = {
    "MMA_OR_TMA": "ONE_LANE",
    "SOFTMAX": "SOFTMAX_LANES",
    "EPILOGUE": "EPILOGUE_LANES",
    "STORE": "STORE_LANES",
    "SCHED_CONSUMERS": "SCHED_CONSUMER_WARPS",
}
BARRIER_SLOT_COUNTS = {
    "k_empty": CFG.SINGLE_STAGE,
    "k_full": CFG.SINGLE_STAGE,
    "v_empty": CFG.SINGLE_STAGE,
    "v_full": CFG.SINGLE_STAGE,
    "do_empty": CFG.DO_STAGES,
    "do_full": CFG.DO_HALF_SLOTS,
    "q_empty": CFG.Q_STAGES,
    "q_full": CFG.Q_STAGES,
    "s_done": CFG.SINGLE_STAGE,
    "dp_done": CFG.SINGLE_STAGE,
    "dq_done": CFG.SINGLE_STAGE,
    "dp_empty": CFG.SINGLE_STAGE,
    "p_ready": CFG.SINGLE_STAGE,
    "ds_ready": CFG.SINGLE_STAGE,
    "ds_smem": CFG.SINGLE_STAGE,
    "dq_empty": CFG.DQ_SLICES,
    "dq_full": CFG.DQ_SLICES,
    "dv_full": CFG.SINGLE_STAGE,
    "dk_full": CFG.SINGLE_STAGE,
    "dvk_empty": CFG.SINGLE_STAGE,
    "sched_full": CFG.SCHED_STAGES,
    "sched_empty": CFG.SCHED_STAGES,
}
BAR_NAME_COUNT = len(BARRIER_SLOT_COUNTS)
BAR_ACTIVE_SLOT_COUNT = sum(BARRIER_SLOT_COUNTS.values())
# Per-slot contributor derivation (one CTA): elected MMA/TMA=ONE_LANE(1),
# bare softmax=8*32=SOFTMAX_LANES(256), epilogue=4*32=EPILOGUE_LANES(128),
# bare store=1*32=STORE_LANES(32), elected consumer warps=8+4+1+1+1=15.
# name, source byte offset, active slots, named init count, producer kind.
# Reserved padding is not active: dq_done[1:3], dp_empty[1], ds_smem[1].
BAR_LAYOUT = (
    ("k_empty", 230400, CFG.SINGLE_STAGE, CFG.ONE_LANE, "mma"),
    ("k_full", 230416, CFG.SINGLE_STAGE, CFG.ONE_LANE, "tma"),
    ("v_empty", 230432, CFG.SINGLE_STAGE, CFG.ONE_LANE, "mma"),
    ("v_full", 230448, CFG.SINGLE_STAGE, CFG.ONE_LANE, "tma"),
    ("do_empty", 230464, CFG.DO_STAGES, CFG.ONE_LANE, "mma"),
    ("do_full", 230480, CFG.DO_HALF_SLOTS, CFG.ONE_LANE, "tma"),
    ("q_empty", 230496, CFG.Q_STAGES, CFG.ONE_LANE, "mma"),
    ("q_full", 230512, CFG.Q_STAGES, CFG.ONE_LANE, "tma"),
    ("s_done", 230528, CFG.SINGLE_STAGE, CFG.ONE_LANE, "mma"),
    ("dp_done", 230544, CFG.SINGLE_STAGE, CFG.ONE_LANE, "mma"),
    ("dq_done", 230560, CFG.SINGLE_STAGE, CFG.ONE_LANE, "mma"),
    ("dp_empty", 230592, CFG.SINGLE_STAGE, CFG.EPILOGUE_LANES, "thread"),
    ("p_ready", 230608, CFG.SINGLE_STAGE, CFG.SOFTMAX_LANES, "thread"),
    ("ds_ready", 230624, CFG.SINGLE_STAGE, CFG.SOFTMAX_LANES, "thread"),
    ("ds_smem", 230640, CFG.SINGLE_STAGE, CFG.EPILOGUE_LANES, "thread"),
    ("dq_empty", 230656, CFG.DQ_SLICES, CFG.STORE_LANES, "thread"),
    ("dq_full", 230688, CFG.DQ_SLICES, CFG.EPILOGUE_LANES, "thread"),
    ("dv_full", 230720, CFG.SINGLE_STAGE, CFG.EPILOGUE_LANES, "thread"),
    ("dk_full", 230736, CFG.SINGLE_STAGE, CFG.EPILOGUE_LANES, "thread"),
    ("dvk_empty", 230752, CFG.SINGLE_STAGE, CFG.STORE_LANES, "thread"),
    ("sched_full", 230768, CFG.SCHED_STAGES, CFG.ONE_LANE, "tma"),
    ("sched_empty", 230784, CFG.SCHED_STAGES, CFG.SCHED_CONSUMER_WARPS, "thread"),
)

# Descriptor declarations are metadata only; a writer must instantiate SmemTile/MmaDesc.
OPERAND_DESCRIPTOR = {
    "elems_per_stage": CFG.M * CFG.D,
    "leading_byte_offset": 16,
    "leading_transpose_byte_offset": CFG.M * CFG.SWIZZLE_BYTES,
    "stride_byte_offset": 8 * CFG.SWIZZLE_BYTES,
    "layout": CFG.SWIZZLE_LAYOUT,
    "desc_version": CFG.DESC_VERSION,
    "tma_granu_elems": CFG.HEAD_BOX,
    "tma_subtile_stride_elems": CFG.M * CFG.HEAD_BOX,
    "tma_loads_per_tile": CFG.DO_HALF_SLOTS,
}
DQ_STORE_DESCRIPTOR = {
    **OPERAND_DESCRIPTOR,
    "tma_granu_elems": CFG.DQ_BOX,
    "tma_subtile_stride_elems": CFG.M * CFG.DQ_BOX,
    "tma_loads_per_tile": CFG.DQ_SLICES,
}
MMA_DESCRIPTORS = (
    ("S_and_dP", CFG.M, CFG.N, CFG.D, False, False),
    ("dV_and_dK", CFG.M, CFG.N, CFG.D, False, True),
    ("dQ", CFG.M, CFG.N, CFG.D, True, True),
)


def kernel_smem_bytes():
    return CFG.SMEM_BYTES


def validate_layout():
    source = SourceConfig()
    if not (all(getattr(CFG, f.name) == getattr(source, f.name) for f in fields(SourceConfig))):
        raise ValueError("SM107 d128 layout invariant failed: all(getattr(CFG, f.name) == getattr(source, f.name) for f in fields(SourceConfig))")
    off = 0
    for row, expected in zip(smem_layout(), SMEM_OFFSETS):
        if not (off == expected and off % 1024 == 0):
            raise ValueError("SM107 d128 layout invariant failed: off == expected and off % 1024 == 0")
        off += row["bytes"] * row.get("stages", 1)
    if not (off + CFG.SCAFFOLD_BYTES == kernel_smem_bytes() == 231424):
        raise ValueError("SM107 d128 layout invariant failed: off + CFG.SCAFFOLD_BYTES == kernel_smem_bytes() == 231424")
    if not (CFG.DELTA == CFG.STATS + CFG.N * CFG.BPE_ACC):
        raise ValueError("SM107 d128 layout invariant failed: CFG.DELTA == CFG.STATS + CFG.N * CFG.BPE_ACC")
    if not (CFG.TMEM_HANDLE == off + 400):
        raise ValueError("SM107 d128 layout invariant failed: CFG.TMEM_HANDLE == off + 400")
    if not (CFG.SCHED_PAYLOAD == off + 416):
        raise ValueError("SM107 d128 layout invariant failed: CFG.SCHED_PAYLOAD == off + 416")
    if not (CFG.SCHED_PAYLOAD + CFG.SCHED_STAGES * CFG.SCHED_PAYLOAD_STAGE_BYTES == off + 480):
        raise ValueError("SM107 d128 layout invariant failed: CFG.SCHED_PAYLOAD + CFG.SCHED_STAGES * CFG.SCHED_PAYLOAD_STAGE_BYTES == off + 480")
    if not ((BAR_NAME_COUNT, BAR_ACTIVE_SLOT_COUNT) == (22, 33)):
        raise ValueError("SM107 d128 layout invariant failed: (BAR_NAME_COUNT, BAR_ACTIVE_SLOT_COUNT) == (22, 33)")
    if not (all(base + cols == TMEM_PLANES[i + 1][1] for i, (_, base, cols) in enumerate(TMEM_PLANES[:-1]))):
        raise ValueError("SM107 d128 layout invariant failed: all(base + cols == TMEM_PLANES[i + 1][1] for i, (_, base, cols) in enumerate(TMEM_PLANES[:-1]))")
    if not (TMEM_PLANES[-1][1] + TMEM_PLANES[-1][2] == CFG.TMEM_COLS == 512):
        raise ValueError("SM107 d128 layout invariant failed: TMEM_PLANES[-1][1] + TMEM_PLANES[-1][2] == CFG.TMEM_COLS == 512")
    if not (
        CFG.SOFTMAX_WARPS * CFG.SOFTMAX_REGS + CFG.EPILOGUE_WARPS * CFG.EPILOGUE_REGS + CFG.SERVICE_WARPS * CFG.SERVICE_REGS
        == CFG.ENTRY_REGS * (CFG.THREADS // CFG.WARP_LANES)
    ):
        raise ValueError(
            "SM107 d128 layout invariant failed: CFG.SOFTMAX_WARPS * CFG.SOFTMAX_REGS + CFG.EPILOGUE_WARPS * CFG.EPILOGUE_REGS + CFG.SERVICE_WARPS * CFG.SERVICE_REGS == CFG.ENTRY_REGS * (CFG.THREADS // CFG.WARP_LANES)"
        )
    if not (all(r % 8 == 0 and 24 <= r <= 256 for r in (CFG.SOFTMAX_REGS, CFG.EPILOGUE_REGS, CFG.SERVICE_REGS))):
        raise ValueError(
            "SM107 d128 layout invariant failed: all(r % 8 == 0 and 24 <= r <= 256 for r in (CFG.SOFTMAX_REGS, CFG.EPILOGUE_REGS, CFG.SERVICE_REGS))"
        )
    if not (CFG.SOFTMAX_REGS > CFG.ENTRY_REGS and CFG.EPILOGUE_REGS > CFG.ENTRY_REGS > CFG.SERVICE_REGS):
        raise ValueError("SM107 d128 layout invariant failed: CFG.SOFTMAX_REGS > CFG.ENTRY_REGS and CFG.EPILOGUE_REGS > CFG.ENTRY_REGS > CFG.SERVICE_REGS")


validate_layout()
