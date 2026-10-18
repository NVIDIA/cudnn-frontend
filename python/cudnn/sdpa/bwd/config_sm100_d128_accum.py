# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d128 accumulation geometry. Offsets are bytes from one aligned slab."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    M: int = 128
    N: int = 128
    D: int = 128
    K_HW: int = 16
    BPE: int = 2
    HEAD_BOX: int = 64
    DQ_BOX: int = 32
    THREADS: int = 512
    TMEM_COLS: int = 512
    SMEM_BYTES: int = 231424
    Q: int = 0
    DO: int = 65536
    K: int = 98304
    V: int = 131072
    ALIAS: int = 163840
    STATS: int = 229376
    DELTA: int = 229888
    TMEM_HANDLE: int = 230800
    SCHED_PAYLOAD: int = 230816
    OPERAND_BYTES: int = 32768
    SLICE_BYTES: int = 16384
    Q_STAGES: int = 2
    SCHED_STAGES: int = 2
    SOFTMAX_LANES: int = 256
    EPILOGUE_LANES: int = 128
    STORE_LANES: int = 32
    ONE_LANE: int = 1
    SCHED_CONSUMER_WARPS: int = 15
    SOFTMAX_REGS: int = 144
    EPILOGUE_REGS: int = 144
    SERVICE_REGS: int = 80
    SOFTMAX_BARRIER: int = 1
    DESC_VERSION: int = 1
    SWIZZLE_BYTES: int = 128
    SWIZZLE_LAYOUT: int = 2


CFG = Config()

# name, byte offset, active slots, init count, producer kind.
# Unused declaration padding remains: dq_done[1:3], dp_empty[1], ds_smem[1].
BAR_LAYOUT = (
    ("k_empty", 230400, 1, 1, "mma"),
    ("k_full", 230416, 1, 1, "tma"),
    ("v_empty", 230432, 1, 1, "mma"),
    ("v_full", 230448, 1, 1, "tma"),
    ("do_empty", 230464, 1, 1, "mma"),
    ("do_full", 230480, 2, 1, "tma"),
    ("q_empty", 230496, 2, 1, "mma"),
    ("q_full", 230512, 2, 1, "tma"),
    ("s_done", 230528, 1, 1, "mma"),
    ("dp_done", 230544, 1, 1, "mma"),
    ("dq_done", 230560, 1, 1, "mma"),
    ("dp_empty", 230592, 1, 128, "thread"),
    ("p_ready", 230608, 1, 256, "thread"),
    ("ds_ready", 230624, 1, 256, "thread"),
    ("ds_smem", 230640, 1, 128, "thread"),
    ("dq_empty", 230656, 4, 32, "thread"),
    ("dq_full", 230688, 4, 128, "thread"),
    ("dv_full", 230720, 1, 128, "thread"),
    ("dk_full", 230736, 1, 128, "thread"),
    ("dvk_empty", 230752, 1, 32, "thread"),
    ("sched_full", 230768, 2, 1, "tma"),
    ("sched_empty", 230784, 2, 15, "thread"),
)


def validate_config():
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    assert (CFG.M, CFG.N, CFG.D, CFG.K_HW, CFG.BPE) == (128, 128, 128, 16, 2)
    assert (CFG.THREADS, CFG.TMEM_COLS, CFG.Q_STAGES, CFG.SCHED_STAGES) == (512, 512, 2, 2)
    assert (CFG.DESC_VERSION, CFG.SWIZZLE_BYTES, CFG.SWIZZLE_LAYOUT) == (1, 128, 2)
    assert (CFG.HEAD_BOX, CFG.DQ_BOX, CFG.OPERAND_BYTES, CFG.SLICE_BYTES) == (64, 32, 32768, 16384)
    assert (CFG.SOFTMAX_LANES, CFG.EPILOGUE_LANES, CFG.STORE_LANES, CFG.ONE_LANE, CFG.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 15)
    assert (CFG.SOFTMAX_REGS, CFG.EPILOGUE_REGS, CFG.SERVICE_REGS) == (144, 144, 80)
    assert (CFG.SOFTMAX_LANES * CFG.SOFTMAX_REGS + CFG.EPILOGUE_LANES * CFG.EPILOGUE_REGS + 4 * 32 * CFG.SERVICE_REGS) == 65536
    assert CFG.SOFTMAX_BARRIER == 1  # CTA synchronization reserves barrier0.
    regions = (
        (CFG.Q, 2 * CFG.OPERAND_BYTES),
        (CFG.DO, CFG.OPERAND_BYTES),
        (CFG.K, CFG.OPERAND_BYTES),
        (CFG.V, CFG.OPERAND_BYTES),
        (CFG.ALIAS, 2 * CFG.OPERAND_BYTES),
        (CFG.STATS, 512),
        (CFG.DELTA, 512),
    )
    assert tuple(start for start, _ in regions) == (0, 65536, 98304, 131072, 163840, 229376, 229888)
    assert all(start + size == regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    assert regions[-1][0] + regions[-1][1] == 230400
    thread_counts = dict(dp_empty=128, p_ready=256, ds_ready=256, ds_smem=128, dq_empty=32, dq_full=128, dv_full=128, dk_full=128, dvk_empty=32, sched_empty=15)
    assert len(BAR_LAYOUT) == 22 and sum(item[2] for item in BAR_LAYOUT) == 33
    addresses = set()
    for name, offset, stages, count, kind in BAR_LAYOUT:
        assert stages in (1, 2, 4) and offset % 8 == 0
        assert count == (thread_counts[name] if kind == "thread" else CFG.ONE_LANE)
        assert kind in ("thread", "mma", "tma")
        for i in range(stages):
            address = offset + i * 8
            assert 230400 <= address < CFG.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert (CFG.TMEM_HANDLE, CFG.SCHED_PAYLOAD, CFG.SMEM_BYTES) == (230800, 230816, 231424)
    assert CFG.TMEM_HANDLE + 4 <= CFG.SCHED_PAYLOAD
    assert CFG.SCHED_PAYLOAD + 64 == 230880
    assert (230880 + 1023) // 1024 * 1024 == CFG.SMEM_BYTES


validate_config()
