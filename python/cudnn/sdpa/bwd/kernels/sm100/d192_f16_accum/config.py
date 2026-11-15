# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense D_QK = 192 / D_V = 128 half accumulation geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    M: int = 128
    N: int = 64
    DQK: int = 192
    DV: int = 128
    K_HW: int = 16
    BPE: int = 2
    HEAD_BOX: int = 64
    DQ_BOX: int = 32
    DQ_CHUNK: int = 64
    DQ_CHUNKS: int = 3
    THREADS: int = 512
    TMEM_COLS: int = 512
    TMEM_S: int = 0
    TMEM_DP: int = 64
    TMEM_DQ: int = 128
    TMEM_DV: int = 192
    TMEM_DK: int = 320
    Q: int = 0
    DO: int = 49152
    K: int = 81920
    V: int = 131072
    DS: int = 163840
    DQ_STAGE: int = 180224
    STATS: int = 212992
    DELTA: int = 213504
    TMEM_HANDLE: int = 214432
    SCHED_PAYLOAD: int = 214448
    SMEM_BYTES: int = 215040
    Q_BYTES: int = 24576
    DO_BYTES: int = 16384
    K_BYTES: int = 49152
    V_BYTES: int = 32768
    DS_BYTES: int = 16384
    DQ_STAGE_BYTES: int = 16384
    SLICE_BYTES: int = 8192
    Q_STAGES: int = 2
    DO_STAGES: int = 2
    DQ_STAGES: int = 2
    SCHED_STAGES: int = 2
    SOFTMAX_LANES: int = 256
    EPILOGUE_LANES: int = 128
    STORE_LANES: int = 32
    ONE_LANE: int = 1
    SCHED_CONSUMER_WARPS: int = 15
    SOFTMAX_REGS: int = 144
    EPILOGUE_REGS: int = 128
    SERVICE_REGS: int = 96
    SOFTMAX_BARRIER: int = 1
    DESC_VERSION: int = 1
    SWIZZLE_BYTES: int = 128
    SWIZZLE_LAYOUT: int = 2


CFG = Config()

_BAR_BASE = 214016

BAR_LAYOUT = (
    ("k_empty", _BAR_BASE + 0, 1, 32, "thread"),
    ("k_full", _BAR_BASE + 8, 1, 1, "tma"),
    ("v_full", _BAR_BASE + 16, 1, 1, "tma"),
    ("do_empty", _BAR_BASE + 24, 2, 1, "mma"),
    ("do_full", _BAR_BASE + 40, 4, 1, "tma"),
    ("q_empty", _BAR_BASE + 72, 2, 1, "mma"),
    ("q_full", _BAR_BASE + 88, 2, 1, "tma"),
    ("s_done", _BAR_BASE + 104, 1, 1, "mma"),
    ("dp_done", _BAR_BASE + 112, 1, 1, "mma"),
    ("dq_done", _BAR_BASE + 120, 1, 1, "mma"),
    ("dq_read", _BAR_BASE + 128, 1, 128, "thread"),
    ("p_ready", _BAR_BASE + 136, 1, 256, "thread"),
    ("ds_ready", _BAR_BASE + 144, 1, 256, "thread"),
    ("ds_smem", _BAR_BASE + 152, 1, 256, "thread"),
    ("dq_empty", _BAR_BASE + 160, 2, 32, "thread"),
    ("dq_full", _BAR_BASE + 176, 2, 128, "thread"),
    ("dv_full", _BAR_BASE + 192, 1, 128, "thread"),
    ("dk_full", _BAR_BASE + 200, 1, 128, "thread"),
    ("dvk_empty", _BAR_BASE + 208, 1, 32, "thread"),
    ("sched_full", _BAR_BASE + 216, 2, 1, "tma"),
    ("sched_empty", _BAR_BASE + 232, 2, 15, "thread"),
)


def validate_config():
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    assert (CFG.M, CFG.N, CFG.DQK, CFG.DV, CFG.K_HW, CFG.BPE) == (128, 64, 192, 128, 16, 2)
    assert (CFG.THREADS, CFG.TMEM_COLS, CFG.Q_STAGES, CFG.DO_STAGES, CFG.DQ_STAGES, CFG.SCHED_STAGES) == (512, 512, 2, 2, 2, 2)
    assert (CFG.DESC_VERSION, CFG.SWIZZLE_BYTES, CFG.SWIZZLE_LAYOUT) == (1, 128, 2)
    assert (CFG.HEAD_BOX, CFG.DQ_BOX, CFG.DQ_CHUNK, CFG.DQ_CHUNKS) == (64, 32, 64, 3) and CFG.DQ_CHUNK * CFG.DQ_CHUNKS == CFG.DQK
    assert CFG.Q_BYTES == CFG.N * CFG.DQK * CFG.BPE and CFG.DO_BYTES == CFG.N * CFG.DV * CFG.BPE
    assert CFG.K_BYTES == CFG.M * CFG.DQK * CFG.BPE and CFG.V_BYTES == CFG.M * CFG.DV * CFG.BPE
    assert CFG.DS_BYTES == CFG.M * CFG.N * CFG.BPE and CFG.DQ_STAGE_BYTES == CFG.N * CFG.DQ_CHUNK * 4 and CFG.SLICE_BYTES == CFG.N * CFG.DQ_BOX * 4
    assert CFG.M * CFG.DV * CFG.BPE + CFG.M * CFG.DQK * CFG.BPE <= CFG.K_BYTES + CFG.V_BYTES and CFG.V == CFG.K + CFG.K_BYTES
    assert (CFG.TMEM_S, CFG.TMEM_DP, CFG.TMEM_DQ, CFG.TMEM_DV, CFG.TMEM_DK) == (0, 64, 128, 192, 320)
    assert CFG.TMEM_DK + CFG.DQK == CFG.TMEM_COLS and CFG.TMEM_DV + CFG.DV == CFG.TMEM_DK
    assert (CFG.SOFTMAX_LANES, CFG.EPILOGUE_LANES, CFG.STORE_LANES, CFG.ONE_LANE, CFG.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 15)
    assert (CFG.SOFTMAX_REGS, CFG.EPILOGUE_REGS, CFG.SERVICE_REGS) == (144, 128, 96)
    assert (CFG.SOFTMAX_LANES * CFG.SOFTMAX_REGS + CFG.EPILOGUE_LANES * CFG.EPILOGUE_REGS + 4 * 32 * CFG.SERVICE_REGS) == 65536
    assert CFG.SOFTMAX_BARRIER == 1
    regions = (
        (CFG.Q, CFG.Q_STAGES * CFG.Q_BYTES),
        (CFG.DO, CFG.DO_STAGES * CFG.DO_BYTES),
        (CFG.K, CFG.K_BYTES),
        (CFG.V, CFG.V_BYTES),
        (CFG.DS, CFG.DS_BYTES),
        (CFG.DQ_STAGE, CFG.DQ_STAGES * CFG.DQ_STAGE_BYTES),
        (CFG.STATS, 512),
        (CFG.DELTA, 512),
    )
    assert tuple(start for start, _ in regions) == (0, 49152, 81920, 131072, 163840, 180224, 212992, 213504)
    assert all(start + size == regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    assert regions[-1][0] + regions[-1][1] == _BAR_BASE
    thread_counts = dict(
        k_empty=32, dq_read=128, p_ready=256, ds_ready=256, ds_smem=256, dq_empty=32, dq_full=128, dv_full=128, dk_full=128, dvk_empty=32, sched_empty=15
    )
    assert len(BAR_LAYOUT) == 21 and sum(item[2] for item in BAR_LAYOUT) == 31
    addresses = set()
    for name, offset, stages, count, kind in BAR_LAYOUT:
        assert stages in (1, 2, 4) and offset % 8 == 0
        assert count == (thread_counts[name] if kind == "thread" else CFG.ONE_LANE)
        assert kind in ("thread", "mma", "tma")
        for i in range(stages):
            address = offset + i * 8
            assert _BAR_BASE <= address < CFG.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert _BAR_BASE + 248 <= CFG.TMEM_HANDLE
    assert (CFG.TMEM_HANDLE, CFG.SCHED_PAYLOAD, CFG.SMEM_BYTES) == (214432, 214448, 215040)
    assert CFG.TMEM_HANDLE + 4 <= CFG.SCHED_PAYLOAD and CFG.SCHED_PAYLOAD + 64 <= CFG.SMEM_BYTES <= 227 * 1024
    assert (CFG.SCHED_PAYLOAD + 64 + 1023) // 1024 * 1024 == CFG.SMEM_BYTES


validate_config()
