# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d128 per-tensor FP8 accumulation geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    M: int = 128
    N: int = 128
    D: int = 128
    K_HW: int = 32
    BPE: int = 1
    HEAD_BOX: int = 128
    IN_SWZ: int = 128
    DQ_BOX: int = 32
    THREADS: int = 512
    TMEM_COLS: int = 512
    SMEM_BYTES: int = 215040
    Q: int = 0
    DO: int = 32768
    K: int = 49152
    V: int = 65536
    ALIAS: int = 81920
    P: int = 147456
    DS: int = 180224
    STATS: int = 212992
    DELTA: int = 213504
    TMEM_HANDLE: int = 214416
    SCHED_PAYLOAD: int = 214432
    TMEM_DV: int = 256
    TMEM_DK: int = 384
    OPERAND_BYTES: int = 16384
    RING_BYTES: int = 16384
    SLICE_BYTES: int = 16384
    DQ_SLICES: int = 4
    ALIAS_SLICES: int = 4
    Q_STAGES: int = 2
    P_STAGES: int = 2
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
    NARROW_SWIZZLE_LAYOUT: int = 4


HEAD_DIMS = (64, 128)
_SLICE_BYTES = 16384
_BAR_BYTES = 400
_SCHED_BYTES = 64


def make_config(d: int) -> Config:
    """The slab geometry of the FP8 accumulation kernel at head dimension ``d``."""
    if d not in HEAD_DIMS:
        raise ValueError(f"fp8 accumulation head dimension must be one of {HEAD_DIMS}; got {d}")
    operand = 128 * d
    ring = 128 * 128
    dq_slices = d // 32
    alias_slices = max(dq_slices, 2 * 128 * d * 2 // _SLICE_BYTES)
    q = 0
    do = q + 2 * operand
    k = do + operand
    v = k + operand
    alias = v + operand
    p_ring = alias + alias_slices * _SLICE_BYTES
    ds_ring = p_ring + 2 * ring
    stats = ds_ring + 2 * ring
    delta = stats + 512
    bars = delta + 512
    tmem_handle = bars + _BAR_BYTES
    sched = tmem_handle + 16
    smem = (sched + _SCHED_BYTES + 1023) // 1024 * 1024
    return Config(
        D=d,
        HEAD_BOX=min(d, 128),
        IN_SWZ=min(d, 128),
        SMEM_BYTES=smem,
        Q=q,
        DO=do,
        K=k,
        V=v,
        ALIAS=alias,
        P=p_ring,
        DS=ds_ring,
        STATS=stats,
        DELTA=delta,
        TMEM_HANDLE=tmem_handle,
        SCHED_PAYLOAD=sched,
        TMEM_DV=256,
        TMEM_DK=256 + d,
        OPERAND_BYTES=operand,
        RING_BYTES=ring,
        DQ_SLICES=dq_slices,
        ALIAS_SLICES=alias_slices,
    )


def make_bar_layout(cfg: Config):
    """name, byte offset, active slots, init count, producer kind."""
    base = cfg.DELTA + 512
    return (
        ("k_empty", base + 0, 1, 1, "mma"),
        ("k_full", base + 16, 1, 1, "tma"),
        ("v_empty", base + 32, 1, 1, "mma"),
        ("v_full", base + 48, 1, 1, "tma"),
        ("do_empty", base + 64, 1, 1, "mma"),
        ("do_full", base + 80, 1, 1, "tma"),
        ("q_empty", base + 96, 2, 1, "mma"),
        ("q_full", base + 112, 2, 1, "tma"),
        ("s_done", base + 128, 1, 1, "mma"),
        ("dp_done", base + 144, 1, 1, "mma"),
        ("dq_done", base + 160, 1, 1, "mma"),
        ("s_free", base + 176, 1, 256, "thread"),
        ("dp_empty", base + 192, 1, 128, "thread"),
        ("p_full", base + 208, 2, 256, "thread"),
        ("p_empty", base + 224, 2, 1, "mma"),
        ("ds_full", base + 240, 2, 256, "thread"),
        ("ds_empty", base + 256, 2, 1, "mma"),
        ("dq_empty", base + 272, 4, 32, "thread"),
        ("dq_full", base + 304, 4, 128, "thread"),
        ("dv_full", base + 336, 1, 128, "thread"),
        ("dk_full", base + 344, 1, 128, "thread"),
        ("dvk_empty", base + 352, 1, 32, "thread"),
        ("sched_full", base + 360, 2, 1, "tma"),
        ("sched_empty", base + 376, 2, 15, "thread"),
    )


CFG = make_config(128)
BAR_LAYOUT = make_bar_layout(CFG)


def validate_config(cfg: Config = CFG, bar_layout=None):
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    bar_layout = make_bar_layout(cfg) if bar_layout is None else bar_layout
    bar_base = cfg.DELTA + 512
    assert (cfg.M, cfg.N, cfg.K_HW, cfg.BPE) == (128, 128, 32, 1) and cfg.D in HEAD_DIMS
    assert (cfg.THREADS, cfg.TMEM_COLS, cfg.Q_STAGES, cfg.P_STAGES, cfg.SCHED_STAGES) == (512, 512, 2, 2, 2)
    assert (cfg.DESC_VERSION, cfg.SWIZZLE_BYTES, cfg.SWIZZLE_LAYOUT, cfg.NARROW_SWIZZLE_LAYOUT) == (1, 128, 2, 4)
    assert cfg.HEAD_BOX == cfg.IN_SWZ == min(cfg.D, 128) and cfg.DQ_BOX == 32 and cfg.SLICE_BYTES == 16384
    assert cfg.OPERAND_BYTES == cfg.M * cfg.D * cfg.BPE and cfg.RING_BYTES == cfg.M * cfg.N * cfg.BPE and cfg.SLICE_BYTES == cfg.M * cfg.DQ_BOX * 4
    assert cfg.DQ_SLICES == cfg.D // 32 and cfg.ALIAS_SLICES >= cfg.DQ_SLICES and cfg.ALIAS_SLICES * cfg.SLICE_BYTES >= 2 * cfg.M * cfg.D * 2
    assert (cfg.TMEM_DV, cfg.TMEM_DK) == (256, 256 + cfg.D) and cfg.TMEM_DK + cfg.D <= cfg.TMEM_COLS
    assert (cfg.SOFTMAX_LANES, cfg.EPILOGUE_LANES, cfg.STORE_LANES, cfg.ONE_LANE, cfg.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 15)
    assert (cfg.SOFTMAX_REGS, cfg.EPILOGUE_REGS, cfg.SERVICE_REGS) == (144, 144, 80)
    assert (cfg.SOFTMAX_LANES * cfg.SOFTMAX_REGS + cfg.EPILOGUE_LANES * cfg.EPILOGUE_REGS + 4 * 32 * cfg.SERVICE_REGS) == 65536
    assert cfg.SOFTMAX_BARRIER == 1
    regions = (
        (cfg.Q, cfg.Q_STAGES * cfg.OPERAND_BYTES),
        (cfg.DO, cfg.OPERAND_BYTES),
        (cfg.K, cfg.OPERAND_BYTES),
        (cfg.V, cfg.OPERAND_BYTES),
        (cfg.ALIAS, cfg.ALIAS_SLICES * cfg.SLICE_BYTES),
        (cfg.P, cfg.P_STAGES * cfg.RING_BYTES),
        (cfg.DS, cfg.P_STAGES * cfg.RING_BYTES),
        (cfg.STATS, 512),
        (cfg.DELTA, 512),
    )
    assert all(start % 1024 == 0 for start, _ in regions[:7])
    assert all(start + size == regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    assert regions[-1][0] + regions[-1][1] == bar_base
    thread_counts = dict(s_free=256, dp_empty=128, p_full=256, ds_full=256, dq_empty=32, dq_full=128, dv_full=128, dk_full=128, dvk_empty=32, sched_empty=15)
    assert len(bar_layout) == 24 and sum(item[2] for item in bar_layout) == 38
    addresses = set()
    for name, offset, stages, count, kind in bar_layout:
        assert stages in (1, 2, 4) and offset % 8 == 0
        assert count == (thread_counts[name] if kind == "thread" else cfg.ONE_LANE)
        assert kind in ("thread", "mma", "tma")
        for i in range(stages):
            address = offset + i * 8
            assert bar_base <= address < cfg.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert cfg.TMEM_HANDLE == bar_base + _BAR_BYTES and cfg.TMEM_HANDLE + 16 == cfg.SCHED_PAYLOAD
    assert (cfg.SCHED_PAYLOAD + _SCHED_BYTES + 1023) // 1024 * 1024 == cfg.SMEM_BYTES <= 227 * 1024


for _d in HEAD_DIMS:
    validate_config(make_config(_d))
