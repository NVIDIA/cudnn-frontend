# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d64 / d128 query-owner dQ geometry: one 128-row query tile per owner walking 128-row key tiles."""

from dataclasses import dataclass

HEAD_DIMS = (64, 128)
_BAR_SLOTS = (
    ("q_empty", 1, 1, "mma"),
    ("q_full", 1, 1, "tma"),
    ("do_empty", 1, 1, "mma"),
    ("do_full", 1, 1, "tma"),
    ("k_empty", "K_STAGES", 1, "mma"),
    ("k_full", "K_STAGES", 1, "tma"),
    ("v_empty", "V_STAGES", 1, "mma"),
    ("v_full", "V_STAGES", 1, "tma"),
    ("s_done", 1, 1, "mma"),
    ("dp_done", 1, 1, "mma"),
    ("dq_done", 1, 1, "mma"),
    ("s_free", 1, 256, "thread"),
    ("ds_ready", 1, 256, "thread"),
    ("dq_read", 1, 128, "thread"),
    ("sched_full", 2, 1, "tma"),
    ("sched_empty", 2, 15, "thread"),
)


@dataclass(frozen=True)
class Config:
    M: int
    N: int
    D: int
    K_HW: int
    BPE: int
    HEAD_BOX: int
    THREADS: int
    TMEM_COLS: int
    TMEM_S: int
    TMEM_DP: int
    TMEM_DQ: int
    Q: int
    DO: int
    K: int
    V: int
    BARS: int
    TMEM_HANDLE: int
    SCHED_PAYLOAD: int
    SMEM_BYTES: int
    OPERAND_BYTES: int
    SLICE_BYTES: int
    K_STAGES: int
    V_STAGES: int
    SCHED_STAGES: int
    SOFTMAX_LANES: int
    EPILOGUE_LANES: int
    STORE_LANES: int
    SCHED_CONSUMER_WARPS: int
    SOFTMAX_REGS: int
    EPILOGUE_REGS: int
    SERVICE_REGS: int
    DESC_VERSION: int
    SWIZZLE_BYTES: int
    SWIZZLE_LAYOUT: int


def _slots(slots, k_stages, v_stages):
    return {"K_STAGES": k_stages, "V_STAGES": v_stages}.get(slots, slots)


def make_config(d: int) -> Config:
    """The slab geometry of the dQ kernel at head dimension ``d``; CPU only."""
    if d not in HEAD_DIMS:
        raise ValueError(f"dQ head dimension must be one of {HEAD_DIMS}; got {d}")
    operand = 128 * d * 2
    k_stages, v_stages = 3, 2
    q = 0
    do = q + operand
    k = do + operand
    v = k + k_stages * operand
    bars = v + v_stages * operand
    bar_bytes = 8 * sum(_slots(slots, k_stages, v_stages) for _, slots, _, _ in _BAR_SLOTS)
    tmem_handle = (bars + bar_bytes + 15) // 16 * 16
    sched = tmem_handle + 16
    smem = (sched + 64 + 1023) // 1024 * 1024
    return Config(
        M=128,
        N=128,
        D=d,
        K_HW=16,
        BPE=2,
        HEAD_BOX=64,
        THREADS=512,
        TMEM_COLS=512,
        TMEM_S=0,
        TMEM_DP=128,
        TMEM_DQ=256,
        Q=q,
        DO=do,
        K=k,
        V=v,
        BARS=bars,
        TMEM_HANDLE=tmem_handle,
        SCHED_PAYLOAD=sched,
        SMEM_BYTES=smem,
        OPERAND_BYTES=operand,
        SLICE_BYTES=16384,
        K_STAGES=k_stages,
        V_STAGES=v_stages,
        SCHED_STAGES=2,
        SOFTMAX_LANES=256,
        EPILOGUE_LANES=128,
        STORE_LANES=32,
        SCHED_CONSUMER_WARPS=15,
        SOFTMAX_REGS=144,
        EPILOGUE_REGS=128,
        SERVICE_REGS=96,
        DESC_VERSION=1,
        SWIZZLE_BYTES=128,
        SWIZZLE_LAYOUT=2,
    )


def make_bar_layout(cfg: Config):
    """name, byte offset, active slots, init count, producer kind."""
    layout, off = [], cfg.BARS
    for name, slots, count, kind in _BAR_SLOTS:
        slots = _slots(slots, cfg.K_STAGES, cfg.V_STAGES)
        layout.append((name, off, slots, count, kind))
        off += 8 * slots
    return tuple(layout)


CFG = make_config(128)
BAR_LAYOUT = make_bar_layout(CFG)


def validate_config(cfg: Config = CFG, bar_layout=None):
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    bar_layout = make_bar_layout(cfg) if bar_layout is None else bar_layout
    assert (cfg.M, cfg.N, cfg.K_HW, cfg.BPE, cfg.HEAD_BOX) == (128, 128, 16, 2, 64) and cfg.D in HEAD_DIMS
    assert (cfg.THREADS, cfg.TMEM_COLS, cfg.K_STAGES, cfg.V_STAGES, cfg.SCHED_STAGES) == (512, 512, 3, 2, 2)
    assert (cfg.DESC_VERSION, cfg.SWIZZLE_BYTES, cfg.SWIZZLE_LAYOUT) == (1, 128, 2)
    assert cfg.OPERAND_BYTES == cfg.M * cfg.D * cfg.BPE and cfg.SLICE_BYTES == cfg.M * cfg.HEAD_BOX * cfg.BPE
    assert cfg.OPERAND_BYTES == (cfg.D // cfg.HEAD_BOX) * cfg.SLICE_BYTES
    assert (cfg.TMEM_S, cfg.TMEM_DP, cfg.TMEM_DQ) == (0, 128, 256) and cfg.TMEM_DQ + cfg.D <= cfg.TMEM_COLS
    assert (cfg.SOFTMAX_LANES, cfg.EPILOGUE_LANES, cfg.STORE_LANES, cfg.SCHED_CONSUMER_WARPS) == (256, 128, 32, 15)
    assert (cfg.SOFTMAX_REGS, cfg.EPILOGUE_REGS, cfg.SERVICE_REGS) == (144, 128, 96)
    assert (cfg.SOFTMAX_LANES * cfg.SOFTMAX_REGS + cfg.EPILOGUE_LANES * cfg.EPILOGUE_REGS + 4 * 32 * cfg.SERVICE_REGS) == 65536
    regions = (
        (cfg.Q, cfg.OPERAND_BYTES),
        (cfg.DO, cfg.OPERAND_BYTES),
        (cfg.K, cfg.K_STAGES * cfg.OPERAND_BYTES),
        (cfg.V, cfg.V_STAGES * cfg.OPERAND_BYTES),
    )
    assert all(start % 1024 == 0 for start, _ in regions)
    assert all(start + size == regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    assert regions[-1][0] + regions[-1][1] == cfg.BARS
    if cfg.D == 128:
        assert tuple(start for start, _ in regions) == (0, 32768, 65536, 163840) and cfg.BARS == 229376
    thread_counts = dict(s_free=256, ds_ready=256, dq_read=128, sched_empty=15)
    assert len(bar_layout) == len(_BAR_SLOTS) == 16 and sum(item[2] for item in bar_layout) == 24
    stage_slots = dict(k_empty=cfg.K_STAGES, k_full=cfg.K_STAGES, v_empty=cfg.V_STAGES, v_full=cfg.V_STAGES, sched_full=2, sched_empty=2)
    end = cfg.BARS
    for name, offset, slots, count, kind in bar_layout:
        assert offset == end and offset % 8 == 0 and slots == stage_slots.get(name, 1) and kind in ("thread", "tma", "mma")
        assert count == (thread_counts[name] if kind == "thread" else 1)
        end += 8 * slots
    assert end <= cfg.TMEM_HANDLE and cfg.TMEM_HANDLE % 16 == 0 and cfg.SCHED_PAYLOAD == cfg.TMEM_HANDLE + 16
    assert (cfg.SCHED_PAYLOAD + 64 + 1023) // 1024 * 1024 == cfg.SMEM_BYTES <= 227 * 1024


for _d in HEAD_DIMS:
    validate_config(make_config(_d))
