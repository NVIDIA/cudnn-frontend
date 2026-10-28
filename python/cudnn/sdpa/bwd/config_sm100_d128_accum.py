# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense d64 / d128 accumulation geometry."""

from dataclasses import dataclass

HEAD_DIMS = (64, 128)
_SLICE_BYTES = 16384
_BAR_BYTES = 400
_SCHED_BYTES = 64


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
    # --- dkv_accum = fp32_reduce (the grouped THD kernel only): the fp32 dV / dK staging over the ALIAS slab and the TMEM planes it reads ---
    BPE_ACC: int = 4  # fp32 accumulator bytes (the dQ image and the fp32 dK / dV staging)
    D_HALF: int = 64  # one TMEM readout half of a 128-column accumulator plane
    DV_COL: int = 256  # TMEM column of the dV accumulator plane (the bodies' `tmem_base + 256`)
    DK_COL: int = 384  # TMEM column of the dK accumulator plane (the bodies' `tmem_base + 384`)
    DKV_ACC_SLICES: int = 4  # fp32 [M x DQ_BOX] reduce-add slices per [M x D] tile = D // DQ_BOX = the dQ ring's slot count
    DKV_ACC_SLICE_BYTES: int = 16384  # M x DQ_BOX x BPE_ACC = SLICE_BYTES: the four slices fill the 64 KiB ALIAS slab exactly
    ALIAS_SLICES: int = 4
    DQ_SLICES: int = 4
    DO_SUBTILES: int = 2

    def dkv_acc_slot(self, index):
        """Byte base of fp32 staging slice ``index`` of a dV / dK round: the dQ fp32 image's slice ``index`` over the ALIAS slab."""
        return self.ALIAS + index * self.DKV_ACC_SLICE_BYTES


def make_config(d: int) -> Config:
    """The slab geometry of the accumulation kernel at head dimension ``d``."""
    if d not in HEAD_DIMS:
        raise ValueError(f"accumulation head dimension must be one of {HEAD_DIMS}; got {d}")
    operand = 128 * d * 2
    dq_slices = d // 32
    alias_slices = max(2, dq_slices, 2 * operand // _SLICE_BYTES)
    q = 0
    do = q + 2 * operand
    k = do + operand
    v = k + operand
    alias = v + operand
    stats = alias + alias_slices * _SLICE_BYTES
    delta = stats + 512
    bars = delta + 512
    tmem_handle = bars + _BAR_BYTES
    sched = tmem_handle + 16
    smem = (sched + _SCHED_BYTES + 1023) // 1024 * 1024
    return Config(
        D=d,
        HEAD_BOX=64,
        SMEM_BYTES=smem,
        Q=q,
        DO=do,
        K=k,
        V=v,
        ALIAS=alias,
        STATS=stats,
        DELTA=delta,
        TMEM_HANDLE=tmem_handle,
        SCHED_PAYLOAD=sched,
        OPERAND_BYTES=operand,
        ALIAS_SLICES=alias_slices,
        DQ_SLICES=dq_slices,
        DO_SUBTILES=d // 64,
        D_HALF=d // 2,
        DV_COL=512 - 2 * d,
        DK_COL=512 - d,
        DKV_ACC_SLICES=d // 32,
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
        ("do_full", base + 80, 2, 1, "tma"),
        ("q_empty", base + 96, 2, 1, "mma"),
        ("q_full", base + 112, 2, 1, "tma"),
        ("s_done", base + 128, 1, 1, "mma"),
        ("dp_done", base + 144, 1, 1, "mma"),
        ("dq_done", base + 160, 1, 1, "mma"),
        ("dp_empty", base + 192, 1, 128, "thread"),
        ("p_ready", base + 208, 1, 256, "thread"),
        ("ds_ready", base + 224, 1, 256, "thread"),
        ("ds_smem", base + 240, 1, 128, "thread"),
        ("dq_empty", base + 256, 4, 32, "thread"),
        ("dq_full", base + 288, 4, 128, "thread"),
        ("dv_full", base + 320, 1, 128, "thread"),
        ("dk_full", base + 336, 1, 128, "thread"),
        ("dvk_empty", base + 352, 1, 32, "thread"),
        ("sched_full", base + 368, 2, 1, "tma"),
        ("sched_empty", base + 384, 2, 15, "thread"),
    )


CFG = make_config(128)
BAR_LAYOUT = make_bar_layout(CFG)


def validate_config(cfg: Config = CFG, bar_layout=None):
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    bar_layout = make_bar_layout(cfg) if bar_layout is None else bar_layout
    assert (cfg.M, cfg.N, cfg.K_HW, cfg.BPE) == (128, 128, 16, 2) and cfg.D in HEAD_DIMS
    assert (cfg.THREADS, cfg.TMEM_COLS, cfg.Q_STAGES, cfg.SCHED_STAGES) == (512, 512, 2, 2)
    assert (cfg.DESC_VERSION, cfg.SWIZZLE_BYTES, cfg.SWIZZLE_LAYOUT) == (1, 128, 2)
    assert (cfg.HEAD_BOX, cfg.DQ_BOX, cfg.SLICE_BYTES) == (64, 32, 16384)
    assert cfg.OPERAND_BYTES == cfg.M * cfg.D * cfg.BPE and cfg.SLICE_BYTES == cfg.M * cfg.DQ_BOX * 4
    assert cfg.DQ_SLICES == cfg.D // 32 and cfg.DO_SUBTILES == cfg.D // cfg.HEAD_BOX
    assert cfg.ALIAS_SLICES >= max(2, cfg.DQ_SLICES) and cfg.ALIAS_SLICES * cfg.SLICE_BYTES >= 2 * cfg.OPERAND_BYTES
    assert (cfg.SOFTMAX_LANES, cfg.EPILOGUE_LANES, cfg.STORE_LANES, cfg.ONE_LANE, cfg.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 15)
    assert (cfg.SOFTMAX_REGS, cfg.EPILOGUE_REGS, cfg.SERVICE_REGS) == (144, 144, 80)
    assert (cfg.SOFTMAX_LANES * cfg.SOFTMAX_REGS + cfg.EPILOGUE_LANES * cfg.EPILOGUE_REGS + 4 * 32 * cfg.SERVICE_REGS) == 65536
    assert cfg.SOFTMAX_BARRIER == 1
    assert 256 + 2 * cfg.D <= cfg.TMEM_COLS
    regions = (
        (cfg.Q, 2 * cfg.OPERAND_BYTES),
        (cfg.DO, cfg.OPERAND_BYTES),
        (cfg.K, cfg.OPERAND_BYTES),
        (cfg.V, cfg.OPERAND_BYTES),
        (cfg.ALIAS, cfg.ALIAS_SLICES * cfg.SLICE_BYTES),
        (cfg.STATS, 512),
        (cfg.DELTA, 512),
    )
    if cfg.D == 128:
        assert tuple(start for start, _ in regions) == (0, 65536, 98304, 131072, 163840, 229376, 229888)
        assert (cfg.TMEM_HANDLE, cfg.SCHED_PAYLOAD, cfg.SMEM_BYTES) == (230800, 230816, 231424)
    assert all(start + size == regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    bars = regions[-1][0] + regions[-1][1]
    thread_counts = dict(dp_empty=128, p_ready=256, ds_ready=256, ds_smem=128, dq_empty=32, dq_full=128, dv_full=128, dk_full=128, dvk_empty=32, sched_empty=15)
    assert len(bar_layout) == 22 and sum(item[2] for item in bar_layout) == 33
    addresses = set()
    for name, offset, stages, count, kind in bar_layout:
        assert stages in (1, 2, 4) and offset % 8 == 0
        assert count == (thread_counts[name] if kind == "thread" else cfg.ONE_LANE)
        assert kind in ("thread", "mma", "tma")
        for i in range(stages):
            address = offset + i * 8
            assert bars <= address < cfg.TMEM_HANDLE and address not in addresses
            addresses.add(address)
    assert cfg.TMEM_HANDLE == bars + _BAR_BYTES and cfg.TMEM_HANDLE + 4 <= cfg.SCHED_PAYLOAD
    assert cfg.SCHED_PAYLOAD + _SCHED_BYTES <= cfg.SMEM_BYTES <= 227 * 1024
    assert (cfg.SCHED_PAYLOAD + _SCHED_BYTES + 1023) // 1024 * 1024 == cfg.SMEM_BYTES
    # dkv_accum = fp32_reduce (fp32_reduce): the fp32 dV / dK staging = the dQ fp32 image's four [M x DQ_BOX] SW128 slices over the ALIAS slab (one fp32
    # [M x D] tile per round; dV then dK), one dq_empty ring slot per slice (the dK round's WAR waits), the TMEM planes the staging reads.
    assert (cfg.BPE_ACC, cfg.D_HALF, cfg.DKV_ACC_SLICES, cfg.DKV_ACC_SLICE_BYTES) == (4, cfg.D // 2, cfg.D // cfg.DQ_BOX, cfg.M * cfg.DQ_BOX * cfg.BPE_ACC)
    assert cfg.DKV_ACC_SLICE_BYTES == cfg.SLICE_BYTES and cfg.DKV_ACC_SLICES * cfg.DKV_ACC_SLICE_BYTES == 2 * cfg.OPERAND_BYTES == cfg.M * cfg.D * cfg.BPE_ACC
    assert [cfg.dkv_acc_slot(i) for i in range(cfg.DKV_ACC_SLICES)] == [cfg.ALIAS + i * cfg.SLICE_BYTES for i in range(cfg.DKV_ACC_SLICES)]
    assert cfg.DV_COL + cfg.D == cfg.DK_COL and cfg.DK_COL + cfg.D == cfg.TMEM_COLS and cfg.D_HALF * 2 == cfg.D
    if cfg.D == 128:
        assert cfg.SCHED_PAYLOAD + 64 == 230880 and (230880 + 1023) // 1024 * 1024 == cfg.SMEM_BYTES
        assert cfg.DKV_ACC_SLICES == next(item[2] for item in bar_layout if item[0] == "dq_empty")  # one ring slot per slice
        assert (cfg.DV_COL, cfg.DK_COL) == (256, 384)


for _d in HEAD_DIMS:
    validate_config(make_config(_d))
