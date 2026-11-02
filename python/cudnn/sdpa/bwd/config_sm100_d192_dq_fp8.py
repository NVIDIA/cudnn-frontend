# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense per-tensor FP8 dQ geometry for D_QK in {128, 192} with D_V = 128: one 128-row query tile per owner walking 128-row key tiles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    M: int
    N: int
    DQK: int
    DV: int
    K_HW: int
    BPE: int
    HEAD_BOX: int
    QK_SWZ: int
    QK_BOX: int
    V_SWZ: int
    V_BOX: int
    THREADS: int
    TMEM_COLS: int
    TMEM_S: int
    TMEM_DP: int
    TMEM_DQ: int
    Q: int
    DO: int
    K: int
    V: int
    DS: int
    OUT: int
    TMEM_HANDLE: int
    SCHED_PAYLOAD: int
    SMEM_BYTES: int
    Q_BYTES: int
    DO_BYTES: int
    K_BYTES: int
    V_BYTES: int
    RING_BYTES: int
    OUT_BYTES: int
    K_STAGES: int
    V_STAGES: int
    DS_STAGES: int
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
    NARROW_SWIZZLE_BYTES: int
    NARROW_SWIZZLE_LAYOUT: int
    MX: int
    T_SHARED: int
    OUT_SPLIT: int
    K_STAGE_BYTES: int
    KT: int
    KT_STAGES: int
    Q_TX: int
    DO_TX: int
    K_TX: int
    KT_TX: int
    V_TX: int
    SF_QK_BYTES: int
    SF_V_BYTES: int
    SF_KT_BYTES: int
    SF_Q: int
    SF_DO: int
    SF_K: int
    SF_KT: int
    SF_V: int
    SF_DS: int
    SF_COLS_QK: int
    SF_COLS_V: int
    TMEM_SF_Q: int
    TMEM_SF_DO: int
    TMEM_SF_K: int
    TMEM_SF_KT: int
    TMEM_SF_V: int
    TMEM_SF_DS: int


_BAR_SLOTS = (
    ("q_empty", 1, 1, "mma"),
    ("q_full", 1, 1, "tma"),
    ("do_empty", 1, 1, "mma"),
    ("do_full", 1, 1, "tma"),
    ("k_empty", 3, 1, "mma"),
    ("k_full", 3, 1, "tma"),
    ("kt_empty", 2, 1, "mma"),
    ("kt_full", 3, 1, "tma"),
    ("k_used", 3, 1, "mma"),
    ("v_empty", 2, 1, "mma"),
    ("v_full", 2, 1, "tma"),
    ("s_done", 1, 1, "mma"),
    ("dp_done", 1, 1, "mma"),
    ("dq_done", 1, 1, "mma"),
    ("s_free", 1, 256, "thread"),
    ("dp_empty", 1, 256, "thread"),
    ("ds_full", 2, 256, "thread"),
    ("ds_empty", 2, 1, "mma"),
    ("dq_read", 1, 128, "thread"),
    ("out_full", 1, 128, "thread"),
    ("out_empty", 1, 32, "thread"),
    ("sched_full", 2, 1, "tma"),
    ("sched_empty", 2, 15, "thread"),
)


def make_config(dqk, dv, mx=False):
    """The geometry and barrier layout for one (D_QK, D_V) pair; CPU only."""
    if (dqk, dv) not in ((64, 64), (128, 128), (192, 128)):
        raise ValueError("the dQ kernel serves (D_QK, D_V) in (64, 64), (128, 128), (192, 128)")
    m = n = 128
    t_shared = bool(mx) and dqk == 192
    q_bytes, do_bytes, k_bytes, v_bytes, ring = m * dqk, m * dv, n * dqk, n * dv, m * n
    k_stage = k_bytes
    k_stages, kt_stages, v_stages, ds_stages = 3, (2 if (mx and not t_shared) else 0), 2, 2
    groups = (dqk + 127) // 128
    sf_qk, sf_v, sf_kt = 512 * groups, 512, 512 * groups
    q, do = 0, q_bytes
    k = do + do_bytes
    kt = k + k_stages * k_stage
    v = kt + kt_stages * k_bytes
    ds = v + v_stages * v_bytes
    out = ds + ds_stages * ring
    out_bytes = m * (128 if t_shared else dqk) * 2
    sf_q = out + out_bytes
    sf_do = sf_q + (sf_qk if mx else 0)
    sf_k = sf_do + (sf_v if mx else 0)
    sf_kt_off = sf_k + (k_stages * sf_qk if mx else 0)
    sf_v_off = sf_kt_off + ((k_stages if t_shared else kt_stages) * sf_kt if mx else 0)
    sf_ds = sf_v_off + (v_stages * sf_v if mx else 0)
    bar_base = sf_ds + (ds_stages * 512 if mx else 0)
    layout, off = [], bar_base
    for name, slots, count, kind in _BAR_SLOTS:
        layout.append((name, off, slots, count, kind))
        off += 8 * slots
    tmem_handle = (off + 15) // 16 * 16
    sched_payload = tmem_handle + 16
    smem = (sched_payload + 64 + 1023) // 1024 * 1024
    qk_swz = 128 if (dqk % 128) == 0 else 64
    v_swz = 128 if (dv % 128) == 0 else 64
    cols_qk, cols_v = 4 * groups, 4
    t_q = 256 + dqk
    t_do = t_q + cols_qk
    t_k = t_do + cols_v
    t_kt = t_k if t_shared else t_k + k_stages * cols_qk
    t_v = (t_k + k_stages * cols_qk) if t_shared else t_kt + kt_stages * cols_qk
    t_ds = t_v + v_stages * cols_v
    cfg = Config(
        M=m,
        N=n,
        DQK=dqk,
        DV=dv,
        K_HW=32,
        BPE=1,
        HEAD_BOX=128,
        QK_SWZ=qk_swz,
        QK_BOX=qk_swz,
        V_SWZ=v_swz,
        V_BOX=v_swz,
        THREADS=512,
        TMEM_COLS=512,
        TMEM_S=0,
        TMEM_DP=128,
        TMEM_DQ=256,
        Q=q,
        DO=do,
        K=k,
        V=v,
        DS=ds,
        OUT=out,
        TMEM_HANDLE=tmem_handle,
        SCHED_PAYLOAD=sched_payload,
        SMEM_BYTES=smem,
        Q_BYTES=q_bytes,
        DO_BYTES=do_bytes,
        K_BYTES=k_bytes,
        V_BYTES=v_bytes,
        RING_BYTES=ring,
        OUT_BYTES=out_bytes,
        K_STAGES=k_stages,
        V_STAGES=v_stages,
        DS_STAGES=ds_stages,
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
        NARROW_SWIZZLE_BYTES=64,
        NARROW_SWIZZLE_LAYOUT=4,
        MX=int(bool(mx)),
        T_SHARED=int(t_shared),
        OUT_SPLIT=int(t_shared),
        K_STAGE_BYTES=k_stage,
        KT=kt,
        KT_STAGES=kt_stages,
        Q_TX=q_bytes + (sf_qk if mx else 0),
        DO_TX=do_bytes + (sf_v if mx else 0),
        K_TX=k_stage + (sf_qk if mx else 0),
        KT_TX=k_bytes + sf_kt,
        V_TX=v_bytes + (sf_v if mx else 0),
        SF_QK_BYTES=sf_qk,
        SF_V_BYTES=sf_v,
        SF_KT_BYTES=sf_kt,
        SF_Q=sf_q,
        SF_DO=sf_do,
        SF_K=sf_k,
        SF_KT=sf_kt_off,
        SF_V=sf_v_off,
        SF_DS=sf_ds,
        SF_COLS_QK=cols_qk,
        SF_COLS_V=cols_v,
        TMEM_SF_Q=t_q,
        TMEM_SF_DO=t_do,
        TMEM_SF_K=t_k,
        TMEM_SF_KT=t_kt,
        TMEM_SF_V=t_v,
        TMEM_SF_DS=t_ds,
    )
    validate_config(cfg, tuple(layout))
    return cfg, tuple(layout)


def validate_config(cfg, bar_layout):
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    assert (cfg.M, cfg.N, cfg.K_HW, cfg.BPE, cfg.HEAD_BOX) == (128, 128, 32, 1, 128) and (cfg.DQK, cfg.DV) in ((64, 64), (128, 128), (192, 128))
    assert (cfg.THREADS, cfg.TMEM_COLS, cfg.K_STAGES, cfg.KT_STAGES, cfg.V_STAGES, cfg.DS_STAGES, cfg.SCHED_STAGES) == (
        512,
        512,
        3,
        2 if (cfg.MX and not cfg.T_SHARED) else 0,
        2,
        2,
        2,
    )
    assert cfg.T_SHARED == int(bool(cfg.MX) and cfg.DQK == 192) and cfg.OUT_SPLIT == cfg.T_SHARED
    assert cfg.QK_SWZ == cfg.QK_BOX == (128 if cfg.DQK % 128 == 0 else 64) and cfg.V_SWZ == cfg.V_BOX == (128 if cfg.DV % 128 == 0 else 64)
    assert (cfg.TMEM_S, cfg.TMEM_DP, cfg.TMEM_DQ) == (0, 128, 256) and cfg.TMEM_DQ + cfg.DQK <= cfg.TMEM_COLS
    assert cfg.Q_BYTES == cfg.M * cfg.DQK and cfg.DO_BYTES == cfg.M * cfg.DV and cfg.K_BYTES == cfg.N * cfg.DQK and cfg.V_BYTES == cfg.N * cfg.DV
    assert cfg.RING_BYTES == cfg.M * cfg.N and cfg.OUT_BYTES == cfg.M * (128 if cfg.OUT_SPLIT else cfg.DQK) * 2
    assert (cfg.SOFTMAX_LANES * cfg.SOFTMAX_REGS + cfg.EPILOGUE_LANES * cfg.EPILOGUE_REGS + 4 * 32 * cfg.SERVICE_REGS) == 65536
    groups = (cfg.DQK + 127) // 128
    assert cfg.K_STAGE_BYTES == cfg.K_BYTES
    assert (cfg.SF_QK_BYTES, cfg.SF_V_BYTES, cfg.SF_KT_BYTES, cfg.SF_COLS_QK, cfg.SF_COLS_V) == (512 * groups, 512, 512 * groups, 4 * groups, 4)
    assert cfg.Q_TX == cfg.Q_BYTES + (cfg.SF_QK_BYTES if cfg.MX else 0) and cfg.DO_TX == cfg.DO_BYTES + (512 if cfg.MX else 0)
    assert cfg.K_TX == cfg.K_BYTES + (cfg.SF_QK_BYTES if cfg.MX else 0) and cfg.KT_TX == cfg.K_BYTES + cfg.SF_KT_BYTES
    assert cfg.V_TX == cfg.V_BYTES + (512 if cfg.MX else 0)
    kt_tiles = cfg.K_STAGES if cfg.T_SHARED else cfg.KT_STAGES
    sf_bytes = (cfg.SF_QK_BYTES, 512, cfg.K_STAGES * cfg.SF_QK_BYTES, kt_tiles * cfg.SF_KT_BYTES, 1024, 1024) if cfg.MX else (0,) * 6
    regions = (
        (cfg.Q, cfg.Q_BYTES),
        (cfg.DO, cfg.DO_BYTES),
        (cfg.K, cfg.K_STAGES * cfg.K_STAGE_BYTES),
        (cfg.KT, cfg.KT_STAGES * cfg.K_BYTES),
        (cfg.V, cfg.V_STAGES * cfg.V_BYTES),
        (cfg.DS, cfg.DS_STAGES * cfg.RING_BYTES),
        (cfg.OUT, cfg.OUT_BYTES),
        (cfg.SF_Q, sf_bytes[0]),
        (cfg.SF_DO, sf_bytes[1]),
        (cfg.SF_K, sf_bytes[2]),
        (cfg.SF_KT, sf_bytes[3]),
        (cfg.SF_V, sf_bytes[4]),
        (cfg.SF_DS, sf_bytes[5]),
    )
    assert all(start % 1024 == 0 for start, _ in regions[:7]) and all(start % 128 == 0 for start, _ in regions[7:])
    assert all(start + size <= regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    bar_base = bar_layout[0][1]
    assert regions[-1][0] + regions[-1][1] <= bar_base
    if cfg.T_SHARED:
        assert cfg.TMEM_SF_KT == cfg.TMEM_SF_K
        sf_cols = (cfg.SF_COLS_QK, cfg.SF_COLS_V, cfg.K_STAGES * cfg.SF_COLS_QK, 8, 8)
        tmem_sf = (cfg.TMEM_SF_Q, cfg.TMEM_SF_DO, cfg.TMEM_SF_K, cfg.TMEM_SF_V, cfg.TMEM_SF_DS)
    else:
        sf_cols = (cfg.SF_COLS_QK, cfg.SF_COLS_V, cfg.K_STAGES * cfg.SF_COLS_QK, cfg.KT_STAGES * cfg.SF_COLS_QK, 8, 8)
        tmem_sf = (cfg.TMEM_SF_Q, cfg.TMEM_SF_DO, cfg.TMEM_SF_K, cfg.TMEM_SF_KT, cfg.TMEM_SF_V, cfg.TMEM_SF_DS)
    assert tmem_sf[0] == cfg.TMEM_DQ + cfg.DQK and all(tmem_sf[i] + sf_cols[i] == tmem_sf[i + 1] for i in range(len(tmem_sf) - 1))
    assert all(c % 4 == 0 for c in tmem_sf) and (not cfg.MX or tmem_sf[-1] + sf_cols[-1] <= cfg.TMEM_COLS)
    thread_counts = dict(s_free=256, dp_empty=256, ds_full=256, dq_read=128, out_full=128, out_empty=32, sched_empty=15)
    assert len(bar_layout) == 23 and sum(item[2] for item in bar_layout) == 38
    end = bar_base
    for name, offset, slots, count, kind in bar_layout:
        assert offset == end and offset % 8 == 0 and slots in (1, 2, 3) and kind in ("thread", "tma", "mma")
        assert count == (thread_counts[name] if kind == "thread" else 1)
        end += 8 * slots
    assert end <= cfg.TMEM_HANDLE and cfg.TMEM_HANDLE % 16 == 0 and cfg.SCHED_PAYLOAD == cfg.TMEM_HANDLE + 16
    assert (cfg.SCHED_PAYLOAD + 64 + 1023) // 1024 * 1024 == cfg.SMEM_BYTES <= 227 * 1024
