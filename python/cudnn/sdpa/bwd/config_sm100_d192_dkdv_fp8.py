# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense per-tensor FP8 dK / dV geometry for D_QK in {128, 192} with D_V = 128: the key-owner half of the two-kernel backward."""

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
    TMEM_DV: int
    TMEM_DK: int
    TMEM_P: int
    P_COLS: int
    Q: int
    DO: int
    K: int
    V: int
    P: int
    DS: int
    STATS: int
    DELTA: int
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
    Q_STAGES: int
    DO_STAGES: int
    P_STAGES: int
    SCHED_STAGES: int
    SOFTMAX_LANES: int
    EPILOGUE_LANES: int
    STORE_LANES: int
    ONE_LANE: int
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
    P_SMEM: int
    Q_STAGE_BYTES: int
    DO_STAGE_BYTES: int
    QT: int
    QT_STAGES: int
    Q_TX: int
    QT_TX: int
    DO_TX: int
    DOT_TX: int
    K_TX: int
    V_TX: int
    SF_QK_BYTES: int
    SF_V_BYTES: int
    SF_QT_BYTES: int
    SF_K: int
    SF_V: int
    SF_Q: int
    SF_QT: int
    SF_DO: int
    SF_DOT: int
    SF_DS: int
    SF_P: int
    SF_COLS_QK: int
    SF_COLS_V: int
    TMEM_SF_K: int
    TMEM_SF_V: int
    TMEM_SF_Q: int
    TMEM_SF_QT: int
    TMEM_SF_DO: int
    TMEM_SF_DOT: int
    TMEM_SF_DS: int
    TMEM_SF_P: int


_BAR_SLOTS = (
    ("k_empty", 1, 32, "thread"),
    ("k_full", 1, 1, "tma"),
    ("v_full", 1, 1, "tma"),
    ("do_empty", 2, 1, "mma"),
    ("do_full", 2, 1, "tma"),
    ("q_empty", 3, 1, "mma"),
    ("q_full", 3, 1, "tma"),
    ("qt_empty", 2, 1, "mma"),
    ("qt_full", 3, 1, "tma"),
    ("q_used", 3, 1, "mma"),
    ("do_used", 2, 1, "mma"),
    ("dot_full", 2, 1, "tma"),
    ("s_done", 1, 1, "mma"),
    ("dp_done", 1, 1, "mma"),
    ("dvk_done", 1, 1, "mma"),
    ("s_free", 1, 256, "thread"),
    ("dp_empty", 1, 256, "thread"),
    ("p_full", 2, 256, "thread"),
    ("p_empty", 2, 1, "mma"),
    ("ds_full", 2, 256, "thread"),
    ("ds_empty", 2, 1, "mma"),
    ("dv_full", 1, 128, "thread"),
    ("dk_full", 1, 128, "thread"),
    ("dvk_empty", 1, 32, "thread"),
    ("sched_full", 2, 1, "tma"),
    ("sched_empty", 2, 15, "thread"),
    ("stats_full", 2, 256, "thread"),
)


def make_config(dqk, dv, mx=False):
    """The geometry and barrier layout for one (D_QK, D_V) pair; CPU only."""
    if (dqk, dv) not in ((64, 64), (128, 128), (192, 128)):
        raise ValueError("the dK / dV kernel serves (D_QK, D_V) in (64, 64), (128, 128), (192, 128)")
    m = n = 128
    t_shared = bool(mx) and dqk == 192
    p_smem = t_shared
    p_region = (not mx) or t_shared
    q_bytes, do_bytes, k_bytes, v_bytes, ring = n * dqk, n * dv, m * dqk, m * dv, m * n
    q_stage, do_stage = q_bytes, do_bytes * (2 if (mx and not t_shared) else 1)
    q_stages, qt_stages, do_stages, p_stages = 3, (2 if (mx and not t_shared) else 0), 2, 2
    groups = (dqk + 127) // 128
    sf_qk, sf_v, sf_qt = 512 * groups, 512, 512 * groups
    q = 0
    qt = q_stages * q_stage
    do = qt + qt_stages * q_bytes
    k = do + do_stages * do_stage
    v = k + k_bytes
    p = v + v_bytes
    ds = p + (p_stages * ring if p_region else 0)
    stats = ds + p_stages * ring
    delta = stats + 1024
    sf_k = delta + 1024
    sf_v_off = sf_k + (sf_qk if mx else 0)
    sf_q = sf_v_off + (sf_v if mx else 0)
    sf_qt_off = sf_q + (q_stages * sf_qk if mx else 0)
    sf_do = sf_qt_off + ((q_stages if t_shared else qt_stages) * sf_qt if mx else 0)
    sf_dot = sf_do + (do_stages * sf_v if mx else 0)
    sf_ds = sf_dot + (do_stages * sf_v if mx else 0)
    sf_p = sf_ds + (p_stages * 512 if mx else 0)
    bar_base = sf_p + (512 if mx else 0)
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
    tmem_sf = 128 + dv + dqk
    t_k = tmem_sf
    t_v = t_k + cols_qk
    t_q = t_v + cols_v
    t_qt = t_q if t_shared else t_q + q_stages * cols_qk
    t_do = (t_q + q_stages * cols_qk) if t_shared else t_qt + qt_stages * cols_qk
    t_dot = t_do if t_shared else t_do + do_stages * cols_v
    t_ds = t_dot + do_stages * cols_v
    t_p = t_ds + p_stages * cols_v
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
        TMEM_DP=0,
        TMEM_DV=128,
        TMEM_DK=128 + dv,
        TMEM_P=448,
        P_COLS=32,
        Q=q,
        DO=do,
        K=k,
        V=v,
        P=p,
        DS=ds,
        STATS=stats,
        DELTA=delta,
        OUT=k,
        TMEM_HANDLE=tmem_handle,
        SCHED_PAYLOAD=sched_payload,
        SMEM_BYTES=smem,
        Q_BYTES=q_bytes,
        DO_BYTES=do_bytes,
        K_BYTES=k_bytes,
        V_BYTES=v_bytes,
        RING_BYTES=ring,
        OUT_BYTES=m * (dv + dqk) * 2,
        Q_STAGES=q_stages,
        DO_STAGES=do_stages,
        P_STAGES=p_stages,
        SCHED_STAGES=2,
        SOFTMAX_LANES=256,
        EPILOGUE_LANES=128,
        STORE_LANES=32,
        ONE_LANE=1,
        SCHED_CONSUMER_WARPS=15,
        SOFTMAX_REGS=160,
        EPILOGUE_REGS=96,
        SERVICE_REGS=96,
        DESC_VERSION=1,
        SWIZZLE_BYTES=128,
        SWIZZLE_LAYOUT=2,
        NARROW_SWIZZLE_BYTES=64,
        NARROW_SWIZZLE_LAYOUT=4,
        MX=int(bool(mx)),
        T_SHARED=int(t_shared),
        P_SMEM=int(p_smem),
        Q_STAGE_BYTES=q_stage,
        DO_STAGE_BYTES=do_stage,
        QT=qt,
        QT_STAGES=qt_stages,
        Q_TX=q_stage + (sf_qk if mx else 0),
        QT_TX=q_bytes + sf_qt,
        DO_TX=do_stage + ((sf_v if t_shared else 2 * sf_v) if mx else 0),
        DOT_TX=do_bytes + sf_v,
        K_TX=k_bytes + (sf_qk if mx else 0),
        V_TX=v_bytes + (sf_v if mx else 0),
        SF_QK_BYTES=sf_qk,
        SF_V_BYTES=sf_v,
        SF_QT_BYTES=sf_qt,
        SF_K=sf_k,
        SF_V=sf_v_off,
        SF_Q=sf_q,
        SF_QT=sf_qt_off,
        SF_DO=sf_do,
        SF_DOT=sf_dot,
        SF_DS=sf_ds,
        SF_P=sf_p,
        SF_COLS_QK=cols_qk,
        SF_COLS_V=cols_v,
        TMEM_SF_K=t_k,
        TMEM_SF_V=t_v,
        TMEM_SF_Q=t_q,
        TMEM_SF_QT=t_qt,
        TMEM_SF_DO=t_do,
        TMEM_SF_DOT=t_dot,
        TMEM_SF_DS=t_ds,
        TMEM_SF_P=t_p,
    )
    validate_config(cfg, tuple(layout))
    return cfg, tuple(layout)


def validate_config(cfg, bar_layout):
    """Fixed source geometry, lane counts and disjoint slab regions; CPU only."""
    assert (cfg.M, cfg.N, cfg.K_HW, cfg.BPE, cfg.HEAD_BOX) == (128, 128, 32, 1, 128) and (cfg.DQK, cfg.DV) in ((64, 64), (128, 128), (192, 128))
    assert (cfg.THREADS, cfg.TMEM_COLS, cfg.Q_STAGES, cfg.QT_STAGES, cfg.DO_STAGES, cfg.P_STAGES, cfg.SCHED_STAGES) == (
        512,
        512,
        3,
        2 if (cfg.MX and not cfg.T_SHARED) else 0,
        2,
        2,
        2,
    )
    assert cfg.T_SHARED == int(bool(cfg.MX) and cfg.DQK == 192) and cfg.P_SMEM == cfg.T_SHARED
    assert (cfg.DESC_VERSION, cfg.SWIZZLE_BYTES, cfg.SWIZZLE_LAYOUT, cfg.NARROW_SWIZZLE_BYTES, cfg.NARROW_SWIZZLE_LAYOUT) == (1, 128, 2, 64, 4)
    assert cfg.QK_SWZ == cfg.QK_BOX == (128 if cfg.DQK % 128 == 0 else 64) and cfg.V_SWZ == cfg.V_BOX == (128 if cfg.DV % 128 == 0 else 64)
    assert cfg.Q_BYTES == cfg.N * cfg.DQK * cfg.BPE and cfg.DO_BYTES == cfg.N * cfg.DV * cfg.BPE
    assert cfg.K_BYTES == cfg.M * cfg.DQK * cfg.BPE and cfg.V_BYTES == cfg.M * cfg.DV * cfg.BPE
    assert cfg.RING_BYTES == cfg.M * cfg.N * cfg.BPE and cfg.OUT_BYTES == cfg.M * (cfg.DV + cfg.DQK) * 2
    assert (cfg.TMEM_S, cfg.TMEM_DP, cfg.TMEM_DV) == (0, 0, 128) and cfg.TMEM_DV + cfg.DV == cfg.TMEM_DK
    assert cfg.TMEM_DK + cfg.DQK <= cfg.TMEM_P and cfg.P_COLS * 4 == cfg.N and cfg.TMEM_P + cfg.P_STAGES * cfg.P_COLS == cfg.TMEM_COLS
    assert all(c % 4 == 0 for c in (cfg.TMEM_DV, cfg.TMEM_DK))
    assert (cfg.SOFTMAX_LANES, cfg.EPILOGUE_LANES, cfg.STORE_LANES, cfg.ONE_LANE, cfg.SCHED_CONSUMER_WARPS) == (256, 128, 32, 1, 15)
    assert (cfg.SOFTMAX_REGS, cfg.EPILOGUE_REGS, cfg.SERVICE_REGS) == (160, 96, 96)
    assert (cfg.SOFTMAX_LANES * cfg.SOFTMAX_REGS + cfg.EPILOGUE_LANES * cfg.EPILOGUE_REGS + 4 * 32 * cfg.SERVICE_REGS) == 65536
    groups = (cfg.DQK + 127) // 128
    ring_mx = cfg.MX and not cfg.T_SHARED
    assert cfg.Q_STAGE_BYTES == cfg.Q_BYTES and cfg.DO_STAGE_BYTES == cfg.DO_BYTES * (2 if ring_mx else 1)
    assert (cfg.SF_QK_BYTES, cfg.SF_V_BYTES, cfg.SF_QT_BYTES, cfg.SF_COLS_QK, cfg.SF_COLS_V) == (512 * groups, 512, 512 * groups, 4 * groups, 4)
    qt_tiles = cfg.Q_STAGES if cfg.T_SHARED else cfg.QT_STAGES
    sf_bytes = (cfg.SF_QK_BYTES, cfg.SF_V_BYTES, cfg.Q_STAGES * cfg.SF_QK_BYTES, qt_tiles * cfg.SF_QT_BYTES, 1024, 1024, 1024, 512) if cfg.MX else (0,) * 8
    assert cfg.Q_TX == cfg.Q_BYTES + (cfg.SF_QK_BYTES if cfg.MX else 0) and cfg.QT_TX == cfg.Q_BYTES + cfg.SF_QT_BYTES
    assert cfg.DO_TX == cfg.DO_STAGE_BYTES + ((512 if cfg.T_SHARED else 1024) if cfg.MX else 0) and cfg.DOT_TX == cfg.DO_BYTES + 512
    assert cfg.K_TX == cfg.K_BYTES + (cfg.SF_QK_BYTES if cfg.MX else 0) and cfg.V_TX == cfg.V_BYTES + (512 if cfg.MX else 0)
    regions = (
        (cfg.Q, cfg.Q_STAGES * cfg.Q_STAGE_BYTES),
        (cfg.QT, cfg.QT_STAGES * cfg.Q_BYTES),
        (cfg.DO, cfg.DO_STAGES * cfg.DO_STAGE_BYTES),
        (cfg.K, cfg.K_BYTES),
        (cfg.V, cfg.V_BYTES),
        (cfg.P, cfg.P_STAGES * cfg.RING_BYTES if ((not cfg.MX) or cfg.T_SHARED) else 0),
        (cfg.DS, cfg.P_STAGES * cfg.RING_BYTES),
        (cfg.STATS, 1024),
        (cfg.DELTA, 1024),
        (cfg.SF_K, sf_bytes[0]),
        (cfg.SF_V, sf_bytes[1]),
        (cfg.SF_Q, sf_bytes[2]),
        (cfg.SF_QT, sf_bytes[3]),
        (cfg.SF_DO, sf_bytes[4]),
        (cfg.SF_DOT, sf_bytes[5]),
        (cfg.SF_DS, sf_bytes[6]),
        (cfg.SF_P, sf_bytes[7]),
    )
    assert all(start + size <= regions[i + 1][0] for i, (start, size) in enumerate(regions[:-1]))
    assert all(start % 1024 == 0 for start, _ in regions[:7]) and all(start % 128 == 0 for start, _ in regions[9:])
    bar_base = bar_layout[0][1]
    assert regions[-1][0] + regions[-1][1] <= bar_base
    assert cfg.OUT == cfg.K and cfg.OUT + cfg.OUT_BYTES <= cfg.STATS and cfg.V == cfg.K + cfg.K_BYTES and cfg.P == cfg.V + cfg.V_BYTES
    if cfg.T_SHARED:
        assert cfg.TMEM_SF_QT == cfg.TMEM_SF_Q and cfg.TMEM_SF_DOT == cfg.TMEM_SF_DO
        sf_cols = (cfg.SF_COLS_QK, cfg.SF_COLS_V, cfg.Q_STAGES * cfg.SF_COLS_QK, 8, 8, 4)
        tmem_sf = (cfg.TMEM_SF_K, cfg.TMEM_SF_V, cfg.TMEM_SF_Q, cfg.TMEM_SF_DO, cfg.TMEM_SF_DS, cfg.TMEM_SF_P)
    else:
        sf_cols = (cfg.SF_COLS_QK, cfg.SF_COLS_V, cfg.Q_STAGES * cfg.SF_COLS_QK, cfg.QT_STAGES * cfg.SF_COLS_QK, 8, 8, 8, 4)
        tmem_sf = (cfg.TMEM_SF_K, cfg.TMEM_SF_V, cfg.TMEM_SF_Q, cfg.TMEM_SF_QT, cfg.TMEM_SF_DO, cfg.TMEM_SF_DOT, cfg.TMEM_SF_DS, cfg.TMEM_SF_P)
    assert tmem_sf[0] == cfg.TMEM_DK + cfg.DQK and all(tmem_sf[i] + sf_cols[i] == tmem_sf[i + 1] for i in range(len(tmem_sf) - 1))
    sf_end = tmem_sf[-1] + sf_cols[-1]
    assert all(c % 4 == 0 for c in tmem_sf) and (not cfg.MX or sf_end <= (cfg.TMEM_COLS if cfg.P_SMEM else cfg.TMEM_P))
    thread_counts = dict(k_empty=32, s_free=256, dp_empty=256, p_full=256, ds_full=256, dv_full=128, dk_full=128, dvk_empty=32, sched_empty=15, stats_full=256)
    assert len(bar_layout) == 27 and sum(item[2] for item in bar_layout) == 47
    end = bar_base
    for name, offset, slots, count, kind in bar_layout:
        assert offset == end and offset % 8 == 0 and slots in (1, 2, 3) and kind in ("thread", "tma", "mma")
        assert count == (thread_counts[name] if kind == "thread" else 1)
        end += 8 * slots
    assert end <= cfg.TMEM_HANDLE and cfg.TMEM_HANDLE % 16 == 0 and cfg.SCHED_PAYLOAD == cfg.TMEM_HANDLE + 16
    assert (cfg.SCHED_PAYLOAD + 64 + 1023) // 1024 * 1024 == cfg.SMEM_BYTES <= 227 * 1024
