# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Support rules and static per-case facts for the DSA benchmark suite.

Every architecture / dtype / shape / API-existence skip rule of the suite
lives in :func:`decide_support`, a pure function of the case fields and the
compute capability. The runner evaluates it before spawning a worker (a
skipped row costs no subprocess) and the worker re-evaluates it so a manual
invocation gets the same answer; the worker keeps its exit-3 path only for
declines the kernels raise themselves. The reasons are the suite's own
wording, so no kernel message reaches a CSV.

This module imports nothing beyond the standard library: the runner may run
on a machine without torch or CUDA. The closed-form byte footprints
(:func:`record_bytes`, :func:`index_key_bytes`) are defined here for the same
reason and re-exported by :mod:`kv_pool`, which owns the torch generators.

A *case* is a flat ``dict`` carrying the preset geometry (``num_q_heads``,
``head_dim_qk``, ``head_dim_vo``, ``topk``, ``indexer_topk``,
``num_kv_heads``, ``kv_layout``, ``index_granularity``, ``indexer_heads``,
``indexer_pool_ratio``, ``mtp_rows``) and the case axes (``component``,
``phase``, ``kv_dtype``, ``paged``, ``indexer_mode``, ``data_type``,
``q_seqlen``, ``kv_seqlen``, ``batch_size``). The runner's expanded case
(with a ``model`` preset and ``profile_pass``) is accepted too: missing
geometry is read from the preset and missing axes take the legacy defaults.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# (num_q_heads, head_dim_qk) -> indexer_topk values the sparse core forward kernels are built
# for. Mirrors the forward API's variant table; the core rows of this suite are scoped to these
# pairs so context and bprop rows pair up.
CORE_VARIANTS: Dict[Tuple[int, int], Tuple[int, ...]] = {
    (64, 512): (0, 512, 1024, 2048),
    (64, 576): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024),
}

# Hierarchical indexer (DSv4.1 after layer 20): the consumer scores a fixed candidate list of
# CANDIDATE_COUNT ids chosen by the source layer in CANDIDATE_BLOCK-token blocks.
CANDIDATE_COUNT = 16384
CANDIDATE_BLOCK = 8

# Rows compiled through the shared CuTe DSL compiler (one target per device). The sparse core
# forward compiles its own sm_100f image on every 10.x part and is exempt.
SHARED_COMPILER_ROWS: FrozenSet[Tuple[str, str]] = frozenset(
    {
        ("core", "bprop"),
        ("indexer", "context"),
        ("indexer", "generation"),
        ("indexer", "bprop"),
    }
)

PHASES = ("context", "generation", "bprop")
PROFILE_PASS_FOR_PHASE = {"context": "fwd", "generation": "fwd", "bprop": "bwd"}
_PHASE_FOR_PROFILE_PASS = {"fwd": "context", "bwd": "bprop"}

_GEOMETRY_DEFAULTS = dict(
    head_dim_qk=512,
    head_dim_vo=512,
    topk=2048,
    indexer_topk=0,
    num_kv_heads=1,
    kv_layout="shared",
    index_granularity=1,
    indexer_heads=0,
    indexer_pool_ratio=1,
    mtp_rows=1,
)
_AXIS_DEFAULTS = dict(component="core", kv_dtype="bfloat16", paged=False, indexer_mode=None, batch_size=1, data_type="bfloat16")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute of a preset-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def indexer_topk_select(preset_like: Any) -> int:
    """Index entries the indexer selects per query.

    ``indexer_topk`` when the preset sets one (the prefix the core kernel
    also emits an LSE over), otherwise ``topk // index_granularity``: 512 for
    DSv4.1, 2048 for GLM-5.3, 512 four-token blocks for GLM-5.3-Flash and QSA.
    Accepts a :class:`ModelPreset` or a flat case ``dict``.
    """
    indexer_topk = int(_get(preset_like, "indexer_topk", 0) or 0)
    if indexer_topk:
        return indexer_topk
    return int(_get(preset_like, "topk")) // int(_get(preset_like, "index_granularity", 1) or 1)


def case_fields(case: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a runner case into the field dict :func:`decide_support` reads.

    Geometry missing from ``case`` is read from ``case["model"]`` when the
    runner's preset is attached, else from the legacy defaults; ``phase``
    falls back to ``profile_pass`` (fwd -> context, bwd -> bprop) and the new
    axes to their legacy defaults, so a legacy case needs no extra keys.
    """
    model = case.get("model")
    fields: Dict[str, Any] = {}
    for key, default in _GEOMETRY_DEFAULTS.items():
        fields[key] = case[key] if key in case else _get(model, key, default)
    fields["num_q_heads"] = case["num_q_heads"] if "num_q_heads" in case else _get(model, "num_q_heads")
    for key, default in _AXIS_DEFAULTS.items():
        fields[key] = case.get(key, default)
    fields["phase"] = case.get("phase") or _PHASE_FOR_PROFILE_PASS.get(case.get("profile_pass", "fwd"), "context")
    fields["q_seqlen"] = case.get("q_seqlen")
    fields["kv_seqlen"] = case.get("kv_seqlen")
    return fields


def uses_shared_compiler(case: Dict[str, Any]) -> bool:
    """True when the row's kernels compile through the shared DSL compiler (everything but the core forward)."""
    f = case_fields(case)
    return (f["component"], f["phase"]) in SHARED_COMPILER_ROWS


def phases_for_config(cfg: Any) -> List[str]:
    """Phases a config enumerates: ``cfg.phases`` or the list ``profile_pass`` implies.

    ``both`` -> ``["context", "bprop"]`` (today's fwd then bwd), ``fwd`` ->
    ``["context"]``, ``bwd`` -> ``["bprop"]``.
    """
    phases = _get(cfg, "phases")
    if phases is not None:
        return list(phases)
    profile_pass = _get(cfg, "profile_pass", "both")
    if profile_pass == "both":
        return ["context", "bprop"]
    if profile_pass in _PHASE_FOR_PROFILE_PASS:
        return [_PHASE_FOR_PROFILE_PASS[profile_pass]]
    raise ValueError(f"profile_pass must be fwd, bwd or both, got {profile_pass!r}")


def record_bytes(head_dim: int, kv_dtype: str, kv_layout: str = "shared", num_kv_heads: int = 1) -> int:
    """Per-token bytes of the core KV record in storage format ``kv_dtype``.

    Shared record (K = V latent): ``bfloat16`` 2 * D; ``mxfp8`` D e4m3 + D/32
    e8m0; ``nvfp4`` D/2 e2m1 + D/16 e4m3 block scales. Separate K and V:
    ``num_kv_heads * 2`` such records. 512 -> 1024 / 528 / 288, 576 -> 1152 /
    594 / 324, QSA (256, separate, 2 heads) -> 2048 / 1056 / 576.
    """
    d = int(head_dim)
    if kv_dtype in ("bfloat16", "float16"):
        one = 2 * d
    elif kv_dtype == "mxfp8":
        one = d + d // 32
    elif kv_dtype == "nvfp4":
        one = d // 2 + d // 16
    else:
        raise ValueError(f"kv_dtype must be bfloat16, mxfp8 or nvfp4, got {kv_dtype!r}")
    if kv_layout == "shared":
        return one
    if kv_layout == "separate":
        return int(num_kv_heads) * 2 * one
    raise ValueError(f"kv_layout must be shared or separate, got {kv_layout!r}")


def index_key_bytes(kv_dtype: str, head_dim: int = 128, pool_ratio: int = 1) -> float:
    """Per-token amortized bytes of one index key (the indexer's K pool).

    ``bfloat16`` 2 * D = 256; ``mxfp8`` D + D/32 = 132 (e4m3 + e8m0 per 32);
    ``nvfp4`` D/2 + D/32 = 68, the MXFP4 (e2m1 + e8m0 per 32) production
    format of the DSv4.1 indexer. One key covers ``pool_ratio`` tokens
    (64 / 33 / 17 at ratio 4).
    """
    d = int(head_dim)
    if kv_dtype in ("bfloat16", "float16"):
        one = 2 * d
    elif kv_dtype == "mxfp8":
        one = d + d // 32
    elif kv_dtype == "nvfp4":
        one = d // 2 + d // 32
    else:
        raise ValueError(f"kv_dtype must be bfloat16, mxfp8 or nvfp4, got {kv_dtype!r}")
    return one / int(pool_ratio)


def bytes_per_token(case: Dict[str, Any]) -> int:
    """Static per-token footprint of the row: the KV record for core rows, the amortized index key for indexer rows."""
    f = case_fields(case)
    if f["component"] == "indexer":
        return int(round(index_key_bytes(f["kv_dtype"], 128, f["indexer_pool_ratio"])))
    return record_bytes(f["head_dim_qk"], f["kv_dtype"], f["kv_layout"], f["num_kv_heads"])


def static_columns(case: Dict[str, Any]) -> Dict[str, Any]:
    """The appended :class:`BenchmarkResult` columns known before any kernel runs (every row, skipped ones included)."""
    f = case_fields(case)
    return dict(
        component=f["component"],
        phase=f["phase"],
        kv_dtype=f["kv_dtype"],
        paged=bool(f["paged"]),
        batch_size=int(f["batch_size"]),
        indexer_mode=f["indexer_mode"] if f["component"] == "indexer" else None,
        bytes_per_token=bytes_per_token(f),
    )


def dsl_target_error(cc: Optional[Tuple[int, int]]) -> Optional[str]:
    """Message when the installed CuTe DSL cannot target ``cc`` (or is below the package floor), else ``None``.

    The one probe both the runner (once per run) and the worker consult; only
    rule 15 reads it, on compute capability (10, 7) rows that compile through
    the shared compiler. ``cudnn`` is imported lazily here so this module stays
    importable without it; any import or probe failure reads as no verdict.
    """
    if cc is None:
        return None
    try:
        from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

        message = cutedsl_arch_requirement_error(cc)
        if message:
            return str(message)
        _installed, version = cutedsl_state()
        return "CuTe DSL too old" if cutedsl_too_old(version) else None
    except Exception:
        return None


def decide_support(case: Dict[str, Any], cc: Optional[Tuple[int, int]], dsl_target_error: Optional[str] = None) -> Optional[str]:
    """Reason this case is not served by the suite's kernels, or ``None`` when it should run.

    First matching rule wins. ``cc`` is the device's compute capability as
    ``(major, minor)``; ``None`` (no torch or no CUDA in the caller) is
    treated as no capable GPU, so every architecture-gated row skips.
    ``dsl_target_error`` is the runner's one-time probe of the installed CuTe
    DSL (``cudnn.frost.buffers.cutedsl_arch_requirement_error(cc)``, or a
    too-old marker); it only matters on compute capability (10, 7) for rows
    that compile through the shared compiler. Reasons are the suite's own
    wording and never quote a kernel message.
    """
    f = case_fields(case)
    component, phase, kv_dtype, paged, mode = f["component"], f["phase"], f["kv_dtype"], bool(f["paged"]), f["indexer_mode"]
    major = cc[0] if cc is not None else None

    # Storage formats and precision.
    if kv_dtype == "nvfp4":
        return "no cuDNN FE kernel consumes NVFP4 KV or MXFP4 index keys (storage-format coverage row)"
    if component == "core" and kv_dtype == "mxfp8":
        return "sparse core attention kernels take bf16/fp16 KV; no MXFP8 KV core kernel"
    if phase == "bprop" and kv_dtype != "bfloat16":
        return "no low-precision backward kernel"

    # Core geometry.
    if component == "core" and f["kv_layout"] == "separate":
        return "QSA-shaped GQA d256 separate-K/V sparse attention is not wired in this suite (cc 10.7 sparse GQA kernel is a follow-up)"
    if component == "core":
        pair = (int(f["num_q_heads"]), int(f["head_dim_qk"]))
        if pair not in CORE_VARIANTS:
            # The backward kernels accept more head counts, but core rows are scoped to the forward
            # variant table so every context row has its bprop twin.
            return f"sparse core kernels serve (num_q_heads, head_dim_qk) in {sorted(CORE_VARIANTS)}, not {pair} (core rows are scoped to the forward variants)"
        if int(f["indexer_topk"]) not in CORE_VARIANTS[pair]:
            return f"indexer_topk={f['indexer_topk']} is not a compiled prefix for (num_q_heads, head_dim_qk)={pair}; kernels take {CORE_VARIANTS[pair]}"

    # Core architecture.
    if component == "core" and phase in ("context", "generation") and major != 10:
        return "sparse core forward requires compute capability 10.x"
    if component == "core" and phase == "bprop" and major not in (9, 10):
        return "sparse core backward requires compute capability 9.x or 10.x"

    # Indexer rows.
    if component == "indexer":
        heads = int(f["indexer_heads"])
        if heads == 0:
            return "model has no indexer rows"
        if heads not in (32, 64):
            return f"indexer kernels require 32 or 64 query heads per KV head (model has {heads})"
        if paged:
            return "indexer kernels take dense BSHD/THD keys; no paged index-key input"
        if phase in ("context", "generation") and major != 10:
            return "indexer forward/top-k fused kernel requires compute capability 10.x"
        if phase == "bprop" and major not in (9, 10):
            return "indexer backward requires compute capability 9.x or 10.x"
        if phase == "bprop" and major == 9 and heads < 64:
            return f"indexer backward on compute capability 9.x requires at least 64 indexer heads (model has {heads})"
        if mode == "hierarchical" and kv_dtype != "bfloat16":
            return "candidate-restricted indexer scoring is bf16-only"
        if mode == "hierarchical" and phase == "bprop":
            return "hierarchical rows are forward-only"

    # Per-request top-k against the pool (core rows: the gather lists hold ``topk`` unique ids of one
    # request). Indexer rows never read ``topk``: they select ``indexer_topk_select`` entries and pad
    # rows with fewer visible keys, exactly as every short causal context row is served.
    topk, kv_seqlen = int(f["topk"]), int(f["kv_seqlen"])
    if component == "core" and topk > kv_seqlen:
        if phase == "generation":
            return f"per-request kv_seqlen ({kv_seqlen}) is shorter than topk ({topk})"
        return f"topk ({topk}) exceeds kv_seqlen ({kv_seqlen})"

    # Shared-compiler target on compute capability (10, 7).
    if cc == (10, 7) and dsl_target_error and (component, phase) in SHARED_COMPILER_ROWS:
        return "installed CuTe DSL lacks the target for compute capability 10.7"
    return None
