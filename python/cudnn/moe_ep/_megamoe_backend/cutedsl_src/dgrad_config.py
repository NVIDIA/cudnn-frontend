"""Public, host-only configuration for the Rubin MegaMoE dgrad kernel.

Resolution never reads process environment or changes requested outputs. Callers
may inspect the returned mapping before constructing a kernel. The kernel also
exposes it as ``resolved_dgrad_config`` after construction.
"""

from collections.abc import Mapping
from enum import Enum


class _Unspecified(Enum):
    VALUE = "unspecified"


# Distinguish an omitted tuning option from an explicit value (including None).
DGRAD_UNSPECIFIED = _Unspecified.VALUE

_DEFAULTS = dict(
    load_balance_mode="static",
    force_static_sched=True,
    clc_bundle_size=None,
    num_sched_stages=None,
    token_back_mode="epi_warps",
    epi_flag_batch=(1, 1),
    flag_batch=1,
    token_in_async_flags=False,
    token_in_transfer_warp_count=4,
    schedule_mode="grouped",
    dfc2_subtile_publish=False,
    dfc2_c_pipe_stages=1,
    dfc2_d_pipe_stages=1,
    dfc2_acc_early_release=False,
    phase_interleave_prologue_tiles=None,
    phase_interleave_defer_consumers_until_full=False,
    phase_interleave_fuse_ready_probe_and_producer_claim=False,
    use_scaled_cvt=False,
    num_ctas_grad_y2_col_quant=2368,
    input_dedup_mode="off",
    dfc1_m_group=1,
)

_ROLLING_SCHEDULE = dict(
    load_balance_mode="atomic_counter",
    schedule_mode="phase_interleave",
    dfc2_subtile_publish=True,
    dfc2_c_pipe_stages=2,
    dfc2_d_pipe_stages=1,
    dfc2_acc_early_release=True,
    phase_interleave_prologue_tiles=0,
    phase_interleave_defer_consumers_until_full=True,
    phase_interleave_fuse_ready_probe_and_producer_claim=True,
)


def _require_values(actual, expected, label):
    differences = [
        f"{key}={actual.get(key)!r} (requires {value!r})"
        for key, value in expected.items() if actual.get(key) != value
    ]
    if differences:
        raise ValueError(f"{label}: " + "; ".join(differences))


def resolve_dgrad_config(problem: Mapping, implementation: Mapping) -> dict:
    """Resolve defaults, the optional preset, and explicit tuning overrides.

    ``enable_dgrad_optimizations=True`` selects ``ds3_ep4_v1``. Its initial
    qualification is T4096/H7168/I2048/E32/K8, EP4, E4M3, quantized combine,
    unclamped SwiGLU, all three auxiliary outputs, and either weight layout.
    Unsupported problems/functions raise; there is no automatic partial preset.

    Omitted options use the preset; explicitly supplied options take precedence
    and are reported in ``dgrad_optimization_overrides`` when they differ.
    ``dgrad_schedule`` retains the tester's legacy baseline/optimized schedule
    selection, independently of the more narrowly qualified complete preset.
    Kernel constructors perform their resource-specific checks after resolution.
    """
    explicit = {k: v for k, v in implementation.items() if v is not DGRAD_UNSPECIFIED}
    # Keep legacy disabled spellings readable; active experiments are archived.
    for key, default in (
        ("phase_interleave_defer_linear1_until_input_ready", False),
        ("prefetch_tma_descriptors", False),
        ("dfc1_weight_l2_prefetch_depth", 0),
    ):
        value = explicit.pop(key, default)
        if type(value) is not type(default):
            raise TypeError(f"{key} must be {type(default).__name__}")
        if value != default:
            raise NotImplementedError(f"{key} is an archived experiment")
    enabled = explicit.pop("enable_dgrad_optimizations", False)
    if type(enabled) is not bool:
        raise TypeError("enable_dgrad_optimizations must be bool")
    legacy_schedule = explicit.pop("dgrad_schedule", None)
    if legacy_schedule not in (None, "baseline", "optimized"):
        raise ValueError(f"unsupported dgrad_schedule={legacy_schedule!r}")
    if enabled and legacy_schedule == "baseline":
        raise ValueError("dgrad_schedule='baseline' conflicts with enable_dgrad_optimizations=True")

    resolved = dict(_DEFAULTS)
    if legacy_schedule == "optimized":
        resolved.update(_ROLLING_SCHEDULE)
    preset = {}
    if enabled:
        _require_values(problem, dict(
            world_size=4, expert_count=32, topk=8, max_tokens_per_rank=4096,
            max_recv_size_per_rank=131072, hidden_size=7168,
            intermediate_gateup_size=2048, quant_kind="mxfp8_e4m3",
            gate_up_clamp=None,
        ), "ds3_ep4_v1 unsupported problem")
        if str(problem.get("combine_format")) != "32e4m3xe8m0":
            raise ValueError("ds3_ep4_v1 requires combine_format='32e4m3xe8m0'")
        # These describe tensor/output contracts, not performance choices.
        # Require the validated contract instead of silently enabling outputs.
        _require_values(explicit, dict(
            mma_tiler_mnk=(256, 256, 128), cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True, token_padding_block=128, sf_padding_block=128,
            sf_vec_size=32, drop_on_overflow=True, act_func="swiglu",
            fc2_in_kernel_topk_reduce=False, dfc2_recompute=True,
            dfc2_col_output=True, enable_grad_y2_col_quant=True,
        ), "ds3_ep4_v1 unsupported functional/layout configuration")
        if explicit.get("weight_storage_mode", "contiguous") not in ("contiguous", "discrete"):
            raise ValueError("ds3_ep4_v1 requires contiguous or discrete weights")
        preset = dict(_ROLLING_SCHEDULE, num_sched_stages=2,
                      group_hint=explicit["launch_cluster_count"],
                      epi_flag_batch=(4, 2), flag_batch=1, token_in_async_flags=True,
                      token_in_transfer_warp_count=4, token_back_mode="epi_warps",
                      input_dedup_mode="rank", dfc1_m_group=16,
                      use_scaled_cvt=True, num_ctas_grad_y2_col_quant=-1)
        resolved.update(preset)
    # None is the explicit spelling of the hardware-resident group count.
    if explicit.get("group_hint", DGRAD_UNSPECIFIED) is None:
        explicit["group_hint"] = explicit["launch_cluster_count"]
    resolved.update(explicit)
    resolved["epi_flag_batch"] = tuple(resolved["epi_flag_batch"]) if resolved["epi_flag_batch"] is not None else (1, 1)

    if type(resolved["token_in_async_flags"]) is not bool:
        raise TypeError("token_in_async_flags must be bool")
    if resolved["token_in_async_flags"] and resolved["flag_batch"] != 1:
        raise ValueError("asynchronous token-in flags require flag_batch=1")
    if resolved["token_in_transfer_warp_count"] != 4:
        raise ValueError("token_in_transfer_warp_count=5 was an archived experiment; use 4")

    rolling = resolved["schedule_mode"] == "phase_interleave"
    if rolling and (resolved["load_balance_mode"] != "atomic_counter"
                    or resolved["token_back_mode"] != "epi_warps"):
        raise ValueError("phase_interleave requires atomic_counter and epi_warps")
    if not rolling and (resolved["phase_interleave_prologue_tiles"] is not None
                        or resolved["phase_interleave_defer_consumers_until_full"]
                        or resolved["phase_interleave_fuse_ready_probe_and_producer_claim"]):
        raise ValueError("phase-interleave options require schedule_mode='phase_interleave'")
    if resolved["dfc1_m_group"] not in (1, 16) or type(resolved["dfc1_m_group"]) is not int:
        raise ValueError("dfc1_m_group must be 1 or 16")
    if resolved["dfc1_m_group"] != 1 and not rolling:
        raise ValueError("M16 traversal requires phase_interleave")
    if resolved["input_dedup_mode"] not in ("off", "rank"):
        raise ValueError("input_dedup_mode must be 'off' or 'rank'")
    if resolved["num_ctas_grad_y2_col_quant"] == -1 and not resolved.get("enable_grad_y2_col_quant", False):
        raise ValueError("automatic grad_y2 grid requires enable_grad_y2_col_quant=True")
    if resolved["input_dedup_mode"] == "rank":
        _require_values(resolved, dict(flag_batch=1, token_in_transfer_warp_count=4,
                                      token_back_mode="epi_warps"), "rank input dedup")
    if resolved["dfc2_subtile_publish"]:
        extent = problem["intermediate_gateup_size"]
        tile_n = resolved["mma_tiler_mnk"][1]
        bits = 2 if resolved["use_2cta_instrs"] else 1
        if extent <= 0 or extent % tile_n or extent // tile_n * bits > 30:
            raise ValueError("dfc2_subtile_publish requires aligned N and at most 30 ready-mask bits")

    overrides = {k: v for k, v in explicit.items() if k in preset and v != preset[k]}
    resolved["enable_dgrad_optimizations"] = enabled
    resolved["dgrad_optimization_profile"] = (
        "ds3_ep4_v1_custom" if overrides else "ds3_ep4_v1"
    ) if enabled else (legacy_schedule or "explicit")
    resolved["dgrad_optimization_overrides"] = overrides
    return resolved


__all__ = ["resolve_dgrad_config"]
