# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``torchrun`` entry point for the #752 CP frozen-partial replay.

CLI contract (from the execution plan)::

    run_cp_reference.py --mode frozen-partials|actual-attention
                        [--fixture <file>]      # required for frozen-partials
                        --output-dir <dir>

Everything else is read from the fixture header and written back into the
report, so a run cannot hide a difference behind a default.

Modes
-----
``frozen-partials``
    Replay a frozen fixture. **No real attention is computed**: the partials come
    from the fixture (synthesised by the FP64 oracle), and the run exercises the
    merge schedule, the dtype/rounding path, the token reassembly and the
    multi-rank plumbing. This is the mode the CP2/CP4 evidence uses.
``actual-attention``
    R2: compute per-step partials on the GPU with the pinned TransformerEngine
    ``fused_attn_fwd`` and merge them. It refuses to run unless the pinned TE
    revision is importable in this interpreter and hash-matches the recorded pin;
    on refusal it writes a ``status=R2_UNVERIFIED`` report and exits 2. It never
    falls back to a look-alike implementation.

Robustness requirements from the plan, and where they are implemented:

* device bound from ``LOCAL_RANK`` -> :func:`local_preflight`
* GPU count validated against world size -> :func:`local_preflight`
* all ranks agree on the configuration **before** the first configuration
  collective -> two-phase check in :func:`run` (local digest, then
  ``all_gather_object`` of the digest, then of the payload hash)
* per-rank errors reported clearly -> every failure writes ``rank_<r>.json``
  with the traceback and the rank's device, then re-raises
* process group always destroyed -> ``finally`` in :func:`run`
* no indefinite wait if a rank dies -> bounded ``init_process_group`` timeout
  plus torchrun's own job teardown; see ``--timeout-sec``
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from cp_numerics import te_adapter as tea  # noqa: E402
from cp_numerics import trace_schema as ts  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_R2_UNVERIFIED = 2
EXIT_PRECONDITION = 3

#: Recorded in every report. ``torch.equal`` is never used for the bitwise
#: comparisons below: it treats ``+0.0`` and ``-0.0`` as equal and hides dtype
#: promotion. The bitwise numbers compare integer bit patterns of the *same*
#: dtype instead.
BITWISE_POLICY = (
    "bitwise comparisons use the integer bit pattern of the same dtype; +0.0 and -0.0 are distinct; "
    "NaN is compared by bit pattern and also counted separately"
)

#: ULP is reported as "how many representable steps of the *actual* dtype lie
#: between the produced value and the correctly rounded FP64 reference".
ULP_POLICY = (
    "max_ulp = max |bitkey(actual) - bitkey(round_to_actual_dtype(reference_fp64))| over elements where both "
    "are finite; it therefore includes the <= 0.5 ULP error of rounding the FP64 reference into the actual dtype"
)


# ---------------------------------------------------------------------------
# bit-pattern helpers (never torch.equal)
# ---------------------------------------------------------------------------


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor.contiguous().view(torch.int32)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.contiguous().view(torch.int16)
    if tensor.dtype == torch.float64:
        return tensor.contiguous().view(torch.int64)
    raise TypeError(f"no integer view registered for {tensor.dtype}")


def bitwise_mismatch_count(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bit patterns differ. Requires identical dtype/shape."""
    if a.dtype != b.dtype:
        raise TypeError(f"bitwise compare needs identical dtypes, got {a.dtype} and {b.dtype}")
    if a.shape != b.shape:
        raise ValueError(f"bitwise compare needs identical shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    return int((_bits(a) != _bits(b)).sum().item())


def _total_order_key(tensor: torch.Tensor) -> torch.Tensor:
    """Unsigned monotone key: ``key(a) - key(b)`` is the ULP distance.

    Uses the standard transform ``key = b >= 0 ? b + 2^(n-1) : 2^n - 1 - b`` on the
    unsigned bit pattern, which keeps ``+0.0`` and ``-0.0`` distinct.
    """
    if tensor.dtype == torch.float32:
        width, mask, sign = 32, 0xFFFFFFFF, 0x80000000
    elif tensor.dtype in (torch.float16, torch.bfloat16):
        width, mask, sign = 16, 0xFFFF, 0x8000
    else:
        raise TypeError(f"ULP key is defined for 16/32-bit floats, not {tensor.dtype}")
    del width
    unsigned = _bits(tensor).to(torch.int64) & mask
    return torch.where(unsigned >= sign, mask - unsigned, unsigned + sign)


# ---------------------------------------------------------------------------
# error statistics
# ---------------------------------------------------------------------------


def error_stats(actual: torch.Tensor, reference_fp64: torch.Tensor) -> Dict[str, Any]:
    """Abs/rel/ULP error over finite elements; non-finite values counted separately."""
    if actual.shape != reference_fp64.shape:
        raise ValueError(f"shape mismatch: actual {tuple(actual.shape)} vs reference {tuple(reference_fp64.shape)}")
    act = actual.detach().to(torch.float64)
    ref = reference_fp64.detach().to(torch.float64)
    finite = torch.isfinite(act) & torch.isfinite(ref)
    stats: Dict[str, Any] = {
        "elements": int(act.numel()),
        "finite_elements": int(finite.sum().item()),
        "actual_nan": int(torch.isnan(act).sum().item()),
        "actual_posinf": int(torch.isposinf(act).sum().item()),
        "actual_neginf": int(torch.isneginf(act).sum().item()),
        "reference_nan": int(torch.isnan(ref).sum().item()),
        "reference_posinf": int(torch.isposinf(ref).sum().item()),
        "reference_neginf": int(torch.isneginf(ref).sum().item()),
        "max_abs": None,
        "max_rel": None,
        "max_ulp": None,
        "ulp_reference_overflowed_dtype": None,
        "first_mismatch_flat_index": None,
        "first_mismatch_value_actual": None,
        "first_mismatch_value_reference": None,
    }
    if stats["finite_elements"] == 0:
        return stats
    a = act[finite]
    r = ref[finite]
    diff = (a - r).abs()
    denom = r.abs().clamp_min(torch.finfo(torch.float64).tiny)
    stats["max_abs"] = float(diff.max().item())
    stats["max_rel"] = float((diff / denom).max().item())
    argmax = int(diff.argmax().item())
    flat_index = int(torch.nonzero(finite.view(-1), as_tuple=False)[argmax].item())
    stats["first_mismatch_flat_index"] = flat_index
    stats["first_mismatch_value_actual"] = float(act.view(-1)[flat_index].item())
    stats["first_mismatch_value_reference"] = float(ref.view(-1)[flat_index].item())

    stats["max_abs_over_max_reference"] = float(diff.max().item() / r.abs().max().clamp_min(torch.finfo(torch.float64).tiny).item())
    if actual.dtype in (torch.float16, torch.bfloat16, torch.float32):
        rounded = reference_fp64.detach().to(actual.dtype)
        comparable = (torch.isfinite(rounded) & finite).cpu()
        stats["ulp_reference_overflowed_dtype"] = int((~torch.isfinite(rounded) & torch.isfinite(ref)).sum().item())
        if bool(comparable.any()):
            up = (_total_order_key(actual.detach().cpu()) - _total_order_key(rounded.cpu())).abs()
            sel = up[comparable]
            stats["max_ulp"] = int(sel.max().item())
            # ULP is extremely sensitive at near-zero outputs, where the absolute
            # error stays ~1e-8 while one ULP shrinks to 1e-11. Report the whole
            # distribution rather than one flattering maximum.
            quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64)
            stats["ulp_percentiles"] = {f"p{int(q * 1000) / 10:g}": int(torch.quantile(sel.to(torch.float64), q).item()) for q in quantiles}
            stats["ulp_gt_4_count"] = int((sel > 4).sum().item())
            stats["ulp_finite_elements"] = int(sel.numel())
    return stats


def locate_flat_index(index: int, shape: Sequence[int], chunk_len: int, cp_size: int) -> Dict[str, int]:
    """Map a flat index of a ``[B, S, H, D]`` tensor onto rank/chunk/token/head."""
    b, s, h, d = shape
    del b
    token = (index // (h * d)) % s
    head = (index // d) % h
    chunk = token // chunk_len
    rank = chunk if chunk < cp_size else 2 * cp_size - 1 - chunk
    return {"batch": index // (s * h * d), "token": token, "head": head, "chunk": chunk, "rank": rank, "dim": index % d}


# ---------------------------------------------------------------------------
# preflight (no collectives here)
# ---------------------------------------------------------------------------


def local_preflight(args: argparse.Namespace, rank: int, local_rank: int, world_size: int) -> Dict[str, Any]:
    """Everything that must be true on *this* rank before any collective."""
    info: Dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "env": {
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "python": sys.version,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this rank")
    device_count = torch.cuda.device_count()
    info["visible_device_count"] = device_count
    if device_count < world_size:
        raise RuntimeError(
            f"world_size={world_size} but only {device_count} CUDA devices are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); refusing to run"
        )
    if local_rank >= device_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} is out of range for {device_count} visible devices")
    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    info["device"] = {
        "index": local_rank,
        "name": props.name,
        "capability": [props.major, props.minor],
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "uuid": str(getattr(props, "uuid", "")),
    }

    if args.mode in ("frozen-partials", "actual-attention"):
        if not args.fixture:
            raise RuntimeError(f"--mode {args.mode} requires --fixture")
        if not os.path.exists(args.fixture):
            raise RuntimeError(f"fixture not found: {args.fixture}")
        fixture = tea.load_fixture(args.fixture)
        header = fixture.header
        info["fixture"] = {
            "path": os.path.abspath(args.fixture),
            "name": header.name,
            "lane": header.lane,
            "sha256": header.fixture_sha256,
            "cp_size": header.cp_size,
            "config_digest": tea.fixture_config_digest(header),
        }
        if header.cp_size != world_size:
            raise RuntimeError(
                f"fixture {header.name!r} carries cp_size={header.cp_size} but world_size={world_size}; " "a P=2 fixture must not be replayed under a P=4 label"
            )
        tea.check_schedule_matches_trace(header)
        records = [ts.TraceRecord.from_dict(d) for d in header.trace]
        results = ts.check_invariants(
            records,
            cp_size=header.cp_size,
            seq_len=header.seq_len,
            chunk_len=header.chunk_len,
            causal=header.causal,
            valid_len=header.valid_len,
        )
        info["trace_invariants"] = [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results]
        failed = [r for r in results if not r.ok]
        if failed:
            raise RuntimeError("trace invariants failed: " + "; ".join(f"{r.name}: {r.detail}" for r in failed))
        ok, why = tea.te_requires_divisibility(header)
        info["te_lane_applicable"] = {"ok": ok, "reason": why}
        if not ok:
            raise RuntimeError(f"the TE merge lane is not applicable to this fixture: {why}")
    else:
        info["fixture"] = None
    return info


# ---------------------------------------------------------------------------
# merge + measurement
# ---------------------------------------------------------------------------


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *, warmup: int, repeats: int) -> Tuple[Any, Dict[str, Any]]:
    """Run ``fn`` with CUDA-event timing; return ``(last result, timing stats)``."""
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    _sync()
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    return result, {
        "repeats": len(samples),
        "warmup": max(0, warmup),
        "samples_ms": samples,
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
    }


def _select_half(tensor: torch.Tensor, part: str, chunk_len: int, seq_axis: int) -> torch.Tensor:
    """Slice the rank's stored query axis into one of its two output halves."""
    if part == tea.PART_WHOLE:
        return tensor
    index = [slice(None)] * tensor.dim()
    index[seq_axis] = slice(0, chunk_len) if part == tea.PART_FIRST else slice(chunk_len, 2 * chunk_len)
    return tensor[tuple(index)]


def naive_sequential_merge(
    partial_out: Sequence[torch.Tensor],
    partial_lse: Sequence[torch.Tensor],
    o_local_shape: Sequence[int],
    out_dtype: torch.dtype,
    *,
    rank: int,
    cp_size: int,
    causal: bool,
) -> torch.Tensor:
    """The obvious first implementation, measured as the comparison baseline.

    One ``max``/``sum`` pass for the global LSE -- no ``log1p``, no per-step
    correction -- followed by a left-to-right weighted accumulation in local
    storage order. It reads exactly the same bytes as the TE lane, so the
    throughput comparison is apples to apples on I/O even though the numerics
    differ. The step *selection* comes from the same schedule metadata
    (:func:`te_adapter.half_step_lists`) because a baseline that attends to the
    wrong KV range would be measuring garbage rather than a merge.
    """
    chunk_len = int(o_local_shape[2]) if len(o_local_shape) == 5 else int(o_local_shape[1]) // 2
    halves = tea.half_step_lists(cp_size, rank, causal)
    merged_halves: List[torch.Tensor] = []
    for selectors in halves:
        outs = [_select_half(partial_out[step], part, chunk_len, 1) for step, part in selectors]
        lses = [_select_half(partial_lse[step], part, chunk_len, -1).to(torch.float32) for step, part in selectors]
        stacked = torch.stack(lses, dim=0)
        finite = torch.isfinite(stacked)
        running_max = stacked.max(dim=0).values
        safe_max = torch.where(torch.isfinite(running_max), running_max, torch.zeros_like(running_max))
        weights = torch.exp(stacked - safe_max.unsqueeze(0))  # [N, B, H, Sq]
        weights = torch.where(finite, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=0)
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        acc: Optional[torch.Tensor] = None
        for index, out_i in enumerate(outs):
            weight = (weights[index] / safe_denom).movedim(1, 2).unsqueeze(-1)
            term = out_i.to(torch.float32) * weight
            acc = term if acc is None else acc + term
        assert acc is not None
        merged_halves.append(acc.to(out_dtype))
    if len(o_local_shape) == 5:
        return torch.stack(merged_halves, dim=1).reshape(*o_local_shape)
    return torch.cat(merged_halves, dim=1).reshape(*o_local_shape)


def _local_shape(header: tea.FixtureHeader) -> Tuple[int, ...]:
    if header.causal:
        return (header.batch, 2, header.chunk_len, header.num_heads, header.head_dim)
    return (header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)


def _flat_shape(header: tea.FixtureHeader) -> Tuple[int, ...]:
    return (header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)


def run_merge_lanes(
    fixture: tea.FrozenFixture,
    rank: int,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    compiled: bool,
) -> Dict[str, Any]:
    header = fixture.header
    cp_size = header.cp_size
    out_dtype = tea.dtype_from_name(header.o_dtype)
    o_local_shape = _local_shape(header)
    flat_shape = _flat_shape(header)

    partial_out = [fixture.rank_step_out(rank, step).to(device) for step in range(cp_size)]
    partial_lse = [fixture.rank_step_lse(rank, step).to(device) for step in range(cp_size)]

    def te_call() -> Tuple[torch.Tensor, torch.Tensor]:
        return tea.merge_te_fidelity(
            rank=rank,
            cp_size=cp_size,
            causal=header.causal,
            partial_out=partial_out,
            partial_lse=partial_lse,
            o_local_shape=o_local_shape,
            compiled=compiled,
        )

    te_result, te_timing = timed(te_call, warmup=warmup, repeats=repeats)
    emu_result, emu_timing = timed(
        lambda: tea.merge_emulator(
            rank=rank,
            cp_size=cp_size,
            causal=header.causal,
            partial_out=partial_out,
            partial_lse=partial_lse,
            o_local_shape=o_local_shape,
            out_dtype=out_dtype,
        ),
        warmup=warmup,
        repeats=repeats,
    )
    naive_result, naive_timing = timed(
        lambda: naive_sequential_merge(
            partial_out,
            partial_lse,
            o_local_shape,
            out_dtype,
            rank=rank,
            cp_size=cp_size,
            causal=header.causal,
        ),
        warmup=warmup,
        repeats=repeats,
    )

    payload_bytes = tea.merge_payload_bytes(rank, header)
    out_bytes = int(te_result[0].numel() * te_result[0].element_size())
    for timing in (te_timing, emu_timing, naive_timing):
        timing["payload_bytes_read"] = payload_bytes
        timing["output_bytes_written"] = out_bytes
        timing["gbps_payload"] = (payload_bytes / 1e9) / (timing["median_ms"] / 1e3) if timing["median_ms"] > 0 else None

    repeat_out, _ = te_call()
    return {
        "rank": rank,
        "local_shape": list(o_local_shape),
        "partial_shapes": [list(t.shape) for t in partial_out],
        "lse_shapes": [list(t.shape) for t in partial_lse],
        "partial_storage_dtype": str(partial_out[0].dtype).replace("torch.", ""),
        "accumulator_dtype": str(te_result[0].dtype).replace("torch.", ""),
        "lse_dtype": str(te_result[1].dtype).replace("torch.", ""),
        "lse_base": header.lse_base,
        "te_out": te_result[0].reshape(flat_shape).detach().cpu(),
        "te_lse": te_result[1].detach().cpu(),
        "emu_out": emu_result[0].reshape(flat_shape).detach().cpu(),
        "naive_out": naive_result.reshape(flat_shape).detach().cpu(),
        "te_timing": te_timing,
        "emu_timing": emu_timing,
        "naive_timing": naive_timing,
        "bitwise_self_repeat_mismatches": bitwise_mismatch_count(te_result[0].reshape(flat_shape), repeat_out.reshape(flat_shape)),
        "compiled": compiled,
    }


# ---------------------------------------------------------------------------
# actual-attention (R2) -- refuses rather than approximates
# ---------------------------------------------------------------------------


def _te_imported_module() -> Tuple[Optional[str], str]:
    """``(source path of the TE the interpreter would run, refusal reason)``.

    Binds the pin to the INSTALLED TransformerEngine rather than to a checkout
    passed on the command line: the kernels that produce R2's numbers come from
    the imported package, so hashing a ``--te-repo`` copy proves nothing about
    them, and a developer pointing ``--te-repo`` at the pin while running a
    different install would get a green report for a kernel it did not use
    (reviewer finding on PR #1142).
    """
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import context_parallel as te_cp
    except (ImportError, OSError) as exc:
        # ImportError when TE is absent; OSError when its CUDA libraries cannot
        # be dlopen'd in this interpreter. Both mean "cannot verify".
        return None, f"transformer_engine is not importable in this interpreter: {exc!r}"

    installed = os.path.abspath(te_cp.__file__ or "")
    if not installed.endswith(tea.TE_SOURCE_RELPATH.replace("/", os.sep)):
        return None, f"imported CP module {installed} does not match the pinned relpath {tea.TE_SOURCE_RELPATH}"

    with open(installed, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if not tea.TE_SOURCE_SHA256:
        return None, "no TE source sha256 is recorded; the pin cannot be verified"
    if digest != tea.TE_SOURCE_SHA256:
        return None, f"imported TE source sha256 {digest} != recorded pin {tea.TE_SOURCE_SHA256} ({installed})"
    return installed, ""


def _te_repo_cross_check(te_repo: Optional[str]) -> str:
    """Optional extra check: the checkout named on the command line, if given.

    Returns a refusal reason, or an empty string when it agrees (or was not
    given). Never raises for a missing path -- a fail-closed report must still
    be written.
    """
    if not te_repo:
        return ""
    candidate = os.path.join(te_repo, tea.TE_SOURCE_RELPATH)
    if not os.path.exists(candidate):
        return f"--te-repo names no pinned source at {candidate}"
    if tea.TE_SOURCE_SHA256:
        digest = tea.helper_source_sha256(te_repo)
        if digest != tea.TE_SOURCE_SHA256:
            return f"--te-repo source sha256 {digest} != recorded pin {tea.TE_SOURCE_SHA256}"
    return ""


def _te_installed_matches_pin(te_repo: Optional[str]) -> Tuple[bool, str]:
    """Does the TE this interpreter imports match the pin? (Cross-checks --te-repo.)"""
    installed, why = _te_imported_module()
    if why:
        return False, why
    repo_why = _te_repo_cross_check(te_repo)
    if repo_why:
        return False, repo_why
    return True, f"imported TE source verified: {installed}"


def run_actual_attention(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, output_dir: str) -> int:
    """R2 lane: per-step partials from the pinned TE fused attention, then TE's merge.

    Refuses unless ``--te-repo`` hashes to the pinned ``context_parallel.py``: a
    look-alike TransformerEngine would produce numbers that look like TE
    fidelity and are not.
    """
    ok, why = _te_installed_matches_pin(args.te_repo)
    installed_path, _installed_why = _te_imported_module()
    base: Dict[str, Any] = {
        "schema": "cp_numerics_run/1",
        "mode": "actual-attention",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "utc": _utc_now(),
        "te_binding": tea.describe_te_binding(),
        "te_imported_module": installed_path,
        "te_repo": os.path.abspath(args.te_repo) if args.te_repo else None,
        # Only hashed once the check has passed: hashing it earlier turned a
        # missing path into a traceback instead of the fail-closed report.
        "te_repo_sha256": tea.helper_source_sha256(args.te_repo) if (ok and args.te_repo) else None,
        "te_compute_dtype": args.te_compute_dtype,
    }
    if not ok:
        base["status"] = "R2_UNVERIFIED"
        base["reason"] = why
        base["note"] = (
            "This lane computes per-step partials with the pinned TransformerEngine fused attention and merges "
            "them with te_adapter.merge_te_fidelity. It refuses outright when the pinned revision is not "
            "importable rather than substituting a look-alike implementation."
        )
        _write_json(os.path.join(output_dir, f"rank_{rank}_actual_attention.json"), base)
        if rank == 0:
            print(json.dumps(base, indent=2))
        return EXIT_R2_UNVERIFIED

    if not args.fixture:
        raise RuntimeError("--mode actual-attention requires --fixture (it supplies the Q/K/V the attention runs on)")
    if not os.path.exists(args.fixture):
        raise RuntimeError(f"fixture not found: {args.fixture}")
    fixture = tea.load_fixture(args.fixture)
    header = fixture.header
    if header.cp_size != world_size:
        raise RuntimeError(f"fixture {header.name!r} carries cp_size={header.cp_size} but world_size={world_size}")

    # Imported here, after the pin check, so the frozen-partial mode and the
    # fail-closed report never need TransformerEngine on the import path.
    from cp_numerics import actual_attention as act  # noqa: PLC0415

    device = torch.device(f"cuda:{local_rank}")
    result = act.run_actual_attention_on_fixture(
        fixture=fixture,
        rank=rank,
        world_size=world_size,
        compiled=bool(args.compiled_helpers),
        device=device,
        compute_dtype=torch.float16 if args.te_compute_dtype == "float16" else torch.bfloat16,
    )
    report = dict(base)
    report["status"] = result.status
    report["reason"] = ""
    report.update(
        {
            "per_step": result.per_step,
            "merged": result.merged,
            "fixture": result.fixture,
            "notes": result.notes,
        }
    )
    _write_json(os.path.join(output_dir, f"rank_{rank}_actual_attention.json"), report)
    if rank == 0:
        summary = {
            "status": report["status"],
            "fixture": result.fixture["name"],
            "ranks": world_size,
            "steps_this_rank": len(result.per_step),
            "merged_o": result.merged["o"],
            "merged_lse": result.merged["lse"],
            "merge_label": result.merged["merge_label"],
            "partial_lse_source": result.merged["partial_lse_source"],
        }
        print(json.dumps(summary, indent=2))
    return EXIT_OK


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_bits(tensor.detach().cpu().contiguous()).numpy().tobytes()).hexdigest()


def _local_expected_slice(expected: torch.Tensor, rank: int, cp_size: int, chunk_len: int) -> torch.Tensor:
    """The rank's two query chunks, concatenated in the rank's storage order."""
    first, second = ts.rank_chunk_indices(rank, cp_size)
    return torch.cat(
        [
            expected[:, first * chunk_len : (first + 1) * chunk_len],
            expected[:, second * chunk_len : (second + 1) * chunk_len],
        ],
        dim=1,
    )


def _local_expected_lse(expected_lse: torch.Tensor, rank: int, cp_size: int, chunk_len: int) -> torch.Tensor:
    first, second = ts.rank_chunk_indices(rank, cp_size)
    return torch.cat(
        [
            expected_lse[:, :, first * chunk_len : (first + 1) * chunk_len],
            expected_lse[:, :, second * chunk_len : (second + 1) * chunk_len],
        ],
        dim=-1,
    )


def _owner_of_chunk(chunk: int, cp_size: int) -> Tuple[int, int]:
    """``(rank, half)`` that stores global ``chunk``."""
    if chunk < cp_size:
        return chunk, 0
    return 2 * cp_size - 1 - chunk, 1


def _global_lse_from_ranks(per_rank_lse: Sequence[torch.Tensor], cp_size: int, chunk_len: int, device: torch.device) -> torch.Tensor:
    """Reassemble per-rank LSE tensors into global order (verification only)."""
    pieces = []
    for chunk in range(2 * cp_size):
        rank, half = _owner_of_chunk(chunk, cp_size)
        pieces.append(per_rank_lse[rank].to(device)[..., half * chunk_len : (half + 1) * chunk_len])
    return torch.cat(pieces, dim=-1)


def _token_order_ok(fixture: tea.FrozenFixture, cp_size: int, chunk_len: int, device: torch.device) -> bool:
    """Reassemble per-rank token-id tensors and check they come back ascending."""
    token_ids = fixture.tensors["token_ids"]
    per_rank = {}
    for r in range(cp_size):
        first, second = ts.rank_chunk_indices(r, cp_size)
        ids = torch.cat(
            [
                token_ids[first * chunk_len : (first + 1) * chunk_len],
                token_ids[second * chunk_len : (second + 1) * chunk_len],
            ]
        )
        per_rank[r] = ids.view(1, 2 * chunk_len, 1, 1).to(device)
    restored = ts.restore_global_order(per_rank, cp_size, chunk_len).reshape(-1)
    return bool(restored.shape[0] == fixture.header.seq_len and torch.equal(restored.cpu(), torch.arange(fixture.header.seq_len)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP frozen-partial / actual-attention reference runner (#752)")
    parser.add_argument("--mode", required=True, choices=["frozen-partials", "actual-attention"])
    parser.add_argument("--fixture", default=None, help="frozen-partial fixture; required for --mode frozen-partials")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--te-repo", default=None, help="optional TE checkout cross-check for --mode actual-attention")
    parser.add_argument(
        "--te-compute-dtype",
        default="float16",
        choices=("float16", "bfloat16"),
        help="dtype the fused kernel runs in; fp32 is not fused-capable at the pinned revision",
    )
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="process-group timeout; bounds how long a surviving rank waits for a dead one",
    )
    parser.add_argument("--compiled-helpers", action="store_true", help="run the TE helpers through torch.compile")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    start_wall = time.time()
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if min(rank, local_rank, world_size) < 0:
        print(
            "ERROR: RANK/LOCAL_RANK/WORLD_SIZE are not set; launch this entry point with torchrun "
            "(e.g. torchrun --standalone --nnodes=1 --nproc-per-node=2 ...).",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION

    if args.mode == "actual-attention":
        return run_actual_attention(args, rank, local_rank, world_size, output_dir)

    report: Dict[str, Any] = {
        "schema": "cp_numerics_run/1",
        "mode": args.mode,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "started_utc": _utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "bitwise_policy": BITWISE_POLICY,
        "ulp_policy": ULP_POLICY,
        "te_binding": tea.describe_te_binding(),
        "status": "STARTED",
    }

    # ---- phase 1: purely local checks, no collective ----------------------
    try:
        report["preflight"] = local_preflight(args, rank, local_rank, world_size)
    except Exception as exc:
        report["status"] = "PREFLIGHT_FAILED"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        report["finished_utc"] = _utc_now()
        _write_json(os.path.join(output_dir, f"rank_{rank}.json"), report)
        print(f"[rank {rank}] preflight failed: {exc}", file=sys.stderr, flush=True)
        return EXIT_PRECONDITION

    fixture = tea.load_fixture(args.fixture)
    header = fixture.header
    device = torch.device("cuda", local_rank)
    cpl = header.cp_size
    chunk_len = header.chunk_len
    flat_shape = _flat_shape(header)

    dist_init_done = False
    try:
        # ---- phase 2: collective agreement, before any merge work ---------
        dist.init_process_group(backend="nccl", timeout=_dt.timedelta(seconds=args.timeout_sec))
        dist_init_done = True
        report["process_group"] = {
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "rank": dist.get_rank(),
            "timeout_sec": args.timeout_sec,
            "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        }
        digests: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(digests, report["preflight"]["fixture"]["config_digest"])
        if len(set(digests)) != 1:
            raise RuntimeError(f"ranks disagree on the fixture configuration: {digests}")
        payload_hashes: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(payload_hashes, report["preflight"]["fixture"]["sha256"])
        if len(set(payload_hashes)) != 1:
            raise RuntimeError(f"ranks loaded different fixture payloads: {payload_hashes}")
        report["fixture_sha256_agreed"] = payload_hashes[0]

        # ---- phase 3: merge + measurement --------------------------------
        merge = run_merge_lanes(
            fixture,
            rank,
            warmup=args.timing_warmup,
            repeats=args.timing_repeats,
            device=device,
            compiled=args.compiled_helpers,
        )
        expected = fixture.tensors["expected_out_fp64"]
        local_stats = error_stats(merge["te_out"], _local_expected_slice(expected, rank, cpl, chunk_len))
        report["local_error_vs_fp64"] = {
            "te_lane": local_stats,
            "emulator_lane": error_stats(merge["emu_out"], _local_expected_slice(expected, rank, cpl, chunk_len)),
            "naive_lane": error_stats(merge["naive_out"], _local_expected_slice(expected, rank, cpl, chunk_len)),
        }
        report["local_lse_error_vs_fp64"] = error_stats(
            merge["te_lse"].view(header.batch, header.num_heads, 2 * chunk_len),
            _local_expected_lse(fixture.tensors["expected_lse_fp64"], rank, cpl, chunk_len),
        )
        report["local_timing"] = {
            "te_lane": merge["te_timing"],
            "emulator_lane": merge["emu_timing"],
            "naive_lane": merge["naive_timing"],
        }
        report["local_out_digest"] = _digest(merge["te_out"])
        report["bitwise_self_repeat_mismatches"] = merge["bitwise_self_repeat_mismatches"]
        report["observed_dtypes"] = {
            "partial_storage": merge["partial_storage_dtype"],
            "accumulator": merge["accumulator_dtype"],
            "lse": merge["lse_dtype"],
            "lse_base": merge["lse_base"],
        }

        # ---- phase 4: gather for global reassembly (verification only) ----
        gathered_te: List[Optional[torch.Tensor]] = [None] * world_size
        gathered_lse: List[Optional[torch.Tensor]] = [None] * world_size
        gathered_emu: List[Optional[torch.Tensor]] = [None] * world_size
        gathered_naive: List[Optional[torch.Tensor]] = [None] * world_size
        gathered_timing: List[Optional[Dict[str, Any]]] = [None] * world_size
        t_gather0 = time.time()
        dist.all_gather_object(gathered_te, merge["te_out"])
        dist.all_gather_object(gathered_lse, merge["te_lse"])
        dist.all_gather_object(gathered_emu, merge["emu_out"])
        dist.all_gather_object(gathered_naive, merge["naive_out"])
        dist.all_gather_object(
            gathered_timing,
            {
                "rank": rank,
                "te_median_ms": merge["te_timing"]["median_ms"],
                "te_min_ms": merge["te_timing"]["min_ms"],
                "emu_median_ms": merge["emu_timing"]["median_ms"],
                "naive_median_ms": merge["naive_timing"]["median_ms"],
                "naive_min_ms": merge["naive_timing"]["min_ms"],
                "payload_bytes_read": merge["te_timing"]["payload_bytes_read"],
                "gbps_payload": merge["te_timing"]["gbps_payload"],
                "bitwise_self_repeat_mismatches": merge["bitwise_self_repeat_mismatches"],
            },
        )
        _sync()
        report["gather_sec"] = time.time() - t_gather0
        report["per_rank_timing"] = gathered_timing

        if rank == 0:
            expected_device = expected.to(device)
            per_rank_te = {r: gathered_te[r].to(device) for r in range(world_size)}
            global_te = ts.restore_global_order(per_rank_te, cpl, chunk_len)
            global_emu = ts.restore_global_order({r: gathered_emu[r].to(device) for r in range(world_size)}, cpl, chunk_len)
            global_naive = ts.restore_global_order({r: gathered_naive[r].to(device) for r in range(world_size)}, cpl, chunk_len)

            replay_per_rank = {}
            for r in range(world_size):
                po = [fixture.rank_step_out(r, s).to(device) for s in range(cpl)]
                pl = [fixture.rank_step_lse(r, s).to(device) for s in range(cpl)]
                out_r, _ = tea.merge_te_fidelity(
                    rank=r,
                    cp_size=cpl,
                    causal=header.causal,
                    partial_out=po,
                    partial_lse=pl,
                    o_local_shape=_local_shape(header),
                    compiled=args.compiled_helpers,
                )
                replay_per_rank[r] = out_r.reshape(flat_shape)
            global_replay = ts.restore_global_order(replay_per_rank, cpl, chunk_len)

            global_stats = error_stats(global_te, expected_device)
            global_stats["first_mismatch_location"] = (
                locate_flat_index(global_stats["first_mismatch_flat_index"], global_te.shape, chunk_len, cpl)
                if global_stats.get("first_mismatch_flat_index") is not None
                else None
            )
            lse_global = _global_lse_from_ranks(gathered_lse, cpl, chunk_len, device)
            report["global"] = {
                "shape": list(global_te.shape),
                "dtype": str(global_te.dtype).replace("torch.", ""),
                "error_vs_fp64": global_stats,
                "emulator_error_vs_fp64": error_stats(global_emu, expected_device),
                "naive_error_vs_fp64": error_stats(global_naive, expected_device),
                "lse_error_vs_fp64": error_stats(lse_global, fixture.tensors["expected_lse_fp64"].to(device)),
                "bitwise_gather_vs_single_process_replay_mismatches": bitwise_mismatch_count(global_te, global_replay),
                "token_order_restored": _token_order_ok(fixture, cpl, chunk_len, device),
                "per_rank_digests": {r: _digest(gathered_te[r]) for r in range(world_size)},
            }
        report["status"] = "OK"
        report["finished_utc"] = _utc_now()
        report["wall_clock_sec"] = time.time() - start_wall
        _write_json(os.path.join(output_dir, f"rank_{rank}.json"), report)
        if rank == 0:
            _write_json(os.path.join(output_dir, "report.json"), report)
            print(json.dumps(_headline(report), indent=2))
        return EXIT_OK

    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        report["finished_utc"] = _utc_now()
        report["wall_clock_sec"] = time.time() - start_wall
        try:
            _write_json(os.path.join(output_dir, f"rank_{rank}.json"), report)
        except Exception:
            pass
        print(f"[rank {rank}] FAILED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return EXIT_FAIL
    finally:
        if dist_init_done and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _headline(report: Dict[str, Any]) -> Dict[str, Any]:
    glob = report.get("global") or {}
    return {
        "mode": report["mode"],
        "status": report["status"],
        "world_size": report["world_size"],
        "fixture": (report.get("preflight") or {}).get("fixture"),
        "te_binding": report["te_binding"],
        "observed_dtypes": report.get("observed_dtypes"),
        "global_error_vs_fp64": glob.get("error_vs_fp64"),
        "global_first_mismatch_location": (glob.get("error_vs_fp64") or {}).get("first_mismatch_location"),
        "global_bitwise_gather_vs_single_process_replay": glob.get("bitwise_gather_vs_single_process_replay_mismatches"),
        "token_order_restored": glob.get("token_order_restored"),
        "local_bitwise_self_repeat_mismatches": report.get("bitwise_self_repeat_mismatches"),
        "per_rank_timing": report.get("per_rank_timing"),
        "gather_sec": report.get("gather_sec"),
        "wall_clock_sec": report.get("wall_clock_sec"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
