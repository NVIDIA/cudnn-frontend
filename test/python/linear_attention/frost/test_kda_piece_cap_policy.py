# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the KDA-forward split-table piece cap."""

import ast
import inspect
from pathlib import Path

import pytest

from cudnn.linear_attention.frost.kda_engine import choose_kda_fwd_piece_cap

pytestmark = pytest.mark.L0

_ROOT = Path(__file__).resolve().parents[4]
_ENGINE_DIR = _ROOT / "python" / "cudnn" / "linear_attention" / "frost"


@pytest.mark.parametrize(
    "total_tokens,n_heads_out,want",
    [
        pytest.param(8192, 96, 2, id="kimi_8k"),
        pytest.param(16384, 96, 2, id="kimi_16k"),
        pytest.param(16384, 64, 2, id="glm_16k"),
        pytest.param(32768, 64, 0, id="glm_32k"),
    ],
)
def test_kda_fwd_piece_cap_evidence_boundary(total_tokens, n_heads_out, want):
    from cudnn.linear_attention.frost.common.split_k import compute_ideal_chunks

    num_sms = 148
    ideal_chunks = compute_ideal_chunks(total_tokens, n_heads_out, num_sms, b_t=16)
    assert (
        choose_kda_fwd_piece_cap(
            device_cc=(10, 0),
            checkpointed=False,
            batch_size=1,
            n_heads_out=n_heads_out,
            n_tiles=n_heads_out,
            num_sms=num_sms,
            ideal_chunks=ideal_chunks,
        )
        == want
    )


@pytest.mark.parametrize(
    "device_cc,checkpointed,batch_size,n_heads_out,n_tiles,num_sms",
    [
        pytest.param((10, 3), False, 1, 64, 64, 148, id="sm103"),
        pytest.param((10, 0), True, 1, 64, 64, 148, id="checkpointed"),
        pytest.param((10, 0), False, 2, 64, 128, 148, id="ragged_batch"),
        pytest.param((10, 0), False, 1, 80, 80, 148, id="unsupported_heads"),
        pytest.param((10, 0), False, 1, 96, 96, 80, id="filled_grid"),
    ],
)
def test_kda_fwd_piece_cap_rejects_unmeasured_regimes(device_cc, checkpointed, batch_size, n_heads_out, n_tiles, num_sms):
    assert (
        choose_kda_fwd_piece_cap(
            device_cc=device_cc,
            checkpointed=checkpointed,
            batch_size=batch_size,
            n_heads_out=n_heads_out,
            n_tiles=n_tiles,
            num_sms=num_sms,
            ideal_chunks=333,
        )
        == 0
    )


def test_kda_fwd_piece_cap_ideal_boundary_is_inclusive():
    common = dict(device_cc=(10, 0), checkpointed=False, batch_size=1, n_heads_out=64, n_tiles=64, num_sms=148)
    assert choose_kda_fwd_piece_cap(**common, ideal_chunks=5 * common["num_sms"]) == 2
    assert choose_kda_fwd_piece_cap(**common, ideal_chunks=5 * common["num_sms"] + 1) == 0


def test_table_recipe_replays_runtime_piece_cap():
    from cudnn.linear_attention.frost.common.split_k import TableRecipe, run_table

    calls = []
    recipe = TableRecipe(
        compiled=lambda *args: calls.append(args),
        split=True,
        safe_gate=False,
        n_heads_out=64,
        n_tiles=64,
        num_sms=148,
        ideal_chunks=443,
        piece_cap=2,
        batch_size=1,
        log2_threshold=-1.0,
        gate_scale_log2=0.0,
        n_scan_ctas=7,
        n_walk_ctas=64,
    )
    gate = object()
    cu_seqlens = object()
    chunk_scratch = object()
    item_scratch = object()
    work_items = object()
    work_count = object()
    scheduler = object()

    run_table(
        recipe,
        gate,
        None,
        None,
        cu_seqlens,
        chunk_scratch,
        item_scratch,
        work_items,
        work_count,
        scheduler,
        0,
    )

    assert len(calls) == 1
    assert calls[0][:4] == (64, 443, 2, 1)
    assert calls[0][6:13] == (gate, None, None, cu_seqlens, chunk_scratch, item_scratch, work_items)


def _build_split_table_calls(path):
    tree = ast.parse(path.read_text())
    calls = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for function_node in (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for node in ast.walk(function_node):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "build_split_table":
                    calls.append((class_node.name, {keyword.arg for keyword in node.keywords}))
    return calls


def test_only_kda_forward_explicitly_owns_piece_cap():
    calls = {}
    for filename in ("kda_engine.py", "gdn_engine.py", "gdn2_engine.py"):
        for class_name, keywords in _build_split_table_calls(_ENGINE_DIR / filename):
            calls[(filename, class_name)] = keywords

    assert set(calls) == {
        ("kda_engine.py", "CompiledKda"),
        ("kda_engine.py", "CompiledKdaBwd"),
        ("gdn_engine.py", "CompiledGdn"),
        ("gdn_engine.py", "CompiledGdnBwd"),
        ("gdn2_engine.py", "CompiledGdn2"),
        ("gdn2_engine.py", "CompiledGdn2Bwd"),
    }
    explicit = {owner for owner, keywords in calls.items() if "piece_cap" in keywords}
    assert explicit == {("kda_engine.py", "CompiledKda")}


def test_piece_cap_defaults_off_and_does_not_shrink_workspace_bound():
    from cudnn.linear_attention.frost.common.split_k import build_split_table, max_work_items

    assert inspect.signature(build_split_table).parameters["piece_cap"].default == 0
    assert "piece_cap" not in inspect.signature(max_work_items).parameters


def test_piece_cap_is_runtime_not_a_compile_cache_key():
    tree = ast.parse((_ENGINE_DIR / "common" / "split_k.py").read_text())
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_split_table")
    key_assignment = next(
        node for node in ast.walk(build) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "key" for target in node.targets)
    )
    assert not any(isinstance(node, ast.Name) and node.id == "piece_cap" for node in ast.walk(key_assignment.value))
