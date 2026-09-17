# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the pinned DeepSeek RoPE functions without importing the full model.

The mathematical functions remain DeepSeek's implementation and attribution.
This loader evaluates the two unmodified AST nodes from a user-supplied source
checkout; no third-party kernel is vendored into this benchmark."""

import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import torch


def source_functions(source):
    path = Path(source) / "model.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    names = {"precompute_freqs_cis", "apply_rotary_emb"}
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    # A fresh namespace gives each owner its own source lru_cache. The source
    # signature has no device argument, so its cache must not span devices.
    namespace = dict(torch=torch, math=math, lru_cache=lru_cache)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["precompute_freqs_cis"], namespace["apply_rotary_emb"]


@torch.inference_mode(False)
@torch.no_grad()
def source_table(source, capacity, *, compressed, device):
    if type(capacity) is not int or capacity < 1 or type(compressed) is not bool:
        raise ValueError("positive capacity and explicit compressed mode required")
    config_path = Path(source) / "config.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"
    config = json.loads(config_path.read_text())
    precompute, apply = source_functions(source)
    with torch.device(device):
        table = precompute(
            config["rope_head_dim"],
            capacity,
            config["original_seq_len"] if compressed else 0,
            config["compress_rope_theta"] if compressed else config["rope_theta"],
            config["rope_factor"],
            config["beta_fast"],
            config["beta_slow"],
        )
    assert table.dtype == torch.complex64 and table.shape == (capacity, 32)
    return table, apply
