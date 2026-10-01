# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned compressed-KV consumer geometry, precision and shared-latent order."""

# Source, numerical and admission checks below rely on assertions.
if not __debug__:
    raise RuntimeError("This benchmark/auditor requires Python assertions; run without -O/-OO or PYTHONOPTIMIZE.")

import ast
import hashlib
import json
from pathlib import Path


def model_cases(config):
    assert (config["head_dim"], config["rope_head_dim"]) == (512, 64)
    assert {config["compress_ratios"][i] for i in config["kv_source_layers"]} == {1, 2}
    cases = []
    for source_sequence in (4096, 16384):
        for batch in (1, 4):
            for ratio in (1, 2):
                sequence = source_sequence // ratio
                cases.append(
                    dict(
                        tag=f"prefill_b{batch}_s{source_sequence}_ratio{ratio}",
                        batch=batch,
                        sequence=sequence,
                        tokens=batch * sequence,
                        heads=1,
                        head_dim=512,
                        rope="compressed",
                        start_pos=0,
                        phase="prefill",
                        position_stride=ratio,
                        source_sequence=source_sequence,
                        source_start_pos=0,
                    )
                )
    for batch in (1, 4):
        for ratio in (1, 2):
            cases.append(
                dict(
                    tag=f"decode_b{batch}_ratio{ratio}",
                    batch=batch,
                    sequence=1,
                    tokens=batch,
                    heads=1,
                    head_dim=512,
                    rope="compressed",
                    start_pos=4096,
                    phase="decode",
                    position_stride=ratio,
                    source_sequence=1,
                    source_start_pos=4096 if ratio == 1 else 4097,
                )
            )
    return cases


def verify_source(source_path):
    source = Path(source_path)
    hashes = {
        "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
        "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
        "config.json": "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809",
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    tree = ast.parse((source / "model.py").read_text())
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    function = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_kv")
    calls = [(n.lineno, ast.unparse(n.func), ast.unparse(n)) for n in ast.walk(function) if isinstance(n, ast.Call)]
    index = next(line for line, fn, text in calls if fn == "self._compress_topk_idxs")
    rope = next(line for line, fn, text in calls if fn == "apply_rotary_emb")
    quant, expression = next((line, text) for line, fn, text in calls if fn == "fp4_act_quant")
    assert index < rope < quant and expression == "fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)"
    return dict(
        revision="dba1be0a40aa45a94ad051997016db3960a90277",
        files=hashes,
        source_lines=dict(indexer=index, rope=rope, quant=quant),
        source_call=expression,
        cases=model_cases(json.loads((source / "config.json").read_text())),
        quantization=dict(value="E2M1", group=16, scale="E4M3", minimum_scale=2**-9, maximum_scale=448.0, global_scale=None, output="BF16 QDQ"),
        sharing="Indexer consumes the unrotated latent before this measured stage; cache publication follows it.",
        excluded=["compressor/projection/norm", "indexer execution", "cache writes", "packed-cache consumers", "STE/backward", "model throughput"],
    )
