# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-source window/DSpark RoPE + FP8 QDQ workloads, not packed-cache tests."""

import ast
import hashlib
import json
from pathlib import Path


def model_cases(config):
    assert (config["head_dim"], config["rope_head_dim"], config["dspark_block_size"]) == (512, 64, 5)
    cases = []
    for sequence in (4096, 16384):
        for batch in (1, 4):
            for rope in ("base", "compressed"):
                cases.append(
                    dict(
                        tag=f"b{batch}_s{sequence}_window_{rope}",
                        batch=batch,
                        sequence=sequence,
                        tokens=batch * sequence,
                        heads=1,
                        head_dim=512,
                        rope=rope,
                        start_pos=0,
                        phase="prefill",
                    )
                )
    for batch in (1, 4):
        for name, sequence, offset in (("main", 1, 4096), ("verify", config["dspark_block_size"], 4097)):
            cases.append(
                dict(
                    tag=f"b{batch}_dspark_{name}",
                    batch=batch,
                    sequence=sequence,
                    tokens=batch * sequence,
                    heads=1,
                    head_dim=512,
                    rope="base",
                    start_pos=offset,
                    phase="decode",
                )
            )
    return cases


def verify_source(source_path, quantization="fp8"):
    SOURCE = Path(source_path)
    model = SOURCE / "model.py"
    kernel = SOURCE / "kernel.py"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    tree = ast.parse(model.read_text())
    globals_ = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(globals_["fp8_block_size"]) == 32
    assert ast.literal_eval(globals_["scale_fmt"]) == "ue8m0"
    assert ast.unparse(globals_["scale_dtype"]) == "torch.float8_e8m0fnu"
    source_calls = {}
    for cls_name, func_name in (("Attention", "_window_kv"), ("DSparkAttention", "forward")):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == func_name)
        pairs = []
        for index, statement in enumerate(function.body[:-1]):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == "apply_rotary_emb":
                following = function.body[index + 1]
                if isinstance(following, ast.Expr) and isinstance(following.value, ast.Call) and ast.unparse(following.value.func) == "act_quant":
                    assert [ast.unparse(arg) for arg in following.value.args[1:]] == ["fp8_block_size", "scale_fmt", "scale_dtype", "True"]
                    pairs.append(dict(rope_line=statement.lineno, quant_line=following.lineno, tensor=ast.unparse(following.value.args[0])))
        assert len(pairs) == (1 if cls_name == "Attention" else 2)
        source_calls[cls_name + "." + func_name] = pairs
    args = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelArgs")
    batch_default = next(node.value for node in args.body if isinstance(node, ast.AnnAssign) and node.target.id == "max_batch_size")
    assert ast.literal_eval(batch_default) == 4
    assert hashlib.sha256((SOURCE / "config.json").read_bytes()).hexdigest() == "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"
    config = json.loads((SOURCE / "config.json").read_text())
    if quantization == "fp4":
        assert ast.literal_eval(globals_["fp4_block_size"]) == 32
        assert (config["index_head_dim"], config["rope_head_dim"], config["index_n_heads"]) == (128, 64, 32)
        indexer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Indexer")
        forward = next(node for node in indexer.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call) and ast.unparse(node.func) == "fp4_act_quant"]
        assert len(calls) == 2
        assert {ast.unparse(call.args[0]) for call in calls} == {"k", "q"}
        assert all([ast.unparse(arg) for arg in call.args[1:]] == ["fp4_block_size", "True"] for call in calls)
        return dict(
            revision="dba1be0a40aa45a94ad051997016db3960a90277",
            files={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in ("model.py", "kernel.py", "config.json")},
            source_calls={"Indexer.forward": [{"line": call.lineno, "tensor": ast.unparse(call.args[0])} for call in calls]},
            quantization=dict(value="E2M1", group=32, scale="UE8M0", amax_floor=6 * 2.0**-126, max_value=6.0, output="BF16 QDQ"),
            sharing="Owner computes index keys before Attention overwrites the shared RoPE-free compressed latent",
            excluded=["projection/norm", "index cache write", "index scores/top-k", "training STE/backward", "model throughput"],
        )
    assert quantization == "fp8"
    return dict(
        revision="dba1be0a40aa45a94ad051997016db3960a90277",
        source_calls=source_calls,
        files={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in ("model.py", "kernel.py", "config.json")},
        cases=model_cases(config),
        quantization=dict(value="E4M3", group=32, scale="UE8M0", amax_floor=1e-4, max_value=448.0, output="BF16 QDQ"),
        batches_scope="1 and source runtime default max_batch_size 4; not a checkpoint batch-size parameter",
        position_scope="Prefill starts at zero; DSpark main token at 4096 and next five draft tokens at 4097..4101",
        excluded=["projection/norm", "ring-cache write", "packed cache", "attention", "training STE/backward", "model throughput"],
    )
