#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build structured documentation ledgers from authored pages and canonical samples."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

try:
    from tools.doc.contracts import candidate_availability_contract
    from tools.doc.run_doc_samples import fern_region_references, fern_sample_references
except ModuleNotFoundError:  # Direct execution adds tools/doc, rather than the repository root, to sys.path.
    from contracts import candidate_availability_contract
    from run_doc_samples import fern_region_references, fern_sample_references

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
API_ROOT = DOCS_ROOT / "api"
SAMPLE_ROOT = DOCS_ROOT / "samples"

SAMPLE_OVERRIDES = {
    ("python", "matmul"): {
        "purpose": "Build and execute the Python BF16 matmul quick start.",
    },
    ("cpp", "matmul"): {
        "purpose": "Build and execute the C++ BF16 matmul quick start.",
    },
    ("python", "add_broadcast"): {
        "apis": ["cudnn.pygraph.add"],
        "owners": ["api/python/add.md", "tutorials/pointwise-reduction.md"],
    },
    ("cpp", "add_broadcast"): {
        "apis": ["cudnn_frontend::graph::Graph::add"],
        "owners": ["api/cpp/add.md", "tutorials/pointwise-reduction.md"],
    },
    ("python", "fusion"): {"apis": ["cudnn.pygraph.matmul", "cudnn.pygraph.bias", "cudnn.pygraph.relu"]},
    ("cpp", "fusion"): {"apis": ["cudnn_frontend::graph::Graph::matmul", "cudnn_frontend::graph::Graph::pointwise"]},
    ("python", "custom_engine"): {
        "apis": ["cudnn.pygraph.plans", "cudnn.pygraph.select_plan", "cudnn.pygraph.selected_engine"],
        "owners": ["how-to/custom-engine.md"],
    },
    ("python", "sdpa_bf16_causal"): {"apis": ["cudnn.pygraph.sdpa", "cudnn.pygraph.sdpa_backward"]},
    ("cpp", "sdpa_bf16_causal"): {"apis": ["cudnn_frontend::graph::Graph::sdpa", "cudnn_frontend::graph::Graph::sdpa_backward"]},
    ("cpp", "sdpa_block_mask"): {
        "apis": ["cudnn_frontend::graph::Graph::sdpa"],
        "owners": ["api/cpp/sdpa.md", "api/python/sdpa.md"],
        "requirements": ["sm100", "cudnn>=92600"],
        "api_availability": "On SM10x, unified SDPA block masks require cuDNN 9.26+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "expected": "packed UINT8 block-mask SDPA passed",
        "verification": "compiled with CUDA 13.1 headers; SM100 execution requires cuDNN 9.26+",
    },
    ("python", "pytorch_autograd"): {
        "apis": ["cudnn.Graph", "torch.autograd.Function"],
        "owners": ["tutorials/pytorch-integration.md"],
        "expected": "PyTorch autograd integration ok: forward=1 backward=1",
        "verification": "executed on CUDA 13.0/cuDNN 9.15.1/SM100",
    },
    ("cpp", "graph_cache"): {
        "apis": ["cudnn_frontend::graph::Graph"],
        "owners": ["tutorials/pytorch-integration.md"],
        "expected": "C++ graph cache ok: builds=1 executions=2",
        "verification": "executed on CUDA 13.1/cuDNN 9.15.1/SM100",
    },
    ("python", "block_scaled_matmul"): {
        "apis": [
            "cudnn.pygraph.block_scale_dequantize",
            "cudnn.pygraph.matmul",
            "cudnn.pygraph.block_scale_quantize",
        ],
        "owners": [
            "api/python/block_scale_dequantize.md",
            "api/python/block_scale_quantize.md",
            "tutorials/block-scaling.md",
        ],
        "requirements": ["sm100", "cudnn>=91400"],
        "expected": "MXFP8 block-scaled matmul ok: zero result and output scales produced",
        "verification": "executed on CUDA 13.0/cuDNN 9.15.1/SM100",
    },
    ("cpp", "block_scaled_matmul"): {
        "apis": [
            "cudnn_frontend::graph::Graph::block_scale_dequantize",
            "cudnn_frontend::graph::Graph::matmul",
            "cudnn_frontend::graph::Graph::block_scale_quantize",
        ],
        "owners": [
            "api/cpp/block_scale_dequantize.md",
            "api/cpp/block_scale_quantize.md",
            "tutorials/block-scaling.md",
        ],
        "requirements": ["sm100", "cudnn>=91400"],
        "expected": "MXFP8 block-scaled matmul ok: zero result and output scales produced",
        "verification": "executed on CUDA 13.1/cuDNN 9.15.1/SM100",
    },
    ("python", "rope_basic"): {
        "requirements": ["cudnn>=92400"],
        "api_availability": "Standalone sample requires cuDNN 9.24+; syntax-checked with cuDNN 9.15.1",
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "expected": "rope result matches reference",
        "verification": "syntax-checked; execution requires cuDNN 9.24+",
    },
    ("python", "rope_backward_basic"): {
        "requirements": ["cudnn>=92400"],
        "api_availability": "Standalone sample requires cuDNN 9.24+; syntax-checked with cuDNN 9.15.1",
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "expected": "rope backward matches reference",
        "verification": "syntax-checked; execution requires cuDNN 9.24+",
    },
    ("cpp", "rope_basic"): {
        "requirements": ["cudnn>=92400"],
        "api_availability": "Standalone sample requires cuDNN 9.24+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "expected": "rope result matches reference",
        "verification": "compiled against current headers; execution requires cuDNN 9.24+",
    },
    ("cpp", "rope_backward_basic"): {
        "requirements": ["cudnn>=92400"],
        "api_availability": "Standalone sample requires cuDNN 9.24+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "expected": "rope backward matches reference",
        "verification": "compiled against current headers; execution requires cuDNN 9.24+",
    },
    ("python", "moe_grouped_matmul_backward"): {
        "apis": ["cudnn.pygraph.moe_grouped_matmul_bwd"],
        "owners": ["api/python/moe_grouped_matmul_bwd.md", "tutorials/moe-grouped-matmul.md"],
        "requirements": ["sm90", "cudnn>=92200", "cublaslt>=130500"],
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "expected": "moe grouped matmul backward ok: dweight verified per expert",
        "verification": "syntax-checked and source-verified; execution requires cuDNN 9.22+ and cublasLt 13.5+",
    },
    ("cpp", "moe_grouped_matmul_backward"): {
        "apis": ["cudnn_frontend::graph::Graph::moe_grouped_matmul_bwd"],
        "owners": ["api/cpp/moe_grouped_matmul_bwd.md", "tutorials/moe-grouped-matmul.md"],
        "requirements": ["sm90", "cudnn>=92200", "cublaslt>=130500"],
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "expected": "moe grouped matmul backward ok: dweight verified per expert",
        "verification": "compiled against current headers; execution requires cuDNN 9.22+ and cublasLt 13.5+",
    },
    ("python", "transpose_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; syntax-checked with cuDNN 9.15.1",
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "verification": "syntax-checked; execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("python", "slice_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; syntax-checked with cuDNN 9.15.1",
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "verification": "syntax-checked; standalone execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("python", "concatenate_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; syntax-checked with cuDNN 9.15.1",
        "fallback_command": ["{python}", "-m", "py_compile", "{path}"],
        "verification": "syntax-checked; execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("cpp", "transpose_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "verification": "compiled against current headers; execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("cpp", "slice_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "verification": "compiled against current headers; standalone execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("cpp", "concatenate_basic"): {
        "requirements": ["sm100", "cudnn>=92200"],
        "api_availability": "Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; compiled but not executed with cuDNN 9.15.1",
        "fallback_command": [
            "nvcc",
            "-std=c++17",
            "-Iinclude",
            "{path}",
            "-lcudnn",
            "-lnvrtc",
            "-o",
            "{binary}",
        ],
        "verification": "compiled against current headers; execution requires CUDA 13.1+, cuDNN 9.22+, and SM100+",
    },
    ("cpp", "resample_basic"): {
        "owners": ["api/cpp/resample.md", "how-to/resampling.md"],
        "purpose": "Execute 2-D max pooling and compare with an index-based reference.",
        "expected": "max-pooling output: shape=[1, 1, 2, 2], values=[6, 8, 14, 16]",
    },
}

POINTWISE = {
    "abs",
    "add",
    "add_square",
    "bias",
    "binary_select",
    "ceil",
    "cmp_eq",
    "cmp_ge",
    "cmp_gt",
    "cmp_le",
    "cmp_lt",
    "cmp_neq",
    "cos",
    "div",
    "elu",
    "elu_backward",
    "erf",
    "exp",
    "floor",
    "gelu",
    "gelu_approx_tanh",
    "gelu_approx_tanh_backward",
    "gelu_backward",
    "identity",
    "leaky_relu",
    "leaky_relu_backward",
    "log",
    "logical_and",
    "logical_not",
    "logical_or",
    "max",
    "min",
    "mod",
    "mul",
    "neg",
    "pow",
    "reciprocal",
    "relu",
    "relu_backward",
    "rsqrt",
    "scale",
    "sigmoid",
    "sigmoid_backward",
    "sin",
    "softplus",
    "softplus_backward",
    "sqrt",
    "sub",
    "swish",
    "swish_backward",
    "tan",
    "tanh",
    "tanh_backward",
}
NORMALIZATION = {
    "adalayernorm",
    "adalayernorm_backward",
    "batchnorm",
    "batchnorm_backward",
    "batchnorm_inference",
    "bn_finalize",
    "dbn_weight",
    "genstats",
    "instancenorm",
    "instancenorm_backward",
    "layernorm",
    "layernorm_backward",
    "rmsnorm",
    "rmsnorm_backward",
}
ATTENTION = {
    "diagonal_band_mask",
    "sdpa",
    "sdpa_backward",
    "sdpa_fp8",
    "sdpa_fp8_backward",
    "sdpa_mxfp8",
    "sdpa_mxfp8_backward",
}
CONVOLUTION = {"conv_fprop", "conv_dgrad", "conv_wgrad"}
SHAPE = {"concatenate", "reshape", "resample", "slice", "transpose"}
LOW_PRECISION = {"block_scale_quantize", "block_scale_dequantize", "moe_grouped_matmul", "moe_grouped_matmul_bwd"}
ROUTING = {
    "nodes",
    "plans",
    "select_plan",
    "selected_engine",
    "tensors",
    "context",
    "find_tensor",
    "get_inputs",
    "get_node",
    "get_outputs",
    "inspect",
}


def source_target_commit() -> str:
    compatibility = json.loads((DOCS_ROOT / "compatibility.json").read_text(encoding="utf-8"))
    target = compatibility.get("target_commit")
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-f]{40}", target):
        raise RuntimeError("docs/compatibility.json must contain a full target_commit SHA")
    subprocess.run(
        ["git", "cat-file", "-e", f"{target}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return target


def write_or_check(path: Path, data: dict, check: bool) -> bool:
    content = json.dumps(data, indent=2, sort_keys=False) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return True
    if check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def page_summary(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    paragraph: list[str] = []
    for line in lines[1:]:
        if line.startswith("## "):
            break
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            break
    summary = " ".join(paragraph)
    summary = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", summary)
    return title, summary or f"Reference for {title}."


def _source_ref(path: Path, line: int) -> str:
    return f"{path.relative_to(REPO_ROOT).as_posix()}:{line}"


@lru_cache(maxsize=None)
def _source_lines(path: Path) -> tuple[str, ...]:
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _matching_lines(path: Path, pattern: str, *, start: int = 1, end: int | None = None) -> list[int]:
    expression = re.compile(pattern)
    return [
        number for number, line in enumerate(_source_lines(path), start=1) if number >= start and (end is None or number <= end) and expression.search(line)
    ]


@lru_cache(maxsize=None)
def python_source_locations(stem: str) -> tuple[str, str, str, str]:
    path = REPO_ROOT / "python" / "cudnn" / "_pygraph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pygraph = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pygraph")
    direct = next(
        (node.lineno for node in pygraph.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == stem),
        None,
    )
    if direct is not None:
        reference = _source_ref(path, direct)
        return reference, reference, stem, stem

    quoted_key = rf'["\']{re.escape(stem)}["\']'
    if stem in POINTWISE:
        declarations = _matching_lines(path, quoted_key, start=500, end=630)
        installers = _matching_lines(path, r"^def _install_pointwise_builders\(")
    elif stem in NORMALIZATION | CONVOLUTION | SHAPE | LOW_PRECISION | {"reduction", "rope", "rope_backward"}:
        declarations = _matching_lines(path, quoted_key, start=1500, end=1884)
        installers = _matching_lines(path, r"^def _install_structured_builders\(")
    elif stem in ATTENTION:
        declarations = _matching_lines(path, quoted_key, start=1885, end=2047)
        installers = _matching_lines(path, r"^def _install_captured_builders\(")
    else:
        declarations = []
        installers = []
    if declarations and installers:
        installer_name = {
            "pointwise": "_install_pointwise_builders",
            "structured": "_install_structured_builders",
            "captured": "_install_captured_builders",
        }
        if stem in POINTWISE:
            implementation_token = installer_name["pointwise"]
        elif stem in ATTENTION:
            implementation_token = installer_name["captured"]
        else:
            implementation_token = installer_name["structured"]
        return _source_ref(path, declarations[0]), _source_ref(path, installers[0]), stem, implementation_token

    init_path = REPO_ROOT / "python" / "cudnn" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"))
    wrapper = next(
        (node for node in init_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == f"_{stem}"),
        None,
    )
    if wrapper is not None:
        implementation_pattern = rf"\bPyGraph::{re.escape(stem)}\s*\("
        implementations = [
            (binding_path, line)
            for binding_path in sorted((REPO_ROOT / "python" / "pygraph").glob("*.cpp"))
            for line in _matching_lines(binding_path, implementation_pattern)
        ]
        declaration = _source_ref(init_path, wrapper.lineno)
        implementation = _source_ref(*implementations[0]) if implementations else declaration
        return declaration, implementation, f"_{stem}", stem if implementations else f"_{stem}"

    binding_pattern = rf'\.def\("{re.escape(stem)}"'
    implementation_pattern = rf"\bPyGraph::{re.escape(stem)}\s*\("
    for binding_path in sorted((REPO_ROOT / "python" / "pygraph").glob("*.cpp")):
        bindings = _matching_lines(binding_path, binding_pattern)
        if not bindings:
            continue
        implementations = _matching_lines(binding_path, implementation_pattern)
        declaration = _source_ref(binding_path, bindings[0])
        implementation = _source_ref(binding_path, implementations[0]) if implementations else declaration
        return declaration, implementation, stem, stem

    raise RuntimeError(f"cannot resolve exact Python source for {stem}")


@lru_cache(maxsize=None)
def cpp_source_locations(stem: str) -> tuple[str, str, str, str]:
    graph_interface = REPO_ROOT / "include" / "cudnn_frontend" / "graph_interface.h"
    node_interface = REPO_ROOT / "include" / "cudnn_frontend" / "node_interface.h"
    public_name = {
        "sdpa_mxfp8": "sdpa_fp8",
        "sdpa_mxfp8_backward": "sdpa_fp8_backward",
        "tensor_scalar": "tensor",
    }.get(stem, "pointwise" if stem in POINTWISE or stem == "gen_index" else stem)

    definitions: list[tuple[Path, int]] = []
    definition_pattern = rf"\b(?:Graph|INode)::{re.escape(public_name)}\s*\("
    for path in [graph_interface, *sorted((REPO_ROOT / "include" / "cudnn_frontend" / "node").glob("*.h"))]:
        definitions.extend((path, line) for line in _matching_lines(path, definition_pattern))

    graph_definitions = [(path, line) for path, line in definitions if "Graph::" in _source_lines(path)[line - 1]]
    node_definitions = [(path, line) for path, line in definitions if "INode::" in _source_lines(path)[line - 1]]
    declaration_pattern = rf"^[^(]*\b{re.escape(public_name)}\s*\("
    graph_declarations = _matching_lines(graph_interface, declaration_pattern, start=603)
    node_declarations = _matching_lines(node_interface, declaration_pattern, start=390)

    if graph_definitions and graph_declarations:
        declaration = _source_ref(graph_interface, graph_declarations[0])
        implementation = _source_ref(*graph_definitions[0])
    elif node_definitions and node_declarations:
        declaration = _source_ref(node_interface, node_declarations[0])
        implementation = _source_ref(*node_definitions[0])
    elif graph_declarations:
        declaration = _source_ref(graph_interface, graph_declarations[0])
        implementation = declaration
    elif node_declarations:
        declaration = _source_ref(node_interface, node_declarations[0])
        implementation = declaration
    else:
        raise RuntimeError(f"cannot resolve exact C++ source for {stem} (public symbol {public_name})")
    return declaration, implementation, public_name, public_name


def absence_source_locations(language: str) -> tuple[str, str, str, str]:
    if language == "python":
        path = REPO_ROOT / "python" / "cudnn" / "_pygraph.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pygraph = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pygraph")
        getattr_method = next(node for node in pygraph.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__")
        reference = _source_ref(path, getattr_method.lineno)
        return reference, reference, "__getattr__", "__getattr__"
    path = REPO_ROOT / "include" / "cudnn_frontend" / "graph_interface.h"
    class_lines = _matching_lines(path, r"^class Graph\b")
    if not class_lines:
        raise RuntimeError("cannot locate the C++ Graph surface used as absence evidence")
    reference = _source_ref(path, class_lines[0])
    return reference, reference, "class Graph", "class Graph"


def family(stem: str) -> tuple[str, str]:
    if stem == "matmul":
        return "core", "how-to/matmul-layouts.md#choose-a-layout"
    if stem in POINTWISE or stem in {"reduction", "gen_index"}:
        return "core", "tutorials/pointwise-reduction.md#pointwise-addition"
    if stem in NORMALIZATION:
        return "normalization", "tutorials/normalization.md#build-forward"
    if stem in ATTENTION:
        return ("low-precision" if "fp8" in stem else "attention"), "tutorials/attention.md#build-forward-and-backward-graphs"
    if stem in CONVOLUTION:
        return "convolution", "tutorials/convolution.md#build-forward"
    if stem in SHAPE:
        return "core", "how-to/shape-operations.md#choose-an-operation"
    if stem in LOW_PRECISION:
        return "low-precision", "concepts/numerical-precision.md#block-scaling"
    if stem in ROUTING:
        return "python-routing", "how-to/custom-engine.md#inspect-automatic-selection"
    if stem in {"rope", "rope_backward"}:
        return "attention", "how-to/rope.md#build-rope"
    return "core", "concepts/api-layers.md#graph-lifecycle"


def sample_id(language: str, stem: str) -> str | None:
    extension = "py" if language == "python" else "cc"
    exact = SAMPLE_ROOT / language / f"{stem}_basic.{extension}"
    if exact.exists():
        return f"{language}-{stem.replace('_', '-')}-basic"
    if language == "python" and stem == "sdpa":
        return "python-sdpa-bf16-causal"
    if language == "python" and stem in ROUTING:
        return "python-custom-engine"
    return None


def cpp_type_learning_anchor(name: str) -> str:
    normalized = name.casefold()
    if any(token in normalized for token in ("sdpa", "softmax", "diagonalband", "pagedcache")):
        return "concepts/attention.md#tensor-layouts"
    if "moe" in normalized:
        return "tutorials/moe-grouped-matmul.md#build-grouped-execution"
    if "block_scale" in normalized:
        return "concepts/numerical-precision.md#block-scaling"
    if any(token in normalized for token in ("kernelcache", "deviceproperties")):
        return "concepts/compilation-caching.md#cache-identity"
    if "rope" in normalized:
        return "how-to/rope.md#build-rope"
    if any(token in normalized for token in ("norm", "bn_", "genstats")):
        return "tutorials/normalization.md#choose-axes-and-state"
    if any(token in normalized for token in ("conv_", "convolution")):
        return "tutorials/convolution.md#define-layouts-and-attributes"
    if "matmul" in normalized:
        return "how-to/matmul-layouts.md#choose-a-layout"
    if any(token in normalized for token in ("pointwise", "reduction")):
        return "tutorials/pointwise-reduction.md#principles"
    if normalized == "rng_attributes":
        return "tutorials/pointwise-reduction.md#pointwise-addition"
    if any(token in normalized for token in ("reshape", "slice", "transpose", "concatenate")):
        return "how-to/shape-operations.md#choose-an-operation"
    if "resample" in normalized:
        return "how-to/resampling.md#determine-output-shape"
    if any(token in normalized for token in ("execution_plan", "engine", "knob")):
        return "concepts/execution-plans.md#c-planning"
    if any(token in normalized for token in ("tensor", "scalar")):
        return "concepts/graphs-and-tensors.md#tensor-identity"
    if normalized == "graph":
        return "concepts/api-layers.md#graph-lifecycle"
    if normalized.endswith("node"):
        return "concepts/api-layers.md#graph-lifecycle"
    return ""


def python_export_learning_anchor(name: str, optional: bool) -> str:
    normalized = name.casefold()
    compact = normalized.replace("_", "")
    if optional:
        if normalized == "bsa":
            return "reference/oss-block-sparse-attention.md#install-and-import"
        if "block_sparse_attention_forward" in normalized:
            return "reference/oss-block-sparse-attention.md#block_sparse_attention_forward"
        if "block_sparse_attention_backward" in normalized:
            return "reference/oss-block-sparse-attention.md#block_sparse_attention_backward"
        if normalized == "dsa":
            return "reference/oss-deepseek-sparse-attention.md#dsa-namespace"
        if normalized == "nsa":
            return "reference/oss-native-sparse-attention.md#nsa-namespace"
        if normalized in {"grouped_gemm"}:
            return "reference/oss-grouped-gemm.md#describe-expert-groups"
        if normalized in {"discrete_grouped_gemm"}:
            return "reference/oss-discrete-grouped-gemm.md#build-pointer-arrays"
        if normalized.startswith("discretegroupedgemm") or normalized.startswith("discrete_grouped_gemm"):
            suffix = "dswiglu" if "dswiglu" in normalized else "swiglu"
            symbol = (
                f"DiscreteGroupedGemm{suffix.capitalize()}Sm100"
                if not normalized.startswith("discrete_grouped")
                else f"discrete_grouped_gemm_{suffix}_wrapper_sm100"
            )
            return f"reference/oss-discrete-grouped-gemm.md#{symbol.casefold()}"
        if normalized.startswith("groupedgemm") or normalized.startswith("grouped_gemm"):
            variants = (
                "glu_hadamard_quant",
                "glu_hadamard",
                "dswiglu",
                "dsrelu",
                "dglu",
                "swiglu",
                "srelu",
                "wgrad",
                "quant",
                "glu",
            )
            variant = next((item for item in variants if item.replace("_", "") in compact), "unfused")
            if normalized.startswith("groupedgemm"):
                class_name = {
                    "unfused": "GroupedGemmSm100",
                    "glu_hadamard_quant": "GroupedGemmGluHadamardQuantSm100",
                    "glu_hadamard": "GroupedGemmGluHadamardSm100",
                    "dswiglu": "GroupedGemmDswigluSm100",
                    "dsrelu": "GroupedGemmDsreluSm100",
                    "dglu": "GroupedGemmDgluSm100",
                    "swiglu": "GroupedGemmSwigluSm100",
                    "srelu": "GroupedGemmSreluSm100",
                    "wgrad": "GroupedGemmWgradSm100",
                    "quant": "GroupedGemmQuantSm100",
                    "glu": "GroupedGemmGluSm100",
                }[variant]
                return f"reference/oss-grouped-gemm.md#{class_name.casefold()}"
            wrapper = "grouped_gemm_wrapper_sm100" if variant == "unfused" else f"grouped_gemm_{variant}_wrapper_sm100"
            return f"reference/oss-grouped-gemm.md#{wrapper}"
        if normalized.startswith("gemm") or normalized.startswith("gemm_"):
            variants = ("proj_rope_mxfp8", "dsrelu", "swiglu", "srelu", "amax")
            variant = next(item for item in variants if item.replace("_", "") in compact)
            if normalized.startswith("gemm_"):
                return f"reference/oss-dense-gemm.md#gemm_{variant}_wrapper_sm100"
            class_name = {
                "proj_rope_mxfp8": "GemmProjRopeMxfp8Sm100",
                "dsrelu": "GemmDsreluSm100",
                "swiglu": "GemmSwigluSm100",
                "srelu": "GemmSreluSm100",
                "amax": "GemmAmaxSm100",
            }[variant]
            return f"reference/oss-dense-gemm.md#{class_name.casefold()}"
        if "sdpa_fwd" in normalized:
            return "reference/oss-sdpa-d256.md#sdpa_fwd_wrapper_sm100_d256"
        if "sdpa_bwd" in normalized:
            return "reference/oss-sdpa-d256.md#sdpa_bwd_wrapper_sm100_d256"
        if normalized == "sdpafwdsm100d256":
            return "reference/oss-sdpa-d256.md#sdpafwdsm100d256"
        if normalized == "sdpabwdsm100d256":
            return "reference/oss-sdpa-d256.md#sdpabwdsm100d256"
        if normalized == "rmsnormrhtamaxsm100":
            return "reference/oss-normalization.md#rmsnormrhtamaxsm100"
        if normalized == "rmsnorm_rht_amax_wrapper_sm100":
            return "reference/oss-normalization.md#rmsnorm_rht_amax_wrapper_sm100"
        return "reference/oss-engines.md#choose-an-entry-point"
    if "causal_conv1d" in normalized:
        return "how-to/causal-convolution.md#low-level-pointer-bindings"
    if "moe" in normalized:
        return "tutorials/moe-grouped-matmul.md#distinguish-wrappers"
    if name == "Graph":
        return "how-to/python-graph.md#declare-the-call-contract"
    if name in {"GraphContext", "pygraph"}:
        return "concepts/api-layers.md#graph-lifecycle"
    if normalized in {"graph", "jit", "graph_cache"}:
        return "how-to/graph-cache.md#choose-the-decorator"
    if normalized in {"node", "nodetype"}:
        return "concepts/python-ir-routing.md#operation-declaration-mechanisms"
    if any(token in normalized for token in ("handle", "stream", "last_error")):
        return "how-to/runtime-resources.md#own-the-handle"
    if "kernel_cache" in normalized or "device_properties" in normalized:
        return "concepts/compilation-caching.md#cache-identity"
    if any(token in normalized for token in ("tensor", "data_type", "scalar_type", "reordering")):
        return "concepts/layouts.md#data-types"
    if any(token in normalized for token in ("heur", "knob", "plan_policy", "note")):
        return "concepts/execution-plans.md#notes-and-policy"
    if any(token in normalized for token in ("attention", "diagonal")):
        return "concepts/attention.md#masks"
    if "norm_" in normalized:
        return "tutorials/normalization.md#choose-axes-and-state"
    if "reduction" in normalized:
        return "tutorials/pointwise-reduction.md#principles"
    if "reshape" in normalized:
        return "how-to/shape-operations.md#reshape"
    if "version" in normalized:
        return "getting-started/verify-environment.md#interpret-the-result"
    if "error" in normalized:
        return "how-to/troubleshoot.md#validate-the-description"
    return "concepts/api-layers.md#graph-lifecycle"


def optional_namespace_learning_anchor(namespace: str, name: str) -> str:
    normalized = name.casefold()
    if namespace == "BSA":
        return f"reference/oss-block-sparse-attention.md#{normalized}"
    if namespace == "DSA":
        if "indexer_forward" in normalized:
            return "reference/oss-deepseek-sparse-attention.md#indexer_forward_wrapper"
        if any(token in normalized for token in ("topk", "top_k", "local_to_global", "compactify")):
            return "reference/oss-deepseek-sparse-attention.md#indexer_top_k_wrapper"
        if "recompute" in normalized:
            return "reference/oss-deepseek-sparse-attention.md#score-recomputation"
        if "backward" in normalized:
            return "reference/oss-deepseek-sparse-attention.md#backward-stages"
        return "reference/oss-deepseek-sparse-attention.md#class-apis"
    nsa_sections = {
        "CompressionAttention": "compression_attention_wrapper",
        "compression_attention_wrapper": "compression_attention_wrapper",
        "TopKReduction": "topk_reduction_wrapper",
        "topk_reduction_wrapper": "topk_reduction_wrapper",
        "SelectionAttention": "selection_attention_wrapper",
        "selection_attention_wrapper": "selection_attention_wrapper",
        "SlidingWindowAttention": "sliding_window_attention_wrapper",
        "sliding_window_attention_wrapper": "sliding_window_attention_wrapper",
    }
    return f"reference/oss-native-sparse-attention.md#{nsa_sections[name]}"


def discover_optional_namespace_members() -> dict[str, dict[str, str]]:
    """Return advertised namespace members and their defining source lines."""
    packages = {
        "BSA": REPO_ROOT / "python" / "cudnn" / "block_sparse_attention" / "__init__.py",
        "DSA": REPO_ROOT / "python" / "cudnn" / "deepseek_sparse_attention" / "__init__.py",
        "NSA": REPO_ROOT / "python" / "cudnn" / "native_sparse_attention" / "__init__.py",
    }
    discovered: dict[str, dict[str, str]] = {}
    for namespace, path in packages.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        members: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "_SYMBOLS" for target in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        members[key.value] = f"{path.relative_to(REPO_ROOT).as_posix()}:{key.lineno}"
            if isinstance(node, ast.ClassDef) and node.name == f"{namespace}Namespace":
                for member in node.body:
                    if not isinstance(member, ast.Assign):
                        continue
                    for target in member.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            members[target.id] = f"{path.relative_to(REPO_ROOT).as_posix()}:{target.lineno}"
        if not members:
            raise RuntimeError(f"no advertised members discovered for {namespace} in {path.relative_to(REPO_ROOT)}")
        discovered[namespace] = members
    return discovered


def candidate_compatibility_id(candidate: dict) -> str:
    anchor = candidate["learning_anchor"]
    if candidate["disposition"] == "Legacy":
        return "legacy"
    if anchor.startswith("reference/oss-"):
        return "oss"
    if "attention" in anchor:
        return "attention"
    if "normalization" in anchor:
        return "normalization"
    if "convolution" in anchor:
        return "convolution"
    if "moe" in anchor or "block-scal" in anchor or "numerical-precision" in anchor:
        return "low-precision"
    if "python-ir" in anchor or "custom-engine" in anchor:
        return "python-routing"
    return "core"


PYTHON_ENUMS = {
    "attention_implementation",
    "behavior_note",
    "build_plan_policy",
    "cuda_graph_retention",
    "data_type",
    "diagonal_alignment",
    "heur_mode",
    "knob_type",
    "moe_grouped_matmul_mode",
    "norm_forward_phase",
    "numerical_note",
    "reduction_mode",
    "reshape_mode",
    "scalar_type",
    "tensor_reordering",
}
PYTHON_TYPES = {"decline_types", "handle", "knob", "tensor"}
PYTHON_NAMESPACES = {"BSA", "CSA", "DSA", "NSA"}
PYTHON_PURPOSE_OVERRIDES = {
    "mask_to_kv_lengths": "Convert a prefix-shaped Boolean attention mask to right-padded KV sequence lengths, returning `None` for a non-prefix mask.",
    "prefix_mask_to_kv_lengths": "Convert an asserted prefix-shaped Boolean attention mask to right-padded KV sequence lengths without host synchronization.",
    "triangle_attention": "Run triangle attention with the cuEquivariance-compatible signature and optional output, stream, and prefix-mask controls.",
    "triangle_multiplicative_update": "Run triangle multiplicative update with the cuEquivariance-compatible signature and optional output and stream controls.",
    "set_reordering_type": "Set the tensor memory reordering layout, such as `F8_128x4`.",
}


def candidate_kind(candidate: dict) -> str:
    """Classify a family-owned symbol from its declaration and qualified owner."""
    qualified = candidate["qualified_name"]
    short_name = qualified.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    if candidate["language"] == "cpp":
        path_value, _, line_value = candidate["evidence"].rpartition(":")
        declaration = (REPO_ROOT / path_value).read_text(encoding="utf-8").splitlines()[int(line_value) - 1]
        if re.search(r"\benum\s+class\b", declaration):
            return "enum"
        if re.search(r"\bstruct\b", declaration):
            return "struct"
        return "class"
    if qualified.startswith("cudnn.pygraph.") or re.match(r"^cudnn\.(?:Graph|GraphContext|Node|Tensor)\.", qualified):
        return "method"
    if short_name in PYTHON_ENUMS or short_name == "NodeType":
        return "enum"
    if short_name in PYTHON_NAMESPACES:
        return "namespace"
    if short_name in PYTHON_TYPES:
        return "type"
    if short_name.isupper():
        return "constant"
    if short_name[:1].isupper() or short_name.endswith("Error"):
        return "class"
    return "function"


@lru_cache(maxsize=1)
def python_doc_summaries() -> dict[str, list[tuple[str, str]]]:
    """Index the first authored docstring paragraph for public Python definitions."""
    summaries: dict[str, list[tuple[str, str]]] = {}
    for path in sorted((REPO_ROOT / "python" / "cudnn").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=True)
            if not docstring:
                continue
            first_paragraph = re.split(r"\n\s*\n", docstring, maxsplit=1)[0]
            paragraph = " ".join(line.strip() for line in first_paragraph.splitlines() if line.strip())
            summary = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", paragraph, maxsplit=1)[0]
            if summary and len(summary) <= 240:
                summaries.setdefault(node.name, []).append((relative, summary.replace("|", "\\|")))
    return summaries


def python_doc_summary(candidate: dict) -> str | None:
    if candidate["language"] != "python":
        return None
    short_name = candidate["qualified_name"].rsplit(".", 1)[-1]
    matches = python_doc_summaries().get(short_name, [])
    if not matches:
        return None
    evidence_path = candidate["evidence"].rpartition(":")[0]
    return next((summary for path, summary in matches if path == evidence_path), matches[0][1])


def candidate_summary(candidate: dict) -> str:
    """Return a concise, identifier-preserving purpose for an index entry."""
    short_name = candidate["qualified_name"].rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    if candidate["language"] == "python" and short_name in PYTHON_PURPOSE_OVERRIDES:
        return PYTHON_PURPOSE_OVERRIDES[short_name]
    source_summary = python_doc_summary(candidate)
    if source_summary:
        return source_summary
    qualified = candidate["qualified_name"]
    kind = candidate["kind"]
    if kind == "method":
        owner = qualified.rsplit(".", 1)[0]
        return f"Call `{short_name}` on `{owner}`."
    if kind == "function":
        return f"Call the `{short_name}` public function."
    if kind == "enum":
        return f"Select a value from the `{short_name}` enumeration."
    if kind == "constant":
        return f"Read the `{short_name}` public constant."
    if kind == "namespace":
        return f"Access the `{short_name}` optional-kernel namespace."
    return f"Use the public `{short_name}` {kind}."


def candidate_stability(disposition: str) -> str:
    return {"Legacy": "Legacy", "Experimental": "Experimental", "Unreleased": "Unreleased"}.get(disposition, "Stable")


def python_symbol_evidence(relative_path: str, name: str) -> str:
    """Locate a Python definition or export entry whose line contains ``name``."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return f"{relative_path}:{node.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == name:
            return f"{relative_path}:{node.lineno}"
        if isinstance(node, ast.alias) and (node.asname == name or node.name == name):
            return f"{relative_path}:{node.lineno}"
    raise RuntimeError(f"cannot find {name!r} in {relative_path}")


def cpp_type_definitions(path: Path) -> list[tuple[str, int]]:
    """Return namespace-qualified, top-level C++ type definitions from ``path``."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    namespace: list[str] = []
    namespace_frames: list[tuple[int, list[str]]] = []
    brace_depth = 0
    definitions: list[tuple[str, int]] = []
    offset = 0
    namespace_pattern = re.compile(r"^[ \t]*namespace[ \t]+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)[ \t]*\{")
    type_pattern = re.compile(r"^[ \t]*(?:class|struct|enum[ \t]+class)[ \t]+([A-Z][A-Za-z0-9_]*)\b")

    for line_number, line in enumerate(lines, start=1):
        namespace_match = namespace_pattern.match(line)
        if namespace_match:
            previous = namespace.copy()
            parts = namespace_match.group(1).split("::")
            namespace = parts if parts[0] == "cudnn_frontend" else [*namespace, *parts]
            opening_depth = brace_depth + 1
            namespace_frames.append((opening_depth, previous))

        type_match = type_pattern.match(line)
        namespace_depth = namespace_frames[-1][0] if namespace_frames else -1
        if type_match and brace_depth == namespace_depth and namespace in (["cudnn_frontend"], ["cudnn_frontend", "graph"]):
            absolute_match_end = offset + type_match.end()
            next_open = text.find("{", absolute_match_end)
            next_semicolon = text.find(";", absolute_match_end)
            if next_open != -1 and (next_semicolon == -1 or next_open < next_semicolon):
                definitions.append(("::".join([*namespace, type_match.group(1)]), line_number))

        brace_depth += line.count("{") - line.count("}")
        while namespace_frames and brace_depth < namespace_frames[-1][0]:
            _, namespace = namespace_frames.pop()
        offset += len(line) + 1

    return definitions


def add_candidate_contract(candidate: dict, compatibility_by_id: dict[str, dict]) -> dict:
    anchor = candidate["learning_anchor"]
    candidate["compatibility_id"] = candidate_compatibility_id(candidate)
    candidate["kind"] = candidate_kind(candidate)
    candidate["summary"] = candidate_summary(candidate)
    candidate["stability"] = candidate_stability(candidate["disposition"])
    candidate["availability_contract"] = candidate_availability_contract(candidate, compatibility_by_id[candidate["compatibility_id"]])
    incomplete_families = (
        "reference/oss-dense-gemm.md#",
        "reference/oss-discrete-grouped-gemm.md#",
        "reference/oss-grouped-gemm.md#",
    )
    if candidate["disposition"] == "Legacy":
        candidate["constraint_note"] = (
            "Header inventory verifies this legacy symbol; its descriptor or operation determines backend support, and no broader version or architecture claim is made."
        )
    elif anchor.startswith(incomplete_families):
        candidate["constraint_note"] = "The linked wrapper section is authoritative for its dtype and scale-layout matrix."
    elif candidate["compatibility_id"] == "oss" and anchor.endswith("#choose-an-entry-point"):
        candidate["constraint_note"] = "The optional export was import-verified; execution support remains package-, dtype-, shape-, and architecture-specific."
    elif candidate["kind"] in {"constant", "enum", "namespace", "type"}:
        candidate["constraint_note"] = "This host-side symbol does not execute independently; the consuming operation or engine determines runtime support."
    elif candidate["language"] == "cpp" and candidate["qualified_name"].endswith("_attributes"):
        candidate["constraint_note"] = (
            "This metadata type does not execute independently; the operation configured by these attributes determines runtime support."
        )
    else:
        candidate["constraint_note"] = (
            "This is the symbol's mapped compatibility envelope; the linked page states the operation- or workflow-specific constraints."
        )
    candidate["constraint_status"] = "documented"
    return candidate


def build_api_manifest() -> dict:
    ending_commit = source_target_commit()
    compatibility_by_id = {feature["id"]: feature for feature in json.loads((DOCS_ROOT / "compatibility.json").read_text(encoding="utf-8"))["features"]}
    pages: dict[str, set[str]] = {
        language: {path.stem for path in (API_ROOT / language).glob("*.md") if path.stem != "index"} for language in ("python", "cpp")
    }
    rows: list[dict] = []
    for language in ("python", "cpp"):
        for path in sorted((API_ROOT / language).glob("*.md")):
            if path.stem == "index":
                continue
            title, summary = page_summary(path)
            stem = path.stem
            compatibility_id, learning_anchor = family(stem)
            qualified = f"cudnn.pygraph.{stem}" if language == "python" else f"cudnn_frontend::graph::Graph::{stem}"
            counterpart_language = "cpp" if language == "python" else "python"
            counterpart = (
                (f"cudnn_frontend::graph::Graph::{stem}" if language == "python" else f"cudnn.pygraph.{stem}") if stem in pages[counterpart_language] else None
            )
            sid = sample_id(language, stem)
            unreleased = language == "python" and stem in ROUTING
            not_exposed = stem in {"estimate_run_times"}
            if not_exposed:
                source_declaration, implementation, declaration_token, implementation_token = absence_source_locations(language)
                source_resolution = "absence-evidence"
            else:
                source_declaration, implementation, declaration_token, implementation_token = (
                    python_source_locations(stem) if language == "python" else cpp_source_locations(stem)
                )
                source_resolution = "exact"
            sample_override = SAMPLE_OVERRIDES.get((language, f"{stem}_basic"), {})
            rows.append(
                {
                    "language": language,
                    "qualified_name": qualified,
                    "title": title,
                    "kind": "method",
                    "summary": summary,
                    "disposition": "Not exposed" if not_exposed else ("Unreleased" if unreleased else "Public"),
                    "stability": "Historical" if not_exposed else ("Experimental" if "oss" in stem else "Stable"),
                    "availability": sample_override.get("api_availability")
                    or (
                        "Not exposed by the current Python binding or C++ Graph class"
                        if not_exposed
                        else ("Current checkout" if unreleased else "See local constraints")
                    ),
                    "source_declaration": source_declaration,
                    "implementation": implementation,
                    "source_resolution": source_resolution,
                    "source_declaration_token": declaration_token,
                    "implementation_token": implementation_token,
                    "evidence": [source_declaration, implementation, f"docs/{path.relative_to(DOCS_ROOT).as_posix()}"],
                    "page": path.relative_to(DOCS_ROOT).as_posix(),
                    "learning_anchor": learning_anchor,
                    "counterpart": counterpart,
                    "aliases": [],
                    "compatibility_id": compatibility_id,
                    "samples": [sid] if sid else [],
                    "verification": sample_override.get("verification")
                    or (
                        "binding absence verified"
                        if not_exposed
                        else (
                            "source-verified; canonical sample passed on CUDA 13.1/cuDNN 9.15.1/SM100"
                            if sid
                            else "source-verified; no compatible canonical sample published"
                        )
                    ),
                }
            )

    documented = {row["qualified_name"] for row in rows}
    old_python = {row["title"] for row in rows if row["language"] == "python"}
    tree = ast.parse((REPO_ROOT / "python" / "cudnn" / "_pygraph.py").read_text(encoding="utf-8"))
    pygraph = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "pygraph")
    current_methods = {node.name: node for node in pygraph.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")}
    unresolved: list[dict] = []
    for name in sorted(current_methods.keys() - old_python):
        unresolved.append(
            {
                "language": "python",
                "qualified_name": f"cudnn.pygraph.{name}",
                "disposition": "Unreleased",
                "evidence": f"python/cudnn/_pygraph.py:{current_methods[name].lineno}",
                "learning_anchor": "concepts/api-layers.md#graph-lifecycle",
                "suggestion": f"Add api/python/{name}.md using the callable template.",
            }
        )

    init_tree = ast.parse((REPO_ROOT / "python" / "cudnn" / "__init__.py").read_text(encoding="utf-8"))
    top_level: dict[str, str] = {
        name: python_symbol_evidence(path, name)
        for name, path in {
            "Graph": "python/cudnn/wrapper.py",
            "GraphContext": "python/cudnn/_pygraph.py",
            "Node": "python/cudnn/nodes.py",
            "NodeType": "python/cudnn/graph_types.py",
            "Tensor": "python/cudnn/graph_types.py",
            "graph": "python/cudnn/graph.py",
            "graph_cache": "python/cudnn/graph.py",
            "jit": "python/cudnn/graph.py",
            "pygraph": "python/cudnn/_pygraph.py",
        }.items()
    }
    optional_top_level: set[str] = set()
    for node in init_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) for target in node.targets):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "symbols_to_import" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                for element in node.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        top_level[element.value] = f"python/cudnn/__init__.py:{element.lineno}"
            if "_LAZY_OPTIONAL_IMPORTS" in names and isinstance(node.value, ast.Dict):
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        top_level[key.value] = f"python/cudnn/__init__.py:{key.lineno}"
                        optional_top_level.add(key.value)
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_optional_symbol"
            and isinstance(node.iter, (ast.List, ast.Tuple))
        ):
            for element in node.iter.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    top_level[element.value] = f"python/cudnn/__init__.py:{element.lineno}"
    for qualified, evidence_path, evidence_name, anchor in (
        ("cudnn.ops.causal_conv1d", "python/cudnn/ops/__init__.py", "causal_conv1d", "how-to/causal-convolution.md#causal_conv1d"),
        (
            "cudnn.ops.causal_conv1d_nwh",
            "python/cudnn/ops/__init__.py",
            "causal_conv1d_nwh",
            "how-to/causal-convolution.md#causal_conv1d_nwh",
        ),
        (
            "cudnn.ops.b2b_causal_conv1d",
            "python/cudnn/ops/__init__.py",
            "b2b_causal_conv1d",
            "how-to/causal-convolution.md#b2b_causal_conv1d",
        ),
        (
            "cudnn.experimental.ops.moe_grouped_matmul",
            "python/cudnn/experimental/ops/__init__.py",
            "moe_grouped_matmul",
            "tutorials/moe-grouped-matmul.md#distinguish-wrappers",
        ),
        ("cudnn.yarn.compute_yarn_freqs", "python/cudnn/yarn.py", "compute_yarn_freqs", "how-to/rope.md#integrate-with-attention"),
        ("cudnn.yarn.compute_yarn_inv_freq", "python/cudnn/yarn.py", "compute_yarn_inv_freq", "how-to/rope.md#integrate-with-attention"),
        ("cudnn.yarn.yarn_get_mscale", "python/cudnn/yarn.py", "yarn_get_mscale", "how-to/rope.md#integrate-with-attention"),
    ):
        if qualified not in documented:
            unresolved.append(
                {
                    "language": "python",
                    "qualified_name": qualified,
                    "disposition": "Experimental" if ".experimental." in qualified else "Public",
                    "evidence": python_symbol_evidence(evidence_path, evidence_name),
                    "learning_anchor": anchor,
                    "suggestion": "Add a callable reference page and canonical sample.",
                }
            )
    engines_tree = ast.parse((REPO_ROOT / "python" / "cudnn" / "engines" / "__init__.py").read_text(encoding="utf-8"))
    engines_all = next(
        node.value
        for node in engines_tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    for element in engines_all.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            continue
        qualified = f"cudnn.engines.{element.value}"
        if qualified in documented:
            continue
        unresolved.append(
            {
                "language": "python",
                "qualified_name": qualified,
                "disposition": "Unreleased",
                "evidence": f"python/cudnn/engines/__init__.py:{element.lineno}",
                "learning_anchor": "how-to/custom-engine.md#inspect-automatic-selection",
                "suggestion": "Add a page matching the exported engine type, constant, or callable and link its canonical sample region.",
            }
        )
    for name, evidence in sorted(top_level.items()):
        qualified = f"cudnn.{name}"
        if qualified in documented:
            continue
        optional = name in optional_top_level
        unresolved.append(
            {
                "language": "python",
                "qualified_name": qualified,
                "disposition": "Experimental" if optional else "Public",
                "evidence": evidence,
                "learning_anchor": python_export_learning_anchor(name, optional),
                "suggestion": "Add a class, enum, type, or callable page matching the exported symbol kind.",
            }
        )

    for namespace, members in discover_optional_namespace_members().items():
        for name, evidence in members.items():
            qualified = f"cudnn.{namespace}.{name}"
            if qualified in documented:
                continue
            unresolved.append(
                {
                    "language": "python",
                    "qualified_name": qualified,
                    "disposition": "Experimental",
                    "evidence": evidence,
                    "learning_anchor": optional_namespace_learning_anchor(namespace, name),
                    "suggestion": "Promote to an individual callable or class page if the namespace becomes stable.",
                }
            )

    class_sources = {
        "Graph": REPO_ROOT / "python" / "cudnn" / "wrapper.py",
        "GraphContext": REPO_ROOT / "python" / "cudnn" / "_pygraph.py",
        "Tensor": REPO_ROOT / "python" / "cudnn" / "graph_types.py",
        "Node": REPO_ROOT / "python" / "cudnn" / "nodes.py",
    }
    for class_name, path in class_sources.items():
        class_tree = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(node for node in ast.walk(class_tree) if isinstance(node, ast.ClassDef) and node.name == class_name)
        for member in class_node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name.startswith("_") and member.name not in {"__call__", "__enter__", "__exit__"}:
                continue
            qualified = f"cudnn.{class_name}.{member.name}"
            if qualified in documented:
                continue
            unresolved.append(
                {
                    "language": "python",
                    "qualified_name": qualified,
                    "disposition": "Unreleased" if class_name in {"GraphContext", "Tensor", "Node"} else "Public",
                    "evidence": f"{path.relative_to(REPO_ROOT).as_posix()}:{member.lineno}",
                    "learning_anchor": (
                        "concepts/graphs-and-tensors.md#tensor-identity"
                        if class_name == "Tensor"
                        else (
                            "concepts/python-ir-routing.md#operation-declaration-mechanisms"
                            if class_name == "Node"
                            else "how-to/python-graph.md#build-on-context-exit"
                        )
                    ),
                    "suggestion": "Add an owner-qualified method page and a focused region in a canonical sample.",
                }
            )

    cpp_candidates: dict[str, tuple[str, str, str]] = {}
    modern_headers = [
        REPO_ROOT / "include" / "cudnn_frontend" / "graph_properties.h",
        REPO_ROOT / "include" / "cudnn_frontend" / "graph_interface.h",
        REPO_ROOT / "include" / "cudnn_frontend" / "knobs.h",
        REPO_ROOT / "include" / "cudnn_frontend" / "plans.h",
        REPO_ROOT / "include" / "cudnn_frontend" / "backend" / "kernel_cache.h",
        REPO_ROOT / "include" / "cudnn_frontend" / "backend" / "device_properties.h",
        *sorted((REPO_ROOT / "include" / "cudnn_frontend" / "node").glob("*.h")),
    ]
    for header in modern_headers:
        for qualified, line in cpp_type_definitions(header):
            name = qualified.rsplit("::", 1)[-1]
            if name in {"Attributes"}:
                continue
            cpp_candidates[qualified] = (
                "Public",
                f"{header.relative_to(REPO_ROOT).as_posix()}:{line}",
                cpp_type_learning_anchor(name),
            )
    # Uppercase suffixes are the legacy builder headers. Lowercase suffixes are
    # current umbrella, utility, and compatibility headers and must not inherit
    # the legacy disposition merely because of their filename prefix.
    for header in sorted((REPO_ROOT / "include").glob("cudnn_frontend_[A-Z]*.h")):
        for qualified, line in cpp_type_definitions(header):
            if qualified.count("::") != 1:
                continue
            cpp_candidates[qualified] = (
                "Legacy",
                f"{header.relative_to(REPO_ROOT).as_posix()}:{line}",
                "how-to/legacy-migration.md#map-the-construction-model",
            )
    for qualified, (disposition, evidence, anchor) in sorted(cpp_candidates.items()):
        if qualified in documented:
            continue
        unresolved.append(
            {
                "language": "cpp",
                "qualified_name": qualified,
                "disposition": disposition,
                "evidence": evidence,
                "learning_anchor": anchor,
                "suggestion": "Add a type or enum reference page; add owner-qualified member pages for its public methods.",
            }
        )
    unresolved = [add_candidate_contract(candidate, compatibility_by_id) for candidate in unresolved]
    return {
        "schema_version": 1,
        "target_commit": ending_commit,
        "generated_by": "tools/doc/build_doc_manifests.py",
        "apis": sorted(rows, key=lambda row: (row["language"], row["qualified_name"].casefold())),
        "unresolved_candidates": sorted(unresolved, key=lambda row: (row["language"], row["qualified_name"].casefold())),
    }


def owner_for_sample(language: str, stem: str) -> list[str]:
    normalized = stem
    for suffix in (
        "_basic",
        "_example_2",
        "_bf16_causal",
        "_inference_no_stats",
        "_dropout",
        "_block_mask",
        "_custom_score_modifier",
        "_paged_kv_cache",
        "_sink_token",
        "_sliding_window_chunked_prefill",
        "_variable_length_padded",
    ):
        normalized = normalized.removesuffix(suffix)
    if language == "python" and stem == "moe_grouped_matmul":
        return ["api/python/moe_grouped_matmul.md", "tutorials/moe-grouped-matmul.md"]
    page = API_ROOT / language / f"{normalized}.md"
    if page.exists():
        return [page.relative_to(DOCS_ROOT).as_posix()]
    if stem == "matmul" and language in {"python", "cpp"}:
        return ["getting-started/quickstart.md"]
    if stem == "custom_engine":
        return ["how-to/custom-engine.md"]
    if stem == "pytorch_cached_graph":
        return ["tutorials/pytorch-integration.md"]
    if stem == "pytorch_custom_operator":
        return ["how-to/pytorch-custom-operator.md"]
    if stem == "causal_conv1d":
        return ["how-to/causal-convolution.md"]
    if stem == "compile_time_constants":
        return ["how-to/scalars-and-constants.md"]
    if stem == "oss_rmsnorm_silu":
        return ["reference/oss-normalization.md"]
    return [f"samples/{language}/README.md"]


def purpose(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(text))
        except SyntaxError:
            doc = None
        if doc:
            return doc.splitlines()[0].strip()
    for line in text.splitlines():
        if path.suffix in {".cc", ".cpp"} and not line.lstrip().startswith("//"):
            continue
        stripped = line.lstrip("/ ").strip()
        if stripped and not stripped.startswith(("Copyright", "SPDX")):
            return stripped.rstrip(".") + "."
    return f"Exercise {path.stem.replace('_', ' ')}."


def build_sample_manifest() -> dict:
    rows: list[dict] = []
    for language in ("python", "cpp"):
        suffixes = {".py"} if language == "python" else {".cc", ".cpp"}
        for path in sorted((SAMPLE_ROOT / language).iterdir()):
            if path.suffix not in suffixes:
                continue
            stem = path.stem
            sid = f"{language}-{stem.replace('_', '-')}"
            mode = "execute" if language == "python" else "compile-run"
            command = (
                ["{python}", "{path}", "--strict"]
                if language == "python"
                else ["nvcc", "-std=c++17", "-Iinclude", "{path}", "-lcudnn", "-lnvrtc", "-o", "{binary}"]
            )
            sample_override = SAMPLE_OVERRIDES.get((language, stem), {})
            owners = sample_override.get("owners", owner_for_sample(language, stem))
            api_name = stem
            for suffix in ("_basic", "_example_2"):
                api_name = api_name.removesuffix(suffix)
            api_qualified = (
                "cudnn.Graph"
                if language == "python" and stem == "pytorch_cached_graph"
                else (f"cudnn.pygraph.{api_name}" if language == "python" else f"cudnn_frontend::graph::Graph::{api_name}")
            )
            requirements = ["python", "cuda"] if language == "python" else ["linux", "cuda", "nvcc"]
            verification = "executed on CUDA 13.0/cuDNN 9.15.1/SM100" if language == "python" else "executed on CUDA 13.1/cuDNN 9.15.1/SM100"
            expected = "exit:0"
            fallback_command = None
            if language == "python" and stem == "causal_conv1d":
                api_qualified = "cudnn.ops.causal_conv1d"
                requirements.append("causal_conv1d_bindings")
                fallback_command = ["{python}", "-m", "py_compile", "{path}"]
                verification = "syntax-checked and source-verified; installed extension lacks optional Causal Conv1d bindings"
            elif language == "python" and stem == "moe_grouped_matmul":
                requirements.append("cudnn>=91800")
                fallback_command = ["{python}", "-m", "py_compile", "{path}"]
                verification = "syntax-checked and source-verified; installed cuDNN 9.15.1 is below the current test gate"
            elif language == "python" and stem == "pytorch_custom_operator":
                api_qualified = "torch.ops.cudnn_docs.matmul"
            elif language == "python" and stem == "oss_rmsnorm_silu":
                api_qualified = "cudnn.pygraph.rmsnorm"
                requirements.append("sm100")
            elif language == "cpp" and stem == "compile_time_constants":
                api_qualified = "cudnn_frontend::graph::Graph::tensor"
                mode = "compile-run"
                requirements.append("cudnn>=92200")
                fallback_command = command.copy()
                verification = "compiled against current headers; execution requires cuDNN 9.22+ and CUDA 13.1+"
            requirements.extend(sample_override.get("requirements", []))
            fallback_command = sample_override.get("fallback_command", fallback_command)
            expected = sample_override.get("expected", expected)
            verification = sample_override.get("verification", verification)
            apis = sample_override.get("apis", [api_qualified])
            row = {
                "id": sid,
                "path": path.relative_to(DOCS_ROOT).as_posix(),
                "language": language,
                "mode": mode,
                "purpose": sample_override.get("purpose", purpose(path)),
                "owners": owners,
                "apis": apis,
                "regions": re.findall(
                    r"--8<--\s+\[start:([^]]+)\]",
                    path.read_text(encoding="utf-8", errors="replace"),
                ),
                "command": command,
                "requirements": requirements,
                "timeout_seconds": 180,
                "expected": expected,
                "verification": verification,
            }
            if fallback_command is not None:
                row["fallback_command"] = fallback_command
            rows.append(row)
    for path in sorted((SAMPLE_ROOT / "shell").glob("*.sh")):
        execute = path.stem == "run_quickstarts"
        row = {
            "id": f"shell-{path.stem.replace('_', '-')}",
            "path": path.relative_to(DOCS_ROOT).as_posix(),
            "language": "shell",
            "mode": "execute",
            "purpose": (
                "Run the published Python and C++ quick starts."
                if execute
                else "Install the released package, optional OSS dependencies, or this source checkout."
            ),
            "owners": ["getting-started/verify-environment.md" if execute else "getting-started/installation.md"],
            "apis": [],
            "regions": re.findall(
                r"--8<--\s+\[start:([^]]+)\]",
                path.read_text(encoding="utf-8", errors="replace"),
            ),
            "command": ["bash", "{path}"],
            "requirements": ["linux", "cuda", "nvcc"] if execute else ["bash", "disposable_python_environment"],
            "timeout_seconds": 300,
            "expected": "matmul ok" if execute else "exit:0",
            "verification": ("executed on CUDA 13.1/cuDNN 9.15.1/SM100" if execute else "bash syntax checked"),
        }
        if not execute:
            row["fallback_command"] = ["bash", "-n", "{path}"]
        rows.append(row)
    cmake_path = SAMPLE_ROOT / "cmake" / "CMakeLists.txt"
    if cmake_path.exists():
        rows.append(
            {
                "id": "cmake-source-integration",
                "path": cmake_path.relative_to(DOCS_ROOT).as_posix(),
                "language": "cmake",
                "mode": "configure-build",
                "purpose": "Configure and compile the C++ quick start through the exported frontend target.",
                "owners": ["getting-started/installation.md"],
                "apis": ["cudnn_frontend::graph::Graph::matmul"],
                "regions": [],
                "command": [
                    "cmake",
                    "-S",
                    "docs/samples/cmake",
                    "-B",
                    "{build_dir}",
                    "-DCUDNN_FRONTEND_SOURCE={repo}",
                ],
                "requirements": ["linux", "cuda", "nvcc", "cmake"],
                "timeout_seconds": 300,
                "expected": "exit:0",
                "verification": "configured and compiled with CMake 3.31.4 and CUDA 13.1",
            }
        )
    by_path = {f"docs/{row['path']}": row for row in rows}
    by_region = {f"/snippets/generated/{Path(row['path']).parent.name}/{Path(row['path']).stem}/{region}.mdx": row for row in rows for region in row["regions"]}
    for page in sorted(DOCS_ROOT.rglob("*.md")):
        targets = [f"docs/{target}" for target in fern_sample_references(page)]
        region_rows = [by_region[target] for target in fern_region_references(page) if target in by_region]
        owner = page.relative_to(DOCS_ROOT).as_posix()
        for row in [*(by_path[target] for target in targets if target in by_path), *region_rows]:
            if owner not in row["owners"]:
                row["owners"].append(owner)
    return {"schema_version": 1, "generated_by": "tools/doc/build_doc_manifests.py", "samples": rows}


def build_old_docs_ledger() -> dict:
    guide_map = {
        "installation": "getting-started/installation.md",
        "quickstart": "getting-started/quickstart.md",
        "quickstart_shorter": "tutorials/first-graph.md",
        "graph_lifecycle": "how-to/explicit-lifecycle.md",
        "python_graph_wrapper": "how-to/python-graph.md",
        "matmul": "how-to/matmul-layouts.md",
        "pointwise": "tutorials/pointwise-reduction.md",
        "convolutions": "tutorials/convolution.md",
        "normalizations": "tutorials/normalization.md",
        "attention": "tutorials/attention.md",
        "attention_recipes": "how-to/attention-recipes.md",
        "pytorch_integration": "tutorials/pytorch-integration.md",
        "pytorch_custom_ops": "how-to/pytorch-custom-operator.md",
        "block_scaling": "tutorials/block-scaling.md",
        "moe_grouped_matmul": "tutorials/moe-grouped-matmul.md",
        "rope": "how-to/rope.md",
        "resampling": "how-to/resampling.md",
        "causal_conv1d": "how-to/causal-convolution.md",
        "concatenate": "how-to/shape-operations.md#concatenate",
        "reshape": "how-to/shape-operations.md#reshape",
        "slice": "how-to/shape-operations.md#slice",
        "transpose": "how-to/shape-operations.md#transpose",
        "index": "index.md",
    }
    v1_map = {
        "CompileTimeConstants": "how-to/scalars-and-constants.md",
        "adding_torch_custom_ops": "how-to/pytorch-custom-operator.md",
        "cuda-graphs": "how-to/cuda-graphs.md",
        "custom-execution-plan": "how-to/engine-knobs.md#build-a-custom-plan",
        "deviceless-ahead-of-time-compilation": "how-to/deployment.md#run-the-three-stage-deviceless-flow",
        "dynamic-kernel-cache": "how-to/dynamic-shapes.md#query-runtime-workspace-before-allocation",
    }
    operation_map = {
        "Attention": "concepts/attention-support-history.md",
        "BlockScaling": "tutorials/block-scaling.md",
        "CausalConv1d": "how-to/causal-convolution.md",
        "Concatenate": "how-to/shape-operations.md#concatenate",
        "Convolutions": "tutorials/convolution.md",
        "Matmul": "how-to/matmul-layouts.md",
        "MoeGroupedMatmul": "tutorials/moe-grouped-matmul.md",
        "Normalizations": "tutorials/normalization.md",
        "Pointwise": "tutorials/pointwise-reduction.md",
        "Resampling": "how-to/resampling.md",
        "Reshape": "how-to/shape-operations.md#reshape",
        "RoPE": "how-to/rope.md",
        "Slice": "how-to/shape-operations.md#slice",
        "Transpose": "how-to/shape-operations.md#transpose",
    }
    oss_map = {
        "sdpa_bwd_d256": "reference/oss-sdpa-d256.md#sdpa_bwd_wrapper_sm100_d256",
        "sdpa_fwd_d256": "reference/oss-sdpa-d256.md#sdpa_fwd_wrapper_sm100_d256",
        "bsa": "reference/oss-block-sparse-attention.md",
        "dsa": "reference/oss-deepseek-sparse-attention.md",
        "nsa": "reference/oss-native-sparse-attention.md",
        "overview": "reference/oss-engines.md",
        "rmsnorm_rht_amax": "reference/oss-normalization.md#rmsnorm_rht_amax_wrapper_sm100",
        "rmsnorm_silu": "reference/oss-normalization.md#run-the-rmsnorm-silu-graph-fusion",
    }
    topics: list[dict] = []
    for path in sorted((REPO_ROOT / "old_docs").rglob("*.md")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if path.name == "SKILLS.md":
            topics.append(
                {
                    "source": relative,
                    "disposition": "excluded",
                    "destination": None,
                    "reason": "Internal authoring instructions, not user documentation.",
                }
            )
            continue
        parts = path.parts
        if "api" in parts and path.parent.name in {"python", "cpp"}:
            destination = f"api/{path.parent.name}/{path.name}"
            disposition = "retained"
            reason = "Per-call contract retained and reconciled against current declarations."
        elif "user_guide" in parts:
            destination = guide_map.get(path.stem, "index.md")
            disposition = "merged"
            reason = "Topic moved into the grouped learning model."
        elif "fe-oss-apis" in parts:
            if path.stem.startswith("discrete_grouped_gemm_"):
                destination = f"reference/oss-discrete-grouped-gemm.md#{path.stem}_wrapper_sm100"
            elif path.stem.startswith("grouped_gemm_"):
                destination = f"reference/oss-grouped-gemm.md#{path.stem.removesuffix('_unified')}_wrapper_sm100"
            elif path.stem.startswith("gemm_"):
                destination = f"reference/oss-dense-gemm.md#{path.stem}_wrapper_sm100"
            else:
                destination = oss_map[path.stem]
            disposition = "merged"
            if path.stem == "rmsnorm_silu":
                reason = "The former standalone name is mapped to the current Graph-pattern OSS engine and its execution constraints."
            elif path.stem == "grouped_gemm_quant_unified":
                reason = "The old unified entry point is covered by the current grouped_gemm_quant wrapper contract."
            else:
                reason = "Current API-local contract and support constraints are retained in the corresponding OSS family guide."
        elif "operations" in parts and path.stem in operation_map:
            destination = operation_map[path.stem]
            disposition = "merged"
            reason = "Operation-family guidance moved to the grouped learning model."
        elif path.stem in v1_map:
            destination = v1_map[path.stem]
            disposition = "merged"
            reason = "End-user workflow retained in the current grouped documentation."
        elif path.stem == "python_graph_and_execution_backends":
            destination = "concepts/python-ir-routing.md"
            disposition = "merged"
            reason = "Python graph routing, plan spaces, UID ownership, and backend selection moved to a dedicated concepts page."
        else:
            destination = "samples/index.md" if path.name == "samples.md" else None
            disposition = "merged" if destination else "excluded"
            reason = "Mapped to the published sample catalog." if destination else "Non-user authoring material."
        topics.append({"source": relative, "disposition": disposition, "destination": destination, "reason": reason})
    return {"schema_version": 1, "topics": topics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outputs = {
        API_ROOT / "manifest.json": build_api_manifest(),
        SAMPLE_ROOT / "manifest.json": build_sample_manifest(),
        DOCS_ROOT / "old-docs-disposition.json": build_old_docs_ledger(),
    }
    drift = [path for path, data in outputs.items() if not write_or_check(path, data, args.check)]
    if drift:
        for path in drift:
            print(f"DRIFT: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    print(f"{'checked' if args.check else 'built'} {len(outputs)} documentation manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
