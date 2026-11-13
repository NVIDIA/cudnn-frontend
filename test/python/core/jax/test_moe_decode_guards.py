# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only preflight checks for JAX MoE/decode availability."""

import importlib.abc
from pathlib import Path
import sys
from types import ModuleType

import pytest

pytest.importorskip("jax", minversion="0.9.1")

pytestmark = pytest.mark.L0
_SOURCE = Path(__file__).resolve().parents[4] / "python" / "cudnn"
_BLOCKED = ("cutlass", "cuda", "nvshmem", "cudnn._cake", "cudnn.jax", "cudnn.moe_ep._megamoe_backend")


@pytest.fixture
def source_package(monkeypatch):
    # Isolate the source modules without loading the platform's native cuDNN
    # extension. Restore any real package already loaded by another test.
    for name in tuple(sys.modules):
        if name == "cudnn" or name.startswith("cudnn.") or any(name == prefix or name.startswith(prefix + ".") for prefix in _BLOCKED):
            monkeypatch.delitem(sys.modules, name)
    for name, relative in (("cudnn", ""), ("cudnn.gemm", "gemm"), ("cudnn.gemm.frost", "gemm/frost")):
        package = ModuleType(name)
        package.__path__ = [str(_SOURCE / relative)]
        package.__file__ = str(_SOURCE / relative / "__init__.py")
        monkeypatch.setitem(sys.modules, name, package)
    sys.modules["cudnn"]._pybind_module = ModuleType("cudnn._compiled_module")

    class BlockRuntimeImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in _BLOCKED):
                raise AssertionError(f"availability guard imported {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [BlockRuntimeImports(), *sys.meta_path])
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "cudnn" or name.startswith("cudnn."):
                sys.modules.pop(name)


def test_fp8_decode_export_declines_old_dsl_before_runtime_imports(source_package, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.1")))
    from cudnn.ops import fp8_projection_decode_forward_jax

    assert callable(fp8_projection_decode_forward_jax)
    with pytest.raises(NotImplementedError, match=r"requires.*4\.9\.0.*4\.8\.1"):
        fp8_projection_decode_forward_jax(None, None, None)


@pytest.mark.parametrize("entry", ["prepare", "constructor"])
def test_ep_declines_before_runtime_imports(source_package, entry):
    from cudnn.moe_ep import MoeEpConfig, MoeEpJax, MoeEpModelConfig, MoeEpParallelConfig, prepare_moe_ep_jax

    config = MoeEpConfig(
        model=MoeEpModelConfig(num_experts=4, hidden_size=128, intermediate_size=256, top_k=2),
        parallel=MoeEpParallelConfig(max_tokens_per_rank=8),
    )
    with pytest.raises(NotImplementedError, match="temporarily unavailable.*standard CuTe DSL JAX bridge"):
        if entry == "prepare":
            prepare_moe_ep_jax(config, device=None, rank=0, world_size=2)
        else:
            MoeEpJax(None, None, None)
