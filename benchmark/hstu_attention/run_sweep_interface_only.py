# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the qlen=1 sweep without requiring a built cuDNN Python extension.

Set ``HSTU_SOURCE_ROOT`` to benchmark another source tree (for example the
pre-optimization baseline) while keeping this sweep driver unchanged.
``HSTU_BENCHMARK_DRIVER`` may select either the broad sweep or the focused
correctness/target benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import types

driver_root = Path(__file__).resolve().parents[2]
source_root = Path(os.environ.get("HSTU_SOURCE_ROOT", driver_root)).resolve()

cudnn_package = types.ModuleType("cudnn")
cudnn_package.__path__ = [str(source_root / "python" / "cudnn")]
cudnn_package.__file__ = str(source_root / "python" / "cudnn" / "__init__.py")
sys.modules["cudnn"] = cudnn_package

# Loading _interface does not require the compiled FE extension.  Bypass the
# public package __init__, which intentionally imports that optional extension.
hstu_package = types.ModuleType("cudnn.hstu_attention")
hstu_package.__path__ = [str(source_root / "python" / "cudnn" / "hstu_attention")]
sys.modules["cudnn.hstu_attention"] = hstu_package


class HSTUBwdSm100:
    """Minimal explicit-API adapter used only by this standalone benchmark."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def check_support(self):
        return True

    def _call(self, do, q, k, v, dq, dk, dv, cu_q, cu_k, *, compile_only):
        from cudnn.hstu_attention import _interface

        return _interface.hstu_varlen_bwd_100(
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            self.kwargs["max_seqlen_q"],
            self.kwargs["max_seqlen_k"],
            dq,
            dk,
            dv,
            self.kwargs["window_size"][0],
            self.kwargs["window_size"][1],
            self.kwargs["alpha"],
            None,
            False,
            self.kwargs["scaling_seqlen"],
            _compile_only=compile_only,
        )

    def compile(self):
        return self._call(
            self.kwargs["sample_do"],
            self.kwargs["sample_q"],
            self.kwargs["sample_k"],
            self.kwargs["sample_v"],
            self.kwargs["sample_dq"],
            self.kwargs["sample_dk"],
            self.kwargs["sample_dv"],
            self.kwargs["sample_cu_seqlens_q"],
            self.kwargs["sample_cu_seqlens_k"],
            compile_only=True,
        )

    def execute(self, do, q, k, v, dq, dk, dv, cu_q, cu_k):
        return self._call(do, q, k, v, dq, dk, dv, cu_q, cu_k, compile_only=False)


class HSTUFwdSm100:
    """Minimal explicit-API adapter used only by this standalone benchmark."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def check_support(self):
        return True

    def _call(self, q, k, v, out, cu_q, cu_k, *, compile_only):
        from cudnn.hstu_attention import _interface

        return _interface.hstu_varlen_fwd_100(
            q,
            k,
            v,
            cu_q,
            cu_k,
            self.kwargs["max_seqlen_q"],
            self.kwargs["max_seqlen_k"],
            self.kwargs["window_size"][0],
            self.kwargs["window_size"][1],
            self.kwargs["alpha"],
            None,
            scaling_seqlen=self.kwargs["scaling_seqlen"],
            out=out,
            _compile_only=compile_only,
        )

    def compile(self):
        return self._call(
            self.kwargs["sample_q"],
            self.kwargs["sample_k"],
            self.kwargs["sample_v"],
            self.kwargs["sample_o"],
            self.kwargs["sample_cu_seqlens_q"],
            self.kwargs["sample_cu_seqlens_k"],
            compile_only=True,
        )

    def execute(self, q, k, v, out, cu_q, cu_k):
        return self._call(q, k, v, out, cu_q, cu_k, compile_only=False)


cudnn_package.HSTUBwdSm100 = HSTUBwdSm100
cudnn_package.HSTUFwdSm100 = HSTUFwdSm100
benchmark_root = driver_root / "benchmark" / "hstu_attention"
driver_name = os.environ.get("HSTU_BENCHMARK_DRIVER", "sweep_hstu_qlen1.py")
if driver_name not in ("benchmark_hstu_qlen1.py", "sweep_hstu_qlen1.py", "profile_hstu_qlen1.py"):
    raise ValueError(f"Unsupported HSTU benchmark driver: {driver_name}")
sys.path.insert(0, str(benchmark_root))
runpy.run_path(str(benchmark_root / driver_name), run_name="__main__")
