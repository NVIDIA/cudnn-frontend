"""Unit tests for the norm cuDNN-graph engine layer (facts + probe logic).

These exercise the pure, frontend-independent logic — ``graph_analyzer.analyze``
(node -> :class:`NormGraphFacts`), the per-engine ``mismatch`` accept/reject, and
engine registration — using synthetic cuDNN norm nodes that mirror the real
``cudnn._pygraph`` schema (node types ``LAYERNORM``/``LAYERNORM_BWD``/...; ports
``input``/``scale``/``bias``/``grad``; outputs ``Y``/``mean``/``inv_var``/``DX``/
``DScale``/``DBias``; backward takes ``mean``/``inv_variance`` as inputs).

End-to-end graph execution (``select_engines`` -> ``execute``) requires a built
cuDNN frontend and a real graph, and is not exercised here.

    python cudnn/norm/tests/test_engines.py
"""

import dataclasses
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDNN_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Stub cudnn with the norm-node attribute surface the analyzer imports.
if "cudnn" not in sys.modules or not hasattr(sys.modules["cudnn"], "data_type"):
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_CUDNN_DIR]
    stub.pygraph = type("pygraph", (), {})
    stub.data_type = types.SimpleNamespace(HALF="HALF", BFLOAT16="BFLOAT16", FLOAT="FLOAT")
    sys.modules["cudnn"] = stub

import torch  # noqa: E402

from cudnn.norm import graph_analyzer as ga  # noqa: E402
from cudnn.norm.config_sm100 import NormVariant as NV  # noqa: E402
from cudnn.norm.fprop import engines as fe  # noqa: E402
from cudnn.norm.bprop import engines as be  # noqa: E402

_DT = sys.modules["cudnn"].data_type


class _T:
    def __init__(self, dim):
        self._d = tuple(dim)

    def get_dim(self):
        return self._d

    def get_stride(self):
        s = [1] * len(self._d)
        for i in range(len(self._d) - 2, -1, -1):
            s[i] = s[i + 1] * self._d[i + 1]
        return tuple(s)

    def get_data_type(self):
        return _DT.HALF

    def get_name(self):
        return ""

    def get_uid(self):
        return 0


class _Node:
    def __init__(self, ntype, params, inputs, outputs):
        self.node_type = types.SimpleNamespace(name=ntype)
        self.params = params
        self.inputs = inputs
        self.outputs = outputs


class _Graph:
    def __init__(self, node):
        self.nodes = [node]


def _facts(node):
    return dataclasses.replace(ga.analyze(_Graph(node)), device_cc=(10, 0))


def main():
    ok = True

    def expect(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    D = 256
    # LayerNorm forward (real cuDNN spelling)
    ln = _Node("LAYERNORM", {"epsilon": 1e-5},
               {"input": _T([8, D]), "scale": _T([D]), "bias": _T([D])},
               {"Y": _T([8, D]), "mean": _T([8]), "inv_var": _T([8])})
    f = _facts(ln)
    expect(f.variant == NV.LAYER_NORM and f.phase == "fprop", "LAYERNORM -> LN fprop")
    expect(f.normalized_shape == (D,) and f.has_beta and f.has_mean, "LN normalized_shape/flags")
    # One fprop engine + one bprop engine, each serving all variants.
    fcap = fe.ENGINE_SPECS[0].capabilities
    bcap = be.ENGINE_SPECS[0].capabilities
    expect(fe.mismatch(fcap, f) is None, "fprop engine accepts LN")

    # RMSNorm forward (no bias, no mean output)
    rms = _Node("RMSNORM", {"epsilon": 1e-5},
                {"input": _T([8, D]), "scale": _T([D])},
                {"Y": _T([8, D]), "inv_var": _T([8])})
    fr = _facts(rms)
    expect(fr.variant == NV.RMS_NORM and not fr.has_mean and not fr.has_beta, "RMSNORM flags")
    expect(fe.mismatch(fcap, fr) is None, "fprop engine accepts RMS (same engine)")

    # BatchNorm forward with running stats
    bn = _Node("BATCHNORM", {"epsilon": 1e-5, "momentum": 0.1},
               {"input": _T([16, 8, 4]), "scale": _T([8]), "bias": _T([8]),
                "in_running_mean": _T([8]), "in_running_var": _T([8])},
               {"Y": _T([16, 8, 4]), "mean": _T([8]), "inv_var": _T([8]),
                "next_running_mean": _T([8]), "next_running_var": _T([8])})
    fbn = _facts(bn)
    expect(fbn.variant == NV.BATCH_NORM and fbn.wants_running_stats and fbn.training, "BATCHNORM running-stats")
    expect(fe.mismatch(fcap, fbn) is None, "fprop engine accepts BatchNorm (same engine)")

    # BatchNorm inference -> training False
    bni = _Node("BATCHNORM_INFERENCE", {},
                {"input": _T([16, 8, 4]), "mean": _T([8]), "inv_variance": _T([8]),
                 "scale": _T([8]), "bias": _T([8])},
                {"Y": _T([16, 8, 4])})
    expect(not _facts(bni).training, "BATCHNORM_INFERENCE -> training=False")

    # LayerNorm backward (mean/inv_variance are inputs; grad = DY)
    lnb = _Node("LAYERNORM_BWD", {},
                {"grad": _T([8, D]), "input": _T([8, D]), "scale": _T([D]),
                 "mean": _T([8]), "inv_variance": _T([8])},
                {"DX": _T([8, D]), "DScale": _T([D]), "DBias": _T([D])})
    fb = _facts(lnb)
    expect(fb.phase == "bprop" and fb.variant == NV.LAYER_NORM, "LAYERNORM_BWD -> LN bprop")
    expect(fb.dy_t is not None and fb.mean_t is not None and fb.dscale_t is not None, "LN bwd ports resolved")
    expect(be.mismatch(bcap, fb) is None, "bprop engine accepts LN bwd")
    expect(fe.mismatch(fcap, fb) is not None, "fprop engine rejects bprop graph")

    # Non-norm graph -> analyze returns None
    other = _Node("MATMUL", {}, {"a": _T([4, 4])}, {"c": _T([4, 4])})
    expect(ga.analyze(_Graph(other)) is None, "non-norm node -> analyze None")

    # Registration: exactly two norm engines (one per phase).
    from cudnn.frost.dispatch import _ENGINES
    names = set(_ENGINES)
    norm_engines = {n for n in names if n.startswith("norm_")}
    expect(norm_engines == {"norm_fprop_sm100", "norm_bprop_sm100"}, f"2 norm engines: {sorted(norm_engines)}")
    expect(fe.mismatch(fcap, f, requested=object()) is not None, "wrong knob vocabulary rejected")

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
