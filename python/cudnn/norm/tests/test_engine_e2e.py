"""End-to-end drive of the norm cuDNN-graph engines: probe -> build -> lower ->
execute, with synthetic cuDNN nodes + real torch buffers, validated vs PyTorch.

This exercises the whole engine->kernel path — variant dispatch through the
single fprop / bprop engine, buffer resolution (``resolve_variant_pack``), the
kernel launch, and the copy-into-output-buffers step — everything except the C++
``cudnn.pygraph`` lifecycle (which needs a built cuDNN frontend). Complements
``test_engines.py`` (which covers facts/probe logic in isolation).

    python cudnn/norm/tests/test_engine_e2e.py
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUDNN_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if "cudnn" not in sys.modules or not hasattr(sys.modules["cudnn"], "data_type"):
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_CUDNN_DIR]
    stub.pygraph = type("pygraph", (), {})
    stub.data_type = types.SimpleNamespace(HALF="HALF", BFLOAT16="BFLOAT16", FLOAT="FLOAT")
    sys.modules["cudnn"] = stub

import torch
import torch.nn.functional as F

from cudnn.norm.fprop import engines as fe
from cudnn.norm.bprop import engines as be

_DT = sys.modules["cudnn"].data_type
_TDT = {torch.float16: _DT.HALF, torch.bfloat16: _DT.BFLOAT16, torch.float32: _DT.FLOAT}

_OK = True


def _expect(c, m):
    global _OK
    _OK &= bool(c)
    print(f"  [{'PASS' if c else 'FAIL'}] {m}")


class _T:
    """Fake IR tensor wrapping a real torch buffer (the variant pack keys on it)."""

    def __init__(self, buf):
        self.buf = buf

    def get_dim(self):
        return tuple(self.buf.shape)

    def get_stride(self):
        return tuple(self.buf.stride())

    def get_data_type(self):
        return _TDT[self.buf.dtype]

    def get_name(self):
        return ""

    def get_uid(self):
        return 0


class _Node:
    def __init__(self, nt, params, inputs, outputs):
        self.node_type = types.SimpleNamespace(name=nt)
        self.params = params
        self.inputs = inputs
        self.outputs = outputs


class _Graph:
    def __init__(self, node):
        self.nodes = [node]


def _mk(shape, dt):
    return _T(torch.randn(*shape, device="cuda", dtype=dt))


def _mkf(n):
    return _T(torch.empty(n, device="cuda", dtype=torch.float32))


def _tol(dt, base):
    return base if dt != torch.float32 else base * 5e-3


def main():
    assert torch.cuda.is_available(), "CUDA required"
    fspec = fe.ENGINE_SPECS[0]
    bspec = be.ENGINE_SPECS[0]

    for dt in (torch.float16, torch.float32):
        print(f"\n=== dtype={dt} ===")
        torch.manual_seed(0)
        D, N = 256, 8

        # --- LayerNorm forward through the fprop engine ---
        xt = _mk((N, D), dt); gt = _mk((D,), dt); bt = _mk((D,), dt)
        yt = _T(torch.empty(N, D, device="cuda", dtype=dt)); mt = _mkf(N); ivt = _mkf(N)
        node = _Node("LAYERNORM", {"epsilon": 1e-5},
                     {"input": xt, "scale": gt, "bias": bt},
                     {"Y": yt, "mean": mt, "inv_var": ivt})
        g = _Graph(node)
        _expect(fe.probe(fspec, g), "LAYERNORM: fprop probe True")
        fe.build(fspec, g)({xt: xt.buf, gt: gt.buf, bt: bt.buf, yt: yt.buf, mt: mt.buf, ivt: ivt.buf})
        ref = F.layer_norm(xt.buf.float(), (D,), gt.buf.float(), bt.buf.float(), 1e-5)
        _expect((yt.buf.float() - ref).abs().max().item() <= _tol(dt, 6e-2), "LAYERNORM: Y matches torch")

        # --- LayerNorm backward through the bprop engine (reuse mean/inv_var) ---
        dyt = _T(torch.randn(N, D, device="cuda", dtype=dt))
        dxt = _T(torch.empty(N, D, device="cuda", dtype=dt)); dst = _mkf(D); dbt = _mkf(D)
        mtin = _T(mt.buf); ivtin = _T(ivt.buf)
        bnode = _Node("LAYERNORM_BWD", {},
                      {"grad": dyt, "input": xt, "scale": gt, "mean": mtin, "inv_variance": ivtin},
                      {"DX": dxt, "DScale": dst, "DBias": dbt})
        bg = _Graph(bnode)
        _expect(be.probe(bspec, bg), "LAYERNORM_BWD: bprop probe True")
        _expect(fe.probe(fspec, bg) is False, "LAYERNORM_BWD: fprop probe False")
        be.build(bspec, bg)({dyt: dyt.buf, xt: xt.buf, gt: gt.buf, mtin: mtin.buf,
                             ivtin: ivtin.buf, dxt: dxt.buf, dst: dst.buf, dbt: dbt.buf})
        xr = xt.buf.float().detach().requires_grad_(True)
        gr = gt.buf.float().detach().requires_grad_(True)
        br = bt.buf.float().detach().requires_grad_(True)
        yref = F.layer_norm(xr, (D,), gr, br, 1e-5)
        dxr, dgr, dbr = torch.autograd.grad(yref, [xr, gr, br], grad_outputs=dyt.buf.float())
        tol = _tol(dt, 4e-2)
        _expect((dxt.buf.float() - dxr).abs().max().item() <= tol * max(1, dxr.abs().max().item()), "LAYERNORM_BWD: DX matches")
        _expect((dst.buf - dgr).abs().max().item() <= tol * max(1, dgr.abs().max().item()), "LAYERNORM_BWD: DScale matches")

        # --- GroupNorm + BatchNorm forward through the SAME fprop engine (variant dispatch) ---
        torch.manual_seed(1)
        xg = _mk((4, 8, 32), dt); gg = _mk((8,), dt); bg2 = _mk((8,), dt)
        yg = _T(torch.empty(4, 8, 32, device="cuda", dtype=dt)); mg = _mkf(16); ig = _mkf(16)
        gnode = _Node("GROUPNORM", {"epsilon": 1e-5, "num_groups": 4},
                      {"input": xg, "scale": gg, "bias": bg2}, {"Y": yg, "mean": mg, "inv_var": ig})
        gG = _Graph(gnode)
        _expect(fe.probe(fspec, gG), "GROUPNORM: fprop probe True (same engine)")
        fe.build(fspec, gG)({xg: xg.buf, gg: gg.buf, bg2: bg2.buf, yg: yg.buf, mg: mg.buf, ig: ig.buf})
        gref = F.group_norm(xg.buf.float(), 4, gg.buf.float(), bg2.buf.float(), 1e-5)
        _expect((yg.buf.float() - gref).abs().max().item() <= _tol(dt, 6e-2), "GROUPNORM: Y matches torch")

        torch.manual_seed(2)
        xb = _mk((16, 8, 4), dt); gbt = _mk((8,), dt); bbt = _mk((8,), dt)
        yb = _T(torch.empty(16, 8, 4, device="cuda", dtype=dt)); mb = _mkf(8); ib = _mkf(8)
        bnnode = _Node("BATCHNORM", {"epsilon": 1e-5, "momentum": 0.1},
                       {"input": xb, "scale": gbt, "bias": bbt}, {"Y": yb, "mean": mb, "inv_var": ib})
        bnG = _Graph(bnnode)
        _expect(fe.probe(fspec, bnG), "BATCHNORM: fprop probe True (same engine)")
        fe.build(fspec, bnG)({xb: xb.buf, gbt: gbt.buf, bbt: bbt.buf, yb: yb.buf, mb: mb.buf, ib: ib.buf})
        bref = F.batch_norm(xb.buf.float(), None, None, gbt.buf.float(), bbt.buf.float(), True, 0.1, 1e-5)
        _expect((yb.buf.float() - bref).abs().max().item() <= _tol(dt, 6e-2), "BATCHNORM: Y matches torch")

    print("\n" + ("ALL PASS" if _OK else "SOME FAILED"))
    sys.exit(0 if _OK else 1)


if __name__ == "__main__":
    main()
