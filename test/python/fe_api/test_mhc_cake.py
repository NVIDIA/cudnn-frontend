# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from cudnn.mhc import api

pytestmark = pytest.mark.L0


def test_vendor_files():
    root = Path(api.__file__).resolve().parents[1] / "_cake"
    for package, record in json.loads((root / "VENDORED_MHC.json").read_text())["packages"].items():
        for name, expected in record["files"].items():
            data = (root / "_vendor" / package / name).read_bytes()
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_cpu_rejected_before_load(monkeypatch):
    monkeypatch.setattr(api, "_backend", lambda *_: pytest.fail("CPU input reached native loader"))
    with pytest.raises(ValueError, match="CUDA"):
        api.mhc_head_forward(torch.empty((3, 4, 4096)))


@pytest.mark.parametrize("value", [0, 65, True, 1.5])
def test_invalid_sinkhorn_count(value):
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        api._constants(1e-5, 1e-6, value)


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from cudnn._cake import loader

    try:
        loader.check_toolkit(loader.device_arch(), "mHC tests")
    except (RuntimeError, NotImplementedError) as e:
        pytest.skip(str(e))
    return torch.device("cuda", torch.cuda.current_device())


def _data(device, h):
    gen = torch.Generator().manual_seed(2032)

    def rand(shape):
        return torch.randn(shape, generator=gen).to(device)

    return dict(
        x=rand((3, 4, h)).bfloat16(),
        fn=rand((24, 4 * h)) * 0.01,
        scale=1 + rand((3,)) * 0.1,
        base=rand((24,)) * 0.1,
        branch=rand((3, h)).bfloat16(),
        dy=rand((3, 4, h)).bfloat16(),
        du=rand((3, h)).bfloat16(),
    )


def _reference(a, iters):
    x, fn, s, b, branch = [a[key].cpu().double().requires_grad_() for key in ("x", "fn", "scale", "base", "branch")]
    flat = x.flatten(1)
    r = (flat.square().mean(1, keepdim=True) + 1e-5).rsqrt()
    z = (flat * r) @ fn.T
    pre = (z[:, :4] * s[0] + b[:4]).sigmoid() + 1e-6
    post = 2 * (z[:, 4:8] * s[1] + b[4:8]).sigmoid()
    c = (z[:, 8:] * s[2] + b[8:]).view(-1, 4, 4).softmax(-1) + 1e-6
    c = c / (c.sum(1, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        c = c / (c.sum(2, keepdim=True) + 1e-6)
        c = c / (c.sum(1, keepdim=True) + 1e-6)
    u = (pre[:, :, None] * x).sum(1)
    y = post[:, :, None] * branch[:, None, :] + torch.einsum("tij,tid->tjd", c, x)
    dx, dfn, ds, db, dbranch = torch.autograd.grad((u, y), (x, fn, s, b, branch), (a["du"].cpu().double(), a["dy"].cpu().double()))
    return dict(u=u.detach(), y=y.detach(), dx=dx, dfn=dfn, dscale=ds, dbase=db, grad_branch=dbranch)


@pytest.mark.parametrize("hidden,iters", [(4096, 20), (5120, 7), (7168, 20)])
def test_full_connection_reference_capture(gpu, hidden, iters):
    from cudnn._cake import loader

    if loader.device_arch() == "sm_107a" and hidden != 4096:
        pytest.skip("SM107 serves H4096")
    a = _data(gpu, hidden)
    x = a["x"]
    args = (x, a["fn"], a["scale"], a["base"])
    u, stats = api.mhc_prepare(*args, sinkhorn_iters=iters)
    y = api.mhc_finish(x, a["branch"], stats)["out"]
    db, gs = api.mhc_backward_prepare(x, a["branch"], stats, a["dy"])
    backargs = (x, a["dy"], a["du"], a["fn"], a["scale"], a["base"], stats, gs)
    grads = api.mhc_backward_finish(*backargs, sinkhorn_iters=iters)
    fw = torch.empty(api.mhc_workspace_size(3, hidden, device=gpu), dtype=torch.uint8, device=gpu)
    bw = torch.empty(api.mhc_workspace_size(3, hidden, device=gpu, backward=True), dtype=torch.uint8, device=gpu)

    def run(stream=None):
        api.mhc_prepare(*args, sinkhorn_iters=iters, out=u, stats=stats, workspace=fw, stream=stream)
        api.mhc_finish(x, a["branch"], stats, out=y, stream=stream)
        api.mhc_backward_prepare(x, a["branch"], stats, a["dy"], grad_branch=db, grad_stats=gs, stream=stream)
        api.mhc_backward_finish(*backargs, sinkhorn_iters=iters, workspace=bw, stream=stream, **grads)

    def check():
        ref = _reference(a, iters)
        for key, value in dict(u=u, y=y, grad_branch=db, **grads).items():
            got = value.cpu().double()
            want = ref[key]
            if value.dtype == torch.bfloat16:
                torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
            else:
                assert float((got - want).norm() / want.norm()) < 2e-4, key
                assert float(((got - want).abs() / torch.maximum(want.abs(), want.square().mean().sqrt())).max()) < 1e-3, key

    check()
    side = torch.cuda.Stream(device=gpu)
    side.wait_stream(torch.cuda.current_stream(gpu))
    run(side)
    torch.cuda.current_stream(gpu).wait_stream(side)
    check()
    torch.cuda.synchronize(gpu)
    count = torch.cuda.memory_stats(gpu)["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats(gpu)["allocation.all.allocated"] == count
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.synchronize(gpu)
    check()
    first = [v.clone() for v in grads.values()]
    run()
    assert all(torch.equal(a, b) for a, b in zip(first, grads.values()))
    with pytest.raises(ValueError, match="alias"):
        api.mhc_finish(x, a["branch"], stats, out=x)
    with pytest.raises(ValueError, match="workspace"):
        api.mhc_prepare(*args, workspace=fw[:1])


def test_head_native_replay(gpu):
    x = _data(gpu, 4096)["x"]
    dy = x[:, 0, :].contiguous()
    out = api.mhc_head_forward(x)["out"]
    dx = api.mhc_head_backward(dy)["out"]

    def run():
        api.mhc_head_forward(x, out=out)
        api.mhc_head_backward(dy, out=dx)

    def check():
        torch.testing.assert_close(out, x.float().mean(1).bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(dx, (dy.float() * 0.25).bfloat16()[:, None, :].expand_as(x), atol=0, rtol=0)

    run()
    check()
    torch.cuda.synchronize(gpu)
    count = torch.cuda.memory_stats(gpu)["allocation.all.allocated"]
    run()
    assert torch.cuda.memory_stats(gpu)["allocation.all.allocated"] == count
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.neg_()
    dy.neg_()
    graph.replay()
    check()
