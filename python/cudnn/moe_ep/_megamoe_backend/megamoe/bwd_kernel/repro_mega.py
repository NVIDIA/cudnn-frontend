# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal mega-backward repro with step prints (debugging aid)."""

import faulthandler
import os
import sys

faulthandler.enable()
os.environ.setdefault("MEGA_NO_DIST", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dataclasses

import torch

import megamoe.repo_path  # noqa: F401


def log(msg):
    print(f"[repro] {msg}", flush=True)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:29541", world_size=1, rank=0
        )

    from pt import EpConfig, QuantConfig
    from megamoe.forward import MegaMoeForwardConfig
    from megamoe.training import MegaMoeHybridMxfp8Layer

    T, H, I, E, K = 4096, 1024, 512, 32, 4
    gen = torch.Generator(device=device).manual_seed(4242)
    x = (torch.randn((T, H), device=device, generator=gen) / 10.0).bfloat16()
    scores = torch.rand((T, E), device=device, generator=gen)
    _, ids = scores.topk(K, dim=-1)
    w = torch.rand((T, K), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    dout = torch.randn((T, H), device=device, generator=gen).bfloat16()
    w13 = (torch.randn((E, 2 * I, H), device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((E, H, I), device=device, generator=gen) * 0.05).bfloat16()

    ep_cfg = EpConfig(num_experts=E, top_k=K, hidden_size=H,
                      intermediate_size=I, ep_size=1, ep_rank=0)
    mm_cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=T, hidden=H, intermediate=I,
        num_total_experts=E, num_topk=K,
    )
    mm_cfg = dataclasses.replace(
        mm_cfg, impl=dataclasses.replace(mm_cfg.impl, generate_c=True)
    )
    log("building layer")
    layer = MegaMoeHybridMxfp8Layer(
        ep_cfg, mm_cfg, w13, w2,
        qcfg=QuantConfig(fprop_fmt="mxfp8", quant_bprop=True), bwd_impl="mega",
    )
    log("forward")
    x_l = x.detach().clone().requires_grad_()
    tw_l = tw.detach().clone().requires_grad_()
    out = layer(x_l, ids, tw_l)
    torch.cuda.synchronize()
    log(f"forward done, out norm {out.float().norm().item():.4f}")
    log("backward (compiles bwd kernel on first call)")
    dx, dtw, dw13, dw2 = torch.autograd.grad(
        out, (x_l, tw_l, layer.w13, layer.w2), dout
    )
    torch.cuda.synchronize()
    log(f"backward done: |dx|={dx.float().norm():.4f} |dtw|={dtw.float().norm():.4f} "
        f"|dw13|={dw13.float().norm():.4f} |dw2|={dw2.float().norm():.4f}")
    nan = any(t.float().isnan().any().item() for t in (dx, dtw, dw13, dw2))
    log(f"nan check: {'FAIL' if nan else 'ok'}")
    layer.finalize()
    dist.destroy_process_group()
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
