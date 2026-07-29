# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Parity driver: cudnn.MoeEp on the MegaMoE backend vs MoeEpReference.

Runs at kernel-supported shapes (H, I multiples of 128) on SM100 with the
megamoe training repo importable (CUDNN_MEGAMOE_ROOT).  Not a pytest module:
the first call compiles the CuTe DSL kernels (minutes).

Single rank:
    CUDNN_MOE_EP_BACKEND=megamoe MEGA_NO_DIST=1 python megamoe_backend_parity.py
Multi rank (EP):
    CUDNN_MOE_EP_BACKEND=megamoe torchrun --nproc_per_node=4 --standalone \
        megamoe_backend_parity.py

The backend computes in MXFP8 (device-quantized activations/weights), so
outputs are compared against the FP32 reference at MXFP8-level tolerance.
route_metadata must match exactly.
"""

import importlib.util
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLONE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

# Import the moe_ep package straight from source files so this driver does
# not need the compiled cudnn pybind module.
_spec = importlib.util.spec_from_file_location(
    "cudnn_moe_ep",
    os.path.join(_CLONE, "python", "cudnn", "moe_ep", "__init__.py"),
    submodule_search_locations=[os.path.join(_CLONE, "python", "cudnn", "moe_ep")],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["cudnn_moe_ep"] = _pkg
_spec.loader.exec_module(_pkg)
MoeEp = _pkg.MoeEp

sys.path.insert(0, os.path.join(_CLONE, "test", "python"))
from fe_api.moe_ep.moe_ep_reference import MoeEpReference  # noqa: E402

TOKENS = 128
HIDDEN = 1024
INTERMEDIATE = 512
EXPERTS = 8
TOPK = 4
SEED = 1234


def rel_err(a, b):
    denom = b.float().norm().item()
    return (a.float() - b.float()).norm().item() / max(denom, 1e-30)


def as_float(output):
    if isinstance(output, torch.Tensor):
        return output.float()
    return output.dequantize()


def main():
    multi = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if multi:
        sys.path.insert(0, os.environ["CUDNN_MEGAMOE_ROOT"])
        import megamoe.repo_path  # noqa: F401
        from src.bootstrap import init_dist_and_nvshmem

        _, rank, world, _ = init_dist_and_nvshmem()
        ep_group = dist.group.WORLD
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        os.environ.setdefault("MEGA_NO_DIST", "1")
        rank, world, ep_group = 0, 1, None
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)

    gen = torch.Generator(device=device).manual_seed(SEED + rank)
    x = (torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10).bfloat16()
    # PR layouts: fc1 (E_local, H, 2I) gate-first, fc2 (E_local, I, H)
    e_local = EXPERTS // world
    fc1_w = (torch.randn((e_local, HIDDEN, 2 * INTERMEDIATE), device=device, generator=gen) * 0.05).bfloat16()
    fc2_w = (torch.randn((e_local, INTERMEDIATE, HIDDEN), device=device, generator=gen) * 0.05).bfloat16()
    scores = torch.rand((TOKENS, EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    ids = ids.long()
    ids[0, TOPK - 1] = -1  # exercise a dropped route
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    tw[0, TOPK - 1] = 0.0
    gout = torch.randn((TOKENS, HIDDEN), device=device, generator=gen)

    kwargs = dict(
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        ep_group=ep_group,
        max_tokens_per_rank=TOKENS,
        generate_c=True,
    )
    api = MoeEp(**kwargs)
    ref = MoeEpReference(**kwargs)
    args = (x, fc1_w, fc2_w, ids, tw)

    print(f"[rank {rank}] forward (compiles kernels on first call)...", flush=True)
    out, fc1_c, meta = api(*args)
    ref_out, ref_fc1_c, ref_meta = ref(*args)

    fwd_err = rel_err(as_float(out), as_float(ref_out))
    meta_ok = torch.equal(meta, ref_meta)
    fc1_err = rel_err(fc1_c, ref_fc1_c) if fc1_c.shape == ref_fc1_c.shape else float("inf")
    print(
        f"[rank {rank}] fwd rel_err={fwd_err:.3e}  "
        f"fc1_c shape={tuple(fc1_c.shape)} vs ref {tuple(ref_fc1_c.shape)} "
        f"rel_err={fc1_err:.3e}  metadata equal={meta_ok}",
        flush=True,
    )

    print(f"[rank {rank}] backward (compiles bwd kernel on first call)...", flush=True)
    dx, dw1, dw2, dtw = api.backward(gout, *args, fc1_c, meta)
    rdx, rdw1, rdw2, rdtw = ref.backward(gout, *args, ref_fc1_c, ref_meta)

    errs = {
        "grad_activation": rel_err(dx, rdx),
        "grad_fc1_weight": rel_err(dw1, rdw1),
        "grad_fc2_weight": rel_err(dw2, rdw2),
        "grad_topk_weights": rel_err(dtw, rdtw),
    }
    dtw_drop_zero = bool((dtw[0, TOPK - 1] == 0).item())
    print(
        f"[rank {rank}] bwd rel_err: " + "  ".join(f"{k}={v:.3e}" for k, v in errs.items()) + f"  dtw[-1 slot]==0: {dtw_drop_zero}",
        flush=True,
    )

    # host-side output quantizer must match the reference bit for bit
    from fe_api.moe_ep.moe_ep_reference import quantize_blockwise

    _megamoe = sys.modules["cudnn_moe_ep._megamoe"]
    bf16_out = as_float(out).bfloat16()
    got_q = _megamoe._quantize_output(bf16_out, _pkg.MoeFormat.MXFP8).dequantize()
    want_q = quantize_blockwise(bf16_out, "mxfp8", axis=1).dequantize()
    quant_exact = torch.equal(got_q, want_q)
    print(f"[rank {rank}] mxfp8 output quantizer bit-exact vs reference: {quant_exact}", flush=True)

    tol = 0.10
    failures = [k for k, v in {**errs, "forward": fwd_err}.items() if v > tol]
    ok = not failures and meta_ok and dtw_drop_zero and quant_exact
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL ' + str(failures)}", flush=True)
    if multi:
        dist.barrier()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
