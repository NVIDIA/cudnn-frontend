# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M0 probe for the bprop megakernel (BWD_DESIGN.md): validate on GPU every
pool/stash contract the backward builds on.

    C1  pool row layout + token_src_metadata: expert e's arrivals live at
        128-aligned slot doff[e], and each row's (src_token, src_topk)
        decodes to a routing entry for expert e.
    C2  l1_topk_weights_buffer[row] == topk_weights[src_token, src_topk].
    C3  fc1_c stash row == RAW pre-SwiGLU gate+up: dequant(my_activation
        [src_token]) @ fakequant(w13_interleaved[e]).T  — and is NOT
        tw-weighted.
    C4  combine_quant staging (T, K, H) persists after forward and holds the
        per-copy weighted y: fakequant(tw * silu(g) * u) @ fakequant(w2).T
        (dtw v0 source).
    C5  padding rows: report whether fc1_c / pool rows inside the 128-pad
        tails are zero (wgrad-from-stash masking decision).
    C6  pool order stability: run the same forward twice, compare metadata
        snapshots (informational — the backward gathers BY metadata, so
        instability across launches is fine; instability of fc1_c vs its
        OWN launch's metadata would not be).

Launch (single rank):   MEGA_NO_DIST=1 python -m megamoe.tests.probe_bwd_contracts
"""

import os
import sys

os.environ.setdefault("MEGA_NO_DIST", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dataclasses

import torch

import megamoe.repo_path  # noqa: F401

from megamoe.forward import MegaMoeForwardConfig, MegaMoeMxfp8Forward
from megamoe.pools import (
    decode_token_src_metadata,
    expert_slot_offsets,
    local_pool_views,
    shared_pool_views,
)
from megamoe.training import dequant_mxfp8_pool

from pt.quant import fake_quant_mxfp8

TOKENS = 4096
HIDDEN = 1024
INTERMEDIATE = 512
NUM_EXPERTS = 32
TOPK = 4
SEED = 4242

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def gen_problem(device):
    gen = torch.Generator(device=device).manual_seed(SEED)
    x = (torch.randn((TOKENS, HIDDEN), device=device, generator=gen) / 10.0).bfloat16()
    scores = torch.rand((TOKENS, NUM_EXPERTS), device=device, generator=gen)
    _, ids = scores.topk(TOPK, dim=-1)
    w = torch.rand((TOKENS, TOPK), device=device, generator=gen) + 0.1
    tw = (w / w.sum(-1, keepdim=True)).float()
    w13 = (torch.randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                       device=device, generator=gen) * 0.05).bfloat16()
    w2 = (torch.randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE),
                      device=device, generator=gen) * 0.05).bfloat16()
    return x, ids.long(), tw, w13, w2


def dequant_weight(q: torch.Tensor, sf_plain: torch.Tensor) -> torch.Tensor:
    """(N, K) fp8 + (N, K//32) E8M0 -> fp32."""
    scale = torch.exp2(sf_plain.view(torch.uint8).float() - 127.0)
    return q.float() * scale.repeat_interleave(32, dim=-1)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    x, ids, tw, w13, w2 = gen_problem(device)

    cfg = MegaMoeForwardConfig(
        max_tokens_per_rank=TOKENS, hidden=HIDDEN, intermediate=INTERMEDIATE,
        num_total_experts=NUM_EXPERTS, num_topk=TOPK,
    )
    cfg = dataclasses.replace(cfg, impl=dataclasses.replace(cfg.impl, generate_c=True))
    fwd = MegaMoeMxfp8Forward(cfg, rank=0, world_size=1)
    fwd.load_weights(w13, w2)

    out = fwd(x, ids, tw).clone()
    torch.cuda.synchronize()

    lv = local_pool_views(fwd)
    sv = shared_pool_views(fwd)
    src_rank, src_token, src_topk, _, _ = decode_token_src_metadata(
        lv["token_src_metadata"]
    )

    counts = torch.bincount(ids.reshape(-1), minlength=NUM_EXPERTS).tolist()
    doff, _ = expert_slot_offsets(counts, pad=128)

    # fake-quant weights in kernel layout (the exact bytes the kernel read)
    qw = fwd._weights
    w13_int_fq = dequant_weight(
        qw.fc1_weight.permute(0, 2, 1).reshape(-1, HIDDEN),
        qw.fc1_weight_sf_plain.reshape(-1, HIDDEN // 32),
    ).view(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2_fq = dequant_weight(
        qw.fc2_weight.permute(0, 2, 1).reshape(-1, INTERMEDIATE),
        qw.fc2_weight_sf_plain.reshape(-1, INTERMEDIATE // 32),
    ).view(NUM_EXPERTS, HIDDEN, INTERMEDIATE)

    x_deq_all = dequant_mxfp8_pool(
        fwd.my_activation[:TOKENS], fwd.my_activation_sf[:TOKENS], HIDDEN
    )  # fp32 (T, H): exactly the quantized activation the kernel multiplied

    combine_staged = sv["combine_quant"]  # (T, K, H) bf16

    c1_ok = c2_ok = True
    c3_max = c3_tw_max = c4_max = 0.0
    pad_fc1_max = pad_pool_max = 0.0
    fc1_c = fwd.fc1_c
    pool_tokens = lv["l1_token_buffer"]

    for e in range(NUM_EXPERTS):
        n = counts[e]
        if n == 0:
            continue
        rows = torch.arange(doff[e], doff[e] + n, device=device)
        st, sk = src_token[rows].long(), src_topk[rows].long()

        # C1: metadata routes every slot row to this expert
        c1_ok &= bool((ids[st, sk] == e).all()) and bool((src_rank[rows] == 0).all())

        # C2: pool topk weights match
        c2_ok &= bool(
            torch.allclose(lv["l1_topk_weights_buffer"][rows], tw[st, sk], atol=0)
        )

        # C3: fc1_c == raw gate+up of THIS row's token
        ref = (x_deq_all[st] @ w13_int_fq[e].t()).to(torch.bfloat16).float()
        got = fc1_c[rows].float()
        c3_max = max(c3_max, (got - ref).abs().max().item())
        c3_tw_max = max(
            c3_tw_max, (got - ref * tw[st, sk, None].float()).abs().max().item()
        )

        # C4: staged per-copy y == fq(tw * silu(gate) * up) @ w2
        pair = got.view(n, INTERMEDIATE // 32, 2, 32)
        gate, up = pair[:, :, 0].reshape(n, -1), pair[:, :, 1].reshape(n, -1)
        a = torch.nn.functional.silu(gate) * up * tw[st, sk, None].float()
        a_q = fake_quant_mxfp8(a.to(torch.bfloat16)).float()
        y_ref = a_q @ w2_fq[e].t()
        y_got = combine_staged[st, sk].float()
        c4_max = max(c4_max, (y_got - y_ref).abs().max().item())

        # C5: padding tail of this expert's slot
        pad_rows = torch.arange(
            doff[e] + n, doff[e] + -(-n // 128) * 128, device=device
        )
        if pad_rows.numel():
            pad_fc1_max = max(pad_fc1_max, fc1_c[pad_rows].float().abs().max().item())
            pad_pool_max = max(
                pad_pool_max,
                pool_tokens[pad_rows].view(torch.float8_e4m3fn)[:, :HIDDEN]
                .float().abs().max().item(),
            )

    check("C1 slot layout + metadata routing", c1_ok)
    check("C2 pool topk weights", c2_ok)
    check("C3 fc1_c = raw gate+up", c3_max < 0.5, f"max_abs_diff={c3_max:.3e}")
    check(
        "C3b fc1_c is pre-tw (tw-folded ref must mismatch)",
        c3_tw_max > 10 * max(c3_max, 1e-6),
        f"tw-folded diff={c3_tw_max:.3e}",
    )
    check("C4 combine staging holds per-copy weighted y", c4_max < 0.5,
          f"max_abs_diff={c4_max:.3e}")
    print(f"  [INFO] C5 padding rows: fc1_c max_abs={pad_fc1_max:.3e}  "
          f"pool max_abs={pad_pool_max:.3e}")

    # C6: order stability across an identical relaunch
    meta_snap = lv["token_src_metadata"].clone()
    out2 = fwd(x, ids, tw).clone()
    torch.cuda.synchronize()
    stable = bool(torch.equal(meta_snap, lv["token_src_metadata"]))
    print(f"  [INFO] C6 pool order stable across relaunch: {stable}")
    check("C6b relaunch output parity",
          (out - out2).abs().max().item() < 1e-2)

    # sanity: dtw from staging (the v0 recipe) vs autograd-free finite formula
    dout = torch.randn_like(out).float()
    dtw_v0 = torch.einsum("th,tkh->tk", dout, combine_staged.float()[:TOKENS]) / tw
    print(f"  [INFO] dtw v0 sample (t=0): {dtw_v0[0].tolist()}")

    fwd.finalize()
    print("PROBE " + ("PASS" if not _failures else f"FAIL ({_failures})"))
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
