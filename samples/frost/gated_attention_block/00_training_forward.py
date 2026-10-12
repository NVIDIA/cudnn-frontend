# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 00: gated attention block, TRAINING forward (``save_for_backward=True``).

One ``GatedAttentionBlockFwd`` runs the whole sub-layer -- QKV + GATE projection,
QK-RMSNorm + partial RoPE, causal GQA SDPA, sigmoid gate, out projection -- and,
declared with ``save_for_backward=True``, writes every tensor the backward needs
into a caller-owned ``SavedForBackward`` record instead of its workspace: the
projection GEMM writes ``saved.proj_slab``, the SDPA writes the PRE-gate
``saved.o`` and ``saved.lse``, norm + RoPE writes ``saved.rstd_q`` / ``rstd_k``.
``saved_slab_views`` spells the slab's four column bands (``q_pre``, ``gate``,
``k_pre``, ``v``) as zero-copy ``[B, S, heads, D]`` views.  This is the
``proj_slab`` save mode (the declaration default); ``saved_gate_copy=True``
keeps the slab in the workspace and copies the GATE band into a compact
``saved.gate`` instead.

Self-checks, torch fp32 math on the record alone: ``proj_slab`` is ``h @ W_qkvg^T``;
``rstd_q`` / ``rstd_k`` are the RMSNorm statistics of the PRE-norm ``q_pre`` /
``k_pre`` bands; ``out`` is ``(o * sigmoid(gate)) @ W_o^T``, i.e. ``saved.o`` is
the pre-gate attention output and ``gate`` the GATE band the block gated with.

Rubin (compute capability 10.7) only; prints SKIP on any other device.  The
block's two projections ride the opt-in FROST GEMM engine, so
``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`` must be in the environment before
``import cudnn`` (set below if unset).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")  # opt-in FROST engines; read at `import cudnn`

import torch  # noqa: E402

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, SavedForBackward, saved_slab_views  # noqa: E402

_RUBIN = (10, 7)


def _rope_tables(batch: int, seq_len: int, rope_dim: int, *, base: float = 1_000_000.0, device, dtype):
    """``cos, sin`` of shape ``[B, S, rope_dim]`` in the rotate-half convention (the two halves duplicated) --
    what HF ``apply_rotary_pos_emb`` and vLLM's ``RotaryEmbedding`` hand a RoPE kernel."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)  # [S, rope_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1).expand(batch, seq_len, rope_dim)
    return emb.cos().to(dtype).contiguous(), emb.sin().to(dtype).contiguous()


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float().flatten(), ref.float().flatten()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def main(batch: int = 1, seq_len: int = 512) -> None:
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    if cc != _RUBIN:
        print(f"[00] SKIP  gated attention block training forward: needs Rubin (compute capability 10.7), found {cc}")
        return
    torch.manual_seed(0)
    dev, dtype = torch.device("cuda"), torch.bfloat16
    geom = GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)  # causal, QK-norm on (the defaults)
    B, S, T = batch, seq_len, batch * seq_len

    # Inputs in the block's declared layouts: nn.Linear-shaped weights ([out, in]); W_qkvg's column blocks are Q | GATE | K | V.
    h = torch.randn(B, S, geom.d_model, device=dev).to(dtype)
    w_qkvg = (torch.randn(geom.n_qkvg, geom.d_model, device=dev) * 0.02).to(dtype)
    w_o = (torch.randn(geom.d_model, geom.h_q * geom.d_head, device=dev) * 0.02).to(dtype)
    w_q_norm = torch.ones(geom.d_head, device=dev, dtype=dtype)
    w_k_norm = torch.ones(geom.d_head, device=dev, dtype=dtype)
    cos, sin = _rope_tables(B, S, geom.rope_dim, device=dev, dtype=dtype)
    out = torch.empty(B, S, geom.d_model, device=dev, dtype=dtype)

    # Declare the TRAINING forward, compile once, size the workspace (no slab and no O in it: the record holds them).
    blk = GatedAttentionBlockFwd(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, geom, save_for_backward=True)
    try:
        blk.check_support()  # a typed decline (an engine or feature this device / build does not serve) is a SKIP, not a failure
    except NotImplementedError as exc:
        print(f"[00] SKIP  gated attention block training forward: declined by check_support(): {exc}")
        return
    blk.compile()
    workspace = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device=dev)

    # The caller-owned record.  The projection GEMM writes proj_slab; q_pre / gate / k_pre / v are its column bands.
    proj_slab = torch.empty(T, geom.n_qkvg, dtype=dtype, device=dev)  # [B*S, n_qkvg], contiguous
    q_pre, gate, k_pre, v = saved_slab_views(proj_slab, geom, B, S)
    saved = SavedForBackward(
        h=h,  # the SAME tensor execute() runs on (verified)
        gate=gate,  # views of proj_slab -- or None, and the backward derives them from the slab
        q_pre=q_pre,
        k_pre=k_pre,
        o=torch.empty(B, S, geom.h_q, geom.d_head, dtype=dtype, device=dev),  # PRE-gate attention output, compact
        lse=torch.empty(B, geom.h_q, S, dtype=torch.float32, device=dev),  # natural-log softmax stats
        rstd_q=torch.empty(B, S, geom.h_q, dtype=torch.float32, device=dev),  # RMSNorm 1/rms per head (None iff qk_norm=False)
        rstd_k=torch.empty(B, S, geom.h_kv, dtype=torch.float32, device=dev),
        proj_slab=proj_slab,
        seq_lens=None,  # dense forward; under padding, the very tensor passed to execute(seq_lens=)
    )
    blk.execute(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, saved=saved)
    torch.cuda.synchronize()

    mib = lambda t: t.numel() * t.element_size() / 2**20  # noqa: E731
    print(f"[00] out {tuple(out.shape)} {out.dtype}   workspace {mib(workspace):.1f} MiB   saved.proj_slab {tuple(proj_slab.shape)} ({mib(proj_slab):.1f} MiB)")
    for name, t in (("q_pre", q_pre), ("gate", gate), ("k_pre", k_pre), ("v", v)):
        print(f"[00]   saved_slab_views -> {name:6s} {str(tuple(t.shape)):18s} strides {tuple(t.stride())}   (a view of proj_slab)")
    for name in ("o", "lse", "rstd_q", "rstd_k"):
        t = getattr(saved, name)
        print(f"[00]   saved.{name:7s} {str(tuple(t.shape)):18s} {t.dtype}")

    # Self-checks against torch (fp32 math), on the record alone.
    e_slab = _rel_err(proj_slab, h.view(T, geom.d_model).float() @ w_qkvg.float().T)
    assert e_slab < 1e-2, f"proj_slab vs h @ W_qkvg^T: rel err {e_slab:.3g}"
    e_rq = _rel_err(saved.rstd_q, torch.rsqrt(q_pre.float().pow(2).mean(-1) + geom.qk_norm_eps))  # rstd of the PRE-norm band
    e_rk = _rel_err(saved.rstd_k, torch.rsqrt(k_pre.float().pow(2).mean(-1) + geom.qk_norm_eps))
    assert max(e_rq, e_rk) < 1e-2, f"rstd vs rsqrt(mean(pre^2) + eps): rel err {e_rq:.3g} / {e_rk:.3g}"
    assert torch.isfinite(saved.lse).all() and torch.isfinite(saved.o.float()).all()
    o_gated = saved.o.float() * torch.sigmoid(gate.float())  # saved.o is PRE-gate; the gated O lands in the workspace
    e_out = _rel_err(out, o_gated.reshape(T, geom.h_q * geom.d_head) @ w_o.float().T)
    assert e_out < 2e-2, f"out vs (o * sigmoid(gate)) @ W_o^T: rel err {e_out:.3g}"
    print(
        f"[00] PASS  training forward  B={B} S={S} h_q={geom.h_q} h_kv={geom.h_kv} d_head={geom.d_head} d_model={geom.d_model}  "
        f"(slab rel {e_slab:.1e}, rstd rel {max(e_rq, e_rk):.1e}, out rel {e_out:.1e})"
    )


if __name__ == "__main__":
    main()
