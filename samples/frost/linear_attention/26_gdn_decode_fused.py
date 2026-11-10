# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 26: the fused GDN (Gated DeltaNet) decode step, driven from a slot pool.

``cudnn.GdnDecodeFused`` is one launch per decode step of a GDN sub-layer.  The model's in-projection has
produced the step's ``x`` (the Q | K | V channels) and ``z`` (the output gate); the launch then runs the
``b`` / ``a`` gate projection of the normed hidden, the causal conv + SiLU over the sequence's conv history,
the gated delta rule on the sequence's recurrent state, the gated RMSNorm and the static 8-bit (e4m3) output
quantization::

    q_t, k_t, v_t = silu(conv(x_{t-3..t}))      q_t and k_t scaled to unit length, q_t also by 128 ** -0.5
    beta_t  = sigmoid(b_t)                      decay_t = exp(-exp(A_log) * softplus(a_t + dt_bias))
    S_t     = decay_t S_{t-1} + beta_t (v_t - decay_t S_{t-1} k_t) k_t^T
    o_t     = S_t q_t
    out_t   = e4m3(rmsnorm(o_t) * w * silu(z_t) / out_scale)

The recurrent state and the conv history live in a caller-owned slot pool (``state`` [slots, V, 128, 128]
fp32, ``conv_state`` [slots, C, 3] bf16).  A step names one slot per row and advances those slots in place;
slot 0 is the null slot (a row that names it is padding).

The script draws seeded weights and 32 steps of inputs for 4 sequences, runs the steps at 1 row (sequence 0
alone) and at 4 rows, and checks:

* the 8-bit output and the state against a plain PyTorch step that keeps every bf16 / fp32 cast of the fused
  step and accumulates in fp64, chained over all steps on its own pool;
* a replay from the pool's snapshot, byte for byte;
* sequence 0 alone against sequence 0 among 4, byte for byte, and that a step touches only the slots it names.

It prints the compile time of each row count and the memory the run took.  The entry serves compute capability
10.x and 12.x GPUs with nvidia-cutlass-dsl >= 4.8.0; elsewhere it declines by type and the script prints that
sentence after SKIP.  The entry's contract (operand layouts, forms, declines) is described in
docs/operations/linear_attention/gdn_decode_flashinfer_backend.md.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import cudnn
import torch

D = 128  # the head dim of the fused step
TAPS = 3  # conv width 4: three history taps and the step's own token
_BITS = {torch.float8_e4m3fn: torch.uint8, torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _draw(steps, seqs, K, V, hidden, device, seed):
    """Seeded weights, a slot pool and ``steps`` inputs for ``seqs`` sequences, drawn on the CPU generator so the values do not depend on the GPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = (2 * K + V) * D  # the Q | K | V channels the conv runs over

    def rn(*shape, std=1.0):
        return (torch.randn(*shape, generator=g) * std).to(device)

    weights = SimpleNamespace(
        conv_weight=rn(C, TAPS + 1, std=0.3).bfloat16(),
        a_log=rn(V, std=0.5),
        dt_bias=rn(V, std=0.5).bfloat16(),
        norm_weight=(1 + rn(D, std=0.2)).bfloat16(),
        out_scale=torch.tensor([0.05], device=device),
        ba_weight=rn(2 * V, hidden, std=hidden**-0.5).bfloat16(),  # the b row of every V head, then the a rows
    )
    pool = dict(  # slot 0 is the null slot; sequence n owns slot n + 1
        state=rn(seqs + 1, V, D, D, std=0.1),
        conv_state=rn(seqs + 1, TAPS, C).bfloat16().transpose(-1, -2),  # [slots, C, 3] over channel-innermost storage
    )
    inputs = SimpleNamespace(
        qkvz=rn(steps, seqs, C + V * D).bfloat16(),  # the in-projection's output row: x = [:C], z = [C:]
        hidden=rn(steps, seqs, hidden).bfloat16(),  # the normed hidden the b / a projection reads
    )
    return weights, pool, inputs


def _operands(rows, t, w, pool, inputs, slots):
    """The entry's operands for token ``t`` of the first ``rows`` sequences; ``x`` and ``z`` are two row-strided views of ``qkvz``."""
    C, qkvz = w.conv_weight.shape[0], inputs.qkvz[t]
    x, z = qkvz[:, :C], qkvz[:, C:].view(qkvz.shape[0], -1, D)
    return (rows, x, pool["conv_state"], w.conv_weight, w.a_log, w.dt_bias, pool["state"], slots, z, w.norm_weight, w.out_scale, inputs.hidden[t], w.ba_weight)


def _run(step, rows, steps, w, pool, inputs, slots):
    """``steps`` fused decode steps; returns the 8-bit outputs [steps, rows, V, 128] (the entry returns its own buffer, rewritten by the next launch)."""
    return torch.stack([step(*_operands(rows, t, w, pool, inputs, slots)).clone() for t in range(steps)])


def _bf16(t):
    return t.bfloat16().double()


def _reference_step(rows, t, w, pool, inputs, slots, eps):
    """The same step in plain PyTorch on ``pool``: every bf16 / fp32 cast of the fused step kept, every sum accumulated in fp64."""
    V, C = w.a_log.numel(), w.conv_weight.shape[0]
    K, slots = (C // D - V) // 2, slots[:rows].long()
    qkvz, hidden = inputs.qkvz[t, :rows], inputs.hidden[t, :rows]
    x, z = qkvz[:, :C], qkvz[:, C:].view(rows, V, D).double()
    # causal conv over the three history taps and the new token (every product rounded to bf16), SiLU, bf16; the history shifts by one
    taps = torch.cat([pool["conv_state"][slots], x[:, :, None]], dim=-1)  # [rows, C, 4]
    acc = _bf16(taps.double() * w.conv_weight.double()).sum(-1)
    conv = _bf16(acc * torch.sigmoid(acc))
    pool["conv_state"][slots] = taps[:, :, 1:]
    q, k, v = (part.view(rows, -1, D) for part in conv.split([K * D, K * D, V * D], dim=-1))
    q = q * D**-0.5 / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = (part.repeat_interleave(V // K, dim=1) for part in (q, k))  # V head h reads K head h // (V / K)
    # the write strength and the decay from the b / a projection of the normed hidden (bf16 results)
    ba = _bf16(hidden.double() @ w.ba_weight.double().T)
    beta = torch.sigmoid(ba[:, :V])
    decay = torch.exp(-torch.exp(w.a_log.double()) * torch.nn.functional.softplus(ba[:, V:] + w.dt_bias.double()))
    # the gated delta rule; the state is stored in fp32 and the output reads the stored state
    S = pool["state"][slots].double() * decay[:, :, None, None]
    S = S + torch.einsum("rhm,rhc->rhmc", (v - torch.einsum("rhmc,rhc->rhm", S, k)) * beta[:, :, None], k)
    pool["state"][slots] = S.float()
    o = _bf16(torch.einsum("rhmc,rhc->rhm", pool["state"][slots].double(), q))
    # gated RMSNorm, bf16, the static 8-bit quantization
    out = _bf16(o * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps) * w.norm_weight.double() * z * torch.sigmoid(z))
    return (out / w.out_scale.double()).to(torch.float8_e4m3fn)


def _same_bytes(a, b) -> bool:
    return torch.equal(a.view(_BITS[a.dtype]), b.view(_BITS[b.dtype]))


def _rms_ratio(out, ref):
    out, ref = out.double(), ref.double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _codes(q):
    """e4m3 bytes as signed integers: neighbouring representable values are one apart."""
    b = q.view(torch.uint8).to(torch.int16)
    return torch.where(b >= 128, 128 - b, b)


def main(steps: int = 32, row_counts=(1, 4), K: int = 16, V: int = 48, hidden: int = 5120, eps: float = 1e-6, seed: int = 0) -> None:
    if not torch.cuda.is_available():
        print("[26] SKIP  needs a CUDA GPU")
        return
    if not hasattr(cudnn, "GdnDecodeFused"):
        print(f"[26] SKIP  needs a cuDNN Frontend that carries cudnn.GdnDecodeFused (1.33.0 or newer); this is {cudnn.__version__}")
        return
    from cudnn.linear_attention.frost import GdnGeometry

    device = torch.device("cuda", torch.cuda.current_device())
    free_start = torch.cuda.mem_get_info(device)[0]
    torch.cuda.reset_peak_memory_stats(device)
    seqs = max(row_counts)
    weights, pool, inputs = _draw(steps, seqs, K, V, hidden, device, seed)
    slots = torch.arange(1, seqs + 1, dtype=torch.int32, device=device)
    snapshot = {name: t.clone() for name, t in pool.items()}

    def restore():
        for name, t in pool.items():
            t.copy_(snapshot[name])

    # ---- the fused step: compile once per row count, run, then replay from the snapshot --------------------------
    runs = {}
    try:
        step = cudnn.GdnDecodeFused(device, eps=eps, geometry=GdnGeometry(K, V, hidden))
        for rows in row_counts:
            t0 = time.perf_counter()
            step.compile(*_operands(rows, 0, weights, pool, inputs, slots))
            runs[rows] = dict(compile_s=time.perf_counter() - t0)
    except NotImplementedError as exc:  # the entry's typed decline: a GPU class or a CuTe DSL it does not serve
        print(f"[26] SKIP  {exc}")
        return
    print("[26] compile (the first one imports the CuTe DSL): " + ", ".join(f"rows={rows} {run['compile_s']:.1f} s" for rows, run in runs.items()))
    for rows, run in runs.items():
        restore()
        run["q"] = _run(step, rows, steps, weights, pool, inputs, slots)
        run["pool"] = {name: t.clone() for name, t in pool.items()}
        restore()
        replay = _run(step, rows, steps, weights, pool, inputs, slots)
        torch.cuda.synchronize()
        assert _same_bytes(run["q"], replay) and all(_same_bytes(pool[name], run["pool"][name]) for name in pool), f"rows={rows}: the replay differs"
        for name, t in run["pool"].items():  # the null slot and the slots of the sequences that did not step are untouched
            untouched = _same_bytes(t[0], snapshot[name][0]) and _same_bytes(t[rows + 1 :], snapshot[name][rows + 1 :])
            assert untouched, f"rows={rows}: {name} moved in a slot the steps did not name"

    # ---- the plain PyTorch step, chained over every step on its own pool ----------------------------------------
    # The fused step sums in fp32 and has its own exp / SiLU forms, so now and then one bf16 rounding of the chain goes the other way:
    # an 8-bit output moves by a code, and such a flip in a gate value moves one head's state by up to a bf16 step until the decay
    # washes it out.  Hence a count on the 8-bit outputs and an rms ratio (not a max) on the state.
    ref_pool = {name: t.clone() for name, t in snapshot.items()}
    q_ref = torch.stack([_reference_step(seqs, t, weights, ref_pool, inputs, slots, eps) for t in range(steps)])
    worst_equal, worst_rms = 1.0, 0.0
    for rows, run in runs.items():
        live = slice(1, rows + 1)
        q, state, state_ref = run["q"], run["pool"]["state"][live], ref_pool["state"][live]
        differ = (q.view(torch.uint8) != q_ref[:, :rows].view(torch.uint8)).sum().item()
        equal = 1 - differ / q.numel()
        apart = (_codes(q) - _codes(q_ref[:, :rows])).abs().max().item()
        r_state = _rms_ratio(state, state_ref)
        e_state = ((state.double() - state_ref.double()).abs().max() / state_ref.abs().max()).item()
        assert _same_bytes(run["pool"]["conv_state"][live], ref_pool["conv_state"][live]), f"rows={rows}: the conv history is not the last three tokens"
        assert equal > 0.999, f"rows={rows}: {equal:.4%} of the 8-bit outputs equal the reference"
        assert r_state < 1e-3, f"rows={rows}: state rms ratio {r_state:.3g}"
        worst_equal, worst_rms = min(worst_equal, equal), max(worst_rms, r_state)
        print(
            f"[26] rows={rows}  8-bit outputs equal {equal:.3%} ({differ} of {q.numel()} differ, largest code distance {apart})  "
            f"state rms {r_state:.1e} (max {e_state:.1e} of the largest entry)  replay bitwise"
        )

    # ---- a sequence's bytes do not depend on its neighbours: every run against the widest one ------------------------
    wide = runs[seqs]
    for rows, run in runs.items():
        if rows == seqs:
            continue
        live = slice(1, rows + 1)
        same_slots = all(_same_bytes(run["pool"][name][live], wide["pool"][name][live]) for name in pool)
        assert _same_bytes(run["q"], wide["q"][:, :rows]), f"rows={rows}: the 8-bit outputs differ from the same sequences among {seqs}"
        assert same_slots, f"rows={rows}: the slots differ from the same sequences among {seqs}"

    mib = 2**20
    taken = (free_start - torch.cuda.mem_get_info(device)[0]) / mib
    peak, reserved = torch.cuda.max_memory_allocated(device) / mib, torch.cuda.max_memory_reserved(device) / mib
    print(f"[26] memory: tensors peak {peak:.0f} MiB (allocator {reserved:.0f} MiB); device {taken:+.0f} MiB since the start (other processes count)")
    print(
        f"[26] PASS  gdn decode step (fused)     K={K} V={V} hidden={hidden} steps={steps} rows={list(row_counts)} "
        f"(8-bit equal {worst_equal:.3%}, state rms {worst_rms:.1e}, replay and batch invariance bitwise)"
    )


if __name__ == "__main__":
    main()
