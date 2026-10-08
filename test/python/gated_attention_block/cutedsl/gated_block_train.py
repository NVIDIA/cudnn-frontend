# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's CONVERGENCE harness -- the core: a small decoder trained with the block as its attention layer,
one arm per block dtype path, deterministic and logged step by step.

The model (``ToyDecoder``) is a pre-norm decoder: ``x = x + Block(RMSNorm(x)); x = x + MLP(RMSNorm(x))`` over ``n_layers`` layers,
a tied embedding / LM head, fp32 logits and next-token cross-entropy.  The parameters are fp32 masters; the block and the MLP
consume bf16 copies (``w.to(bfloat16)``, an autograd-tracked cast, so the block's bf16 weight gradients reach the masters through
it); RMSNorm and the residual stream stay fp32; the MLP is two bf16 GEMMs with a GELU, identical across arms and never quantized --
the study isolates the block.  AdamW on the masters (``TrainConfig``: betas 0.9 / 0.95, weight decay 0.1 on the matrices, lr 3e-4
with a linear warm-up and a cosine decay to 10 %, global-norm clipping at 1.0 applied AFTER the gradient metrics are taken).

The attention layer is ``GatedBlockFn``, one ``torch.autograd.Function`` per layer call: ``forward`` quantizes the layer's bf16
operands the way the arm prescribes (nothing for bf16), re-points the ONE compiled ``GatedAttentionBlockFwd`` of the arm at the
layer's spec with ``update_quant_scales`` and runs the UNFUSED training forward into the layer's caller-owned record and output
buffer; ``backward`` re-points the ONE compiled ``GatedAttentionBlockBwd`` at the SAME spec (autograd runs it long after every
layer's forward, so the layer's quantization spec travels with it) and runs it into the layer's gradient buffers over the layer's own workspace
(``quant_scalars`` reads one workspace, so one per layer).  Everything is declared and compiled once per arm and re-bound per step
and layer: no allocation in the block, no recompile.

The arms (``ARMS``; ``Arm`` spells the recipe knobs):

* ``bf16`` -- the bf16 block, the reference trajectory.
* ``mxfp8`` -- ``MxQuantSpec``: ``h`` / ``W_qkvg`` block-quantized every step (e4m3 codes + E8M0 / 32 blobs, the test tree's
  ``quantize_block_inputs_mxfp8``), the transposed artifacts of the backward re-quantized along the other axis from the SAME bf16
  tensors, ``W_o`` per-tensor e4m3 with ``descale_w_o`` from THIS step's amax -- so ``update_quant_scales`` runs per layer and step
  here too --, ``scale_o = 1.0``, the gradient scale of dY derived in-kernel.
* ``fp8-recal1`` -- ``QuantSpec``: ``descale_h`` / ``descale_w_qkvg`` / ``descale_w_o`` from the tensors quantized THIS step (exact,
  current), the activation scales ``scale_q / scale_k / scale_v / scale_o`` RECALIBRATED every step from the previous step's record
  (window 1: ``scale_x = grad_scale_from_amax(amax_x(t - 1), margin)`` -- the kernels' own power-of-two rule, zero headroom), the
  gradient scales derived in-kernel ("current"), ``scale_dp`` from the previous step's ``amax_dp`` through the kernels' own rule
  ``grad_scale_from_amax(amax_dp, dp_margin_log2)`` (``Arm.dp_scale_rule="amax"``, four octaves of headroom by default).  Two
  products, not one: the LAGGED ``amax_dp(t - 1) * scale_dp(t)`` lands in (14, 28] BY CONSTRUCTION (the band the SDPA suites'
  ``get_fp8_scale_factor`` gives for an ``amax_dp >= 0.0625``); the SAME-STEP ``amax_dp(t) * scale_dp(t)`` -- what sets the e4m3 dS
  resolution and what the ``<= 448`` check reads -- is off it by the step's ``amax_dp`` growth and is only REPORTED (derivable from
  the row's ``quant_scalars.amax_dp`` x ``scale_dp``; 1.8 .. 80 measured over a 300-step synthetic smoke run, so the four octaves
  are what absorb an early-training 3x step-over-step growth; a > 16x growth aborts the run, a detector).  The helper itself is
  ``dp_scale_rule="helper"`` -- its ``epsilon`` floor caps the scale at 512 once ``amax_dp`` is below 0.0625, which a mean-reduced
  loss reaches at once: the e4m3 dS then sits in the subnormal range and the dQ / dK-fed gradients collapse.
  Step 0 runs one calibration forward at unit activation scales (discarded; it yields the amax) and one
  discarded backward at ``scale_dp = 1.0`` (it yields ``amax_dp``), so the LOGGED step 0 already runs calibrated -- marked
  ``calib_fwd`` / ``calib_bwd`` in its row.  The lagged recipe saturates silently where an activation's amax grows into the top of
  its band, so the harness COUNTS it: ``sat_x = amax_x(t) * scale_x(t) > 448`` and ``n_clip_x`` = the clipped-element count, per step,
  layer and tensor -- reported, never asserted.
* ``torch-bf16`` -- the pure-torch reference block (``gated_attention_block_reference``) under autograd: the control.

The ``delayed`` recipe (``Arm(grad_scaling="delayed")``) hands the backward the caller's ``scale_dy / scale_do / scale_dqkvg``
(``scale_dy`` alone under MXFP8): ``grad_scale_from_amax`` of the max over the last ``grad_window`` steps' amax of each gradient, or
whatever a ``scale_feed`` callable returns per step and layer -- the replay pin feeds a "current" run's logged scales and gets its
gradients back bitwise.  Without a feed, step 0 BOOTSTRAPS the histories from discarded backward passes
(``ConvergenceRun.bootstrap_rungs``): a gradient's published amax is valid only once every scale UPSTREAM of it is -- a unit scale
flushes a training gradient (amax ~ 1e-3) to zero in e4m3 (subnormal 2^-9) and everything computed from that tensor is garbage -- so
the rungs run in dependency order, each pass at the scales seeded so far: ``scale_dy`` (dY is the backward's INPUT gradient,
``dL/dout`` -- nothing quantized precedes it, so its amax is scale-independent; the block's input gradient is ``dh``), then
``scale_do`` (dO comes from the quantized dY), then ``scale_dp`` (dP from the quantized dO), then
``scale_dqkvg`` (its dQ / dK from the quantized dS); MXFP8 has ``scale_dy`` alone, and the "current" fp8 arm's ladder is ``scale_dp``
alone, as before.  A lagged scale has ZERO headroom at margin 0 by construction, so under ``delayed`` ``amax * scale > 448`` on a
gradient is COUNTED per step, layer and gradient (the row's ``grad_sat``; reported, never asserted); under ``current`` the kernel
derives every gradient scale from the same step's amax, so the same product is ASSERTED there (a violation is a bug).

Determinism: every random draw -- the init, the data -- is on a CPU ``torch.Generator`` (torch's CUDA Philox lays draws out by SM
count, so one seed means different tensors on two parts); ``run()`` turns ``torch.use_deterministic_algorithms(True)`` on for its
duration and requires ``CUBLAS_WORKSPACE_CONFIG`` to name a deterministic workspace (``:4096:8`` or ``:16:8``, set BEFORE the first
cuBLAS call of the process -- the CLI sets it, a test process sets it itself); the block's reductions are order-free.  Each JSONL row
carries ``row_digest`` = sha256 of its canonical JSON minus ``NONDET_KEYS`` (``wall_ms`` / ``timestamp`` / ``host`` / ``pid``) and
``ROW_META_KEYS`` (``replica``), the run
``run_digest`` = sha256 over the row digests, and ``grad_sha256`` = sha256 of the bytes of every layer's ``dw_qkvg`` and ``dw_o``.
Two runs of one arm from one seed must agree on every digest.  The digest covers the ``recipe`` dict too, so it changes with the
harness VERSION at identical numerics (a new recipe knob = a new digest): a comparison across versions or across node classes
compares ``grad_sha256`` and the numeric keys -- every row key minus ``NUMERIC_COMPARE_IGNORE`` -- never the digest alone.  Nothing
here is a performance number: ``wall_ms`` is informational.

Data: ``SyntheticTokens`` -- a copy task (``S / 2`` random tokens followed by their copy: the second half is predictable through
attention only, so the loss on it falls from ``ln V`` toward 0 as the block learns induction) and a second-order Markov chain with a
fixed random transition rule (``next = (a * prev + b * prev2 + offset[k]) mod V``, ``k`` drawn from a fixed distribution ``p``) whose
conditional entropy ``H(p)`` is the analytic loss floor, mixed 1:1 per batch over a 4096 vocabulary.  ``loss_floor_nats`` gives the
batch's floor.

CLI::

    python gated_block_train.py --arm fp8-recal1 --geometry smoke --steps 300 --seed 0 --out <dir> [--name <run>] [--replica 1]
                                [--grad-scaling delayed --grad-window 16] [--margin-log2 2] [--bwd-knobs fuse_gate_bwd,fuse_wgrad_overlap]
                                [--replay-scales-from <jsonl of a "current" run>] [--act-window 1] [--dp-scale-rule amax|helper] [--dp-margin-log2 4]

prints the imported ``cudnn`` module and the device, one line per step, and the median ms per step over steps >= 1 (host-bound;
informational), and writes ``<out>/<name>.jsonl`` + ``<out>/<name>.manifest.json``.  The quantized arms need the Rubin (compute
capability 10.7) block; ``torch-bf16`` runs on any CUDA device.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

if __name__ == "__main__":
    # The CLI opts in to the FROST engines (the block's GEMMs ride an opt-in engine row) and pins cuBLAS's deterministic workspace
    # BEFORE torch / cudnn are imported.  Not at import time of the module: a test process gets the opt-in from conftest.py per test,
    # and a module-level default would leak the flag into the rest of a pytest session.
    os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_PYTHON = os.path.dirname(os.path.dirname(_HERE))  # test/python: the SDPA suites' `sdpa.helpers` (get_fp8_scale_factor)
for _p in (_TEST_PYTHON, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cudnn.gated_attention_block import (  # noqa: E402
    GatedAttentionBlockBwd,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    MxQuantSpec,
    QuantSpec,
    SavedForBackward,
    saved_slab_views,
)
from cudnn.gated_attention_block.kernels.quantize import grad_scale_from_amax  # noqa: E402
from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference, quant_e4m3, quantize_block_inputs_mxfp8  # noqa: E402
from sdpa.helpers import get_fp8_scale_factor  # noqa: E402

__all__ = [
    "ARMS",
    "Arm",
    "ConvergenceRun",
    "DETERMINISTIC_CUBLAS_CONFIGS",
    "E4M3_MAX",
    "GEOMETRIES",
    "GatedBlockFn",
    "NONDET_KEYS",
    "NUMERIC_COMPARE_IGNORE",
    "ROW_META_KEYS",
    "RunResult",
    "SyntheticTokens",
    "ToyDecoder",
    "TrainConfig",
    "TrainGeometry",
    "canonical_json",
    "row_digest",
    "run_digest",
    "run_training",
]

E4M3_MAX = 448.0
_E4M3 = torch.float8_e4m3fn
_BF16 = torch.bfloat16
NONDET_KEYS = frozenset({"wall_ms", "timestamp", "host", "pid"})
ROW_META_KEYS = frozenset({"replica"})  # written into the row AFTER the digest, like the NONDET keys: two replicas of one seed digest equal
# The digest is keyed to the harness VERSION as well as to the numerics: ``recipe`` (the arm's knobs, which grow with the harness) sits
# inside it, so two versions digest DIFFERENT at identical numerics.  A cross-version or cross-node-class comparison drops these keys
# (plus NONDET_KEYS / ROW_META_KEYS) and compares what is left -- ``grad_sha256``, ``loss``, ``grad_norms``, ``quant_scalars``, ... .
NUMERIC_COMPARE_IGNORE = frozenset({"recipe", "row_digest"})
DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")
_ACTS = ("q", "k", "v", "o")
_GRAD_NAMES = ("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm")
_FP8_GRAD_SCALES = ("scale_dy", "scale_do", "scale_dqkvg")
_MX_GRAD_SCALES = ("scale_dy",)
_MX_ARTIFACTS = ("h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf")
_BWD_KNOBS = ("fuse_gate_bwd", "fuse_wgrad_overlap")
# The per-layer METRICS tensor (device fp32, one per layer, read back ONCE per step): the activation amax and clipped-element counts
# of the forward's record (fp8 arms) and the fp32 norms of the five block gradients.
_MET = {name: i for i, name in enumerate([f"amax_{x}" for x in _ACTS] + [f"n_clip_{x}" for x in _ACTS] + [f"norm_{g}" for g in _GRAD_NAMES])}


def _print(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Geometry, training constants, arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainGeometry:
    """The decoder's shape: the block's head geometry (causal, QK-RMSNorm on) plus the model axes around it."""

    d_model: int
    h_q: int
    h_kv: int
    d_head: int
    rope_dim: int
    n_layers: int
    batch: int
    seq_len: int
    vocab: int

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model

    @property
    def tokens_per_step(self) -> int:
        return self.batch * self.seq_len

    @property
    def block_kw(self) -> dict:
        return dict(d_model=self.d_model, h_q=self.h_q, h_kv=self.h_kv, d_head=self.d_head, rope_dim=self.rope_dim)

    def block_geometry(self) -> GatedAttentionBlockGeometry:
        return GatedAttentionBlockGeometry(**self.block_kw)

    def ref_geometry(self) -> RefGeometry:
        return RefGeometry(**self.block_kw)


GEOMETRIES = {
    # the accept suites' head geometry at two model sizes; both satisfy the quantized backward's rules (d_model % 256 == 0, T % 32 == 0)
    "smoke": TrainGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64, n_layers=2, batch=2, seq_len=1024, vocab=4096),
    "headline": TrainGeometry(d_model=1024, h_q=8, h_kv=2, d_head=256, rope_dim=64, n_layers=4, batch=4, seq_len=2048, vocab=50304),
}


@dataclass(frozen=True)
class TrainConfig:
    """One optimizer recipe for every arm."""

    lr: float = 3e-4
    warmup_steps: int = 200
    min_lr_frac: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1  # on the matrices; none on the norm weights
    clip_norm: float = 1.0
    init_std: float = 0.02
    rms_eps: float = 1e-6

    def lr_at(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if total_steps <= self.warmup_steps:
            return self.lr
        frac = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return self.lr * (self.min_lr_frac + (1.0 - self.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))


@dataclass(frozen=True)
class Arm:
    """One block dtype path plus its recipe knobs.

    ``family``: ``None`` = the bf16 block; ``"fp8"`` = per-tensor e4m3 (``QuantSpec``); ``"mxfp8"`` = block-scaled (``MxQuantSpec``);
    ``"torch"`` = the pure-torch reference block under autograd; ``"torch_fn"`` = the same reference block driven through
    ``GatedBlockFn`` (the trainer plumbing on any CUDA device).  ``grad_scaling`` / ``grad_window`` / ``margin_log2``: the gradient
    recipe of a quantized backward (``"current"`` derives the scales in-kernel; ``"delayed"`` hands it the max over the last
    ``grad_window`` amax; the margin is the "current" recipe's octaves of headroom, the declaration attribute).  ``act_window``
    (fp8): the activation amax window of the forward recipe -- 1 recalibrates from the previous step, 0 keeps unit activation
    scales.  ``bwd_knobs``: performance-only backward knobs, bitwise the default.  ``dp_scale_rule`` / ``dp_margin_log2`` (fp8): how the next
    step's ``scale_dp`` follows this step's ``amax_dp`` -- ``"amax"`` = ``grad_scale_from_amax(amax_dp, dp_margin_log2)`` (the kernels'
    rule: the LAGGED product ``amax_dp(t) * scale_dp(t + 1)`` in ``(448 / 2**(m+1), 448 / 2**m]`` by construction; the same-step
    ``amax_dp(t + 1) * scale_dp(t + 1)`` is off it by the step's amax growth and only reported), ``"helper"`` = the SDPA suites'
    ``get_fp8_scale_factor`` (a 512 cap below ``amax_dp = 0.0625``)."""

    name: str
    family: Optional[str]
    grad_scaling: str = "current"
    grad_window: int = 16
    margin_log2: int = 0
    act_window: int = 1
    bwd_knobs: Tuple[str, ...] = ()
    dp_scale_rule: str = "amax"
    dp_margin_log2: int = 4

    def __post_init__(self):
        if self.family not in (None, "fp8", "mxfp8", "torch", "torch_fn"):
            raise ValueError(f"Arm.family must be None, 'fp8', 'mxfp8', 'torch' or 'torch_fn', got {self.family!r}")
        if self.grad_scaling not in ("current", "delayed"):
            raise ValueError(f"Arm.grad_scaling must be 'current' or 'delayed', got {self.grad_scaling!r}")
        if not self.quantized and (self.grad_scaling != "current" or self.margin_log2 != 0):
            raise ValueError(f"{self.name}: a gradient recipe belongs to a quantized arm (family fp8 / mxfp8)")
        if self.grad_window < 1 or self.act_window < 0 or not (0 <= self.margin_log2 <= 8):
            raise ValueError(f"{self.name}: grad_window >= 1, act_window >= 0, margin_log2 in [0, 8]")
        if any(k not in _BWD_KNOBS for k in self.bwd_knobs):
            raise ValueError(f"{self.name}: bwd_knobs must be among {_BWD_KNOBS}, got {self.bwd_knobs}")
        if self.bwd_knobs and self.family in ("torch", "torch_fn"):
            raise ValueError(f"{self.name}: the torch reference has no backward knobs")
        if self.dp_scale_rule not in ("amax", "helper") or not (0 <= self.dp_margin_log2 <= 8):
            raise ValueError(f"{self.name}: dp_scale_rule must be 'amax' or 'helper' and dp_margin_log2 in [0, 8]")

    @property
    def quantized(self) -> bool:
        return self.family in ("fp8", "mxfp8")

    @property
    def uses_block(self) -> bool:
        return self.family in (None, "fp8", "mxfp8")

    @property
    def grad_scale_names(self) -> Tuple[str, ...]:
        return _FP8_GRAD_SCALES if self.family == "fp8" else (_MX_GRAD_SCALES if self.family == "mxfp8" else ())

    def with_(self, **kw) -> "Arm":
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> dict:
        return dict(dataclasses.asdict(self), bwd_knobs=list(self.bwd_knobs))


ARMS = {
    "bf16": Arm("bf16", None),
    "mxfp8": Arm("mxfp8", "mxfp8"),
    "fp8-recal1": Arm("fp8-recal1", "fp8", act_window=1),
    "torch-bf16": Arm("torch-bf16", "torch"),
    "torch-bf16-fn": Arm("torch-bf16-fn", "torch_fn"),
}


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)


def row_digest(row: dict) -> str:
    """sha256 of the row's canonical JSON minus ``NONDET_KEYS``, ``ROW_META_KEYS`` and the digest itself."""
    skip = NONDET_KEYS | ROW_META_KEYS | {"row_digest"}
    return hashlib.sha256(canonical_json({k: v for k, v in row.items() if k not in skip}).encode()).hexdigest()


def run_digest(rows: List[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(r["row_digest"].encode())
    return h.hexdigest()


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


# ---------------------------------------------------------------------------
# Data: the synthetic token stream (CPU generator)
# ---------------------------------------------------------------------------


class SyntheticTokens:
    """Batches of ``[batch, seq_len + 1]`` tokens from a CPU generator: even rows the copy task, odd rows the Markov chain."""

    def __init__(self, geom: TrainGeometry, seed: int, *, support: int = 4):
        self.geom = geom
        self.seed = seed
        v = geom.vocab
        self._g = torch.Generator().manual_seed(seed)
        g = self._g
        self.mul_1 = int(torch.randint(1, v, (1,), generator=g))
        self.mul_2 = int(torch.randint(1, v, (1,), generator=g))
        self.offsets = torch.randperm(v, generator=g)[:support].tolist()  # distinct offsets -> distinct next tokens
        w = torch.rand(support, generator=g, dtype=torch.float64) + 0.25
        self.probs = w / w.sum()
        self.markov_entropy_nats = float(-(self.probs * self.probs.log()).sum())
        self.n_drawn = 0

    def _copy_sequence(self) -> torch.Tensor:
        n = self.geom.seq_len + 1
        half = self.geom.seq_len // 2
        first = torch.randint(0, self.geom.vocab, (n - half,), generator=self._g)
        return torch.cat((first, first[:half]))

    def _markov_sequence(self) -> torch.Tensor:
        n, v = self.geom.seq_len + 1, self.geom.vocab
        x = torch.randint(0, v, (2,), generator=self._g).tolist()
        ks = torch.multinomial(self.probs, n - 2, replacement=True, generator=self._g).tolist()
        for k in ks:
            x.append((self.mul_1 * x[-1] + self.mul_2 * x[-2] + self.offsets[k]) % v)
        return torch.tensor(x, dtype=torch.int64)

    def batch(self) -> torch.Tensor:
        """The next batch of the stream (sequential: run ``batch()`` in step order for a reproducible trajectory)."""
        rows = [self._copy_sequence() if b % 2 == 0 else self._markov_sequence() for b in range(self.geom.batch)]
        self.n_drawn += 1
        return torch.stack(rows)

    def loss_floor_nats(self) -> float:
        """The analytic per-token floor of a batch: ``ln V`` on the random tokens, 0 on the copied half, ``H(p)`` on the chain."""
        s, v = self.geom.seq_len, self.geom.vocab
        half = s // 2
        ln_v = math.log(v)
        copy_floor = ((s - half) * ln_v + half * 0.0) / s  # targets j < s - half are random draws, the rest copies
        markov_floor = (ln_v + (s - 1) * self.markov_entropy_nats) / s  # target 0 is a random draw, the rest the chain
        per_row = [copy_floor if b % 2 == 0 else markov_floor for b in range(self.geom.batch)]
        return sum(per_row) / len(per_row)

    def as_dict(self) -> dict:
        return dict(
            kind="synthetic", seed=self.seed, vocab=self.geom.vocab, markov_entropy_nats=self.markov_entropy_nats, loss_floor_nats=self.loss_floor_nats()
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class ToyDecoder:
    """A pre-norm decoder with fp32 masters; ``forward(tokens, targets, block)`` calls ``block(layer, h_bf16, w_qkvg, w_o, w_q_norm,
    w_k_norm)`` for the attention sub-layer and returns the fp32 mean cross-entropy.  The init is drawn from a CPU generator in a
    FIXED order (the test tree's convention: matrices ``randn * init_std``, norm weights 1.0), so every arm starts identical."""

    def __init__(self, geom: TrainGeometry, cfg: TrainConfig, seed: int, device: torch.device):
        self.geom, self.cfg, self.device = geom, cfg, device
        g = torch.Generator().manual_seed(seed)
        dm, bg = geom.d_model, geom.block_geometry()

        def randn(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float32) * cfg.init_std

        p: Dict[str, torch.Tensor] = {"emb": randn(geom.vocab, dm)}
        for l in range(geom.n_layers):
            p[f"ln1.{l}"] = torch.ones(dm)
            p[f"w_qkvg.{l}"] = randn(bg.n_qkvg, dm)
            p[f"w_o.{l}"] = randn(dm, geom.h_q * geom.d_head)
            p[f"w_q_norm.{l}"] = torch.ones(geom.d_head)
            p[f"w_k_norm.{l}"] = torch.ones(geom.d_head)
            p[f"ln2.{l}"] = torch.ones(dm)
            p[f"mlp_w1.{l}"] = randn(geom.d_ff, dm)
            p[f"mlp_w2.{l}"] = randn(dm, geom.d_ff)
        p["lnf"] = torch.ones(dm)
        self.params = {k: v.to(device).requires_grad_(True) for k, v in p.items()}

    def param_groups(self) -> list:
        decay = [v for k, v in self.params.items() if v.dim() == 2]
        no_decay = [v for k, v in self.params.items() if v.dim() != 2]
        return [dict(params=decay, weight_decay=self.cfg.weight_decay), dict(params=no_decay, weight_decay=0.0)]

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, block: Callable) -> torch.Tensor:
        p, g, eps = self.params, self.geom, self.cfg.rms_eps
        x = F.embedding(tokens, p["emb"])  # fp32 [B, S, d_model]
        for l in range(g.n_layers):
            h = _rms_norm(x, p[f"ln1.{l}"], eps).to(_BF16)  # the block's input: the normed residual stream, one tracked bf16 cast
            out = block(l, h, p[f"w_qkvg.{l}"], p[f"w_o.{l}"], p[f"w_q_norm.{l}"], p[f"w_k_norm.{l}"])
            x = x + out.float()
            h2 = _rms_norm(x, p[f"ln2.{l}"], eps).to(_BF16)
            m = F.linear(F.gelu(F.linear(h2, p[f"mlp_w1.{l}"].to(_BF16))), p[f"mlp_w2.{l}"].to(_BF16))
            x = x + m.float()
        xn = _rms_norm(x, p["lnf"], eps)
        logits = xn.reshape(-1, g.d_model) @ p["emb"].t()  # fp32, tied
        return F.cross_entropy(logits, targets.reshape(-1))


# ---------------------------------------------------------------------------
# The block as an autograd Function, and the per-layer state behind it
# ---------------------------------------------------------------------------


class GatedBlockFn(torch.autograd.Function):
    """``forward(h, w_qkvg, w_o, w_q_norm, w_k_norm, run, layer)`` -> the layer's output buffer; ``backward`` returns the block's
    gradients for the five tensors (``dw_q_norm`` / ``dw_k_norm`` cast to the bf16 of the norm-weight copies)."""

    @staticmethod
    def forward(ctx, h, w_qkvg, w_o, w_q_norm, w_k_norm, run, layer):
        out, spec = run._forward_layer(layer, h, w_qkvg, w_o, w_q_norm, w_k_norm)
        ctx.run, ctx.layer, ctx.spec = run, layer, spec
        return out

    @staticmethod
    def backward(ctx, dy):
        grads = ctx.run._backward_layer(ctx.layer, dy, ctx.spec)
        return grads["dh"], grads["dw_qkvg"], grads["dw_o"], grads["dw_q_norm"].to(_BF16), grads["dw_k_norm"].to(_BF16), None, None


class _LayerState:
    """Everything one layer owns: the record buffers, the output and gradient buffers, the backward workspace, the device scalars of
    the recipe, the amax histories, the current step's quantization spec and the metrics tensor."""

    def __init__(self, run: "ConvergenceRun", layer: int):
        g, geom, dev = run.block_geom, run.geom, run.device
        b, s = geom.batch, geom.seq_len
        t = b * s
        self.layer = layer
        self.proj_slab = torch.empty(t, g.n_qkvg, dtype=_BF16, device=dev)
        self.q_pre, self.gate, self.k_pre, self.v = saved_slab_views(self.proj_slab, g, b, s)
        self.o = torch.empty(b, s, g.h_q, g.d_head, dtype=_BF16, device=dev)
        self.lse = torch.empty(b, g.h_q, s, dtype=torch.float32, device=dev)
        self.rstd_q = torch.empty(b, s, g.h_q, dtype=torch.float32, device=dev)
        self.rstd_k = torch.empty(b, s, g.h_kv, dtype=torch.float32, device=dev)
        self.out = torch.empty(b, s, g.d_model, dtype=_BF16, device=dev)
        self.grads = dict(
            dh=torch.empty(b, s, g.d_model, dtype=_BF16, device=dev),
            dw_qkvg=torch.empty(g.n_qkvg, g.d_model, dtype=_BF16, device=dev),
            dw_o=torch.empty(g.d_model, g.h_q * g.d_head, dtype=_BF16, device=dev),
            dw_q_norm=torch.empty(g.d_head, dtype=torch.float32, device=dev),
            dw_k_norm=torch.empty(g.d_head, dtype=torch.float32, device=dev),
        )
        self.ws_bwd: Optional[torch.Tensor] = None
        self.met = torch.zeros(len(_MET), dtype=torch.float32, device=dev)
        arm = run.arm
        self.scale_dp = torch.full((1,), 1.0, dtype=torch.float32, device=dev) if arm.family == "fp8" else None
        self.scale_dp_host = 1.0
        self.grad_scales = {n: torch.full((1,), 1.0, dtype=torch.float32, device=dev) for n in arm.grad_scale_names} if arm.grad_scaling == "delayed" else {}
        self.grad_scales_host: Dict[str, float] = {}
        self.act_hist = {x: collections.deque(maxlen=max(1, arm.act_window)) for x in _ACTS}
        self.grad_hist = {n: collections.deque(maxlen=arm.grad_window) for n in arm.grad_scale_names}
        self.spec = None
        self.inputs: dict = {}
        self.saved: Optional[SavedForBackward] = None
        self.act_scales: Dict[str, float] = {}

    def record(self, h: torch.Tensor) -> SavedForBackward:
        return SavedForBackward(
            h=h, gate=self.gate, q_pre=self.q_pre, k_pre=self.k_pre, o=self.o, lse=self.lse, rstd_q=self.rstd_q, rstd_k=self.rstd_k, proj_slab=self.proj_slab
        )


@dataclass
class RunResult:
    rows: List[dict]
    manifest: dict
    kept_grads: Dict[int, Dict[str, torch.Tensor]]
    run: "ConvergenceRun"


class ConvergenceRun:
    """One arm's trajectory: the model, the data stream, the compiled block pair (declared lazily at the first forward / backward),
    the per-layer state and the row writer.  ``run(steps)`` trains and returns the rows; see the module docstring."""

    def __init__(
        self,
        arm: Arm,
        geom: TrainGeometry,
        *,
        seed: int = 0,
        data_seed: Optional[int] = None,
        cfg: TrainConfig = TrainConfig(),
        device="cuda",
        replica: int = 0,
        name: Optional[str] = None,
        scale_feed: Optional[Callable[[int, int], Dict[str, float]]] = None,
        log: Callable = _print,
    ):
        self.arm, self.geom, self.cfg, self.seed, self.replica = arm, geom, cfg, seed, replica
        self.data_seed = seed if data_seed is None else data_seed
        self.name = name or f"{arm.name}_r{replica}"
        self.device = torch.device(device)
        self.log = log
        self.scale_feed = scale_feed
        self.block_geom = geom.block_geometry()
        self.ref_geom = geom.ref_geometry()
        self.model = ToyDecoder(geom, cfg, seed, self.device)
        self.data = SyntheticTokens(geom, self.data_seed)
        self.cos, self.sin = build_rope_tables(geom.seq_len, geom.rope_dim, batch=geom.batch, device=self.device, dtype=_BF16)
        self.cos, self.sin = self.cos.contiguous(), self.sin.contiguous()
        self.opt = torch.optim.AdamW(self.model.param_groups(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, foreach=True)
        self.layers = [_LayerState(self, l) for l in range(geom.n_layers)]
        self.fwd: Optional[GatedAttentionBlockFwd] = None
        self.bwd: Optional[GatedAttentionBlockBwd] = None
        self.ws_fwd: Optional[torch.Tensor] = None
        self._replay = None  # the fp8 recipe's norm+RoPE replay scratch (nq, nk, rstd_q, rstd_k)
        self.step = -1
        self.calibrating = False
        self._calib_bwd = 0  # discarded step-0 backward passes per layer (the bootstrap ladder's rungs); the row logs it as a bool
        self._grad_sat: Optional[Dict[str, List[bool]]] = None

    # -- environment ------------------------------------------------------------------------------------------------------

    @staticmethod
    def check_environment() -> None:
        cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cfg not in DETERMINISTIC_CUBLAS_CONFIGS:
            raise RuntimeError(
                f"CUBLAS_WORKSPACE_CONFIG must name a deterministic cuBLAS workspace ({' or '.join(DETERMINISTIC_CUBLAS_CONFIGS)}) and be set "
                f"before the process's first cuBLAS call; got {cfg!r}.  torch.use_deterministic_algorithms(True) refuses cuBLAS GEMMs without it."
            )

    # -- the block call (the model's attention sub-layer) ---------------------------------------------------------------------

    def block(self, l: int, h16: torch.Tensor, w_qkvg32, w_o32, wq32, wk32) -> torch.Tensor:
        w_qkvg16, w_o16, wq16, wk16 = w_qkvg32.to(_BF16), w_o32.to(_BF16), wq32.to(_BF16), wk32.to(_BF16)  # tracked casts
        if self.arm.family == "torch":
            return gated_attention_block_reference(h16, w_qkvg16, wq16, wk16, self.cos, self.sin, w_o16, geom=self.ref_geom).out
        return GatedBlockFn.apply(h16, w_qkvg16, w_o16, wq16, wk16, self, l)

    # -- forward ----------------------------------------------------------------------------------------------------------------

    def _activation_scales(self, st: _LayerState) -> Dict[str, float]:
        if self.calibrating or self.arm.act_window == 0:
            return {f"scale_{x}": 1.0 for x in _ACTS}
        for x in _ACTS:
            if not st.act_hist[x]:
                raise RuntimeError(f"layer {st.layer}: no activation amax history for {x} -- the step-0 calibration forward did not run")
        return {f"scale_{x}": grad_scale_from_amax(max(st.act_hist[x]), self.arm.margin_log2) for x in _ACTS}

    def _quantize_layer_inputs(self, st: _LayerState, h16, w_qkvg16, w_o16, wq16, wk16):
        """The arm's quantization of this layer's bf16 operands -> ``(inputs, spec)``; ``inputs`` carries ``h`` / ``w_qkvg`` / ``w_o``
        in the block's dtypes, the MXFP8 blobs under ``sf`` and the backward's artifacts under ``art``."""
        fam = self.arm.family
        if fam == "fp8":
            amax = torch.stack([h16.detach().abs().amax(), w_qkvg16.detach().abs().amax(), w_o16.detach().abs().amax()]).float()
            s_h, s_wq, s_wo = (E4M3_MAX / amax.clamp_min(1e-8)).tolist()  # the test tree's amax_scale, three tensors in ONE sync
            act = self._activation_scales(st)
            st.act_scales = act
            spec = QuantSpec(descale_h=1.0 / s_h, descale_w_qkvg=1.0 / s_wq, descale_w_o=1.0 / s_wo, **act)
            inputs = dict(h=quant_e4m3(h16, s_h), w_qkvg=quant_e4m3(w_qkvg16, s_wq), w_o=quant_e4m3(w_o16, s_wo), sf={}, art={})
        elif fam == "mxfp8":
            src = dict(h=h16.detach(), w_qkvg=w_qkvg16.detach(), w_o=w_o16.detach(), w_q_norm=wq16, w_k_norm=wk16, cos=self.cos, sin=self.sin)
            mx, desc = quantize_block_inputs_mxfp8(src, backward=True)
            spec = MxQuantSpec(descale_w_o=desc["descale_w_o"], scale_o=1.0)
            inputs = dict(
                h=mx["h"], w_qkvg=mx["w_qkvg"], w_o=mx["w_o"], sf=dict(h_sf=mx["h_sf"], w_qkvg_sf=mx["w_qkvg_sf"]), art={k: mx[k] for k in _MX_ARTIFACTS}
            )
        else:
            spec = None
            inputs = dict(h=h16, w_qkvg=w_qkvg16, w_o=w_o16, sf={}, art={})
        inputs.update(w_q_norm=wq16, w_k_norm=wk16)
        return inputs, spec

    def _declare_fwd(self, inputs: dict, spec, out: torch.Tensor) -> None:
        kw: dict = dict(save_for_backward=True)
        if spec is not None:
            kw["quant"] = spec
        if self.arm.family == "mxfp8":
            kw.update(sample_h_sf=inputs["sf"]["h_sf"], sample_w_qkvg_sf=inputs["sf"]["w_qkvg_sf"])
        self.fwd = GatedAttentionBlockFwd(
            inputs["h"], inputs["w_qkvg"], inputs["w_q_norm"], inputs["w_k_norm"], self.cos, self.sin, inputs["w_o"], out, self.block_geom, **kw
        )
        self.fwd.check_support()
        self.fwd.compile()
        self.ws_fwd = torch.empty(self.fwd.get_workspace_size(), dtype=torch.uint8, device=self.device)
        if self.arm.family == "fp8":
            g, t = self.block_geom, self.geom.tokens_per_step
            self._replay = (
                torch.empty(t, g.h_q, g.d_head, dtype=_BF16, device=self.device),
                torch.empty(t, g.h_kv, g.d_head, dtype=_BF16, device=self.device),
                torch.empty(t, g.h_q, dtype=torch.float32, device=self.device),
                torch.empty(t, g.h_kv, dtype=torch.float32, device=self.device),
            )

    def _forward_layer(self, l: int, h16, w_qkvg16, w_o16, wq16, wk16):
        st = self.layers[l]
        if self.arm.family == "torch_fn":
            with torch.no_grad():
                ref = gated_attention_block_reference(h16, w_qkvg16, wq16, wk16, self.cos, self.sin, w_o16, geom=self.ref_geom)
            st.out.copy_(ref.out)
            st.inputs = dict(h=h16, w_qkvg=w_qkvg16, w_o=w_o16, w_q_norm=wq16, w_k_norm=wk16)
            st.spec = None
            return st.out, None
        inputs, spec = self._quantize_layer_inputs(st, h16, w_qkvg16, w_o16, wq16, wk16)
        st.inputs, st.spec = inputs, spec
        st.saved = st.record(inputs["h"])
        if self.fwd is None:
            self._declare_fwd(inputs, spec, st.out)
        if spec is not None:
            self.fwd.update_quant_scales(spec)
        self.fwd.execute(inputs["h"], inputs["w_qkvg"], wq16, wk16, self.cos, self.sin, inputs["w_o"], st.out, self.ws_fwd, saved=st.saved, **inputs["sf"])
        if self.arm.family == "fp8":
            self._record_activation_amax(st, wq16, wk16)
        return st.out, spec

    def _record_activation_amax(self, st: _LayerState, wq16, wk16) -> None:
        """This step's activation amax of the record -- the forward's OWN norm+RoPE stage replayed out of place over the slab's
        pre-norm Q / K bands (bitwise the tensors the quantize launches read), the slab's V band, and the torch gated O -- plus
        the clipped-element counts at this step's scales, into the layer's metrics tensor (read back once per step)."""
        nq, nk, rq, rk = self._replay
        st_fwd = self.fwd._norm_rope
        st_fwd.execute(st.q_pre, st.k_pre, wq16, wk16, self.cos, self.sin, q_out=nq, k_out=nk, rstd_q=rq, rstd_k=rk)
        og = st.o.float() * torch.sigmoid(st.gate.float())
        for x, ten in (("q", nq), ("k", nk), ("v", st.v), ("o", og)):
            a = ten.float().abs()
            st.met[_MET[f"amax_{x}"]] = a.amax()
            st.met[_MET[f"n_clip_{x}"]] = (a * st.act_scales[f"scale_{x}"] > E4M3_MAX).sum().float()

    # -- backward ---------------------------------------------------------------------------------------------------------------

    def _declare_bwd(self, dy: torch.Tensor, st: _LayerState) -> None:
        inp, kw = st.inputs, {k: True for k in self.arm.bwd_knobs}
        if st.spec is not None:
            kw.update(quant=st.spec, grad_scaling=self.arm.grad_scaling)
            if self.arm.grad_scaling == "current":
                kw["grad_scale_margin_log2"] = self.arm.margin_log2
        self.bwd = GatedAttentionBlockBwd(dy, st.saved, inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], self.cos, self.sin, inp["w_o"], self.block_geom, **kw)
        self.bwd.check_support()
        self.bwd.compile()
        for other in self.layers:
            other.ws_bwd = torch.empty(self.bwd.get_workspace_size(), dtype=torch.uint8, device=self.device)

    def _bwd_kwargs(self, st: _LayerState) -> dict:
        kw: dict = {}
        if self.arm.family == "fp8":
            kw["scale_dp"] = st.scale_dp
        if self.arm.grad_scaling == "delayed":
            kw.update(st.grad_scales)
        if self.arm.family == "mxfp8":
            kw.update(st.inputs["art"])
        return kw

    def _execute_bwd(self, st: _LayerState, dy: torch.Tensor) -> None:
        inp = st.inputs
        self.bwd.execute(
            dy,
            st.saved,
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            self.cos,
            self.sin,
            inp["w_o"],
            workspace=st.ws_bwd,
            **st.grads,
            **self._bwd_kwargs(st),
        )

    def _next_scale_dp(self, amax_dp: float) -> float:
        """The fp8 SDPA row's dP scale for the NEXT execute from this execute's ``amax_dp`` (``Arm.dp_scale_rule``)."""
        if self.arm.dp_scale_rule == "helper":
            return float(get_fp8_scale_factor(amax_dp, _E4M3))
        return float(grad_scale_from_amax(amax_dp, self.arm.dp_margin_log2))

    def _read_scalars(self, st: _LayerState) -> Dict[str, float]:
        d = self.bwd.quant_scalars(st.ws_bwd)
        return dict(zip(d.keys(), torch.cat(list(d.values())).tolist()))

    def bootstrap_rungs(self) -> List[str]:
        """The scales the arm seeds at step 0 from DISCARDED backward passes, in dependency order -- one pass per rung, each at the
        scales seeded so far (unit elsewhere), because a gradient's published amax is valid only once every scale UPSTREAM of it is:
        dY's is scale-independent (dY is the backward's INPUT, ``dL/dout`` -- nothing quantized precedes it; the block's input gradient
        is ``dh``), dO's needs ``scale_dy`` (dO is computed from the quantized dY), dP's needs ``scale_do``, dQKVG's needs ``scale_dp``
        (its dQ / dK come from the quantized dS).  A unit scale flushes a training
        gradient (amax ~ 1e-3) to zero in e4m3 (subnormal 2^-9), so a downstream amax read at unit upstream scales is garbage --
        seeding ``scale_dqkvg`` from it saturates the first logged pass by orders of magnitude.  "current" fp8: ``scale_dp`` alone (the
        kernel derives the three gradient scales from the same step's amax); "delayed" without a feed: the gradient scales too (fp8:
        dy, do, dp, dqkvg; MXFP8: dy); a feed supplies them from step 0."""
        seed = self.arm.grad_scaling == "delayed" and self.scale_feed is None
        if self.arm.family == "fp8":
            return (["scale_dy", "scale_do"] if seed else []) + ["scale_dp"] + (["scale_dqkvg"] if seed else [])
        if self.arm.family == "mxfp8" and seed:
            return ["scale_dy"]
        return []

    def _backward_layer(self, l: int, dy: torch.Tensor, spec) -> dict:
        st = self.layers[l]
        if self.arm.family == "torch_fn":
            return self._torch_fn_backward(st, dy)
        dy = dy.contiguous()
        if self.bwd is None:
            self._declare_bwd(dy, st)
        if spec is not None:
            self.bwd.update_quant_scales(spec)
        if self.step == 0:
            # the step-0 bootstrap ladder (``bootstrap_rungs``): one DISCARDED backward per rung, its published amax seeding that
            # rung's scale for every later pass; the gradients are overwritten by the logged pass below; one host sync per rung,
            # step 0 only.  The seeded amax enters the gradient histories through the logged pass's readback, not here.
            rungs = self.bootstrap_rungs()
            if rungs and self.arm.family == "fp8":
                st.scale_dp.fill_(1.0)
                st.scale_dp_host = 1.0
            for rung in rungs:
                self._execute_bwd(st, dy)
                torch.cuda.synchronize()
                sc = self._read_scalars(st)
                if rung == "scale_dp":
                    st.scale_dp_host = self._next_scale_dp(sc["amax_dp"])
                    st.scale_dp.fill_(st.scale_dp_host)
                else:
                    self._set_delayed_scales(st, {rung: grad_scale_from_amax(sc["amax_" + rung[len("scale_") :]], self.arm.margin_log2)})
            self._calib_bwd = len(rungs)
        self._execute_bwd(st, dy)
        for name in _GRAD_NAMES:
            st.met[_MET[f"norm_{name}"]] = st.grads[name].float().norm()
        return st.grads

    def _torch_fn_backward(self, st: _LayerState, dy: torch.Tensor) -> dict:
        """The reference block's gradients through autograd, into the layer's buffers (the trainer plumbing on any CUDA device)."""
        inp = st.inputs
        with torch.enable_grad():
            leaves = {k: inp[k].detach().requires_grad_(True) for k in ("h", "w_qkvg", "w_o", "w_q_norm", "w_k_norm")}
            ref = gated_attention_block_reference(
                leaves["h"], leaves["w_qkvg"], leaves["w_q_norm"], leaves["w_k_norm"], self.cos, self.sin, leaves["w_o"], geom=self.ref_geom
            )
            g = torch.autograd.grad(ref.out, list(leaves.values()), dy)
        for name, src in zip(("dh", "dw_qkvg", "dw_o", "dw_q_norm", "dw_k_norm"), g):
            st.grads[name].copy_(src)
        for name in _GRAD_NAMES:
            st.met[_MET[f"norm_{name}"]] = st.grads[name].float().norm()
        return st.grads

    # -- the step ------------------------------------------------------------------------------------------------------------------

    def _set_delayed_scales(self, st: _LayerState, vals: Dict[str, float]) -> None:
        for n, v in vals.items():
            st.grad_scales[n].fill_(float(v))
            st.grad_scales_host[n] = float(v)

    def _begin_step(self, step: int) -> None:
        self.step = step
        self._calib_bwd = 0
        self._grad_sat = {n: [] for n in self.arm.grad_scale_names} if self.arm.grad_scaling == "delayed" else None
        if self.arm.grad_scaling != "delayed":
            return
        for st in self.layers:
            if self.scale_feed is not None:
                vals = self.scale_feed(step, st.layer)
                missing = [n for n in self.arm.grad_scale_names if n not in vals]
                if missing:
                    raise ValueError(f"scale_feed(step={step}, layer={st.layer}) lacks {missing}")
                self._set_delayed_scales(st, {n: vals[n] for n in self.arm.grad_scale_names})
            elif step > 0:
                self._set_delayed_scales(st, {n: grad_scale_from_amax(max(st.grad_hist[n]), self.arm.margin_log2) for n in self.arm.grad_scale_names})
            # step 0 without a feed: the bootstrap ladder in _backward_layer seeds each scale from a discarded pass (bootstrap_rungs)

    def _needs_activation_calibration(self) -> bool:
        return self.arm.family == "fp8" and self.arm.act_window > 0

    def _seed_activation_histories(self) -> None:
        torch.cuda.synchronize()
        for st in self.layers:
            met = st.met.tolist()
            for x in _ACTS:
                st.act_hist[x].append(met[_MET[f"amax_{x}"]])

    def _grad_norm(self) -> float:
        sq = torch.stack([p.grad.float().pow(2).sum() for p in self.model.params.values()]).sum()
        return float(sq.sqrt().item())

    def _torch_arm_metrics(self) -> None:
        """The ``torch`` arm runs no Function: its gradient norms are those of the fp32 master gradients of the block's parameters
        (``dh`` has no master and stays NaN), written into the metrics tensor the readback reads."""
        p = self.model.params
        for st in self.layers:
            st.met[_MET["norm_dh"]] = float("nan")
            for name, key in (("dw_qkvg", "w_qkvg"), ("dw_o", "w_o"), ("dw_q_norm", "w_q_norm"), ("dw_k_norm", "w_k_norm")):
                st.met[_MET[f"norm_{name}"]] = p[f"{key}.{st.layer}"].grad.float().norm()

    def _grad_hash_tensors(self, st: _LayerState) -> Tuple[torch.Tensor, torch.Tensor]:
        """What ``grad_sha256`` hashes for one layer: the block's bf16 ``dw_qkvg`` / ``dw_o`` buffers, or the fp32 master gradients of
        those two weights for the ``torch`` arm (no block buffers there)."""
        if self.arm.family == "torch":
            p = self.model.params
            return p[f"w_qkvg.{st.layer}"].grad, p[f"w_o.{st.layer}"].grad
        return st.grads["dw_qkvg"], st.grads["dw_o"]

    def _clip(self, total_norm: float) -> None:
        coef = self.cfg.clip_norm / (total_norm + 1e-6)
        if coef < 1.0:
            for p in self.model.params.values():
                p.grad.mul_(coef)

    def _read_back(self) -> Tuple[List[list], Optional[List[Dict[str, float]]]]:
        """After the step's backward: ONE sync, every layer's metrics and scalar block; the gradient-scale check (asserted under
        "current", counted under "delayed") and the dP check; the next step's ``scale_dp``; the activation and gradient amax
        histories."""
        torch.cuda.synchronize()
        mets = torch.stack([st.met for st in self.layers]).tolist()
        scalars = None
        if self.arm.quantized:
            scalars = []
            for st, met in zip(self.layers, mets):
                sc = self._read_scalars(st)
                for n in self.arm.grad_scale_names:
                    what = n[len("scale_") :]
                    amax = sc["amax_" + what]
                    if amax != amax:
                        raise RuntimeError(f"step {self.step} layer {st.layer}: the published amax of {what} is NaN")
                    sat = not (amax * sc[n] <= E4M3_MAX)
                    if self.arm.grad_scaling == "current":
                        if sat:
                            raise RuntimeError(
                                f"step {self.step} layer {st.layer}: amax * {n} = {amax:.6g} * {sc[n]:.6g} > {E4M3_MAX}: the e4m3 gradient would "
                                f"saturate -- under grad_scaling='current' the kernel derives {n} from this step's amax, so this is a bug"
                            )
                    else:
                        self._grad_sat[n].append(bool(sat))  # the lagged recipe's cost (zero headroom at margin 0): reported, never asserted
                    st.grad_hist[n].append(amax)
                if self.arm.family == "fp8":
                    prod = sc["amax_dp"] * st.scale_dp_host
                    if not (prod <= E4M3_MAX) or prod != prod:
                        raise RuntimeError(
                            f"step {self.step} layer {st.layer}: amax_dp * scale_dp = {sc['amax_dp']:.6g} * {st.scale_dp_host:.6g} > {E4M3_MAX}: the e4m3 dS would saturate"
                        )
                    st.scale_dp_host = self._next_scale_dp(sc["amax_dp"])
                    st.scale_dp.fill_(st.scale_dp_host)
                    for x in _ACTS:
                        st.act_hist[x].append(met[_MET[f"amax_{x}"]])
                scalars.append(sc)
        return mets, scalars

    def _row(self, step: int, loss: float, lr: float, grad_norm: float, mets: List[list], scalars, calib_fwd: bool, wall_ms: float) -> dict:
        g, arm = self.geom, self.arm
        layers = self.layers
        row: dict = dict(
            step=step,
            tokens_seen=(step + 1) * g.tokens_per_step,
            loss=loss,
            lr=lr,
            grad_norm_total=grad_norm,
            calib_fwd=calib_fwd,
            calib_bwd=bool(self._calib_bwd),
            grad_norms={name: [m[_MET[f"norm_{name}"]] for m in mets] for name in _GRAD_NAMES},
            arm=arm.name,
            recipe=dict(
                family=arm.family,
                grad_scaling=arm.grad_scaling,
                grad_window=arm.grad_window,
                margin_log2=arm.margin_log2,
                act_window=arm.act_window,
                bwd_knobs=list(arm.bwd_knobs),
                dp_scale_rule=arm.dp_scale_rule,
                dp_margin_log2=arm.dp_margin_log2,
            ),
            geometry=dataclasses.asdict(g),
            seed=self.seed,
            data_seed=self.data_seed,
        )
        if arm.quantized:
            fields = [f.name for f in dataclasses.fields(type(layers[0].spec)) if isinstance(getattr(layers[0].spec, f.name), float)]
            row["spec"] = {f: [float(getattr(st.spec, f)) for st in layers] for f in fields}
            row["quant_scalars"] = {k: [sc[k] for sc in scalars] for k in scalars[0]}
            if arm.family == "fp8":
                # the scale_dp the logged backward RAN at (the next step's is set at readback), the activation amax of this step's
                # record and the saturation counters at this step's scales
                row["scale_dp"] = [sc["descale_dp"] and 1.0 / sc["descale_dp"] for sc in scalars]
                row["act_amax"] = {x: [m[_MET[f"amax_{x}"]] for m in mets] for x in _ACTS}
                row["sat"] = {x: [bool(m[_MET[f"amax_{x}"]] * st.act_scales[f"scale_{x}"] > E4M3_MAX) for m, st in zip(mets, layers)] for x in _ACTS}
                row["n_clip"] = {x: [int(round(m[_MET[f"n_clip_{x}"]])) for m in mets] for x in _ACTS}
            if arm.grad_scaling == "delayed":
                row["given_scales"] = {n: [st.grad_scales_host.get(n) for st in layers] for n in arm.grad_scale_names}
                row["grad_sat"] = dict(self._grad_sat)  # per gradient scale and layer: amax * scale > 448 this step (reported)
        h = hashlib.sha256()
        for st in layers:
            for ten in self._grad_hash_tensors(st):
                h.update(_tensor_bytes(ten))
        row["grad_sha256"] = h.hexdigest()
        row["grad_sha256_of"] = "master_fp32_w_qkvg_w_o" if arm.family == "torch" else "block_bf16_dw_qkvg_dw_o"
        row["row_digest"] = row_digest(row)
        row.update(
            replica=self.replica, wall_ms=wall_ms, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=socket.gethostname(), pid=os.getpid()
        )
        return row

    def run(self, steps: int, *, jsonl_path: Optional[str] = None, keep_grads_at: Tuple[int, ...] = ()) -> RunResult:
        """Train ``steps`` steps (``steps >= 1``: a run with no step has no row to digest and is refused); returns the rows (and the
        fp32 master gradients, pre-clip, of the steps in ``keep_grads_at``).  Deterministic algorithms are forced for the duration and
        restored after."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1 (a run with no step has no row to digest), got {steps}")
        self.check_environment()
        prev_det = torch.are_deterministic_algorithms_enabled()
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._run(steps, jsonl_path, set(keep_grads_at))
        finally:
            torch.use_deterministic_algorithms(prev_det)
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    def _run(self, steps: int, jsonl_path: Optional[str], keep: set) -> RunResult:
        rows: List[dict] = []
        kept: Dict[int, Dict[str, torch.Tensor]] = {}
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f = open(jsonl_path, "w") if jsonl_path else None
        params = self.model.params
        try:
            for step in range(steps):
                t0 = time.perf_counter()
                tokens = self.data.batch().to(self.device)
                inp, tgt = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
                calib_fwd = False
                if step == 0 and self._needs_activation_calibration():
                    self.calibrating = True
                    self.step = 0
                    with torch.no_grad():
                        self.model.forward(inp, tgt, self.block)
                    self._seed_activation_histories()
                    self.calibrating = False
                    calib_fwd = True
                self._begin_step(step)
                for p in params.values():
                    p.grad = None
                loss = self.model.forward(inp, tgt, self.block)
                loss.backward()
                if self.arm.family == "torch":
                    self._torch_arm_metrics()
                mets, scalars = self._read_back()
                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    raise RuntimeError(f"step {step}: non-finite loss {loss_val}")
                grad_norm = self._grad_norm()
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"step {step}: non-finite gradient norm {grad_norm}")
                for name, vals in ((n, [m[_MET[f"norm_{n}"]] for m in mets]) for n in _GRAD_NAMES):
                    if not all(math.isfinite(v) for v in vals) and not (name == "dh" and self.arm.family == "torch"):
                        raise RuntimeError(f"step {step}: non-finite block gradient {name}: {vals}")
                if step in keep:
                    kept[step] = {k: p.grad.detach().clone() for k, p in params.items()}
                self._clip(grad_norm)
                lr = self.cfg.lr_at(step, steps)
                for grp in self.opt.param_groups:
                    grp["lr"] = lr
                self.opt.step()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1e3
                row = self._row(step, loss_val, lr, grad_norm, mets, scalars, calib_fwd, wall_ms)
                rows.append(row)
                if f is not None:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    f.flush()
                self.log(
                    f"[{self.name}] step {step:5d} loss {loss_val:.5f} lr {lr:.3e} |g| {grad_norm:.4f} wall {wall_ms:8.1f} ms"
                    f"{'  (calib fwd)' if calib_fwd else ''}{f'  (calib bwd x{self._calib_bwd} per layer)' if self._calib_bwd else ''}"
                )
        finally:
            if f is not None:
                f.close()
        walls = sorted(r["wall_ms"] for r in rows[1:])
        median_ms = walls[len(walls) // 2] if walls else None
        manifest = dict(
            name=self.name,
            arm=self.arm.as_dict(),
            geometry=dataclasses.asdict(self.geom),
            config=dataclasses.asdict(self.cfg),
            data=self.data.as_dict(),
            seed=self.seed,
            replica=self.replica,
            steps=steps,
            n_rows=len(rows),
            bootstrap_rungs=self.bootstrap_rungs(),
            run_digest=run_digest(rows),
            first_row_digest=rows[0]["row_digest"] if rows else None,
            last_loss=rows[-1]["loss"] if rows else None,
            ms_per_step_median_steps_ge_1=median_ms,
            deterministic_algorithms=True,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            torch=torch.__version__,
            cudnn_module=sys.modules["cudnn"].__file__,
            device=torch.cuda.get_device_name(self.device),
            compute_capability=list(torch.cuda.get_device_capability(self.device)),
            sm_count=torch.cuda.get_device_properties(self.device).multi_processor_count,
            host=socket.gethostname(),
            pid=os.getpid(),
            started=started,
            finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return RunResult(rows=rows, manifest=manifest, kept_grads=kept, run=self)


def run_training(
    arm: Arm, geom: TrainGeometry, *, steps: int, seed: int = 0, jsonl_path: Optional[str] = None, keep_grads_at: Tuple[int, ...] = (), **kw
) -> RunResult:
    """Build a :class:`ConvergenceRun` and run it (``kw`` -> the constructor: ``data_seed``, ``cfg``, ``replica``, ``name``, ``scale_feed``, ``log``)."""
    return ConvergenceRun(arm, geom, seed=seed, **kw).run(steps, jsonl_path=jsonl_path, keep_grads_at=keep_grads_at)


def load_rows(jsonl_path: str) -> List[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scale_feed_from_rows(rows: List[dict], names: Tuple[str, ...]) -> Callable[[int, int], Dict[str, float]]:
    """A ``scale_feed`` that replays a logged run's published gradient scales (``quant_scalars``) per step and layer."""

    def feed(step: int, layer: int) -> Dict[str, float]:
        if step >= len(rows):
            raise IndexError(f"scale feed: the logged run has {len(rows)} rows, step {step} requested")
        return {n: rows[step]["quant_scalars"][n][layer] for n in names}

    return feed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _arm_from_args(a) -> Arm:
    arm = ARMS[a.arm]
    changes = {}
    if a.grad_scaling is not None:
        changes["grad_scaling"] = a.grad_scaling
    if a.grad_window is not None:
        changes["grad_window"] = a.grad_window
    if a.margin_log2 is not None:
        changes["margin_log2"] = a.margin_log2
    if a.act_window is not None:
        changes["act_window"] = a.act_window
    if a.bwd_knobs:
        changes["bwd_knobs"] = tuple(k for k in a.bwd_knobs.split(",") if k)
    if a.dp_scale_rule is not None:
        changes["dp_scale_rule"] = a.dp_scale_rule
    if a.dp_margin_log2 is not None:
        changes["dp_margin_log2"] = a.dp_margin_log2
    return arm.with_(**changes) if changes else arm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--geometry", default="smoke", choices=sorted(GEOMETRIES))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True, help="directory for <name>.jsonl and <name>.manifest.json")
    ap.add_argument("--grad-scaling", choices=("current", "delayed"), default=None)
    ap.add_argument("--grad-window", type=int, default=None)
    ap.add_argument("--margin-log2", type=int, default=None)
    ap.add_argument("--act-window", type=int, default=None)
    ap.add_argument("--bwd-knobs", default="", help="comma-separated: fuse_gate_bwd,fuse_wgrad_overlap")
    ap.add_argument("--dp-scale-rule", choices=("amax", "helper"), default=None, help="fp8: how scale_dp follows amax_dp (Arm.dp_scale_rule)")
    ap.add_argument("--dp-margin-log2", type=int, default=None, help="fp8: the octaves of headroom of the 'amax' dP rule (Arm.dp_margin_log2)")
    ap.add_argument("--replay-scales-from", default=None, help="a 'current' run's JSONL whose published gradient scales feed this 'delayed' run")
    a = ap.parse_args(argv)
    if a.steps < 1:
        ap.error(f"--steps must be >= 1 (a run with no step has no row to digest), got {a.steps}")

    import cudnn

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    _print(f"cudnn: {cudnn.__file__} | torch {torch.__version__} | {props.name} cc {torch.cuda.get_device_capability(dev)} {props.multi_processor_count} SMs")
    _print(
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} CUDNN_FRONTEND_ENABLE_FROST_ENGINES={os.environ.get('CUDNN_FRONTEND_ENABLE_FROST_ENGINES')}"
    )
    arm, geom = _arm_from_args(a), GEOMETRIES[a.geometry]
    feed = None
    if a.replay_scales_from:
        if arm.grad_scaling != "delayed":
            raise SystemExit("--replay-scales-from needs --grad-scaling delayed")
        feed = scale_feed_from_rows(load_rows(a.replay_scales_from), arm.grad_scale_names)
    os.makedirs(a.out, exist_ok=True)
    name = a.name or f"{arm.name}_r{a.replica}"
    jsonl = os.path.join(a.out, name + ".jsonl")
    _print(f"arm {arm.as_dict()} geometry {a.geometry} {dataclasses.asdict(geom)} steps {a.steps} seed {a.seed} -> {jsonl}")
    res = run_training(arm, geom, steps=a.steps, seed=a.seed, data_seed=a.data_seed, replica=a.replica, name=name, scale_feed=feed, jsonl_path=jsonl)
    with open(os.path.join(a.out, name + ".manifest.json"), "w") as f:
        json.dump(res.manifest, f, indent=1, sort_keys=True)
    m = res.manifest
    _print(f"[{name}] run_digest {m['run_digest']}  rows {m['n_rows']}  last loss {m['last_loss']:.5f}  floor {m['data']['loss_floor_nats']:.4f} nats")
    _print(f"[{name}] median ms/step over steps >= 1: {m['ms_per_step_median_steps_ge_1']} (host-bound, informational -- not a performance number)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
