# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provider table and the regime table of the dense Qwen family, as data.

For every block the rows are ordered; the first row whose GPU classes, sequence range, tokens-per-sequence range and stack
contain the run serves the block.  The whole table is printed before every run.  A row with a ``fallback`` reason is a
reference chain or a multiply over a dequantised copy; ``band`` is the tie-out band of the row against the reference block
(``model_samples.tie_out.BANDS``) and is never widened in place.  ``form`` is what the row's adapter passes to its entry: fields
of the entry's own form, which the kernel check compares with the entry's answer.  A stack adds its rows here; it removes
none.

The stacks:

* ``fe-cublas`` -- the reference path: the fused linear-attention step on its entry's own pick, every multiply on cuBLAS over
  a bf16 copy, the rest in PyTorch;
* ``native`` -- the target kernel set of this model's decode step (``kernel_census.json`` beside this file): per block the
  kernel the target launches, in the target's form.  A block whose kernel is not served here yet stays on its fallback row,
  and a run prints it as ``not yet``; how a row stands to the target is decided against the census file, not written here;
* ``fe-entries`` -- every cuDNN Frontend row of ``native``, with every block a library or a temporary kernel serves there
  on its fallback row: no package beyond cuDNN Frontend and PyTorch, so the rows of the entries stay verified wherever the
  area's GPU test runs.  Until ``native`` has such a row the two stacks are the same rows;
* ``reference`` -- every block in plain PyTorch.

``readme_table()`` renders the tables as the family README shows them; a host test keeps the two in agreement.
"""

from __future__ import annotations

from ..blocks import GIB, ProviderRow, Regime
from ..census import form_text

ANY = (("any",), (1, None), (1, None))
BOTH = ("fe-cublas", "reference")
#: the stacks that are held against the family's census of the target kernel set; a run of one always checks its kernels
TARGET_STACKS = ("native", "fe-entries")
#: every stack with provider rows (all but the reference), and every stack
PROVIDER_STACKS = ("fe-cublas", *TARGET_STACKS)
EVERY = (*BOTH, *TARGET_STACKS)

PROVIDERS = {
    "embedding": [ProviderRow(*ANY, "torch.index_select", EVERY, serves="any")],
    "add_norm_quantize": [ProviderRow(*ANY, "torch_ref.add_norm_quantize", EVERY, serves="any", fallback="reference chain")],
    "projection.fp8.in": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "linear_attention.step": [
        ProviderRow(
            ("cc10.x", "cc12.x"),
            (1, 65535),
            (1, 1),
            "fe.gdn_decode_fused",
            "fe-cublas",
            serves="rows 1-65,535",
            measured=("cc 10.0, 1 row",),
            band="gdn_step",
            batch_invariant=("cc10.0", "cc10.3", "cc12.0"),
            # the entry's own pick per rows, launched without a wait on its predecessor and without an early release of its
            # successor.  Both are what the entry does when nothing is passed; the row says so, and the kernel check holds it
            form=(("PDL", False), ("PDL_RELEASE", False)),
        ),
        ProviderRow(
            ("cc10.x", "cc12.x"),
            (1, 65535),
            (1, 1),
            "fe.gdn_decode_fused",
            TARGET_STACKS,
            serves="rows 1-65,535",
            measured=("cc 10.0, 1 row",),
            band="gdn_step",
            batch_invariant=("cc10.0",),  # the class this form has run on; the row above lists what its form has
            # the target's form: the entry's own pick per rows, launched as a dependent of its predecessor (it waits for it
            # only before it reads the in-projection's rows) and releasing its successor at entry.  The launch before it must
            # not write this layer's state: in this stack that launch is the in-projection's, as in the target
            form=(("PDL", True), ("PDL_RELEASE", True)),
        ),
        ProviderRow(*ANY, "torch_ref.gdn_decode_step", "reference", serves="any", fallback="reference"),
    ],
    "projection.fp8.out": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "mlp.gate_up": [
        ProviderRow(
            *ANY,
            "cublas.bf16_dequantized+torch_ref.silu_mul_quantize",
            PROVIDER_STACKS,
            serves="any",
            fallback="bf16 copy of 4-bit weights",
            band="bf16_rounding",
        ),
        ProviderRow(*ANY, "torch_ref.linear+torch_ref.silu_mul_quantize", "reference", serves="any", fallback="reference"),
    ],
    "mlp.down": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "attention.prepare": [ProviderRow(*ANY, "torch_ref.attention_prepare", EVERY, serves="any", fallback="reference chain")],
    "attention.core": [ProviderRow(*ANY, "torch_ref.attention", EVERY, serves="any", fallback="reference chain over a plain per-layer cache")],
    "attention.gate_quantize": [
        ProviderRow(
            ("cc10.x",),
            (1, 160),
            (1, None),
            "fe.attention_gate_quant_decode",
            TARGET_STACKS,
            serves="rows 1-160",
            band="gate_fp8",
            batch_invariant=("cc10.0",),
            # the target's form: 256 threads per thread block, an ordinary launch that releases its successor at entry.  The
            # launch after it in this stack carries no dependent-launch attribute, so the release changes nothing here
            form=(("THREADS", 256), ("PDL_RELEASE", True)),
            rows=(1, 160),
        ),
        ProviderRow(*ANY, "torch_ref.gate_quantize", EVERY, serves="any", fallback="reference chain"),
    ],
    "output_head": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "token_choice": [ProviderRow(*ANY, "torch.argmax", EVERY, serves="any")],
}

REGIMES = [
    Regime(
        ("any",),
        (1, None),
        "launch-bound",
        "every block of these stacks is launched from Python, one launch or more per block; no byte-bound form is declared, "
        "and the sequence count at which one would win is not measured on any class",
    ),
]

#: how each block is named in text
BLOCK_TITLES = {
    "embedding": "token embedding lookup",
    "add_norm_quantize": "residual add + RMSNorm + quantise",
    "projection.fp8.in": "8-bit in-projection (linear attention QKVZ / attention QKV)",
    "linear_attention.step": "linear-attention step",
    "projection.fp8.out": "8-bit out-projection",
    "mlp.gate_up": "MLP gate / up + SiLU x up + quantise",
    "mlp.down": "MLP down",
    "attention.prepare": "attention preparation",
    "attention.core": "attention core",
    "attention.gate_quantize": "gate + quantise",
    "output_head": "output head",
    "token_choice": "token choice",
}

#: device memory the first launch of the fused linear-attention step reserves outside PyTorch's allocator, per GPU class: an
#: allowance for the memory sentence of the plan, which is printed before anything is launched.  The run then measures the
#: launch and prints what it took (this sample on one compute capability 12.0 GPU: 3,124 MiB at the first form, none after)
FIRST_LAUNCH_RESERVE = {"cc12.x": 3 * GIB}


#: the logits band of the token check, declared here and not taken from a run: the logits of the provider stack and of the
#: reference stack, each run end to end on its own state, may differ by this many bf16 units of the run's largest reference
#: logit.  It is declared for the reduced seeded model (measured by this sample: 1.0 unit with two layers, 1.2 with four, 1.4
#: with eight); a real checkpoint declares its own
LOGITS_BAND_UNITS = 4.0
#: the logits band for a checkpoint read from local disk (the published depth, trained weights): none.  The token line of
#: such a run is REPORTED and gated on finite logits only (``tie_out.token_check`` with ``band_units=None``).
#: The model quantises its MLP inputs to 4 bits per element and its projection inputs to 8, so a difference of one bf16
#: rounding in a multiply becomes a different code in the next quantiser, 64 layers deep.  Measured by this sample on the
#: published 27B-class checkpoint: two REFERENCE stacks that differ only in their accumulation precision (fp64 against IEEE
#: fp32) were up to 52 units (13.0 logits) apart over 80 steps and chose another greedy token at 13 of them; the provider
#: stack was up to 54 units (13.5 logits) from the fp64 reference over the same steps.  In three 16-step windows of the
#: kind a run checks, the provider stack was 8.7 to 10.3 logits from the reference and a second reference 7.5 to 11.4.
#: No band separates a correct stack from a failed one here.  In bf16 units of the largest logit a band is 8, 16 or 32
#: logits wide by that logit's binade, and the distance of two correct stacks does not follow the binade: 128 units --
#: beyond which a stack that returns zeros passes -- are 8 logits where a window's largest logit is below 16, less than
#: two correct stacks were apart.  And a greedy token is decidable only where its margin exceeds twice the band, which no
#: step of those runs reached.  What verifies a provider on a checkpoint is the per-block check, teacher-forced; the run
#: prints the distance, the count of equal greedy tokens and the same for a second reference, to be read side by side
CHECKPOINT_LOGITS_BAND_UNITS = None


def readme_table(calls: dict, stack: str = "fe-cublas", cc=(10, 0)) -> str:
    """The block table of the family README (markdown) for ``stack`` on compute capability ``cc``, with ``calls`` per step of
    the full model: per block the first row of the stack that serves the class, with the form its entry is called in."""
    from ..providers import ENTRY_NAMES

    lines = ["| block | calls per step (64 layers) | provider key | entry [form] |", "|---|---|---|---|"]
    for block, rows in PROVIDERS.items():
        row = next(r for r in rows if stack in r.stacks and r.serves_class(cc))
        entry = ENTRY_NAMES[row.key] + form_text(row.form) + (f", fallback: {row.fallback}" if row.fallback else "")
        lines.append(f"| {BLOCK_TITLES[block]} | {calls[block]} | `{row.key}` | {entry} |")
    return "\n".join(lines)


def readme_regimes() -> str:
    lines = ["| GPU classes | sequences | regime | note |", "|---|---|---|---|"]
    for r in REGIMES:
        upper = "and more" if r.sequences[1] is None else f"to {r.sequences[1]}"
        lines.append(f"| {', '.join(r.classes)} | {r.sequences[0]} {upper} | {r.name} | {r.note} |")
    return "\n".join(lines)
