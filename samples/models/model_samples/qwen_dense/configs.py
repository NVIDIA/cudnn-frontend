# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provider table and the regime table of the dense Qwen family, as data.

For every block the rows are ordered; the first row whose GPU classes, sequence range, tokens-per-sequence range and stack
contain the run serves the block.  The whole table is printed before every run.  A row with a ``fallback`` reason is a
reference chain or a multiply over a dequantised copy; ``band`` is the tie-out band of the row against the reference block
(``model_samples.tie_out.BANDS``) and is never widened in place.  The next stack adds its rows here; it removes none.

``readme_table()`` renders the tables as the family README shows them; a host test keeps the two in agreement.
"""

from __future__ import annotations

from ..blocks import GIB, ProviderRow, Regime

ANY = (("any",), (1, None), (1, None))
BOTH = ("fe-cublas", "reference")

PROVIDERS = {
    "embedding": [ProviderRow(*ANY, "torch.index_select", BOTH, serves="any")],
    "add_norm_quantize": [ProviderRow(*ANY, "torch_ref.add_norm_quantize", BOTH, serves="any", fallback="reference chain")],
    "projection.fp8.in": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", "fe-cublas", serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
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
        ),
        ProviderRow(*ANY, "torch_ref.gdn_decode_step", "reference", serves="any", fallback="reference"),
    ],
    "projection.fp8.out": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", "fe-cublas", serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "mlp.gate_up": [
        ProviderRow(
            *ANY, "cublas.bf16_dequantized+torch_ref.silu_mul_quantize", "fe-cublas", serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"
        ),
        ProviderRow(*ANY, "torch_ref.linear+torch_ref.silu_mul_quantize", "reference", serves="any", fallback="reference"),
    ],
    "mlp.down": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", "fe-cublas", serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "attention.prepare": [ProviderRow(*ANY, "torch_ref.attention_prepare", BOTH, serves="any", fallback="reference chain")],
    "attention.core": [ProviderRow(*ANY, "torch_ref.attention", BOTH, serves="any", fallback="reference chain over a plain per-layer cache")],
    "attention.gate_quantize": [ProviderRow(*ANY, "torch_ref.gate_quantize", BOTH, serves="any", fallback="reference chain")],
    "output_head": [
        ProviderRow(*ANY, "cublas.bf16_dequantized", "fe-cublas", serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "token_choice": [ProviderRow(*ANY, "torch.argmax", BOTH, serves="any")],
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


def readme_table(calls: dict) -> str:
    """The block table of the family README (markdown) for the ``fe-cublas`` stack, with ``calls`` per step of the full model."""
    from ..providers import ENTRY_NAMES

    lines = ["| block | calls per step (64 layers) | provider key | entry |", "|---|---|---|---|"]
    for block, rows in PROVIDERS.items():
        row = next(r for r in rows if "fe-cublas" in r.stacks)
        entry = ENTRY_NAMES[row.key] + (f", fallback: {row.fallback}" if row.fallback else "")
        lines.append(f"| {BLOCK_TITLES[block]} | {calls[block]} | `{row.key}` | {entry} |")
    return "\n".join(lines)


def readme_regimes() -> str:
    lines = ["| GPU classes | sequences | regime | note |", "|---|---|---|---|"]
    for r in REGIMES:
        upper = "and more" if r.sequences[1] is None else f"to {r.sequences[1]}"
        lines.append(f"| {', '.join(r.classes)} | {r.sequences[0]} {upper} | {r.name} | {r.note} |")
    return "\n".join(lines)
