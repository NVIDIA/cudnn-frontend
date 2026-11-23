# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of a checkpoint's drafter (``samples/models/model_samples/qwen_dense/drafter.py``): its loader with the
8-bit quantisation it applies when it loads (``weights.quantize_fp8_per_channel``), and its draft pass on reference blocks.

* The quantisation, against the format definition: a quantiser written here from the E4M3 layout (a sign bit, four exponent
  bits with bias 7, three mantissa bits, subnormals, no infinity; nearest value, a tie to the even code, saturating), with
  the scale and the quotient as single fp32 operations.  Nothing compares the loader with a kernel that reads its codes,
  so this test is the pin.
* The draft vocabulary: each id once, in ascending order, inside the model's vocabulary, with what was dropped counted;
  every file the reader refuses, each with its own sentence; the README's recipe, run as it is written.
* A checkpoint directory generated here, with a drafter: the manifest from the headers alone; the rows cut from the output
  head against the bytes of the shard file, read here by the header's offsets and not through the reader under test; the
  8-bit weights and the order of the parts of the merged ones; every drafter the loader declines, by name.

* The draft pass.  Every block of it is a reference chain in every stack, so no tie-out judges it: each new function is
  held here to a version written from its definition -- the quantiser of a row of a step (a scale per row, a
  multiplication by the reciprocal of the scale), the joint quantiser of the two normed halves in front of the first
  multiply, the gate with one rounding and SiLU x up with two, the token choice over listed rows.  A pass over several
  rows is held to that many passes of one row, rows and cache byte for byte; what each pass of a step is handed, by a
  recording stand-in for the blocks; and the model's stored forms and the drafter with its copies, loaded together from
  a checkpoint generated here.  The blocks themselves, as a plan builds them, are held to the functions they stand
  for, and what each declares about its operands is read through the tap of the tie-out, on each sequence alone.

No real checkpoint, tokenizer or template file is committed.  No GPU, no weights."""

import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
README = MODELS / "qwen_dense" / "README.md"
MODEL, DRAFTER = "model.language_model.", "mtp."  # the two prefixes of the generated checkpoint, as a published one has them
#: this family's names of the drafter's own tensors, in the order of its manifest; a name on disk is the prefix + the name
OWN = (
    "pre_fc_norm_embedding",
    "pre_fc_norm_hidden",
    "layers.0.input_layernorm",
    "layers.0.self_attn.q_norm",
    "layers.0.self_attn.k_norm",
    "layers.0.post_attention_layernorm",
    "norm",
    "fc",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
)


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


# --------------------------------------------------------------------------- the 8-bit format, written from its definition


def _e4m3_value(byte: int) -> float:
    """The value of one E4M3 code byte from the format: a sign bit, four exponent bits with bias 7, three mantissa bits,
    subnormals at exponent 0, no infinity."""
    sign, exponent, mantissa = byte >> 7, (byte >> 3) & 0xF, byte & 0x7
    value = (mantissa / 8.0) * 2.0**-6 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if sign else value


#: the values of the codes 0x00 .. 0x7E, in ascending order: 0x7E is the largest, 448; 0x7F is not a number
VALUES = [_e4m3_value(byte) for byte in range(0x7F)]


def _fp32(value: float) -> float:
    """A double rounded to the nearest fp32.  A quotient of two fp32 values formed in double and rounded once more is the
    fp32 quotient: 53 bits are enough for that second rounding to change nothing."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _code(quotient: float) -> int:
    """The E4M3 code of a value: the nearest of the format's values, a tie to the even code, beyond +-448 the largest; the
    sign bit is the value's, also where it rounds to zero."""
    magnitude = min(abs(quotient), 448.0)
    code = min(range(0x7F), key=lambda c: (abs(VALUES[c] - magnitude), c % 2))
    return code | (0x80 if math.copysign(1.0, quotient) < 0 else 0)


def _per_channel(rows) -> tuple:
    """``(codes, scales)`` of a weight given as rows of numbers: per row ``scale = fp32(largest magnitude / 448)``, not below
    ``fp32(1 / (448 * 512))``, and ``code = e4m3(fp32(value / scale))``."""
    floor = _fp32(1.0 / (448.0 * 512.0))
    codes, scales = [], []
    for row in rows:
        scale = max(_fp32(max(abs(value) for value in row) / 448.0), floor)
        scales.append(scale)
        codes.append([_code(_fp32(value / scale)) for value in row])
    return codes, scales


def _per_row(rows) -> tuple:
    """``(codes, scales)`` of the rows of a step given as rows of numbers, each quantised at a scale of its own: per row
    ``scale = fp32(largest magnitude / 448)``, not below ``fp32(1 / (448 * 512))``, and ``code = e4m3(fp32(value * fp32(1 /
    scale)))``: the reciprocal of the scale and the product are fp32 operations."""
    floor = _fp32(1.0 / (448.0 * 512.0))
    codes, scales = [], []
    for row in rows:
        scale = max(_fp32(max(abs(value) for value in row) / 448.0), floor)
        reciprocal = _fp32(1.0 / scale)
        scales.append(scale)
        codes.append([_code(_fp32(value * reciprocal)) for value in row])
    return codes, scales


def _bf16_scaled(value: float) -> tuple:
    """``(significand, exponent)`` of a positive double with the significand scaled to [128, 256): bf16 has 8 bits of it."""
    mantissa, exponent = math.frexp(value)
    return mantissa * 256.0, exponent


def _bf16(value: float) -> float:
    """The nearest bf16 value of a double: a significand of 8 bits, a tie to the even one.  For values in the normal
    range of the format (nothing here is subnormal or overflows)."""
    if value == 0.0:
        return value
    scaled, exponent = _bf16_scaled(abs(value))
    whole = math.floor(scaled)
    rest = scaled - whole
    whole += 1 if rest > 0.5 or (rest == 0.5 and whole % 2 == 1) else 0
    return math.copysign(whole / 256.0 * 2.0**exponent, value)


def _cast(value: float) -> float:
    """A double as the reference blocks round it to bf16: through fp32, as PyTorch's cast of an fp64 tensor does."""
    return _bf16(_fp32(value))


def _clear_of_a_midpoint(value: float, relative: float = 2.0**-20) -> bool:
    """Whether a relative error of up to ``relative`` cannot move the bf16 rounding of ``value``: its distance from the
    midpoint of the two bf16 values around it is more than that error."""
    if value == 0.0:
        return True
    scaled, _ = _bf16_scaled(abs(value))
    return abs(scaled - math.floor(scaled) - 0.5) > 256.0 * relative


def test_the_e4m3_values_of_this_file_are_the_formats():
    """The table the quantiser below rests on, against values that are stated, and against PyTorch's own decode of every
    code that is a number."""
    assert VALUES[:3] == [0.0, 2.0**-9, 2.0**-8] and VALUES[8] == 2.0**-6 and VALUES[0x38] == 1.0 and VALUES[0x7E] == 448.0
    assert all(low < high for low, high in zip(VALUES, VALUES[1:])), "ascending with the code: the neighbours of a value are the codes beside it"
    decoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().tolist()
    assert [decoded[byte] for byte in range(0x7F)] == VALUES and [decoded[byte] for byte in range(0x80, 0xFF)] == [-value for value in VALUES]
    assert math.isnan(decoded[0x7F]) and math.isnan(decoded[0xFF])
    assert [_code(v) for v in (21.0, 23.0, 449.0, -1e9, -0.0, 0.0, 2.0**-10, -(2.0**-10), 0.75 * 2.0**-9)] == [0x5A, 0x5C, 0x7E, 0xFE, 0x80, 0, 0, 0x80, 1]
    # ... and the bf16 rounding of this file: stated values, and PyTorch's own cast of fp32 values (one rounding)
    assert [_bf16(v) for v in (1.0, 1.00390625, 1.001953125, 1.005859375, -3.0078125, 0.1, 0.0)] == [1.0, 1.0, 1.0, 1.0078125, -3.0, 0.10009765625, 0.0]
    generator = torch.Generator().manual_seed(3)
    drawn = torch.cat([torch.randn(2048, generator=generator) * scale for scale in (1e-3, 1.0, 300.0)])
    assert drawn.to(torch.bfloat16).double().tolist() == [_bf16(value) for value in drawn.double().tolist()]
    assert _clear_of_a_midpoint(1.0) and not _clear_of_a_midpoint(1.00390625) and not _clear_of_a_midpoint(1.00390625 * (1 + 2.0**-21))


def test_the_per_channel_8_bit_quantisation_against_the_format_definition():
    from model_samples import weights
    from model_samples.weights import ScaledTensor, dequantize, quantize_fp8_per_channel

    # Stated rows; every value is exact in bf16.  What each row holds:
    #   0  a largest element of 13/16: the scale is not a power of two.  Elements 1 and 2 are chosen so that their quotient
    #      by that scale is, in fp32, exactly the midpoint of two codes (1.3125 and 2.625): the division rounds to the even
    #      code; a multiplication by the reciprocal of the scale lands just above the midpoint and gives the next code
    #   1  all zero: the scale is the floor, the codes are 0
    #   2  a largest element below 1/512: the scale stays at the floor, and the codes are those of the quotient by the floor
    #   3  a largest element of 1.75, so the scale is 1/256 exactly; 21/256 and 23/256 are midpoints: one tie goes down,
    #      one goes up, both to the even code; the largest elements are +-448, the largest codes
    #   4  the largest element is negative; a negative zero keeps its sign bit; a value far below the smallest code is 0
    rows = [
        [0.8125, 39.0 / 16384.0, 39.0 / 8192.0, -0.40625],
        [0.0, 0.0, 0.0, 0.0],
        [2.0**-10, -(2.0**-11), 2.0**-13, 0.0],
        [1.75, 21.0 / 256.0, 23.0 / 256.0, -1.75],
        [-0.5, 0.25, -0.0, 2.0**-20],
    ]
    x = torch.tensor(rows, dtype=torch.bfloat16)
    assert x.double().tolist() == rows, "the stated values are bf16 values"
    stored = quantize_fp8_per_channel(x)
    assert isinstance(stored, ScaledTensor) and stored.codes.dtype == torch.uint8 and tuple(stored.codes.shape) == (5, 4)
    assert stored.scale.dtype == torch.float32 and tuple(stored.scale.shape) == (5,), "one fp32 scale per row"
    codes, scales = _per_channel(rows)
    assert stored.codes.tolist() == codes and stored.scale.tolist() == scales, "byte for byte the format's quantiser, and its scales bit for bit"
    floor = _fp32(1.0 / (448.0 * 512.0))
    assert scales == [_fp32(0.8125 / 448.0), floor, floor, 1.0 / 256.0, _fp32(0.5 / 448.0)] and weights.PER_CHANNEL_FLOOR == 512
    assert codes == [
        [0x7E, 0x3A, 0x42, 0xF6],
        [0, 0, 0, 0],
        [0x76, 0xEE, 0x5E, 0],
        [0x7E, 0x5A, 0x5C, 0xFE],
        [0xFE, 0x76, 0x80, 0],
    ], "stated bytes: midpoints to the even code, +-448 as 0x7E / 0xFE, never the NaN code"
    scale = scales[0]
    assert [_fp32(value / scale) for value in rows[0][1:3]] == [1.3125, 2.625], "the two quotients of row 0 are midpoints of two codes"
    reciprocal = [_code(_fp32(value * _fp32(1.0 / scale))) for value in rows[0][1:3]]
    assert reciprocal == [0x3B, 0x43], "what a multiplication by the reciprocal gives for the same two elements: the code above"

    # random rows, among them one below the floor and one of zeros: every byte and every scale
    generator = torch.Generator().manual_seed(11)
    y = (torch.randn((48, 80), generator=generator) * 0.05).to(torch.bfloat16)
    y[5] = y[5] * 2.0**-9
    y[7] = 0.0
    y[9, 3] = -4.0  # the largest element of its row, negative
    stored = quantize_fp8_per_channel(y)
    codes, scales = _per_channel(y.double().tolist())
    assert stored.codes.tolist() == codes and stored.scale.tolist() == scales
    assert scales[5] == scales[7] == floor and int(stored.codes[9, 3]) == 0xFE and 0x7F not in set(sum(codes, [])) and 0xFF not in set(sum(codes, []))
    assert sum(1 for row in codes for byte in row if byte & 0x7F == 0x7E) >= 46, "every row above the floor reaches the largest code at its largest element"

    # a row is quantised by itself: the weights of one multiply can be concatenated before or after, in any chunking
    first, second = y[:13], y[13:]
    parts = [quantize_fp8_per_channel(first), quantize_fp8_per_channel(second)]
    assert torch.equal(torch.cat([part.codes for part in parts]), stored.codes) and torch.equal(torch.cat([part.scale for part in parts]), stored.scale)
    for chunk in (1, 7, 48, 4096):
        again = quantize_fp8_per_channel(y, rows_per_chunk=chunk)
        assert torch.equal(again.codes, stored.codes) and torch.equal(again.scale, stored.scale), chunk
    alone = quantize_fp8_per_channel(y[20:21])
    assert torch.equal(alone.codes, stored.codes[20:21]) and torch.equal(alone.scale, stored.scale[20:21])

    # what a multiply over a copy reads: the code values, exact in bf16, and the scale of every row
    copy = dequantize(stored)
    assert copy.data.dtype == torch.bfloat16 and copy.scale is stored.scale
    values = torch.tensor([0.0 if byte & 0x7F == 0x7F else _e4m3_value(byte) for byte in range(256)], dtype=torch.float64)
    assert torch.equal(copy.data.double(), values[stored.codes.long()])
    live = torch.tensor([row not in (5, 7) for row in range(48)])
    step = torch.pow(2.0, torch.floor(torch.log2(copy.data.double().abs().clamp_min(2.0**-6))) - 3)
    error = (copy.data.double() * stored.scale.double().unsqueeze(1) - y.double()).abs()
    assert bool((error[live] <= 0.5 * (step * stored.scale.double().unsqueeze(1))[live] * (1 + 1e-5)).all()), "within half a code step of the weight"

    # what the quantiser refuses: a tensor without rows and columns, and a row with a value that is not finite
    for wrong in (torch.zeros(8, dtype=torch.bfloat16), torch.zeros((2, 3, 4), dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match=re.escape(f"the per-channel 8-bit quantisation takes a weight [out, in]; got {list(wrong.shape)}")):
            quantize_fp8_per_channel(wrong)
    for value, row in ((float("nan"), 3), (float("inf"), 0), (float("-inf"), 47)):
        spoilt = y.clone()
        spoilt[row, 2] = value
        for chunk in (2048, 2):
            with pytest.raises(ValueError, match=rf"^row {row} of the weight holds a value that is not finite: nothing is quantised$"):
                quantize_fp8_per_channel(spoilt, rows_per_chunk=chunk)


# --------------------------------------------------------------------------- the draft vocabulary


def test_a_draft_vocabulary_is_read_once_each_in_order_and_inside_the_vocabulary(tmp_path):
    from model_samples.qwen_dense.drafter import DraftVocab

    def written(name, content, as_json=True):
        path = tmp_path / name
        path.write_text(json.dumps(content) if as_json else content)
        return str(path)

    listed = DraftVocab.read(written("ids.json", [7, 3, 3, 271, 272, -1, 0, 1000000, 7]), 272)
    assert listed.ids == (0, 3, 7, 271), "each id once, ascending, and only ids of the model: 272 is one past its last"
    assert (listed.vocab, listed.source, listed.listed, listed.repeated, listed.outside) == (272, "ids.json", 9, 2, 3)
    said = "draft vocabulary ids.json: 4 ids of the model's 272, in ascending order (9 entries listed; dropped: 2 listed before, 3 outside the vocabulary)"
    assert listed.text() == said and str(tmp_path) not in listed.text()
    clean = DraftVocab.read(written("clean.json", [5, 1, 2]), 272)
    assert (
        clean.ids == (1, 2, 5) and clean.text() == "draft vocabulary clean.json: 3 ids of the model's 272, in ascending order (3 entries listed, none dropped)"
    )
    twice = DraftVocab.read(written("twice.json", [4, 4]), 272)
    assert twice.ids == (4,) and twice.text().endswith("(2 entries listed; dropped: 1 listed before)")
    beyond = DraftVocab.read(written("beyond.json", [4, 300]), 272)
    assert beyond.ids == (4,) and beyond.text().endswith("(2 entries listed; dropped: 1 outside the vocabulary)")
    last, alone = DraftVocab.read(written("last.json", [271]), 271 + 1), DraftVocab.read(written("alone.json", [0]), 1)
    assert last.ids == (271,) and alone.ids == (0,), "the last id of a model, and the one id of a model that has one"

    (tmp_path / "folder.json").mkdir()
    (tmp_path / "bytes.json").write_bytes(b"[1, 2\xff]")
    for path, sentence in (
        (str(tmp_path / "nowhere.json"), "draft vocabulary nowhere.json: cannot be read (No such file or directory)"),
        (str(tmp_path / "folder.json"), "draft vocabulary folder.json: cannot be read (Is a directory)"),
        (written("cut.json", "[1, 2", as_json=False), "draft vocabulary cut.json: not a JSON file ("),
        (str(tmp_path / "bytes.json"), "draft vocabulary bytes.json: not a JSON file ("),
        (written("object.json", {"ids": [1, 2]}), "draft vocabulary object.json: holds an object where a JSON list of token ids is expected"),
        (written("number.json", 7), "draft vocabulary number.json: holds a number where a JSON list of token ids is expected"),
        (written("fraction.json", 7.5), "draft vocabulary fraction.json: holds a number where a JSON list of token ids is expected"),
        (written("string.json", "1, 2"), "draft vocabulary string.json: holds a string where a JSON list of token ids is expected"),
        (written("null.json", None), "draft vocabulary null.json: holds null where a JSON list of token ids is expected"),
        (written("true.json", True), "draft vocabulary true.json: holds true or false where a JSON list of token ids is expected"),
        (written("text.json", [1, "12"]), "draft vocabulary text.json: entry 1 is '12' where a token id (a whole number) is expected"),
        (written("float.json", [3.0, 4]), "draft vocabulary float.json: entry 0 is 3.0 where a token id (a whole number) is expected"),
        (written("bool.json", [1, 2, True]), "draft vocabulary bool.json: entry 2 is True where a token id (a whole number) is expected"),
        (written("none.json", [None]), "draft vocabulary none.json: entry 0 is None where a token id (a whole number) is expected"),
        (written("nested.json", [1, [2, 3]]), "draft vocabulary nested.json: entry 1 is [2, 3] where a token id (a whole number) is expected"),
        (written("long.json", ["x" * 100]), "draft vocabulary long.json: entry 0 is '" + "x" * 39 + " where a token id (a whole number) is expected"),
        (written("empty.json", []), "draft vocabulary empty.json: none of its 0 entries is a token id of this model (0 to 271)"),
        (written("outside.json", [272, -5]), "draft vocabulary outside.json: none of its 2 entries is a token id of this model (0 to 271)"),
    ):
        with pytest.raises(ValueError) as refusal:
            DraftVocab.read(path, 272)
        text = str(refusal.value)
        assert (text.startswith(sentence) if sentence.endswith("(") else text == sentence) and str(tmp_path) not in text, text

    # the list of the seeded model: drawn from a seed, as many ids as asked for (a quarter of the vocabulary unless said)
    drawn = DraftVocab.seeded(272)
    assert len(drawn.ids) == 68 and all(low < high for low, high in zip(drawn.ids, drawn.ids[1:])) and 0 <= drawn.ids[0] and drawn.ids[-1] < 272
    assert drawn == DraftVocab.seeded(272, seed=0) and drawn.ids != DraftVocab.seeded(272, seed=1).ids
    assert drawn.text() == "draft vocabulary seeded list (seed 0): 68 ids of the model's 272, in ascending order (68 entries listed, none dropped)"
    assert DraftVocab.seeded(272, rows=272, seed=3).ids == tuple(range(272)) and len(DraftVocab.seeded(272, rows=1).ids) == 1
    assert set(DraftVocab.seeded(272, rows=5, seed=2).ids) != set(DraftVocab.seeded(272, rows=5, seed=4).ids)
    for rows in (0, 273, -1, True, 2.5, "4"):
        with pytest.raises(ValueError, match=re.escape(f"a seeded draft vocabulary holds between 1 and 272 ids; got {rows!r}")):
            DraftVocab.seeded(272, rows=rows)


def _readme_section() -> str:
    """The family README's section on the drafter, as it is written there."""
    return README.read_text().split("\n## The checkpoint's drafter\n", 1)[1].split("\n## ", 1)[0]


def _readme_blocks() -> tuple:
    """The three fenced ``python`` blocks of the README's section on the drafter, as they are written there: the call of
    the loader, the recipe for a draft vocabulary, and a draft step behind the loader's call."""
    blocks = re.findall(r"```python\n(.*?)```", _readme_section(), flags=re.S)
    assert len(blocks) == 3 and "QwenDenseDrafter.describe(" in blocks[0] and "--dump-tokens" in blocks[1], "the call, then the recipe"
    assert "drafter.draft(" in blocks[2] and "QwenDenseDrafter.describe(" not in blocks[2], "then a draft step, on what the call loaded"
    return tuple(blocks)


def _scripts(monkeypatch) -> tuple:
    """The two scripts of the family as modules, ``(00_decode, 02_speculative_decode)``, for their own functions; what loading
    them changes in the process leaves with the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # a script puts samples/models and its own folder on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    loaded = []
    for script in ("00_decode.py", "02_speculative_decode.py"):
        spec = importlib.util.spec_from_file_location(f"model_sample_qwen_dense_{script[:2]}_for_the_recipe", MODELS / "qwen_dense" / script)
        loaded.append(importlib.util.module_from_spec(spec))
        spec.loader.exec_module(loaded[-1])
    return tuple(loaded)


def _takes(script, flag: str) -> bool:
    """Whether a script takes ``flag`` on a run with a checkpoint: it parses the flag with a value and does not end in its
    usage error.  A flag that is declared to be refused by a sentence is not taken; neither is one the script does not have."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            script._arguments(["--model", "d", flag, "value"])
        except SystemExit:
            return False
    return True


def test_the_readmes_recipe_makes_a_draft_vocabulary_the_loader_reads(tmp_path, monkeypatch):
    """The recipe is run as it is written in the family README, on files the sample's own ``--dump-tokens`` writer wrote:
    every id the model read or wrote is listed, the rest of the list is the lowest ids not among them, and the loader
    reads the list without dropping an entry.  The same lines with fewer rows than there are ids keep the most frequent."""
    from model_samples.qwen_dense.drafter import RECIPE, DraftVocab

    assert RECIPE == 'samples/models/qwen_dense/README.md, "The checkpoint\'s drafter"' and "--dump-tokens FILE" in README.read_text()
    sample, speculative = _scripts(monkeypatch)
    # a flag the section names is a flag a script of this folder TAKES.  A script that declares a flag to refuse it holds the flag's
    # string as well: each script is asked, with the flag and a value, whether it ends in its usage error
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*[a-z]", _readme_section()))
    assert {"--model", "--dump-tokens", "--draft-vocab", "--speculative-config"} <= flags
    assert [flag for flag in sorted(flags) if not (_takes(sample, flag) or _takes(speculative, flag))] == []
    assert (_takes(sample, "--draft-vocab"), _takes(speculative, "--draft-vocab")) == (False, True), "declared by both, taken by one"
    assert (_takes(sample, "--speculative-config"), _takes(speculative, "--speculative-config")) == (False, True)
    assert (_takes(sample, "--dump-census"), _takes(speculative, "--dump-census")) == (True, False) and not _takes(sample, "--no-such-flag")
    assert "`draft_vocab=`" in _readme_section() and "`02_speculative_decode.py` takes the list as `--draft-vocab FILE`" in _readme_section()
    monkeypatch.chdir(tmp_path)
    one = [{"sequences": 1, "prompt": 0, "prompt_ids": [[248045, 846, 198, 70000]], "ids": [[70000, 70000, 12, 90000, 248046]]}]
    two = [
        {"sequences": 2, "prompt": 0, "prompt_ids": [[248045, 846], [248045, 9]], "ids": [[70000, 3], [200000, 12]]},
        {"sequences": 1, "prompt": 1, "prompt_ids": [[248045, 846, 5]], "ids": [[70000]]},
    ]
    refused = (sample._dump("ids_1.json", "fe-cublas", one), sample._dump("ids_2.json", "native", two))  # written before anything is asserted
    assert refused == ("", ""), "the writer returns what went wrong: nothing, twice"
    seen = {248045, 846, 198, 70000, 12, 90000, 248046, 9, 3, 200000, 5}

    recipe = _readme_blocks()[1]
    assert 'dumps = ["ids_1.json", "ids_2.json"]' in recipe and recipe.count("rows = 65536") == 1, "the two lines a reader edits"
    exec(compile(recipe, "the README's recipe", "exec"), {})
    listed = json.loads((tmp_path / "draft_vocab.json").read_text())
    assert listed == sorted(listed) and len(listed) == len(set(listed)) == 65536 and seen <= set(listed)
    rest = sorted(set(listed) - seen)
    assert rest == [token for token in range(65536) if token not in seen][: 65536 - len(seen)], "then the lowest ids that the dumps do not hold"
    vocabulary = DraftVocab.read("draft_vocab.json", 248320)
    assert vocabulary.ids == tuple(listed) and (vocabulary.listed, vocabulary.repeated, vocabulary.outside) == (65536, 0, 0)
    small = DraftVocab.read("draft_vocab.json", 15520)  # the same list for a model with fewer ids: the ids it has
    assert small.ids == tuple(range(15520)) and small.outside == 65536 - 15520

    exec(compile(recipe.replace("rows = 65536", "rows = 3"), "the README's recipe with 3 rows", "exec"), {})
    assert json.loads((tmp_path / "draft_vocab.json").read_text()) == [846, 70000, 248045], "the three ids the dumps hold most often: 3, 5 and 4 times"


def test_the_readmes_recipe_reads_the_ids_of_both_scripts(tmp_path, monkeypatch):
    """The recipe as it is written, on one file the plain script's ``--dump-tokens`` writer wrote and one the speculative
    script's wrote: the second holds the kept count of every step beside the same ``prompt_ids`` and ``ids``, and the
    recipe lists what both runs read and wrote."""
    sample, speculative = _scripts(monkeypatch)
    monkeypatch.chdir(tmp_path)
    plain = [{"sequences": 1, "prompt": 0, "prompt_ids": [[248045, 846, 198]], "ids": [[70000, 12, 248046]]}]
    drafted = [{"sequences": 1, "prompt": 0, "prompt_ids": [[248045, 9, 198]], "ids": [[70000, 3, 90000, 5, 248046]], "kept": [2, 0]}]
    assert (sample._dump("ids_1.json", "native", plain), speculative._dump("ids_2.json", "native", drafted, draft_tokens=5)) == ("", "")
    written = json.loads((tmp_path / "ids_2.json").read_text())
    assert (written["sample"], written["draft_tokens"], written["greedy"]) == ("qwen_dense/02_speculative_decode", 5, True)
    assert written["runs"] == drafted and json.loads((tmp_path / "ids_1.json").read_text())["sample"] == "qwen_dense/00_decode"
    recipe = _readme_blocks()[1]
    exec(compile(recipe.replace("rows = 65536", "rows = 12"), "the README's recipe on the files of both scripts", "exec"), {})
    listed = json.loads((tmp_path / "draft_vocab.json").read_text())
    seen = {248045, 846, 198, 70000, 12, 248046, 9, 3, 90000, 5}
    assert set(listed) == seen | {0, 1} and listed == sorted(listed), "the ids both runs read and wrote, then the lowest ids not among them"
    assert "The files of both scripts of this folder are read alike" in " ".join(_readme_section().split())


# --------------------------------------------------------------------------- a checkpoint on local disk, with a drafter


def _tiny_model():
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=("linear_attention", "full_attention"),
        hidden=64,
        vocab=272,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=2,
        attn_kv_heads=1,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _config(geometry) -> dict:
    """The ``config.json`` of a checkpoint of this geometry with a drafter, with the keys a published checkpoint carries."""
    text = {
        "model_type": "qwen3_5_text",
        "layer_types": list(geometry.layer_kinds),
        "hidden_size": geometry.hidden,
        "vocab_size": geometry.vocab,
        "intermediate_size": geometry.mlp,
        "linear_num_key_heads": geometry.la_key_heads,
        "linear_num_value_heads": geometry.la_value_heads,
        "linear_key_head_dim": geometry.la_head_dim,
        "linear_value_head_dim": geometry.la_head_dim,
        "linear_conv_kernel_dim": geometry.la_conv_width,
        "num_attention_heads": geometry.attn_heads,
        "num_key_value_heads": geometry.attn_kv_heads,
        "head_dim": geometry.attn_head_dim,
        "rope_parameters": {"partial_rotary_factor": geometry.rotary_dim / geometry.attn_head_dim, "rope_theta": geometry.rope_theta},
        "rms_norm_eps": geometry.norm_eps,
        "attn_output_gate": True,
        "hidden_act": "silu",
        "eos_token_id": 271,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }
    return {"model_type": "qwen3_5", "text_config": text}


def _disk_tensors(model, drafter, seed: int) -> dict:
    """The tensors of a seeded checkpoint with a drafter, in the layout a quantised checkpoint stores: the language model
    under one prefix, the output head under none, the drafter's own tensors -- bf16 -- under theirs."""
    from model_samples import Checkpoint

    out = {}
    source = Checkpoint.synthetic(model, seed)
    for spec in model.tensor_specs():
        key = spec.name if spec.name.startswith("lm_head.") else MODEL + spec.name
        stored = source.load(spec.name, "cpu")
        if spec.fmt == "scalar":
            out[key] = torch.tensor(stored, dtype=torch.float32)
        elif spec.fmt == "fp8":
            out[key] = stored.codes.view(torch.float8_e4m3fn)
            out[key + "_scale"] = torch.tensor(stored.scale, dtype=torch.float32)
        elif spec.fmt == "nvfp4":
            out[key] = stored.codes
            out[key + "_scale"] = stored.block_scales.view(torch.float8_e4m3fn)
            out[key + "_scale_2"] = torch.tensor(stored.scale, dtype=torch.float32)
        else:
            out[key] = stored
    own = Checkpoint.synthetic(drafter, seed)
    for name in OWN:
        out[DRAFTER + name + ".weight"] = own.load("drafter." + name + ".weight", "cpu")
    return out


def _save_safetensors(path: Path, tensors: dict) -> None:
    """One shard written from the format's definition: eight bytes of little-endian header length, the JSON header padded
    with spaces to a multiple of eight bytes, the tensors' bytes in header order."""
    spelling = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name in sorted(tensors):
        tensor = tensors[name].contiguous()
        raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
        header[name] = {"dtype": spelling[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def _write_checkpoint(directory: Path, tensors: dict, config: dict, shards: int = 2) -> Path:
    """A checkpoint directory as a sharded checkpoint is published: numbered shards, the index file, ``config.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.safetensors"):
        old.unlink()
    names, where = sorted(tensors), {}
    for index in range(shards):
        shard = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        _save_safetensors(directory / shard, {name: tensors[name] for name in names[index::shards]})
        where.update({name: shard for name in names[index::shards]})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": where}))
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def _raw(directory: Path, key: str) -> tuple:
    """``(dtype, shape, bytes)`` of one tensor, read from its shard file by the offsets of the shard's header: the bytes of
    the checkpoint as they are on disk, without the reader under test."""
    for shard in sorted(directory.glob("*.safetensors")):
        content = shard.read_bytes()
        (length,) = struct.unpack("<Q", content[:8])
        entry = json.loads(content[8 : 8 + length]).get(key)
        if entry is not None:
            start, stop = entry["data_offsets"]
            return entry["dtype"], tuple(entry["shape"]), content[8 + length + start : 8 + length + stop]
    raise KeyError(key)


def _bytes(tensor) -> bytes:
    return bytes(tensor.contiguous().reshape(-1).view(torch.uint8).tolist())


def test_a_drafter_on_disk_its_manifest_its_head_rows_and_its_8_bit_weights(tmp_path, monkeypatch, capsys):
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint
    from model_samples.providers import Quant
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.qwen_dense.drafter import MULTIPLIES, NORMS, ManifestRow
    from model_samples.timing import Stages
    from model_samples.weights import BlockScaledTensor, ScaledTensor, quantize_fp8_per_channel

    tiny = _tiny_model()
    seeded = QwenDenseDrafter.seeded(tiny, seed=5)
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, seeded, 5), _config(tiny.geometry))
    (tmp_path / "ids.json").write_text(json.dumps([5, 270, 3, 5, 300, 17, 64, 1, -2]))
    ids = [1, 3, 5, 17, 64, 270]

    read = []
    plain_read = Checkpoint._read
    monkeypatch.setattr(Checkpoint, "_read", lambda checkpoint, key: (read.append(key), plain_read(checkpoint, key))[1])
    disk = Checkpoint.describe(str(directory))
    model = QwenDense.describe(disk)
    of_model, model_keys = dict(disk.specs), dict(disk._keys)
    drafter = QwenDenseDrafter.describe(disk, model, draft_vocab=str(tmp_path / "ids.json"))
    assert disk._open == {} and read == [], "described and bound from the headers: no shard is opened, no tensor read"
    assert (drafter.prefix, drafter.label, drafter.name) == ("mtp.", "checkpoint tiny-checkpoint (local)", "qwen_dense drafter")
    assert drafter.vocabulary.ids == tuple(ids) and (drafter.vocabulary.listed, drafter.vocabulary.repeated, drafter.vocabulary.outside) == (9, 1, 2)

    # the manifest: every tensor of the drafter, the checkpoint's own name for it, how it is stored, what a load makes of it
    eight = "8-bit codes, one scale per row"
    manifest = [
        ("mtp.pre_fc_norm_embedding.weight", "bf16 [64]", "pre_fc_norm_embedding: as stored"),
        ("mtp.pre_fc_norm_hidden.weight", "bf16 [64]", "pre_fc_norm_hidden: as stored"),
        ("mtp.layers.0.input_layernorm.weight", "bf16 [64]", "input_norm: as stored"),
        ("mtp.layers.0.self_attn.q_norm.weight", "bf16 [8]", "q_norm: as stored"),
        ("mtp.layers.0.self_attn.k_norm.weight", "bf16 [8]", "k_norm: as stored"),
        ("mtp.layers.0.post_attention_layernorm.weight", "bf16 [64]", "post_norm: as stored"),
        ("mtp.norm.weight", "bf16 [64]", "final_norm: as stored"),
        ("mtp.fc.weight", "bf16 [64, 128]", f"fc: {eight}"),
        ("mtp.layers.0.self_attn.q_proj.weight", "bf16 [32, 64]", f"rows 0 to 31 of qkv: {eight}"),
        ("mtp.layers.0.self_attn.k_proj.weight", "bf16 [8, 64]", f"rows 32 to 39 of qkv: {eight}"),
        ("mtp.layers.0.self_attn.v_proj.weight", "bf16 [8, 64]", f"rows 40 to 47 of qkv: {eight}"),
        ("mtp.layers.0.self_attn.o_proj.weight", "bf16 [64, 16]", f"o: {eight}"),
        ("mtp.layers.0.mlp.gate_proj.weight", "bf16 [96, 64]", f"rows 0 to 95 of gate_up: {eight}"),
        ("mtp.layers.0.mlp.up_proj.weight", "bf16 [96, 64]", f"rows 96 to 191 of gate_up: {eight}"),
        ("mtp.layers.0.mlp.down_proj.weight", "bf16 [64, 96]", f"down: {eight}"),
        ("model.language_model.embed_tokens.weight", "bf16 [272, 64]", "embedding: the model's table"),
        ("lm_head.weight", "4-bit [272, 64]", "head: its rows at the 6 ids of the draft vocabulary, as stored"),
        ("lm_head.input_scale", "fp32 scalar", "head_quant: the static scale of the head's 4-bit operand"),
    ]
    assert [(row.key, row.stored, row.loaded) for row in drafter.manifest()] == manifest
    names = ["drafter." + name + ".weight" for name in OWN] + ["embed_tokens.weight", "lm_head.weight", "lm_head.input_scale"]
    assert [row.name for row in drafter.manifest()] == names == [spec.name for spec in drafter.tensor_specs()] and len(names) == 18
    assert all(isinstance(row, ManifestRow) for row in drafter.manifest()) and drafter.keys == {row.name: row.key for row in drafter.manifest()}
    assert [DRAFTER + part + ".weight" for _, part in NORMS] + [DRAFTER + part + ".weight" for _, parts in MULTIPLIES for part in parts] == [
        key for key, _, _ in manifest[:15]
    ]
    on_disk = sorted(key for key in disk.tensors if key.startswith(DRAFTER))
    assert on_disk == sorted(key for key, _, _ in manifest[:15]), "the list holds every tensor the checkpoint has under the drafter's prefix"
    text = drafter.manifest_text().splitlines()
    assert text[0] == "drafter of checkpoint tiny-checkpoint (local): 15 tensors of its own under 'mtp.', 3 of the model's" and len(text) == 23
    assert text[1].split() == ["tensor", "stored", "at", "load"] and text[2].split(None, 3)[:3] == ["mtp.pre_fc_norm_embedding.weight", "bf16", "[64]"]
    assert text[20] == "  " + drafter.vocabulary.text() and text[21].startswith("  loaded, besides the model's embedding table: 0.00 GiB (8-bit weights ")
    assert text[22].startswith("  and the bf16 copies its reference blocks multiply on (a load keeps them unless told copies=False): 0.00 GiB (")
    assert str(tmp_path) not in drafter.manifest_text()

    # bound beside the model's list: the model's tensors as they were, the drafter's own added; twice changes nothing
    assert all(disk.specs[name] == spec and disk._keys[name] == model_keys[name] for name, spec in of_model.items())
    assert set(disk.specs) == set(of_model) | set(names) and len(disk.specs) == len(of_model) + 15
    again = QwenDenseDrafter.describe(disk, model, draft_vocab=str(tmp_path / "ids.json"))
    assert again == drafter and set(disk.specs) == set(of_model) | set(names) and read == []

    stages, device, asked = Stages(), torch.device("cpu"), []
    plain_load = Checkpoint.load
    monkeypatch.setattr(
        Checkpoint, "load", lambda checkpoint, name, where, as_format="stored": (asked.append((name, where)), plain_load(checkpoint, name, where, as_format))[1]
    )
    weights = drafter.load(disk, device, stages)
    assert list(stages.seconds) == [
        "read",
        "quantise",
        "merge",
        "dequantise",
    ], "the load of a checkpoint: read, quantise, merge where a weight has parts, and the copies"
    # the output head is read whole on the CPU, whatever the device is, and only its listed rows go to the device; every other tensor is read to the device
    assert [where for name, where in asked if name == "lm_head.weight"] == ["cpu"] and len(asked) == 18
    assert all(where is device for name, where in asked if name != "lm_head.weight")
    head_keys = ["lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2", "lm_head.input_scale"]
    wanted = [key for key, _, _ in manifest[:16]] + head_keys
    assert sorted(set(read)) == sorted(wanted), "what a load reads of the checkpoint: the drafter's list and nothing else"

    # the head: the rows of the checkpoint's output head at the listed ids, byte for byte as the shard file holds them
    dtype, shape, codes = _raw(directory, "lm_head.weight")
    assert (dtype, shape) == ("U8", (272, 32))
    _, _, scales = _raw(directory, "lm_head.weight_scale")
    assert isinstance(weights.head, BlockScaledTensor) and tuple(weights.head.codes.shape) == (6, 32) and tuple(weights.head.block_scales.shape) == (6, 4)
    assert weights.head.shape == (6, 64) and weights.head.codes.dtype == weights.head.block_scales.dtype == torch.uint8
    assert _bytes(weights.head.codes) == b"".join(codes[32 * token : 32 * token + 32] for token in ids)
    assert _bytes(weights.head.block_scales) == b"".join(scales[4 * token : 4 * token + 4] for token in ids)
    shifted = b"".join(codes[32 * (token + 1) : 32 * (token + 1) + 32] for token in ids)
    assert _bytes(weights.head.codes) != shifted and len(set(codes[32 * token : 32 * token + 32] for token in range(272))) == 272, "no two rows are alike"
    assert weights.head.scale == struct.unpack("<f", _raw(directory, "lm_head.weight_scale_2")[2])[0], "the head's own global scale"
    assert weights.head_quant == Quant("nvfp4", struct.unpack("<f", _raw(directory, "lm_head.input_scale")[2])[0])
    assert weights.head_ids.dtype == torch.int32 and weights.head_ids.tolist() == ids

    # the norm weights and the embedding table: the checkpoint's bytes
    for name, part in NORMS:
        dtype, shape, raw = _raw(directory, DRAFTER + part + ".weight")
        loaded = getattr(weights, name)
        assert dtype == "BF16" and loaded.dtype == torch.bfloat16 and tuple(loaded.shape) == shape and _bytes(loaded) == raw, name
    assert _bytes(weights.embedding) == _raw(directory, MODEL + "embed_tokens.weight")[2] and tuple(weights.embedding.shape) == (272, 64)

    # the five weights: the 8-bit quantisation of the checkpoint's bf16 rows, the parts of a merged weight in their order
    def on_disk_bf16(part):
        dtype, shape, raw = _raw(directory, DRAFTER + part + ".weight")
        assert dtype == "BF16"
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)

    shapes = {"fc": (64, 128), "qkv": (48, 64), "o": (64, 16), "gate_up": (192, 64), "down": (64, 96)}
    for name, parts in MULTIPLIES:
        loaded = getattr(weights, name)
        assert isinstance(loaded, ScaledTensor) and tuple(loaded.codes.shape) == shapes[name] and tuple(loaded.scale.shape) == (shapes[name][0],), name
        start = 0
        for part in parts:  # each part by itself, at its rows
            alone = quantize_fp8_per_channel(on_disk_bf16(part))
            stop = start + alone.codes.shape[0]
            assert torch.equal(loaded.codes[start:stop], alone.codes) and torch.equal(loaded.scale[start:stop], alone.scale), (name, part)
            start = stop
        assert start == shapes[name][0]
    assert [name for name, _ in MULTIPLIES] == list(shapes) and [parts for _, parts in MULTIPLIES][1][0].endswith("q_proj"), "query first"
    assert not torch.equal(on_disk_bf16("layers.0.mlp.gate_proj"), on_disk_bf16("layers.0.mlp.up_proj")), "two parts that an exchange would show"
    codes, scales = _per_channel(on_disk_bf16("layers.0.self_attn.o_proj").double().tolist())
    assert weights.o.codes.tolist() == codes and weights.o.scale.tolist() == scales, "and against the format's quantiser of this file, end to end"

    # what the loaded drafter holds, against the arithmetic on its tensor list
    held = drafter.load_bytes()
    assert held["weights"] == sum(getattr(weights, name).codes.numel() + 4 * getattr(weights, name).scale.numel() for name, _ in MULTIPLIES) == 32448
    assert held["norms"] == sum(2 * getattr(weights, name).numel() for name, _ in NORMS) == 672
    assert held["head"] == weights.head.codes.numel() + weights.head.block_scales.numel() + 4 * weights.head_ids.numel() == 6 * (32 + 4 + 4)

    # the model's table handed over: the drafter keeps no second copy and does not read it
    read.clear()
    table = torch.zeros((272, 64), dtype=torch.bfloat16)
    shared = drafter.load(disk, "cpu", embedding=table)
    assert (
        shared.embedding is table and MODEL + "embed_tokens.weight" not in read and sorted(set(read)) == sorted(set(wanted) - {MODEL + "embed_tokens.weight"})
    )
    assert torch.equal(shared.qkv.codes, weights.qkv.codes) and torch.equal(shared.head.codes, weights.head.codes)

    # the same drafter from memory: seeded tensors, the list of the file
    from_memory = dataclasses.replace(seeded, vocabulary=drafter.vocabulary)
    drawn = Stages()
    memory = from_memory.load(Checkpoint.synthetic(from_memory, seed=5), "cpu", drawn)
    assert list(drawn.seconds) == ["draw", "quantise", "merge", "dequantise"]
    for name, _ in NORMS:
        assert torch.equal(getattr(memory, name), getattr(weights, name)), name
    for name, _ in MULTIPLIES:
        assert torch.equal(getattr(memory, name).codes, getattr(weights, name).codes) and torch.equal(getattr(memory, name).scale, getattr(weights, name).scale)
    assert torch.equal(memory.head.codes, weights.head.codes) and torch.equal(memory.head.block_scales, weights.head.block_scales)
    assert _fp32(memory.head.scale) == weights.head.scale and _fp32(memory.head_quant.scale) == weights.head_quant.scale
    assert torch.equal(memory.head_ids, weights.head_ids) and torch.equal(memory.embedding, weights.embedding)
    disk.close()

    # the lines of the family README that call the loader, as they are written there but for the two paths
    call = _readme_blocks()[0]
    assert call.count('"/data/Qwen3.8-27B-NVFP4"') == 1 and call.count('"draft_vocab.json"') == 1
    call = call.replace('"/data/Qwen3.8-27B-NVFP4"', repr(str(directory))).replace('"draft_vocab.json"', repr(str(tmp_path / "ids.json")))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the lines put samples/models on the path
    capsys.readouterr()
    read.clear()
    scope = {}
    exec(compile(call, "the README's call of the loader", "exec"), scope)
    assert capsys.readouterr().out == drafter.manifest_text() + "\n" and scope["drafter"] == drafter
    assert torch.equal(scope["weights"].gate_up.codes, weights.gate_up.codes) and torch.equal(scope["weights"].head.codes, weights.head.codes)
    # called as it is written there, with no table handed over, the load reads the model's table too: the section says so
    its_key = MODEL + "embed_tokens.weight"
    assert its_key in read and sorted(set(read)) == sorted(wanted) and _bytes(scope["weights"].embedding) == _raw(directory, its_key)[2]

    # ... and the lines of the README that run a draft step, as they are written there, on what its call of the loader loaded
    read.clear()
    exec(compile(_readme_blocks()[2], "the README's draft step", "exec"), scope)
    proposed, printed = scope["drafts"], capsys.readouterr().out
    assert printed == f"{proposed.tolist()}\n" and read == [], "the proposals are printed, and no tensor of the checkpoint is read"
    assert proposed.dtype == torch.int32 and tuple(proposed.shape) == (1, 5) and set(proposed.flatten().tolist()) <= set(ids)
    assert tuple(scope["hidden"].shape) == (1, 64) and scope["blocks"].stack == "reference" and (scope["blocks"].sequences, scope["blocks"].tokens) == (1, 1)
    written = scope["cache"].k.any(dim=-1).flatten().tolist()
    assert written == [True] * 5 + [False] * 59, "one row at position 0 and four passes behind it: positions 0 to 4 of 64"
    # the same step by its definition: the body on the row it is handed, the head, then each pass on its own row and proposal
    again = drafter.state(1, 64, "cpu")
    row = drafter.body(scope["call"], scope["weights"], again, scope["hidden"], scope["ids"], 0)
    by_hand = [drafter.propose(scope["call"], scope["weights"], row)]
    for index in range(1, 5):
        row = drafter.body(scope["call"], scope["weights"], again, row, by_hand[-1], index, index)
        by_hand.append(drafter.propose(scope["call"], scope["weights"], row, index))
    assert torch.equal(torch.stack(by_hand, dim=1), proposed) and _bytes(again.k) == _bytes(scope["cache"].k)
    scope["checkpoint"].close()


def test_a_drafter_this_sample_does_not_read_is_declined_by_name(tmp_path):
    """Every drafter the loader declines, and every list it refuses, with its own sentence: the checkpoint is judged first
    (a ``NotImplementedError``: a SKIP that names the tensor or the setting), the list after it (a ``ValueError``)."""
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    tiny = _tiny_model()
    tensors = _disk_tensors(tiny, QwenDenseDrafter.seeded(tiny, seed=5), 5)
    config = _config(tiny.geometry)
    (tmp_path / "ids.json").write_text(json.dumps([1, 2, 3]))
    ids = str(tmp_path / "ids.json")

    def described(change=None, config_change=None, vocabulary=ids):
        changed, settings = dict(tensors), json.loads(json.dumps(config))
        if change is not None:
            change(changed)
        if config_change is not None:
            config_change(settings["text_config"])
        disk = Checkpoint.describe(str(_write_checkpoint(tmp_path / "other", changed, settings)))
        model = QwenDense.describe(disk)
        bound = set(disk.specs)
        try:
            return QwenDenseDrafter.describe(disk, model, draft_vocab=vocabulary)
        except (NotImplementedError, ValueError):
            if not (change is None and config_change is None):  # a checkpoint that is declined binds nothing of its drafter
                assert set(disk.specs) == bound
            raise

    def drop_the_drafter(t):
        for key in [key for key in t if key.startswith(DRAFTER)]:
            del t[key]

    def a_second_drafter(t):
        t.update({"draft." + key[len(DRAFTER) :]: value for key, value in t.items() if key.startswith(DRAFTER)})

    def one_more(t):
        t[DRAFTER + "layers.1.input_layernorm.weight"] = torch.zeros(64, dtype=torch.bfloat16)

    def two_more(t):
        one_more(t)
        t[DRAFTER + "extra.weight"] = torch.zeros((272, 64), dtype=torch.bfloat16)

    label = r"checkpoint other \(local\)"
    reads = "it reads a drafter stored in bf16, whose multiplies it quantises to 8 bits when it loads"
    fc, k_proj = DRAFTER + "fc.weight", DRAFTER + "layers.0.self_attn.k_proj.weight"
    # every checkpoint is written and described in a statement of its own: an assert states what came back, it does no work
    as_generated = described()
    assert as_generated.vocabulary.ids == (1, 2, 3), "the checkpoint as generated is read"
    for key in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"):  # a config without the two keys: one layer, no table of its own
        without = described(config_change=lambda c, key=key: c.pop(key))
        assert without.prefix == DRAFTER, key
    declined = (
        (dict(change=drop_the_drafter), rf"^{label} holds no drafter \(no tensor ends in '.pre_fc_norm_hidden.weight' with 'fc.weight' beside it\)$"),
        (dict(change=lambda t: t.pop(fc)), rf"^{label} holds no drafter \("),
        (dict(change=a_second_drafter), rf"^{label}: the drafter's tensors are not under one prefix \(found: 'draft.', 'mtp.'\)$"),
        (
            dict(change=lambda t: t.pop(DRAFTER + "norm.weight")),
            rf"^{label} has no tensor 'mtp.norm.weight' \(a plain tensor\): not a checkpoint this sample reads$",
        ),
        (
            dict(change=lambda t: t.update({fc: t[fc].float().to(torch.float8_e4m3fn)})),
            rf"^{label}: tensor 'mtp.fc.weight' is F8_E4M3 \[64, 128\] where this sample reads BF16 \[64, 128\] \(a plain tensor\): {reads}$",
        ),
        (
            dict(change=lambda t: t.update({k_proj: torch.zeros((16, 64), dtype=torch.bfloat16)})),
            rf"^{label}: tensor '{k_proj}' is BF16 \[16, 64\] where this sample reads BF16 \[8, 64\] \(a plain tensor\): {reads}$",
        ),
        (dict(change=one_more), rf"^{label} holds a drafter tensor this sample does not read, 'mtp.layers.1.input_layernorm.weight': {reads}$"),
        (dict(change=two_more), rf"^{label} holds a drafter tensor this sample does not read, 'mtp.extra.weight' \(and 1 more\): {reads}$"),
        (dict(config_change=lambda c: c.update(mtp_num_hidden_layers=2)), rf"^{label} has a drafter of 2 layers; this sample reads a drafter of one layer$"),
        (dict(config_change=lambda c: c.update(mtp_num_hidden_layers=0)), rf"^{label} has a drafter of 0 layers; this sample reads a drafter of one layer$"),
        (
            dict(config_change=lambda c: c.update(mtp_use_dedicated_embeddings=True)),
            rf"^{label} has a drafter with an embedding table of its own; this sample reads one that shares the model's$",
        ),
    )
    for changes, sentence in declined:
        for vocabulary in (ids, None):  # declined whatever the list is: the checkpoint is judged first
            with pytest.raises(NotImplementedError, match=sentence) as refusal:
                described(vocabulary=vocabulary, **changes)
            assert str(tmp_path) not in str(refusal.value)

    wrong_kind = (
        (dict(mtp_num_hidden_layers="1"), "config.json has mtp_num_hidden_layers = '1' where a whole number is expected"),
        (dict(mtp_num_hidden_layers=True), "config.json has mtp_num_hidden_layers = True where a whole number is expected"),
        (dict(mtp_num_hidden_layers=1.0), "config.json has mtp_num_hidden_layers = 1.0 where a whole number is expected"),
        (dict(mtp_use_dedicated_embeddings="no"), "config.json has mtp_use_dedicated_embeddings = 'no' where true or false is expected"),
        (dict(mtp_use_dedicated_embeddings=0), "config.json has mtp_use_dedicated_embeddings = 0 where true or false is expected"),
    )
    for entry, sentence in wrong_kind:
        with pytest.raises(ValueError, match=rf"^{label}: {re.escape(sentence)}$"):
            described(config_change=lambda c, entry=entry: c.update(entry))

    # no list: an error that says what the list is and where the recipe for one is
    with pytest.raises(ValueError) as refusal:
        described(vocabulary=None)
    assert str(refusal.value) == (
        "a checkpoint's drafter needs a draft vocabulary: a JSON list of the token ids its head is cut to. "
        'A recipe that makes one from the files --dump-tokens writes: samples/models/qwen_dense/README.md, "The checkpoint\'s drafter"'
    )
    assert "\n## The checkpoint's drafter\n" in README.read_text(), "the section the sentence names is there"
    # and a list the reader refuses is refused here in the reader's words
    with pytest.raises(ValueError, match=r"^draft vocabulary nowhere.json: cannot be read \(No such file or directory\)$"):
        described(vocabulary=str(tmp_path / "nowhere.json"))
    (tmp_path / "outside.json").write_text(json.dumps([272, 500]))
    with pytest.raises(ValueError, match=r"^draft vocabulary outside.json: none of its 2 entries is a token id of this model \(0 to 271\)$"):
        described(vocabulary=str(tmp_path / "outside.json"))


def test_two_tensor_lists_of_one_checkpoint_share_a_name_only_for_the_same_tensor(tmp_path):
    """``Checkpoint.bind`` binds a second list beside the first.  A name of the second that the first holds must be the same
    tensor of the checkpoint in the same format and shape; otherwise the lists are wrong, and nothing of the second is
    bound."""
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.weights import TensorSpec

    tiny = _tiny_model()
    tensors = _disk_tensors(tiny, QwenDenseDrafter.seeded(tiny, seed=5), 5)
    disk = Checkpoint.describe(str(_write_checkpoint(tmp_path / "tiny-checkpoint", tensors, _config(tiny.geometry))))
    QwenDense.describe(disk)
    bound, keys = dict(disk.specs), dict(disk._keys)
    norm = bound["norm.weight"]
    disk.bind([norm], {"norm.weight": MODEL + "norm.weight"})  # the same tensor again: nothing changes
    assert disk.specs == bound and disk._keys == keys
    sentence = (
        "tensor 'norm.weight' is bound to this checkpoint as another tensor already: two tensor lists of one checkpoint share a name only for the same tensor"
    )
    other = [TensorSpec("extra.weight", (64,), "bf16"), norm]
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):  # the same name for another tensor of the checkpoint
        disk.bind(other, {"extra.weight": DRAFTER + "norm.weight", "norm.weight": DRAFTER + "norm.weight"})
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):  # the same tensor under another specification
        disk.bind([dataclasses.replace(norm, std=1.0)], {"norm.weight": MODEL + "norm.weight"})
    assert disk.specs == bound and disk._keys == keys, "a list that does not fit binds nothing, not even its first tensor"
    disk.bind([TensorSpec("extra.weight", (64,), "bf16")], {"extra.weight": DRAFTER + "norm.weight"})
    assert set(disk.specs) == set(bound) | {"extra.weight"} and torch.equal(disk.load("extra.weight", "cpu"), tensors[DRAFTER + "norm.weight"])
    disk.close()


def test_the_seeded_drafter_draws_its_tensors_and_its_list_from_a_seed():
    """The drafter of the seeded model: its list and its tensors come from the seed, and the head it cuts its rows from is
    the model's own seeded head.  And the tensor list at the published widths, as arithmetic: nothing of that size is
    drawn here."""
    from model_samples import Checkpoint
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.qwen_dense.drafter import MULTIPLIES, NORMS, DraftVocab

    tiny = _tiny_model()
    drafter = QwenDenseDrafter.seeded(tiny, seed=5)
    assert (drafter.label, drafter.prefix, drafter.keys) == ("seeded weights", "", {}) and len(drafter.vocabulary.ids) == 68
    assert drafter.manifest_text().splitlines()[0] == "drafter of seeded weights: 15 tensors of its own, 3 of the model's"
    assert [row.key for row in drafter.manifest()] == [row.name for row in drafter.manifest()], "no checkpoint: a tensor is named as the family names it"
    weights = drafter.load(Checkpoint.synthetic(drafter, seed=5), "cpu")
    again = QwenDenseDrafter.seeded(tiny, seed=5).load(Checkpoint.synthetic(drafter, seed=5), "cpu")
    other = QwenDenseDrafter.seeded(tiny, seed=6)
    for name, _ in MULTIPLIES:
        assert torch.equal(getattr(again, name).codes, getattr(weights, name).codes) and torch.equal(getattr(again, name).scale, getattr(weights, name).scale)
    assert all(torch.equal(getattr(again, name), getattr(weights, name)) for name, _ in NORMS) and torch.equal(again.head.codes, weights.head.codes)
    assert other.vocabulary.ids != drafter.vocabulary.ids
    assert not torch.equal(other.load(Checkpoint.synthetic(other, seed=6), "cpu").fc.codes, weights.fc.codes), "another seed: other weights, another list"

    head = Checkpoint.synthetic(tiny, seed=5).load("lm_head.weight", "cpu")  # the model's own checkpoint, drawn from the model's list
    index = torch.tensor(drafter.vocabulary.ids)
    assert torch.equal(weights.head.codes, head.codes[index]) and torch.equal(weights.head.block_scales, head.block_scales[index])
    assert weights.head.scale == head.scale and weights.head_ids.tolist() == list(drafter.vocabulary.ids)
    assert torch.equal(weights.embedding, Checkpoint.synthetic(tiny, seed=5).load("embed_tokens.weight", "cpu")), "and the model's table"
    with pytest.raises(KeyError, match="no tensor named 'drafter.pre_fc_norm_embedding.weight' in this checkpoint"):
        drafter.load(Checkpoint.synthetic(tiny, seed=5), "cpu")  # the model's list holds no tensor of the drafter

    # the published widths: the reduced model of the default run, and the 64-layer model with a list of 65,536 ids
    reduced = QwenDenseDrafter.seeded(QwenDense.reduced())
    shapes = {row.name[len("drafter.") : -len(".weight")]: row.stored for row in reduced.manifest()[:15]}
    assert shapes["fc"] == "bf16 [5120, 10240]" and shapes["layers.0.self_attn.q_proj"] == "bf16 [12288, 5120]"
    assert shapes["layers.0.self_attn.k_proj"] == shapes["layers.0.self_attn.v_proj"] == "bf16 [1024, 5120]"
    assert shapes["layers.0.self_attn.o_proj"] == "bf16 [5120, 6144]" and shapes["layers.0.mlp.down_proj"] == "bf16 [5120, 17408]"
    assert shapes["layers.0.mlp.gate_proj"] == shapes["layers.0.mlp.up_proj"] == "bf16 [17408, 5120]" and shapes["layers.0.self_attn.q_norm"] == "bf16 [256]"
    assert [row.loaded for row in reduced.manifest()[8:11]] == [
        "rows 0 to 12,287 of qkv: 8-bit codes, one scale per row",
        "rows 12,288 to 13,311 of qkv: 8-bit codes, one scale per row",
        "rows 13,312 to 14,335 of qkv: 8-bit codes, one scale per row",
    ]
    assert len(reduced.vocabulary.ids) == 15520 // 4 == 3880
    elements = 5120 * 10240 + 14336 * 5120 + 5120 * 6144 + 34816 * 5120 + 5120 * 17408  # the five weights, one byte each
    rows = 5120 + 14336 + 5120 + 34816 + 5120  # one fp32 scale each
    assert reduced.load_bytes() == {"weights": elements + 4 * rows, "norms": 2 * (5 * 5120 + 2 * 256), "head": 3880 * (2560 + 320 + 4)}
    full = QwenDenseDrafter.seeded(QwenDense(), rows=65536)
    assert full.load_bytes() == {"weights": 424931328, "norms": 52224, "head": 65536 * 2884} and full.manifest()[16].stored == "4-bit [248320, 5120]"
    copies_line = (
        "  and the bf16 copies its reference blocks multiply on (a load keeps them unless told copies=False): 1.42 GiB "
        "(of the five weights 0.79, of the head rows 0.62)"
    )
    loaded_line = "  loaded, besides the model's embedding table: 0.57 GiB (8-bit weights 0.40, norm weights 0.00, head rows 0.18)"
    assert full.manifest_text().splitlines()[-2:] == [loaded_line, copies_line]
    assert full.copy_bytes() == {"weights": 2 * elements, "head": 2 * 65536 * 5120} and reduced.copy_bytes() == {
        "weights": 2 * elements,
        "head": 2 * 3880 * 5120,
    }
    assert full.memory_bytes(1, 2051) == sum(full.load_bytes().values()) + sum(full.copy_bytes().values()) + 2 * 4 * 2051 * 256 * 2

    # the manifest the family README prints for the published checkpoint is this rendering, for that checkpoint's names:
    # a host test cannot read the checkpoint, so it holds the README's block to the tensor list at the published widths
    of_model = {
        "embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "lm_head.input_scale": "lm_head.input_scale",
    }
    keys = {row.name: of_model.get(row.name, "mtp." + row.name[len("drafter.") :]) for row in full.manifest()}
    listed = DraftVocab(tuple(range(65536)), 248320, "draft_vocab.json", 65536)
    published = dataclasses.replace(full, label="checkpoint Qwen3.8-27B-NVFP4 (local)", prefix="mtp.", keys=keys, vocabulary=listed)
    section = _readme_section()
    printed = re.findall(r"```text\n(.*?)```", section, flags=re.S)
    assert len(printed) == 1 and printed[0].rstrip("\n") == published.manifest_text()
    # the last line of that manifest leaves the model's embedding table out, and the README's own call of the loader reads
    # it: the section states the table's size at these widths, and the argument that hands the model's table over instead
    geometry = full.geometry
    table = f"{geometry.vocab:,} x {geometry.hidden:,} elements in bf16, {2 * geometry.vocab * geometry.hidden / 2.0**30:.2f} GiB"
    assert table == "248,320 x 5,120 elements in bf16, 2.37 GiB" and table in " ".join(section.split())
    assert "drafter.load(checkpoint, device, embedding=table)" in section


# --------------------------------------------------------------------------- the draft pass: each function against its definition


def test_the_row_quantiser_against_the_format_definition():
    """``weights.quantize_fp8_per_row``, the quantiser of the rows of a step: scales and codes against a quantiser written
    here from the format (``_per_row``).  Its rule is a third one beside the static scale and the weight's, and it is fixed
    down to its operation: a multiplication by the fp32 reciprocal of the scale, where the weight's rule divides."""
    from model_samples import weights
    from model_samples.providers import FP8_ROW, ROW_QUANT, Quant, torch_ref
    from model_samples.weights import ScaledTensor, quantize_fp8_per_channel, quantize_fp8_per_row

    # Stated rows; every value is exact in bf16.  What each row holds:
    #   0  a largest element of 13/16.  The quotients of elements 1 and 2 by the scale are, in fp32, the midpoints of two
    #      codes (1.3125 and 2.625): the product with the reciprocal of the scale lands just above each and gives the code
    #      above, where a division rounds to the even code below
    #   1  all zero: the scale is the floor, the codes are 0
    #   2  a largest element below 1/512: the scale stays at the floor
    #   3  a largest element of 1.0234375: its product with the reciprocal of its own scale is above 448 in fp32 and
    #      saturates at the largest code, never the code that is not a number; the other elements are far smaller
    #   4  the largest element is negative and 256 times row 0's: its scale is 256 times row 0's
    rows = [
        [0.8125, 39.0 / 16384.0, 39.0 / 8192.0, -0.40625],
        [0.0, 0.0, 0.0, 0.0],
        [2.0**-10, -(2.0**-11), 2.0**-13, 0.0],
        [1.0234375, -1.0234375, 2.0**-8, -0.0],
        [-208.0, 0.8125, 39.0 / 64.0, 2.0**-20],
    ]
    x = torch.tensor(rows, dtype=torch.bfloat16)
    assert x.double().tolist() == rows, "the stated values are bf16 values"
    stored = quantize_fp8_per_row(x)
    assert isinstance(stored, ScaledTensor) and stored.codes.dtype == torch.uint8 and tuple(stored.codes.shape) == (5, 4)
    assert stored.scale.dtype == torch.float32 and tuple(stored.scale.shape) == (5,), "one fp32 scale per row of the step"
    codes, scales = _per_row(rows)
    assert stored.codes.tolist() == codes and stored.scale.tolist() == scales, "byte for byte the format's quantiser, and its scales bit for bit"
    floor = _fp32(1.0 / (448.0 * 512.0))
    assert scales == [_fp32(0.8125 / 448.0), floor, floor, _fp32(1.0234375 / 448.0), _fp32(208.0 / 448.0)] and weights.PER_ROW_FLOOR == 512
    assert codes == [
        [0x7E, 0x3B, 0x43, 0xF6],
        [0, 0, 0, 0],
        [0x76, 0xEE, 0x5E, 0],
        [0x7E, 0xFE, 0x3E, 0x80],
        [0xFE, 0x3E, 0x3B, 0],
    ], "stated bytes"
    # the operation: the same two elements under the weight's rule, which divides, are the codes below
    scale = scales[0]
    assert [_fp32(value / scale) for value in rows[0][1:3]] == [1.3125, 2.625], "the two quotients of row 0 are midpoints of two codes"
    assert [_code(_fp32(value / scale)) for value in rows[0][1:3]] == [0x3A, 0x42] and quantize_fp8_per_channel(x).codes[0].tolist() == [0x7E, 0x3A, 0x42, 0xF6]
    assert torch.equal(quantize_fp8_per_channel(x).scale, stored.scale), "the two rules share the scale and differ in the operation behind it"
    # saturation: the largest element of row 3 times the reciprocal of its scale is above 448 in fp32
    assert _fp32(1.0234375 * _fp32(1.0 / scales[3])) > 448.0 and 0x7F not in set(sum(codes, [])) and 0xFF not in set(sum(codes, []))
    # a scale per row, not one per step: rows 0 and 4 have two magnitudes, and under the step's one scale row 0 would lose its codes
    assert scales[4] == 256.0 * scales[0] and codes[4][1] == 0x3E and codes[0][0] == 0x7E, "0.8125 is the largest code in row 0 and the code of 1.75 in row 4"
    one_scale = [[_code(_fp32(value * _fp32(1.0 / scales[4]))) for value in row] for row in rows]
    assert (
        one_scale[0] == [0x3E, 3, 5, 0xB6] and one_scale[0] != codes[0]
    ), "under the scale of the step's largest row, row 0 is left with codes 256 times smaller"

    # random rows of several magnitudes, among them one below the floor and one of zeros: every byte and every scale
    generator = torch.Generator().manual_seed(17)
    y = (torch.randn((40, 96), generator=generator) * torch.logspace(-3, 2, 40).unsqueeze(1)).to(torch.bfloat16)
    y[5] = y[5] * 2.0**-9
    y[7] = 0.0
    y[9, 3] = -400.0  # the largest element of its row, negative
    stored = quantize_fp8_per_row(y)
    codes, scales = _per_row(y.double().tolist())
    assert stored.codes.tolist() == codes and stored.scale.tolist() == scales
    assert scales[5] == scales[7] == floor and int(stored.codes[9, 3]) == 0xFE and len(set(scales)) == 39
    assert sum(1 for row in codes for byte in row if byte & 0x7F == 0x7E) >= 38, "every row above the floor reaches the largest code at its largest element"
    divided = quantize_fp8_per_channel(y)  # the weight's rule on the same rows: the same scales, and the codes of ITS definition
    assert torch.equal(divided.scale, stored.scale) and divided.codes.tolist() == _per_channel(y.double().tolist())[0]

    # a row is quantised by itself: the rows of a step together, or one by one, in any order
    for row in (0, 5, 7, 39):
        alone = quantize_fp8_per_row(y[row : row + 1])
        assert torch.equal(alone.codes, stored.codes[row : row + 1]) and torch.equal(alone.scale, stored.scale[row : row + 1]), row
    flipped = quantize_fp8_per_row(y.flip(0))
    assert torch.equal(flipped.codes, stored.codes.flip(0)) and torch.equal(flipped.scale, stored.scale.flip(0))
    # rows in another dtype are rounded to bf16 first: the largest magnitude and the codes are those of the rounded values
    wide = y.float() * (1.0 + 2.0**-12)
    assert not torch.equal(wide.to(torch.bfloat16).float(), wide)
    rounded = quantize_fp8_per_row(wide)
    assert torch.equal(rounded.codes, quantize_fp8_per_row(wide.to(torch.bfloat16)).codes) and torch.equal(
        rounded.scale, quantize_fp8_per_row(wide.to(torch.bfloat16)).scale
    )
    assert rounded.scale.tolist() == _per_row(wide.to(torch.bfloat16).double().tolist())[1]
    for wrong in (torch.zeros(8, dtype=torch.bfloat16), torch.zeros((2, 3, 4), dtype=torch.bfloat16)):
        with pytest.raises(ValueError, match=re.escape(f"the per-row 8-bit quantisation takes the rows of a step [rows, width]; got {list(wrong.shape)}")):
            quantize_fp8_per_row(wrong)

    # the format a block is handed for it: a Quant without a scale; a static format states one
    assert ROW_QUANT == Quant(FP8_ROW) and ROW_QUANT.scale is None and FP8_ROW == "fp8_row"
    through = torch_ref.quantize(y, ROW_QUANT)
    assert torch.equal(through.codes, stored.codes) and torch.equal(through.scale, stored.scale)
    sentence = 'a quantisation format with a static scale states it, and "fp8_row" forms its scale per call; '
    for arguments, has in (
        (("fp8",), '"fp8" states no scale'),
        (("nvfp4", None), '"nvfp4" states no scale'),
        (("fp8_row", 0.5), '"fp8_row" states the scale 0.5'),
    ):
        with pytest.raises(ValueError, match=f"^{re.escape(sentence + has)}$"):
            Quant(*arguments)
    assert Quant("fp8", 0.0).scale == 0.0 and Quant("nvfp4", 0.25) == Quant("nvfp4", 0.25)


def test_the_joint_quantiser_of_the_two_normed_halves():
    """The operand of the drafter's first multiply (``torch_ref.pre_fc_norm_quantize``): the embedding rows and the hidden
    rows each normed with its own weight and rounded to bf16, the embedding half first, and ONE scale per row over both
    halves.  Against the norm formula evaluated element by element and the row quantiser of this file."""
    from model_samples.providers import ROW_QUANT, torch_ref

    generator = torch.Generator().manual_seed(23)
    rows, width, eps = 5, 24, 1.0e-6
    embedding = (torch.randn((rows, width), generator=generator) * 0.05).to(torch.bfloat16)
    hidden = (torch.randn((rows, width), generator=generator) * 3.0).to(torch.bfloat16)
    weight_embedding = (torch.randn(width, generator=generator) * 0.02).to(torch.bfloat16)
    weight_hidden = (torch.randn(width, generator=generator) * 0.02 + 2.0).to(torch.bfloat16)  # the hidden half three times the embedding half

    def normed(values, weight):
        """bf16(v * rsqrt(mean(v^2) + eps) * (1 + w)) of every row, element by element, in doubles."""
        out = []
        for row in values.double().tolist():
            inverse = 1.0 / math.sqrt(sum(v * v for v in row) / len(row) + eps)
            out.append([_cast(v * inverse * (1.0 + w)) for v, w in zip(row, weight.double().tolist())])
        return out

    first, second = normed(embedding, weight_embedding), normed(hidden, weight_hidden)
    assert torch_ref.rms_norm(embedding, weight_embedding, eps, torch.float64).double().tolist() == first, "the norm of this file is the reference's"
    assert torch_ref.rms_norm(hidden.float(), weight_hidden, eps, torch.float64).double().tolist() == second
    codes, scales = _per_row([one + two for one, two in zip(first, second)])
    stored = torch_ref.pre_fc_norm_quantize(embedding, hidden, weight_embedding, weight_hidden, ROW_QUANT, eps, torch.float64)
    assert tuple(stored.codes.shape) == (rows, 2 * width) and tuple(stored.scale.shape) == (rows,)
    assert stored.codes.tolist() == codes and stored.scale.tolist() == scales, "the embedding half first, one scale per row over both halves"
    # one scale over both: it is the hidden half's largest magnitude here, and the embedding half quantised alone has another
    largest = [max(abs(v) for v in row) for row in second]
    assert scales == [_fp32(value / 448.0) for value in largest] and all(max(abs(v) for v in row) < 0.5 * big for row, big in zip(first, largest))
    alone_codes, alone_scales = _per_row(first)
    assert alone_scales != scales and [row[:width] for row in codes] != alone_codes, "a scale per half would give the embedding half other codes"
    assert all(0x7E not in {byte & 0x7F for byte in row[:width]} for row in codes), "no element of the smaller half reaches the largest code"
    # the halves exchanged: other bytes, whether the weights go with them or not
    exchanged = torch_ref.pre_fc_norm_quantize(hidden, embedding, weight_hidden, weight_embedding, ROW_QUANT, eps, torch.float64)
    assert torch.equal(exchanged.codes, torch.cat((stored.codes[:, width:], stored.codes[:, :width]), dim=1)) and not torch.equal(exchanged.codes, stored.codes)
    crossed = torch_ref.pre_fc_norm_quantize(hidden, embedding, weight_embedding, weight_hidden, ROW_QUANT, eps, torch.float64)
    assert not torch.equal(crossed.codes, stored.codes) and not torch.equal(crossed.codes, exchanged.codes)
    # a row by itself; and without a quantisation the chain has nothing to hand on
    alone = torch_ref.pre_fc_norm_quantize(embedding[2:3], hidden[2:3], weight_embedding, weight_hidden, ROW_QUANT, eps, torch.float64)
    assert torch.equal(alone.codes, stored.codes[2:3]) and torch.equal(alone.scale, stored.scale[2:3])
    assert torch_ref.pre_fc_norm_quantize(embedding, hidden, weight_embedding, weight_hidden, None, eps, torch.float64) is None


def test_the_drafters_gate_and_silu_against_their_formulas():
    """The gate of the drafter's attention output, ``bf16(o * sigmoid(gate))`` with ONE rounding
    (``torch_ref.gate_row_quantize``), and ``bf16(bf16(SiLU(gate)) * up)`` with two (``torch_ref.silu_mul``), each against
    its formula evaluated element by element in doubles.  An fp32 sigmoid is not the double's to the last bit, so an
    element is judged where its value is clear of a bf16 rounding midpoint by more than that -- nearly all are.  And the
    model's own chains are byte for byte what they were: ``gate_quantize`` rounds the sigmoid first, which is another
    function, and ``silu_mul_quantize`` is ``silu_mul`` with its quantisation behind it."""
    import torch.nn.functional as F

    from model_samples.providers import ROW_QUANT, Quant, torch_ref
    from model_samples.weights import quantize_fp8, quantize_nvfp4

    bf16 = torch.bfloat16
    generator = torch.Generator().manual_seed(29)
    o = (torch.randn((6, 512), generator=generator) * 2.0).to(bf16)
    gate = (torch.randn((6, 512), generator=generator) * 3.0).to(bf16)

    # the gate: one rounding
    gated = torch_ref.gate_row_quantize(o, gate, ROW_QUANT)
    values = (
        gated.codes.view(torch.float8_e4m3fn).double() * gated.scale.double().unsqueeze(1)
    ).tolist()  # the codes back at their scale: only to know they are codes of it
    assert tuple(gated.codes.shape) == (6, 512) and tuple(gated.scale.shape) == (6,) and all(len(row) == 512 for row in values)
    judged, product = 0, []
    for row_o, row_gate in zip(o.double().tolist(), gate.double().tolist()):
        product.append([a * (1.0 / (1.0 + math.exp(-g))) for a, g in zip(row_o, row_gate)])
    made = (o.float() * torch.sigmoid(gate.float())).to(bf16)  # the chain's two operations, as they are written in the block
    for row, (row_made, row_product) in enumerate(zip(made.double().tolist(), product)):
        for column, (got, exact) in enumerate(zip(row_made, row_product)):
            if _clear_of_a_midpoint(exact):
                judged += 1
                assert got == _bf16(exact), (row, column)
    assert judged == 6 * 512 - 2, "all but two elements are judged: those sit next to a rounding midpoint"
    codes, scales = _per_row(made.double().tolist())
    assert gated.codes.tolist() == codes and gated.scale.tolist() == scales, "and the rounded rows are quantised per row"
    # a stated element: 1.2109375 * sigmoid(-1.625) = 0.19917.., which is bf16 0.19921875; with the sigmoid rounded to bf16
    # first (0.1640625) the product is 0.19867.., which is bf16 0.1982421875 -- the model's chain, another function
    a, g = torch.tensor([[1.2109375]], dtype=bf16), torch.tensor([[-1.625]], dtype=bf16)
    exact = 1.2109375 / (1.0 + math.exp(1.625))
    assert _clear_of_a_midpoint(exact) and _bf16(exact) == 0.19921875 and _bf16(1.2109375 * _bf16(1.0 / (1.0 + math.exp(1.625)))) == 0.1982421875
    # (a row of one element shows its bf16 value in its scale: the element is the row's largest, and the scale is it / 448)
    one_rounding, two_roundings = torch_ref.gate_row_quantize(a, g, ROW_QUANT), torch_ref.gate_quantize(a, g, ROW_QUANT)
    assert one_rounding.scale.tolist() == [_fp32(0.19921875 / 448.0)] and two_roundings.scale.tolist() == [_fp32(0.1982421875 / 448.0)]
    assert one_rounding.codes.tolist() == two_roundings.codes.tolist() == [[0x7E]]
    # ... and that chain is what it was: the sigmoid rounded, then the product, then the static quantisation
    scale = 8.0 / 448.0
    was = quantize_fp8((o.float() * torch.sigmoid(gate.float()).to(bf16).float()).to(bf16), scale)
    assert torch.equal(torch_ref.gate_quantize(o, gate, Quant("fp8", scale)).codes, was.codes)
    differing = int(((o.float() * torch.sigmoid(gate.float()).to(bf16).float()).to(bf16) != made).sum())
    assert 100 < differing < 2000, "the two functions differ on a share of the elements, by one bf16 unit"

    # SiLU x up: two roundings
    gate_up = torch.cat((gate, o), dim=1)  # the gate half, then the up half
    act = torch_ref.silu_mul(gate_up)
    assert act.dtype == bf16 and tuple(act.shape) == (6, 512)
    judged = 0
    for row, (row_gate, row_up, row_act) in enumerate(zip(gate.double().tolist(), o.double().tolist(), act.double().tolist())):
        for column, (g, u, got) in enumerate(zip(row_gate, row_up, row_act)):
            silu = g / (1.0 + math.exp(-g))
            if _clear_of_a_midpoint(silu):
                judged += 1
                assert got == _bf16(_bf16(silu) * u), (row, column)  # the product of two bf16 values is exact in a double: rounded once
    # 19 of 3,072 are not judged, one in 162, where values spread evenly would leave one in 2,048: a gate value on the bf16 grid
    # repeats, and its SiLU with it
    assert judged == 6 * 512 - 19
    # the SiLU rounded before the product is the function: on some elements a single rounding of the product is another value
    single = (F.silu(gate.float()) * o.float()).to(bf16)
    assert 100 < int((single != act).sum()) < 2000
    # the model's chain: byte for byte what it was, for both of its quantisations, and the first two operations are silu_mul
    for quant, quantiser in ((Quant("nvfp4", 8.0 / 2688.0), quantize_nvfp4), (Quant("fp8", scale), quantize_fp8)):
        was = quantiser((F.silu(gate_up[:, :512].float()).to(bf16).float() * gate_up[:, 512:].float()).to(bf16), quant.scale)
        now = torch_ref.silu_mul_quantize(gate_up, quant)
        assert all(torch.equal(a, b) for a, b in zip(now.tensors(), was.tensors())) and now.scale == was.scale
    per_row = torch_ref.silu_mul_quantize(gate_up, ROW_QUANT)
    assert per_row.codes.tolist() == _per_row(act.double().tolist())[0]


def test_the_drafters_token_choice_takes_the_first_of_equal_columns_and_returns_its_id():
    """``torch_ref.listed_argmax``: per row the column of the largest logit, the first of equal ones, and the token id of
    that column -- on stated logits, and through the block of the key ``torch.argmax+lookup``."""
    from model_samples import providers
    from model_samples.blocks import DeviceFacts
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import QwenDenseDrafter, configs

    logits = torch.tensor(
        [
            [1.0, 5.0, 5.0, -2.0, 5.0],  # three equal largest values: the first of them, column 1
            [7.0, 7.0, 7.0, 7.0, 7.0],  # every column equal: column 0
            [-1.0, -3.0, -1.0, -2.0, -9.0],  # negative values, two equal: column 0
            [0.0, 0.5, 0.25, 0.5, 0.125],  # columns 1 and 3: column 1
            [-4.0, -4.0, -4.0, -4.0, -3.5],  # the largest value in the last column
        ],
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([11, 22, 33, 44, 55], dtype=torch.int32)  # ascending, as a draft vocabulary's are: the lowest column is the lowest id
    chosen = torch_ref.listed_argmax(logits, ids)
    assert chosen.tolist() == [22, 11, 11, 22, 55] and chosen.dtype == torch.int32 and tuple(chosen.shape) == (5,)
    assert torch_ref.listed_argmax(logits[3:4], ids).tolist() == [22] and torch_ref.listed_argmax(logits.float(), ids).tolist() == [22, 11, 11, 22, 55]
    assert torch_ref.listed_argmax(logits, torch.arange(5, dtype=torch.int32)).tolist() == [
        1,
        0,
        0,
        1,
        4,
    ], "the column itself is not the token: the id of the column is"
    assert torch_ref.listed_argmax(logits.flip(1), ids).tolist() == [11, 11, 33, 22, 11], "the first of equal columns of the flipped rows"
    # the block of the drafter's table: the same function, a reference chain, not the sample's host path
    drafter = QwenDenseDrafter.seeded(_tiny_model(), seed=5)
    row = configs.DRAFTER_PROVIDERS["draft.token_choice"][0]
    block = providers.factories()[row.key](row, drafter, DeviceFacts((10, 0)), {})
    assert (row.key, row.fallback, block.entry) == ("torch.argmax+lookup", "reference chain", "torch.argmax + the token id of the chosen row")
    assert block(logits, ids).tolist() == [22, 11, 11, 22, 55] and [kernel.relation for kernel in block.kernels] == ["fallback"]


# --------------------------------------------------------------------------- the draft pass: its structure


def _drafter_on_the_cpu(seed: int = 5, stack: str = "reference"):
    """``(drafter, weights, call, log)``: the seeded drafter of the tiny geometry loaded on the CPU with its copies, the
    lookup of a plan of its provider table as a family's step takes it, and the ``(block, pass)`` of every call made."""
    from model_samples import Checkpoint
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDenseDrafter

    drafter = QwenDenseDrafter.seeded(_tiny_model(), seed=seed)
    weights = drafter.load(Checkpoint.synthetic(drafter, seed=seed), "cpu")
    blocks = plan(drafter.provider_table, drafter.regimes, DeviceFacts((10, 0)), 1, 1, stack, model=drafter, source=drafter.table_source)
    log = []

    def call(block, index, *operands):
        log.append((block, index))
        return blocks[block](*operands)

    return drafter, weights, call, log


def _step_rows(drafter, sequences: int, tokens: int, seed: int):
    """Hidden rows ``[sequences x tokens, hidden]`` bf16 and ids ``[sequences x tokens]`` int32, drawn from ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    hidden = (torch.randn((sequences * tokens, drafter.geometry.hidden), generator=generator) * 1.5).to(torch.bfloat16)
    return hidden, torch.randint(0, drafter.geometry.vocab, (sequences * tokens,), generator=generator, dtype=torch.int32)


def test_every_block_of_the_drafters_plan_is_the_function_it_stands_for(monkeypatch):
    """The blocks of the drafter's plan, as ``plan`` builds them from ``DRAFTER_PROVIDERS``, each against the function it
    stands for.  The tests above call the functions, the stand-in test below replaces the blocks, and a pass compared
    with passes of itself moves with a defect: an error INSIDE a block's class -- operands exchanged, another function
    called, an epsilon or an accumulation of its own -- is held here.  The blocks of the new keys on stated operands
    whose roles cannot be exchanged without other bytes; then one body pass through the plan's blocks against the same
    pass written out from the functions, rows and cache; then the head, block by block."""
    from model_samples import weights as formats
    from model_samples.providers import ROW_QUANT, AttentionParams, Quant, torch_ref

    drafter, weights, call, _ = _drafter_on_the_cpu()
    bf16, acc, eps, width = torch.bfloat16, torch.float64, drafter.norm_eps, drafter.geometry.hidden
    generator = torch.Generator().manual_seed(41)

    def same(got, want) -> bool:
        """Two quantised operands of one kind with the same bytes and the same scale: a float, or one per row."""
        if type(got) is not type(want) or type(got.scale) is not type(want.scale):
            return False
        scales = torch.equal(got.scale, want.scale) if isinstance(want.scale, torch.Tensor) else got.scale == want.scale
        return scales and all(_bytes(a) == _bytes(b) for a, b in zip(got.tensors(), want.tensors()))

    # the two norms in front of the first multiply: a small embedding half, a large hidden half, two weights
    embedding = (torch.randn((5, width), generator=generator) * 0.05).to(bf16)
    hidden = (torch.randn((5, width), generator=generator) * 3.0).to(bf16)
    weight_embedding = (torch.randn(width, generator=generator) * 0.02).to(bf16)
    weight_hidden = (torch.randn(width, generator=generator) * 0.02 + 2.0).to(bf16)
    got = call("draft.pre_fc_norm", 0, embedding, hidden, weight_embedding, weight_hidden, ROW_QUANT)
    want = torch_ref.pre_fc_norm_quantize(embedding, hidden, weight_embedding, weight_hidden, ROW_QUANT, eps, acc)
    assert same(got, want), "the embedding half first with its own weight, one scale over both, the model's epsilon"
    others = (
        torch_ref.pre_fc_norm_quantize(hidden, embedding, weight_hidden, weight_embedding, ROW_QUANT, eps, acc),  # the halves exchanged
        torch_ref.pre_fc_norm_quantize(embedding, hidden, weight_embedding, weight_hidden, ROW_QUANT, 100.0 * eps, acc),  # another epsilon
    )
    assert not any(same(other, want) for other in others), "these operands tell each of them from the block"
    # ... and the accumulation of the run, which rows of this size cannot tell from fp32 sums: what the block hands the function
    handed = []
    monkeypatch.setattr(torch_ref, "pre_fc_norm_quantize", lambda *arguments: handed.append(arguments[4:]))
    call("draft.pre_fc_norm", 0, embedding, hidden, weight_embedding, weight_hidden, ROW_QUANT)
    monkeypatch.undo()
    assert handed == [(ROW_QUANT, eps, torch.float64)], "its quantisation, the model's epsilon, fp64"

    # the gate: attention rows, then the gate, one rounding -- with the stated element of the function's test, whose scale
    # shows its bf16 value (0.19921875; the model's chain, which rounds the sigmoid first, gives 0.1982421875)
    o = (torch.randn((6, 512), generator=generator) * 2.0).to(bf16)
    gate = (torch.randn((6, 512), generator=generator) * 3.0).to(bf16)
    got, want = call("draft.gate_quantize", 0, o, gate, ROW_QUANT), torch_ref.gate_row_quantize(o, gate, ROW_QUANT)
    assert same(got, want) and not same(torch_ref.gate_row_quantize(gate, o, ROW_QUANT), want)
    stated = call("draft.gate_quantize", 0, torch.tensor([[1.2109375]], dtype=bf16), torch.tensor([[-1.625]], dtype=bf16), ROW_QUANT)
    assert stated.scale.tolist() == [_fp32(0.19921875 / 448.0)] and stated.codes.tolist() == [[0x7E]]

    # SiLU x up: the gate half first, the SiLU rounded before the product
    gate_up = torch.cat((gate, o), dim=1)
    act, formula = call("draft.silu_mul", 0, gate_up), torch_ref.silu_mul(gate_up)
    exchanged, unrounded = torch_ref.silu_mul(torch.cat((o, gate), dim=1)), (torch.nn.functional.silu(gate.float()) * o.float()).to(bf16)
    assert _bytes(act) == _bytes(formula) and _bytes(exchanged) != _bytes(formula) and _bytes(unrounded) != _bytes(formula)

    # the quantisation of a row: the format it is asked for, per row for the MLP's operand and 4-bit for the head's
    per_row = call("draft.activation_quantize", 0, o, ROW_QUANT)
    four_bit = call("draft.head.quantize", 0, hidden, Quant("nvfp4", 8.0 / 2688.0))
    assert same(per_row, formats.quantize_fp8_per_row(o)) and same(four_bit, formats.quantize_nvfp4(hidden, 8.0 / 2688.0))
    assert tuple(per_row.scale.shape) == (6,) and four_bit.scale == 8.0 / 2688.0, "a scale per row there, the head's static scale here"
    assert isinstance(per_row, formats.ScaledTensor) and isinstance(four_bit, formats.BlockScaledTensor)

    # one body pass through the plan's blocks = D0 .. D15 written out from the functions: two sequences, three rows each
    sequences, tokens, position = 2, 3, 2
    hidden, ids = _step_rows(drafter, sequences, tokens, seed=43)
    through, written = drafter.state(sequences, 12, "cpu"), drafter.state(sequences, 12, "cpu")
    rows = drafter.body(call, weights, through, hidden, ids, position)
    g, copies = drafter.geometry, weights.copies
    params = AttentionParams(g.attn_heads, g.attn_kv_heads, g.attn_head_dim, g.rotary_dim, g.rope_theta, weights.q_norm, weights.k_norm)
    embedded = weights.embedding.index_select(0, ids)
    q = torch_ref.pre_fc_norm_quantize(embedded, hidden, weights.pre_fc_norm_embedding, weights.pre_fc_norm_hidden, ROW_QUANT, eps, acc)
    stream = torch_ref.linear(q, copies["fc"], acc)
    q, _, residual = torch_ref.add_norm_quantize(stream, None, weights.input_norm, ROW_QUANT, eps, acc)
    qh, gate = torch_ref.attention_prepare_tokens(torch_ref.linear(q, copies["qkv"], acc), written, position, tokens, params, eps, acc)
    q = torch_ref.gate_row_quantize(torch_ref.attention_tokens(qh, written, position, tokens, acc), gate, ROW_QUANT)
    q, _, residual = torch_ref.add_norm_quantize(torch_ref.linear(q, copies["o"], acc), residual, weights.post_norm, ROW_QUANT, eps, acc)
    q = formats.quantize_fp8_per_row(torch_ref.silu_mul(torch_ref.linear(q, copies["gate_up"], acc)))
    _, normed, _ = torch_ref.add_norm_quantize(torch_ref.linear(q, copies["down"], acc), residual, weights.final_norm, None, eps, acc)
    assert _bytes(rows) == _bytes(normed), "the body through the plan's blocks is the pass written out from the functions"
    assert _bytes(through.k) == _bytes(written.k) and _bytes(through.v) == _bytes(written.v) and bool(written.k[:, :, 2:5].any(dim=-1).all())

    # the head, block by block: the operand in 4 bits at the head's static scale, the logits over the listed rows, the id
    chosen = rows.unflatten(0, (sequences, tokens))[:, 1]
    operand = formats.quantize_nvfp4(chosen, weights.head_quant.scale)
    logits = torch_ref.linear(operand, copies["head"], acc)
    got = call("draft.head.quantize", 0, chosen, weights.head_quant)
    got_logits = call("draft.head", 0, got, copies["head"])
    proposed = drafter.propose(call, weights, chosen)
    assert same(got, operand) and _bytes(got_logits) == _bytes(logits) and tuple(logits.shape) == (sequences, len(drafter.vocabulary.ids))
    assert proposed.tolist() == torch_ref.listed_argmax(logits, weights.head_ids).tolist()


def test_every_block_of_a_pass_gives_a_sequence_alone_the_bytes_it_has_among_others():
    """What a block of the drafter's plan declares about its operands -- which have one leading row per row of the step
    (``per_row``) and which is state the call writes (``state_args``) -- read the way a checked run reads it.  The tap of
    the tie-out runs every block of a step over several sequences once more on ONE sequence, its operands cut to that
    sequence's rows by ``per_row`` (an 8-bit operand with a scale per row through ``ScaledTensor.rows``, the cache
    through ``KvCache.rows``), and compares the bytes.  No session taps a draft pass yet: this test is what reads the
    declarations, on a pass over two sequences of three rows, once for each sequence."""
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.tie_out import BlockTap

    drafter, weights, plain, _ = _drafter_on_the_cpu()
    declared = {
        "draft.embedding": ((0,), ()),
        "draft.pre_fc_norm": ((0, 1), ()),
        "draft.projection": ((0,), ()),
        "draft.norm_quantize": ((0, 1), ()),
        "draft.attention.prepare": ((0, 1), (1,)),
        "draft.attention.core": ((0, 1), ()),
        "draft.gate_quantize": ((0, 1), ()),
        "draft.silu_mul": ((0,), ()),
        "draft.activation_quantize": ((0,), ()),
        "draft.final_norm": ((0, 1), ()),
        "draft.head.quantize": ((0,), ()),
        "draft.head": ((0,), ()),
        "draft.token_choice": ((0,), ()),
    }
    sequences, tokens, position = 2, 3, 2
    facts, table, source = DeviceFacts((10, 0)), drafter.provider_table, drafter.table_source
    body = plan(table, drafter.regimes, facts, sequences, tokens, "reference", model=drafter, source=source)
    head = plan(table, drafter.regimes, facts, sequences, 1, "reference", model=drafter, source=source)  # the head runs on one row per sequence
    assert {name: (tuple(block.per_row), tuple(block.state_args)) for name, block in body.items()} == declared
    assert {name: (tuple(block.per_row), tuple(block.state_args)) for name, block in head.items()} == declared

    hidden, ids = _step_rows(drafter, sequences, tokens, seed=47)
    front = [_step_rows(drafter, sequences, 1, seed=200 + at) for at in range(position)]  # what stands in front: two positions
    untapped = drafter.state(sequences, 12, "cpu")
    for at, (row, token) in enumerate(front):
        drafter.body(plain, weights, untapped, row, token, at)
    rows = drafter.body(plain, weights, untapped, hidden, ids, position)
    proposals = drafter.propose(plain, weights, rows.unflatten(0, (sequences, tokens))[:, 1])

    taps = BlockTap(body, body), BlockTap(head, head)
    for alone in range(sequences):  # the sequence the tap runs alone
        cache = drafter.state(sequences, 12, "cpu")
        for at, (row, token) in enumerate(front):
            drafter.body(plain, weights, cache, row, token, at)
        for tap in taps:
            tap.step = alone
        tapped = drafter.body(lambda block, index, *operands: taps[0](block, index, body[block], operands), weights, cache, hidden, ids, position)
        kept = tapped.unflatten(0, (sequences, tokens))[:, 1]
        chosen = drafter.propose(lambda block, index, *operands: taps[1](block, index, head[block], operands), weights, kept)
        # the tap hands back what the blocks made over both sequences, and leaves the cache as they left it
        assert _bytes(tapped) == _bytes(rows) and chosen.tolist() == proposals.tolist(), alone
        assert _bytes(cache.k) == _bytes(untapped.k) and _bytes(cache.v) == _bytes(untapped.v), alone
    stats = {**taps[0].stats, **taps[1].stats}
    calls = drafter.calls()
    assert sorted(stats) == sorted(declared) and sum(calls.values()) == 18
    assert {name: of.batch_calls for name, of in stats.items()} == {name: sequences * calls[name] for name in declared}, "every call, once per sequence"
    assert [name for name, of in stats.items() if of.batch_equal != of.batch_calls] == [], "a sequence alone: the bytes it has among the others"


def test_a_pass_over_several_rows_is_that_many_passes_of_one_row():
    """The body of a pass over ``T`` rows per sequence against ``T`` passes of one row at the positions of those rows: the
    final-norm rows and the cache, byte for byte, at 2 and at 6 rows per sequence, for one sequence and for two.  The
    first pass of a step runs over the rows of the verify step in front of it and every other pass at one row; a pass
    over a prompt runs at one row per position: they must be one function of a position's own two inputs and of what
    stands in front of it."""
    from model_samples.state import KvCache

    drafter, weights, call, log = _drafter_on_the_cpu()
    prefix, capacity = 3, 12
    for sequences in (1, 2):
        for tokens in (2, 6):
            at_once, one_by_one = drafter.state(sequences, capacity, "cpu"), drafter.state(sequences, capacity, "cpu")
            assert isinstance(at_once, KvCache) and tuple(at_once.k.shape) == tuple(at_once.v.shape) == (sequences, 1, capacity, 8)
            assert at_once.k.dtype == at_once.v.dtype == torch.bfloat16 and at_once.tokens == 1 and not at_once.k.any() and not at_once.v.any()
            for position in range(prefix):  # what stands in front: three positions written by passes of one row, in both caches
                hidden, ids = _step_rows(drafter, sequences, 1, seed=100 + position)
                for cache in (at_once, one_by_one):
                    drafter.body(call, weights, cache, hidden, ids, position)
            before = at_once.k.clone()
            hidden, ids = _step_rows(drafter, sequences, tokens, seed=7)
            log.clear()
            rows = drafter.body(call, weights, at_once, hidden, ids, prefix)
            assert log == [(block, 0) for block, _ in drafter.wiring()[:15]] and tuple(rows.shape) == (sequences * tokens, 64) and rows.dtype == torch.bfloat16
            by_token = hidden.unflatten(0, (sequences, tokens)), ids.unflatten(0, (sequences, tokens))
            singles = [drafter.body(call, weights, one_by_one, by_token[0][:, t], by_token[1][:, t], prefix + t) for t in range(tokens)]
            assert _bytes(rows) == _bytes(torch.stack(singles, dim=1).flatten(0, 1)), (sequences, tokens)
            assert _bytes(at_once.k) == _bytes(one_by_one.k) and _bytes(at_once.v) == _bytes(one_by_one.v), (sequences, tokens)
            # the pass wrote the positions of its rows and no other; the cache keeps no count of its own
            written = slice(prefix, prefix + tokens)
            assert torch.equal(at_once.k[:, :, :prefix], before[:, :, :prefix]) and not at_once.k[:, :, prefix + tokens :].any() and at_once.tokens == 1
            assert bool(at_once.k[:, :, written].any(dim=-1).all()) and bool(at_once.v[:, :, written].any(dim=-1).all())
            # a row reads what stands in front of it and nothing behind it
            changed = hidden.clone()
            changed[tokens - 1 :: tokens] += 1.0  # the last row of every sequence
            again = drafter.body(call, weights, drafter.state(sequences, capacity, "cpu"), changed, ids, prefix).unflatten(0, (sequences, tokens))
            empty = drafter.body(call, weights, drafter.state(sequences, capacity, "cpu"), hidden, ids, prefix).unflatten(0, (sequences, tokens))
            assert _bytes(again[:, : tokens - 1]) == _bytes(empty[:, : tokens - 1]) and _bytes(again[:, tokens - 1]) != _bytes(empty[:, tokens - 1])
            assert _bytes(empty) != _bytes(rows.unflatten(0, (sequences, tokens))), "the same rows behind an empty cache: other rows"
            assert _bytes(empty[:, 0]) != _bytes(empty[:, 1]), "two positions, two rows"

    # the head on one row per sequence: three calls, ids of the draft vocabulary, the same for a sequence alone
    log.clear()
    proposed = drafter.propose(call, weights, rows.unflatten(0, (2, 6))[:, 4])
    assert log == [(block, 0) for block, _ in drafter.wiring()[15:]] and proposed.dtype == torch.int32 and tuple(proposed.shape) == (2,)
    assert set(proposed.tolist()) <= set(drafter.vocabulary.ids)
    alone = drafter.propose(call, weights, rows.unflatten(0, (2, 6))[1:, 4])
    assert alone.tolist() == proposed.tolist()[1:]

    # what a pass refuses, each with its sentence
    cache = drafter.state(2, capacity, "cpu")
    hidden, ids = _step_rows(drafter, 2, 3, seed=9)
    refused = (
        (dict(hidden=hidden[:5]), "a draft pass takes a whole number of rows per sequence of its cache (2) and one id per row; got 5 hidden rows and 6 ids"),
        (dict(ids=ids[:4]), "a draft pass takes a whole number of rows per sequence of its cache (2) and one id per row; got 6 hidden rows and 4 ids"),
        (
            dict(hidden=hidden[:5], ids=ids[:5]),  # an id per row, and no whole number of rows per sequence
            "a draft pass takes a whole number of rows per sequence of its cache (2) and one id per row; got 5 hidden rows and 5 ids",
        ),
        (
            dict(hidden=hidden[:1], ids=ids[:1]),
            "a draft pass takes a whole number of rows per sequence of its cache (2) and one id per row; got 1 hidden rows and 1 ids",
        ),
        (dict(position=10), "the drafter's cache was built for 12 positions per sequence; this pass writes 10 to 12"),
        (dict(position=-1), "the drafter's cache was built for 12 positions per sequence; this pass writes -1 to 1"),
    )
    for changes, sentence in refused:
        operands = dict(hidden=hidden, ids=ids, position=0)
        operands.update(changes)
        with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
            drafter.body(call, weights, cache, operands["hidden"], operands["ids"], operands["position"])
    assert not cache.k.any(), "a refused pass wrote nothing"
    at_the_end = drafter.body(call, weights, cache, hidden, ids, 9)
    assert tuple(at_the_end.shape) == (6, 64) and bool(cache.k[:, :, 9:].any(dim=-1).all()), "the last three positions of the cache"
    for sequences, positions in ((0, 4), (2, 0)):
        with pytest.raises(
            ValueError, match=f"^a drafter's cache needs at least one sequence and one position; got sequences={sequences}, capacity={positions}$"
        ):
            drafter.state(sequences, positions, "cpu")

    # a step whose later passes would write past the cache is refused before its first pass: it wrote nothing
    fresh = drafter.state(2, capacity, "cpu")
    with pytest.raises(ValueError, match="^the drafter's cache was built for 12 positions per sequence; the passes of this step write up to position 12$"):
        drafter.draft(call, weights, fresh, hidden, ids, 6, 2, 5)  # three rows at 6 to 8, the row at 8 kept: passes at 9, 10, 11 and 12
    assert not fresh.k.any() and not fresh.v.any(), "a refused step wrote nothing"
    # ... what the first pass itself would write past it is the pass's own refusal, also for a step of that one pass
    with pytest.raises(ValueError, match="^the drafter's cache was built for 12 positions per sequence; this pass writes 10 to 12$"):
        drafter.draft(call, weights, fresh, hidden, ids, 10, 2, 1)
    assert not fresh.k.any(), "nor did that one"
    fits = drafter.draft(call, weights, fresh, hidden, ids, 6, 2, 4)  # the last of the four passes writes position 11
    assert tuple(fits.shape) == (2, 4) and bool(fresh.k[:, :, 6:].any(dim=-1).all()) and not fresh.k[:, :, :6].any()


def test_the_passes_of_a_step_read_what_the_step_says():
    """``QwenDenseDrafter.draft`` through a recording stand-in for the blocks, for every kept count 0 .. k at one sequence
    and at two: pass 0 is the body on what the step hands over, and of its output the row ``kept`` of every sequence goes
    to the head; pass ``j`` is handed the output row of pass ``j - 1``, the proposal of pass ``j - 1`` and the position
    ``position + kept + j``.  And every block of a pass is handed what the table of the module says: the weight of its
    call site, the residual stream where one is added, the quantisation of its operand."""
    from types import SimpleNamespace

    from model_samples.providers import ROW_QUANT, AttentionParams, Quant
    from model_samples.qwen_dense import QwenDenseDrafter
    from model_samples.qwen_dense.drafter import BODY, HEAD
    from model_samples.state import KvCache

    drafter = QwenDenseDrafter.seeded(_tiny_model(), seed=5)
    plain = ("embedding", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "input_norm", "q_norm", "k_norm", "post_norm", "final_norm", "head_ids")
    copies = {name: SimpleNamespace(name=name) for name in ("fc", "qkv", "o", "gate_up", "down", "head")}
    weights = SimpleNamespace(head_quant=Quant("nvfp4", 0.5), copies=copies, **{name: SimpleNamespace(name=name) for name in plain})
    position, k = 4, 5
    tokens = k + 1
    sites = drafter.sites()
    assert sites == tuple(site for _, site in (*BODY, *HEAD)) and [block for block, _ in drafter.wiring()] == [block for block, _ in (*BODY, *HEAD)]

    def run(sequences, kept, passes=k):
        """One ``draft`` on stand-in blocks: ``(drafts, calls, made)`` -- the calls as (block, pass, operands), and per
        (block, pass, site) what the stand-in handed back."""
        calls, made, rows = [], {}, [0]
        cache = KvCache(torch.zeros((sequences, 1, 32, 2), dtype=torch.bfloat16), torch.zeros((sequences, 1, 32, 2), dtype=torch.bfloat16))

        def call(block, index, *operands):
            site = sites[len(calls) % len(sites)]  # of this call, by its place in the pass
            calls.append((block, index, operands))
            if block == "draft.embedding":
                rows[0] = operands[0].shape[0]  # the rows of this pass: one per id
            # (a block with several results hands them back as a list here: the pass unpacks either)
            if block == "draft.final_norm":  # the output rows of the pass: row r of pass j holds 100 (j + 1) + r
                rows_out = torch.arange(rows[0], dtype=torch.float32).unsqueeze(1).expand(rows[0], 4) + 100.0 * (index + 1)
                out = [None, rows_out, "the stream behind the final norm"]
            elif block == "draft.token_choice":
                out = torch.full((sequences,), 1000 + index, dtype=torch.int32)
            elif block == "draft.norm_quantize":
                out = [f"codes of {site} in pass {index}", f"normed rows of {site}", f"the stream behind {site} in pass {index}"]
            elif block == "draft.attention.prepare":
                out = [f"q of pass {index}", f"gate of pass {index}"]
            else:
                out = f"{block} {site} of pass {index}".replace("  ", " ")
            made[(block, index, site)] = out
            return out

        hidden = torch.arange(sequences * tokens, dtype=torch.float32).unsqueeze(1).expand(sequences * tokens, 4).to(torch.bfloat16)
        ids = torch.arange(sequences * tokens, dtype=torch.int32) + 50
        drafts = drafter.draft(call, weights, cache, hidden, ids, position, kept, passes)
        return drafts, calls, made, (cache, hidden, ids)

    for sequences in (1, 2):
        for kept in range(tokens):
            drafts, calls, made, (cache, hidden, ids) = run(sequences, kept)
            assert drafts.dtype == torch.int32 and drafts.tolist() == [[1000 + j for j in range(k)]] * sequences, "one column per pass, in order"
            assert [(block, index) for block, index, _ in calls] == [(block, index) for index in range(k) for block, _ in (*BODY, *HEAD)]
            assert len(calls) == k * drafter.calls_per_pass() == k * 18
            by_pass = [{(block, site): operands for (block, index, operands), (_, site) in zip(calls, (*BODY, *HEAD) * k) if index == j} for j in range(k)]
            for j, of in enumerate(by_pass):
                rows = sequences * tokens if j == 0 else sequences
                at = position if j == 0 else position + kept + j
                # what the pass is handed: the step's rows and ids, then the pass's own output row and its own last proposal
                handed_ids, table = of[("draft.embedding", "")]
                embedded, handed_hidden, norm_e, norm_h, quant = of[("draft.pre_fc_norm", "")]
                assert table is weights.embedding and embedded == f"draft.embedding of pass {j}" and quant is ROW_QUANT
                assert norm_e is weights.pre_fc_norm_embedding and norm_h is weights.pre_fc_norm_hidden
                if j == 0:
                    assert handed_ids is ids and handed_hidden is hidden
                else:
                    assert handed_ids is made[("draft.token_choice", j - 1, "")], "the proposal of the pass before"
                    before = made[("draft.final_norm", j - 1, "")][1]
                    if j == 1:  # of pass 0's rows, the row `kept` of every sequence: a view
                        chosen = torch.tensor([[100.0 + sequence * tokens + kept] * 4 for sequence in range(sequences)])
                        assert torch.equal(handed_hidden, chosen) and handed_hidden.untyped_storage().data_ptr() == before.untyped_storage().data_ptr()
                    else:
                        assert handed_hidden is before, "the output row of the pass before"
                assert handed_ids.shape[0] == handed_hidden.shape[0] == rows
                # the attention blocks: the drafter's cache as a view of this pass's rows per sequence, at the pass's position
                qkv, view, where, params = of[("draft.attention.prepare", "")]
                query, view_core, where_core = of[("draft.attention.core", "")]
                assert view.k is cache.k and view.v is cache.v and view.tokens == rows // sequences and (where, where_core) == (at, at)
                assert view_core.k is cache.k and view_core.tokens == view.tokens and query == f"q of pass {j}" and qkv == f"draft.projection qkv of pass {j}"
                g = drafter.geometry
                assert params == AttentionParams(g.attn_heads, g.attn_kv_heads, g.attn_head_dim, g.rotary_dim, g.rope_theta, weights.q_norm, weights.k_norm)
                # every other block: its operand, the weight of its call site (handed over last, by name), the residual where one
                # is added, its quantisation
                stream = f"draft.projection fc of pass {j}"
                assert of[("draft.projection", "fc")] == (f"draft.pre_fc_norm of pass {j}", copies["fc"], "fc")
                assert of[("draft.norm_quantize", "input")] == (
                    stream,
                    None,
                    weights.input_norm,
                    ROW_QUANT,
                    "input",
                ), "the first multiply's rows ARE the stream: no add"
                assert of[("draft.projection", "qkv")] == (f"codes of input in pass {j}", copies["qkv"], "qkv")
                assert of[("draft.gate_quantize", "")] == (f"draft.attention.core of pass {j}", f"gate of pass {j}", ROW_QUANT)
                assert of[("draft.projection", "o")] == (f"draft.gate_quantize of pass {j}", copies["o"], "o")
                behind_input = f"the stream behind input in pass {j}"
                assert of[("draft.norm_quantize", "mlp")] == (f"draft.projection o of pass {j}", behind_input, weights.post_norm, ROW_QUANT, "mlp")
                assert of[("draft.projection", "gate_up")] == (f"codes of mlp in pass {j}", copies["gate_up"], "gate_up")
                assert of[("draft.silu_mul", "")] == (f"draft.projection gate_up of pass {j}",)
                assert of[("draft.activation_quantize", "")] == (f"draft.silu_mul of pass {j}", ROW_QUANT)
                assert of[("draft.projection", "down")] == (f"draft.activation_quantize of pass {j}", copies["down"], "down")
                assert of[("draft.final_norm", "")] == (f"draft.projection down of pass {j}", f"the stream behind mlp in pass {j}", weights.final_norm, None)
                # the head: one row per sequence -- of pass 0 the row `kept`, of a later pass its one row -- at the head's own quantisation
                head_rows, head_quant = of[("draft.head.quantize", "")]
                out = made[("draft.final_norm", j, "")][1]
                expected = out.unflatten(0, (sequences, tokens))[:, kept] if j == 0 else out
                assert torch.equal(head_rows, expected) and tuple(head_rows.shape) == (sequences, 4) and head_quant is weights.head_quant
                assert (
                    head_rows.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
                ), "a view of the pass's rows: nothing is copied for the row select"
                assert of[("draft.head", "")] == (f"draft.head.quantize of pass {j}", copies["head"])
                logits, listed = of[("draft.token_choice", "")]
                assert logits == f"draft.head of pass {j}" and listed is weights.head_ids
            assert not cache.k.any() and cache.tokens == 1, "the stand-ins wrote nothing, and the cache handed in keeps its own count"

    # one pass: the body, the row select, the head; and a step behind one row per sequence (the first step of a run)
    drafts, calls, _, _ = run(1, 3, passes=1)
    assert drafts.tolist() == [[1000]] and len(calls) == 18
    calls = []
    cache = KvCache(torch.zeros((2, 1, 8, 2), dtype=torch.bfloat16), torch.zeros((2, 1, 8, 2), dtype=torch.bfloat16))

    handed = {"draft.final_norm": [None, torch.zeros((2, 4)), None], "draft.norm_quantize": [None, None, None], "draft.attention.prepare": [None, None]}

    def recorded(block, index, *operands):
        calls.append((block, index, operands))
        return handed.get(block, torch.zeros(2, dtype=torch.int32))

    one_row = torch.zeros((2, 4), dtype=torch.bfloat16)
    behind_one_row = drafter.draft(recorded, weights, cache, one_row, torch.zeros(2, dtype=torch.int32), 1, 0, 3)
    assert tuple(behind_one_row.shape) == (2, 3)
    assert [operands[2] for block, _, operands in calls if block == "draft.attention.prepare"] == [1, 2, 3], "one row per sequence: positions 1, 2, 3"
    assert [operands[1].tokens for block, _, operands in calls if block == "draft.attention.core"] == [1, 1, 1]

    # what a step refuses: a kept count that is no row of a sequence, no pass, and a load without its copies
    for kept in (tokens, -1, True, 1.0, None):
        with pytest.raises(ValueError, match=re.escape(f"a draft step continues from one of the 6 rows per sequence it is handed, 0 to 5; got kept={kept!r}")):
            run(1, kept)
    for passes in (0, -2, True, 2.0):
        with pytest.raises(ValueError, match=re.escape(f"a draft step has at least one pass; got passes={passes!r}")):
            run(1, 0, passes=passes)
    weights.copies = None
    sentence = (
        "the reference blocks of a draft pass multiply on bf16 copies of the drafter's weights, and this load kept none: load the drafter with copies=True"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
        run(1, 0)
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
        drafter.propose(recorded, weights, one_row)
    # fewer rows than sequences: there is no row to keep, and the first pass says what it takes
    sentence = "a draft pass takes a whole number of rows per sequence of its cache (2) and one id per row; got 1 hidden rows and 1 ids"
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
        drafter.draft(recorded, weights, cache, one_row[:1], torch.zeros(1, dtype=torch.int32), 1, 0, 3)


def test_the_models_stored_forms_and_the_drafter_loaded_together(tmp_path):
    """One checkpoint on disk, generated here, with a drafter: the model loaded with the stored forms a plan of the stack
    ``native`` asks for, then the drafter with the model's table handed over and with the copies its reference blocks
    multiply on.  Each copy is the code values of the tensor the load keeps beside it; ``memory_bytes()`` is what is
    held; the model's stored weights are the bytes of their load after the drafter's; and a pass runs on what was loaded."""
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint
    from model_samples.blocks import DeviceFacts, plan
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.qwen_dense.drafter import MULTIPLIES, NORMS
    from model_samples.timing import Stages
    from model_samples.weights import Dequantized, dequantize

    tiny = _tiny_model()
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, QwenDenseDrafter.seeded(tiny, seed=5), 5), _config(tiny.geometry))
    (tmp_path / "ids.json").write_text(json.dumps(list(range(0, 272, 3))))
    disk = Checkpoint.describe(str(directory))
    model = QwenDense.describe(disk)
    drafter = QwenDenseDrafter.describe(disk, model, draft_vocab=str(tmp_path / "ids.json"))
    assert drafter.geometry.layer_kinds == ("full_attention",) and model.geometry.layer_kinds == ("linear_attention", "full_attention")
    assert dataclasses.replace(drafter.geometry, layer_kinds=model.geometry.layer_kinds) == model.geometry, "the model's widths, one attention layer"

    # the forms a plan of the cuDNN Frontend rows keeps on compute capability 10.0: the weights its entries multiply on; a plan of
    # ``native`` keeps three more there, the weights its library rows multiply on
    entries = plan(QwenDense.provider_table, QwenDense.regimes, DeviceFacts((10, 0)), 1, 1, "fe-entries", model=QwenDense.reduced())
    kept = tuple(name for name, block in entries.items() if block.stored_weights)
    assert kept == ("projection.fp8.in", "mlp.gate_up")
    native = plan(QwenDense.provider_table, QwenDense.regimes, DeviceFacts((10, 0)), 1, 1, "native", model=QwenDense.reduced())
    on_stored = tuple(name for name, block in native.items() if block.stored_weights)
    assert on_stored == ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up", "mlp.down", "output_head")
    loaded = model.load(disk, "cpu", Stages(), stored=kept, uncopied=kept)
    assert loaded.stored == loaded.uncopied == kept and len(loaded.alone) == 4, "two layers, two weights each, on their stored format alone"

    stages = Stages()
    weights = drafter.load(disk, "cpu", stages, embedding=loaded.embedding)
    assert list(stages.seconds) == ["read", "quantise", "merge", "dequantise"] and weights.embedding is loaded.embedding
    assert sorted(weights.copies) == ["down", "fc", "gate_up", "head", "o", "qkv"]
    values = torch.tensor([0.0 if byte & 0x7F == 0x7F else _e4m3_value(byte) for byte in range(256)], dtype=torch.float64)
    for name, _ in MULTIPLIES:  # the copy of each weight: the values of ITS codes, exact in bf16, with its scales and the codes beside it
        copy, stored = weights.copies[name], getattr(weights, name)
        assert isinstance(copy, Dequantized) and copy.data.dtype == torch.bfloat16 and copy.stored is stored and copy.scale is stored.scale
        assert torch.equal(copy.data.double(), values[stored.codes.long()]), name
        assert copy.bf16() is copy.data
    head = weights.copies["head"]
    assert head.stored is weights.head and head.scale == weights.head.scale and tuple(head.data.shape) == (91, 64) and head.data.dtype == torch.bfloat16
    assert torch.equal(
        head.data, dequantize(weights.head).data
    ), "the listed head rows as the multiply over a copy reads them (weights.dequantize has its own pin)"
    assert not torch.equal(weights.copies["fc"].data[:48, :64], weights.copies["qkv"].data), "two weights, two copies"

    # what is held, against the arithmetic on the tensor list
    def held(tensors):
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    codes = held(tensor for name, _ in MULTIPLIES for tensor in (getattr(weights, name).codes, getattr(weights, name).scale))
    norms = held(getattr(weights, name) for name, _ in NORMS)
    rows = held((weights.head.codes, weights.head.block_scales, weights.head_ids))
    assert drafter.load_bytes() == {"weights": codes, "norms": norms, "head": rows}
    elements = 64 * 128 + 48 * 64 + 64 * 16 + 192 * 64 + 64 * 96  # the five weights of the tiny geometry
    copied = {"weights": held(weights.copies[name].data for name, _ in MULTIPLIES), "head": held([head.data])}
    assert drafter.copy_bytes() == copied == {"weights": 2 * elements, "head": 2 * 91 * 64}
    cache = drafter.state(2, 40, "cpu")
    assert held(cache.tensors()) == 2560, "keys and values, two sequences, one head, 40 positions of 8 elements, 2 bytes each"
    assert drafter.memory_bytes(2, 40) == codes + norms + rows + sum(copied.values()) + 2560, "what the load holds, its copies and the cache: nothing else"
    assert drafter.memory_bytes(2, 40, copies=False) == drafter.memory_bytes(2, 40) - sum(copied.values())
    assert drafter.memory_bytes(1, 40) == drafter.memory_bytes(2, 40) - 1280 and drafter.memory_bytes(2, 20) == drafter.memory_bytes(2, 40) - 1280

    # a pass on what was loaded: proposals from the listed ids; the model's stored weights are the bytes of their load
    blocks = plan(drafter.provider_table, drafter.regimes, DeviceFacts((10, 0)), 2, 1, "native", model=drafter, source=drafter.table_source)
    hidden, ids = _step_rows(drafter, 2, 3, seed=11)
    drafts = drafter.draft(lambda block, index, *operands: blocks[block](*operands), weights, cache, hidden, ids, 5, 1, 4)
    assert tuple(drafts.shape) == (2, 4) and drafts.dtype == torch.int32 and set(drafts.flatten().tolist()) <= set(range(0, 272, 3))
    assert bool(cache.k[:, :, 5:10].any(dim=-1).all()) and not cache.k[:, :, :5].any() and not cache.k[:, :, 10:].any(), "positions 5 to 7, then 7 to 9"
    assert all(weight.intact() for _, _, weight in loaded.alone) and all(weight.data is None for _, _, weight in loaded.alone)

    # a load without the copies: the codes alone, no stage for them, and a pass names the argument
    bare_stages = Stages()
    bare = drafter.load(disk, "cpu", bare_stages, embedding=loaded.embedding, copies=False)
    assert bare.copies is None and list(bare_stages.seconds) == ["read", "quantise", "merge"]
    assert torch.equal(bare.fc.codes, weights.fc.codes) and torch.equal(bare.head.codes, weights.head.codes)
    with pytest.raises(ValueError, match="load the drafter with copies=True$"):
        drafter.body(lambda block, index, *operands: blocks[block](*operands), bare, cache, hidden, ids, 5)
    disk.close()


def test_the_readmes_speculative_loop_runs_on_the_checkpoint_the_loaders_call_described(tmp_path):
    """The speculative loop of the family README, as it is written, with the names of the loader's call above it --
    ``checkpoint``, ``model``, ``drafter`` -- for a checkpoint on disk that is generated here.  The session that loads
    is handed the checkpoint, so the table the loop runs on and the drafter's weights are the file's, and the loop's
    tokens are a plain session's on that checkpoint.  With ``checkpoint`` ``None`` a session draws seeded weights: handed
    the drafter of the checkpoint it is refused by name where it is built, and the seeded pair -- what the sentence
    under the loop says ``None`` is for -- runs the same lines."""
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint, TextIO
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    tiny = _tiny_model()
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, QwenDenseDrafter.seeded(tiny, seed=5), 5), _config(tiny.geometry))
    (tmp_path / "ids.json").write_text(json.dumps(list(range(0, 272, 3))))
    checkpoint = Checkpoint.describe(str(directory))  # the call of the loader, as the README writes it
    model = QwenDense.describe(checkpoint)
    drafter = QwenDenseDrafter.describe(checkpoint, model, draft_vocab=str(tmp_path / "ids.json"))
    part = README.read_text().split("**The speculative step.**", 1)[1].split("**Memory.** The memory sentence of a verify plan", 1)[0]
    (code,) = re.findall(r"```python\n(.*?)```", part, flags=re.S)
    prompt = torch.randint(0, 272, (5,), generator=torch.Generator().manual_seed(1))
    names = dict(model=model, text=TextIO.synthetic(272), drafter=drafter, stack="reference", device="cpu", sys=sys, prompt=prompt)

    def run(handed):
        scope = dict(names, checkpoint=handed, positions=5 + 24 + 4, limit=24)
        exec(compile(code, "the README's speculative loop", "exec"), scope)
        return scope["plain"], scope["session"], scope["new"]

    plain, session, new = run(checkpoint)
    table = checkpoint.load("embed_tokens.weight", "cpu")
    own = drafter.load(checkpoint, "cpu", embedding=table)  # the drafter as the loader's call loads it
    assert plain.checkpoint is checkpoint and torch.equal(session.weights.embedding, table), "the loop ran on the file's weights"
    assert session.draft_weights.embedding is session.weights.embedding and torch.equal(session.draft_weights.fc.codes, own.fc.codes)
    assert torch.equal(session.draft_weights.head.codes, own.head.codes) and session.draft_weights.head_ids.tolist() == list(range(0, 272, 3))
    ids = plain.prefill([prompt])
    stream = plain.read(ids)
    while len(stream) < len(new):
        ids = plain.step(ids)
        stream += plain.read(ids)
    assert len(new) >= 24 and new == stream, "the tokens of the loop are a plain session's on that checkpoint"
    # handed no checkpoint, the session that loads would draw seeded weights: the drafter of the checkpoint is refused there, by name
    refusal = "the drafter was described from a checkpoint and the weights of this session are seeded: hand the session that loads that checkpoint "
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}" + r"\(checkpoint=\), or build the drafter of the seeded model$"):
        run(None)
    # ... and the seeded pair runs the same lines with ``None``: another table, and another drafter
    names.update(model=tiny, drafter=QwenDenseDrafter.seeded(tiny))
    seeded_plain, seeded, _ = run(None)
    assert seeded_plain.checkpoint.synthetic_weights and not torch.equal(seeded.weights.embedding, table)
    assert seeded.draft_weights.embedding is seeded.weights.embedding and not torch.equal(seeded.draft_weights.fc.codes, own.fc.codes)
    checkpoint.close()


@pytest.mark.parametrize("links", [0, 1, 2])
def test_a_session_loads_the_drafter_of_the_checkpoint_its_weights_were_read_from(tmp_path, links):
    """A checkpoint on disk, generated here, with its drafter.  A session with the drafter that is handed the checkpoint
    and loads it itself (no link), that runs on the weights of the session that was handed it (one link, ``share=``),
    or on those of a session that shares that one's (two links): its drafter's tensors are the file's -- what the
    loader's call loads -- beside the table the model was loaded with.  A session in between holds no checkpoint."""
    pytest.importorskip("safetensors")
    from model_samples import Checkpoint, Session, TextIO
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.qwen_dense.drafter import MULTIPLIES, NORMS

    tiny = _tiny_model()
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, QwenDenseDrafter.seeded(tiny, seed=5), 5), _config(tiny.geometry))
    (tmp_path / "ids.json").write_text(json.dumps(list(range(0, 272, 3))))
    checkpoint = Checkpoint.describe(str(directory))
    model = QwenDense.describe(checkpoint)
    drafter = QwenDenseDrafter.describe(checkpoint, model, draft_vocab=str(tmp_path / "ids.json"))
    text, common = TextIO.synthetic(272), dict(stack="reference", device="cpu", capacity=16)
    chain = [Session(model, text, checkpoint=checkpoint, **common)]
    for _ in range(links - 1):
        chain.append(Session(model, text, share=chain[-1], **common))
    how = dict(share=chain[-1]) if links else dict(checkpoint=checkpoint)
    chain.append(Session(model, text, draft_tokens=3, drafter=drafter, **common, **how))
    for each in chain:
        each.load()
    session, table = chain[-1], checkpoint.load("embed_tokens.weight", "cpu")
    own = drafter.load(checkpoint, "cpu", embedding=table)  # the drafter as the loader's call loads it
    assert [each.checkpoint for each in chain[1:-1]] == [None] * max(links - 1, 0) and all(each.weights is chain[0].weights for each in chain[:-1])
    assert torch.equal(session.weights.embedding, table) and session.draft_weights.embedding is session.weights.embedding
    for name, _ in MULTIPLIES:
        loaded = getattr(session.draft_weights, name)
        assert torch.equal(loaded.codes, getattr(own, name).codes) and torch.equal(loaded.scale, getattr(own, name).scale), name
    assert all(torch.equal(getattr(session.draft_weights, name), getattr(own, name)) for name, _ in NORMS)
    head = session.draft_weights.head
    assert torch.equal(head.codes, own.head.codes) and torch.equal(head.block_scales, own.head.block_scales) and head.scale == own.head.scale
    assert session.draft_weights.head_ids.tolist() == list(range(0, 272, 3))
    checkpoint.close()


def test_the_readme_says_what_the_drafters_pass_is_and_what_runs_it():
    """The sentences of the family README's section on the drafter that state a count, a figure or a behaviour of the
    draft pass, each against the code: the calls of a pass, its call sites, the rule of the row quantiser, what a load
    and its cache hold at the published widths, what serves the blocks and what does not run them yet.  The section
    states no count of the launches of a pass.  And the sentences of the area README and of the guide."""
    from model_samples import providers, weights
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter, configs
    from model_samples.qwen_dense.drafter import BODY, HEAD

    section = " ".join(_readme_section().split())
    # what there is, and what does not run it
    said = (
        "**What is here today is its loader and its draft pass on reference blocks, both reached from the library**",
        'A session that is handed the drafter runs its passes in front of every verify step (`Session.speculate`, in the section "The verify step"), '
        'and `02_speculative_decode.py` is the script that drafts with it (the section "Speculative decoding").',
        "`00_decode.py` neither reads the drafter nor drafts: it declares `--speculative-config` and `--draft-vocab` to refuse them with the name of "
        "that script.",
        "The drafter has a draft pass on reference blocks, reached from the library (`QwenDenseDrafter.body`, `propose`, `draft`): "
        "every block of it is a reference chain in every stack, and its embedding lookup is PyTorch's gather.",
        "A session that is handed the drafter runs its passes (`Session.speculate`, in the next section); `02_speculative_decode.py` takes the list "
        "as `--draft-vocab FILE`, the loader as `draft_vocab=`.",
        "The drafter's blocks are held to their definitions and to each other at one row and at several, and their proposals are reported against "
        "the model's own tokens on a checkpoint; they are not compared with another implementation of the drafter in this repository "
        "(one second evaluation of its network was made by hand, outside this repository: it is among the runs below).",
        "*Against the model's tokens* (reported, no gate): the checkpoint run below, with the same pass run on wrong inputs beside it.",
        "The blocks that call them are held one layer up: every block of the drafter's plan, as the plan builds it, against the function it stands for",
        "and every block of a pass over two sequences gives, for each sequence, the bytes that sequence gives alone,",
        "No kernel serves a block of the drafter and none has read these 8-bit weights.",
        # how the pass is reached: the lines themselves are run where the README's call of the loader is run
        "A draft step from Python, behind the call of the loader above (its `model`, `drafter` and `weights`):",
        "they read nothing of the checkpoint.",
        "every row of this table serves all of them, so one plan runs a pass over any number of rows.",
        "Here the hidden rows are the caller's to supply; a session that is handed the drafter supplies its own -- the rows its model's last step "
        'left -- and runs these passes in front of every verify step (`Session.speculate`, in the section "The verify step").',
        # the cache: what a row reads of it, and what a step is refused for (held by the test of a pass over several rows)
        "so a position in front that no pass wrote is read as those zeros and nothing refuses it: filling the cache over a prompt, "
        "one body pass per position, is the caller's.",
        "A step whose passes would write past the cache is refused before its first pass.",
        # ... and who fills it over a prompt (held by the test of the speculative step: the cache behind the prompt of a session)
        "A session with the drafter does it in `prefill`.",
    )
    assert [sentence for sentence in said if sentence not in section] == []
    gone = (
        "nothing runs the loaded drafter",
        "its loader, and nothing else",
        "there is no draft step, no block and no provider row",
        "to the model's own tokens on a checkpoint;",  # the checkpoint run is reported: it holds nothing
        "no session runs a draft pass",  # a session that is handed the drafter does
        "hands them out yet",
    )
    assert [words for words in gone if words in section] == []
    # ... and no sentence of another section says that nothing drafts: a session with a drafter does, and no SCRIPT does
    whole = " ".join(README.read_text().split())
    elsewhere = (
        "the step a session makes of it and the drafter's passes (**The speculative step**, below). `02_speculative_decode.py` runs the second "
        '(the section "Speculative decoding"); both are reached from the helper layer as well,',
        "how many were kept; handed the drafter, the session drafts them itself (**The speculative step**, below).",
        "**What is not here.** Kernels for the drafter: every block of its passes but the embedding lookup is a reference chain, so no rate is "
        "printed for such a run.",
        "Without a drafter nothing decides how many tokens to keep: the caller says it.",
    )
    assert [sentence for sentence in elsewhere if sentence not in whole] == []
    assert [words for words in ("othing drafts yet", "runs nothing)", "No drafter runs", "o session drafts", "o session runs") if words in whole] == []
    assert all(callable(getattr(QwenDenseDrafter, name)) for name in ("body", "propose", "draft", "state", "memory_bytes", "sites"))
    # ... from the table: every block a reference chain in every stack, but the lookup, which is the gather
    rows = {block: rows[0] for block, rows in configs.DRAFTER_PROVIDERS.items()}
    assert all(row.fallback.startswith("reference chain") and row.stacks == configs.EVERY for block, row in rows.items() if block != "draft.embedding")
    assert (rows["draft.embedding"].key, rows["draft.embedding"].fallback) == ("torch.index_select", "") and not hasattr(QwenDenseDrafter, "checked_stacks")
    # a session takes the drafter and hands its passes the rows of its own state; the drafter knows no session
    loop = (MODELS / "model_samples" / "loop.py").read_text()
    assert not any("Session" in name for name in dir(QwenDenseDrafter)) and "drafter=None," in loop and "state.hidden, state.greedy" in loop

    # the counts of a pass
    drafter = QwenDenseDrafter.seeded(QwenDense(), rows=65536)
    assert "One pass is 18 block calls. The body is 15 of them:" in section and "The head is 3 calls on one row per sequence:" in section
    assert (drafter.calls_per_pass(), len(BODY), len(HEAD)) == (18, 15, 3)
    assert "The multiply is called at five sites and the norm + quantise at two" in section
    assert (
        (drafter.calls()["draft.projection"], drafter.calls()["draft.norm_quantize"])
        == (5, 2)
        == (len(configs.DRAFT_PROJECTION_SITES), len(configs.DRAFT_NORM_SITES))
    )
    assert "Every row of this table serves every stack, every GPU class and any row count." in section, "held by the plan test, row by row"
    assert "for every kept count 0 to 5" in section, "the stand-in test of the passes runs k = 5"
    # the rule of the row quantiser, in the section's numbers
    assert "divided by 448 and never below 1 / (448 x 512); a code is the element *multiplied by the reciprocal* of that scale" in section
    assert (
        "Where a quotient sits exactly between two codes the two rules can give two codes: the product then lands just above the midpoint and takes "
        "the code above, where the division takes the even one. A host test holds a row with two such elements."
    ) in section, "the stated row of the quantiser's test; at most midpoints the two rules give one code"
    assert "where a quotient sits exactly between two codes the two can give different codes." in section, "the weights' rule, the same statement"
    assert "the scale of the operand's row and the scale of the weight's row, the two multiplied in the accumulation type." in section
    assert (weights.E4M3_MAX, weights.PER_ROW_FLOOR) == (448.0, 512) and callable(weights.quantize_fp8_per_row) and providers.FP8_ROW == "fp8_row"
    # what a load and its cache hold at the published widths with 65,536 ids (the manifest's last line is held with the manifest)
    copies = sum(drafter.copy_bytes().values()) / 2.0**30
    assert f"{copies:.2f} GiB with 65,536 ids, the last line of the manifest. `copies=False` leaves them out." == (
        "1.42 GiB with 65,536 ids, the last line of the manifest. `copies=False` leaves them out."
    )
    assert (
        "which the reference blocks of the draft pass multiply on: 1.42 GiB with 65,536 ids, the last line of the manifest. `copies=False` leaves them out."
        in section
    )
    held = f"{drafter.memory_bytes(1, 2051) / 2.0**30:.2f} GiB at the published widths with 65,536 ids, one sequence and 2,051 positions"
    assert held.startswith("2.00 GiB") and held in section and "A pass on a load without its copies is an error that names `copies=True`." in section
    # the order of the section: the loader, the recipe, the pass, what holds it, the runs, what is not here
    anchors = (
        "**It finds the drafter's tensors",
        "**A recipe for the list.**",
        "**The draft pass.**",
        "| block | calls per pass | provider key | entry [form] |",
        "| block | call site | what differs there |",
        "**What holds the pass.**",
        "**The runs behind this section**",
        "**What is not here.**",
    )
    assert [section.find(anchor) for anchor in anchors] == sorted(section.find(anchor) for anchor in anchors) and all(anchor in section for anchor in anchors)
    # the runs of the pass, as they were printed (a GPU class, a date, a list and its protocol beside every figure)
    runs = section.split("**The runs of the draft pass**", 1)[1].split("**What is not here.**", 1)[0]
    assert runs.startswith(" (2026-11-18; PyTorch 2.13.0; a GPU of compute capability 10.0, a 148-SM part shared with other users: correctness only,")
    printed = (
        "A pass over six rows of a sequence gave the rows and the cache of six passes of one row, byte for byte, at one sequence and at two.",
        "`draft` with every kept count 0 to 5 behind six rows, and behind one row, gave the five proposals and the cache",
        "by name: the embedding lookup is PyTorch's gather, and the id of the chosen row and the concatenations launch other PyTorch kernels than the gather.",
        "A second run printed the same bytes.",
        "(reported; no line of it is a gate of this sample)",
        # a dated run names the rows it ran on: the stack has had three library rows since, and its runs need the library's package
        "The model decoded three prompts greedily on the cuDNN Frontend rows of the stack `native` (the rows `fe-entries` selects today without the attention preparation's row), "
        "70 new tokens each: the default prompt and two arithmetic word problems whose text is not part of this repository, 58, 117 and 78 prompt tokens.",
        "the 342 ids of six dump files, then the lowest ids; the default prompt is one of the three prompts behind those dumps, the other two are not.",
        "The first proposal equals the model's next token at 167 of 207 positions (46, 56 and 65 of 69 by prompt), "
        "the second at 128 of 204, the third at 94 of 201, the fourth at 70 of 198, the fifth at 54 of 195.",
        "At 4 of the 207 positions the model's token is not in the list.",
        "kept 2.37 drafted tokens a step on average over 195 positions: none at 37 of them, all five at 42.",
        "exchanged, 0 of 207; the hidden row replaced by zeros, 8 of 207; the hidden row of the position before, 108 of 207; "
        "the hidden row of the position after, 150 of 207.",
        "gave the five proposals of the pass behind one row at 36 of 36 positions.",
        "gave the pass's first proposal at 200 of 207 positions and the model's token at 166. It is not part of this repository.",
        "With a cache of 194 positions the loaded drafter held 2,135,219,200 bytes, which is `memory_bytes(1, 194)`.",
    )
    assert [sentence for sentence in printed if sentence not in runs] == []
    # ... the figures that are arithmetic are held as arithmetic: the positions of three runs of 70 new tokens, the kept counts, the bytes
    prompts, new_tokens, by_prompt, by_kept = (58, 117, 78), 70, (46, 56, 65), (37, 40, 32, 28, 16, 42)
    counted = tuple(len(prompts) * (new_tokens - j) for j in range(1, 6))  # draft j of a position needs j tokens of the run behind it
    assert counted == (207, 204, 201, 198, 195) and sum(by_prompt) == 167 and sum(by_kept) == counted[4]
    assert f"{sum(kept * positions for kept, positions in enumerate(by_kept)) / sum(by_kept):.2f}" == "2.37", "drafted tokens kept, 0 to 5, by positions"
    assert drafter.memory_bytes(1, max(prompts) + new_tokens + 5 + 2) == 2135219200 and max(prompts) + new_tokens + 5 + 2 == 194
    # no count of the launches of a pass, of any kind of them: the kernel census holds none, and the section states none
    counted = r"(?:[\d,]+|one|two|three|four|five|six|no|none) (?:kernel |gather |copy )?(?:launch(?:es)?|gathers?|kernels?|cop(?:y|ies))"
    assert re.findall(rf"\b{counted} (?:per|a|of a|of one|in a) (?:draft )?(?:pass|step|call)\b", section) == []
    assert re.findall(r"[\d,]+ (?:kernel )?launch(?:es)?\b|\blaunch(?:es)? (?:of|per) (?:a|one|the) (?:draft )?pass\b", section) == []

    # the area README and the guide
    area = " ".join((MODELS / "README.md").read_text().split())
    assert (
        "the drafter's draft pass on reference blocks, reached from the library (`QwenDenseDrafter.body`, `propose`, `draft`: every block of it a "
        "reference chain in every stack, its embedding lookup PyTorch's gather)"
    ) in area and "no session runs a pass" not in area
    assert "(`DRAFTER_PROVIDERS`: 18 block calls a pass, every block a reference chain but the embedding lookup;" in area
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    assert (
        "The list of provider keys is closed; it changes only together with a kernel, with the reference chain of a block that a kernel is to serve, or "
        "with a declared library entry." in guide
    )
    assert "changes only together with a kernel, with the reference chain of a block that a kernel is to serve, or with a declared library entry" in " ".join(
        (providers.__doc__ or "").split()
    )
    assert "\n## What building the drafter's pass taught\n" in (MODELS / "AGENTS.md").read_text()
    assert "- A count of a pass is a count of block calls, and it is the wiring's: 18, of which the body is 15 and the head 3." in guide
    assert "that a comparison of tokens could hold" in guide and "with 2,048 rows the same quantiser failed on 954 scales" in guide
    assert "Four things hold such a pass, and each sees what the others do not:" in guide and "(an error inside a function);" in guide
    assert "(an error inside a block's class: operands exchanged, another function called, an epsilon of its own);" in guide
    assert "eight made inside the classes of the new blocks passed every test." in guide, "the defects of the test above: what it is there for"
    assert "multiplies by a reciprocal that is one number per operand, a constant of the checkpoint;" in guide
    assert 'so "one in two thousand" was one in a hundred and sixty.' in guide, "the count is the gate / SiLU test's: 19 of 3,072"
    assert (
        "token at 167 of 207 positions. With the two halves in front of the first multiply exchanged it was 0, with the hidden row replaced by zeros 8" in guide
    )
    assert "and with the hidden row of the position before, 108:" in guide
    for key in ("torch_ref.pre_fc_norm_quantize", "torch_ref.gate_row_quantize", "torch_ref.silu_mul", "torch_ref.quantize", "torch.argmax+lookup"):
        assert f"`{key}`" in guide and key in providers.KEYS, key
