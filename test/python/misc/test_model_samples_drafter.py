# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the loader of a checkpoint's drafter (``samples/models/model_samples/qwen_dense/drafter.py``) and of the
8-bit quantisation it applies when it loads (``weights.quantize_fp8_per_channel``).

* The quantisation, against the format definition: a quantiser written here from the E4M3 layout (a sign bit, four exponent
  bits with bias 7, three mantissa bits, subnormals, no infinity; nearest value, a tie to the even code, saturating), with
  the scale and the quotient as single fp32 operations.  Nothing compares the loader with a kernel that reads its codes,
  so this test is the pin.
* The draft vocabulary: each id once, in ascending order, inside the model's vocabulary, with what was dropped counted;
  every file the reader refuses, each with its own sentence; the README's recipe, run as it is written.
* A checkpoint directory generated here, with a drafter: the manifest from the headers alone; the rows cut from the output
  head against the bytes of the shard file, read here by the header's offsets and not through the reader under test; the
  8-bit weights and the order of the parts of the merged ones; every drafter the loader declines, by name.

No real checkpoint, tokenizer or template file is committed.  No GPU, no weights."""

import dataclasses
import importlib.util
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


def test_the_e4m3_values_of_this_file_are_the_formats():
    """The table the quantiser below rests on, against values that are stated, and against PyTorch's own decode of every
    code that is a number."""
    assert VALUES[:3] == [0.0, 2.0**-9, 2.0**-8] and VALUES[8] == 2.0**-6 and VALUES[0x38] == 1.0 and VALUES[0x7E] == 448.0
    assert all(low < high for low, high in zip(VALUES, VALUES[1:])), "ascending with the code: the neighbours of a value are the codes beside it"
    decoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().tolist()
    assert [decoded[byte] for byte in range(0x7F)] == VALUES and [decoded[byte] for byte in range(0x80, 0xFF)] == [-value for value in VALUES]
    assert math.isnan(decoded[0x7F]) and math.isnan(decoded[0xFF])
    assert [_code(v) for v in (21.0, 23.0, 449.0, -1e9, -0.0, 0.0, 2.0**-10, -(2.0**-10), 0.75 * 2.0**-9)] == [0x5A, 0x5C, 0x7E, 0xFE, 0x80, 0, 0, 0x80, 1]


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


def _readme_blocks() -> tuple:
    """The two fenced ``python`` blocks of the README's section on the drafter, as they are written there: the call of the
    loader, then the recipe for a draft vocabulary."""
    section = README.read_text().split("\n## The checkpoint's drafter\n", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    assert len(blocks) == 2 and "QwenDenseDrafter.describe(" in blocks[0] and "--dump-tokens" in blocks[1], "the call, then the recipe"
    return tuple(blocks)


def test_the_readmes_recipe_makes_a_draft_vocabulary_the_loader_reads(tmp_path, monkeypatch):
    """The recipe is run as it is written in the family README, on files the sample's own ``--dump-tokens`` writer wrote:
    every id the model read or wrote is listed, the rest of the list is the lowest ids not among them, and the loader
    reads the list without dropping an entry.  The same lines with fewer rows than there are ids keep the most frequent."""
    from model_samples.qwen_dense.drafter import RECIPE, DraftVocab

    assert RECIPE == 'samples/models/qwen_dense/README.md, "The checkpoint\'s drafter"' and "--dump-tokens FILE" in README.read_text()
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_for_the_recipe", MODELS / "qwen_dense" / "00_decode.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
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
    assert text[0] == "drafter of checkpoint tiny-checkpoint (local): 15 tensors of its own under 'mtp.', 3 of the model's" and len(text) == 22
    assert text[1].split() == ["tensor", "stored", "at", "load"] and text[2].split(None, 3)[:3] == ["mtp.pre_fc_norm_embedding.weight", "bf16", "[64]"]
    assert text[20] == "  " + drafter.vocabulary.text() and text[21].startswith("  loaded, besides the model's embedding table: 0.00 GiB (8-bit weights ")
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
    assert list(stages.seconds) == ["read", "quantise", "merge"], "the load of a checkpoint: read, quantise, and merge where a weight has parts"
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
    assert list(drawn.seconds) == ["draw", "quantise", "merge"]
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
    scope = {}
    exec(compile(call, "the README's call of the loader", "exec"), scope)
    assert capsys.readouterr().out == drafter.manifest_text() + "\n" and scope["drafter"] == drafter
    assert torch.equal(scope["weights"].gate_up.codes, weights.gate_up.codes) and torch.equal(scope["weights"].head.codes, weights.head.codes)


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
    assert full.manifest_text().splitlines()[-1] == (
        "  loaded, besides the model's embedding table: 0.57 GiB (8-bit weights 0.40, norm weights 0.00, head rows 0.18)"
    )

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
    section = README.read_text().split("\n## The checkpoint's drafter\n", 1)[1].split("\n## ", 1)[0]
    printed = re.findall(r"```text\n(.*?)```", section, flags=re.S)
    assert len(printed) == 1 and printed[0].rstrip("\n") == published.manifest_text()
