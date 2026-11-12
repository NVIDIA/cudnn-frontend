# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the weight formats of the model samples (``samples/models/model_samples/weights.py``): the bf16 copies a
multiply reads, against a decode of the stored codes written here from the format definitions -- 8-bit E4M3 with a scale;
packed 4-bit E2M1, low nibble first, with E4M3 block scales and a global scale -- with an all-zero and a saturated block,
both merge rules, and the reference multiply on a product computed by hand.  The provider stack and the reference stack of a
sample read the same copies and use the same quantisers, so the tie-out between them cannot see an error in either: these
tests are the pin.

And the reading of a checkpoint from local disk: the header of a safetensors file read without its tensors; and, where the
``safetensors`` package is installed, a checkpoint directory generated here in the layout of the quantised open-weight
checkpoints -- described, bound through the family's name map and read back bit for bit; decoded on the reference stack
on the CPU to the bytes the same tensors give from memory; its manifest hash; the declines of a checkpoint the family does
not read; and the sample's memory sentence for a checkpoint, printed before a tensor is read.  No real checkpoint, tokenizer
or template file is committed.  No GPU, no weights."""

import json
import os
import re
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
#: the eight magnitudes of an E2M1 code by its three low bits; bit 3 is the sign
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _e4m3_value(byte: int) -> float:
    """The value of one E4M3 code byte from the format: a sign bit, four exponent bits with bias 7, three mantissa bits,
    subnormals at exponent 0, no infinity."""
    sign, exponent, mantissa = byte >> 7, (byte >> 3) & 0xF, byte & 0x7
    value = (mantissa / 8.0) * 2.0**-6 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if sign else value


def _e4m3(codes):
    return torch.tensor([_e4m3_value(byte) for byte in range(256)], dtype=torch.float64)[codes.long()]


def _e4m3_step(values):
    """The spacing of E4M3 at these magnitudes."""
    return torch.pow(2.0, torch.floor(torch.log2(values.abs().clamp_min(2.0**-6))) - 3)


def _e2m1(packed):
    """Packed 4-bit codes -> values: element ``2k`` is the low nibble of byte ``k``, element ``2k + 1`` the high one."""
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    return torch.tensor([*E2M1, *(-v for v in E2M1)], dtype=torch.float64)[nibbles.long()]


def test_an_8_bit_copy_is_the_decode_of_its_codes():
    from model_samples import weights

    x = torch.randn((64, 80), generator=torch.Generator().manual_seed(1)) * 0.02
    x[0, 0], x[0, 1] = 1.0, -1.0  # beyond 448 times the scale: the quantiser saturates
    scale = 0.1 / 448.0
    stored = weights.quantize_fp8(x, scale)
    assert stored.codes.dtype == torch.uint8 and stored.codes[0, :2].tolist() == [0x7E, 0xFE], "saturated at +-448, never the NaN code"
    copy = weights.dequantize(stored)
    mine = _e4m3(stored.codes)
    assert copy.data.dtype == torch.bfloat16 and copy.scale == scale
    assert torch.equal(copy.data.double(), mine), "the copy holds the code values exactly: four significant bits in bf16"
    exact = x.double() / scale
    inside = (mine - exact).abs() <= 0.5 * _e4m3_step(mine) + 1e-6 * exact.abs()  # round to nearest, in code units
    assert bool(inside.flatten()[2:].all()) and mine[0, :2].tolist() == [448.0, -448.0]


def test_a_4_bit_copy_is_the_decode_of_its_nibbles_low_nibble_first():
    from model_samples import weights

    # one block of 16 whose values are the code values themselves: the block scale is 1 and nothing is rounded
    block = torch.tensor([[0.5, 6.0, -1.0, 1.5, 2.0, -3.0, 4.0, 0.0, -0.5, 1.0, -6.0, 3.0, -1.5, -2.0, -4.0, 0.5]])
    stored = weights.quantize_nvfp4(block, 1.0)

    def nibble(value):
        return E2M1.index(abs(value)) | (8 if value < 0 else 0)

    pairs = zip(block[0, 0::2].tolist(), block[0, 1::2].tolist())
    assert stored.codes.tolist() == [[nibble(low) | (nibble(high) << 4) for low, high in pairs]], "low nibble first"
    assert stored.block_scales.tolist() == [[0x38]], "the E4M3 code of 1"
    assert stored.shape == (1, 16) and weights.dequantize(stored).data.tolist() == block.tolist()

    x = torch.randn((64, 160), generator=torch.Generator().manual_seed(2)) * 0.02
    x[3, 16:32] = 0.0  # an all-zero block
    x[5, 0] = 50.0  # a block whose block scale saturates
    scale = 6.5 * 0.02 / (448.0 * 6.0)
    stored = weights.quantize_nvfp4(x, scale)
    copy = weights.dequantize(stored)
    block_scales = _e4m3(stored.block_scales).repeat_interleave(16, dim=-1)
    mine = _e2m1(stored.codes) * block_scales
    assert copy.data.dtype == torch.bfloat16 and copy.scale == scale
    assert torch.equal(copy.data.double(), mine), "code value times block scale is exact in bf16: at most six significant bits"
    assert int(stored.block_scales[3, 1]) == 0 and not bool(stored.codes[3, 8:16].any()) and not bool(copy.data[3, 16:32].any())
    assert int(stored.block_scales[5, 0]) == 0x7E and float(mine[5, 0]) == 6.0 * 448.0, "saturated: the largest code at the largest block scale"
    error = (mine * scale - x.double()).abs()
    error[5, :16] = 0.0
    assert bool((error <= block_scales * scale * (1.0 + 1e-5)).all()), "within one code step of its block where the block scale is not saturated"


def test_the_merge_rules():
    from model_samples import weights

    generator = torch.Generator().manual_seed(3)
    small = weights.quantize_fp8(torch.randn((8, 32), generator=generator) * 0.01, 0.01 / 100)
    large = weights.quantize_fp8(torch.randn((4, 32), generator=generator) * 0.03, 0.03 / 100)
    merged = weights.merge_scaled([small, large])  # the default rule: one scale, the largest
    assert merged.scale == large.scale and merged.codes.shape == (12, 32)
    assert torch.equal(merged.codes[8:], large.codes), "the part that has the largest scale keeps its codes"
    before, after = _e4m3(small.codes) * small.scale, _e4m3(merged.codes[:8])
    exact = before / merged.scale
    assert bool(((after - exact).abs() <= 0.5 * _e4m3_step(after) + 1e-6 * exact.abs()).all()), "the other part is re-quantised to nearest at that scale"
    assert not torch.equal(merged.codes[:8], small.codes)
    own = weights.merge_scaled([small, large], "own_scale")
    assert torch.equal(own.codes, torch.cat([small.codes, large.codes])), "every part keeps its codes"
    assert torch.allclose(own.scale, torch.tensor([small.scale] * 8 + [large.scale] * 4)) and own.scale.dtype == torch.float32
    with pytest.raises(ValueError, match="merge rule"):
        weights.merge_scaled([small, large], "smallest_scale")
    with pytest.raises(ValueError, match="one scale each"):
        weights.merge_scaled([own, large])
    gate = weights.quantize_nvfp4(torch.randn((2, 16), generator=generator), 0.1)
    up = weights.quantize_nvfp4(torch.randn((3, 16), generator=generator), 0.1)
    both = weights.merge_block_scaled([gate, up])
    assert both.scale == 0.1 and torch.equal(both.codes, torch.cat([gate.codes, up.codes]))
    assert torch.equal(both.block_scales, torch.cat([gate.block_scales, up.block_scales]))
    with pytest.raises(ValueError, match="share their global scale"):
        weights.merge_block_scaled([gate, weights.quantize_nvfp4(torch.randn((3, 16), generator=generator), 0.2)])


def test_the_reference_multiply_on_a_product_computed_by_hand():
    from model_samples import weights
    from model_samples.providers import torch_ref

    operand = weights.quantize_fp8(torch.tensor([[1.0, -2.0, 0.5, 4.0]]), 0.5)  # the codes 2, -4, 1, 8
    assert _e4m3(operand.codes).tolist() == [[2.0, -4.0, 1.0, 8.0]]
    rows = torch.tensor([[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.bfloat16)
    # (2 * 2 - 4 * 1 + 0 - 8) * 0.5 * 0.25 = -1 and (2 - 4 + 1 + 8) * 0.5 * 0.25 = 0.875
    for acc in (torch.float64, torch.float32):
        product = torch_ref.linear(operand, weights.Dequantized(rows, 0.25), acc)
        assert product.dtype == torch.bfloat16 and product.tolist() == [[-1.0, 0.875]]
    per_row = weights.Dequantized(rows, torch.tensor([0.25, 0.5]))  # a merged weight that kept one scale per row
    assert torch_ref.linear(operand, per_row, torch.float64).tolist() == [[-1.0, 1.75]]
    packed = weights.quantize_nvfp4(torch.tensor([[0.5, 6.0, -1.0, 1.5, 2.0, -3.0, 4.0, 0.0, -0.5, 1.0, -6.0, 3.0, -1.5, -2.0, -4.0, 0.5]]), 2.0)
    ones = weights.Dequantized(torch.ones((1, 16), dtype=torch.bfloat16), 1.0)
    assert torch_ref.linear(packed, ones, torch.float64).tolist() == [[0.5]], "a 4-bit operand: its code values, its block scale and its global scale"


# --------------------------------------------------------------------------- a checkpoint on local disk

PREFIX = "model.language_model."


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
    """The ``config.json`` of a checkpoint of this geometry, with the keys the published checkpoints carry."""
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
    }
    return {"model_type": "qwen3_5", "text_config": text}


def _fp32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32))


def _disk_tensors(model, source) -> dict:
    """The tensors of a seeded checkpoint in the layout a quantised checkpoint stores: the language model under one prefix,
    the output head under none; 8-bit codes with ``weight_scale``; packed 4-bit codes with block scales in ``weight_scale``
    and the global scale in ``weight_scale_2``; fp32 scalars."""
    out = {}
    for spec in model.tensor_specs():
        key = spec.name if spec.name.startswith("lm_head.") else PREFIX + spec.name
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
    return out


def _save_safetensors(path: Path, tensors: dict) -> None:
    """One shard written from the format's definition -- eight bytes of little-endian header length, the JSON header, the
    tensors' bytes in header order -- so that the test needs the ``safetensors`` package for what the helper layer uses it
    for, the reading, and nothing else."""
    spelling = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name in sorted(tensors):
        tensor = tensors[name].contiguous()
        raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
        header[name] = {"dtype": spelling[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # the header is padded with spaces to a multiple of eight bytes
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def _write_checkpoint(directory: Path, tensors: dict, config: dict, shards: int = 2) -> Path:
    """A checkpoint directory as a sharded checkpoint is published: numbered shards, the index file that maps every tensor to
    its shard, ``config.json``."""
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


class _FromMemory:
    """The seeded tensors as a checkpoint on disk gives them back: every scale rounded to the fp32 it is stored as."""

    synthetic_weights = True

    def __init__(self, source):
        self.source = source

    def load(self, name, device, as_format="stored"):
        from model_samples.weights import dequantize

        stored = self.source.load(name, device)
        if isinstance(stored, float):
            return _fp32(stored)
        if hasattr(stored, "scale"):
            stored.scale = _fp32(stored.scale)
        return dequantize(stored) if as_format == "dequantized" else stored


def test_a_safetensors_header_is_read_without_its_tensors(tmp_path):
    from model_samples.weights import read_header

    header = {
        "__metadata__": {"format": "pt"},
        "w": {"dtype": "F8_E4M3", "shape": [2, 3], "data_offsets": [0, 6]},
        "s": {"dtype": "F32", "shape": [], "data_offsets": [6, 10]},
    }
    raw = json.dumps(header).encode()
    path = tmp_path / "shard.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)  # the header alone: the ten bytes of tensors are not there, and are not asked for
    tensors, read, data_bytes = read_header(str(path))
    assert tensors == {"w": ("F8_E4M3", (2, 3)), "s": ("F32", ())} and read == raw
    assert data_bytes == 10, "where the last tensor ends: a whole file holds 8 + the header + this many bytes"

    def entry(offsets):
        body = json.dumps({"w": {"dtype": "F32", "shape": [1], **offsets}}).encode()
        return struct.pack("<Q", len(body)) + body

    for name, content in (
        ("short.safetensors", b"abc"),
        ("text.safetensors", b"this is not a tensor file at all"),
        ("json.safetensors", struct.pack("<Q", 4) + b"nope"),
        ("list.safetensors", struct.pack("<Q", 6) + b"[1, 2]"),  # JSON, but not a header
        ("entry.safetensors", struct.pack("<Q", 24) + b'{"w": {"dtype": "F32"}} '),  # an entry without a shape
        ("number.safetensors", struct.pack("<Q", 8) + b'{"w": 3}'),
        ("nowhere.safetensors", entry({})),  # an entry that does not say where its bytes are
        ("backwards.safetensors", entry({"data_offsets": [4, 0]})),
        ("fraction.safetensors", entry({"data_offsets": [0, 4.0]})),
        ("cut.safetensors", (struct.pack("<Q", len(raw)) + raw)[:-9]),  # a file that ends inside its header
    ):
        (tmp_path / name).write_bytes(content)
        with pytest.raises(ValueError, match=f"{name} is not a safetensors file") as refusal:
            read_header(str(tmp_path / name))
        assert str(tmp_path) not in str(refusal.value), "the file's own name, never a path"


def test_a_checkpoint_on_disk_is_described_bound_and_read_bit_for_bit(tmp_path):
    pytest.importorskip("safetensors")
    import model_samples
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.tie_out import bytes_equal
    from model_samples.weights import BlockScaledTensor, ScaledTensor

    tiny = _tiny_model()
    seeded = model_samples.Checkpoint.synthetic(tiny, seed=5)
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, seeded), _config(tiny.geometry))

    disk = model_samples.Checkpoint.describe(str(directory))
    assert not disk.synthetic_weights and disk.specs == {}, "a description binds nothing by itself"
    assert disk.summary().startswith("checkpoint tiny-checkpoint (local): 2 shards, ") and str(tmp_path) not in disk.summary()
    assert len(disk.manifest_hash) == 12 and disk.summary().endswith(f"manifest hash {disk.manifest_hash}")
    assert model_samples.Checkpoint.describe(str(directory)).manifest_hash == disk.manifest_hash
    assert disk._open == {}, "a description opens no shard for reading"

    model = QwenDense.describe(disk)
    assert model.geometry == tiny.geometry and model.reductions == () and model.calls_per_step() == 20
    assert tiny.logits_band_units == configs.LOGITS_BAND_UNITS == 4.0, "the seeded model's declared logits band"
    assert model.logits_band_units is configs.CHECKPOINT_LOGITS_BAND_UNITS is None, "a described checkpoint declares none: its token line is reported"
    keys = model.checkpoint_keys(disk)
    assert keys["lm_head.weight"] == "lm_head.weight" and keys["embed_tokens.weight"] == PREFIX + "embed_tokens.weight"
    assert keys["layers.0.linear_attn.in_proj_qkv.input_scale"] == PREFIX + "layers.0.linear_attn.in_proj_qkv.input_scale"
    assert set(disk.specs) == {spec.name for spec in tiny.tensor_specs()} and disk._open == {}, "bound from the headers: still nothing read"
    assert disk.bytes_of(disk.names) == seeded.bytes_of(seeded.names)

    kinds = set()
    for spec in tiny.tensor_specs():
        want, got = seeded.load(spec.name, "cpu"), disk.load(spec.name, "cpu")
        kinds.add(type(got).__name__)
        if spec.fmt == "scalar":
            assert got == _fp32(want), spec.name
            continue
        assert type(got) is type(want) and bytes_equal(got, want), spec.name
        if isinstance(got, (ScaledTensor, BlockScaledTensor)):
            assert got.scale == _fp32(want.scale) and got.codes.dtype == torch.uint8, spec.name
    assert kinds == {"float", "Tensor", "ScaledTensor", "BlockScaledTensor"}, "every storage format was read"
    copy = disk.load("layers.0.mlp.down_proj.weight", "cpu", as_format="dequantized")
    assert copy.data.dtype == torch.bfloat16 and copy.data.shape == (64, 96) and copy.scale == _fp32(seeded.load("layers.0.mlp.down_proj.weight", "cpu").scale)
    with pytest.raises(KeyError, match="no tensor named"):
        disk.load("layers.9.mlp.down_proj.weight", "cpu")

    # the reference stack on the CPU, from disk and from memory: the same tensors through the name map, the merges and the loop
    text = model_samples.TextIO.synthetic(tiny.geometry.vocab)
    prompts = [text.encode("ab" + str(s)) for s in range(2)]

    def decode(model, checkpoint):
        session = model_samples.Session(model, text, sequences=2, stack="reference", device="cpu", capacity=16, checkpoint=checkpoint)
        session.plan_table()
        session.load()
        ids = session.prefill(prompts)
        for _ in range(5):
            ids = session.step(ids)
        return session, session.record.logits[: session.record.count].clone(), session.record.outputs[: session.record.count].clone()

    from_disk, logits, ids = decode(model, disk)
    _, logits_memory, ids_memory = decode(tiny, _FromMemory(seeded))
    assert from_disk.record.count == 8 and bytes_equal(logits, logits_memory) and bytes_equal(ids, ids_memory), "a decode from disk = the decode from memory"
    assert "read" in from_disk.stages.seconds and "draw" not in from_disk.stages.seconds, "the load of a checkpoint is the stage read"
    disk.close()
    assert disk._open == {}


def test_a_checkpoint_the_family_does_not_read_is_declined_by_name(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    import model_samples
    from model_samples.qwen_dense import QwenDense

    tiny = _tiny_model()
    tensors = _disk_tensors(tiny, model_samples.Checkpoint.synthetic(tiny, seed=5))
    config = _config(tiny.geometry)
    reference = model_samples.Checkpoint.describe(str(_write_checkpoint(tmp_path / "good", tensors, config))).manifest_hash

    def described(change=None, config_change=None, name="other"):
        changed, settings = dict(tensors), json.loads(json.dumps(config))
        if change is not None:
            change(changed)
        if config_change is not None:
            config_change(settings)
        return model_samples.Checkpoint.describe(str(_write_checkpoint(tmp_path / name, changed, settings)))

    assert described().manifest_hash == reference, "the same shards under another directory name: the same manifest"
    down = PREFIX + "layers.1.mlp.down_proj.weight"
    assert described(lambda t: t.update({down: t[down].flip(0)})).manifest_hash == reference, "the manifest pins names, formats, shapes and sizes, not values"
    assert described(lambda t: t.update({"extra.weight": torch.zeros(3)})).manifest_hash != reference
    last = PREFIX + "norm.weight"  # the last name of the last shard: renamed to a name of the same length, every size and offset stays
    renamed = described(lambda t: t.update({last[:-8] + "n.weight": t.pop(last)}), name="renamed")
    good = model_samples.Checkpoint.describe(str(tmp_path / "good"))
    assert renamed.files == good.files and renamed.manifest_hash != reference, "the manifest pins the header bytes, not only shard names and sizes"

    with pytest.raises(NotImplementedError, match=rf"checkpoint other \(local\) has no tensor '{down}_scale_2' \(the 4-bit global scale\)"):
        QwenDense.describe(described(lambda t: t.pop(down + "_scale_2")))
    q = PREFIX + "layers.1.self_attn.q_proj.weight"
    with pytest.raises(NotImplementedError, match=r"q_proj.weight' is BF16 \[32, 64\] where this sample reads F8_E4M3 \[32, 64\] \(8-bit codes\)") as refusal:
        QwenDense.describe(described(lambda t: t.update({q: torch.zeros((32, 64), dtype=torch.bfloat16)})))
    assert "8-bit projections and a 4-bit MLP and output head" in str(refusal.value) and str(tmp_path) not in str(refusal.value)
    with pytest.raises(NotImplementedError, match=r"lm_head.weight' is U8 \[272, 16\] where this sample reads U8 \[272, 32\]"):
        QwenDense.describe(described(lambda t: t.update({"lm_head.weight": t["lm_head.weight"][:, :16]})))
    with pytest.raises(NotImplementedError, match="is a model of type 'llama'; this sample reads the dense Qwen 3.5 architecture"):
        QwenDense.describe(described(config_change=lambda c: c["text_config"].update(model_type="llama")))
    with pytest.raises(NotImplementedError, match="config.json has no 'linear_num_key_heads'"):
        QwenDense.describe(described(config_change=lambda c: c["text_config"].pop("linear_num_key_heads")))
    with pytest.raises(NotImplementedError, match="has an attention layer without its output gate"):
        QwenDense.describe(described(config_change=lambda c: c["text_config"].update(attn_output_gate=False)))
    second = {
        "draft." + name[len(PREFIX) :]: value
        for name, value in tensors.items()
        if name.startswith(PREFIX + "embed") or name.startswith(PREFIX + "layers.0.input")
    }
    with pytest.raises(NotImplementedError, match="not under one prefix"):
        QwenDense.describe(described(lambda t: t.update(second)))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="--model empty: no config.json there; pass a checkpoint directory on local disk"):
        model_samples.Checkpoint.describe(str(empty))
    (empty / "config.json").write_text("{}")
    with pytest.raises(ValueError, match=r"--model empty: no \*.safetensors shard there"):
        model_samples.Checkpoint.describe(str(empty))
    for content, sentence in (
        ("{not json", "--model empty: config.json is not a JSON file"),
        ("[1]", "--model empty: config.json does not hold a JSON object"),
    ):
        (empty / "config.json").write_text(content)
        with pytest.raises(ValueError, match=re.escape(sentence)) as refusal:
            model_samples.Checkpoint.describe(str(empty))
        assert str(tmp_path) not in str(refusal.value)
    monkeypatch.setitem(sys.modules, "safetensors", None)  # as if the package were not installed
    with pytest.raises(NotImplementedError, match=r"safetensors package .*: pip install -r samples/models/requirements.txt"):
        model_samples.Checkpoint.describe(str(tmp_path / "good"))


def _sample_on_stated_facts(tmp_path, monkeypatch, free):
    """The sample script as a module on stated device facts with ``free[0]`` bytes free, beside a checkpoint directory
    generated here with its text files: ``(sample, directory)``.  A tensor read or a load is an ``AssertionError``: what
    these tests look at comes before both.  What the script changes in the process leaves with the test."""
    pytest.importorskip("safetensors")
    pytest.importorskip("jinja2")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("cudnn")
    import importlib.util

    import model_samples
    from model_samples import blocks
    from tokenizers import models, pre_tokenizers

    tiny = _tiny_model()
    directory = _write_checkpoint(tmp_path / "tiny-checkpoint", _disk_tensors(tiny, model_samples.Checkpoint.synthetic(tiny, seed=5)), _config(tiny.geometry))
    words = {word: index for index, word in enumerate(["[unk]", "why", "is", "the", "sky", "blue", "?", "<|im_end|>"])}
    tokenizer = tokenizers.Tokenizer(models.WordLevel(words, unk_token="[unk]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "chat_template.jinja").write_text("{% for message in messages %}{{ message.content + ' ' }}{% endfor %}")
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 7}))

    def no_read(checkpoint, key):
        raise AssertionError(f"read tensor {key}")

    def no_load(session):
        raise AssertionError("reached a load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, free[0])))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    monkeypatch.setattr(model_samples.Checkpoint, "_read", no_read)
    monkeypatch.setattr(model_samples.Session, "load", no_load)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # the script sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_with_a_checkpoint", MODELS / "qwen_dense" / "00_decode.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample, directory


def test_the_sample_states_its_memory_need_for_a_checkpoint_before_it_reads_a_tensor(tmp_path, monkeypatch, capsys):
    """``--model`` on stated device facts: the checkpoint is described from its headers, every prompt is encoded, the plan and
    the memory sentence are printed, and a GPU with too little free memory is a SKIP with exit code 0 -- all before a tensor
    of a shard is read."""
    free = [1 * 1024**3]
    sample, directory = _sample_on_stated_facts(tmp_path, monkeypatch, free)
    arguments = ["--model", str(directory), "--prompt", "why is the sky blue ?", "--prompt", "why ?", "--sequences", "1,2"]
    code = sample.main(arguments)
    out = capsys.readouterr().out
    assert code == 0 and str(tmp_path) not in out, out
    assert out.splitlines()[0].startswith("[00] qwen_dense decode | checkpoint tiny-checkpoint (local): 2 shards, ") and "manifest hash" in out
    skip = (
        "[00] SKIP stack fe-cublas needs about 2.0 GiB on this GPU for 1 and 2 sequences, one count after the other; 1.0 GiB are free: free memory on this GPU"
    )
    assert out.splitlines()[-1].startswith(skip), out
    free[0] = 8 * 1024**3
    with pytest.raises(AssertionError, match="reached a load"):
        sample.main(arguments)
    out = capsys.readouterr().out
    assert "this stack needs about 2.0 GiB on this GPU for 1 and 2 sequences, one count after the other; 8.0 GiB are free" in out and "SKIP" not in out
    assert "stack fe-cublas, 1 sequence, cc 10.0" in out and "[00] tokenizer and chat template of checkpoint tiny-checkpoint" in out
    monkeypatch.setitem(sys.modules, "safetensors", None)  # as if the package were not installed
    code = sample.main(arguments)
    last = capsys.readouterr().out.splitlines()[-1]
    assert (
        code == 0
        and last.startswith("[00] SKIP a checkpoint is read through the safetensors package")
        and last.endswith("pip install -r samples/models/requirements.txt")
    )
    for wrong in (["--prompt", "hi"], ["--messages", "chat.json"], ["--model", str(directory), "--layers", "4"]):
        with pytest.raises(SystemExit):
            sample.main(wrong)
    capsys.readouterr()


def test_what_a_user_can_get_wrong_ends_in_an_error_line_and_never_in_a_traceback(tmp_path, monkeypatch, capsys):
    """A directory that is not one, a hub name, a file that is not what its name says, a copy of a checkpoint that is not
    whole, messages the template cannot render, options that are not options, a tokenizer of another model and a place that
    cannot be written: each is one ``[00] ERROR`` line with exit code 1, before the plan, and none names a local path."""
    sample, directory = _sample_on_stated_facts(tmp_path, monkeypatch, [8 * 1024**3])
    import tokenizers
    from tokenizers import models

    model = ["--model", str(directory)]

    def last(arguments):
        code = sample.main(arguments)
        out = capsys.readouterr().out
        assert str(tmp_path) not in out and "served by" not in out, out
        return code, out.splitlines()[-1]

    nothing = "not a directory on local disk; pass a checkpoint directory (a hub name is not fetched: nothing is downloaded; "
    nothing += "the README beside the sample says where its checkpoint comes from)"
    assert last(["--model", str(tmp_path / "nowhere")]) == (1, f"[00] ERROR --model nowhere: {nothing}")
    assert last(["--model", "some-org/Some-Model"]) == (1, f"[00] ERROR --model Some-Model: {nothing}")
    assert last(model + ["--messages", str(tmp_path / "missing.json")]) == (1, "[00] ERROR --messages missing.json: cannot be read (No such file or directory)")
    (tmp_path / "malformed.json").write_text("{not json")
    code, line = last(model + ["--messages", str(tmp_path / "malformed.json")])
    assert code == 1 and line.startswith("[00] ERROR --messages malformed.json: not a JSON file ("), line
    (tmp_path / "options.json").write_text(json.dumps({"messages": [{"role": "user", "content": "why ?"}], "chat_template_kwargs": {"tools": []}}))
    refused = '[00] ERROR --messages options.json: "chat_template_kwargs" may not hold tools; each has its own entry in the file'
    assert last(model + ["--messages", str(tmp_path / "options.json")]) == (1, refused)
    (tmp_path / "number.json").write_text(json.dumps([{"role": "user", "content": 3}]))  # the template adds text to the content
    code, line = last(model + ["--messages", str(tmp_path / "number.json")])
    assert code == 1 and line.startswith("[00] ERROR the checkpoint's chat template refused the messages: unsupported operand"), line
    (tmp_path / "blank.json").write_text(json.dumps([{"role": "user", "content": ""}]))  # rendered to a space, encoded to nothing
    nothing_to_feed = "the checkpoint's chat template and tokenizer turn it into no token; there is nothing to decode from"
    assert last(model + ["--messages", str(tmp_path / "blank.json")]) == (1, f"[00] ERROR --messages blank.json: {nothing_to_feed}")
    assert last(model + ["--prompt", "why ?", "--prompt", " "]) == (1, f"[00] ERROR --prompt 2: {nothing_to_feed}")
    for target in (tmp_path / "no-such-folder" / "ids.json", tmp_path / "folder"):
        (tmp_path / "folder").mkdir(exist_ok=True)
        for arguments in (model, []):  # with a checkpoint and for the default run
            assert last(arguments + ["--dump-tokens", str(target)]) == (1, f"[00] ERROR --dump-tokens {target.name}: cannot be written there")
        # the same file after a run, had the check before the run passed: the reason, with the file's own name
        assert sample._dump(str(target), "fe-cublas", []) in (
            f"--dump-tokens {target.name}: No such file or directory",
            f"--dump-tokens {target.name}: Is a directory",
        )
    assert sample._dump(str(tmp_path / "ids.json"), "fe-cublas", [{"sequences": 1, "prompt": 0, "prompt_ids": [[1, 2]], "ids": [[3]]}]) == ""
    written = json.loads((tmp_path / "ids.json").read_text())
    assert written["stack"] == "fe-cublas" and written["greedy"] is True and written["runs"][0]["ids"] == [[3]]
    assert written["runs"][0]["prompt_ids"] == [[1, 2]], "one prompt per sequence, as one list of produced ids per sequence"

    (directory / "generation_config.json").write_text("{not json")
    code, line = last(model)
    assert code == 1 and line.startswith("[00] ERROR --model tiny-checkpoint: generation_config.json is not a JSON file ("), line
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": "7"}))  # a string would be read letter by letter
    wrong_kind = "eos_token_id in generation_config.json is '7' where a token id or a list of token ids is expected"
    assert last(model) == (1, f"[00] ERROR --model tiny-checkpoint: {wrong_kind}")
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 7}))
    # an entry of config.json that holds something else than what it stands for: the file and the entry, not a traceback
    config = (directory / "config.json").read_text()
    for change, sentence in (
        (lambda c: c.update(text_config=None), "config.json has a text_config that is not a JSON object"),
        (lambda c: c["text_config"].update(head_dim=None), "config.json has head_dim = None where a whole number is expected"),
        (lambda c: c["text_config"].update(hidden_size="large"), "config.json has hidden_size = 'large' where a whole number is expected"),
        (lambda c: c["text_config"].update(hidden_size=64.5), "config.json has hidden_size = 64.5 where a whole number is expected"),
        (lambda c: c["text_config"].update(rms_norm_eps=True), "config.json has rms_norm_eps = True where a number is expected"),
        (lambda c: c["text_config"].update(layer_types=3), "config.json has layer_types = 3 where a list of layer names is expected"),
        (lambda c: c["text_config"].update(rope_parameters=[1]), "config.json has rope_parameters that are not a JSON object"),
        (lambda c: c["text_config"].update(linear_key_head_dim="8"), "config.json has linear_key_head_dim = '8' where a whole number is expected"),
    ):
        changed = json.loads(config)
        change(changed)
        (directory / "config.json").write_text(json.dumps(changed))
        assert last(model) == (1, f"[00] ERROR checkpoint tiny-checkpoint (local): {sentence}")
    (directory / "config.json").write_text(config)
    template = (directory / "chat_template.jinja").read_text()
    (directory / "chat_template.jinja").write_text("{% generation %}{{ messages[0].content }}{% endgeneration %}")  # a tag this environment does not have
    code, line = last(model)
    assert code == 1 and line.startswith("[00] ERROR --model tiny-checkpoint: the chat template does not compile ("), line
    (directory / "chat_template.jinja").write_bytes(template.encode() + b"\xff")
    assert last(model) == (1, "[00] ERROR --model tiny-checkpoint: chat_template.jinja is not UTF-8 text")
    (directory / "chat_template.jinja").write_text(template)
    shard = sorted(directory.glob("*.safetensors"))[0]
    kept = shard.read_bytes()
    shard.write_bytes(struct.pack("<Q", 6) + b"[1, 2]")
    assert last(model) == (1, f"[00] ERROR {shard.name} is not a safetensors file (its header is not a JSON object)")
    shard.write_bytes(kept)
    (directory / "a.safetensors").mkdir()  # a shard that cannot be opened: the system's error, with the file's own name
    assert last(model) == (1, "[00] ERROR a.safetensors: Is a directory")
    (directory / "a.safetensors").rmdir()
    # a copy that is not whole -- a shard cut short, one that holds its header alone, one with bytes its header does not
    # describe, one that is not there -- is an error that names the shard, before the plan: not a traceback in the load
    header_alone = kept[: 8 + struct.unpack("<Q", kept[:8])[0]]
    for content in (kept[:-3], header_alone, kept + b"\0"):
        shard.write_bytes(content)
        sizes = f"{shard.name} holds {len(content):,} bytes where its header describes {len(kept):,}: an incomplete or damaged copy"
        assert last(model) == (1, f"[00] ERROR --model tiny-checkpoint: {sizes}")
    shard.unlink()
    gone = f"[00] ERROR --model tiny-checkpoint: {shard.name} is not there: an incomplete copy of the checkpoint"
    assert last(model) == (1, gone), "the index file lists it, and the numbering of the shard that is there implies it"
    index = directory / "model.safetensors.index.json"
    listed = index.read_text()
    index.unlink()
    assert last(model) == (1, gone), "without the index file: the numbering alone"
    other = sorted(directory.glob("*.safetensors"))[0]
    other.rename(directory / "weights.safetensors")
    index.write_text(json.dumps({"weight_map": {"a": "weights.safetensors", "b": shard.name, "c": "third.safetensors"}}))
    two = f"[00] ERROR --model tiny-checkpoint: {shard.name} is not there (and 1 more): an incomplete copy of the checkpoint"
    assert last(model) == (1, two), "shards without numbers: the index file alone"
    for content, sentence in (("{not json", "is not a JSON file ("), (json.dumps({"weight_map": [1]}), "does not hold a weight map (")):
        index.write_text(content)
        code, line = last(model)
        assert code == 1 and line.startswith(f"[00] ERROR --model tiny-checkpoint: model.safetensors.index.json {sentence}"), line
    (directory / "weights.safetensors").rename(other)
    index.write_text(listed)
    shard.write_bytes(kept)
    words = tokenizers.Tokenizer(models.WordLevel({f"w{index}": index for index in range(300)}, unk_token="w0"))
    words.save(str(directory / "tokenizer.json"))
    mismatch = "the tokenizer has 300 ids, the model's embedding table 272 rows: not the tokenizer of this checkpoint"
    assert last(model) == (1, f"[00] ERROR --model tiny-checkpoint: {mismatch}")
    (directory / "tokenizer.json").write_text("{}")
    code, line = last(model)
    assert code == 1 and line.startswith("[00] ERROR --model tiny-checkpoint: tokenizer.json is not a tokenizer the tokenizers package reads ("), line
