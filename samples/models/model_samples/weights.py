# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weights of a model sample in the three storage formats, the merge rule and the dequantisation.

The three formats a quantised open-weight checkpoint of this family stores:

* ``bf16``: a plain tensor;
* 8-bit: E4M3 codes with a scale, ``value = code * scale`` (:class:`ScaledTensor`);
* 4-bit: E2M1 codes packed two per byte (low nibble first), one E4M3 block scale per 16 elements of the last dimension and one
  global scale, ``value = code * block_scale * scale`` (:class:`BlockScaledTensor`).

Activations travel between blocks in the same two containers, so a block that consumes an 8-bit or a 4-bit operand sees the
bytes a kernel on the stored formats would see.  :func:`dequantize` returns the code values in bf16 WITHOUT the outer scale:
an E4M3 code has a 4-bit significand and an E2M1 code times its E4M3 block scale at most 6 bits, so both are exact in bf16
(8 bits), and a multiply over these copies applies the scales to its result.

The quantisers follow the public arithmetic of the two formats (round to nearest even, saturating); they are not promised to
make the midpoint choices of another quantiser.

``Checkpoint.synthetic`` draws every tensor on the CPU generator from a seed and the tensor's name, so a tensor's values depend
on neither the device nor the load order.

``Checkpoint.describe(directory)`` reads a checkpoint from local disk in two steps.  The description reads ``config.json``
and the header of every ``*.safetensors`` shard -- names, dtypes, shapes -- and no tensor, so a sample can plan, state its
memory need and decline before it reads a byte of weights.  A family then binds its tensor list to the description
(``bind``), which checks every tensor's presence, storage format and shape; ``load`` reads one tensor through the
``safetensors`` library (no pickle, nothing executed), imported here and nowhere else.  The layout read is the one the
quantised open-weight checkpoints of this family use:

* 8-bit: ``<name>.weight`` as E4M3 codes ``[out, in]`` and ``<name>.weight_scale``, one fp32 scalar;
* 4-bit: ``<name>.weight`` as packed codes ``[out, in / 2]`` (``uint8``), ``<name>.weight_scale`` as E4M3 block scales
  ``[out, in / 16]`` and ``<name>.weight_scale_2``, one fp32 global scale;
* ``<name>.input_scale``: the static scale of the activations the weight is multiplied with, one fp32 scalar.

Nothing is downloaded: a directory that is not on local disk is an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import torch

E4M3_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16  # elements per 4-bit block scale
#: the eight magnitudes of an E2M1 code, by its three low bits (bit 3 is the sign)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass
class ScaledTensor:
    """8-bit E4M3 codes with a scale: ``value = code * scale``.  ``codes`` holds the code bytes (``uint8``; view them as
    ``torch.float8_e4m3fn`` for a kernel); ``scale`` is a float, or a 1-D tensor with one scale per row of a merged weight."""

    codes: torch.Tensor
    scale: Union[float, torch.Tensor]

    @property
    def shape(self):
        return self.codes.shape

    def rows(self, start: int, stop: int) -> "ScaledTensor":
        return ScaledTensor(self.codes[start:stop], self.scale)

    def clone(self) -> "ScaledTensor":
        return ScaledTensor(self.codes.clone(), self.scale)

    def tensors(self):
        return (self.codes,)


@dataclass
class BlockScaledTensor:
    """Packed 4-bit E2M1 codes (``uint8``, two codes per byte, low nibble first), E4M3 block scales (``uint8`` code bytes, one per
    16 elements of the last dimension) and a global scale: ``value = code * block_scale * scale``."""

    codes: torch.Tensor
    block_scales: torch.Tensor
    scale: float

    @property
    def shape(self):
        return (*self.codes.shape[:-1], self.codes.shape[-1] * 2)

    def rows(self, start: int, stop: int) -> "BlockScaledTensor":
        return BlockScaledTensor(self.codes[start:stop], self.block_scales[start:stop], self.scale)

    def clone(self) -> "BlockScaledTensor":
        return BlockScaledTensor(self.codes.clone(), self.block_scales.clone(), self.scale)

    def tensors(self):
        return (self.codes, self.block_scales)


@dataclass
class Dequantized:
    """A bf16 copy of code values (exact) and the scale a multiply applies to its result (a float or one scale per row)."""

    data: torch.Tensor
    scale: Union[float, torch.Tensor]


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E4M3 code bytes: saturating at +-448, round to nearest even."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_values(codes: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return codes.view(torch.float8_e4m3fn).to(dtype)


def _e2m1(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E2M1 code nibbles (``uint8`` in 0..15): saturating at +-6, round to nearest, ties to the even code."""
    mag = x.abs()
    index = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for i in range(7):  # the code index = the midpoints below the magnitude; a tie goes up from an odd code, down from an even one
        mid = (E2M1_VALUES[i] + E2M1_VALUES[i + 1]) / 2
        index += (mag >= mid) if i % 2 else (mag > mid)
    return index + 8 * (torch.signbit(x) & (index > 0)).to(torch.uint8)


def _e2m1_values(nibbles: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    signed = [*E2M1_VALUES, *[-v for v in E2M1_VALUES]]
    return torch.tensor(signed, dtype=dtype, device=nibbles.device)[nibbles.to(torch.int32)]


def _row_chunks(x: torch.Tensor, rows_per_chunk: int):
    """Slices of a tensor along its first dimension, so that a whole weight is never held in several temporaries at once."""
    if x.dim() < 2 or x.shape[0] <= rows_per_chunk:
        yield slice(None), x
        return
    for a in range(0, x.shape[0], rows_per_chunk):
        yield slice(a, a + rows_per_chunk), x[a : a + rows_per_chunk]


def quantize_fp8(x: torch.Tensor, scale: float, rows_per_chunk: int = 2048) -> ScaledTensor:
    """Static 8-bit quantisation: ``code = e4m3(x * (1 / scale))`` with the reciprocal formed in fp32."""
    inv = torch.tensor(1.0, dtype=torch.float32, device=x.device) / torch.tensor(float(scale), dtype=torch.float32, device=x.device)
    codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    for where, part in _row_chunks(x, rows_per_chunk):
        codes[where] = _e4m3(part.float() * inv)
    return ScaledTensor(codes, float(scale))


def quantize_nvfp4(x: torch.Tensor, scale: float, rows_per_chunk: int = 2048) -> BlockScaledTensor:
    """4-bit quantisation with a static global scale: per block of 16, ``block_scale = e4m3(max|x| / 6 / scale)`` and
    ``code = e2m1(x / (block_scale * scale))``; an all-zero block has block scale 0 and codes 0."""
    if x.shape[-1] % BLOCK:
        raise ValueError(f"the 4-bit format needs a last dimension that is a multiple of {BLOCK}; got {tuple(x.shape)}")
    g = torch.tensor(float(scale), dtype=torch.float32, device=x.device)
    packed = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8, device=x.device)
    block_codes = torch.empty((*x.shape[:-1], x.shape[-1] // BLOCK), dtype=torch.uint8, device=x.device)
    for where, part in _row_chunks(x, rows_per_chunk):
        v = part.float().unflatten(-1, (-1, BLOCK))
        codes = _e4m3(v.abs().amax(-1) / E2M1_MAX / g)
        step = _e4m3_values(codes) * g
        inv = torch.where(step != 0, 1.0 / step, torch.zeros_like(step))
        nibbles = _e2m1(v * inv.unsqueeze(-1)).flatten(-2)
        packed[where] = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
        block_codes[where] = codes
    return BlockScaledTensor(packed, block_codes, float(scale))


def dequantize(t, rows_per_chunk: int = 2048) -> Dequantized:
    """The code values of a stored tensor as a bf16 copy, without the outer scale (exact: see the module docstring), and that
    scale.  A plain tensor is returned as it is with scale 1."""
    if isinstance(t, ScaledTensor):
        return Dequantized(_e4m3_values(t.codes, torch.bfloat16), t.scale)
    if isinstance(t, BlockScaledTensor):
        out = torch.empty(t.shape, dtype=torch.bfloat16, device=t.codes.device)
        lead = t.codes.shape[0] if t.codes.dim() > 1 else 1
        codes, blocks, flat = t.codes.reshape(lead, -1), t.block_scales.reshape(lead, -1), out.view(lead, -1)
        for a in range(0, lead, rows_per_chunk):  # chunked: the nibble lookup needs an index tensor
            c = codes[a : a + rows_per_chunk]
            values = torch.stack((_e2m1_values(c & 0xF), _e2m1_values(c >> 4)), dim=-1).flatten(-2)
            scales = _e4m3_values(blocks[a : a + rows_per_chunk]).repeat_interleave(BLOCK, dim=-1)
            flat[a : a + rows_per_chunk] = (values * scales).to(torch.bfloat16)
        return Dequantized(out, t.scale)
    return Dequantized(t, 1.0)


def merge_scaled(parts: Sequence[ScaledTensor], rule: str = "largest_scale") -> ScaledTensor:
    """Merge 8-bit weights along their first dimension (the projections of one multiply: Q, K and V; ``qkv`` and ``z``).

    ``"largest_scale"`` (the default, what the stock serving engine does): every part whose scale is not the largest is
    dequantised and re-quantised at the largest scale, so the merged weight has ONE scale.  ``"own_scale"`` keeps every part's
    codes and returns one scale per row: closer to the checkpoint, and not comparable with an engine that merges."""
    if rule not in ("largest_scale", "own_scale"):
        raise ValueError(f'merge rule must be "largest_scale" or "own_scale"; got {rule!r}')
    if not parts or any(not isinstance(p, ScaledTensor) or isinstance(p.scale, torch.Tensor) for p in parts):
        raise ValueError("merge_scaled takes 8-bit tensors with one scale each")
    if rule == "own_scale":
        scales = torch.cat([torch.full((p.codes.shape[0],), float(p.scale), dtype=torch.float32, device=p.codes.device) for p in parts])
        return ScaledTensor(torch.cat([p.codes for p in parts]), scales)
    top = max(float(p.scale) for p in parts)
    merged = [p.codes if float(p.scale) == top else quantize_fp8(_e4m3_values(p.codes) * float(p.scale), top).codes for p in parts]
    return ScaledTensor(torch.cat(merged), top)


def merge_block_scaled(parts: Sequence[BlockScaledTensor]) -> BlockScaledTensor:
    """Merge 4-bit weights along their first dimension (the gate and the up projection of one multiply).  The parts must share
    their global scale: the block scales are stored relative to it, so another scale would need a re-quantisation."""
    if not parts or any(not isinstance(p, BlockScaledTensor) for p in parts):
        raise ValueError("merge_block_scaled takes 4-bit tensors")
    if any(float(p.scale) != float(parts[0].scale) for p in parts):
        raise ValueError(f"4-bit parts of one multiply must share their global scale; got {[float(p.scale) for p in parts]}")
    return BlockScaledTensor(torch.cat([p.codes for p in parts]), torch.cat([p.block_scales for p in parts]), float(parts[0].scale))


@dataclass(frozen=True)
class TensorSpec:
    """One tensor of a synthetic checkpoint: ``mean + std * normal`` drawn on the CPU generator, stored in ``fmt`` (``"bf16"``,
    ``"fp32"``, ``"fp8"``, ``"nvfp4"``) or, for ``"scalar"``, the constant ``mean``.  ``scale`` fixes the 8-bit scale or the 4-bit
    global scale (``None``: from the tensor's largest magnitude).  ``tied`` names another tensor of the same shape: this one is
    then ``std`` times a seeded permutation of that tensor's rows instead of a draw of its own."""

    name: str
    shape: tuple
    fmt: str = "bf16"
    mean: float = 0.0
    std: float = 1.0
    scale: Optional[float] = None
    tied: str = ""

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stored_bytes(self) -> int:
        if self.fmt == "fp8":
            return self.numel
        if self.fmt == "nvfp4":
            return self.numel // 2 + self.numel // BLOCK
        return self.numel * (2 if self.fmt == "bf16" else 4)


@dataclass(frozen=True)
class DiskTensor:
    """One tensor of a described checkpoint, from its shard's header: nothing of it is read."""

    dtype: str  # as the header spells it: "BF16", "F32", "F8_E4M3", "U8", ...
    shape: tuple
    file: str  # the shard's file name


#: what a described checkpoint may hold for each format of a family's tensor list: header dtype of the tensor itself
_DISK_DTYPE = {"bf16": "BF16", "fp32": "F32", "fp8": "F8_E4M3", "nvfp4": "U8", "scalar": "F32"}
REQUIREMENTS = "samples/models/requirements.txt"


def read_header(path: str) -> tuple:
    """The header of one safetensors file: ``(tensors, raw header bytes)`` with ``tensors`` name -> ``(dtype, shape)``.
    The format is public: eight bytes of little-endian header length, then that many bytes of JSON; no tensor is read."""
    with open(path, "rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file")
        (length,) = struct.unpack("<Q", first)
        if not 2 <= length <= 100 * 1024 * 1024:
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file (header length {length})")
        raw = f.read(length)
    try:
        header = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"{os.path.basename(path)} is not a safetensors file ({error})") from error
    if not isinstance(header, dict):
        raise ValueError(f"{os.path.basename(path)} is not a safetensors file (its header is not a JSON object)")
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str) or not isinstance(entry.get("shape"), list):
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file (header entry {name!r} has no dtype and shape)")
        tensors[name] = (entry["dtype"], tuple(entry["shape"]))
    return tensors, raw


class Checkpoint:
    """The tensors of one model by name.  ``Checkpoint.synthetic(model, seed)`` builds them in memory from
    ``model.tensor_specs()``; ``Checkpoint.describe(directory)`` describes a checkpoint on local disk, which a family then
    binds its tensor list to."""

    def __init__(self, specs: Iterable[TensorSpec], seed: int, formats: str, label: str):
        self.specs = {s.name: s for s in specs}
        self.seed, self.formats, self.label = int(seed), formats, label
        self.synthetic_weights = True
        self.config: dict = {}
        self.tensors: Dict[str, DiskTensor] = {}  # a described checkpoint: every tensor of every shard, by its own name
        self.files: Dict[str, int] = {}  # shard file name -> bytes
        self.manifest_hash = ""
        self._directory = ""
        self._keys: Dict[str, str] = {}  # a family's tensor name -> the checkpoint's
        self._open: dict = {}

    @classmethod
    def describe(cls, directory: str) -> "Checkpoint":
        """``config.json`` and the shard headers of a checkpoint directory on local disk; no tensor is read.

        Raises ``NotImplementedError`` (a SKIP that names the fix) when the ``safetensors`` package is missing, and
        ``ValueError`` when the directory is not a checkpoint.  ``manifest_hash`` identifies the checkpoint's layout: a
        SHA-256 over every shard's name, size and header bytes (tensor names, dtypes, shapes, offsets), not over its values."""
        try:
            import safetensors  # noqa: F401  the reader of ``load``; imported here so that its absence declines before the plan
        except ImportError as error:
            raise NotImplementedError(f"a checkpoint is read through the safetensors package ({error}): pip install -r {REQUIREMENTS}") from error
        shown = os.path.basename(os.path.normpath(directory)) or "."
        config_path = os.path.join(directory, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(f"--model {shown}: no config.json there; pass a checkpoint directory on local disk (nothing is downloaded)")
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except ValueError as error:
            raise ValueError(f"--model {shown}: config.json is not a JSON file ({error})") from None
        if not isinstance(config, dict):
            raise ValueError(f"--model {shown}: config.json does not hold a JSON object")
        shards = sorted(name for name in os.listdir(directory) if name.endswith(".safetensors"))
        if not shards:
            raise ValueError(f"--model {shown}: no *.safetensors shard there")
        self = cls((), 0, "native", f"checkpoint {shown} (local)")
        self.synthetic_weights, self.config, self._directory = False, config, directory
        digest = hashlib.sha256()
        for shard in shards:
            path = os.path.join(directory, shard)
            tensors, raw = read_header(path)
            size = os.path.getsize(path)
            self.files[shard] = size
            digest.update(shard.encode() + b"\0" + str(size).encode() + b"\0" + raw)
            for name, (dtype, shape) in tensors.items():
                if name in self.tensors:
                    raise ValueError(f"--model {shown}: tensor {name!r} is in {self.tensors[name].file} and in {shard}")
                self.tensors[name] = DiskTensor(dtype, shape, shard)
        self.manifest_hash = digest.hexdigest()[:12]
        return self

    def summary(self) -> str:
        """What was described, for a sample's header: no path, the directory's own name only."""
        total = sum(self.files.values())
        return f"{self.label}: {len(self.files)} shards, {len(self.tensors):,} tensors, {total / 1e9:.1f} GB, manifest hash {self.manifest_hash}"

    def bind(self, specs: Iterable[TensorSpec], keys: Mapping[str, str]) -> None:
        """Attach a family's tensor list to a described checkpoint.  ``keys`` maps each spec's name to the checkpoint's own
        tensor name.  Every tensor must be there in the storage format and shape of its spec; the first that is not raises
        ``NotImplementedError`` naming it: the checkpoint is not one this family reads."""

        def need(key, dtype, shape, what):
            found = self.tensors.get(key)
            if found is None:
                raise NotImplementedError(f"{self.label} has no tensor {key!r} ({what}): not a checkpoint this sample reads")
            if found.dtype != dtype or (shape is not None and tuple(found.shape) != tuple(shape)):
                wanted = f"{dtype} {list(shape)}" if shape is not None else dtype
                raise NotImplementedError(
                    f"{self.label}: tensor {key!r} is {found.dtype} {list(found.shape)} where this sample reads {wanted} ({what}): "
                    "it reads the build with 8-bit projections and a 4-bit MLP and output head"
                )

        bound, names = {}, {}
        for spec in specs:
            key = keys[spec.name]
            if spec.fmt == "scalar":
                need(key, "F32", (), "a static activation scale")
            elif spec.fmt == "fp8":
                need(key, "F8_E4M3", spec.shape, "8-bit codes")
                need(key + "_scale", "F32", (), "the 8-bit weight scale")
            elif spec.fmt == "nvfp4":
                need(key, "U8", (*spec.shape[:-1], spec.shape[-1] // 2), "packed 4-bit codes")
                need(key + "_scale", "F8_E4M3", (*spec.shape[:-1], spec.shape[-1] // BLOCK), "4-bit block scales")
                need(key + "_scale_2", "F32", (), "the 4-bit global scale")
            else:
                need(key, _DISK_DTYPE[spec.fmt], spec.shape, "a plain tensor")
            bound[spec.name], names[spec.name] = spec, key
        self.specs, self._keys = bound, names

    def _read(self, key: str) -> torch.Tensor:
        """One tensor of a described checkpoint on the CPU, through ``safetensors`` (memory-mapped shard, no pickle)."""
        from safetensors import safe_open

        shard = self.tensors[key].file
        handle = self._open.get(shard)
        if handle is None:
            handle = self._open[shard] = safe_open(os.path.join(self._directory, shard), framework="pt", device="cpu")
        return handle.get_tensor(key)

    def close(self) -> None:
        """Drop the memory-mapped shards of a described checkpoint."""
        self._open.clear()

    @classmethod
    def synthetic(cls, model, seed: int = 0, formats: str = "native") -> "Checkpoint":
        """``formats="native"`` stores every tensor in its declared format; ``"bf16"`` stores the drawn values unquantised."""
        if formats not in ("native", "bf16"):
            raise ValueError(f'formats must be "native" or "bf16"; got {formats!r}')
        return cls(model.tensor_specs(), seed, formats, f"seeded weights (seed {int(seed)})")

    @property
    def names(self):
        return tuple(self.specs)

    def bytes_of(self, names: Iterable[str]) -> int:
        """The stored bytes of these tensors: what a block on the stored formats must read."""
        return sum(self.specs[n].stored_bytes for n in names)

    def _draw(self, spec: TensorSpec) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed((self.seed * 1000003 + zlib.crc32(spec.name.encode())) % (2**63))
        if spec.tied:
            source = self.specs[spec.tied]
            if source.tied or tuple(source.shape) != tuple(spec.shape):
                raise ValueError(f"tensor {spec.name!r} is tied to {spec.tied!r}, which must be a drawn tensor of the same shape")
            return self._draw(source)[torch.randperm(spec.shape[0], generator=g)].mul_(spec.std)
        x = torch.randn(spec.shape, generator=g, dtype=torch.float32)
        if spec.std != 1.0:
            x.mul_(spec.std)
        if spec.mean:
            x.add_(spec.mean)
        return x

    def load(self, name: str, device, as_format: str = "stored"):
        """One tensor on ``device``: ``"stored"`` gives a plain tensor, a :class:`ScaledTensor` or a :class:`BlockScaledTensor`;
        ``"dequantized"`` gives the :class:`Dequantized` bf16 copy of it.  A scalar is returned as a Python float."""
        if as_format not in ("stored", "dequantized"):
            raise ValueError(f'as_format must be "stored" or "dequantized"; got {as_format!r}')
        spec = self.specs.get(name)
        if spec is None:
            raise KeyError(f"no tensor named {name!r} in this checkpoint")
        if not self.synthetic_weights:
            key = self._keys[name]
            if spec.fmt == "scalar":
                return float(self._read(key))
            if spec.fmt == "fp8":
                stored = ScaledTensor(self._read(key).view(torch.uint8).to(device), float(self._read(key + "_scale")))
            elif spec.fmt == "nvfp4":
                blocks = self._read(key + "_scale").view(torch.uint8).to(device)
                stored = BlockScaledTensor(self._read(key).to(device), blocks, float(self._read(key + "_scale_2")))
            else:
                stored = self._read(key).to(device)
            return dequantize(stored) if as_format == "dequantized" else stored
        if spec.fmt == "scalar":
            return float(spec.mean)
        x = self._draw(spec).to(device)
        fmt = spec.fmt if self.formats == "native" else ("fp32" if spec.fmt == "fp32" else "bf16")
        if fmt == "fp32":
            stored = x
        elif fmt == "bf16":
            stored = x.to(torch.bfloat16)
        elif fmt == "fp8":
            stored = quantize_fp8(x, spec.scale if spec.scale is not None else float(x.abs().max()) / E4M3_MAX)
        elif fmt == "nvfp4":
            stored = quantize_nvfp4(x, spec.scale if spec.scale is not None else float(x.abs().max()) / (E4M3_MAX * E2M1_MAX))
        else:
            raise ValueError(f"tensor {name!r}: unknown format {spec.fmt!r}")
        return dequantize(stored) if as_format == "dequantized" else stored
