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
on neither the device nor the load order.  Reading a real checkpoint from local disk arrives with the next pull request of this
area.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

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


class Checkpoint:
    """The tensors of one model by name.  ``Checkpoint.synthetic(model, seed)`` builds them in memory from ``model.tensor_specs()``."""

    def __init__(self, specs: Iterable[TensorSpec], seed: int, formats: str, label: str):
        self.specs = {s.name: s for s in specs}
        self.seed, self.formats, self.label = int(seed), formats, label
        self.synthetic_weights = True

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
