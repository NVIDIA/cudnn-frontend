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

One more container carries what a kernel on the stored formats hands to a block that is not one yet.
:class:`TiledBlockScaledTensor` is a 4-bit activation with its block scales in the 128 x 4 layout such a kernel writes
(:func:`tile_scale_index`).  :func:`dequantize` reads it, so the gather out of the layout is launched by the block that
consumes the operand, never by the one that produced it.

A weight is held in the forms the blocks of a run read it in (:class:`Dequantized`): its bf16 copy where a block reads
one, and its stored form where a block multiplies on that -- the merged 8-bit codes with their one scale, or the packed
4-bit codes with their block scales and their global scale, as the loader had them before it dequantised them.  A weight
that every block of the run reads on its stored format has no copy.  The reference blocks of a run's checks read copies:
for such a weight their copy is made for one call and dropped after it (:class:`OnDemand`), which also counts what
that costs.  A family's memory sentence counts what is held and the largest copy made that way.  Such a copy is made
from the bytes the weight's own block multiplies on, so it is a reference only while no block has written into them:
the loader takes a checksum of every weight it keeps that way (:func:`stored_checksum`), and the checks of a run
compare it before their first pass and after their last (``Dequantized.intact``).

A kernel may read its weight in a layout of its own.  The fused gate / up kernel of an MLP reads the merged weight with
its rows in tiles of gate rows and up rows and its block scales in the 128 x 4 layout
(:class:`InterleavedBlockScaledTensor`): the same bytes in another order, built once per weight at load by the
provider that multiplies on it.  A load then holds that ONE tensor, never the checkpoint's order beside it, and
:func:`dequantize` gives its bf16 copy back in the checkpoint's row order, which is what every reader of a copy reads.
A library multiply on the stored 4-bit format reads the rows in the checkpoint's order and the block scales in the
128 x 4 layout (:class:`TiledBlockScaledWeight`): built at load in the same way, and kept as that weight's one stored form.

The quantisers follow the public arithmetic of the two formats (round to nearest even, saturating); they are not promised to
make the midpoint choices of another quantiser.

A weight that a checkpoint stores in bf16 can be brought to 8 bits when it is loaded, with one scale per row -- per output
channel of the multiply (:func:`quantize_fp8_per_channel`).  That rule is fixed down to its division, because the codes are
what a kernel on them reads: a host test pins it against the format definition.  The rows of a step can be brought to 8
bits each at a scale of its own, formed from the row (:func:`quantize_fp8_per_row`): a third rule beside the static scale
of :func:`quantize_fp8` and the weight's, fixed down to its multiplication by the reciprocal of the scale and pinned the
same way.

``Checkpoint.synthetic`` draws every tensor on the CPU generator from a seed and the tensor's name, so a tensor's values depend
on neither the device nor the load order.

``Checkpoint.describe(directory)`` reads a checkpoint from local disk in two steps.  The description reads ``config.json``
and the header of every ``*.safetensors`` shard -- names, dtypes, shapes, offsets -- and no tensor, so a sample can plan,
state its memory need and decline before it reads a byte of weights.  It also checks that the copy is whole: every shard
the index file lists (or the shards' numbering implies) is there, and every shard holds exactly the bytes its header
describes; a cut download is an error here, not in the middle of a load.  A family then binds its tensor list to the description
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
import math
import os
import re
import struct
import sys
import time
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
    ``torch.float8_e4m3fn`` for a kernel); ``scale`` is a float, or a 1-D fp32 tensor with one scale per row: of a merged
    weight whose parts kept their scales, of a weight quantised per channel, or of the rows of a step that were each
    quantised at a scale of their own (:func:`quantize_fp8_per_row`)."""

    codes: torch.Tensor
    scale: Union[float, torch.Tensor]

    @property
    def shape(self):
        return self.codes.shape

    def rows(self, start: int, stop: int) -> "ScaledTensor":
        """The rows ``start`` to ``stop`` with their scale: a scale per row (one entry per row of the codes) is sliced with
        the codes, one scale for every row is handed on as it is."""
        scale = self.scale
        if isinstance(scale, torch.Tensor) and scale.dim() == 1 and scale.shape[0] == self.codes.shape[0]:
            scale = scale[start:stop]
        return ScaledTensor(self.codes[start:stop], scale)

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


#: the 128 x 4 block-scale layout (cuDNN's ``F8_128x4`` order): rows and block-scale columns of one tile, rows of one line
TILE_ROWS = 128
TILE_COLUMNS = 4
TILE_LINE_ROWS = 32
_TILE_INDEX: dict = {}  # (rows, blocks, device) -> the index of ``tile_scale_index``: built once per process


def tile_scale_index(rows: int, blocks: int, device) -> torch.Tensor:
    """``[rows, blocks]`` int64: the byte of the block scale of ``(row, block)`` in a scale buffer in the 128 x 4 layout.  The
    buffer holds ``ceil(rows / 128)`` bands of 128 rows, one after the other; a band holds ``ceil(blocks / 4)`` tiles of 128
    rows x 4 scale columns (512 bytes each), one after the other; inside a tile the scale of ``(r, c)`` is byte
    ``(r % 32) * 16 + (r // 32) * 4 + c``."""
    key = (int(rows), int(blocks), str(device))
    if key not in _TILE_INDEX:
        row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
        block = torch.arange(blocks, dtype=torch.int64).unsqueeze(0)
        tile_bytes = TILE_ROWS * TILE_COLUMNS
        band_bytes = (blocks + TILE_COLUMNS - 1) // TILE_COLUMNS * tile_bytes
        local = row % TILE_ROWS
        inside = (local % TILE_LINE_ROWS) * (tile_bytes // TILE_LINE_ROWS) + (local // TILE_LINE_ROWS) * TILE_COLUMNS + block % TILE_COLUMNS
        _TILE_INDEX[key] = ((row // TILE_ROWS) * band_bytes + (block // TILE_COLUMNS) * tile_bytes + inside).to(device)
    return _TILE_INDEX[key]


@dataclass
class TiledBlockScaledTensor:
    """A 4-bit activation as a kernel on the stored formats writes it: packed E2M1 codes ``[rows, n / 2]`` (``uint8``, low
    nibble first), the E4M3 block scales as code bytes in the 128 x 4 layout (:func:`tile_scale_index`; the buffer has
    ``ceil(rows / 128) * 128`` rows of ``n / 16`` bytes, and only the bytes of the ``rows`` live rows are read) and the
    global scale of :class:`BlockScaledTensor`.  A block that reads the row-major layout asks for it (``row_major``): one
    gather, launched by the block that asks."""

    codes: torch.Tensor
    tiled_scales: torch.Tensor
    scale: float

    @property
    def shape(self):
        return (*self.codes.shape[:-1], self.codes.shape[-1] * 2)

    def row_major(self) -> BlockScaledTensor:
        rows, blocks = self.codes.shape[0], self.codes.shape[-1] * 2 // BLOCK
        scales = self.tiled_scales.reshape(-1)[tile_scale_index(rows, blocks, self.codes.device)]
        return BlockScaledTensor(self.codes, scales, self.scale)

    def rows(self, start: int, stop: int) -> BlockScaledTensor:
        return self.row_major().rows(start, stop)

    def clone(self) -> BlockScaledTensor:
        return self.row_major().clone()

    def tensors(self):
        return self.row_major().tensors()


def tile_scales(block_scales: torch.Tensor) -> torch.Tensor:
    """Row-major block scales ``[rows, blocks]`` in the 128 x 4 layout (:func:`tile_scale_index`): a buffer of whole bands
    of 128 rows and whole tiles of 4 scale columns, ``[ceil(rows / 128) * 128, ceil(blocks / 4) * 4]``, zero where no row
    lives.  Whole bands of whole tiles are a permutation of dimensions; anything else is one scatter.  Either way the
    launches are those of the block that needs the layout."""
    rows, blocks = block_scales.shape
    if rows % TILE_ROWS == 0 and blocks % TILE_COLUMNS == 0:
        lines = TILE_ROWS // TILE_LINE_ROWS
        bands = block_scales.reshape(rows // TILE_ROWS, lines, TILE_LINE_ROWS, blocks // TILE_COLUMNS, TILE_COLUMNS)
        return bands.permute(0, 3, 2, 1, 4).reshape(rows, blocks).contiguous()
    shape = (-(-rows // TILE_ROWS) * TILE_ROWS, -(-blocks // TILE_COLUMNS) * TILE_COLUMNS)
    tiled = torch.zeros(shape, dtype=block_scales.dtype, device=block_scales.device)
    tiled.view(-1)[tile_scale_index(rows, blocks, block_scales.device)] = block_scales
    return tiled


def untile_scales(tiled: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    """The inverse of :func:`tile_scales` for the first ``rows`` rows: ``[rows, blocks]`` row-major block scales out of a
    buffer in the 128 x 4 layout."""
    if rows % TILE_ROWS == 0 and blocks % TILE_COLUMNS == 0 and tiled.numel() == rows * blocks:
        lines = TILE_ROWS // TILE_LINE_ROWS
        bands = tiled.reshape(rows // TILE_ROWS, blocks // TILE_COLUMNS, TILE_LINE_ROWS, lines, TILE_COLUMNS)
        return bands.permute(0, 3, 2, 1, 4).reshape(rows, blocks)
    return tiled.reshape(-1)[tile_scale_index(rows, blocks, tiled.device)]


@dataclass
class TiledBlockScaledWeight(TiledBlockScaledTensor):
    """A 4-bit weight in the layout a library multiply on the stored format reads: the packed E2M1 codes ``[out, in / 2]`` in
    the checkpoint's row order and the E4M3 block scales of those rows as code bytes in the 128 x 4 layout
    (:func:`tile_scales`).  The same bytes as the stored weight, the scale bytes in another order: a provider that
    multiplies on this layout builds it once per weight at load, and a load that keeps it keeps no other form of the stored
    weight.  A reader of a bf16 copy gets one from it (:func:`dequantize`), bit for bit the copy of the stored weight."""

    def row_major(self) -> BlockScaledTensor:
        """The weight with its block scales row by row: whole bands of whole tiles are a permutation of dimensions (every
        weight of the published widths), anything else one gather."""
        rows, blocks = self.codes.shape[0], self.codes.shape[-1] * 2 // BLOCK
        return BlockScaledTensor(self.codes, untile_scales(self.tiled_scales, rows, blocks), self.scale)

    def tensors(self):
        """What the block multiplies on, as it holds it: the bytes a checksum of this weight reads."""
        return (self.codes, self.tiled_scales)


@dataclass
class InterleavedBlockScaledTensor:
    """The merged gate / up weight of an MLP in the layout a fused gate / up kernel multiplies on: packed E2M1 codes
    ``[2 x intermediate, in / 2]`` whose rows come in tiles of ``tile`` gate rows followed by the ``tile`` up rows of the
    same channels (the checkpoint has every gate row first, then every up row), the E4M3 block scales of those rows as
    code bytes in the 128 x 4 layout (:func:`tile_scales`) and the global scale of :class:`BlockScaledTensor`.  The same
    bytes as the merged weight, in another order: a provider that multiplies on this layout builds it once per weight at
    load (its entry's own helper), and a load that keeps it keeps no other form of the stored weight.  A reader of a
    bf16 copy gets one in the CHECKPOINT's row order (:func:`dequantize`)."""

    codes: torch.Tensor
    tiled_scales: torch.Tensor
    scale: float
    tile: int = 64

    @property
    def shape(self):
        return (*self.codes.shape[:-1], self.codes.shape[-1] * 2)

    def tensors(self):
        """What the block multiplies on, as it holds it: the bytes a checksum of this weight reads."""
        return (self.codes, self.tiled_scales)


@dataclass
class Dequantized:
    """A weight as a multiply is handed it, in the forms the plans of a run read: ``data`` is the bf16 copy of its code
    values (exact), ``scale`` the scale a multiply applies to its result (a float or one scale per row), ``stored`` the
    stored tensor the copy is made from (a :class:`ScaledTensor`, a :class:`BlockScaledTensor`, or that weight in the
    layout its provider multiplies on: an :class:`InterleavedBlockScaledTensor`, a :class:`TiledBlockScaledWeight`).  A
    family's loader decides from the plans which of the two a weight has: ``stored`` is ``None`` where no block multiplies
    on the stored format, ``data`` is ``None`` where no block reads a copy.  It has one of them at least.  ``checksum`` is taken by the
    loader of a weight it keeps on its stored format alone (:func:`stored_checksum`): what its stored bytes were at the
    load, for the checks of the run to hold them to (``intact``)."""

    data: Optional[torch.Tensor]
    scale: Union[float, torch.Tensor]
    stored: Optional[object] = None
    checksum: Optional[tuple] = None

    def __post_init__(self):
        if self.data is None and self.stored is None:
            raise ValueError("a weight is held as a bf16 copy, on its stored format, or as both; this one as neither")

    def intact(self) -> bool:
        """Whether the stored tensor holds the bytes it held at the load: its checksum taken again equals the loader's.
        ``True`` for a weight the loader took none of (one that has its bf16 copy: its reference does not read the stored
        bytes).  Reads the device: for a load and for the checks of a run, never inside a step."""
        return self.checksum is None or stored_checksum(self.stored) == self.checksum

    def bf16(self) -> torch.Tensor:
        """The bf16 copy, for a multiply that reads one.  ``ValueError`` where the load kept the stored format alone: no
        block of the plans it was made for reads a copy of this weight."""
        if self.data is None:
            raise ValueError(
                "this multiply reads a bf16 copy of its weight, and the load kept that weight on its stored format alone: the weights of "
                "this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
            )
        return self.data


def _word_sums(tensor: torch.Tensor) -> torch.Tensor:
    """Three 64-bit sums over the bytes of one tensor, each row read as little-endian 64-bit words (a last word that is
    not whole is filled with zeros): all words, the row sums weighted by the row, the column sums weighted by the column.
    Each wraps at 64 bits.  A function of the bytes and the row length alone, however the tensor lies in its storage."""
    tensor = tensor.contiguous()
    tensor = tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor.reshape(-1, 1)
    per_word = 8 // tensor.element_size() if 8 % tensor.element_size() == 0 else 0
    if sys.byteorder == "little" and per_word and tensor.shape[1] % per_word == 0 and tensor.storage_offset() % per_word == 0:
        words = tensor.view(torch.int64)  # rows of whole words that start on a word of the storage (every weight of the published widths): no copy
    else:  # any other tensor: the same words, put together byte by byte
        raw = tensor.view(torch.uint8)
        raw = torch.nn.functional.pad(raw, (0, -raw.shape[1] % 8))
        shifts = torch.arange(0, 64, 8, dtype=torch.int64, device=raw.device)
        words = (raw.reshape(raw.shape[0], -1, 8).to(torch.int64) << shifts).sum(-1)
    rows, columns = words.sum(1), words.sum(0)

    def weighted(sums):
        return (sums * torch.arange(1, sums.numel() + 1, dtype=torch.int64, device=sums.device)).sum()

    return torch.stack((rows.sum(), weighted(rows), weighted(columns)))


def stored_checksum(stored) -> tuple:
    """A checksum of a stored weight (a :class:`ScaledTensor`, a :class:`BlockScaledTensor` or an
    :class:`InterleavedBlockScaledTensor`: the tensors as the weight's block multiplies on them): three 64-bit sums per
    tensor of it -- the codes, the block scales, a scale per row -- over its bytes read as 64-bit words: all of them,
    the row sums weighted by the row and the column sums weighted by the column; and a scale that is one float, as it
    is.  One changed byte moves the first sum; bytes exchanged between two rows, or along a row, move the others.

    The loader takes it of every weight it keeps on its stored format alone, and the checks of a run take it again
    (``Dequantized.intact``): the reference of such a weight is a copy made from these bytes, which a block that writes
    into its weight would change for both sides.  It costs three reductions per tensor and one host read per weight, at
    a load and in the checks, never in a step; it makes no copy of a tensor whose rows are whole 64-bit words."""
    scale = stored.scale
    per_row = isinstance(scale, torch.Tensor)
    sums = torch.cat([_word_sums(tensor) for tensor in (*stored.tensors(), *((scale,) if per_row else ()))]).tolist()
    return (*sums, *(() if per_row else (float(scale),)))


class OnDemand:
    """The bf16 copies a reader makes for itself of the weights a run keeps on their stored format alone -- the reference
    blocks of the checks read copies, and no block of the run's plans does -- and what they cost.

    ``operands(operands)`` returns the operands of one block call with, in place of every weight among them that has no
    copy, a handle that has one, made now from the stored form.  The handle is the caller's for that call and is dropped
    with it, so one such copy exists at a time and the weights of the session are never changed.  The copy is exact
    (:func:`dequantize`): bit for bit the copy a load would have kept, as long as the stored bytes are those of the
    load -- which is not this object's to know: the checks that use it compare the loader's checksum of every such
    weight before and after (``Dequantized.intact``).

    Counted over the life of the object: ``copies`` made, their ``bytes`` in all, the ``largest`` one (what is held at a
    time) and the ``seconds`` they took, the device synchronised at both ends of each -- a reader of this kind is a check,
    never a timed step."""

    def __init__(self):
        self.copies = self.bytes = self.largest = 0
        self.seconds = 0.0

    def operands(self, operands: tuple) -> tuple:
        if not any(isinstance(o, Dequantized) and o.data is None for o in operands):
            return operands
        return tuple(self._with_copy(o) if isinstance(o, Dequantized) and o.data is None else o for o in operands)

    def _with_copy(self, weight: Dequantized) -> Dequantized:
        device = weight.stored.codes.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        copy = dequantize(weight.stored).data
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.seconds += time.perf_counter() - start
        size = copy.numel() * copy.element_size()
        self.copies, self.bytes, self.largest = self.copies + 1, self.bytes + size, max(self.largest, size)
        return Dequantized(copy, weight.scale, weight.stored)


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> E4M3 code bytes: saturating at +-448, round to nearest even."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_values(codes: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return codes.view(torch.float8_e4m3fn).to(dtype)


def _e2m1(x: torch.Tensor, signed_zero: bool = False) -> torch.Tensor:
    """fp32 -> E2M1 code nibbles (``uint8`` in 0..15): saturating at +-6, round to nearest, ties to the even code.  A value
    that rounds to zero gets the code 0 whatever its sign; with ``signed_zero`` it keeps its sign (code 8 for a negative
    one), as a kernel that carries the sign bit through writes it."""
    mag = x.abs()
    index = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for i in range(7):  # the code index = the midpoints below the magnitude; a tie goes up from an odd code, down from an even one
        mid = (E2M1_VALUES[i] + E2M1_VALUES[i + 1]) / 2
        index += (mag >= mid) if i % 2 else (mag > mid)
    negative = torch.signbit(x) if signed_zero else torch.signbit(x) & (index > 0)
    return index + 8 * negative.to(torch.uint8)


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


#: a row of a weight quantised per channel never gets a smaller scale than ``1 / (448 * PER_CHANNEL_FLOOR)``: a row whose
#: largest magnitude is below ``1 / PER_CHANNEL_FLOOR``, an all-zero row among them, keeps that scale
PER_CHANNEL_FLOOR = 512


def quantize_fp8_per_channel(x: torch.Tensor, rows_per_chunk: int = 2048) -> ScaledTensor:
    """8-bit quantisation of a weight ``[out, in]`` with one scale per row, that is per output channel of its multiply:

    * ``scale = max|row| / 448``, a division in fp32, and never below ``1 / (448 * 512)`` formed in fp32: a row whose
      largest magnitude is below 1/512 keeps that scale, so an all-zero row has a scale and the codes 0;
    * ``code = e4m3(row / scale)``: a division in fp32 again -- not a multiplication by the reciprocal of the scale, which
      puts a quotient that sits on the midpoint of two codes on the other side on some elements --, saturating at +-448,
      round to nearest even.

    A row is quantised by itself, so the weights that feed one multiply can be concatenated before or after.  Returns a
    :class:`ScaledTensor` whose ``scale`` is the fp32 tensor ``[out]``.  ``ValueError`` for a tensor that is not ``[out,
    in]`` and for a row that holds a value that is not finite: nothing is quantised then."""
    if x.dim() != 2:
        raise ValueError(f"the per-channel 8-bit quantisation takes a weight [out, in]; got {list(x.shape)}")
    top = torch.tensor(E4M3_MAX, dtype=torch.float32, device=x.device)
    floor = torch.tensor(1.0, dtype=torch.float32, device=x.device) / (top * PER_CHANNEL_FLOOR)
    codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    scale = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    for where, part in _row_chunks(x, rows_per_chunk):
        rows = part.float()
        largest = rows.abs().amax(dim=1)
        if not bool(torch.isfinite(largest).all()):
            row = (where.start or 0) + int((~torch.isfinite(largest)).nonzero()[0])
            raise ValueError(f"row {row} of the weight holds a value that is not finite: nothing is quantised")
        scale[where] = torch.maximum(largest / top, floor)
        codes[where] = _e4m3(rows / scale[where].unsqueeze(1))
    return ScaledTensor(codes, scale)


#: a row of a step quantised at a scale of its own never gets a smaller scale than ``1 / (448 * PER_ROW_FLOOR)``
PER_ROW_FLOOR = 512


def quantize_fp8_per_row(x: torch.Tensor) -> ScaledTensor:
    """8-bit quantisation of the rows of a step ``[rows, width]``, each at a scale of its own that is formed from the row:
    the operand of a multiply whose activation scale is no constant of the checkpoint.

    * The largest magnitude of a row is taken over its bf16 values (rows handed over in another dtype are rounded to bf16
      first, and their codes are those of the rounded values);
    * ``scale = max(largest / 448, 1 / (448 * 512))``: a division in fp32, the floor formed in fp32, so a row of zeros has
      a scale and the codes 0;
    * ``code = e4m3(value * (1 / scale))``, saturating at +-448, round to nearest even: the reciprocal of the scale is
      formed in fp32 and the product is an fp32 multiplication.

    Three fp32 operations, none formed in a wider type and rounded.  This is NOT the rule of a weight quantised per
    channel (:func:`quantize_fp8_per_channel`), which divides every element by the scale: where a quotient is the
    midpoint of two codes the two rules can give two codes (the product then lands just above the midpoint and takes
    the code above, where the division takes the even one).  A row is quantised by itself: the rows of a step, quantised
    together or one by one, give the same bytes.  Nothing is read from the device.  Returns a :class:`ScaledTensor`
    whose ``scale`` is the fp32 tensor ``[rows]``.  ``ValueError`` for a tensor that is not ``[rows, width]``."""
    if x.dim() != 2:
        raise ValueError(f"the per-row 8-bit quantisation takes the rows of a step [rows, width]; got {list(x.shape)}")
    one = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    top = torch.tensor(E4M3_MAX, dtype=torch.float32, device=x.device)
    values = x.to(torch.bfloat16).float()
    scale = torch.maximum(values.abs().amax(dim=1) / top, one / (top * PER_ROW_FLOOR))
    return ScaledTensor(_e4m3(values * (one / scale).unsqueeze(1)), scale)


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
    scale.  A plain tensor is returned as it is with scale 1.  An operand a kernel on the stored formats handed over is
    brought to the stored form first, here, by the block that reads it: block scales in the 128 x 4 layout are gathered
    (:class:`TiledBlockScaledTensor`).  A merged
    gate / up weight kept in a fused kernel's layout (:class:`InterleavedBlockScaledTensor`) gives its copy in the
    checkpoint's row order."""
    if isinstance(t, TiledBlockScaledTensor):
        t = t.row_major()
    if isinstance(t, ScaledTensor):
        return Dequantized(_e4m3_values(t.codes, torch.bfloat16), t.scale)
    if isinstance(t, BlockScaledTensor):
        out = torch.empty(t.shape, dtype=torch.bfloat16, device=t.codes.device)
        lead = t.codes.shape[0] if t.codes.dim() > 1 else 1
        codes, blocks, flat = t.codes.reshape(lead, -1), t.block_scales.reshape(lead, -1), out.view(lead, -1)
        for a in range(0, lead, rows_per_chunk):  # chunked: the nibble lookup needs an index tensor
            flat[a : a + rows_per_chunk] = _block_values(codes[a : a + rows_per_chunk], blocks[a : a + rows_per_chunk])
        return Dequantized(out, t.scale)
    if isinstance(t, InterleavedBlockScaledTensor):
        return Dequantized(_interleaved_values(t, rows_per_chunk), t.scale)
    return Dequantized(t, 1.0)


def _block_values(codes: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """The code values of packed 4-bit rows (two codes per byte, the low nibble first) times the scale of their block of 16,
    as bf16: exact."""
    values = torch.stack((_e2m1_values(codes & 0xF), _e2m1_values(codes >> 4)), dim=-1).flatten(-2)
    return (values * _e4m3_values(block_scales).repeat_interleave(BLOCK, dim=-1)).to(torch.bfloat16)


def _interleaved_values(t: InterleavedBlockScaledTensor, rows_per_chunk: int) -> torch.Tensor:
    """The bf16 copy of a merged gate / up weight kept in a fused kernel's layout, in the CHECKPOINT's row order (every
    gate row, then every up row): what a block that reads a copy reads, bit for bit the copy of the merged weight.
    Chunk by chunk over whole tiles -- a tile's gate rows and its up rows go to their two places -- so the weight is
    never held in a second packed form beside the copy."""
    n, half, tile = t.codes.shape[0], t.codes.shape[0] // 2, t.tile
    blocks = t.codes.shape[1] * 2 // BLOCK
    out = torch.empty((n, t.codes.shape[1] * 2), dtype=torch.bfloat16, device=t.codes.device)
    whole = n % TILE_ROWS == 0 and blocks % TILE_COLUMNS == 0  # whole bands of whole scale tiles: a chunk's scales are a slice
    scales = None if whole else untile_scales(t.tiled_scales, n, blocks)
    unit = math.lcm(2 * tile, TILE_ROWS)  # rows of a chunk: whole weight tiles and whole scale bands
    step = max(rows_per_chunk // unit, 1) * unit
    for a in range(0, n, step):
        b = min(a + step, n)
        rows = untile_scales(t.tiled_scales[a:b], b - a, blocks) if whole else scales[a:b]
        values = _block_values(t.codes[a:b], rows).view((b - a) // (2 * tile), 2, tile, -1)
        out[a // 2 : b // 2] = values[:, 0].reshape((b - a) // 2, -1)
        out[half + a // 2 : half + b // 2] = values[:, 1].reshape((b - a) // 2, -1)
    return out


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
#: what ``Checkpoint.bind`` says a family reads, unless the family says it in its own words
MODEL_BUILD = "it reads the build with 8-bit projections and a 4-bit MLP and output head"


def read_header(path: str) -> tuple:
    """The header of one safetensors file: ``(tensors, raw header bytes, data bytes)`` with ``tensors`` name -> ``(dtype,
    shape)`` and ``data bytes`` the end of the last tensor the header describes, counted from the end of the header.  The
    format is public: eight bytes of little-endian header length, then that many bytes of JSON, then the tensors; a
    complete file holds exactly ``8 + header length + data bytes`` bytes.  No tensor is read."""
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
    tensors, data_bytes = {}, 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str) or not isinstance(entry.get("shape"), list):
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file (header entry {name!r} has no dtype and shape)")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(o) is int for o in offsets) or not 0 <= offsets[0] <= offsets[1]:
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file (header entry {name!r} has no data offsets)")
        tensors[name] = (entry["dtype"], tuple(entry["shape"]))
        data_bytes = max(data_bytes, offsets[1])
    return tensors, raw, data_bytes


#: the index file a sharded checkpoint carries beside its shards, and the way its shards are numbered
INDEX_FILE = "model.safetensors.index.json"
_NUMBERED_SHARD = re.compile(r"(.*)-(\d+)-of-(\d+)\.safetensors")


def _missing_shards(directory: str, shards: Iterable[str]) -> list:
    """The shards a checkpoint directory should hold and does not: those its index file lists, and the gaps in shards
    numbered ``-00001-of-00003``.  A ``ValueError`` where the index file is not what its name says."""
    present, wanted = set(shards), set()
    index = os.path.join(directory, INDEX_FILE)
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                listed = json.load(f)
        except ValueError as error:
            raise ValueError(f"{INDEX_FILE} is not a JSON file ({error})") from None
        files = listed.get("weight_map") if isinstance(listed, dict) else None
        if not isinstance(files, dict) or not all(isinstance(name, str) for name in files.values()):
            raise ValueError(f"{INDEX_FILE} does not hold a weight map (tensor name -> shard file)")
        wanted.update(files.values())
    for name in present:
        numbered = _NUMBERED_SHARD.fullmatch(name)
        if numbered and 0 < int(numbered.group(3)) <= 10000:
            stem, digits, of = numbered.groups()
            wanted.update(f"{stem}-{number:0{len(digits)}d}-of-{of}.safetensors" for number in range(1, int(of) + 1))
    return sorted(wanted - present)


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
        ``ValueError`` when the directory is not a checkpoint or not a whole one: a shard its index file lists (or its
        numbering implies) is not there, or a shard does not hold the bytes its header describes.  ``manifest_hash``
        identifies the checkpoint's layout: a SHA-256 over every shard's name, size and header bytes (tensor names, dtypes,
        shapes, offsets), not over its values."""
        try:
            from safetensors import safe_open  # noqa: F401  the reader of ``_read``; imported here so that its absence declines before the plan
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
        try:
            missing = _missing_shards(directory, shards)
        except ValueError as error:
            raise ValueError(f"--model {shown}: {error}") from None
        if missing:
            more = f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""
            raise ValueError(f"--model {shown}: {missing[0]} is not there{more}: an incomplete copy of the checkpoint")
        self = cls((), 0, "native", f"checkpoint {shown} (local)")
        self.synthetic_weights, self.config, self._directory = False, config, directory
        digest = hashlib.sha256()
        for shard in shards:
            path = os.path.join(directory, shard)
            tensors, raw, data_bytes = read_header(path)
            size, described = os.path.getsize(path), 8 + len(raw) + data_bytes
            if size != described:  # the reader asks for an exact cover: a cut download, or a copy that stopped early
                raise ValueError(f"--model {shown}: {shard} holds {size:,} bytes where its header describes {described:,}: an incomplete or damaged copy")
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

    def bind(self, specs: Iterable[TensorSpec], keys: Mapping[str, str], reads: str = MODEL_BUILD) -> None:
        """Attach a family's tensor list to a described checkpoint.  ``keys`` maps each spec's name to the checkpoint's own
        tensor name.  Every tensor must be there in the storage format and shape of its spec; the first that is not raises
        ``NotImplementedError`` naming it: the checkpoint is not one this family reads (``reads``: the sentence that says
        what it does read).  A second list of the same checkpoint -- its drafter's -- is bound beside the first; a name
        both lists hold must stand for the same tensor in the same format and shape, or the lists are wrong: a
        ``ValueError``.  Nothing is bound unless the whole list fits."""

        def need(key, dtype, shape, what):
            found = self.tensors.get(key)
            if found is None:
                raise NotImplementedError(f"{self.label} has no tensor {key!r} ({what}): not a checkpoint this sample reads")
            if found.dtype != dtype or (shape is not None and tuple(found.shape) != tuple(shape)):
                wanted = f"{dtype} {list(shape)}" if shape is not None else dtype
                raise NotImplementedError(
                    f"{self.label}: tensor {key!r} is {found.dtype} {list(found.shape)} where this sample reads {wanted} ({what}): {reads}"
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
        for name, spec in bound.items():
            if name in self.specs and (self.specs[name] != spec or self._keys[name] != names[name]):
                raise ValueError(
                    f"tensor {name!r} is bound to this checkpoint as another tensor already: "
                    "two tensor lists of one checkpoint share a name only for the same tensor"
                )
        self.specs, self._keys = {**self.specs, **bound}, {**self._keys, **names}

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
