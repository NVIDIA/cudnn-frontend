---
layout: post
title: "The 128×4 Tiled Layout for Block Scaling Factors"
sidebar_title: "128×4 Scale Layout"
date: 2026-04-20
description: "How cuDNN expects MXFP8 and NVFP4 block scaling factors to be laid out in the 128×4 tiled format on Blackwell GPUs, and how to convert to/from row-major."
---

When you pass MXFP8 or NVFP4 tensors as inputs to cuDNN operations on Blackwell, the accompanying **block scaling factors** must be stored in a specific tiled memory layout. This post explains what that layout is, why it exists, and how to work with it.

## Background: MX Block Scaling

[MX (Microscaling)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) is an industry specification established by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm to standardize low-precision block-scaled data types. The core idea: a tensor is divided into fixed-size **blocks**, and each block shares a single **scale factor**. The higher-precision reconstruction is simply `element × scale`.

| Format | Element type | Element bits | Block size | Scale type | Scale bits |
|--------|-------------|-------------|-----------|-----------|-----------|
| MXFP8 | E5M2 / E4M3 | 8 | 32 | E8M0 | 8 |
| MXFP6 | E3M2 / E2M3 | 6 | 32 | E8M0 | 8 |
| MXFP4 | E2M1 | 4 | 32 | E8M0 | 8 |
| MXINT8 | INT8 | 8 | 32 | E8M0 | 8 |

> E*x*M*y* denotes *x* exponent bits and *y* mantissa bits. E8M0 is special — it has no sign bit and represents pure powers of 2.

**NVFP4** is an inference-oriented format that uses **double quantization**: MX-style block scaling *plus* a per-tensor FP32 scale.

| Format | Element type | Block size | Block scale type | Tensor scale type |
|--------|-------------|-----------|-----------------|------------------|
| NVFP4 | E2M1 / E0M3 | 16 | E4M3 | FP32 |
| NVFP4 | E2M1 / E0M3 | 16 | E8M0 | FP32 |

On Blackwell, **MXFP8 is the primary format for LLM training** and **NVFP4 for inference**. cuDNN operations that consume these formats require the block scaling factors to be in the **128×4 tiled layout** described below.

## Why a Special Layout?

Consider a matrix multiply where the hardware processes data in **128-row tiles**. Each row has its own set of block scale factors (one per *block_size* elements along the contraction dimension). In a naive row-major layout, scales for 128 different rows are scattered across 128 separate memory regions — loading them requires 128 cache-line accesses.

The 128×4 tiled layout solves this by **co-locating** the scale factors for 128 consecutive rows into a single contiguous memory chunk. The hardware can then load all scales for one tile in a **single coalesced memory transaction** and feed them directly to the tensor core without any gather/scatter overhead.

```
Row-major (scattered):    row 0 scales | row 1 scales | ... | row 127 scales
                          ↑ 128 separate memory regions for one tile's worth of scales

128×4 tiled (coalesced):  [ row 0..127, cols 0..3 interleaved ]
                          ↑ one contiguous 512-byte chunk
```

## The Logical Scale Tensor

Before getting into the tiled layout, let's define the **logical** scale tensor shape. For a 2D data matrix of shape `[M, K]`:

**MXFP8** (block_size = 32, E8M0 scales):

```
Scale shape: [M, ceil(K / 32)]
Each scale covers 32 consecutive elements along K.
One 128×4 tile of scales → 128 rows × 4 scale cols × 32 elements/scale = 128×128 data block.
```

**NVFP4** (block_size = 16, E4M3 scales):

```
Scale shape: [M, ceil(K / 16)]
Each scale covers 16 consecutive elements along K.
One 128×4 tile of scales → 128 rows × 4 scale cols × 16 elements/scale = 128×64 data block.
```

Both formats share the same tile geometry: **128 rows × 4 scale columns**. The only difference is how many data elements each scale factor covers.

## Tile-Internal Layout

Each 128×4 tile contains 512 scale values. They are *not* stored in simple row-major order within the tile. Instead, the 128 rows are interleaved in groups of 32, matching the warp-group structure of the Blackwell tensor core.

The coordinate mapping between a logical `(outer, inner)` position within a tile and the linear memory `offset` is:

```c
// (outer, inner) → linear offset within a 128×4 tile
// outer: row index within tile, 0–127
// inner: column index within tile, 0–3
offset = (outer % 32) * 16 + (outer / 32) * 4 + inner

// linear offset → (outer, inner)
outer = ((offset % 16) / 4) * 32 + (offset / 16)
inner = offset % 4
```

Here's what the interleaving looks like concretely. Within the tile, memory offset 0–15 contains:

```
offset 0:  (outer=0,  inner=0)     ←─ row 0, col 0
offset 1:  (outer=0,  inner=1)     ←─ row 0, col 1
offset 2:  (outer=0,  inner=2)     ←─ row 0, col 2
offset 3:  (outer=0,  inner=3)     ←─ row 0, col 3
offset 4:  (outer=32, inner=0)     ←─ row 32, col 0
offset 5:  (outer=32, inner=1)     ←─ row 32, col 1
offset 6:  (outer=32, inner=2)     ←─ row 32, col 2
offset 7:  (outer=32, inner=3)     ←─ row 32, col 3
offset 8:  (outer=64, inner=0)     ←─ row 64, col 0
offset 9:  (outer=64, inner=1)     ←─ row 64, col 1
offset 10: (outer=64, inner=2)     ←─ row 64, col 2
offset 11: (outer=64, inner=3)     ←─ row 64, col 3
offset 12: (outer=96, inner=0)     ←─ row 96, col 0
offset 13: (outer=96, inner=1)     ←─ row 96, col 1
offset 14: (outer=96, inner=2)     ←─ row 96, col 2
offset 15: (outer=96, inner=3)     ←─ row 96, col 3
```

Then offset 16–31 covers rows 1, 33, 65, 97 (each with 4 columns), and so on. The pattern: each consecutive 16-element group packs 4 rows (spaced 32 apart) × 4 columns. There are 32 such groups per tile, yielding 32 × 16 = 512 total elements.

The reason for the 32-row stride: it matches the **warp group size** on Blackwell. Each warp group processes 32 rows of data and can extract its 16 scale factors from a contiguous slice of the tile, with zero gather overhead.

```
Warp group 0 (rows 0–31):   offsets [0..3], [16..19], [32..35], ..., [496..499]
                             ↕ stride-16 within tile → contiguous per sub-group

Warp group 1 (rows 32–63):  offsets [4..7], [20..23], [36..39], ..., [500..503]
Warp group 2 (rows 64–95):  offsets [8..11], [24..27], [40..43], ..., [504..507]
Warp group 3 (rows 96–127): offsets [12..15], [28..31], [44..47], ..., [508..511]
```

## Multi-Tile Layout

Real scale tensors are larger than a single 128×4 tile. Tiles are arranged **row-major** across the full scale tensor. Given `sf_inner_dim` scale columns (padded to a multiple of 4), the starting offset of a tile at logical scale position `(sf_outer, sf_inner)` is:

```c
// sf_inner must be a multiple of 4 (tile-aligned)
tile_start = (sf_inner + sf_outer * sf_inner_dim) * 128
```

Worked example — a data matrix of shape `[256, 256]` with MXFP8 (block_size=32):

```
Logical scale shape:   [256, ceil(256/32)] = [256, 8]
Tiles needed:          ceil(256/128) × ceil(8/4) = 2 × 2 = 4 tiles

Memory layout (row-major tiles, each tile = 512 elements):

  ┌───────────────────┬───────────────────┐
  │ Tile A             │ Tile B             │
  │ rows 0–127         │ rows 0–127         │
  │ scale cols 0–3     │ scale cols 4–7     │
  ├───────────────────┼───────────────────┤
  │ Tile C             │ Tile D             │
  │ rows 128–255       │ rows 128–255       │
  │ scale cols 0–3     │ scale cols 4–7     │
  └───────────────────┴───────────────────┘

  Memory: [Tile A (512)] [Tile B (512)] [Tile C (512)] [Tile D (512)]
  Total:  2048 scale elements
```

## Padding Rules

When dimensions don't align to tile boundaries, **pad to full tiles and zero-fill** the out-of-bounds entries:

- **outer dimension** (rows): pad to a multiple of **128**
- **inner dimension** (scale columns): pad to a multiple of **4**

```python
from math import ceil

def padded_scale_shape(M, K, block_size):
    """Compute the padded scale tensor shape for 128×4 tiling."""
    sf_cols = ceil(K / block_size)
    sf_cols_padded = ceil(sf_cols / 4) * 4     # inner: multiple of 4
    sf_rows_padded = ceil(M / 128) * 128       # outer: multiple of 128
    return sf_rows_padded, sf_cols_padded

# MXFP8 example: M=500, K=192, block_size=32
# → sf_cols = 6, sf_cols_padded = 8, sf_rows_padded = 512
# → allocate [512, 8] = 4096 elements, zero-fill OOB

# NVFP4 example: M=500, K=192, block_size=16
# → sf_cols = 12, sf_cols_padded = 12, sf_rows_padded = 512
# → allocate [512, 12] = 6144 elements, zero-fill OOB
```

Additional constraints:
- Scale tensor starting addresses must be **16-byte aligned**
- The tiled layout is **not transposition-invariant** — even when data is transposed, the scale layout stays the same; you must recompute scales in the correct orientation rather than transposing the scale tensor
- Kernels may overwrite out-of-bounds slots with zeros — don't assume OOB values persist

## Putting It Together: cuDNN MXFP8 Attention

In MXFP8 attention, each of Q, K, V has shape `[B, H, S, D]` with block scaling along a specific dimension. The scale tensors adopt the 128×4 tiled layout where:

- **`outer`** = the non-scaling dimension (e.g., sequence positions for row-wise scales)
- **`inner`** = the scaling dimension divided by block_size (e.g., `ceil(D/32)` for head-dimension scales)

```
Q [B, H, S_q, D]  — row-wise scales (block along D):
  Logical:   SF_Q [B, H, S_q,          ceil(D/32)]
  Physical:  SF_Q [B, H, ceil(S_q/128)×128, ceil(ceil(D/32)/4)×4]  in 128×4 tiled layout

V [B, H, S_kv, D] — column-wise scales (block along S_kv):
  Logical:   SF_V [B, H, ceil(S_kv/32), D]
  Physical:  SF_V [B, H, ceil(ceil(S_kv/32)/128)×128, ceil(D/4)×4]  in 128×4 tiled layout
```

## Reference Implementation

Complete Python conversion between row-major and 128×4 tiled layout:

```python
import numpy as np
from math import ceil

def to_128x4_tiled(scales_2d: np.ndarray) -> np.ndarray:
    """Convert a padded [outer, inner] row-major scale tensor to 128×4 tiled layout.

    Requires: outer % 128 == 0 and inner % 4 == 0.
    """
    outer_dim, inner_dim = scales_2d.shape
    assert outer_dim % 128 == 0 and inner_dim % 4 == 0

    num_tile_rows = outer_dim // 128
    num_tile_cols = inner_dim // 4
    out = np.empty(outer_dim * inner_dim, dtype=scales_2d.dtype)

    for r in range(outer_dim):
        for c in range(inner_dim):
            tile_r, tile_c = r // 128, c // 4
            tile_base = (tile_c + tile_r * num_tile_cols) * 512

            lr, lc = r % 128, c % 4  # local coords within tile
            local_off = (lr % 32) * 16 + (lr // 32) * 4 + lc

            out[tile_base + local_off] = scales_2d[r, c]

    return out


def from_128x4_tiled(flat: np.ndarray, outer_dim: int, inner_dim: int) -> np.ndarray:
    """Convert a flat 128×4 tiled buffer back to [outer, inner] row-major."""
    assert outer_dim % 128 == 0 and inner_dim % 4 == 0

    num_tile_cols = inner_dim // 4
    scales_2d = np.empty((outer_dim, inner_dim), dtype=flat.dtype)

    for idx in range(len(flat)):
        tile_idx = idx // 512
        local_off = idx % 512

        tile_r = tile_idx // num_tile_cols
        tile_c = tile_idx % num_tile_cols

        lr = ((local_off % 16) // 4) * 32 + (local_off // 16)
        lc = local_off % 4

        r = tile_r * 128 + lr
        c = tile_c * 4 + lc
        scales_2d[r, c] = flat[idx]

    return scales_2d


def quantize_scales_for_cudnn(scales_2d: np.ndarray, block_size: int) -> np.ndarray:
    """Pad and convert a raw [M, ceil(K/block_size)] scale tensor for cuDNN."""
    M, sf_cols = scales_2d.shape
    padded_rows = ceil(M / 128) * 128
    padded_cols = ceil(sf_cols / 4) * 4

    padded = np.zeros((padded_rows, padded_cols), dtype=scales_2d.dtype)
    padded[:M, :sf_cols] = scales_2d

    return to_128x4_tiled(padded)
```

## Worked Example: Tracing the Mapping

Let's trace a small case to build intuition. Scale tensor shape `[4, 2]`, after padding to `[128, 4]`:

```
Original (4×2):         After padding (128×4):
 row 0: [a, b]          row 0:   [a, b, 0, 0]
 row 1: [c, d]          row 1:   [c, d, 0, 0]
 row 2: [e, f]          row 2:   [e, f, 0, 0]
 row 3: [g, h]          row 3:   [g, h, 0, 0]
                         rows 4–127: all zeros
```

After 128×4 tiling (single tile, 512 elements):

```
offset  0– 3:  row 0   → [a, b, 0, 0]
offset  4– 7:  row 32  → [0, 0, 0, 0]
offset  8–11:  row 64  → [0, 0, 0, 0]
offset 12–15:  row 96  → [0, 0, 0, 0]
offset 16–19:  row 1   → [c, d, 0, 0]
offset 20–23:  row 33  → [0, 0, 0, 0]
...
offset 32–35:  row 2   → [e, f, 0, 0]
...
offset 48–51:  row 3   → [g, h, 0, 0]
...
offsets 64–511: all zeros (rows 4–31 and their 32-stride partners)
```

Rows 0, 32, 64, 96 are grouped first, then 1, 33, 65, 97, and so on. Within each 4-row group, the 4 inner-dimension values are stored contiguously.

## MXFP8 vs NVFP4: Layout Comparison

While both formats share the same 128×4 tile structure, the key differences are in block size and how much data each tile covers:

| | MXFP8 | NVFP4 |
|---|---|---|
| Block size | 32 | 16 |
| Scale type | E8M0 (8-bit) | E4M3 (8-bit) or E8M0 (8-bit) |
| Data covered per tile | 128 rows × 128 cols | 128 rows × 64 cols |
| Scale cols per tile | 4 (each covers 32 elements) | 4 (each covers 16 elements) |
| Tensor-level scale | No | Yes (FP32, double quantization) |
| Use case | Training (Blackwell) | Inference (Blackwell) |

For NVFP4 with double quantization, the block scales in 128×4 tiled layout are multiplied with the per-tensor FP32 scale during dequantization:

```
dequantized_value = element × block_scale × tensor_scale
```

## Common Pitfalls

**Forgetting to pad.** If the scale dimension isn't a multiple of 4 or the spatial dimension isn't a multiple of 128, cuDNN will reject the input or produce incorrect results. Always pad first, zero-fill, then tile.

**Confusing logical and physical shapes.** The logical scale shape `[M, ceil(K/block_size)]` tells you the semantic meaning; the physical allocation `[ceil(M/128)×128, ceil(ceil(K/block_size)/4)×4]` is what you actually allocate and tile.

**Transposing scale tensors.** The 128×4 layout does not support transposition. If your operation needs the data in a transposed orientation, you must **requantize and produce new scale factors** in the correct layout — you cannot simply transpose the existing scale tensor.

**Misaligned allocation.** Scale factor pointers must be 16-byte aligned. Standard CUDA allocation (`cudaMalloc`) guarantees this, but custom allocators or sub-allocation from pools may not.

## Learn More

- **OCP Microscaling Spec:** [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — the standard that defines block scaling with E8M0
- **cuDNN Attention API:** [Fused Attention Operations](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html) — how cuDNN uses the 128×4 layout for SDPA
- **GTC 2025:** [cuDNN on Blackwell](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73071/) — architecture deep dive on MXFP8 and NVFP4 support
