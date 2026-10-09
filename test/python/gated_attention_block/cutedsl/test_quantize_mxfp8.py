# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block's MXFP8 quantize pass (``kernels/quantize_mxfp8.py``): e4m3 codes + F8_128x4 E8M0 scales.

Bit-exactness against ``test/python/sdpa/mxfp8_quant.py::quantize_to_mxfp8`` is the
contract, on BOTH the data and the uint8 scale-factor view: the production Rubin MXFP8
SDPA consumes exactly the oracle's layout and values, the adapter binds the SF blob by
storage order and checks only its byte count, so a wrong byte order or a rounding
corner is numerically wrong downstream and never an error.  Hence ``torch.equal``.

Three tiers:

* shape algebra + the two section-2.3 byte-address formulas pinned against the oracle's
  ``_swizzle_128x4`` / ``swizzle_sf_columnwise`` on random probes -- no GPU;
* an sm_107a trace-compile of both arms with an ``nvdisasm`` spill count -- any box whose
  cutlass-dsl knows ``sm_107a`` (>= 4.8.0.dev0) AND whose ``$CUDA_PATH/bin`` or ``$PATH``
  carries an nvdisasm that decodes it (an A100 box compiles Rubin cubins in a second).
  Anything less is a SKIP, never a failure: the verdict must not depend on which toolkit
  ``CUDA_PATH`` names;
* numerics on a cc 10.x device (``cvt.rp.satfinite.ue8m0x2.f32`` is sm_100+); the block
  targets Rubin, so these run on the SM107 dev node.

The two GEMM-canonical modes (``sf_layout="gemm"``: the padded F8_128x4 blob over ``[rows, K]`` the
block-scale projection GEMMs declare, batch folded into the rows; ``transposed=True``: the
columnwise arm's physically transposed ``[H*D, T]`` store) take the same three tiers against the
block reference's ``mx_quantize_rowwise_2d`` / ``mx_swizzle_sf_rowwise_padded`` (the artifact
builder's own oracle) cross-checked with ``quantize_to_mxfp8``'s ``_d`` / ``_s`` payloads -- bitwise,
the default arms' artifacts untouched.

The PACKED arm (``packed=True``: the block's THD pipeline; SF tiles per sequence in ``cu_seqlens``
order, the V planes adjacent, slack past the live total zero) is pinned the same way: per sequence
against ``quantize_to_mxfp8`` on that sequence's rows re-laid as the Rubin MXFP8 SDPA's THD row
reads them (``_packed_sf_reference``), BITWISE against the dense launch where the two layouts
coincide (``B = 1``: the payload and the Q/K slabs; the V slabs are the dense atoms in the
plane-adjacent order), as an exact PERMUTATION of the dense blobs on a uniform packing, with every
byte written under an E8M0-NaN (``0xFF``) poison (pad blocks and slack ``0x00``), on both length
forms (``[B]`` lengths, ``[B+1]`` prefix sums at a non-zero base), on ``B = 300`` five-token
sequences (the warp scan has no cap on ``B``), and the dual-axis packed pair bitwise the two
standalone packed launches.  The dense artifacts' SASS is unchanged by the arm (the appended
parameters are trace-time constants there).
"""

import glob
import os
import shutil
import struct
import subprocess
import sys
import textwrap
import zlib

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # test/python: the sdpa oracle package

from sdpa.mxfp8_quant import (  # noqa: E402
    _swizzle_128x4,
    e8m0_ceil,
    quantize_blocks,
    quantize_to_mxfp8,
    swizzle_sf_columnwise,
    swizzle_sf_rowwise,
)

from cudnn.frost.tile_dsl.pointwise import E8M0_RCP_E4M3_MAX_BITS  # noqa: E402
from cudnn.gated_attention_block import api as block_api  # noqa: E402
from cudnn.gated_attention_block.api import GatedAttentionBlockGeometry, _sf_slot_bytes  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes, sf_padded_dims  # noqa: E402
from cudnn.gated_attention_block.kernels.quantize_mxfp8 import (  # noqa: E402
    AXIS_COL,
    AXIS_ROW,
    COMPILE_OPTIONS,
    SF_LAYOUT_GEMM,
    SF_LAYOUT_SDPA,
    SF_LAYOUTS,
    QuantizeMxfp8Recipe,
    canonical_atoms_per_band,
    compile_quantize_mxfp8,
    compiled_cache,
    lanes_per_row,
    moved_bytes,
    n_sf_tiles,
    n_sf_tiles_packed_cap,
    run_quantize_mxfp8,
    sf_byte_canonical,
    sf_byte_columnwise,
    sf_byte_columnwise_packed,
    sf_byte_rowwise,
    sf_byte_rowwise_packed,
    sf_bytes,
    sf_bytes_packed,
    sf_tile_bytes,
    validate_mode,
    validate_packed_mode,
    validate_shape,
)

sys.path.insert(0, _HERE)  # the block reference next to this file
from gated_block_reference import (  # noqa: E402
    RefGeometry,
    make_inputs,
    mx_quantize_rowwise_2d,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    quantize_block_inputs_mxfp8,
)

D = 256
N_QKVG = 17408  # the 397B slab width: Q [0, 4*256), K [1024, 1536), V [1536, 2048) at h_q=4 / h_kv=2
ROLES = {"q": (AXIS_ROW, 4, 0), "k": (AXIS_ROW, 2, 1024), "v": (AXIS_COL, 2, 1536)}  # role -> (axis, heads, slab column offset)


def _nvdisasm_candidates():
    """Executables to TRY, most likely to decode sm_107a first; the probe verifies each and skips if none does.

    A public CUDA 13.x ``nvdisasm`` reports ``Cannot decode architecture 'SM107a'``, so the toolkit
    ``$CUDA_PATH`` names is tried first and the one on ``$PATH`` second -- both are hints, not authorities."""
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm107a_known_to_the_dsl() -> bool:
    """cutlass-dsl < 4.8.0.dev0 has no ``sm_107a`` (``Arch.from_string`` KeyError); FROST's floor is 4.7.0."""
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_mx_cvt = pytest.mark.skipif(
    _cc() is None or _cc() < (10, 0), reason=f"cvt.rp.satfinite.ue8m0x2.f32 needs sm_100+ (the block targets Rubin); found {_cc()}"
)


# ---------------------------------------------------------------------------
# Oracle plumbing
# ---------------------------------------------------------------------------


def _oracle(src_thd: torch.Tensor, b: int, s: int, h: int):
    """``quantize_to_mxfp8`` on the block's ``[T, H, D]`` buffer, permuted to BHSD.

    Returns ``(row_data [T,H,D] e4m3, row_sf flat uint8, col_data [T,H,D] e4m3, col_sf flat uint8)``."""
    bhsd = src_thd.reshape(b, s, h, D).permute(0, 2, 1, 3)
    row_d, _, row_sf, col_d, _, col_sf = quantize_to_mxfp8(bhsd, b, h, s, D, with_ref=False)
    to_thd = lambda x: x.permute(0, 2, 1, 3).reshape(b * s, h, D)  # noqa: E731
    return to_thd(row_d), row_sf.reshape(-1), to_thd(col_d), col_sf.reshape(-1)


def _launch(axis: str, src: torch.Tensor, *, batch: int, seq_len: int, h: int):
    """Compile (cached), sentinel-fill dst (0xFF = e4m3 NaN) and sf (0xFF = E8M0 NaN), run, sync."""
    r = compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=D, axis=axis)
    dst = torch.empty(batch * seq_len, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    dst.view(torch.uint8).fill_(0xFF)
    sf = torch.full((sf_bytes(batch, h, seq_len, D),), 0xFF, dtype=torch.uint8, device="cuda")
    run_quantize_mxfp8(r, src, dst, sf, batch=batch, seq_len=seq_len, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst, sf


def _check_against_oracle(axis: str, src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, *, batch: int, seq_len: int, h: int) -> None:
    row_d, row_sf, col_d, col_sf = _oracle(src, batch, seq_len, h)
    ref_d, ref_sf = (row_d, row_sf) if axis == AXIS_ROW else (col_d, col_sf)
    got_d = dst.view(torch.uint8)
    assert ref_sf.numel() == sf.numel(), (ref_sf.numel(), sf.numel())
    n_bad_d = int((got_d != ref_d.view(torch.uint8)).sum().item())
    n_bad_sf = int((sf != ref_sf).sum().item())
    assert n_bad_d == 0, f"{axis}: {n_bad_d} / {got_d.numel()} e4m3 codes differ from the oracle"
    assert n_bad_sf == 0, f"{axis}: {n_bad_sf} / {sf.numel()} SF bytes differ from the oracle (byte ORDER or rounding)"
    # 100 % of both outputs written: a 0xFF sentinel is a NaN in either format and cannot come from finite data.
    assert int((got_d == 0xFF).sum().item()) == 0, "e4m3 sentinel survived: some data cells were never written"
    assert int((sf == 0xFF).sum().item()) == 0, "SF sentinel survived: some scale bytes were never written (E8M0 NaN under the SDPA's whole-tile TMA)"


def _launch_canonical(axis: str, src: torch.Tensor, *, batch: int, seq_len: int, h: int, transposed: bool):
    """The GEMM-canonical modes: compile (cached), sentinel-fill dst (0xFF) and the padded blob (0xFF), run, sync.

    ``src`` ``[T, H, D]``; rowwise the blob is ``sf_blob_bytes(T, H*D)`` and ``dst`` ``[T, H, D]``; transposed the blob is
    ``sf_blob_bytes(H*D, T)`` and ``dst`` the contiguous e4m3 ``[H*D, T]`` matrix."""
    t = batch * seq_len
    r = compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=D, axis=axis, sf_layout=SF_LAYOUT_GEMM, transposed=transposed)
    assert r.sf_layout == SF_LAYOUT_GEMM and r.transposed is transposed
    dst = torch.empty((h * D, t) if transposed else (t, h, D), dtype=torch.float8_e4m3fn, device="cuda")
    dst.view(torch.uint8).fill_(0xFF)
    need = sf_blob_bytes(h * D, t) if transposed else sf_blob_bytes(t, h * D)
    sf = torch.full((need,), 0xFF, dtype=torch.uint8, device="cuda")
    run_quantize_mxfp8(r, src, dst, sf, batch=batch, seq_len=seq_len, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst, sf


def _canonical_oracle(src_thd: torch.Tensor, *, transposed: bool):
    """The block reference's quantization of the ``[T, H*D]`` matrix (rowwise) or of its transpose ``[H*D, T]`` (the
    transposed arm) -- e4m3 codes + the padded canonical blob -- cross-checked against ``quantize_to_mxfp8``'s ``_d`` / ``_s``
    payload of the same matrix (b = h = 1, s = T, d = H*D): the two oracles must agree before either judges the kernel."""
    t = int(src_thd.shape[0])
    x2d = src_thd.reshape(t, -1)
    k = int(x2d.shape[1])
    codes, e = mx_quantize_rowwise_2d(x2d.t().contiguous() if transposed else x2d)
    blob = mx_swizzle_sf_rowwise_padded(e)
    row_d, _, _, col_d, _, _ = quantize_to_mxfp8(x2d.reshape(1, 1, t, k), 1, 1, t, k, with_ref=False)
    twin = col_d.reshape(t, k).t().contiguous() if transposed else row_d.reshape(t, k)
    assert torch.equal(codes.view(torch.uint8), twin.view(torch.uint8)), "the two oracles disagree on the payload"
    return codes, e, blob


def _check_canonical(src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, *, transposed: bool) -> None:
    codes, e, blob = _canonical_oracle(src, transposed=transposed)
    got = dst.view(torch.uint8).reshape(codes.shape)  # rowwise: the [T, H, D] destination IS the [T, H*D] matrix; transposed: [H*D, T] as is
    assert got.numel() == codes.numel() and blob.numel() == sf.numel(), (got.shape, codes.shape, blob.numel(), sf.numel())
    n_bad_d = int((got != codes.view(torch.uint8)).sum().item())
    n_bad_sf = int((sf != blob).sum().item())
    assert n_bad_d == 0, f"transposed={transposed}: {n_bad_d} / {got.numel()} e4m3 codes differ from the oracle"
    assert n_bad_sf == 0, f"transposed={transposed}: {n_bad_sf} / {sf.numel()} SF bytes differ from the padded canonical blob (byte ORDER or rounding)"
    assert int((got == 0xFF).sum().item()) == 0, "e4m3 sentinel survived: some data cells were never written"
    assert int((sf == 0xFF).sum().item()) == 0, "SF sentinel survived: some scale bytes (pad bytes included) were never written"
    rows, k = codes.shape
    assert torch.equal(mx_unswizzle_sf_rowwise(sf, rows, k), e), "the blob read back through the reference's inverse is not the logical scale matrix"


def _tail_sf_bytes(axis: str, *, batch: int, h: int, seq_len: int):
    """Absolute SF byte offsets that belong to pad rows (``s >= S`` in the last tile), per section 2.3."""
    tiles = n_sf_tiles(seq_len)
    offs = set()
    for b in range(batch):
        for hh in range(h):
            if axis == AXIS_ROW:  # every pad ROW owns D/32 scales
                for s in range(seq_len, tiles * 128):
                    for d_idx in range(0, D, 32):
                        offs.add(sf_byte_rowwise(b, hh, s, d_idx, n_heads=h, n_tiles=tiles, d=D))
            else:  # only a 32-token block that is ENTIRELY padding is 0x00 (a straddling block keeps its live rows' amax)
                for s in range(-(-seq_len // 32) * 32, tiles * 128, 32):
                    for dd in range(D):
                        offs.add(sf_byte_columnwise(b, hh, s, dd, n_heads=h, n_tiles=tiles, batch=batch))
    return sorted(offs)


def _lens_tensor(lens, *, cu: bool = False, base: int = 0) -> torch.Tensor:
    """The packed lengths as the kernel takes them: ``[B]`` int32 lengths, or ``[B+1]`` int32 prefix sums at ``base``."""
    if not cu:
        return torch.tensor([int(n) for n in lens], dtype=torch.int32, device="cuda")
    cu_vals, acc = [int(base)], int(base)
    for n in lens:
        acc += int(n)
        cu_vals.append(acc)
    return torch.tensor(cu_vals, dtype=torch.int32, device="cuda")


def _launch_packed(axis: str, src: torch.Tensor, lens, *, h: int, cu: bool = False, cu_base: int = 0, poison: int = 0xFF):
    """Compile (cached) the PACKED arm, poison dst (0xFF = e4m3 NaN) and the whole SF slot incl. its slack (0xFF = E8M0 NaN), run, sync.

    ``src`` is the packed ``[T, H, D]`` (``T = sum(lens)``); the SF slot is ``sf_bytes_packed(B, H, T, D)`` (the capacity bound the
    block carves).  Returns ``(dst, sf, seq_lens)``."""
    b, t = len(lens), int(sum(lens))
    r = compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=D, axis=axis, packed=True)
    assert r.packed is True and r.sf_layout == SF_LAYOUT_SDPA
    dst = torch.full((t, h, D), poison, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    sf = torch.full((sf_bytes_packed(b, h, t, D),), poison, dtype=torch.uint8, device="cuda")
    seq_lens = _lens_tensor(lens, cu=cu, base=cu_base)
    run_quantize_mxfp8(r, src, dst, sf, stream=torch.cuda.current_stream().cuda_stream, seq_lens=seq_lens, num_sequences=b, cu_seqlens=cu)
    torch.cuda.synchronize()
    return dst, sf, seq_lens


def _packed_sf_reference(axis: str, src_thd: torch.Tensor, lens, *, h: int):
    """The PACKED layout's oracle, per sequence: ``quantize_to_mxfp8`` on the sequence's own rows (``b = 1``), its SF tiles re-laid as
    the Rubin MXFP8 SDPA's THD row reads them -- rowwise ``[h, n_tiles, 1024]`` slabs as they come, columnwise the plane-major
    ``[D/128, h, n_tiles, 512]`` atoms transposed to ``[h, n_tiles, D/128, 512]`` (the SDPA THD suite's per-sequence recipe) -- then the
    sequences' tiles concatenated per head in ``cu_seqlens`` order and the slot's slack tiles zero.  An empty sequence contributes no
    rows and no tiles.  Returns ``(data [T, H, D] e4m3, sf flat uint8 of sf_bytes_packed)``."""
    b, t = len(lens), int(sum(lens))
    n_cap = n_sf_tiles_packed_cap(t, b)
    datas, sfs, lo = [], [], 0
    for n in lens:
        n = int(n)
        if n == 0:
            continue
        bhsd = src_thd[lo : lo + n].reshape(1, n, h, D).permute(0, 2, 1, 3)
        row_d, _, row_sf, col_d, _, col_sf = quantize_to_mxfp8(bhsd, 1, h, n, D, with_ref=False)
        n_tiles = n_sf_tiles(n)
        if axis == AXIS_ROW:
            data, sf = row_d, row_sf.view(torch.uint8).reshape(h, n_tiles, sf_tile_bytes(D))
        else:
            planes = D // 128
            data = col_d
            sf = col_sf.view(torch.uint8).reshape(planes, h, n_tiles, 512).permute(1, 2, 0, 3).reshape(h, n_tiles, sf_tile_bytes(D))
        datas.append(data.permute(0, 2, 1, 3).reshape(n, h, D))
        sfs.append(sf)
        lo += n
    data = torch.cat(datas, dim=0)
    sf = torch.cat(sfs, dim=1)
    assert sf.shape[1] <= n_cap, (sf.shape, n_cap)
    slack = torch.zeros((h, n_cap - sf.shape[1], sf_tile_bytes(D)), dtype=torch.uint8, device=sf.device)
    return data, torch.cat([sf, slack], dim=1).reshape(-1)


def _check_packed_against_oracle(axis: str, src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, lens, *, h: int) -> None:
    ref_d, ref_sf = _packed_sf_reference(axis, src, lens, h=h)
    got_d = dst.view(torch.uint8)
    assert ref_sf.numel() == sf.numel(), (ref_sf.numel(), sf.numel())
    n_bad_d = int((got_d != ref_d.view(torch.uint8)).sum().item())
    n_bad_sf = int((sf != ref_sf).sum().item())
    assert n_bad_d == 0, f"{axis} packed {list(lens)}: {n_bad_d} / {got_d.numel()} e4m3 codes differ from the per-sequence oracle"
    assert n_bad_sf == 0, f"{axis} packed {list(lens)}: {n_bad_sf} / {sf.numel()} SF bytes differ from the per-sequence oracle (byte ORDER or rounding)"
    # every byte written: the 0xFF poison is a NaN in both formats and cannot come from finite data -- the slack tiles included
    assert int((got_d == 0xFF).sum().item()) == 0, "e4m3 sentinel survived: some packed data rows were never written"
    assert int((sf == 0xFF).sum().item()) == 0, "SF sentinel survived: a pad block or a slack tile was never written (E8M0 NaN under the SDPA's whole-tile TMA)"


# ---------------------------------------------------------------------------
# Tier 1: shape algebra + layout contracts -- no GPU
# ---------------------------------------------------------------------------


def test_packed_contracts_are_typed():
    """The packed arm's algebra and its typed contract, no device: the capacity bound dominates the live total for any lengths
    (and IS the dense count at ``B = 1``), the byte twins place a slab at ``(h*n_cap + tile) * 1024`` with the V planes adjacent
    (at ``B = 1`` the rowwise twin is the dense one; the columnwise twin differs from the dense D-plane-major one as soon as
    ``H * n_tiles > 1``), and ``packed`` refuses the canonical / transposed modes before any tracing."""
    import random

    rng = random.Random(7)
    for _ in range(200):
        b = rng.randint(1, 40)
        lens = [rng.randint(0, 700) for _ in range(b)]
        t = sum(lens)
        live = sum(n_sf_tiles(n) for n in lens)
        cap = n_sf_tiles_packed_cap(t, b)
        assert live <= cap < live + b, (lens, live, cap)
    for t in (1, 127, 128, 129, 300, 1000):
        assert n_sf_tiles_packed_cap(t, 1) == n_sf_tiles(t) and sf_bytes_packed(1, 3, t, D) == sf_bytes(1, 3, t, D)
    h, n_cap = 3, 5
    for hh in range(h):
        for tile in range(n_cap):
            for r in (0, 31, 32, 127):
                for d_idx in (0, 32, 128, 255):
                    want = (hh * n_cap + tile) * sf_tile_bytes(D) + ((d_idx // 32) // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + (d_idx // 32) % 4
                    assert sf_byte_rowwise_packed(hh, tile, r, d_idx, n_cap=n_cap, d=D) == want
                    assert sf_byte_rowwise_packed(hh, tile, r, d_idx, n_cap=n_cap, d=D) == sf_byte_rowwise(
                        0, hh, tile * 128 + r, d_idx, n_heads=h, n_tiles=n_cap, d=D
                    )
                    plane, dm = d_idx // 128, d_idx % 128
                    want_c = (hh * n_cap + tile) * sf_tile_bytes(D) + plane * 512 + (dm % 32) * 16 + (dm // 32) * 4 + r // 32
                    assert sf_byte_columnwise_packed(hh, tile, r, d_idx, n_cap=n_cap, d=D) == want_c
    # the whole packed V slot is a permutation of the dense B=1 one (every byte lands exactly once); at (h, n) = (1, 1) they coincide
    dense = {sf_byte_columnwise(0, hh, s_, dd, n_heads=h, n_tiles=n_cap, batch=1) for hh in range(h) for s_ in range(0, n_cap * 128, 32) for dd in range(D)}
    packed = {
        sf_byte_columnwise_packed(hh, s_ // 128, s_ % 128, dd, n_cap=n_cap, d=D) for hh in range(h) for s_ in range(0, n_cap * 128, 32) for dd in range(D)
    }
    assert dense == packed == set(range(sf_bytes_packed(1, h, n_cap * 128, D)))
    assert sf_byte_columnwise_packed(0, 0, 40, 130, n_cap=1, d=D) == sf_byte_columnwise(0, 0, 40, 130, n_heads=1, n_tiles=1, batch=1)
    assert sf_byte_columnwise_packed(0, 1, 40, 2, n_cap=2, d=D) != sf_byte_columnwise(
        0, 0, 128 + 40, 2, n_heads=1, n_tiles=2, batch=1
    )  # plane 0 of tile 1: 1024 vs 512
    validate_packed_mode(SF_LAYOUT_SDPA, False, False)
    validate_packed_mode(SF_LAYOUT_SDPA, False, True)
    validate_packed_mode(SF_LAYOUT_GEMM, True, False)
    with pytest.raises(ValueError, match="sf_layout='sdpa'"):
        validate_packed_mode(SF_LAYOUT_GEMM, False, True)
    with pytest.raises(ValueError, match="transposed=True"):
        validate_packed_mode(SF_LAYOUT_SDPA, True, True)
    with pytest.raises(ValueError, match="bool"):
        validate_packed_mode(SF_LAYOUT_SDPA, False, 1)
    with pytest.raises(ValueError, match="sf_layout='sdpa'"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, sf_layout=SF_LAYOUT_GEMM, packed=True)
    with pytest.raises(ValueError, match="sf_layout='sdpa'"):  # the layout check comes first; the transposed one is pinned on validate_packed_mode above
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM, transposed=True, packed=True)
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8_dual

    with pytest.raises(ValueError, match="sf_layout='sdpa'"):
        compile_quantize_mxfp8_dual(dtype_in=torch.bfloat16, h=2, d=D, sf_layout=SF_LAYOUT_GEMM, transposed_second=True, packed=True)


def test_rcp_448_is_the_exact_fp32_rounding():
    """The scale is ONE fp32 multiply by fp32(1/448); the oracle does the same (``torch.tensor(1/448, float32)``)."""
    bits = struct.unpack("<I", struct.pack("<f", 1.0 / 448.0))[0]
    assert bits == 0x3B124925 == E8M0_RCP_E4M3_MAX_BITS
    assert torch.tensor(1.0 / 448.0, dtype=torch.float32).view(torch.int32).item() == 0x3B124925


@pytest.mark.parametrize(
    "amax, byte, rcp",
    [
        (0.0, 0x00, 2.0**127),
        (448.0, 0x7F, 1.0),
        (449.0, 0x80, 0.5),
        (896.0, 0x80, 0.5),
        (224.0, 0x7E, 2.0),
        (2.0, 0x78, 128.0),
        (1.0, 0x77, 256.0),
        (0.5, 0x76, 512.0),
    ],
)
def test_e8m0_oracle_corners(amax, byte, rcp):
    """The corners the kernel must reproduce: 448 * fp32(1/448) == 1.0 exactly (no round-up), 449 rounds UP, powers of two stay."""
    x = torch.zeros(1, 32)
    x[0, 0] = amax
    _, e = quantize_blocks(x, torch.float8_e4m3fn)
    assert int(e.item()) == byte
    assert e8m0_ceil(torch.tensor([amax * (1.0 / 448.0)], dtype=torch.float32)).item() == byte
    assert float(2.0 ** (127 - byte)) == rcp


def test_sf_algebra_at_d256():
    assert sf_tile_bytes(256) == 1024 and sf_tile_bytes(128) == 512 and sf_tile_bytes(512) == 2048
    assert n_sf_tiles(128) == 1 and n_sf_tiles(129) == 2 and n_sf_tiles(1000) == 8
    assert sf_bytes(2, 4, 1000, 256) == 2 * 4 * 8 * 1024  # == the adapter's b*h*n_tiles*SF_SMEM_SIZE
    assert lanes_per_row(256) == 16 and lanes_per_row(512) == 32
    assert moved_bytes(1000, 32, 256) == 1000 * 32 * 256 * 3 + 1000 * 32 * 256 // 32
    assert moved_bytes(7, 2, 256, 2) == 7 * 2 * 256 * 3 + 7 * 2 * 256 // 32


def test_validate_shape_declines_what_the_lanes_cannot_cover():
    for axis in (AXIS_ROW, AXIS_COL):
        validate_shape(256, 256, axis)
        validate_shape(128, 256, axis)
        validate_shape(512, 256, axis)
        with pytest.raises(ValueError, match="multiple of 128"):
            validate_shape(200, 256, axis)
        with pytest.raises(ValueError, match="multiple of 32"):
            validate_shape(256, 40, axis)
        with pytest.raises(ValueError, match="burst"):
            validate_shape(2048, 256, axis)  # 8 KiB SF tile needs 512 burst lanes
    with pytest.raises(ValueError, match="axis"):
        validate_shape(256, 256, "rows")
    for bad_d in (0, -128):  # 0 used to reach a ZeroDivisionError, -128 used to PASS (every modulus held) and reach cute.compile
        with pytest.raises(ValueError, match="positive multiple of 128"):
            validate_shape(bad_d, 256, AXIS_ROW)
        with pytest.raises(ValueError, match="positive multiple of 128"):
            validate_shape(bad_d, 256, AXIS_COL)
    with pytest.raises(ValueError, match="up to 1024"):
        validate_shape(256, 2048, AXIS_ROW)  # past the CTA cap: validation passed, the launch failed untyped
    with pytest.raises(ValueError, match="up to 1024"):
        validate_shape(256, 2048, AXIS_COL)
    with pytest.raises(ValueError, match="divide a warp"):
        validate_shape(384, 256, AXIS_ROW)  # 24 lanes/row
    with pytest.raises(ValueError, match="rows per pass"):
        validate_shape(256, 96, AXIS_ROW)  # 6 rows/pass does not divide 128
    with pytest.raises(ValueError, match="units per tile"):
        validate_shape(128, 512, AXIS_COL)  # 8 units over 16 warps


def test_compile_declines_bad_inputs_before_tracing():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_quantize_mxfp8(dtype_in=torch.float32, h=2, d=256, axis=AXIS_ROW)
    with pytest.raises(ValueError, match="axis"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=256, axis="column")
    with pytest.raises(ValueError, match="positive"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=0, d=256, axis=AXIS_ROW)
    with pytest.raises(ValueError, match="not built"):
        run_quantize_mxfp8(QuantizeMxfp8Recipe(torch.bfloat16, 2, 256, AXIS_ROW), None, None, None, batch=1, seq_len=1, stream=0)


def test_f8_128x4_atom_rule_is_the_quoted_one():
    """``_swizzle_128x4``: atoms (128 rows x 4 cols) row-major; inside, (r, c) at ``(r%32)*16 + (r//32)*4 + c``."""
    g = torch.Generator().manual_seed(0)
    for R, C in ((128, 8), (384, 8), (256, 32), (128, 4)):
        m = torch.randint(0, 256, (R, C), dtype=torch.uint8, generator=g)
        sw = _swizzle_128x4(m).reshape(-1)
        for _ in range(500):
            r, c = int(torch.randint(0, R, (1,), generator=g)), int(torch.randint(0, C, (1,), generator=g))
            atom = (r // 128) * (C // 4) + c // 4
            assert int(sw[atom * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4]) == int(m[r, c])


@pytest.mark.parametrize("b, h, s", [(1, 1, 128), (2, 3, 384), (1, 2, 1000), (2, 2, 256)])
def test_rowwise_sf_byte_formula_matches_the_oracle_swizzle(b, h, s):
    """Section 2.3 Q/K row: tile (b, h, s_tile) = 1024 contiguous bytes; inside ``(c//4)*512 + (s%32)*16 + ((s%128)//32)*4 + c%4``."""
    g = torch.Generator().manual_seed(1)
    tiles = n_sf_tiles(s)
    s_pad = tiles * 128
    row_e = torch.randint(0, 256, (b * h * s_pad, D // 32), dtype=torch.uint8, generator=g)  # the oracle's logical [l*s_padded, D/32]
    sw = swizzle_sf_rowwise(row_e).reshape(-1)
    assert sw.numel() == sf_bytes(b, h, s, D)
    for _ in range(2000):
        bb, hh = int(torch.randint(0, b, (1,), generator=g)), int(torch.randint(0, h, (1,), generator=g))
        ss, dd = int(torch.randint(0, s_pad, (1,), generator=g)), int(torch.randint(0, D, (1,), generator=g))
        off = sf_byte_rowwise(bb, hh, ss, dd, n_heads=h, n_tiles=tiles, d=D)
        assert int(sw[off]) == int(row_e[(bb * h + hh) * s_pad + ss, dd // 32]), (bb, hh, ss, dd)
    # the tile is contiguous and lands where the SDPA's _build_sf_desc expects it
    assert sf_byte_rowwise(b - 1, h - 1, s_pad - 1, D - 1, n_heads=h, n_tiles=tiles, d=D) == sw.numel() - 1


@pytest.mark.parametrize("b, h, s", [(1, 1, 128), (2, 3, 384), (1, 2, 1000), (2, 2, 256)])
def test_columnwise_sf_byte_formula_matches_the_oracle_swizzle(b, h, s):
    """Section 2.3 V row: D-plane-major, plane stride ``B*KH*n_tiles*512`` -- NOT the Q/K per-tile order."""
    g = torch.Generator().manual_seed(2)
    tiles = n_sf_tiles(s)
    s_pad = tiles * 128
    col_e = torch.randint(0, 256, (b * h * s_pad // 32, D), dtype=torch.uint8, generator=g)  # the oracle's TE storage [l*s_padded/32, D]
    sw = swizzle_sf_columnwise(col_e).reshape(-1)
    assert sw.numel() == sf_bytes(b, h, s, D)
    for _ in range(2000):
        bb, hh = int(torch.randint(0, b, (1,), generator=g)), int(torch.randint(0, h, (1,), generator=g))
        ss, dd = int(torch.randint(0, s_pad, (1,), generator=g)), int(torch.randint(0, D, (1,), generator=g))
        off = sf_byte_columnwise(bb, hh, ss, dd, n_heads=h, n_tiles=tiles, batch=b)
        assert int(sw[off]) == int(col_e[(bb * h + hh) * (s_pad // 32) + ss // 32, dd]), (bb, hh, ss, dd)
    # the two D-planes are a whole plane apart, and the second plane begins exactly where the first ends
    assert sf_byte_columnwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, batch=b) == b * h * tiles * 512
    if b * h * tiles > 1:
        assert sf_byte_columnwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, batch=b) != sf_byte_rowwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, d=D)


@pytest.mark.parametrize("rows, k", [(256, 5120), (992, 5120), (1000, 17408), (2016, 5120), (5120, 256), (5120, 992), (17408, 2016)])
def test_canonical_sf_byte_formula_matches_the_oracle_swizzle(rows, k):
    """``sf_byte_canonical(row, k_block, k=K)`` addresses the PADDED canonical blob exactly as ``mx_swizzle_sf_rowwise_padded`` lays it
    out, for the rowwise ``(T, N)`` and the transposed ``(N, T)`` orientations (T in 256 / 992 / 1000 / 2016), pad rows / blocks
    ``0x00``, ``canonical_atoms_per_band`` == ``sf_padded_dims``'s atom count; the byte count is SYMMETRIC in (rows, K)."""
    g = torch.Generator().manual_seed(4)
    e = torch.randint(1, 256, (rows, k // 32), dtype=torch.uint8, generator=g)
    blob = mx_swizzle_sf_rowwise_padded(e)
    assert blob.numel() == sf_blob_bytes(rows, k) == sf_blob_bytes(k, rows) if k % 128 == 0 and rows % 32 == 0 else blob.numel() == sf_blob_bytes(rows, k)
    assert canonical_atoms_per_band(k) == sf_padded_dims(rows, k)[1] // 4 == -(-(k // 32) // 4)
    live = set()
    for _ in range(3000):
        r, c = int(torch.randint(0, rows, (1,), generator=g)), int(torch.randint(0, k // 32, (1,), generator=g))
        off = sf_byte_canonical(r, c, k=k)
        assert int(blob[off]) == int(e[r, c]), (r, c, off)
    for r in range(rows):  # every live byte, by the formula (dense enumeration), then the rest of the blob is pad = 0x00
        for c in range(k // 32):
            live.add(sf_byte_canonical(r, c, k=k))
    assert len(live) == rows * (k // 32)
    mask = torch.ones(blob.numel(), dtype=torch.bool)
    mask[torch.tensor(sorted(live))] = False
    assert int(blob[mask].max().item()) == 0 if mask.any() else True
    assert mask.sum().item() == blob.numel() - rows * (k // 32)


def test_mode_contracts_are_typed():
    """``validate_mode`` / ``compile_quantize_mxfp8`` type the (axis, sf_layout, transposed) triple both ways, and
    ``run_quantize_mxfp8`` refuses the wrong blob count / destination for each mode BEFORE any launch -- exercised on a
    hand-built recipe (``compiled`` a sentinel: every reject fires ahead of the call), so it runs on any CUDA box."""
    validate_mode(AXIS_ROW, SF_LAYOUT_SDPA, False)
    validate_mode(AXIS_COL, SF_LAYOUT_GEMM, True)
    validate_mode(AXIS_ROW, SF_LAYOUT_GEMM, False)
    with pytest.raises(ValueError, match="needs axis='col'"):
        validate_mode(AXIS_ROW, SF_LAYOUT_GEMM, True)
    with pytest.raises(ValueError, match="needs sf_layout='gemm'"):
        validate_mode(AXIS_COL, SF_LAYOUT_SDPA, True)
    with pytest.raises(ValueError, match="needs transposed=True"):
        validate_mode(AXIS_COL, SF_LAYOUT_GEMM, False)  # the columnwise GEMM blob exists only as the [H*D, T] store (review of #1429)
    with pytest.raises(ValueError, match="sf_layout must be one of"):
        validate_mode(AXIS_ROW, "canonical", False)
    with pytest.raises(ValueError, match="transposed must be a bool"):
        validate_mode(AXIS_COL, SF_LAYOUT_GEMM, 1)
    with pytest.raises(ValueError, match="needs axis='col'"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, sf_layout=SF_LAYOUT_GEMM, transposed=True)
    with pytest.raises(ValueError, match="needs sf_layout='gemm'"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_COL, transposed=True)
    with pytest.raises(ValueError, match="needs transposed=True"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM)  # transposed omitted
    with pytest.raises(ValueError, match="needs transposed=True"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM, transposed=False)
    assert SF_LAYOUTS == (SF_LAYOUT_SDPA, SF_LAYOUT_GEMM) == ("sdpa", "gemm")
    if not torch.cuda.is_available():
        pytest.skip("the run_ contract needs CUDA tensors")
    h, b, s = 2, 2, 128
    t = b * s
    x = torch.randn(t, h, D, device="cuda").to(torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream
    dst3 = torch.empty(t, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    dst_t = torch.empty(h * D, t, dtype=torch.float8_e4m3fn, device="cuda")
    sf_sdpa = torch.zeros(sf_bytes(b, h, s, D), dtype=torch.uint8, device="cuda")
    sf_row = torch.zeros(sf_blob_bytes(t, h * D), dtype=torch.uint8, device="cuda")
    sf_t = torch.zeros(sf_blob_bytes(h * D, t), dtype=torch.uint8, device="cuda")
    assert (
        sf_sdpa.numel() == sf_row.numel() == sf_t.numel()
    ), "at batch-folded (T, H*D) the three counts coincide: the MESSAGES, not the counts, tell the modes apart"
    sentinel = object()
    gemm_row = QuantizeMxfp8Recipe(torch.bfloat16, h, D, AXIS_ROW, compiled=sentinel, sf_layout=SF_LAYOUT_GEMM)
    gemm_t = QuantizeMxfp8Recipe(torch.bfloat16, h, D, AXIS_COL, compiled=sentinel, sf_layout=SF_LAYOUT_GEMM, transposed=True)
    sdpa_col = QuantizeMxfp8Recipe(torch.bfloat16, h, D, AXIS_COL, compiled=sentinel)
    with pytest.raises(ValueError, match=r"sf_blob_bytes\(rows=256, K=512\)"):
        run_quantize_mxfp8(gemm_row, x, dst3, sf_row[:-1], batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match=r"sf_blob_bytes\(rows=512, K=256\)"):
        run_quantize_mxfp8(gemm_t, x, dst_t, sf_t[:-512], batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match=r"ceil\(S/128\)"):
        run_quantize_mxfp8(sdpa_col, x, dst3, sf_sdpa[:-1], batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match=r"transposed=True: dst must be the contiguous \[H\*D=512, T=256\]"):
        run_quantize_mxfp8(gemm_t, x, dst3, sf_t, batch=b, seq_len=s, stream=st)  # a [T, H, D] dst to the transposed artifact
    with pytest.raises(ValueError, match=r"transposed=True: dst must be the contiguous"):
        run_quantize_mxfp8(gemm_t, x, dst_t.t().contiguous().t(), sf_t, batch=b, seq_len=s, stream=st)  # a [N, T] VIEW with strides (1, N)
    with pytest.raises(ValueError, match=r"dst must be \[T, H=2, D=256\] \(this artifact is transposed=False\)"):
        run_quantize_mxfp8(gemm_row, x, dst_t, sf_row, batch=b, seq_len=s, stream=st)  # an [N, T] dst to a rowwise artifact
    with pytest.raises(ValueError, match=r"dst must be \[T, H=2, D=256\]"):
        run_quantize_mxfp8(sdpa_col, x, dst_t, sf_sdpa, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="multiple of 32"):
        xr = torch.randn(1000, h, D, device="cuda").to(torch.bfloat16)
        run_quantize_mxfp8(gemm_t, xr, torch.empty(h * D, 1000, dtype=torch.float8_e4m3fn, device="cuda"), sf_t, batch=1, seq_len=1000, stream=st)


@pytest.mark.parametrize("t", [256, 992, 2016])
@pytest.mark.parametrize("n", [5120, 17408])
def test_quantize_mxfp8_stage_sf_bytes_follows_the_layout(t, n):
    """``api._QuantizeMxfp8(...).sf_bytes()``: ``_sf_slot_bytes`` at the defaults (the SDPA adapter's count), ``sf_blob_bytes(T, N)``
    under ``sf_layout="gemm"`` rowwise, ``sf_blob_bytes(N, T)`` transposed; ``check_support`` types ``transposed`` without ``col`` /
    ``gemm``; the stage's ``moved_bytes`` is the same count in every mode."""
    h = n // D
    geom = GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=64)
    mk = lambda **kw: block_api._QuantizeMxfp8(geom, batch=1, seq_len=t, dtype_in=torch.bfloat16, heads=h, name="q", **kw)  # noqa: E731
    base = mk(axis=AXIS_ROW)
    base.check_support()
    assert (base.sf_layout, base.transposed) == (SF_LAYOUT_SDPA, False) and base.sf_bytes() == _sf_slot_bytes(1, h, t, D)
    rowwise = mk(axis=AXIS_ROW, sf_layout=SF_LAYOUT_GEMM)
    rowwise.check_support()
    assert rowwise.sf_bytes() == sf_blob_bytes(t, n)
    transposed = mk(axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM, transposed=True)
    transposed.check_support()
    assert transposed.sf_bytes() == sf_blob_bytes(n, t)
    assert base.moved_bytes() == rowwise.moved_bytes() == transposed.moved_bytes() == moved_bytes(t, h, D)
    for kw in (
        dict(axis=AXIS_ROW, sf_layout=SF_LAYOUT_GEMM, transposed=True),
        dict(axis=AXIS_COL, transposed=True),
        dict(axis=AXIS_COL, sf_layout="gem"),
        dict(axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM),  # the columnwise GEMM blob without the transposed store (review of #1429)
    ):
        with pytest.raises(ValueError, match="transposed=True|sf_layout must be one of|needs transposed=True"):
            mk(**kw).check_support()


@pytest.mark.parametrize("b, s", [(1, 256), (2, 496), (2, 1008), (1, 1000)])
def test_artifact_builder_backward_arm_is_the_transposed_requantization(b, s):
    """``quantize_block_inputs_mxfp8(inp, backward=True)`` appends the caller's four artifacts and nothing else changes:
    ``h_t`` == ``mx_quantize_rowwise_2d(h^T)`` (contiguous e4m3 ``[d_model, T]``, strides ``(T, 1)``) with
    ``h_t_sf.numel() == sf_blob_bytes(d_model, T)``; ``w_qkvg_t`` == ``mx_quantize_rowwise_2d(W^T)`` ``[d_model, N]`` with
    ``sf_blob_bytes(d_model, N)`` bytes; the forward's codes / blobs are byte-identical to the ``backward=False`` call; the
    forward's ``h_sf`` has the SAME byte count as ``h_t_sf`` (the symmetric count: a wrong-orientation blob is a numerics
    matter, never a host reject) and differs from it byte for byte.  At T = 1000 (not a multiple of 32: the token axis
    cannot be block-quantized, and the block declines a weight gradient at such a T) the arm builds ``w_qkvg_t`` /
    ``w_qkvg_t_sf`` only -- no ``h_t`` keys, no error (the data gradient's artifact is served at any T)."""
    geom = RefGeometry(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=64)
    inp = make_inputs(geom, batch=b, seq_len=s, device="cuda" if torch.cuda.is_available() else "cpu")
    fwd, spec_fwd = quantize_block_inputs_mxfp8(inp)
    bwd, spec_bwd = quantize_block_inputs_mxfp8(inp, backward=True)
    t, dm, n = b * s, geom.d_model, int(inp["w_qkvg"].shape[0])
    whole_blocks = t % 32 == 0
    assert spec_fwd == spec_bwd and set(bwd) - set(fwd) == ({"h_t", "h_t_sf", "w_qkvg_t", "w_qkvg_t_sf"} if whole_blocks else {"w_qkvg_t", "w_qkvg_t_sf"})
    for key in fwd:
        assert torch.equal(fwd[key].view(torch.uint8), bwd[key].view(torch.uint8)) if isinstance(fwd[key], torch.Tensor) else fwd[key] == bwd[key], key
    w_t_codes, w_t_e = mx_quantize_rowwise_2d(inp["w_qkvg"].t().contiguous())
    assert torch.equal(bwd["w_qkvg_t"].view(torch.uint8), w_t_codes.view(torch.uint8)) and bwd["w_qkvg_t"].stride() == (n, 1)
    assert bwd["w_qkvg_t_sf"].numel() == sf_blob_bytes(dm, n) and torch.equal(bwd["w_qkvg_t_sf"], mx_swizzle_sf_rowwise_padded(w_t_e))
    assert bwd["w_qkvg_sf"].numel() == bwd["w_qkvg_t_sf"].numel() and not torch.equal(bwd["w_qkvg_sf"], bwd["w_qkvg_t_sf"])
    if not whole_blocks:
        return
    h_t_codes, h_t_e = mx_quantize_rowwise_2d(inp["h"].reshape(t, dm).t().contiguous())
    assert torch.equal(bwd["h_t"].view(torch.uint8), h_t_codes.view(torch.uint8)) and bwd["h_t"].stride() == (t, 1) and bwd["h_t"].dtype == torch.float8_e4m3fn
    assert bwd["h_t_sf"].numel() == sf_blob_bytes(dm, t) and torch.equal(bwd["h_t_sf"], mx_swizzle_sf_rowwise_padded(h_t_e))
    assert bwd["h_sf"].numel() == bwd["h_t_sf"].numel() and not torch.equal(bwd["h_sf"], bwd["h_t_sf"])
    # the transposed codes are NOT the forward codes transposed: the 32-blocks run along the other axis
    assert not torch.equal(bwd["h_t"].view(torch.uint8), bwd["h"].reshape(t, dm).t().contiguous().view(torch.uint8))


def test_oracle_pads_s_to_128_with_zero_scales():
    """The pad rows the kernel must write as 0x00 are 0x00 in the oracle too (both arms), located by the section-2.3 formulas."""
    torch.manual_seed(3)
    b, h, s = 2, 2, 900  # rows 900..1023 pad; columnwise the 32-token blocks at 928 / 960 / 992 are entirely padding
    x = torch.randn(b * s, h, D).to(torch.bfloat16)
    _, row_sf, _, col_sf = _oracle(x, b, s, h)
    for axis, sf in ((AXIS_ROW, row_sf), (AXIS_COL, col_sf)):
        offs = _tail_sf_bytes(axis, batch=b, h=h, seq_len=s)
        assert offs, axis
        assert int(sf[offs].max().item()) == 0, axis
        assert int(sf.ne(0).sum().item()) > 0.9 * (sf.numel() - len(offs)), axis  # everything else is a live scale


# ---------------------------------------------------------------------------
# Tier 2: sm_107a trace-compile + spill count -- any box with the internal toolkit
# ---------------------------------------------------------------------------

_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    axis, sf_layout, transposed, packed, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1", sys.argv[5], sys.argv[6:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    import torch
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import COMPILE_OPTIONS, compile_quantize_mxfp8
    # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which ICEs on sm_107a.
    compile_quantize_mxfp8(
        dtype_in=torch.bfloat16, h=4, d=256, axis=axis, compile_options=COMPILE_OPTIONS + " --gpu-arch sm_107a --keep-cubin",
        sf_layout=sf_layout, transposed=transposed, packed=packed,
    )
    cubins = glob.glob(os.path.join(dump, "*.sm_107a.cubin"))
    if not cubins:
        print("FAIL no .sm_107a.cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:  # vanished / not executable / wedged: the next candidate
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes sm_107a")
        sys.exit(0)
    print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("E8M0CVT", sum(1 for ln in sass if "F2FP" in ln and ".E8." in ln and ".RP" in ln))
    print("E4M3CVT", sum(1 for ln in sass if "F2FP" in ln and "E4M3" in ln))
    print("FMNMX3", sum(1 for ln in sass if "FMNMX3" in ln))
    print("STG128", sum(1 for ln in sass if "STG.E.128" in ln))
    print("STG16", sum(1 for ln in sass if "STG.E.U16" in ln or "STG.E.16" in ln))
    print("LINES", len(sass))
    """)


@pytest.mark.parametrize(
    "axis, sf_layout, transposed, packed, e8m0_cvts",
    [
        (AXIS_ROW, SF_LAYOUT_SDPA, False, False, 8),
        (AXIS_COL, SF_LAYOUT_SDPA, False, False, 2),
        (AXIS_ROW, SF_LAYOUT_GEMM, False, False, 8),
        (AXIS_COL, SF_LAYOUT_GEMM, True, False, 2),
        (AXIS_ROW, SF_LAYOUT_SDPA, False, True, 8),
        (AXIS_COL, SF_LAYOUT_SDPA, False, True, 2),
    ],
    ids=["row", "col", "canonical", "transposed", "packed-row", "packed-col"],
)
def test_sm107_trace_compile_has_no_spills(axis, sf_layout, transposed, packed, e8m0_cvts, tmp_path):
    """Compile for Rubin here (no device match needed), decode with an nvdisasm that knows sm_107a: STL/LDL must be 0 -- for
    the two SDPA-layout arms, the two GEMM-canonical ones and the two PACKED ones (the warp scan adds shuffles, no local memory).

    Also pins that the scale is produced by the hardware ``cvt.rp...ue8m0x2`` (at least one per row pass rowwise = 8
    at 256 threads / D=256; one ``e8m0_pair`` per warp unit columnwise = 2 -- a LOWER bound, ptxas may duplicate), that
    the abs-max tree fuses (FMNMX3 > 0), and the STORE FORM of the columnwise arms: the SDPA arm's 2-byte stores
    (``STG.E.U16`` > 0) against the transposed arm's 16-byte ones (NO ``STG.E.U16``; its data leaves as ``STG.E.128``).
    ``SPILL == 0`` is the load-bearing assertion.

    SKIPS (never fails) when the DSL predates ``sm_107a`` or no candidate nvdisasm decodes it -- an L0 verdict
    must not turn on which toolkit ``CUDA_PATH`` happens to name."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"quantize_mxfp8_{axis}_{sf_layout}_{int(transposed)}_{int(packed)}"
    dump.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _SASS_PROBE, axis, sf_layout, "1" if transposed else "0", "1" if packed else "0", str(dump), *cands],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    if any(ln.startswith("SKIP") for ln in proc.stdout.splitlines()):
        pytest.skip(f"{[ln for ln in proc.stdout.splitlines() if ln.startswith(('SKIP', 'REJECT'))]}")
    keys = ("SPILL", "E8M0CVT", "E4M3CVT", "FMNMX3", "STG128", "STG16", "LINES")
    stats = {k: int(v) for k, v in (ln.split() for ln in proc.stdout.splitlines() if ln.split() and ln.split()[0] in keys)}
    print(
        f"\n[{axis} {sf_layout} transposed={transposed} packed={packed}] sm_107a SASS: {stats} via {[ln for ln in proc.stdout.splitlines() if ln.startswith('NVDISASM')]}"
    )
    assert stats["SPILL"] == 0, f"{axis} {sf_layout} transposed={transposed} packed={packed}: {stats['SPILL']} STL/LDL in the sm_107a cubin"
    assert stats["E8M0CVT"] >= e8m0_cvts, f"{axis}: fewer ue8m0x2 cvts than the source issues -- {stats}"
    assert stats["E4M3CVT"] > 0 and stats["FMNMX3"] > 0, stats
    if axis == AXIS_COL:
        if transposed:
            assert stats["STG16"] == 0 and stats["STG128"] > 1, f"the transposed arm must store its codes as 16-byte vectors, not 2-byte words: {stats}"
        else:
            assert stats["STG16"] > 0, f"the SDPA columnwise arm's 2-byte stores are gone -- the module docstring's traffic account is stale: {stats}"
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"


def test_default_artifacts_are_byte_identical():
    """The appended knobs do not fork the default artifact: the implicit and the explicit ``(sf_layout="sdpa",
    transposed=False, packed=False)`` builds are ONE cache entry, keyed by today's 7-tuple as the PREFIX plus the three appended
    fields, and each new mode is its own entry with the same prefix.  Traced for sm_107a in-process (no device match needed; SKIPS
    where the DSL predates it) -- the SASS of the default arms is unchanged by construction: every new statement sits under a
    ``const_expr`` on the knobs, and the packed arm's appended kernel arguments are trace-time constants (``None``) on the default
    arms, so their kernel parameter lists are unchanged too."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    opts = COMPILE_OPTIONS + " --gpu-arch sm_107a"
    before = len(compiled_cache)
    r1 = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, compile_options=opts)
    r2 = compile_quantize_mxfp8(
        dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, compile_options=opts, sf_layout=SF_LAYOUT_SDPA, transposed=False, packed=False
    )
    assert r1.compiled is r2.compiled and (r1.sf_layout, r1.transposed, r1.packed) == (SF_LAYOUT_SDPA, False, False) == (r2.sf_layout, r2.transposed, r2.packed)
    keys = [k for k, v in compiled_cache.items() if v is r1.compiled]
    assert len(keys) == 1 and keys[0][:6] == ("torch.bfloat16", 2, D, AXIS_ROW, 256, opts) and keys[0][7:] == (SF_LAYOUT_SDPA, False, False), keys
    r3 = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, compile_options=opts, sf_layout=SF_LAYOUT_GEMM)
    assert r3.compiled is not r1.compiled
    k3 = [k for k, v in compiled_cache.items() if v is r3.compiled]
    assert len(k3) == 1 and k3[0][:7] == keys[0][:7] and k3[0][7:] == (SF_LAYOUT_GEMM, False, False)
    r4 = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=D, axis=AXIS_ROW, compile_options=opts, packed=True)
    assert r4.compiled is not r1.compiled and r4.packed is True
    k4 = [k for k, v in compiled_cache.items() if v is r4.compiled]
    assert len(k4) == 1 and k4[0][:7] == keys[0][:7] and k4[0][7:] == (SF_LAYOUT_SDPA, False, True)
    assert len(compiled_cache) >= before + 3 and len({k[7:] for k in (keys[0], k3[0], k4[0])}) == 3


# ---------------------------------------------------------------------------
# Tier 3: numerics on cc 10.x
# ---------------------------------------------------------------------------


@requires_mx_cvt
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("s", [128, 256, 384, 512, 1000, 900])
@pytest.mark.parametrize("role", ["q", "k", "v"])
def test_compact_source_is_bit_exact_vs_oracle(role, s, b, packed):
    """Compact ``[T, H, D]`` bf16 in; e4m3 codes AND the F8_128x4 SF blob equal the oracle byte for byte.

    H_q=4 (q) vs H_kv=2 (k, v) so the head stride of the SF tile index is exercised; S=1000 is the KV tail
    (104 live rows in the last tile; columnwise the 992..1023 block STRADDLES S), S=900 adds three 32-token
    blocks that are entirely padding (columnwise SF must be 0x00 there).  ``packed``: the same ``(b, s)`` as the
    uniform packing ``(s,) * b`` through the PACKED arm against the per-sequence oracle (its slack tiles ``0x00``);
    the e4m3 payload is the dense launch's byte for byte."""
    axis, h, _ = ROLES[role]
    torch.manual_seed(zlib.crc32(f"{role}-{s}-{b}".encode()) & 0xFFFF)  # str hash is per-process randomized; crc32 replays
    x = (torch.randn(b * s, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst, sf = _launch(axis, x, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, x, dst, sf, batch=b, seq_len=s, h=h)
    offs = _tail_sf_bytes(axis, batch=b, h=h, seq_len=s)
    if offs:
        assert int(sf[offs].max().item()) == 0, "pad-row SF bytes must be 0x00"
    if packed:
        dst_p, sf_p, _ = _launch_packed(axis, x, (s,) * b, h=h)
        _check_packed_against_oracle(axis, x, dst_p, sf_p, (s,) * b, h=h)
        assert torch.equal(dst_p.view(torch.uint8), dst.view(torch.uint8)), "the packed payload differs from the dense launch's"


@requires_mx_cvt
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_f16_and_bf16_sources(axis, dtype):
    torch.manual_seed(7)
    b, s, h = 2, 384, 2
    x = (torch.randn(b * s, h, D, device="cuda") * 40.0).to(dtype)
    dst, sf = _launch(axis, x, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, x, dst, sf, batch=b, seq_len=s, h=h)


@requires_mx_cvt
@pytest.mark.parametrize("role", ["q", "k", "v"])
def test_strided_slab_slice_source(role):
    """The source is a column slice of the fused ``[T, 17408]`` projection slab (token stride 17408), as the block hands it."""
    axis, h, col_off = ROLES[role]
    torch.manual_seed(11)
    b, s = 2, 1000
    slab = (torch.randn(b * s, N_QKVG, device="cuda") * 2.0).to(torch.bfloat16)
    src = torch.as_strided(slab, (b * s, h, D), (N_QKVG, D, 1), storage_offset=col_off)
    assert src.stride(0) == N_QKVG and not src.is_contiguous()
    dst, sf = _launch(axis, src, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, src.contiguous(), dst, sf, batch=b, seq_len=s, h=h)
    assert dst.is_contiguous()


@requires_mx_cvt
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_e8m0_corners_on_device(axis):
    """Hand-built blocks: all-zero (SF 0x00, code 0), amax exactly 448 (0x7F: 448 * fp32(1/448) == 1.0, no round-up),
    amax 450 (rounds UP to 0x80), exact powers of two 2.0 / 1.0 / 0.5 (0x78 / 0x77 / 0x76), 446 (0x7F from below),
    a negative-dominated block, and -448 (the sign must not touch the amax: 0x7F, code 0xFE).  Bit-exact vs the
    oracle AND vs the hand-written SF bytes; the e4m3 codes vs torch's saturating cast of ``value * 2^(127 - e)``."""
    b, s, h = 1, 384, 1  # 9 cases: rowwise rows 0..8, columnwise token blocks 0..8 (tokens 0..287 -- needs S > 256)
    cases = [
        (0, 0.0, 0x00),
        (1, 448.0, 0x7F),
        (2, 450.0, 0x80),
        (3, 2.0, 0x78),
        (4, 1.0, 0x77),
        (5, 0.5, 0x76),
        (6, 446.0, 0x7F),
        (7, -3.0, 0x78),
        (8, -448.0, 0x7F),
    ]
    x = torch.zeros(s, h, D, device="cuda", dtype=torch.float32)
    # rowwise: row r, block 0 (d 0..31); columnwise: d-column 0, token block r (s = 32r .. 32r+31).
    for r, amax, _ in cases:
        first, second = (r, r) if axis == AXIS_ROW else (32 * r, 32 * r + 1)
        if axis == AXIS_ROW:
            x[first, 0, 0] = amax
            x[first, 0, 1] = -amax / 4
        else:
            x[first, 0, 0] = amax
            x[second, 0, 0] = -amax / 4
    xb = x.to(torch.bfloat16)
    row_of = (lambda r: r) if axis == AXIS_ROW else (lambda r: 32 * r)
    assert float(xb[row_of(1), 0, 0]) == 448.0 and float(xb[row_of(2), 0, 0]) == 450.0 and float(xb[row_of(8), 0, 0]) == -448.0  # bf16-exact
    dst, sf = _launch(axis, xb, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, xb, dst, sf, batch=b, seq_len=s, h=h)
    tiles = n_sf_tiles(s)
    codes = dst.view(torch.uint8)
    for r, amax, e in cases:
        exp_code = int(torch.tensor([abs(amax) * 2.0 ** (127 - e)]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).item()) | (
            0x80 if amax < 0 else 0
        )
        if axis == AXIS_ROW:
            off = sf_byte_rowwise(0, 0, r, 0, n_heads=h, n_tiles=tiles, d=D)
            code = int(codes[r, 0, 0].item())
        else:
            off = sf_byte_columnwise(0, 0, 32 * r, 0, n_heads=h, n_tiles=tiles, batch=b)
            code = int(codes[32 * r, 0, 0].item())
        assert int(sf[off].item()) == e, f"{axis} amax={amax}: SF 0x{int(sf[off].item()):02X} != 0x{e:02X}"
        assert code == exp_code, f"{axis} amax={amax}: code 0x{code:02X} != 0x{exp_code:02X}"


# ---------------------------------------------------------------------------
# The PACKED (THD) arm: per-sequence-tile-padded SF tiles in cu_seqlens order
# ---------------------------------------------------------------------------

_PACKINGS = [
    (300, 128, 200),  # the block's THD suite packing: a tail tile on every sequence, none a 128-multiple
    (5, 130, 1000),  # a 5-token sequence (one tile), a 2-tile one, a 1000-token one (the KV tail of the dense suite)
    (126, 0, 60),  # an EMPTY sequence in the middle: zero tiles, zero rows
    (300, 200, 0),  # a TRAILING empty sequence: cu_sf[B] == cu_sf[B-1], the slack starts right after the last live tile
    (512,),  # B = 1: the packed layout is the dense one
]


@requires_mx_cvt
@pytest.mark.parametrize("cu", [False, True], ids=["lengths", "prefix"])
@pytest.mark.parametrize("lens", _PACKINGS, ids=lambda l: "x".join(str(n) for n in l))
@pytest.mark.parametrize("role", ["q", "k", "v"])
def test_packed_matches_the_per_sequence_oracle(role, lens, cu):
    """The PACKED arm against ``quantize_to_mxfp8`` run on each sequence's own rows and re-laid as the SDPA's THD row reads the
    tiles (``_packed_sf_reference``): e4m3 codes and SF bytes equal byte for byte, every byte of the slot written (the pad
    blocks of a tail tile and the slack tiles ``0x00``, never the ``0xFF`` poison), empty sequences owning no tiles, on both
    length forms (the ``[B+1]`` prefix sums at base 0 here; a non-zero base in its own cell)."""
    axis, h, _ = ROLES[role]
    torch.manual_seed(zlib.crc32(f"packed-{role}-{lens}-{cu}".encode()) & 0xFFFF)
    t = sum(lens)
    x = (torch.randn(t, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst, sf, _ = _launch_packed(axis, x, lens, h=h, cu=cu)
    _check_packed_against_oracle(axis, x, dst, sf, lens, h=h)
    # the slack tiles (past the live total, up to the capacity) are 0x00: the capacity-slack class stays inert for the SDPA
    live = sum(n_sf_tiles(n) for n in lens)
    n_cap = n_sf_tiles_packed_cap(t, len(lens))
    slack = sf.view(h, n_cap, sf_tile_bytes(D))[:, live:]
    assert slack.numel() == h * (n_cap - live) * sf_tile_bytes(D) and (slack.numel() == 0 or int(slack.max().item()) == 0)


@requires_mx_cvt
@pytest.mark.parametrize("role", ["q", "k", "v"])
@pytest.mark.parametrize("t", [128, 300, 1000])
def test_packed_b1_is_the_dense_layout(role, t):
    """``B = 1``: the packed launch over ``(t,)`` against the dense ``batch=1, seq_len=t`` launch over the SAME bytes -- the e4m3
    payload and the Q / K slabs BITWISE (``cu_sf[0] = 0``, ``n_cap = ceil(t/128)``: the two layouts coincide); the V slot holds the
    dense atoms in the plane-ADJACENT order (``packed.view(h, n, 2, 512).permute(2, 0, 1, 3) == dense.view(2, h, n, 512)``), which
    is the dense slot itself only at ``h * n == 1``.  A difference is a finding, never a tolerance."""
    axis, h, _ = ROLES[role]
    torch.manual_seed(zlib.crc32(f"b1-{role}-{t}".encode()) & 0xFFFF)
    x = (torch.randn(t, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst_d, sf_d = _launch(axis, x, batch=1, seq_len=t, h=h)
    dst_p, sf_p, _ = _launch_packed(axis, x, (t,), h=h)
    assert sf_p.numel() == sf_d.numel() == sf_bytes(1, h, t, D)
    assert torch.equal(dst_p.view(torch.uint8), dst_d.view(torch.uint8)), "B=1 packed payload differs from the dense launch"
    n = n_sf_tiles(t)
    if axis == AXIS_ROW:
        assert torch.equal(sf_p, sf_d), "B=1 packed Q/K scale slabs differ from the dense launch"
    else:
        perm = sf_p.view(h, n, D // 128, 512).permute(2, 0, 1, 3).reshape(-1)
        assert torch.equal(perm, sf_d), "B=1 packed V scale atoms are not the dense atoms in the plane-adjacent order"
        assert (h * n == 1) == torch.equal(sf_p, sf_d)


@requires_mx_cvt
@pytest.mark.parametrize("role", ["q", "k", "v"])
@pytest.mark.parametrize("s, b", [(256, 4), (200, 4), (128, 2)])
def test_packed_uniform_is_a_permutation_of_the_dense_blobs(role, s, b):
    """A UNIFORM packing ``(s,) * b`` against the dense ``(b, s)`` launch over the same bytes: the payload bitwise, the SF slabs an
    exact PERMUTATION -- Q/K ``packed.view(H, B, n, 1024).permute(1, 0, 2, 3) == dense.view(B, H, n, 1024)``, V
    ``packed.view(H, B, n, 2, 512).permute(3, 1, 0, 2, 4) == dense.view(2, B, H, n, 512)`` (the oracle-free pin of the burst
    addressing); at ``s = 200`` every sequence has a tail tile and the slot carries slack tiles past ``B * n``."""
    axis, h, _ = ROLES[role]
    torch.manual_seed(zlib.crc32(f"perm-{role}-{s}-{b}".encode()) & 0xFFFF)
    x = (torch.randn(b * s, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst_d, sf_d = _launch(axis, x, batch=b, seq_len=s, h=h)
    dst_p, sf_p, _ = _launch_packed(axis, x, (s,) * b, h=h)
    assert torch.equal(dst_p.view(torch.uint8), dst_d.view(torch.uint8))
    n = n_sf_tiles(s)
    n_cap = n_sf_tiles_packed_cap(b * s, b)
    live = sf_p.view(h, n_cap, sf_tile_bytes(D))[:, : b * n]  # the slack tiles (zero) are not part of the permutation
    if axis == AXIS_ROW:
        assert torch.equal(live.reshape(h, b, n, sf_tile_bytes(D)).permute(1, 0, 2, 3).reshape(-1), sf_d)
    else:
        assert torch.equal(live.reshape(h, b, n, D // 128, 512).permute(3, 1, 0, 2, 4).reshape(-1), sf_d)
    if n_cap > b * n:
        assert int(sf_p.view(h, n_cap, sf_tile_bytes(D))[:, b * n :].max().item()) == 0


@requires_mx_cvt
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_packed_prefix_sums_at_a_nonzero_base_are_the_lengths_form_bitwise(axis):
    """``cu_seqlens=True`` with a prefix tensor sliced from a larger one (base 100): the kernel reads adjacent DIFFERENCES, so the
    base never matters -- bitwise the ``[B]`` lengths launch and the base-0 prefix launch."""
    h, lens = 2, (300, 128, 200)
    torch.manual_seed(11)
    x = torch.randn(sum(lens), h, D, device="cuda").to(torch.bfloat16)
    d0, s0, _ = _launch_packed(axis, x, lens, h=h)
    d1, s1, _ = _launch_packed(axis, x, lens, h=h, cu=True)
    d2, s2, l2 = _launch_packed(axis, x, lens, h=h, cu=True, cu_base=100)
    assert l2.tolist() == [100, 400, 528, 728]
    assert torch.equal(d0.view(torch.uint8), d1.view(torch.uint8)) and torch.equal(s0, s1)
    assert torch.equal(d0.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s0, s2)


@requires_mx_cvt
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_packed_many_short_sequences_have_no_cap(axis):
    """``B = 300`` sequences of 5..128 tokens (the block's shortest legal sequence is 5): the per-unit warp scan walks ten
    32-sequence chunks with a carried total -- no SMEM array of ``B`` entries, no cap on ``B`` -- and every sequence lands on its own
    tiles (the per-sequence oracle names a wrong one).  Run on both length forms."""
    import random

    rng = random.Random(300)
    lens = [rng.randint(5, 128) for _ in range(300)]
    h = 2
    torch.manual_seed(300)
    x = torch.randn(sum(lens), h, D, device="cuda").to(torch.bfloat16)
    for cu in (False, True):
        dst, sf, _ = _launch_packed(axis, x, lens, h=h, cu=cu)
        _check_packed_against_oracle(axis, x, dst, sf, lens, h=h)


@requires_mx_cvt
def test_packed_two_launches_are_bitwise_identical():
    torch.manual_seed(13)
    h, lens = 2, (300, 128, 200)
    x = torch.randn(sum(lens), h, D, device="cuda").to(torch.bfloat16)
    for axis in (AXIS_ROW, AXIS_COL):
        d1, s1, _ = _launch_packed(axis, x, lens, h=h)
        d2, s2, _ = _launch_packed(axis, x, lens, h=h, poison=0x00)
        assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s1, s2), axis


@requires_mx_cvt
def test_dual_axis_packed_sdpa_pair_is_bitwise_the_two_standalone_packed_launches():
    """The dual-axis arm under ``packed=True`` (the SDPA pair): ``dst`` / ``sf`` equal the standalone packed rowwise launch and
    ``dst_T`` / ``sf_T`` the standalone packed columnwise one, byte for byte -- the same per-sequence decode, the same bursts."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8_dual, run_quantize_mxfp8_dual

    torch.manual_seed(17)
    h, lens = 2, (300, 128, 200)
    b, t = len(lens), sum(lens)
    x = torch.randn(t, h, D, device="cuda").to(torch.bfloat16)
    r = compile_quantize_mxfp8_dual(dtype_in=x.dtype, h=h, d=D, packed=True)
    assert r.packed is True
    dst = torch.full((t, h, D), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    dst_t = torch.full((t, h, D), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    sf = torch.full((sf_bytes_packed(b, h, t, D),), 0xFF, dtype=torch.uint8, device="cuda")
    sf_t = torch.full_like(sf, 0xFF)
    seq_lens = _lens_tensor(lens)
    run_quantize_mxfp8_dual(r, x, dst, sf, dst_t, sf_t, stream=torch.cuda.current_stream().cuda_stream, seq_lens=seq_lens, num_sequences=b)
    torch.cuda.synchronize()
    row_d, row_sf, _ = _launch_packed(AXIS_ROW, x, lens, h=h)
    col_d, col_sf, _ = _launch_packed(AXIS_COL, x, lens, h=h)
    assert torch.equal(dst.view(torch.uint8), row_d.view(torch.uint8)) and torch.equal(sf, row_sf)
    assert torch.equal(dst_t.view(torch.uint8), col_d.view(torch.uint8)) and torch.equal(sf_t, col_sf)
    with pytest.raises(ValueError, match="batch / seq_len are refused"):
        run_quantize_mxfp8_dual(r, x, dst, sf, dst_t, sf_t, batch=1, seq_len=t, stream=torch.cuda.current_stream().cuda_stream)


@requires_mx_cvt
def test_packed_run_declines_a_mismatched_binding():
    """The packed arm's host contract, typed both ways and before any launch: a packed recipe refuses ``batch`` / ``seq_len`` and
    needs ``seq_lens`` (int32, 1-D, contiguous, CUDA, ``B`` or ``B+1`` entries) + ``num_sequences >= 1`` and the capacity byte count;
    a dense recipe refuses the packed trio.  A refused call launches nothing (the sentinel-filled outputs are untouched)."""
    h, lens = 2, (300, 128, 200)
    b, t = len(lens), sum(lens)
    x = torch.randn(t, h, D, device="cuda").to(torch.bfloat16)
    r = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=D, axis=AXIS_ROW, packed=True)
    dst = torch.full((t, h, D), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    sf = torch.full((sf_bytes_packed(b, h, t, D),), 0xFF, dtype=torch.uint8, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    good = _lens_tensor(lens)
    with pytest.raises(ValueError, match="batch / seq_len are refused"):
        run_quantize_mxfp8(r, x, dst, sf, batch=1, seq_len=t, stream=st, seq_lens=good, num_sequences=b)
    with pytest.raises(ValueError, match="num_sequences >= 1"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=good, num_sequences=0)
    with pytest.raises(ValueError, match="int32"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=good.to(torch.int64), num_sequences=b)
    with pytest.raises(ValueError, match="of 3 elements"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=good[:2], num_sequences=b)
    with pytest.raises(ValueError, match="of 4 elements"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=good, num_sequences=b, cu_seqlens=True)
    with pytest.raises(ValueError, match="int32"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=None, num_sequences=b)
    with pytest.raises(ValueError, match="CUDA tensor"):
        run_quantize_mxfp8(r, x, dst, sf, stream=st, seq_lens=good.cpu(), num_sequences=b)
    with pytest.raises(ValueError, match="PACKED layout"):
        run_quantize_mxfp8(r, x, dst, sf[:-1], stream=st, seq_lens=good, num_sequences=b)
    with pytest.raises(ValueError, match="PACKED layout"):  # the DENSE B=1 byte count (10 tiles / head) is not the packed capacity at B = 4 (8 tiles / head)
        run_quantize_mxfp8(
            r, x, dst, torch.zeros(sf_bytes(1, h, t, D), dtype=torch.uint8, device="cuda"), stream=st, seq_lens=_lens_tensor(lens + (0,)), num_sequences=b + 1
        )
    torch.cuda.synchronize()
    assert int((dst.view(torch.uint8) != 0xFF).sum().item()) == 0 and int((sf != 0xFF).sum().item()) == 0, "a refused call launched"
    dense = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=D, axis=AXIS_ROW)
    sf_dense = torch.zeros(sf_bytes(1, h, t, D), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="PACKED arm"):
        run_quantize_mxfp8(dense, x, dst, sf_dense, batch=1, seq_len=t, stream=st, seq_lens=good, num_sequences=b)
    with pytest.raises(ValueError, match="PACKED arm"):
        run_quantize_mxfp8(dense, x, dst, sf_dense, batch=1, seq_len=t, stream=st, cu_seqlens=True)
    with pytest.raises(ValueError, match="required for a dense"):
        run_quantize_mxfp8(dense, x, dst, sf_dense, stream=st)


# ---------------------------------------------------------------------------
# The DUAL-AXIS arm: rowwise + columnwise from one read (the MXFP8 backward's launch fusion)
# ---------------------------------------------------------------------------


def _launch_dual(src: torch.Tensor, *, batch: int, seq_len: int, h: int, canonical: bool, poison: int = 0xFF):
    """Compile (cached) the dual arm, poison every destination byte with ``poison``, run, sync.  Returns ``(dst, sf, dst_T, sf_T)``."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import compile_quantize_mxfp8_dual, run_quantize_mxfp8_dual

    t = batch * seq_len
    r = compile_quantize_mxfp8_dual(dtype_in=src.dtype, h=h, d=D, sf_layout=SF_LAYOUT_GEMM if canonical else SF_LAYOUT_SDPA, transposed_second=canonical)
    dst = torch.full((t, h, D), poison, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    dst_t = torch.full((h * D, t) if canonical else (t, h, D), poison, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    if canonical:
        sf = torch.full((sf_blob_bytes(t, h * D),), poison, dtype=torch.uint8, device="cuda")
        sf_t = torch.full((sf_blob_bytes(h * D, t),), poison, dtype=torch.uint8, device="cuda")
    else:
        sf = torch.full((sf_bytes(batch, h, seq_len, D),), poison, dtype=torch.uint8, device="cuda")
        sf_t = torch.full_like(sf, poison)
    run_quantize_mxfp8_dual(r, src, dst, sf, dst_t, sf_t, batch=batch, seq_len=seq_len, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst, sf, dst_t, sf_t


def test_dual_axis_contracts_are_typed():
    """The dual arm's modes and shapes, typed before any device is touched: the SDPA pair refuses ``transposed_second``, the canonical
    pair REQUIRES it (the columnwise canonical blob has one form), the column pass needs one thread per ``d``, both SF tiles must
    burst in parallel, ``moved_bytes_dual`` is one read + two writes."""
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import moved_bytes_dual, validate_dual_mode, validate_dual_shape

    validate_dual_mode(SF_LAYOUT_SDPA, False)
    validate_dual_mode(SF_LAYOUT_GEMM, True)
    with pytest.raises(ValueError, match="needs sf_layout='gemm'"):
        validate_dual_mode(SF_LAYOUT_SDPA, True)
    with pytest.raises(ValueError, match="needs transposed=True"):
        validate_dual_mode(SF_LAYOUT_GEMM, False)
    validate_dual_shape(D, 256)
    validate_dual_shape(128, 128)
    validate_dual_shape(512, 512)
    with pytest.raises(ValueError, match="one thread per d"):
        validate_dual_shape(D, 128)
    with pytest.raises(ValueError, match="must divide the 32-token sub-tile"):
        validate_dual_shape(D, 1024)  # 1024 / 16 = 64 rows per pass > 32
    assert moved_bytes_dual(1000, 4, D) == 2 * moved_bytes(1000, 4, D) - 1000 * 4 * D * 2  # the second READ is what the fusion saves
    assert moved_bytes_dual(1000, 4, D, src_elem_bytes=2) == 1000 * 4 * D * 4 + 2 * (1000 * 4 * D // 32)
    # the recipe's host checks, on a recipe with compiled=None (no device needed beyond the tensors)
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import QuantizeMxfp8DualRecipe, check_dual_operands

    if not torch.cuda.is_available():
        return
    t, h = 64, 2
    src = torch.empty(t, h, D, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty(t, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(sf_bytes(1, h, t, D), dtype=torch.uint8, device="cuda")
    r_sdpa = QuantizeMxfp8DualRecipe(dtype_in=torch.bfloat16, h=h, d=D)
    assert check_dual_operands(r_sdpa, src, dst, sf, torch.empty_like(dst), torch.empty_like(sf), batch=1, seq_len=t) == (t, h, 0, 0)
    with pytest.raises(ValueError, match="dst_T must be the compact e4m3"):
        check_dual_operands(r_sdpa, src, dst, sf, torch.empty(h * D, t, dtype=torch.float8_e4m3fn, device="cuda"), torch.empty_like(sf), batch=1, seq_len=t)
    with pytest.raises(ValueError, match="sf_T must hold"):
        check_dual_operands(r_sdpa, src, dst, sf, torch.empty_like(dst), sf[:-16], batch=1, seq_len=t)
    r_can = QuantizeMxfp8DualRecipe(dtype_in=torch.bfloat16, h=h, d=D, sf_layout=SF_LAYOUT_GEMM, transposed_second=True)
    blob, blob_t = (
        torch.empty(sf_blob_bytes(t, h * D), dtype=torch.uint8, device="cuda"),
        torch.empty(sf_blob_bytes(h * D, t), dtype=torch.uint8, device="cuda"),
    )
    dst_t = torch.empty(h * D, t, dtype=torch.float8_e4m3fn, device="cuda")
    n_c = sf_padded_dims(t, h * D, 32)[1] // 4
    n_c_t = sf_padded_dims(h * D, t, 32)[1] // 4
    assert check_dual_operands(r_can, src, dst, blob, dst_t, blob_t, batch=1, seq_len=t) == (t, h, n_c, n_c_t)
    with pytest.raises(ValueError, match="multiple of 32"):
        check_dual_operands(
            r_can,
            src[:40],
            dst[:40],
            torch.empty(sf_blob_bytes(40, h * D), dtype=torch.uint8, device="cuda"),
            dst_t[:, :40].contiguous(),
            blob_t,
            batch=1,
            seq_len=40,
        )
    with pytest.raises(ValueError, match="transposed_second=True: dst_T must be the contiguous"):
        check_dual_operands(r_can, src, dst, blob, torch.empty_like(dst), blob_t, batch=1, seq_len=t)
    # the appended per-half flags (the fused epilogue's folded-out halves): a folded-out half passes None and is NOT checked (no
    # stand-in tensor), a tensor bound to it is refused, a traced half left unbound is refused, both halves folded out is refused; a
    # folded-out half's atom count comes back 0 (its blob is never sized -- the transposed one needs T % 32 == 0)
    assert check_dual_operands(r_can, src, dst, blob, None, None, batch=1, seq_len=t, want_col=False) == (t, h, n_c, 0)
    assert check_dual_operands(r_can, src, None, None, dst_t, blob_t, batch=1, seq_len=t, want_row=False) == (t, h, 0, n_c_t)
    assert check_dual_operands(r_sdpa, src, None, None, torch.empty_like(dst), torch.empty_like(sf), batch=1, seq_len=t, want_row=False) == (t, h, 0, 0)
    # a ROWWISE-only canonical launch over a ragged T (the block's dgrad-only cell at T = 1000, need_dw_qkvg=False) is served -- the
    # transposed blob, which the canonical builder refuses at T % 32 != 0, is never asked for; the same T with the transposed half traced is
    # the 32-token-block decline
    t_r = 1000
    src_r = torch.empty(t_r, h, D, dtype=torch.bfloat16, device="cuda")
    dst_r = torch.empty(t_r, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    blob_r = torch.empty(sf_blob_bytes(t_r, h * D), dtype=torch.uint8, device="cuda")
    n_c_r = sf_padded_dims(t_r, h * D, 32)[1] // 4
    assert check_dual_operands(r_can, src_r, dst_r, blob_r, None, None, batch=1, seq_len=t_r, want_col=False) == (t_r, h, n_c_r, 0)
    with pytest.raises(ValueError, match="multiple of 32"):
        check_dual_operands(r_can, src_r, dst_r, blob_r, torch.empty(h * D, t_r, dtype=torch.float8_e4m3fn, device="cuda"), blob_t, batch=1, seq_len=t_r)
    with pytest.raises(ValueError, match="dst_T is bound but its half is not traced"):
        check_dual_operands(r_can, src, dst, blob, dst_t, None, batch=1, seq_len=t, want_col=False)
    with pytest.raises(ValueError, match="sf must be bound: its half is traced"):
        check_dual_operands(r_can, src, dst, None, dst_t, blob_t, batch=1, seq_len=t)
    with pytest.raises(ValueError, match="at least one half"):
        check_dual_operands(r_can, src, None, None, None, None, batch=1, seq_len=t, want_row=False, want_col=False)


_DUAL_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    canonical, dump, cands = sys.argv[1] == "1", sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    import torch
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import COMPILE_OPTIONS, compile_quantize_mxfp8_dual
    compile_quantize_mxfp8_dual(
        dtype_in=torch.bfloat16, h=4, d=256, compile_options=COMPILE_OPTIONS + " --gpu-arch sm_107a --keep-cubin",
        sf_layout="gemm" if canonical else "sdpa", transposed_second=canonical,
    )
    cubins = glob.glob(os.path.join(dump, "*.sm_107a.cubin"))
    if not cubins:
        print("FAIL no .sm_107a.cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes sm_107a")
        sys.exit(0)
    print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("E8M0CVT", sum(1 for ln in sass if "F2FP" in ln and ".E8." in ln and ".RP" in ln))
    print("STG128", sum(1 for ln in sass if "STG.E.128" in ln))
    print("STG16", sum(1 for ln in sass if "STG.E.U16" in ln or "STG.E.16" in ln))
    print("LINES", len(sass))
    """)


@pytest.mark.parametrize("canonical", [False, True], ids=["sdpa-pair", "canonical-pair"])
def test_dual_axis_sm107_trace_compile_has_no_spills(canonical, tmp_path):
    """The dual arm compiled for Rubin here (no device match needed), decoded with an nvdisasm that knows sm_107a: STL/LDL must be 0
    in both modes, the column pass's ``cvt.rp...ue8m0x2`` present, and EVERY payload leaves as 16-byte vectors -- the SDPA pair's
    columnwise payload included (NO ``STG.E.U16``: the 2-byte-store arm of the standalone columnwise kernel is what the fusion
    removes from the chain).  SKIPS, never fails, where the DSL predates sm_107a or no nvdisasm decodes it."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"quantize_mxfp8_dual_{int(canonical)}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _DUAL_SASS_PROBE, "1" if canonical else "0", str(dump), *cands], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    if any(ln.startswith("SKIP") for ln in proc.stdout.splitlines()):
        pytest.skip(f"{[ln for ln in proc.stdout.splitlines() if ln.startswith(('SKIP', 'REJECT'))]}")
    keys = ("SPILL", "E8M0CVT", "STG128", "STG16", "LINES")
    stats = {k: int(v) for k, v in (ln.split() for ln in proc.stdout.splitlines() if ln.split() and ln.split()[0] in keys)}
    print(f"\n[dual canonical={canonical}] sm_107a SASS: {stats}")
    assert stats["SPILL"] == 0, f"dual canonical={canonical}: {stats['SPILL']} STL/LDL in the sm_107a cubin"
    assert stats["E8M0CVT"] >= 2 and stats["STG128"] > 1, stats
    assert stats["STG16"] == 0, f"the dual arm must store every payload as 16-byte vectors, never 2-byte words: {stats}"


@requires_mx_cvt
@pytest.mark.parametrize("b, s", [(1, 256), (2, 992), (1, 1000), (2, 1008), (1, 2016)])
@pytest.mark.parametrize("h", [2, 4])
def test_dual_axis_sdpa_pair_is_bitwise_the_two_standalone_launches(b, s, h):
    """``do8 + sf_do`` AND ``do_T8 + sf_do_T`` from ONE read == the standalone rowwise and columnwise SDPA-layout launches, byte for
    byte, every destination byte 0xFF-POISONED first: every SF byte of every ceil128(S) unit is written (a pad block ``0x00``: the
    e8m0 of a zeroed block -- 992 / 1000 / 1008 are the ragged tails, 1008 at B = 2 the one where an UNpredicated payload store would
    land in batch 1's first 16 tokens), and both halves equal the torch oracle through the standalone checks."""
    torch.manual_seed(1000 * b + s + h)
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    dst, sf, dst_t, sf_t = _launch_dual(x, batch=b, seq_len=s, h=h, canonical=False)
    row_d, row_sf = _launch(AXIS_ROW, x, batch=b, seq_len=s, h=h)
    col_d, col_sf = _launch(AXIS_COL, x, batch=b, seq_len=s, h=h)
    assert torch.equal(dst.view(torch.uint8), row_d.view(torch.uint8)), "the dual arm's rowwise payload differs from the standalone rowwise launch"
    assert torch.equal(sf, row_sf), "the dual arm's rowwise SF tiles differ from the standalone rowwise launch"
    assert torch.equal(dst_t.view(torch.uint8), col_d.view(torch.uint8)), "the dual arm's columnwise payload differs from the standalone columnwise launch"
    assert torch.equal(sf_t, col_sf), "the dual arm's columnwise SF atoms differ from the standalone columnwise launch"
    _check_against_oracle(AXIS_ROW, x, dst, sf, batch=b, seq_len=s, h=h)
    _check_against_oracle(AXIS_COL, x, dst_t, sf_t, batch=b, seq_len=s, h=h)
    tail = _tail_sf_bytes(AXIS_ROW, batch=b, h=h, seq_len=s)
    if tail:
        assert int(sf[tail].max().item()) == 0, "a pad row's rowwise SF byte is not 0x00"
    tail_c = _tail_sf_bytes(AXIS_COL, batch=b, h=h, seq_len=s)
    if tail_c:
        assert int(sf_t[tail_c].max().item()) == 0, "a pad block's columnwise SF byte is not 0x00"


@requires_mx_cvt
@pytest.mark.parametrize("t", [256, 992, 2016])
@pytest.mark.parametrize("n", [5120, 17408])
def test_dual_axis_canonical_pair_is_bitwise_the_two_standalone_launches(t, n):
    """``dqkvg8 + sf_dqkvg`` (rowwise canonical over ``(T, N)``) AND ``dqkvg_t8 + sf_dqkvg_t`` (the TRANSPOSED ``[N, T]`` store with its
    canonical blob over ``(N, T)``) from ONE read == the two standalone canonical launches, byte for byte, on 0xFF-poisoned
    destinations (the pad blocks of the ceil128 units ``0x00``, every blob byte written), and both equal the block reference's padded
    swizzle through the standalone canonical checks.  ``T % 32 == 0`` throughout (the transposed store's whole-block rule)."""
    h = n // D
    torch.manual_seed(t + n)
    x = torch.randn(t, h, D, device="cuda").to(torch.bfloat16)
    dst, sf, dst_t, sf_t = _launch_dual(x, batch=1, seq_len=t, h=h, canonical=True)
    row_d, row_sf = _launch_canonical(AXIS_ROW, x, batch=1, seq_len=t, h=h, transposed=False)
    col_d, col_sf = _launch_canonical(AXIS_COL, x, batch=1, seq_len=t, h=h, transposed=True)
    assert torch.equal(dst.view(torch.uint8), row_d.view(torch.uint8)) and torch.equal(
        sf, row_sf
    ), "the rowwise canonical half differs from the standalone launch"
    assert torch.equal(dst_t.view(torch.uint8), col_d.view(torch.uint8)) and torch.equal(
        sf_t, col_sf
    ), "the transposed canonical half differs from the standalone launch"
    _check_canonical(x, dst, sf, transposed=False)
    _check_canonical(x, dst_t, sf_t, transposed=True)


@requires_mx_cvt
def test_dual_axis_two_launches_are_bitwise_identical_and_match_f16():
    """Determinism (two launches, both pairs) and the f16 source (the ``f16x2_to_f32`` arm of the staging / row pass)."""
    torch.manual_seed(77)
    b, s, h = 2, 1000, 2
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    a = _launch_dual(x, batch=b, seq_len=s, h=h, canonical=False)
    c = _launch_dual(x, batch=b, seq_len=s, h=h, canonical=False)
    assert all(torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(a, c))
    xf = torch.randn(1 * 992, h, D, device="cuda").to(torch.float16)
    dst, sf, dst_t, sf_t = _launch_dual(xf, batch=1, seq_len=992, h=h, canonical=False)
    row_d, row_sf = _launch(AXIS_ROW, xf, batch=1, seq_len=992, h=h)
    col_d, col_sf = _launch(AXIS_COL, xf, batch=1, seq_len=992, h=h)
    assert torch.equal(dst.view(torch.uint8), row_d.view(torch.uint8)) and torch.equal(sf, row_sf)
    assert torch.equal(dst_t.view(torch.uint8), col_d.view(torch.uint8)) and torch.equal(sf_t, col_sf)


@requires_mx_cvt
def test_two_launches_are_bitwise_identical():
    torch.manual_seed(5)
    b, s, h = 2, 1000, 2
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    for axis in (AXIS_ROW, AXIS_COL):
        d1, s1 = _launch(axis, x, batch=b, seq_len=s, h=h)
        d2, s2 = _launch(axis, x, batch=b, seq_len=s, h=h)
        assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s1, s2), axis
    # the two canonical modes (T = 2000 serves the rowwise one; the transposed one needs T % 32 == 0: 2 x 992)
    d1, s1 = _launch_canonical(AXIS_ROW, x, batch=b, seq_len=s, h=h, transposed=False)
    d2, s2 = _launch_canonical(AXIS_ROW, x, batch=b, seq_len=s, h=h, transposed=False)
    assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s1, s2), "canonical rowwise"
    xt = x[: 2 * 992]
    d1, s1 = _launch_canonical(AXIS_COL, xt, batch=2, seq_len=992, h=h, transposed=True)
    d2, s2 = _launch_canonical(AXIS_COL, xt, batch=2, seq_len=992, h=h, transposed=True)
    assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s1, s2), "transposed"


@requires_mx_cvt
@pytest.mark.parametrize("n", [5120, 17408])
@pytest.mark.parametrize("t", [256, 992, 1000, 2016])
def test_canonical_rowwise_is_bitwise_the_padded_swizzle(t, n):
    """``sf_layout="gemm"`` rowwise on the ``[T, N/D, D]`` view of a ``[T, N]`` bf16 gradient (N = the 397B slab width or a
    narrow one): the e4m3 codes equal ``quantize_to_mxfp8``'s rowwise ``_d`` payload AND the block reference's
    ``mx_quantize_rowwise_2d`` byte for byte, the SF blob equals ``mx_swizzle_sf_rowwise_padded(row_e)`` (the padded canonical
    blob the block-scale dgrad binds), every byte written (the sentinel gone, pad rows ``0x00``), the batch folded into the rows
    (B = 2 wherever T is even)."""
    torch.manual_seed(zlib.crc32(f"canon-{t}-{n}".encode()) & 0xFFFF)
    h = n // D
    b = 2 if t % 2 == 0 else 1
    x = (torch.randn(t, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst, sf = _launch_canonical(AXIS_ROW, x, batch=b, seq_len=t // b, h=h, transposed=False)
    assert sf.numel() == sf_blob_bytes(t, n)
    _check_canonical(x, dst, sf, transposed=False)
    # the pad rows of the last 128-row band carry 0x00 in the blob, by the host formula
    if t % 128:
        pad = [sf_byte_canonical(r, c, k=n) for r in range(t, -(-t // 128) * 128) for c in range(0, n // 32, 7)]
        assert int(sf[pad].max().item()) == 0


@requires_mx_cvt
@pytest.mark.parametrize("n", [5120, 17408])
@pytest.mark.parametrize("t", [256, 992, 2016])
def test_transposed_columnwise_is_bitwise_the_padded_swizzle(t, n):
    """``sf_layout="gemm", transposed=True``: the ``[T, N]`` gradient quantized along the TOKENS (32-token blocks) and stored
    as the contiguous e4m3 ``[N, T]`` matrix equals ``mx_quantize_rowwise_2d(x^T)`` (the artifact builder's own contract for
    ``h_t``) and ``quantize_to_mxfp8``'s columnwise ``_s`` payload transposed, byte for byte; the SF blob equals
    ``mx_swizzle_sf_rowwise_padded(col_e^T)`` over ``(rows = N, K = T)`` with the pad blocks of a ragged-128 T ``0x00``;
    T = 992 / 2016 are 31 / 63 blocks (a 128-tile with one pad block)."""
    torch.manual_seed(zlib.crc32(f"transposed-{t}-{n}".encode()) & 0xFFFF)
    h = n // D
    x = (torch.randn(t, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst, sf = _launch_canonical(AXIS_COL, x, batch=2, seq_len=t // 2, h=h, transposed=True)
    assert tuple(dst.shape) == (n, t) and dst.is_contiguous() and sf.numel() == sf_blob_bytes(n, t)
    _check_canonical(x, dst, sf, transposed=True)
    if (t // 32) % 4:
        pad = [sf_byte_canonical(r, c, k=t) for r in range(0, n, 97) for c in range(t // 32, -(-(t // 32) // 4) * 4)]
        assert int(sf[pad].max().item()) == 0


@requires_mx_cvt
def test_transposed_arm_declines_a_ragged_t():
    """T = 1000 (not a multiple of 32) is a typed decline of the transposed arm at run_ -- named 32 -- and nothing launches
    (the sentinel-filled outputs are untouched); the same T is SERVED by the canonical rowwise arm (its pad rows are 0x00)."""
    torch.manual_seed(9)
    h, t = 2, 1000
    x = torch.randn(t, h, D, device="cuda").to(torch.bfloat16)
    r = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=D, axis=AXIS_COL, sf_layout=SF_LAYOUT_GEMM, transposed=True)
    dst = torch.full((h * D, t), 0xFF, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    sf = torch.full((sf_blob_bytes(h * D, 1024),), 0xFF, dtype=torch.uint8, device="cuda")  # the count a padded T would take
    with pytest.raises(ValueError, match="multiple of 32"):
        run_quantize_mxfp8(r, x, dst, sf, batch=1, seq_len=t, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert int((dst.view(torch.uint8) != 0xFF).sum().item()) == 0 and int((sf != 0xFF).sum().item()) == 0, "a refused call launched"
    dst_row, sf_row = _launch_canonical(AXIS_ROW, x, batch=1, seq_len=t, h=h, transposed=False)
    _check_canonical(x, dst_row, sf_row, transposed=False)


@requires_mx_cvt
def test_run_declines_a_mismatched_binding():
    h, b, s = 2, 1, 256
    r = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=D, axis=AXIS_ROW)
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    dst = torch.empty(b * s, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.zeros(sf_bytes(b, h, s, D), dtype=torch.uint8, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        run_quantize_mxfp8(r, x, torch.empty_like(x), sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="H=2"):
        run_quantize_mxfp8(r, torch.randn(b * s, 4, D, device="cuda").to(torch.bfloat16), dst, sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="batch\\*seq_len"):
        run_quantize_mxfp8(r, x, dst, sf, batch=2, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="uint8"):
        run_quantize_mxfp8(r, x, dst, sf.view(torch.float8_e8m0fnu), batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="ceil\\(S/128\\)"):
        run_quantize_mxfp8(r, x, dst, sf[:-1], batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="compiled for"):
        run_quantize_mxfp8(r, x.to(torch.float16), dst, sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="CUDA tensor"):
        run_quantize_mxfp8(r, x.cpu(), dst, sf, batch=b, seq_len=s, stream=st)
    slab = torch.zeros(b * s, N_QKVG, device="cuda", dtype=torch.bfloat16)
    off_by_8_bytes = torch.as_strided(slab, (b * s, h, D), (N_QKVG, D, 1), storage_offset=4)  # strides fine, base 8 B off a 16-B line
    assert off_by_8_bytes.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        run_quantize_mxfp8(r, off_by_8_bytes, dst, sf, batch=b, seq_len=s, stream=st)
