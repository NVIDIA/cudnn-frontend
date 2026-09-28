# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 scale-factor tensors bind by STORAGE order, and bind zero-copy.

An F8_128x4 scale-factor tensor is an opaque byte layout: the kernel's
scale-factor descriptors read the reordered atom stream the producer laid down
in memory, NOT the tensor's logical element order.  Producers legally hand that
buffer over as a PERMUTED VIEW -- the atom-shaped tensor permuted into the
logical ``[.., mn, k]`` shape the graph declares (this is what the repo's own
mxfp8 benchmark harness builds).  On such a view the two orders differ, so
binding through ``.contiguous()`` would both copy and hand the kernel a
*different byte stream* than its descriptors read.

These tests pin physical-order binding in the retained dense tensor adapter
and the shared prepared pointer binder.  They are CPU-only: no GPU, no DSL, no
compile -- everything here is tensor metadata.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

import cudnn.sdpa.fwd.api_dsl as api_dsl
from cudnn.sdpa.fwd.api_dsl import _sf_storage_order_bytes

pytestmark = pytest.mark.L0


def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


# The atom-shaped scale-factor tensor and the permutation that turns it into the
# logical shape a graph declares -- the shape/permutation the mxfp8 benchmark
# harness uses (``benchmark/attention_training/benchmark_single_sdpa.py``).
# l=2 batch*head, 2 mn-atom-tiles, 3 k-atom-tiles, atom (32, 4) x 4.
_ATOM_SHAPE = (2, 2, 3, 32, 4, 4)
_ATOM_PERMUTE = (3, 4, 1, 5, 2, 0)
_ATOM_BYTES = 2 * 2 * 3 * 32 * 4 * 4


def _harness_view(flat: torch.Tensor) -> torch.Tensor:
    """The permuted 6-D view a scale-factor producer hands over."""
    return flat.reshape(_ATOM_SHAPE).permute(*_ATOM_PERMUTE)


def _reordered_bytes() -> torch.Tensor:
    """A distinguishable byte stream, as the reordering would have written it."""
    return torch.arange(_ATOM_BYTES, dtype=torch.int32).remainder(251).sub(125).to(torch.int8)


def test_permuted_view_binds_without_a_copy():
    """The harness view binds through the SAME storage -- no copy kernel."""
    flat = _reordered_bytes()
    view = _harness_view(flat)
    assert not view.is_contiguous(), "the harness view must be non-contiguous, else this test is vacuous"

    bound = _sf_storage_order_bytes(view, "sf_q")

    assert bound.data_ptr() == view.data_ptr()
    assert bound.dim() == 1 and bound.numel() == _ATOM_BYTES
    assert bound.dtype == torch.int8


def test_permuted_view_binds_the_storage_order_not_the_logical_order():
    """The bytes handed to the kernel are the producer's, not the logical ones.

    This is the correctness half: ``.contiguous()`` materializes the LOGICAL
    element order, which for this (non-trivial) permutation is a different byte
    stream -- i.e. scrambled scale factors, not merely a wasted copy.
    """
    flat = _reordered_bytes()
    view = _harness_view(flat)

    bound = _sf_storage_order_bytes(view, "sf_k")
    logical = view.contiguous().reshape(-1)

    assert torch.equal(bound, flat), "storage-order bind must reproduce the buffer the producer wrote"
    assert not torch.equal(bound, logical), "the permutation must be non-trivial, else this test is vacuous"


def test_an_already_contiguous_buffer_is_passed_through():
    """The common case (a flat, storage-order buffer) is untouched."""
    flat = _reordered_bytes()

    bound = _sf_storage_order_bytes(flat.reshape(_ATOM_SHAPE), "sf_v")

    assert bound.data_ptr() == flat.data_ptr()
    assert torch.equal(bound, flat)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="torch has no float8_e8m0fnu")
def test_an_e8m0_permuted_view_binds_as_bytes_zero_copy():
    """Scale factors typed E8M0 rather than int8 take the same path."""
    view = _harness_view(_reordered_bytes().view(torch.float8_e8m0fnu))
    assert not view.is_contiguous()

    bound = _sf_storage_order_bytes(view, "sf_q")

    assert bound.data_ptr() == view.data_ptr()
    assert bound.dtype == torch.int8 and bound.numel() == _ATOM_BYTES


def test_an_overlapping_view_is_rejected_by_name():
    """No byte stream exists for an overlapping tensor -- decline, do not copy."""
    overlapping = torch.empty(8, dtype=torch.int8).as_strided((4, 4), (1, 1))

    with pytest.raises(ValueError, match="sf_k"):
        _sf_storage_order_bytes(overlapping, "sf_k")


def test_a_gapped_slice_is_rejected_by_name():
    """A strided slice is not a dense buffer either: its bytes are not a stream."""
    gapped = torch.zeros((8, 8), dtype=torch.int8)[:, :4]

    with pytest.raises(ValueError, match="sf_v"):
        _sf_storage_order_bytes(gapped, "sf_v")


def test_legacy_dense_adapter_binds_through_the_helper():
    method = "_reshape_sf"
    """Regression guard: neither reshape path may reintroduce ``.contiguous()``.

    The copy is invisible in results -- it is correct for the contiguous inputs
    the kernels' own tests hand over, and only a permuted view exposes it -- so
    nothing downstream would report a revert.
    """
    fn = ast.parse(textwrap.dedent(inspect.getsource(getattr(api_dsl.SdpaFwdDslSm100, method)))).body[0]
    # Statements only: the prose in these docstrings names both spellings.
    code = "\n".join(ast.unparse(node) for node in fn.body if not _is_docstring(node))

    assert "_sf_storage_order_bytes(" in code
    assert "contiguous" not in code


def test_prepared_sf_preserves_storage_order_and_rejects_copies():
    from cudnn.sdpa.fwd.prepared import _sf_byte_count

    view = _harness_view(_reordered_bytes())
    assert _sf_byte_count(tuple(view.shape), tuple(view.stride()), "int8") == _ATOM_BYTES
    wide = _reordered_bytes().view(torch.int32)
    assert _sf_byte_count(tuple(wide.shape), tuple(wide.stride()), "int32") == _ATOM_BYTES
    for shape, strides in (((4, 4), (1, 1)), ((4, 4), (8, 1)), ((4, 4), (-4, 1))):
        with pytest.raises(ValueError):
            _sf_byte_count(shape, strides, "int8")


def _prepared_sf_facts(thd=False):
    from cudnn.sdpa.fwd.prepared import BufferFacts, DenseLaunchSpec, ThdLaunchSpec, QuantizedLaunchSpec

    spec = ThdLaunchSpec() if thd else DenseLaunchSpec()
    spec.device_index, spec.qh, spec.kh = 0, 2, 2
    if not thd:
        spec.b, spec.s_q_max, spec.s_k_max = 1, 256, 256
    spec.quant = QuantizedLaunchSpec(True, 0, (512, 512, 512))
    facts = {
        name: BufferFacts(0x10000 + i * 0x10000, "uint8", (2, 0), 2048, (1, 2, 2, 512), (2048, 1024, 512, 1)) for i, name in enumerate(("sf_q", "sf_k", "sf_v"))
    }
    facts.update(
        {name: BufferFacts(0x100000 + i * 0x10000, "float8_e4m3fn", (2, 0), 65536, (256, 2, 128), (256, 128, 1)) for i, name in enumerate(("q", "k", "v"))}
    )
    return spec, facts


@pytest.mark.parametrize("thd", [False, True])
@pytest.mark.parametrize("name", ["sf_q", "sf_k", "sf_v"])
@pytest.mark.parametrize("bad", ["short", "alignment", "device", "gapped", "overlap", "missing", "whole_tiles"])
def test_prepared_sf_rejects_invalid_runtime_facts(thd, name, bad):
    from cudnn.sdpa.fwd.prepared import _bind_mxfp8_scales

    spec, facts = _prepared_sf_facts(thd)
    f = facts[name]
    if bad == "short":
        f = f._replace(span=1024)
    elif bad == "alignment":
        f = f._replace(ptr=f.ptr + 1)
    elif bad == "device":
        f = f._replace(device=(1, 0))
    elif bad == "gapped":
        f = f._replace(strides=(4096, 2048, 1024, 1))
    elif bad == "overlap":
        f = f._replace(strides=(2048, 512, 512, 1))
    elif bad == "missing":
        f = None
    else:
        f = f._replace(shape=(2047,), strides=(1,))
    facts[name] = f
    with pytest.raises(ValueError):
        _bind_mxfp8_scales(spec, facts)


@pytest.mark.parametrize("thd", [False, True])
def test_prepared_sf_cache_rechecks_pointer_and_observed_span(thd):
    from cudnn.sdpa.fwd.prepared import _bind_mxfp8_scales

    spec, facts = _prepared_sf_facts(thd)
    assert _bind_mxfp8_scales(spec, facts)["sf_tiles"] == (2, 2, 2)
    original = facts["sf_q"]
    for replacement in (original._replace(span=0), original._replace(ptr=original.ptr + 4)):
        facts["sf_q"] = replacement
        with pytest.raises(ValueError):
            _bind_mxfp8_scales(spec, facts)
    facts["sf_q"] = original._replace(span=-1)
    assert _bind_mxfp8_scales(spec, facts)["sf_q_ptr"] == original.ptr


def test_prepared_sf_rejects_different_packed_kv_tile_counts():
    from cudnn.sdpa.fwd.prepared import _bind_mxfp8_scales

    spec, facts = _prepared_sf_facts(True)
    facts["sf_v"] = facts["sf_v"]._replace(shape=(1024,), strides=(1,), span=1024)
    with pytest.raises(ValueError, match="same packed tile count"):
        _bind_mxfp8_scales(spec, facts)
