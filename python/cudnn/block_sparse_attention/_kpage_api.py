# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage entry points behind the ``kpage=`` argument of the BSA wrappers."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .kpage._layout import KPageLayout

KPAGE_MIN_CUTEDSL = (4, 6, 2)


def _require_cutedsl() -> None:
    from ._interface import _cutlass_dsl_version

    version = _cutlass_dsl_version()
    if version < KPAGE_MIN_CUTEDSL:
        raise RuntimeError(
            "KPage block-sparse attention requires nvidia-cutlass-dsl>="
            + ".".join(map(str, KPAGE_MIN_CUTEDSL))
            + "; found "
            + ".".join(map(str, version))
            + ". The other BSA paths remain available."
        )


def _normalize_metadata(
    q2k_block_index: torch.Tensor,
    block_sparse_num: Optional[int],
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
    q_block_sizes: Optional[torch.Tensor],
    num_kv_blocks: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand the BSA metadata to the four tensors the layout builder consumes.

    The batch is folded into the head dimension: the layout tables are per
    (batch, head) and ``block_sizes`` is shared across the batch anyway.
    """
    if q2k_block_index.ndim != 4 or q2k_block_index.dtype != torch.int32 or not q2k_block_index.is_cuda:
        raise ValueError("q2k_block_index must be a rank-4 int32 CUDA tensor")
    if not q2k_block_index.is_contiguous() or (q2k_block_nums is not None and not q2k_block_nums.is_contiguous()):
        raise NotImplementedError("kpage requires contiguous q2k_block_index and q2k_block_nums")
    batch, heads, num_q_blocks, capacity = q2k_block_index.shape
    q2k_block_index = q2k_block_index.view(1, batch * heads, num_q_blocks, capacity)
    if q2k_block_nums is None:
        if block_sparse_num is None:
            block_sparse_num = capacity
        if not 1 <= block_sparse_num <= capacity:
            raise ValueError(f"block_sparse_num must be in [1, {capacity}], got {block_sparse_num}")
        q2k_block_nums = torch.full((1, batch * heads, num_q_blocks), block_sparse_num, dtype=torch.int32, device=q2k_block_index.device)
    else:
        q2k_block_nums = q2k_block_nums.view(1, batch * heads, num_q_blocks)
    if block_sizes is None:
        if num_kv_blocks is None:
            num_kv_blocks = num_q_blocks
        block_sizes = torch.full((num_kv_blocks,), 128, dtype=torch.int32, device=q2k_block_index.device)
    elif block_sizes.ndim != 1:
        raise NotImplementedError("kpage takes block_sizes of shape (num_kv_blocks,)")
    elif num_kv_blocks is not None and block_sizes.numel() != num_kv_blocks:
        raise ValueError(f"block_sizes has {block_sizes.numel()} entries, seqlen_k spans {num_kv_blocks} KV blocks")
    if q_block_sizes is None:
        q_block_sizes = torch.full((num_q_blocks,), 128, dtype=torch.int32, device=q2k_block_index.device)
    return q2k_block_index, q2k_block_nums, block_sizes, q_block_sizes


def _metadata_source(q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums) -> tuple:
    """Identity of the metadata arguments: storage, shape and strides of each tensor.

    A layout remembers what it was built from so a call that passes it with
    other metadata is rejected instead of silently running the wrong pattern.
    The tensors themselves are kept alive by the layout, so storage addresses
    cannot be recycled behind the check.
    """

    def ident(t):
        return None if t is None else (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t)

    # block_sparse_num only matters with fixed counts, and None means "the whole row"
    if q2k_block_nums is not None:
        block_sparse_num = None
    elif block_sparse_num is None:
        block_sparse_num = int(q2k_block_index.shape[-1])
    return (ident(q2k_block_index), block_sparse_num, ident(block_sizes), ident(q2k_block_nums))


def _same_source(a: tuple, b: tuple) -> bool:
    return all(x[:3] == y[:3] if isinstance(x, tuple) and isinstance(y, tuple) else x == y for x, y in zip(a, b))


def prepare_layout(
    q2k_block_index: torch.Tensor,
    block_sparse_num: Optional[int],
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
    *,
    q_block_sizes: Optional[torch.Tensor] = None,
    seqlen_q: Optional[int] = None,
    seqlen_k: Optional[int] = None,
    validate: bool,
    allow_empty_block_nums: bool = False,
    bwd_kv_groups: Optional[int],
    with_backward: bool = False,
) -> KPageLayout:
    _require_cutedsl()
    major, minor = torch.cuda.get_device_capability(q2k_block_index.device)
    if major != 10:
        raise NotImplementedError(f"kpage requires an SM100/SM103 GPU, found SM{major}{minor}")
    for name, seqlen in (("seqlen_q", seqlen_q), ("seqlen_k", seqlen_k)):
        if seqlen is not None and seqlen < 1:
            raise ValueError(f"{name} must be positive, got {seqlen}")
    source = _metadata_source(q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums)
    idx, nums, kv_sizes, q_sizes = _normalize_metadata(
        q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums, q_block_sizes, None if seqlen_k is None else (seqlen_k + 127) // 128
    )
    from .kpage._layout import prepare_layout as _prepare

    with torch.cuda.device(idx.device):
        layout = _prepare(
            idx,
            nums,
            kv_sizes,
            q_sizes,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            validate=validate,
            allow_empty_block_nums=allow_empty_block_nums,
            bwd_kv_groups=bwd_kv_groups,
            source=source,
        )
        if with_backward:
            layout.build_backward()
    return layout


def resolve_layout(
    kpage: Union[bool, KPageLayout],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: Optional[int],
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
    *,
    arch: int,
    sparse_block_size: int,
    allow_empty_block_nums: bool,
    extra: Tuple[Optional[torch.Tensor], ...] = (),
) -> KPageLayout:
    """Check the KPage contract for one call (tensors as BHSD views) and return the layout to run."""
    _require_cutedsl()
    if arch // 10 != 10:  # the kernels are validated on SM100/SM103 only
        raise NotImplementedError(f"kpage requires an SM100/SM103 GPU, found SM{arch}")
    if sparse_block_size != 128:
        raise NotImplementedError("kpage requires sparse_block_size=128")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("kpage requires BF16 q, k, and v")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise NotImplementedError("kpage requires k and v of one shape with the batch and head_dim of q")
    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    if heads % k.shape[1]:
        raise ValueError("the number of query heads must be divisible by the number of KV heads")
    if head_dim != 128:
        raise NotImplementedError("kpage requires head_dim=128")
    for name, t in (("q", q), ("k", k), ("v", v), *zip(("do_tensor", "o_tensor", "lse_tensor", "dq_tensor", "dk_tensor", "dv_tensor"), extra)):
        if t is not None:
            folded = _fold(t, name)
            if not (folded.is_contiguous() or (folded.ndim == 4 and folded.transpose(1, 2).is_contiguous())):
                raise NotImplementedError(f"kpage requires a compact BHSD or BSHD {name}, got strides {tuple(t.stride())}")

    if isinstance(kpage, KPageLayout):
        if kpage.device != q.device:
            raise ValueError(f"the KPageLayout was built on {kpage.device}, q is on {q.device}")
        if (kpage.num_heads, kpage.seqlen_q, kpage.seqlen_k) != (batch * heads, seqlen_q, seqlen_k):
            raise ValueError(
                f"the KPageLayout describes {kpage.num_heads} (batch x heads) x {kpage.seqlen_q} query / {kpage.seqlen_k} key tokens, "
                f"the call has {batch} x {heads} x {seqlen_q} / {seqlen_k}"
            )
        if not _same_source(kpage._source, _metadata_source(q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums)):
            raise ValueError("the KPageLayout was built from different sparse metadata; pass the same tensors, rebuild it, or use kpage=True")
        return kpage
    if kpage is not True:
        raise TypeError(f"kpage must be False, True, or a KPageLayout, got {type(kpage).__name__}")
    return prepare_layout(
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        q2k_block_nums,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        validate=True,
        allow_empty_block_nums=allow_empty_block_nums,
        bwd_kv_groups=None,
    )


def _fold(t: Optional[torch.Tensor], name: str = "tensor") -> Optional[torch.Tensor]:
    """[B, H, ...] -> [1, B * H, ...] view; the kernels run a batch as batch x heads."""
    if t is None:
        return None
    try:
        return t.view(1, t.shape[0] * t.shape[1], *t.shape[2:])
    except RuntimeError:
        raise NotImplementedError(
            f"kpage runs a batch as batch x heads and needs {name} to be viewable as [1, batch * heads, ...]; a BSHD batch > 1 is not"
        ) from None


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout_obj: KPageLayout, softmax_scale: Optional[float]):
    from .kpage import _interface

    batch, heads = q.shape[:2]
    out, lse = _interface.forward(_fold(q), _fold(k), _fold(v), layout_obj, softmax_scale)
    return out.view(batch, heads, *out.shape[2:]), lse.view(batch, heads, lse.shape[2])


def backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    layout_obj: KPageLayout,
    softmax_scale: Optional[float],
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
):
    from .kpage import _interface

    batch = q.shape[0]
    grads = _interface.backward(_fold(do), _fold(q), _fold(k), _fold(v), _fold(o), _fold(lse), layout_obj, softmax_scale, _fold(dq), _fold(dk), _fold(dv))
    return tuple(g.view(batch, g.shape[1] // batch, *g.shape[2:]) for g in grads)
