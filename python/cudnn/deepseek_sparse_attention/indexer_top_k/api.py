"""APIBase wrapper for the DSA indexer top-K CuTe DSL kernel."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict

from .local_to_global_dsl import local_to_global as _local_to_global
from .compactify import compactify as _compactify

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _get_cute_dsl_topk_wrapper():
    from .indexer_top_k_decode_varlen import cute_dsl_topk_wrapper

    return cute_dsl_topk_wrapper


class IndexerTopK(APIBase):
    """Top-K filter using the SM100 CuTe-DSL radix kernel.

    Selects the ``top_k`` largest entries from each row of ``input_values``.

    Parameter conventions (important — easy to misuse)
    --------------------------------------------------
    ``input_values`` has shape ``(n_rows, num_cols)``. The kernel treats the
    rows as belonging to ``batch_size = seq_lens.numel()`` groups, with
    ``n_rows == batch_size * next_n`` exactly. Within each group of
    ``next_n`` rows the kernel applies a **speculative-decoding stagger**:
    the effective valid length for row ``task_id`` inside batch ``b`` is
    ``seq_lens[b] - next_n + (task_id % next_n) + 1`` (i.e. the first row in
    a group sees the shortest prefix, the last row sees the full
    ``seq_lens[b]``).

    Use cases:

    * **Independent per-row top-K over equal-length rows** (the common
      case): set ``next_n = 1``, ``batch_size = n_rows``, ``seq_lens`` a
      length-``n_rows`` tensor. Each row then sees its own ``seq_lens[i]``
      columns.
    * **Speculative decoding / medusa-style drafts**: set ``next_n`` to the
      number of draft tokens per batch; ``seq_lens`` describes the cache
      length per batch. The kernel produces the staggered behaviour
      automatically.

    Setting ``next_n > 1`` with ``batch_size < n_rows / next_n`` causes the
    stagger formula to produce non-positive lengths for early rows; those
    rows receive no kernel writes and the output buffer stays at its
    initial (``-1``) state for them.

    Notes
    -----
    The underlying :func:`cute_dsl_topk_wrapper` already owns a compilation
    cache keyed on the next-power-of-two of ``num_cols``; this class's
    ``compile()`` is a no-op that simply primes that cache.
    """

    def __init__(
        self,
        sample_input_values: torch.Tensor,
        sample_seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
    ):
        super().__init__()
        self.input_desc = self._make_tensor_desc(sample_input_values, name="input_values")
        self.seq_lens_desc = self._make_tensor_desc(sample_seq_lens, name="seq_lens")
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.num_copy_bits = int(num_copy_bits)

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        self._check_dtype(self.input_desc, list(_SUPPORTED_DTYPES), name="input_values")
        self._check_dtype(self.seq_lens_desc, torch.int32, name="seq_lens")
        self._value_error_if(
            self.input_desc.ndim != 2,
            f"input_values must be 2-D (n_rows, num_cols), got {self.input_desc.shape}",
        )
        self._value_error_if(
            self.seq_lens_desc.ndim != 1,
            f"seq_lens must be 1-D, got {self.seq_lens_desc.shape}",
        )
        self._value_error_if(
            self.top_k <= 0 or self.top_k > 2048,
            f"top_k must be in (0, 2048], got {self.top_k}",
        )

        # Enforce the kernel's n_rows == batch_size * next_n invariant
        # up-front so misuse surfaces here rather than as silently-empty
        # rows (the stagger formula produces non-positive lengths when
        # next_n exceeds n_rows per batch).
        n_rows = self.input_desc.shape[0]
        batch_size = self.seq_lens_desc.shape[0]
        self._value_error_if(
            n_rows != batch_size * self.next_n,
            f"n_rows ({n_rows}) must equal seq_lens.numel() * next_n "
            f"({batch_size} * {self.next_n} = {batch_size * self.next_n}). "
            f"For independent top-K over equal-length rows use "
            f"next_n=1 and seq_lens of shape (n_rows,).",
        )

        major, _ = torch.cuda.get_device_capability()
        self._runtime_error_if(
            major < 10,
            f"IndexerTopK requires SM100+ compute capability, found SM{major}",
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        # cute_dsl_topk_wrapper compiles lazily on first call and caches
        # internally; nothing to do here. Keep the _compiled_kernel sentinel
        # non-None so APIBase.__call__() skips a second compile() attempt.
        self._compiled_kernel = _get_cute_dsl_topk_wrapper()

    def execute(
        self,
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._logger.debug("Entering execute")
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            self.compile()

        kernel = self._compiled_kernel or _get_cute_dsl_topk_wrapper()
        indices, values = kernel(
            input_values,
            seq_lens,
            self.top_k,
            self.next_n,
            return_val=self.return_val,
            num_copy_bits=self.num_copy_bits,
        )
        return indices, values


_cache_of_IndexerTopKObjects: dict = {}


def indexer_top_k_wrapper(
    input_values: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper returning ``{'indices', 'values'}``.

    ``input_values`` is ``(n_rows, num_cols)``; the kernel requires
    ``n_rows == seq_lens.numel() * next_n`` and applies a
    speculative-decoding length stagger within each ``next_n``-row group
    (first row sees the shortest prefix, last sees the full ``seq_lens``).
    For "independent top-K over every row" set ``next_n=1`` and make
    ``seq_lens`` a length-``n_rows`` tensor. See :class:`IndexerTopK` for
    full details.

    ``values`` is ``None`` when ``return_val=False``.
    """
    cache_key = (
        input_values.dtype,
        int(input_values.shape[0]),  # n_rows affects wrapper validation and output shape
        input_values.shape[-1],  # num_cols buckets internally
        int(seq_lens.shape[0]),  # batch_size / seq_lens.numel()
        int(top_k),
        int(next_n),
        bool(return_val),
        int(num_copy_bits),
    )
    obj = _cache_of_IndexerTopKObjects.get(cache_key)
    if obj is None:
        obj = IndexerTopK(
            sample_input_values=input_values,
            sample_seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
            num_copy_bits=num_copy_bits,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerTopKObjects[cache_key] = obj

    indices, values = obj.execute(input_values, seq_lens, current_stream=stream)
    return TupleDict(indices=indices, values=values)


def local_to_global_wrapper(
    local_indices: torch.Tensor,
    seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convert local top-K indices to the global index space."""
    indices = _local_to_global(
        local_indices,
        seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        stream=stream,
    )
    return TupleDict(indices=indices)


def compactify_wrapper(
    indices: torch.Tensor,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Pack valid indices row-wise and return ``indices`` plus ``topk_length``."""
    compact_indices, topk_length = _compactify(indices, stream=stream)
    return TupleDict(indices=compact_indices, topk_length=topk_length)
