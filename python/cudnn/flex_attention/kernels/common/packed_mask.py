# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.

"""Host diagnostics for the public arbitrary interval-mask representation."""

import torch


def interval_endpoints_to_dense(endpoints: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    """Materialize ``[0,F0), [F1,F2), ...`` semantics for host diagnostics."""

    if endpoints.ndim != 1 or endpoints.numel() <= 0 or endpoints.numel() % 2 == 0:
        raise ValueError("endpoints must be a non-empty odd rank-1 tensor")
    if seqlen_k < 0:
        raise ValueError("seqlen_k must be non-negative")
    if bool(((endpoints < 0) | (endpoints > seqlen_k)).any().item()):
        raise ValueError("endpoints must lie in [0, seqlen_k]")
    if endpoints.numel() > 1 and bool((endpoints[1:] < endpoints[:-1]).any().item()):
        raise ValueError("endpoints must be nondecreasing")
    columns = torch.arange(seqlen_k, device=endpoints.device)
    visible = columns < endpoints[0]
    for index in range(1, endpoints.numel(), 2):
        visible |= (columns >= endpoints[index]) & (columns < endpoints[index + 1])
    return visible
