# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state a decode loop carries between steps: a plain key/value cache per attention layer, and per linear-attention layer a
pool of recurrent states and conv histories addressed by slot ids.

The set of sequences is fixed when the state is built.  Slot 0 of every pool is the null slot (a row whose slot id is 0 is
padding for the fused step and touches nothing); sequence ``i`` owns slot ``i + 1``.  The conv history is stored tap-major,
``[slots, taps, channels]``, and handed to a block as the ``[slots, channels, taps]`` view of it: the layout a serving engine
keeps and the fused step is validated on.

``snapshot`` / ``restore`` exist for the replay and batch-invariance checks of ``tie_out``; a sample has no other use for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class KvCache:
    """Keys and values of one attention layer, ``[sequences, kv_heads, capacity, head_dim]`` bf16; rows up to the step's
    position are valid."""

    k: torch.Tensor
    v: torch.Tensor

    def tensors(self):
        return (self.k, self.v)

    def rows(self, start: int, stop: int) -> "KvCache":
        """The cache of some of the sequences, as views of the same memory."""
        return KvCache(self.k[start:stop], self.v[start:stop])


@dataclass
class RecurrentSlots:
    """The pool of one linear-attention layer: ``state`` ``[slots, value_heads, head_dim, head_dim]`` fp32 and the conv history,
    stored ``[slots, taps, channels]`` bf16."""

    state: torch.Tensor
    conv_storage: torch.Tensor

    @property
    def conv(self) -> torch.Tensor:
        """The ``[slots, channels, taps]`` view a block takes."""
        return self.conv_storage.transpose(1, 2)

    def tensors(self):
        return (self.state, self.conv_storage)


class DecodeState:
    """Per-layer state of ``sequences`` fixed sequences plus the position of the next token."""

    def __init__(self, layer_kinds: Sequence[str], sequences: int, capacity: int, shapes: Dict[str, tuple], device):
        """``shapes``: ``"state"`` -> (value_heads, head_dim, head_dim), ``"conv"`` -> (taps, channels), ``"kv"`` -> (kv_heads, head_dim)."""
        if sequences < 1 or capacity < 1:
            raise ValueError(f"a decode state needs at least one sequence and one position; got sequences={sequences}, capacity={capacity}")
        self.sequences, self.capacity, self.position = int(sequences), int(capacity), 0
        self.device = torch.device(device)
        self.slot_ids = torch.arange(1, sequences + 1, dtype=torch.int32, device=self.device)
        self.layers: List[object] = []
        for kind in layer_kinds:
            if kind == "linear_attention":
                self.layers.append(
                    RecurrentSlots(
                        torch.zeros((sequences + 1, *shapes["state"]), dtype=torch.float32, device=self.device),
                        torch.zeros((sequences + 1, *shapes["conv"]), dtype=torch.bfloat16, device=self.device),
                    )
                )
            else:
                kv = (sequences, shapes["kv"][0], capacity, shapes["kv"][1])
                self.layers.append(
                    KvCache(torch.zeros(kv, dtype=torch.bfloat16, device=self.device), torch.zeros(kv, dtype=torch.bfloat16, device=self.device))
                )

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for layer in self.layers for t in layer.tensors())

    def snapshot(self) -> dict:
        """A copy of every state tensor and the position."""
        return {"position": self.position, "layers": [tuple(t.clone() for t in layer.tensors()) for layer in self.layers]}

    def restore(self, snap: dict) -> None:
        """Write a snapshot back into the existing allocations."""
        self.position = snap["position"]
        for layer, saved in zip(self.layers, snap["layers"]):
            for t, s in zip(layer.tensors(), saved):
                t.copy_(s)
