# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state a decode loop carries between steps: a plain key/value cache per attention layer, and per linear-attention layer a
pool of recurrent states and conv histories addressed by slot ids.

The set of sequences is fixed when the state is built.  Slot 0 of every pool is the null slot (a row whose slot id is 0 is
padding for the fused step and touches nothing); sequence ``i`` owns slot ``i + 1``.  The conv history is stored tap-major,
``[slots, taps, channels]``, and handed to a block as the ``[slots, channels, taps]`` view of it: the layout a serving engine
keeps and the fused step is validated on.

**A step of several tokens per sequence** (a verify step: ``tokens`` = T > 1) keeps a recurrent state per (sequence, token):
the state after token ``t`` of sequence ``i`` is written to the slot of that pair, so that the step after it can start from
whichever token the run kept.  The slots are numbered token by token -- (sequence ``i``, token ``t``) owns slot ``1 + t *
sequences + i`` -- so token 0 of a sequence is its plain slot ``i + 1``, and the layout of one token per sequence is this
layout at T = 1.  The conv history of a sequence stays in its plain slot and holds ``T + 2`` columns: a step reads its
three taps where the last step left them and writes ``[the last two of them, x of token 0, .., x of token T - 1]``.
:class:`TokenSlots` is that table as a step hands it to its block, and ``DecodeState.accept`` says how many tokens behind
the first the run kept: the state then stands at that token, with nothing copied.

**A decode step on such a state** (the prompt of a session that verifies behind it) works on token 0: the slots of token 0
are the plain slots, the history columns 0 to 2 the plain history, the cache rows the plain cache.  ``DecodeState.decode_layers``
are those views -- the same memory in the layout of one token per sequence, built with the state -- and
``DecodeState.prompt_layers()`` hands them out while the state stands at token 0 in front of its first verify step.

``snapshot`` / ``restore`` exist for the replay and batch-invariance checks of ``tie_out``; a sample has no other use for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class KvCache:
    """Keys and values of one attention layer, ``[sequences, kv_heads, capacity, head_dim]`` bf16; rows up to the step's
    position are valid.  ``tokens``: the tokens per sequence of the steps that write it (a step inserts that many
    positions per sequence)."""

    k: torch.Tensor
    v: torch.Tensor
    tokens: int = 1

    def tensors(self):
        return (self.k, self.v)

    def rows(self, start: int, stop: int) -> "KvCache":
        """The cache of the sequences that own the rows ``start`` to ``stop`` of a step (``tokens`` rows each), as views of
        the same memory."""
        return KvCache(self.k[start // self.tokens : stop // self.tokens], self.v[start // self.tokens : stop // self.tokens], self.tokens)

    def of(self, sequence: int) -> tuple:
        """Where ``sequence`` lives in each tensor of ``tensors()``."""
        return (sequence, sequence)

    def first_token(self) -> "KvCache":
        """The same cache as a decode step takes it: one position per sequence and step."""
        return KvCache(self.k, self.v, 1)


@dataclass
class TokenSlots:
    """The slot table of a step that feeds several tokens per sequence, as its block takes it: ``indices`` ``[sequences,
    tokens]`` int32, the slot of every (sequence, token); ``cu_seqlens`` ``[sequences + 1]`` int32, the first row of every
    sequence among the step's rows (``tokens`` rows each); and how many tokens of the last step were kept.  ``accept`` is
    that number behind the first token, 0 .. tokens - 1: a step starts from the slot of token ``accept`` and from the
    three history columns ``accept .. accept + 2``.  A kernel reads it as ``accepted`` (one int32 per sequence, ``accept +
    1``): a row of ``counts``, a constant table built with the state, so that keeping tokens launches and copies nothing."""

    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    counts: torch.Tensor  # [tokens, sequences rounded up to four] int32: row a holds a + 1
    accept: int = 0

    @classmethod
    def build(cls, sequences: int, tokens: int, device) -> "TokenSlots":
        """The table of ``sequences`` sequences at ``tokens`` tokens each: (sequence ``i``, token ``t``) -> slot ``1 + t *
        sequences + i``."""
        token = torch.arange(tokens, dtype=torch.int32, device=device)
        indices = (1 + token[None, :] * sequences + torch.arange(sequences, dtype=torch.int32, device=device)[:, None]).contiguous()
        cu_seqlens = torch.arange(0, sequences * tokens + 1, tokens, dtype=torch.int32, device=device)
        # a row of four elements or a multiple: every row then starts 16-byte aligned, as a kernel's operand must
        counts = (token[:, None] + 1).expand(tokens, -(-sequences // 4) * 4).contiguous()
        return cls(indices, cu_seqlens, counts)

    @property
    def sequences(self) -> int:
        return int(self.indices.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.indices.shape[1])

    @property
    def accepted(self) -> torch.Tensor:
        """``[sequences]`` int32: the tokens of the last step that were kept, the first included (1 .. tokens)."""
        return self.counts[self.accept, : self.sequences]

    def rows(self, start: int, stop: int) -> "TokenSlots":
        """The table of the sequences that own the rows ``start`` to ``stop`` of a step: their slots in the same pools, in
        tensors of their own (a kernel's operand starts 16-byte aligned, which a row of the table need not)."""
        first, last = start // self.tokens, stop // self.tokens
        cu_seqlens = (self.cu_seqlens[first : last + 1] - self.cu_seqlens[first]).contiguous()
        counts = self.counts[:, : -(-(last - first) // 4) * 4].contiguous()
        return TokenSlots(self.indices[first:last].clone(), cu_seqlens, counts, self.accept)

    def tensors(self):
        return ()  # a table: no call writes it


@dataclass
class RecurrentSlots:
    """The pool of one linear-attention layer: ``state`` ``[slots, value_heads, head_dim, head_dim]`` fp32 and the conv history,
    stored ``[slots, taps, channels]`` bf16.  At ``tokens`` tokens per sequence the state pool holds a slot per (sequence,
    token) and the conv history ``tokens + 2`` columns per sequence, in the sequence's plain slot."""

    state: torch.Tensor
    conv_storage: torch.Tensor
    tokens: int = 1

    @property
    def conv(self) -> torch.Tensor:
        """The ``[slots, channels, taps]`` view a block takes."""
        return self.conv_storage.transpose(1, 2)

    def tensors(self):
        return (self.state, self.conv_storage)

    def of(self, sequence: int) -> tuple:
        """Where ``sequence`` lives in each tensor of ``tensors()``: its state slots, one per token, and its history."""
        sequences = (self.state.shape[0] - 1) // self.tokens
        return ([1 + token * sequences + sequence for token in range(self.tokens)] if self.tokens > 1 else sequence + 1, sequence + 1)

    def first_token(self) -> "RecurrentSlots":
        """Token 0 of this pool in the layout of one token per sequence, on the same memory: the null slot and the slots of
        token 0 (the plain slots), and the history columns 0 to 2.  A decode step that is handed this view reads and writes
        what a step of several tokens reads at count 1; the slots of the later tokens and the later columns are not part
        of it.  The history view keeps the pool's slot stride: ``tokens + 2`` columns apart."""
        sequences = (self.state.shape[0] - 1) // self.tokens
        taps = self.conv_storage.shape[1] - self.tokens + 1
        return RecurrentSlots(self.state[: sequences + 1], self.conv_storage[:, :taps], 1)


class DecodeState:
    """Per-layer state of ``sequences`` fixed sequences plus the position of the next token.  ``tokens``: the tokens per
    sequence a step of the run feeds (1: a decode step; more: a verify step, whose table is ``slots``)."""

    def __init__(self, layer_kinds: Sequence[str], sequences: int, capacity: int, shapes: Dict[str, tuple], device, tokens: int = 1):
        """``shapes``: ``"state"`` -> (value_heads, head_dim, head_dim), ``"conv"`` -> (taps, channels), ``"kv"`` -> (kv_heads, head_dim)."""
        if sequences < 1 or capacity < 1 or tokens < 1:
            raise ValueError(
                f"a decode state needs at least one sequence, one position and one token per sequence; got sequences={sequences}, capacity={capacity}, tokens={tokens}"
            )
        self.sequences, self.capacity, self.position, self.tokens = int(sequences), int(capacity), 0, int(tokens)
        self.device = torch.device(device)
        self.slot_ids = torch.arange(1, sequences + 1, dtype=torch.int32, device=self.device)
        #: the slot table of a step of several tokens per sequence; a decode step takes ``slot_ids``
        self.slots: Optional[TokenSlots] = TokenSlots.build(self.sequences, self.tokens, self.device) if self.tokens > 1 else None
        taps, channels = shapes["conv"]
        self.layers: List[object] = []
        for kind in layer_kinds:
            if kind == "linear_attention":
                self.layers.append(
                    RecurrentSlots(
                        torch.zeros((sequences * self.tokens + 1, *shapes["state"]), dtype=torch.float32, device=self.device),
                        torch.zeros((sequences + 1, taps + self.tokens - 1, channels), dtype=torch.bfloat16, device=self.device),
                        self.tokens,
                    )
                )
            else:
                kv = (sequences, shapes["kv"][0], capacity, shapes["kv"][1])
                self.layers.append(
                    KvCache(torch.zeros(kv, dtype=torch.bfloat16, device=self.device), torch.zeros(kv, dtype=torch.bfloat16, device=self.device), self.tokens)
                )
        #: the layers as a decode step takes them: the layers themselves at one token per sequence, else their token-0 views
        self.decode_layers: List[object] = self.layers if self.tokens == 1 else [layer.first_token() for layer in self.layers]
        #: the steps of several tokens per sequence this state has run: a decode step comes in front of the first of them
        self.verify_steps = 0

    @property
    def accepted(self) -> int:
        """The tokens behind the first that the last step kept: where the state stands (0 for a decode step)."""
        return 0 if self.slots is None else self.slots.accept

    def accept(self, count: int) -> None:
        """Keep ``count`` tokens behind the first of the step that has just run (0 .. tokens - 1): the next step starts from
        the state after that token and ``count + 1`` positions on.  Nothing is copied -- the slots of the later tokens and
        the cache rows behind the kept ones are simply not read again."""
        if self.slots is None:
            raise ValueError("a decode state of one token per sequence keeps every token: there is nothing to accept")
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count < self.tokens:
            raise ValueError(f"a step of {self.tokens} tokens per sequence keeps 0 to {self.tokens - 1} tokens behind the first; got {count!r}")
        self.slots.accept = count
        self.position += count + 1

    def prompt_layers(self) -> List[object]:
        """The layers a decode step is handed (``decode_layers``).  On a state of several tokens per sequence a decode step
        is a step of the prompt: it works on token 0 and comes in front of the first verify step.  Behind one -- or with
        the state standing at a later token -- its views would name slots and columns the run no longer stands on, and
        that is a ``ValueError``."""
        if self.tokens > 1 and (self.accepted or self.verify_steps):
            steps = self.verify_steps
            raise ValueError(
                "a decode step on a state of several tokens per sequence starts from token 0, in front of its first verify step; "
                f"this state stands at token {self.accepted} behind {steps} verify step{'' if steps == 1 else 's'}"
            )
        return self.decode_layers

    def standing(self) -> List[tuple]:
        """Per layer what the next step starts from, as tensors in the layout of one token per sequence: the recurrent state
        ``[sequences, ...]`` and the three history taps ``[sequences, taps, channels]`` of a linear-attention layer, the
        cached keys and values up to the position of an attention layer.  Two runs of the same tokens are compared on it,
        whatever tokens per sequence each fed."""
        kept = self.accepted
        slots = (self.slot_ids + kept * self.sequences).long()
        standing = []
        for layer in self.layers:
            if isinstance(layer, RecurrentSlots):
                taps = layer.conv_storage.shape[1] - self.tokens + 1
                standing.append((layer.state[slots], layer.conv_storage[1:, kept : kept + taps]))
            else:
                standing.append((layer.k[:, :, : self.position], layer.v[:, :, : self.position]))
        return standing

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for layer in self.layers for t in layer.tensors())

    def snapshot(self) -> dict:
        """A copy of every state tensor, the position, the tokens kept and the verify steps run."""
        return {
            "position": self.position,
            "accepted": self.accepted,
            "verify_steps": self.verify_steps,
            "layers": [tuple(t.clone() for t in layer.tensors()) for layer in self.layers],
        }

    def restore(self, snap: dict) -> None:
        """Write a snapshot back into the existing allocations."""
        self.position = snap["position"]
        self.verify_steps = snap.get("verify_steps", 0)
        if self.slots is not None:
            self.slots.accept = snap.get("accepted", 0)
        for layer, saved in zip(self.layers, snap["layers"]):
            for t, s in zip(layer.tensors(), saved):
                t.copy_(s)
