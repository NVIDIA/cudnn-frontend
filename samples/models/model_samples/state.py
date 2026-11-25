# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state a decode loop carries between steps: a key/value cache per attention layer -- plain, or a pool of pages -- and per
linear-attention layer a pool of recurrent states and conv histories addressed by slot ids.

The set of sequences is fixed when the state is built.  Slot 0 of every pool is the null slot (a row whose slot id is 0 is
padding for the fused step and touches nothing); sequence ``i`` owns slot ``i + 1``.  The conv history is stored tap-major,
``[slots, taps, channels]``, and handed to a block as the ``[slots, channels, taps]`` view of it: the layout the fused
step is validated on.

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

**The attention cache has two layouts**, and a state is built in one of them.  The plain one (:class:`KvCache`) holds per
layer two tensors with one run of positions per sequence.  The paged one (:class:`PagedKvCache`, ``DecodeState(page=64)``)
holds per layer ONE pool of pages of ``page`` positions, keys in the first half of its last axis and values in the second
-- the layout a kernel that writes a paged attention cache is handed.  A sequence owns ``pages`` consecutive pages of the
pool for the whole run (the set of sequences is fixed, so no page is ever freed or handed on), and position ``p`` of
sequence ``i`` is the row ``p % page`` of the pool's page ``i * pages + p // page``: its *slot* is the flat index ``i *
pages * page + p``.  Whatever reads or writes a cache in plain PyTorch asks the cache -- ``write(position, k, v)``,
``keys(count, dtype)``, ``values(count, dtype)`` -- and gets the same bytes from either layout: a pool is reached by views
and by indexing with host integers alone, never through an index tensor (a gather is a kernel of its own, the one a
step's embedding lookup launches).  A kernel is handed the pool's two halves (``k_pool`` / ``v_pool``) and, for the step
that starts at a position, the rows of the state's tables (:class:`PageTable`, ``PagedKvCache.at(position)``): the cos / sin
table of every position, and the positions and slots of the step's rows.  The tables are constants, built once with the
state and written by no call, so moving on by one position or by a kept count launches and copies nothing.  A step of
several tokens that keeps fewer than it wrote restores nothing either, in a pool as in the plain cache: the next step
starts at the first rejected position and writes as many positions as the last one did, so every rejected row is written
again before a row behind the kept position is read.  A kernel that READS such a pool is handed two more tables
(``PagedKvCache.attended(position)``): the page table, which says which pages of the pool a sequence owns, and the
positions each sequence holds after the step.  A state holds them where it is given the run's maximum length
(``DecodeState(max_model_len=)``), which a session does where a block of its plans reads them.

**What a drafter reads of the step before it** is part of the state too: ``hidden``, the final-norm rows of the model's last
step, by reference -- the tensor that step's last norm returned, never a copy -- and ``greedy``, the ids that step chose
(row ``t``: the token that follows row ``t``); and ``draft``, the drafter's own attention cache, which a session with a
drafter builds with the state.  A family's step sets the first two; a state without a drafter has no third.

**What a recorded decode step reads** (:class:`Cursor`, ``DecodeState.step_cursor()``).  A step that is recorded once as a
launch sequence and replayed for every token runs no Python between two tokens, so nothing of it may be selected by a
host integer: a recorded launch keeps the ADDRESS of every operand it was recorded with.  The token a step is fed and the
rows of the position it stands at -- the position rows, the slot rows and the lengths, which :meth:`PagedKvCache.at` and
:meth:`PagedKvCache.attended` hand out as rows of the tables -- are therefore held once more at fixed addresses: the ids
buffer and three rows of the cursor, set from the tables (``Cursor.set``) and moved on by one position in place
(``Cursor.advance``: an in-place add per row, the last launches of such a step).  The cursor is built where it is asked
for and is off until its holder switches it on for the one pass in which a step is recorded: only then do the two
accessors hand out its rows.  Every step launched from Python reads the rows of the tables by the host position, as it
does on a state without a cursor.

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

    def write(self, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Insert the key and value rows of one token per sequence, ``[sequences, kv_heads, head_dim]`` each, at ``position``."""
        self.k[:, :, position] = k
        self.v[:, :, position] = v

    def keys(self, count: int, dtype) -> torch.Tensor:
        """The keys of the first ``count`` positions of every sequence, ``[sequences, kv_heads, count, head_dim]`` in ``dtype``:
        what an attention over those positions reads.  In the cache's own dtype it is a view of the cache."""
        return self.k[:, :, :count].to(dtype)

    def values(self, count: int, dtype) -> torch.Tensor:
        """The values of the first ``count`` positions of every sequence, as :meth:`keys`."""
        return self.v[:, :, :count].to(dtype)


@dataclass
class PageTable:
    """What a kernel that prepares the operands of a paged attention reads beside the pool, for the steps of one number of
    tokens per sequence: constants built with the state, written by no call.

    ``cos_sin`` ``[capacity, rotary width]`` bf16: per position the cosines of its rotary angles, then the sines -- every row
    formed by the reference chain's own function for that position, on the state's device, so that a kernel that reads
    the table and the chain that forms the row per call rotate by the same bytes.  ``positions`` ``[capacity - tokens + 1,
    3, rows]`` int64: row ``p`` is for the step that starts at position ``p`` and holds ``p + t`` for token ``t`` of every
    sequence (the rows of a step stand sequence by sequence), as three equal rows -- a text position is one number, and a
    kernel that takes three position rows reads the same one in each.  ``slots`` ``[capacity - tokens + 1, rows]`` int64:
    ``sequence * pages * page + p + t``, the flat index of that position's row in the pool.  No row stands for a step
    whose last token would lie past the capacity.

    ``blocks`` and ``lengths`` are what a kernel that reads the pool through a page table is handed beside it; a state that
    was given no maximum length has neither (``None``).  ``blocks`` ``[sequences + 1, width]`` int32, the page table:
    row ``i`` names the pages of sequence ``i`` in the order of its positions, ``i * pages + j`` in column ``j``, and every
    column past its own pages names its LAST own page: such a column stands for positions the sequence never holds,
    and names a page of the pool all the same.  The width is the pages of the run's maximum length, which such a
    kernel is told beside the table, or the sequence's own pages where those are more, rounded up to a multiple of
    four (:meth:`width`).  The last row is slack: every entry names page 0, so that an entry behind the last
    sequence's row names a page of the pool as well.  ``lengths`` ``[capacity - tokens + 1, sequences rounded up to four]``
    int32: row ``p`` holds ``p + tokens`` in every column, the positions a sequence holds AFTER the step that starts at
    ``p`` (every sequence of a state stands at the same position).  Rows of four elements or a multiple, so every row of
    either table starts 16 bytes aligned -- the layout of ``TokenSlots.counts``.

    A kernel reads two things here that nothing can check for it on the host: the slots of one step are distinct (two
    rows of a launch never write one pool row), and every position is a row of ``cos_sin``.  Both hold by construction
    while a sequence's pages and the cos / sin table hold its capacity, which :meth:`build` asks of its integers before a
    tensor is made.  And both hold on ONE pool alone: ``sequences``, ``page`` and ``pages`` are the integers the slots
    were formed from, which a cache holds against its pool before it hands out a row (:class:`PagedKvCache`)."""

    cos_sin: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    sequences: int  # of the pool the slots are indices into
    page: int  # the positions of a page of that pool
    pages: int  # ... and the pages a sequence owns in it
    tokens: int = 1
    single: Optional["PageTable"] = None  # beside a table of several tokens per sequence: the table of one, for the decode steps of a prompt
    blocks: Optional[torch.Tensor] = None  # the page table; one tensor for both tokens counts of a state
    lengths: Optional[torch.Tensor] = None  # the positions a sequence holds after a step of ``tokens`` tokens
    #: the rows of ONE position of this table at fixed addresses, for a step that is recorded and replayed
    #: (``DecodeState.step_cursor()``); ``None`` until a state is asked for it.  The caches that read this table hand out
    #: its rows while it is switched on, and the table's own rows otherwise
    cursor: Optional["Cursor"] = None

    @staticmethod
    def rotary(capacity: int, rotary_dim: int, theta: float, device) -> torch.Tensor:
        """The cos / sin table of ``capacity`` positions: row ``p`` is what ``torch_ref.rope_cos_sin(p)`` returns, cosines then
        sines.  Position by position with the chain's own function -- a second formula, or a table made on another
        device, can differ from it in the last bit of a cosine."""
        from .providers.torch_ref import rope_cos_sin  # the provider module imports this one: asked for here, when a state is built

        return torch.stack([torch.cat(rope_cos_sin(position, rotary_dim, theta, device)) for position in range(capacity)])

    @staticmethod
    def width(page: int, pages: int, max_model_len: int) -> int:
        """The entries of one sequence's row of the page table: the pages that hold ``max_model_len`` positions, or the
        sequence's own ``pages`` where those are more, rounded up to a multiple of four."""
        return -(-max(pages, -(-max_model_len // page)) // 4) * 4

    @classmethod
    def build(
        cls, sequences: int, tokens: int, capacity: int, page: int, pages: int, cos_sin: torch.Tensor, device, max_model_len: Optional[int] = None
    ) -> "PageTable":
        """The tables of ``sequences`` sequences at ``tokens`` tokens per sequence, on the cos / sin table ``cos_sin`` (shared
        by the tables of one state); with ``max_model_len``, the run's maximum length, the page table and the lengths too.
        Made from host integers and moved to ``device`` once.  ``ValueError`` where the pages of a sequence or the rows of
        ``cos_sin`` do not hold ``capacity`` positions, or where ``capacity`` passes ``max_model_len``."""
        if capacity > pages * page:  # a position past a sequence's pages would have the slot of the next sequence's row
            raise ValueError(f"{capacity} positions per sequence need {-(-capacity // page)} pages of {page}; got pages={pages!r}")
        if capacity > cos_sin.shape[0]:  # ... and one past the table would rotate by what lies behind it
            raise ValueError(f"{capacity} positions per sequence need a cos / sin table of as many rows; got {cos_sin.shape[0]}")
        if max_model_len is not None and capacity > max_model_len:  # a kernel that is told the maximum would be handed a length past it
            raise ValueError(f"{capacity} positions per sequence pass the run's maximum length; got max_model_len={max_model_len!r}")
        steps = max(capacity - tokens + 1, 0)  # a step starts at 0 .. capacity - tokens
        position = torch.arange(steps, dtype=torch.int64)[:, None, None] + torch.arange(tokens, dtype=torch.int64)
        position = position.expand(steps, sequences, tokens)  # [the position a step starts at, sequence, token]: start + token
        slots = torch.arange(sequences, dtype=torch.int64)[None, :, None] * (pages * page) + position
        rows = sequences * tokens
        positions = position.reshape(steps, 1, rows).expand(steps, 3, rows).contiguous()  # three real rows: no axis of stride zero
        single = None if tokens == 1 else cls.build(sequences, 1, capacity, page, pages, cos_sin, device, max_model_len)
        blocks = lengths = None
        if max_model_len is not None:
            blocks = None if single is None else single.blocks
            if blocks is None:
                column = torch.arange(cls.width(page, pages, max_model_len), dtype=torch.int32).clamp(max=pages - 1)  # past the own pages: the last own page
                own = torch.arange(sequences, dtype=torch.int32)[:, None] * pages + column
                blocks = torch.cat([own, torch.zeros_like(own[:1])]).contiguous().to(device)  # the slack row: page 0
            lengths = (torch.arange(steps, dtype=torch.int32)[:, None] + tokens).expand(steps, -(-sequences // 4) * 4).contiguous().to(device)
        table = cos_sin, positions.to(device), slots.reshape(steps, rows).contiguous().to(device)
        return cls(*table, sequences, page, pages, tokens, single, blocks, lengths)

    def first_token(self) -> "PageTable":
        """The table of one token per sequence on the same cos / sin table: what the decode steps of a prompt read on a state
        of several tokens per sequence; the table itself at one token."""
        return self if self.single is None else self.single

    def tensors(self):
        return ()  # a table: no call writes it

    def bytes(self) -> int:
        """Bytes of the tables of a state: the cos / sin table once, and the positions and slots of each tokens count; where
        the state holds them, the page table once and the lengths of each tokens count."""
        held = [self.cos_sin, self.positions, self.slots, *(() if self.single is None else (self.single.positions, self.single.slots))]
        if self.blocks is not None:
            held += [self.blocks, self.lengths, *(() if self.single is None else (self.single.lengths,))]
        return sum(t.numel() * t.element_size() for t in held)


@dataclass
class Cursor:
    """What a recorded decode step reads at FIXED addresses: the ids it is fed and the rows of the position it stands at.

    ``ids`` ``[sequences]`` int32: the ids the step's lookup reads; the step's own last copy writes the ids it chose into
    it, so the next replay feeds on them -- and on whatever the caller, who is handed this tensor behind a replay, has
    written into it since.  ``positions`` ``[3, rows]`` int64, ``slots`` ``[rows]`` int64 and ``lengths``
    ``[sequences rounded up to four]`` int32 (``None`` on a state without that table): the rows ``PagedKvCache.at`` and
    ``.attended`` hand out for one position, as copies of the table's rows in allocations of their own -- the same
    dtypes, shapes and alignment, so a kernel reads them as it reads a row of the table.  ``position`` is the host's
    own account of where the three rows stand; a holder compares it with the state's position in front of every replay
    and sets the rows where the two differ (a step launched from Python moves the state and never the cursor).

    ``on`` says that the caches of the table hand out these rows in the place of the table's: set by the holder for the
    one pass in which a step is recorded, and for nothing else."""

    table: PageTable
    ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    lengths: Optional[torch.Tensor]
    position: int = 0
    on: bool = False

    @classmethod
    def build(cls, table: PageTable, device) -> "Cursor":
        """The cursor of ``table`` at position 0: the rows of the step that starts there, cloned.  ``ValueError`` for a table
        of several tokens per sequence (a recorded step feeds one) and for a table no step starts on."""
        if table.tokens != 1:
            raise ValueError(f"a recorded step feeds one token per sequence; this table is for steps of {table.tokens}")
        if not int(table.positions.shape[0]):
            raise ValueError("a recorded step starts at a position of its state; this table holds none")
        ids = torch.zeros(table.sequences, dtype=torch.int32, device=device)
        lengths = None if table.lengths is None else table.lengths[0].clone()
        return cls(table, ids, table.positions[0].clone(), table.slots[0].clone(), lengths)

    def set(self, position: int) -> None:
        """Stand at ``position``: its rows of the tables copied into the cursor's rows (a copy per row), and the host's
        account with them.  ``IndexError`` for a position no step starts at."""
        steps = int(self.table.positions.shape[0])
        if not 0 <= position < steps:
            raise IndexError(f"position {position} is not one a step starts at: the tables hold {steps} such positions, from 0")
        self.positions.copy_(self.table.positions[position])
        self.slots.copy_(self.table.slots[position])
        if self.lengths is not None:
            self.lengths.copy_(self.table.lengths[position])
        self.position = position

    def advance(self) -> None:
        """One position on, in place: an add per row -- the launches that end a recorded step (``launches``) -- and the
        host's account with them.  A holder that records the adds puts the account back, and moves it once per replay
        (``replayed``)."""
        self.positions.add_(1)
        self.slots.add_(1)
        if self.lengths is not None:
            self.lengths.add_(1)
        self.position += 1

    def replayed(self) -> None:
        """A replay of a recorded step has run the adds of ``advance``: the host's account follows."""
        self.position += 1

    def launches(self) -> int:
        """The launches of ``advance``: one per row the cursor holds."""
        return 2 + (self.lengths is not None)

    def holds(self) -> bool:
        """Whether the rows on the device are those of ``position``, by the arithmetic of the tables written out here
        from host integers (never from a row of the table: a table one row late agrees with itself): every position
        row holds ``position``, row ``i`` of the slots ``i x pages x page + position``, every length ``position + 1``.
        A host read: for a check behind a request, never inside a step."""
        table, at = self.table, self.position
        slots = [sequence * table.pages * table.page + at for sequence in range(table.sequences)]
        held = self.positions.tolist() == [[at] * table.sequences] * 3 and self.slots.tolist() == slots
        return held and (self.lengths is None or self.lengths.tolist() == [at + 1] * int(self.lengths.shape[0]))


@dataclass
class PagedKvCache:
    """Keys and values of one attention layer as ONE pool of pages, ``[sequences * pages, kv_heads, page, 2 * head_dim]`` bf16:
    keys in the first half of the last axis, values in the second; sequence ``i`` owns the pages ``i * pages .. (i + 1) *
    pages - 1``, and its position ``p`` is row ``p % page`` of its page ``p // page``.  ``tokens`` as for :class:`KvCache`.

    ``first`` / ``sequences`` are the sequences this object stands for -- every sequence of the pool, or the window a check
    cuts out to run some sequences alone (:meth:`rows`).  The accessors read and write that window; the pool, and with it
    ``k_pool`` / ``v_pool``, is always the whole pool, because a slot is an index into the whole pool.

    ``table`` is the table of THIS pool and of this object's tokens per sequence: one built for other sequences, pages
    or tokens would hand a kernel slots past the pool's end or inside another sequence's pages, so it is a
    ``ValueError`` here, as is a window that is not of the pool's sequences.  The accessors are indexing, as the plain
    cache's are, and take the positions a session's steps name, which a session holds inside its capacity: a row of a
    sequence's last page past the capacity is a row of the pool like any other, and ``write`` does not refuse it."""

    pool: torch.Tensor
    pages: int  # per sequence
    table: PageTable
    tokens: int = 1
    first: int = 0
    sequences: Optional[int] = None

    def __post_init__(self):
        table, held, page = self.table, int(self.pool.shape[0]), int(self.pool.shape[2])
        if (table.sequences * table.pages, table.pages, table.page, table.tokens) != (held, self.pages, page, self.tokens):
            raise ValueError(
                f"the tables of a paged cache are built for its pool: these are for a pool of {table.sequences * table.pages} pages ({table.pages} a sequence) "
                f"of {table.page} positions, tokens per sequence {table.tokens}; the cache has a pool of {held} pages ({self.pages} a sequence) of {page} "
                f"positions, tokens per sequence {self.tokens}"
            )
        if self.sequences is None:
            self.sequences = table.sequences - self.first
        if not 0 <= self.first <= self.first + self.sequences <= table.sequences:
            raise ValueError(
                f"a paged cache stands for sequences of its pool, which holds {table.sequences}; got first={self.first}, sequences={self.sequences}"
            )

    @property
    def page(self) -> int:
        """The positions of one page."""
        return int(self.pool.shape[2])

    @property
    def k_pool(self) -> torch.Tensor:
        """The key half of the WHOLE pool, ``[sequences * pages, kv_heads, page, head_dim]``: a view, as a kernel takes it."""
        return self.pool[..., : self.pool.shape[-1] // 2]

    @property
    def v_pool(self) -> torch.Tensor:
        """The value half of the whole pool, as :attr:`k_pool`."""
        return self.pool[..., self.pool.shape[-1] // 2 :]

    def _paged(self) -> torch.Tensor:
        """The pages of this object's sequences as ``[sequences, pages, kv_heads, page, 2 * head_dim]``: a view of the pool."""
        return self.pool.view(-1, self.pages, *self.pool.shape[1:])[self.first : self.first + self.sequences]

    def tensors(self):
        return (self.pool[self.first * self.pages : (self.first + self.sequences) * self.pages],)

    def rows(self, start: int, stop: int) -> "PagedKvCache":
        """The cache of the sequences that own the rows ``start`` to ``stop`` of a step (``tokens`` rows each): the same pool,
        the same tables, and those sequences as the window."""
        first, last = start // self.tokens, stop // self.tokens
        return PagedKvCache(self.pool, self.pages, self.table, self.tokens, self.first + first, last - first)

    def of(self, sequence: int) -> tuple:
        """Where ``sequence`` lives in each tensor of ``tensors()``: its pages."""
        return (slice(sequence * self.pages, (sequence + 1) * self.pages),)

    def first_token(self) -> "PagedKvCache":
        """The same pool as a decode step takes it: one position per sequence and step, with the table of one token."""
        return PagedKvCache(self.pool, self.pages, self.table.first_token(), 1, self.first, self.sequences)

    def write(self, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Insert the key and value rows of one token per sequence, ``[sequences, kv_heads, head_dim]`` each, at ``position``:
        two views by host integers and two copies, as on the plain cache."""
        page, row = divmod(position, self.page)
        width = self.pool.shape[-1] // 2
        rows = self._paged()[:, page, :, row]  # [sequences, kv_heads, 2 * head_dim]: the rows of this position
        rows[..., :width] = k
        rows[..., width:] = v

    def _read(self, half: int, count: int, dtype) -> torch.Tensor:
        """One half of the first ``count`` positions of every sequence: the pages that hold them as one view of the pool,
        the page axis moved behind the heads, ONE cast into a contiguous tensor (a copy, in the pool's own dtype), read
        as a run of positions and cut at ``count``."""
        pages = -(-count // self.page)
        width = self.pool.shape[-1] // 2
        held = self._paged()[:, :pages, :, :, half * width : (half + 1) * width].transpose(1, 2)  # [sequences, kv_heads, pages, page, head_dim]
        whole = held.clone(memory_format=torch.contiguous_format) if held.dtype == dtype else held.to(dtype, memory_format=torch.contiguous_format)
        return whole.view(self.sequences, self.pool.shape[1], pages * self.page, width)[:, :, :count]

    def keys(self, count: int, dtype) -> torch.Tensor:
        """The keys of the first ``count`` positions of every sequence, ``[sequences, kv_heads, count, head_dim]`` in ``dtype``:
        the bytes :meth:`KvCache.keys` returns after the same writes.  Always a copy: the pages of a sequence are not one
        run of positions in the pool."""
        return self._read(0, count, dtype)

    def values(self, count: int, dtype) -> torch.Tensor:
        """The values of the first ``count`` positions of every sequence, as :meth:`keys`."""
        return self._read(1, count, dtype)

    def at(self, position: int) -> tuple:
        """``(cos_sin, positions, slots)`` for the step that starts at ``position``: the whole cos / sin table, and the rows of
        the position and slot tables for this object's sequences -- ``[3, rows]`` and ``[rows]`` int64, views inside the
        tables (the token axis contiguous, each aligned to its element).  What an adapter to a kernel hands on; the
        reference chains never ask.  ``IndexError`` for a position no step of this object's tokens starts at: one past
        the last row, and a negative one, which would index the tables from their end."""
        self._starts(position)
        first, last = self.first * self.tokens, (self.first + self.sequences) * self.tokens
        cursor = self._recording(position)
        if cursor is not None:  # a step that is being recorded: the rows at fixed addresses, which its own last launches move on
            return self.table.cos_sin, cursor.positions[:, first:last], cursor.slots[first:last]
        return self.table.cos_sin, self.table.positions[position][:, first:last], self.table.slots[position][first:last]

    def _recording(self, position: int) -> Optional[Cursor]:
        """The table's cursor while it is switched on -- the pass in which a step is recorded -- else ``None``.  The
        cursor stands where the pass is made: a ``ValueError`` where the step asks for another position than the one
        its rows hold, which a replay would then read in its place."""
        cursor = self.table.cursor
        if cursor is None or not cursor.on:
            return None
        if cursor.position != position:
            raise ValueError(
                f"a step is recorded where its cursor stands: the cursor holds the rows of position {cursor.position}, the step asks for {position}"
            )
        return cursor

    def _starts(self, position: int) -> None:
        """``IndexError`` for a position no step of this object's tokens starts at."""
        steps = int(self.table.positions.shape[0])
        if not 0 <= position < steps:
            raise IndexError(
                f"position {position} is not one a step of {self.tokens} token{'s' if self.tokens > 1 else ''} per sequence starts at: "
                f"the tables hold {steps} such positions, from 0"
            )

    def attended(self, position: int) -> tuple:
        """``(blocks, lengths)`` for the step that starts at ``position``: the page-table rows of this object's sequences,
        ``[sequences, width]`` int32, and the positions each of them holds after the step, ``[sequences]`` int32 -- views
        inside the state's tables, contiguous, each starting 16 bytes aligned, for every sequence of the pool and for a
        window of them; behind a window's last row stands the next sequence's row or the slack row.  What an adapter to a
        kernel that reads the pool through a page table hands on beside ``k_pool`` / ``v_pool``; the reference chains never
        ask.  ``IndexError`` as :meth:`at`; ``ValueError`` on a state that was built without the two tables."""
        self._starts(position)
        if self.table.blocks is None:
            raise ValueError(
                "this attention cache has no page table and no lengths: its state was given no maximum length (DecodeState(max_model_len=)), "
                "which a session hands the state where a block of its plans reads them"
            )
        cursor = self._recording(position)
        if cursor is not None:  # as in ``at``: the lengths at a fixed address
            return self.table.blocks[self.first : self.first + self.sequences], cursor.lengths[: self.sequences]
        return self.table.blocks[self.first : self.first + self.sequences], self.table.lengths[position, : self.sequences]


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
    sequence a step of the run feeds (1: a decode step; more: a verify step, whose table is ``slots``).  ``page``: the
    layout of the attention caches -- ``None``, the plain cache; a number of positions, a pool of pages of that many
    (:class:`PagedKvCache`) with the state's position and slot tables (``table``).  ``max_model_len``: the run's maximum
    length, for a paged state whose pool a kernel reads through a page table: the state then holds the page table and
    the lengths as well (``PageTable.blocks`` / ``lengths``)."""

    def __init__(
        self,
        layer_kinds: Sequence[str],
        sequences: int,
        capacity: int,
        shapes: Dict[str, tuple],
        device,
        tokens: int = 1,
        page: Optional[int] = None,
        pages: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ):
        """``shapes``: ``"state"`` -> (value_heads, head_dim, head_dim), ``"conv"`` -> (taps, channels), ``"kv"`` -> (kv_heads,
        head_dim) and, read for a paged cache alone, ``"rotary"`` -> (rotary width, theta).  ``pages``: the pages a
        sequence owns in a paged cache; left out, as many as hold the capacity, ``ceil(capacity / page)``.
        ``max_model_len``: the positions a sequence of the run may reach at most, a whole number that holds the capacity;
        left out, the state has no page table and no lengths."""
        if sequences < 1 or capacity < 1 or tokens < 1:
            raise ValueError(
                f"a decode state needs at least one sequence, one position and one token per sequence; got sequences={sequences}, capacity={capacity}, tokens={tokens}"
            )
        if page is None and pages is not None:
            raise ValueError(f"pages= says how many pages a sequence of a PAGED cache owns: it comes with page=, the positions of a page; got pages={pages!r}")
        if max_model_len is not None:
            if page is None:
                raise ValueError(
                    f"max_model_len= builds the page table and the lengths of a PAGED cache: it comes with page=; got max_model_len={max_model_len!r}"
                )
            if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
                raise ValueError(f"the maximum length of a run is a whole number of positions from 1; got max_model_len={max_model_len!r}")
        if page is not None:
            if any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in (page, 1 if pages is None else pages)):
                raise ValueError(
                    f"the page of an attention cache is a whole number of positions from 1, and a sequence owns a whole number of pages; got page={page!r}, pages={pages!r}"
                )
            pages = -(-int(capacity) // page) if pages is None else pages
        self.sequences, self.capacity, self.position, self.tokens = int(sequences), int(capacity), 0, int(tokens)
        self.device = torch.device(device)
        self.page, self.pages = page, pages
        #: of a paged cache: the cos / sin table and the position and slot tables of this state's steps (and, at several
        #: tokens per sequence, of the decode steps of its prompt); built once, for every attention layer
        self.table: Optional[PageTable] = None
        if page is not None and any(kind != "linear_attention" for kind in layer_kinds):
            width, theta = shapes["rotary"]
            cos_sin = PageTable.rotary(self.capacity, width, theta, self.device)
            self.table = PageTable.build(self.sequences, self.tokens, self.capacity, page, pages, cos_sin, self.device, max_model_len)
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
            elif self.table is not None:  # one pool of pages: keys in the first half of the last axis, values in the second
                heads, width = shapes["kv"]
                pool = torch.zeros((sequences * pages, heads, page, 2 * width), dtype=torch.bfloat16, device=self.device)
                self.layers.append(PagedKvCache(pool, pages, self.table, self.tokens))
            else:
                kv = (sequences, shapes["kv"][0], capacity, shapes["kv"][1])
                self.layers.append(
                    KvCache(torch.zeros(kv, dtype=torch.bfloat16, device=self.device), torch.zeros(kv, dtype=torch.bfloat16, device=self.device), self.tokens)
                )
        #: the layers as a decode step takes them: the layers themselves at one token per sequence, else their token-0 views
        self.decode_layers: List[object] = self.layers if self.tokens == 1 else [layer.first_token() for layer in self.layers]
        #: the steps of several tokens per sequence this state has run: a decode step comes in front of the first of them
        self.verify_steps = 0
        #: of the last step: its final-norm rows ``[rows, hidden]`` bf16, by reference, and the ids it chose ``[rows]`` int32
        self.hidden: Optional[torch.Tensor] = None
        self.greedy: Optional[torch.Tensor] = None
        #: the drafter's own attention cache (``KvCache``), built by a session that has a drafter; it has no position
        self.draft: Optional[KvCache] = None

    def step_cursor(self) -> Cursor:
        """The cursor of this state (:class:`Cursor`): built at the first call, at position 0, and the same object from
        then on -- what is recorded on a state reads the addresses of that state's cursor.  ``ValueError`` on a state
        whose steps feed several tokens per sequence, and on one without the tables of a paged attention cache: the
        rows a cursor holds are rows of those tables."""
        if self.tokens != 1:
            raise ValueError(f"a recorded step feeds one token per sequence; a step of this state feeds {self.tokens}")
        if self.table is None:
            raise ValueError("a recorded step reads the position rows of a paged attention cache; this state has no such tables (DecodeState(page=))")
        if self.table.cursor is None:
            self.table.cursor = Cursor.build(self.table, self.device)
        return self.table.cursor

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
        cached keys and values up to the position of an attention layer (``[sequences, kv_heads, position, head_dim]``
        each: views of a plain cache, copies out of a paged one).  Two runs of the same tokens are compared on it,
        whatever tokens per sequence each fed and whatever layout each cache has."""
        kept = self.accepted
        slots = (self.slot_ids + kept * self.sequences).long()
        standing = []
        for layer in self.layers:
            if isinstance(layer, RecurrentSlots):
                taps = layer.conv_storage.shape[1] - self.tokens + 1
                standing.append((layer.state[slots], layer.conv_storage[1:, kept : kept + taps]))
            else:
                standing.append((layer.keys(self.position, torch.bfloat16), layer.values(self.position, torch.bfloat16)))
        return standing

    def bytes(self) -> int:
        """Bytes of every state tensor and, with a paged cache, of its tables."""
        held = sum(t.numel() * t.element_size() for layer in self.layers for t in layer.tensors())
        return held + (0 if self.table is None else self.table.bytes())

    def clear(self) -> None:
        """Back to position 0 of a new prompt set in the existing allocations: every state tensor zero, the drafter's cache
        too, no token kept, no verify step run, and nothing left of a last step (``hidden``, ``greedy``)."""
        self.position = self.verify_steps = 0
        self.hidden = self.greedy = None
        if self.slots is not None:
            self.slots.accept = 0
        for holder in (*self.layers, *(() if self.draft is None else (self.draft,))):
            for t in holder.tensors():
                t.zero_()

    def snapshot(self) -> dict:
        """A copy of every state tensor, the position, the tokens kept and the verify steps run; and of what a drafter reads
        of the last step: its final-norm rows, its greedy ids and the drafter's cache (``None`` where there is none)."""

        def copied(tensor):
            return None if tensor is None else tensor.clone()

        return {
            "position": self.position,
            "accepted": self.accepted,
            "verify_steps": self.verify_steps,
            "layers": [tuple(t.clone() for t in layer.tensors()) for layer in self.layers],
            "hidden": copied(self.hidden),
            "greedy": copied(self.greedy),
            "draft": None if self.draft is None else tuple(t.clone() for t in self.draft.tensors()),
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
        self.hidden, self.greedy = snap.get("hidden"), snap.get("greedy")  # the snapshot's own copies: no step writes them again
        if self.draft is not None and snap.get("draft") is not None:
            for t, s in zip(self.draft.tensors(), snap["draft"]):
                t.copy_(s)
