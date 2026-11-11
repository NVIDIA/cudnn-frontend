# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text in, token ids; token ids, streamed text out.

``TextIO.synthetic`` is a pure-Python byte-level tokenizer for the seeded default run of a sample: ids 0..255 are the bytes of
UTF-8 text, the ids above them are seeded lower-case pieces that only the model can produce, and the last id is the end token.
It imports nothing beyond the standard library and PyTorch.  A tokenizer and a chat template read from a real checkpoint arrive
with the next pull request of this area.
"""

from __future__ import annotations

import codecs
import json
import random
from typing import Optional, Sequence

import torch


class TextStream:
    """Incremental decoder of one sequence: ``push(token_id)`` returns the text that became complete with this token ("" while a
    multi-byte character is still open)."""

    def __init__(self, pieces: Sequence[bytes]):
        self._pieces = pieces
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        return self._decoder.decode(self._pieces[int(token_id)])

    def close(self) -> str:
        return self._decoder.decode(b"", final=True)


class TextIO:
    """The text side of a model sample: chat messages -> token ids, token ids -> text, the end-token ids."""

    def __init__(self, pieces: Sequence[bytes], eos_ids: Sequence[int], name: str):
        self._pieces = tuple(pieces)
        self.eos_ids = tuple(int(i) for i in eos_ids)
        self.name = name

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    @classmethod
    def synthetic(cls, vocab_size: int, seed: int = 0) -> "TextIO":
        if vocab_size < 258:
            raise ValueError(f"the synthetic tokenizer needs a vocabulary of at least 258 ids (256 bytes, one piece, the end token); got {vocab_size}")
        rng = random.Random(seed)
        letters = "abcdefghijklmnopqrstuvwxyz"
        pieces = [bytes([b]) for b in range(256)]
        for _ in range(256, vocab_size - 1):
            word = "".join(rng.choice(letters) for _ in range(rng.randint(2, 5)))
            pieces.append((" " + word if rng.random() < 0.5 else word).encode())
        pieces.append(b"<|end|>")
        return cls(pieces, (vocab_size - 1,), f"synthetic byte-level tokenizer, {vocab_size} ids, seed {seed}")

    def render_chat(self, messages: Sequence[dict], tools: Optional[Sequence[dict]] = None, add_generation_prompt: bool = True) -> str:
        """The prompt text of chat messages.  Tool schemas are text: they are rendered as one JSON block ahead of the turns.
        Without the generation prompt the last turn is left open, to be continued."""
        parts = []
        if tools:
            parts.append("<tools>\n" + json.dumps(list(tools), sort_keys=True) + "\n</tools>\n")
        for message in messages:
            parts.append(f"<{message['role']}>\n{message['content']}\n")
        if add_generation_prompt:
            parts.append("<assistant>\n")
        elif parts and messages:
            parts[-1] = parts[-1][:-1]
        return "".join(parts)

    def encode(self, text: str) -> torch.Tensor:
        """Token ids of plain text: int32 on the CPU (ids go to the GPU as 32-bit integers)."""
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.int32)

    def encode_chat(self, messages: Sequence[dict], tools: Optional[Sequence[dict]] = None, add_generation_prompt: bool = True) -> torch.Tensor:
        return self.encode(self.render_chat(messages, tools, add_generation_prompt))

    def decode(self, ids) -> str:
        """The text of a whole id sequence at once (what the streamed pieces of :meth:`decoder` add up to)."""
        return b"".join(self._pieces[int(i)] for i in ids).decode("utf-8", errors="replace")

    def decoder(self, prompt_ids=None) -> TextStream:
        """A fresh stream decoder for one sequence; the prompt is not echoed."""
        return TextStream(self._pieces)
