# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text in, token ids; token ids, streamed text out.

``TextIO.synthetic`` is a pure-Python byte-level tokenizer for the seeded default run of a sample: ids 0..255 are the bytes of
UTF-8 text, the ids above them are seeded lower-case pieces that only the model can produce, and the last id is the end token.
It imports nothing beyond the standard library and PyTorch.

``TextIO.from_checkpoint(directory)`` reads the text side of a checkpoint on local disk: its ``tokenizer.json`` through the
``tokenizers`` library, its chat template (``chat_template.jinja``, or the ``chat_template`` entry of
``tokenizer_config.json``) compiled with ``jinja2``, and the end-token ids of its ``generation_config.json``.  The two
packages are imported here and nowhere else in the helper layer.  The template environment is set up as the reference
library of these checkpoints sets up its own: an immutable sandbox with ``trim_blocks`` and ``lstrip_blocks``, the
loop-controls extension, a ``tojson`` filter that does not escape HTML, ``raise_exception`` and ``strftime_now``.  Chat
messages, tool schemas and tool results are all text by the time they are encoded: the template renders them, the
tokenizer encodes the rendered prompt as it is (the template writes the special tokens itself).  Text comes back through
the library's incremental decoder, one per sequence, with special tokens left out.
"""

from __future__ import annotations

import codecs
import json
import os
import random
from typing import Optional, Sequence

import torch

REQUIREMENTS = "samples/models/requirements.txt"


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


class LibraryStream:
    """Incremental decoder of one sequence over a ``tokenizers`` tokenizer: the same ``push`` / ``close`` as :class:`TextStream`."""

    def __init__(self, tokenizer):
        from tokenizers.decoders import DecodeStream

        self._tokenizer = tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)

    def push(self, token_id: int) -> str:
        return self._stream.step(self._tokenizer, int(token_id)) or ""

    def close(self) -> str:
        return ""


def load_messages(path: str) -> tuple:
    """A chat request from a JSON file: ``(messages, tools, template options)``.  The file holds a list of messages, or an
    object with ``messages`` and optionally ``tools`` (JSON tool schemas) and ``chat_template_kwargs`` (options the
    checkpoint's template understands).  A file of another shape is a ``ValueError``."""
    with open(path, encoding="utf-8") as f:
        body = json.load(f)
    if isinstance(body, list):
        body = {"messages": body}
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
        raise ValueError(f'--messages {os.path.basename(path)}: needs a list of messages, or an object with "messages"; every message has a "role"')
    tools, options = body.get("tools"), body.get("chat_template_kwargs") or {}
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f'--messages {os.path.basename(path)}: "tools" is a list of JSON tool schemas')
    if not isinstance(options, dict):
        raise ValueError(f'--messages {os.path.basename(path)}: "chat_template_kwargs" is an object')
    return messages, tools, options


class TextIO:
    """The text side of a model sample: chat messages -> token ids, token ids -> text, the end-token ids."""

    def __init__(self, pieces: Sequence[bytes], eos_ids: Sequence[int], name: str):
        self._pieces = tuple(pieces)
        self.eos_ids = tuple(int(i) for i in eos_ids)
        self.name = name
        self.synthetic_text = True
        self._tokenizer = None  # a tokenizers.Tokenizer for a checkpoint
        self._template = None  # its compiled chat template
        self._template_tokens: dict = {}  # the special-token strings a template may name (bos_token, eos_token, ...)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size(with_added_tokens=True)
        return len(self._pieces)

    @classmethod
    def from_checkpoint(cls, directory: str) -> "TextIO":
        """The tokenizer, the chat template and the end tokens of a checkpoint directory on local disk.  No weights are read.

        Raises ``NotImplementedError`` (a SKIP that names the fix) when ``tokenizers`` or ``jinja2`` is not installed, and
        ``ValueError`` when the directory lacks a file this needs."""
        try:
            import jinja2
            import jinja2.ext
            import jinja2.sandbox
            import tokenizers
        except ImportError as error:
            raise NotImplementedError(
                f"a checkpoint's text is read with the tokenizers and jinja2 packages ({error}): pip install -r {REQUIREMENTS}"
            ) from error
        shown = os.path.basename(os.path.normpath(directory)) or "."

        def read(name, required=True):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                if required:
                    raise ValueError(f"--model {shown}: no {name} there")
                return None
            with open(path, encoding="utf-8") as f:
                return f.read()

        if read("tokenizer.json") is None:  # read once for the error text; the library reads the file itself
            raise ValueError(f"--model {shown}: no tokenizer.json there")
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        settings = json.loads(read("tokenizer_config.json", required=False) or "{}")
        source = read("chat_template.jinja", required=False) or settings.get("chat_template")
        if not isinstance(source, str):
            raise ValueError(f"--model {shown}: no chat template there (chat_template.jinja, or chat_template in tokenizer_config.json)")
        generation = json.loads(read("generation_config.json", required=False) or "{}")
        eos = generation.get("eos_token_id")
        if eos is None:
            config = json.loads(read("config.json", required=False) or "{}")
            eos = config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id"))
        if eos is None:
            raise ValueError(f"--model {shown}: no end-token id there (eos_token_id in generation_config.json or config.json)")

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def strftime_now(fmt):
            import datetime

            return datetime.datetime.now().strftime(fmt)

        environment = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        environment.filters["tojson"] = tojson
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = strftime_now
        self = cls(
            (),
            [eos] if isinstance(eos, int) else eos,
            f"tokenizer and chat template of checkpoint {shown} (tokenizers {tokenizers.__version__}, jinja2 {jinja2.__version__})",
        )
        self.synthetic_text, self._tokenizer, self._template = False, tokenizer, environment.from_string(source)
        for key, value in settings.items():  # the special tokens a template may refer to by name
            if key.endswith("_token") and isinstance(value, dict):
                value = value.get("content")
            if key.endswith("_token") and isinstance(value, str):
                self._template_tokens[key] = value
        return self

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

    def render_chat(self, messages: Sequence[dict], tools: Optional[Sequence[dict]] = None, add_generation_prompt: bool = True, **options) -> str:
        """The prompt text of chat messages.  Tool schemas are text: a checkpoint's template renders them where it wants them;
        the synthetic tokenizer renders them as one JSON block ahead of the turns.  Without the generation prompt the last
        turn is left open, to be continued.  ``options`` are handed to a checkpoint's template (what it understands is the
        template's own business); the synthetic tokenizer takes none."""
        if self._template is not None:
            context = {**self._template_tokens, **options}
            try:
                return self._template.render(
                    messages=list(messages), tools=list(tools) if tools else None, add_generation_prompt=add_generation_prompt, **context
                )
            except Exception as error:  # the template's own refusal (raise_exception) or a construct it cannot render
                if type(error).__module__.partition(".")[0] != "jinja2":
                    raise
                raise ValueError(f"the checkpoint's chat template refused the messages: {error}") from error
        if options:
            raise ValueError(f"the synthetic tokenizer's template takes no options; got {sorted(options)}")
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
        """Token ids of text: int32 on the CPU (ids go to the GPU as 32-bit integers).  A checkpoint's tokenizer encodes the
        text as it is: a rendered prompt already carries its special tokens."""
        if self._tokenizer is not None:
            return torch.tensor(self._tokenizer.encode(text, add_special_tokens=False).ids, dtype=torch.int32)
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.int32)

    def encode_chat(self, messages: Sequence[dict], tools: Optional[Sequence[dict]] = None, add_generation_prompt: bool = True, **options) -> torch.Tensor:
        return self.encode(self.render_chat(messages, tools, add_generation_prompt, **options))

    def decode(self, ids) -> str:
        """The text of a whole id sequence at once (what the streamed pieces of :meth:`decoder` add up to)."""
        if self._tokenizer is not None:
            return self._tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)
        return b"".join(self._pieces[int(i)] for i in ids).decode("utf-8", errors="replace")

    def decoder(self, prompt_ids=None):
        """A fresh stream decoder for one sequence; the prompt is not echoed."""
        if self._tokenizer is not None:
            return LibraryStream(self._tokenizer)
        return TextStream(self._pieces)
