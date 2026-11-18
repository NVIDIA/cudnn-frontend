# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a user of a serving engine brings to the samples of this family, in one module beside the scripts.

A sample is one process with greedy token choice, not a serving engine.  Its user often comes from one, with the request
body a chat client sends and with the option names of a serving command line.  This module reads that body as it is and
answers every field of it:

* **read**: ``messages``, ``tools``, ``chat_template_kwargs``, and the limit of new tokens (``max_tokens`` or its newer name
  ``max_completion_tokens``);
* **stated**: the last part of ``model`` is printed and compared with the name of the checkpoint directory;
* **not used** (``UNUSED``): a field that does not change the text here, ``seed`` and ``stream`` among them, named with
  the reason;
* **neutral** (``NEUTRAL``): a field that would change the output is accepted at the value at which it changes nothing --
  ``temperature`` 0, ``n`` 1, ``top_p`` 1, no stop strings -- and at ``null``;
* **refused**: the same field at another value is a ``ValueError`` that names the field, its value and why this sample
  does not serve it.  Nothing is decoded differently from what a field of the table asks for without an error;
* a field this sample does not know is printed by its name and ignored: it cannot say whether that field would change
  the output.  So the table is where a field goes that asks for something a sample leaves out -- sampling, stop strings,
  constrained output, a processor of the logits, parsing of the output -- under whatever name a client sends it.

What a run prints of a body is the names of its fields and the values this module knows: a number, a boolean, a value of
the table.  It never prints back what somebody else wrote into the body -- a user, metadata -- and of ``model`` it prints
the last part, which is what it compares: a serving engine that serves a directory is asked for it by its path.

It also holds what a run says to that user before anything is loaded -- the decoding rule beside the checkpoint's own
generation settings (``greedy_line``), how many decode steps a prompt takes before the first new token
(``first_token_line``), the limit of new tokens of a request against the context length (``new_tokens``) -- and where the
generated text goes (``TextSink``: standard output, or a file; as it is generated, or once).

A request body never enters the helper layer (``model_samples``): that layer is handed messages, tool schemas and template
options.  Field names are those of the public chat-completions request and of the sampling parameters serving engines add
to it.  This module imports the standard library only; the helper layer is imported where a request is read, after the
sample has imported ``cudnn`` and PyTorch in their order.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: new tokens of a request that names no limit
MAX_TOKENS = 256
#: the default of ``--max-model-len``: the context length the family's census of the target kernel set was recorded with
MAX_MODEL_LEN = 2051

#: the two names of a request's limit of new tokens
LIMITS = ("max_tokens", "max_completion_tokens")
#: what a request body is read for
READ = ("messages", "tools", "chat_template_kwargs", *LIMITS)
#: fields that do not change the text a run of this sample produces, each with the reason a run prints
UNUSED = {
    "seed": "token choice is greedy, nothing is drawn",
    "stream": "the text goes where the command line says, as it is generated unless --no-stream",
    "stream_options": "one process: there is no response to shape",
    "user": "one process: there is no account",
    "metadata": "one process: nothing is stored",
    "store": "one process: nothing is stored",
    "service_tier": "one process: there is no queue",
    "parallel_tool_calls": "the output is text: tool calls in it are not parsed",
}
#: the fields of ``UNUSED`` whose value a run prints back: a number and a boolean.  The others are named, never quoted
_VALUE_SHOWN = ("seed", "stream")
_SAMPLING = "it shapes the distribution a token is drawn from, and nothing is drawn here: token choice is greedy"
_BEAM = "there is no beam search: one sequence per prompt, greedy"
_PENALTY = "no penalty is applied to the logits"
_LOGITS = "nothing processes the logits: the token of the largest one is taken, over the whole vocabulary"
_STOP = "stop strings are not served: a sequence ends at an end token of the checkpoint or at its limit of new tokens"
_LOGPROBS = "a run writes text and, with --dump-tokens, token ids; no log-probabilities"
_REASONING = "reasoning a model writes before its answer is part of the text"
_SPECIAL = "special tokens are left out of the text"
_CONSTRAINED = "constrained output is not served: the text is what the greedy tokens decode to"
_TOOLS = "tool schemas are rendered into the prompt by the checkpoint's chat template; the output is neither constrained nor parsed"
_TEMPLATE = "the prompt ends where the checkpoint's chat template opens the reply"
#: Fields that would change the output, each with the values at which it changes nothing and the reason this sample does
#: not serve another value.  ``null`` is accepted for every field: it is the field left out.
NEUTRAL = {
    "temperature": ((0,), "token choice is greedy: the token of the largest logit at every step"),
    "top_p": ((1,), _SAMPLING),
    "top_k": ((0, -1), _SAMPLING),
    "min_p": ((0,), _SAMPLING),
    "n": ((1,), "a request gets one text; --sequences N decodes the same prompt in N sequences"),
    "best_of": ((1,), "a request gets one text: nothing more is generated to choose among"),
    "use_beam_search": ((False,), _BEAM),
    "length_penalty": ((1,), _BEAM),
    "frequency_penalty": ((0,), _PENALTY),
    "presence_penalty": ((0,), _PENALTY),
    "repetition_penalty": ((1,), _PENALTY),
    "logit_bias": (({},), "no bias is applied to the logits"),
    "logits_processors": (([],), _LOGITS),
    "bad_words": (([],), _LOGITS),
    "allowed_token_ids": ((), _LOGITS),
    "watermarking": ((False,), _LOGITS),
    "stop": (([],), _STOP),
    "include_stop_str_in_output": ((False,), _STOP),
    "stop_token_ids": (([],), "a sequence ends at the end tokens of the checkpoint and at no other"),
    "ignore_eos": ((False,), "an end token ends a sequence; --force-tokens N decodes exactly N tokens instead"),
    "min_tokens": ((0,), "an end token ends a sequence whenever it comes"),
    "repetition_detection": ((), "a sequence ends at an end token of the checkpoint or at its limit of new tokens, whatever it repeats"),
    "thinking_token_budget": ((), f"{_REASONING}, and no budget is put on it"),
    "include_reasoning": ((True,), f"{_REASONING}: the output is not parsed"),
    "logprobs": ((False,), _LOGPROBS),
    "top_logprobs": ((0,), _LOGPROBS),
    "prompt_logprobs": ((), _LOGPROBS),
    "logprob_token_ids": ((), _LOGPROBS),
    "echo": ((False,), "the prompt is not written back with the text"),
    "skip_special_tokens": ((True,), _SPECIAL),
    "spaces_between_special_tokens": ((True,), _SPECIAL),
    "response_format": (({"type": "text"},), _CONSTRAINED),
    "structured_outputs": ((), _CONSTRAINED),
    "structural_tag": ((), _CONSTRAINED),
    "guided_json": ((), _CONSTRAINED),
    "guided_regex": ((), _CONSTRAINED),
    "guided_choice": ((), _CONSTRAINED),
    "guided_grammar": ((), _CONSTRAINED),
    "tool_choice": (("auto",), _TOOLS),
    "function_call": ((), "it is the older name of tool_choice, and the output is neither constrained nor parsed"),
    "functions": ((), "it is the older name of tools, the field tool schemas are read from"),
    "add_generation_prompt": ((True,), _TEMPLATE),
    "continue_final_message": ((False,), _TEMPLATE),
    "add_special_tokens": ((False,), "the prompt is what the checkpoint's chat template renders, encoded as it is: no special token is added"),
    "chat_template": ((), "the messages are rendered by the checkpoint's own chat template"),
    "documents": (([],), "the checkpoint's chat template is given messages and tool schemas, nothing else"),
    "truncate_prompt_tokens": ((), "a prompt is encoded whole"),
    "reasoning_effort": ((), "an option of the checkpoint's chat template goes into chat_template_kwargs"),
    "extra_body": (
        ({},),
        "it is how a client library is handed the fields it adds to a request, not a field of the body it sends: its fields go to the top level of the body",
    ),
}


def _text_io():
    """The text module of the helper layer, imported where a request is read and not with this module: a sample imports
    ``cudnn`` before PyTorch, and the helper layer imports PyTorch."""
    from model_samples import text_io

    return text_io


def _show(value, width: int = 60) -> str:
    """A JSON value as a request writes it, cut where it is long."""
    text = json.dumps(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _name(name: str) -> str:
    """The name of a field as a run prints it: as it stands where it is a plain word, as a JSON string where it is not.  A
    name is whatever the body's author wrote: one with a line break in it would put a line of its own into the run."""
    return name if re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", name) else _show(name)


def _last_part(model: str) -> tuple:
    """The last part of a model name -- what is compared with the checkpoint directory, and all a run prints of it -- and
    how a run shows it: with ``.../`` in front where the name had more parts.  A serving engine that serves a directory
    is asked for it by its path, and a path is never printed."""
    name = model.strip().rstrip("/\\")
    last = re.split(r"[/\\]", name)[-1]
    return last, _show(last if last == name else ".../" + last)


def _same(value, neutral) -> bool:
    """Whether a field's value is one of its neutral values: a number by its value (``0`` and ``0.0``), a boolean only as a
    boolean (``true`` is not ``1``), a list or an object by its content."""
    if isinstance(value, bool) or isinstance(neutral, bool):
        return isinstance(value, bool) and isinstance(neutral, bool) and value == neutral
    if isinstance(neutral, (int, float)):
        return isinstance(value, (int, float)) and value == neutral
    return type(value) is type(neutral) and value == neutral


@dataclass
class Request:
    """One chat request of a run: what the checkpoint's chat template is given, the request's own limit of new tokens,
    and what a run says about the body it came in."""

    messages: list
    tools: Optional[list]
    options: dict  # of the checkpoint's chat template (chat_template_kwargs)
    shown: str  # how a run names it: the flag and the file's own name, never a path
    max_tokens: Optional[int] = None  # the request's own limit of new tokens
    limit_field: str = ""  # the name it came under
    stated: list = field(default_factory=list)  # what the body's line says, part by part; empty for a list of messages

    def line(self) -> str:
        """The line a run prints for a request body: every field of it, answered.  "" for a list of messages or a prompt."""
        return f"{self.shown}: a chat request body. " + " ".join(self.stated) if self.stated else ""


def prompt_request(prompt: str, shown: str) -> Request:
    """One user message, as ``--prompt`` gives it."""
    return Request([{"role": "user", "content": prompt}], None, {}, shown)


def parse_request(body, shown: str, checkpoint: str = "") -> Request:
    """A parsed chat request: a list of messages, or the body a client sends.  ``checkpoint`` is the name of the
    checkpoint directory, which the body's ``model`` is compared with.  A field that would change the output is a
    ``ValueError`` that starts with ``shown`` and names the field; so is a body of another shape."""
    messages, tools, options = _text_io().chat_request(body, shown)
    request = Request(messages, tools, options, shown)
    if not isinstance(body, dict):
        return request
    limits = {}
    for name in LIMITS:
        value = body.get(name)
        if value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f'{shown}: "{name}": {_show(value)} is not a limit of new tokens; it takes a positive whole number')
        limits[name] = value
    if len(set(limits.values())) > 1:
        both = " and ".join(f'"{name}": {value}' for name, value in limits.items())
        raise ValueError(f"{shown}: {both} are two limits of new tokens that differ; a request has one")
    read = [f"messages ({len(messages)})"]
    if tools:
        read.append(f"tools ({len(tools)})")
    if options:
        read.append("chat_template_kwargs (" + ", ".join(_name(option) for option in sorted(options)) + ")")
    for name, value in limits.items():
        request.max_tokens, request.limit_field = value, request.limit_field or name
        read.append(f"{name} {value}")
    unused, neutral, unknown, model = [], [], [], ""
    for name, value in body.items():
        if name in READ or value is None:  # null is the field left out
            continue
        if name == "model":
            last, said = _last_part(value) if isinstance(value, str) else ("", "")
            if not last:
                raise ValueError(f'{shown}: "model": {_show(value)} is not a name')
            same = (
                "names the checkpoint directory"
                if last == checkpoint
                else f"not the name of the checkpoint directory; the run decodes the one it was given, {checkpoint}"
            )
            model = f"model {said}: {same}."  # what stands in front of the last part -- an organisation, a path -- is not printed
        elif name in UNUSED:
            if name == "stream" and not isinstance(value, bool):
                raise ValueError(f'{shown}: "stream": {_show(value)} is neither true nor false')
            named = f"{name} {_show(value, 24)}" if name in _VALUE_SHOWN else name  # a value somebody else wrote is not printed back
            unused.append(f"{named} ({UNUSED[name]})")
        elif name in NEUTRAL:
            values, reason = NEUTRAL[name]
            if not any(_same(value, known) for known in values):
                accepted = ", ".join([_show(known) for known in values] + ["null"])
                raise ValueError(
                    f'{shown}: "{name}": {_show(value)} would change the output and is not served: {reason}. Accepted: {accepted}, or the field left out'
                )
            neutral.append(f"{name} {_show(value, 24)}")
        else:
            unknown.append(_name(name))
    request.stated.append("Read: " + ", ".join(read) + ".")
    if model:
        request.stated.append(model)
    if unused:
        request.stated.append("Not used: " + "; ".join(unused) + ".")
    if neutral:
        request.stated.append("At the value that changes nothing: " + ", ".join(neutral) + ".")
    if unknown:
        request.stated.append("Not known to this sample and ignored: " + ", ".join(unknown) + ".")
    return request


def read_request(source: str, checkpoint: str = "", stdin=None) -> Request:
    """A chat request from a JSON file, or from standard input where ``source`` is ``-``.  A source that cannot be read or
    does not hold JSON is a ``ValueError`` that names the flag and the file's own name, never a path."""
    if source != "-":
        shown = f"--messages {os.path.basename(source)}"
        return parse_request(_text_io().read_json(source, shown), shown, checkpoint)
    shown = "--messages -"
    stream = sys.stdin if stdin is None else stdin
    if stream is None:  # closed before the process started (<&-): the interpreter then has no stream in its place
        raise ValueError(f"{shown}: standard input is closed; pipe a request body or a list of messages into the sample")
    try:
        text = stream.read()
    except (OSError, ValueError) as error:  # a descriptor that went away, bytes that are not text
        raise ValueError(f"{shown}: standard input cannot be read ({getattr(error, 'strerror', None) or type(error).__name__})") from None
    if not text.strip():
        raise ValueError(f"{shown}: standard input is empty; pipe a request body or a list of messages into the sample")
    try:
        body = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{shown}: standard input does not hold JSON ({error})") from None
    return parse_request(body, shown, checkpoint)


def new_tokens(
    request: Request, prompt_tokens: int, max_model_len: int, flag: Optional[int] = None, forced: Optional[int] = None, flag_name: str = "--max-tokens"
):
    """The limit of new tokens of one request, held against the context length: ``(limit, note)``.

    ``--force-tokens`` (``forced``) rules over everything; then the request's own limit; then ``--max-tokens`` (``flag``),
    which is the limit of a request that names none; then the default.  A prompt and its new tokens share the context
    (``max_model_len`` positions): a limit somebody asked for that does not fit is a ``ValueError`` before anything is
    loaded, as is a prompt that leaves no room at all, and the default is cut to the room that is left.  ``note`` is ""
    or the line a run prints when a limit that was given does not apply."""
    note = ""
    if forced is not None:
        limit, source = forced, f"--force-tokens {forced}"
        if request.max_tokens is not None:
            note = (
                f'{request.shown}: its "{request.limit_field}": {request.max_tokens} does not apply beside --force-tokens {forced}, '
                f"which decodes exactly {forced} tokens"
            )
    elif request.max_tokens is not None:
        limit, source = request.max_tokens, f'"{request.limit_field}": {request.max_tokens} of the request'
        if flag is not None and flag != limit:
            note = (
                f'{request.shown}: its own "{request.limit_field}": {limit} applies, not {flag_name} {flag}: the flag is the limit of a request that names none'
            )
    elif flag is not None:
        limit, source = flag, f"{flag_name} {flag}"
    else:
        limit, source = MAX_TOKENS, ""
    room = max_model_len - prompt_tokens
    if room < 1:
        raise ValueError(
            f"{request.shown}: the prompt is {prompt_tokens:,} tokens and --max-model-len is {max_model_len:,}: a prompt and its new tokens share "
            "the context, and no position is left for a new token; raise --max-model-len"
        )
    if limit > room and source:
        raise ValueError(
            f"{request.shown}: {prompt_tokens:,} prompt tokens + {limit:,} new tokens ({source}) are {prompt_tokens + limit:,} positions and "
            f"--max-model-len is {max_model_len:,}; ask for at most {room:,} new tokens, or raise --max-model-len"
        )
    if limit > room:
        note = (
            f"{request.shown}: up to {room:,} new tokens, not the default {MAX_TOKENS}: {prompt_tokens:,} prompt tokens leave that many positions "
            f"under --max-model-len {max_model_len:,}"
        )
        limit = room
    return limit, note


def greedy_line(defaults: dict) -> str:
    """The decoding rule of a run beside the checkpoint's own generation settings (``TextIO.generation_defaults``), so that
    the difference from an engine that applies those settings is never silent."""
    rule = "greedy decoding: the token of the largest logit at every step"
    if not defaults:
        return f"{rule} (the checkpoint's generation_config.json names no sampling setting)"
    named = ", ".join(f"{name} {json.dumps(value)}" for name, value in defaults.items())
    return f"{rule}; the checkpoint's own generation settings ({named} in generation_config.json) are not applied"


def first_token_line(prompts: Sequence[tuple]) -> str:
    """What a prompt costs before the first new token, said before anything is loaded.  ``prompts``: ``(shown, prompt
    tokens)`` per request.  The line gives what is exact, a count of decode steps; the seconds are the report's."""
    first, rest = prompts[0], prompts[1:]
    counts = f"{first[1]:,} decode steps before the first new token of {first[0]}, as long as {first[1]:,} new tokens take"
    counts += "".join(f"; {tokens:,} of {shown}" for shown, tokens in rest)
    return (
        "first token: a prompt goes through the decode step, one step per prompt token (no stack of this sample has a prefill kernel): "
        f"{counts}. No text is written until then; the first token line of the report has the time"
    )


def _silence_stdout() -> None:
    """After a write to standard output failed because its reader went away: point the descriptor at the null device, so
    that the interpreter's own flush at exit does not fail a second time with a message of its own."""
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    target = -1
    try:
        target = sys.stdout.fileno()
        os.dup2(null, target)
    except (OSError, ValueError, AttributeError):  # no descriptor behind it: a captured or a replaced stream
        pass
    finally:
        if null != target:  # opened on the number of a closed standard output, the null device stays where it is
            os.close(null)


class TextSink:
    """Where the generated text of sequence 0 goes: standard output, or the file of ``--text-out``; piece by piece as it
    is generated (``streamed``), or once per prompt.  Every text ends in a line break, and nothing else is written there.

    A destination that stops taking text -- a reader that went away, a full disk, a character the stream cannot encode,
    a standard output that was closed before the process started -- is remembered (``error``) and not written to again:
    the run goes on, and its last line says what is missing.

    A request gets one text.  A run with a list of sequence counts decodes every prompt once per count; the text of a
    prompt is written with the first count, and the sink of a later count (``later()``) takes nothing and says whether
    its sequence 0 produced the same text."""

    def __init__(self, path: Optional[str] = None, streamed: bool = True, stream=None):
        self.path, self.streamed = path, streamed
        self._stream = stream  # a stand-in for standard output (tests)
        self._file = None
        self.error = ""
        self._texts: list = []  # the text of every prompt so far, as the first sequence count produced it
        self._first: Optional[TextSink] = None  # of a later sequence count: the sink of the first one
        self._prompt = 0  # of a later sequence count: the prompt its next text belongs to

    @property
    def name(self) -> str:
        """``stdout``, or the file's own name: never a path."""
        return os.path.basename(os.path.normpath(self.path)) if self.path is not None else "stdout"

    def later(self) -> "TextSink":
        """The sink of a later sequence count of the same run: it writes nothing, and holds what it is told of each
        prompt's text against the text the first count produced."""
        sink = TextSink(self.path, self.streamed, self._stream)
        sink._first = self
        return sink

    def announced(self) -> str:
        """For the line before the decode loop: where the text will go."""
        if self._first is not None:
            return f"the report follows; the text of this prompt went to {self.name} with the first sequence count and is not written again"
        if self.streamed:
            return f"the text of sequence 0 goes to {self.name} as it is generated, the report follows"
        return f"the report follows, then the text of sequence 0 goes to {self.name}; the text is not streamed"

    def reported(self, text: str = "") -> str:
        """For the report's ``text out`` line, once per prompt: where the text of sequence 0 (``text``) went, or why it
        is not there."""
        if self._first is not None:
            first = self._first._texts[self._prompt] if self._prompt < len(self._first._texts) else None
            self._prompt += 1
            same = "the same text as" if text == first else "ANOTHER text than"
            return f"not written again: {same} with the first sequence count, whose text is the one that went to {self.name}"
        self._texts.append(text)
        if self.error:
            return f"could not be written to {self.name} ({self.error})"
        return f"written to {self.name} as it was generated" if self.streamed else f"goes to {self.name} after this report"

    def write(self, text: str) -> None:
        if self.error or self._first is not None or not text:
            return
        try:
            if self.path is not None and self._file is None:
                self._file = open(self.path, "w", encoding="utf-8")
            target = self._file or self._stream or sys.stdout
            if target is None:  # closed before the process started (>&-): the interpreter then has no stream in its place
                self.error = "standard output is closed"
                return
            target.write(text)
            target.flush()
        except (OSError, ValueError) as error:  # ValueError: a character the stream cannot encode, a stream that was closed
            self.error = getattr(error, "strerror", None) or str(error) or type(error).__name__
            if self.path is None and self._stream is None:
                _silence_stdout()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError as error:
                self.error = self.error or error.strerror or type(error).__name__
            self._file = None
