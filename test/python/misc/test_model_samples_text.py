# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the text side of the model samples (``samples/models/model_samples/text_io.py``): the synthetic
tokenizer's round trip and its stream; chat requests read from JSON files; and, where ``jinja2`` and ``tokenizers`` are
installed, the text side of a checkpoint directory built here: a chat template with the constructs a checkpoint's template
uses (tool schemas, a reasoning block, tool calls and tool results, several turns, an option, a refusal) rendered against
stored strings, and a tokenizer trained in the test, encoded, decoded and streamed.  The requirement file of the area and
its dependency-update entry are checked as text.  No real tokenizer or template file is committed; no GPU, no weights.

And what a user of a serving engine brings to a sample (``samples/models/qwen_dense/serving_options.py`` and the flags of
``00_decode.py``): the request body a chat client sends, answered field by field -- every field that is accepted at a
neutral value, and every refusal by its own sentence; the limit of new tokens of a request against ``--max-model-len``;
the line that states the decoding rule beside the checkpoint's own generation settings, and the one that says what a
prompt costs before the first new token; where the generated text goes (standard output as it is generated, a file, or
once after the report) while every line of the run goes to standard error; and the sample's ``main()`` on the CPU, to its
last line, on a checkpoint directory generated here, with a request body from a file and from standard input.

The tests that need ``jinja2``, ``tokenizers`` or ``safetensors`` skip visibly where a package is missing."""

import errno
import importlib.util
import io
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "samples" / "models"
FAMILY = MODELS / "qwen_dense"  # the scripts of the family and, beside them, serving_options.py
GIB = 1024**3
#: what the system says when the reader of a pipe has gone away; an ``OSError`` built with this number is that error
PIPE = os.strerror(errno.EPIPE)

#: a chat template with the constructs of a checkpoint's own: a namespace, a macro, a tool block, a system turn, reasoning
#: put back in front of an assistant's content, tool calls, grouped tool results, an option and a refusal
TEMPLATE = """\
{%- set seen = namespace(tools=0) %}
{%- macro render_content(content) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\\n# Tools\\n\\n<tools>' }}
    {%- for tool in tools %}
        {%- set seen.tools = seen.tools + 1 %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>" }}
    {%- if messages[0].role == 'system' %}
        {{- '\\n\\n' + render_content(messages[0].content) }}
    {%- endif %}
    {{- '<|im_end|>\\n' }}
{%- elif messages[0].role == 'system' %}
    {{- '<|im_start|>system\\n' + render_content(messages[0].content) + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content) %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\\n' }}
        {%- if message.reasoning_content is string %}
            {{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}
        {%- endif %}
        {{- content }}
        {%- if message.tool_calls is defined and message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {{- '\\n<tool_call>\\n<function=' + tool_call.function.name + '>\\n' }}
                {%- for name, value in tool_call.function.arguments|items %}
                    {{- '<parameter=' + name + '>\\n' }}
                    {{- value if value is string else value | tojson }}
                    {{- '\\n</parameter>\\n' }}
                {%- endfor %}
                {{- '</function>\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' + content + '\\n</tool_response>' }}
        {%- if loop.last or loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}
"""

WEATHER = {"type": "function", "function": {"name": "weather", "description": "Weather of a <city> & its sky", "parameters": {"type": "object"}}}


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` and the family's script folder on ``sys.path`` for this module only: the paths, every
    ``model_samples`` module and ``serving_options`` leave with it."""
    sys.path.insert(0, str(MODELS))
    sys.path.insert(0, str(FAMILY))
    yield
    sys.path.remove(str(FAMILY))
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n in ("model_samples", "serving_options") or n.startswith("model_samples.")]:
        del sys.modules[name]


def _text_directory(directory: Path, template: str = TEMPLATE, opens_with_a_marker: bool = False, saved_cut_and_padded: bool = False) -> Path:
    """The text files of a checkpoint directory, built here: a byte-level BPE tokenizer trained on a few sentences, with the
    turn markers as special tokens and the reasoning markers as ordinary added tokens; a chat template; the end tokens.
    ``opens_with_a_marker`` gives the tokenizer a post-processor that puts a turn marker in front of what it encodes;
    ``saved_cut_and_padded`` saves it as after a call that truncated to 3 ids and padded to 8."""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, processors, trainers

    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    corpus = ["why is the sky blue", "the sky is blue because air scatters blue light", "system user assistant tool", "weather in the city today"]
    tokenizer.train_from_iterator(corpus * 4, trainer)
    tokenizer.add_tokens([tokenizers.AddedToken("<think>", special=False), tokenizers.AddedToken("</think>", special=False)])
    if opens_with_a_marker:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<|im_start|> $A", special_tokens=[("<|im_start|>", tokenizer.token_to_id("<|im_start|>"))]
        )
    if saved_cut_and_padded:
        tokenizer.enable_truncation(max_length=3)
        tokenizer.enable_padding(length=8, pad_id=tokenizer.token_to_id("<|endoftext|>"), pad_token="<|endoftext|>")
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "chat_template.jinja").write_text(template, encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>", "pad_token": {"content": "<|endoftext|>"}, "model_max_length": 64}))
    end = [tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")]
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": end, "do_sample": True}))
    return directory


def test_the_synthetic_tokenizer_round_trips_and_streams_what_it_decodes():
    from model_samples import TextIO

    text = TextIO.synthetic(300, seed=3)
    assert text.vocab_size == 300 and text.eos_ids == (299,) and text.synthetic_text
    for sample in ("plain ascii", "naïve café", "日本語 text", "🙂 and \x00 byte", ""):
        ids = text.encode(sample)
        assert ids.dtype == torch.int32 and ids.numel() == len(sample.encode("utf-8")), "ids 0..255 are the bytes of the text"
        assert text.decode(ids.tolist()) == sample
        stream = text.decoder()
        streamed = "".join(stream.push(i) for i in ids.tolist()) + stream.close()
        assert streamed == sample, "the streamed pieces add up to the full decode"
    stream = text.decoder()
    pieces = [stream.push(i) for i in "é".encode("utf-8")]
    assert pieces == ["", "é"], "an open multi-byte character gives no text yet"
    produced = [70, 260, 280, 299]  # a byte, two seeded pieces, the end token
    stream = text.decoder()
    streamed = "".join(stream.push(i) for i in produced) + stream.close()
    assert streamed == text.decode(produced) and text.decode(produced).endswith("<|end|>")
    assert TextIO.synthetic(300, seed=3).decode(produced) == text.decode(produced) != TextIO.synthetic(300, seed=4).decode(produced)
    chat = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yes"}]
    assert text.render_chat(chat) == "<user>\nhi\n<assistant>\nyes\n<assistant>\n"
    assert text.render_chat(chat, add_generation_prompt=False) == "<user>\nhi\n<assistant>\nyes", "without the generation prompt the last turn stays open"
    with_tools = text.render_chat(chat[:1], tools=[{"name": "b", "arguments": {}}])
    assert with_tools.startswith('<tools>\n[{"arguments": {}, "name": "b"}]\n</tools>\n<user>'), "tool schemas are text"
    assert text.encode_chat(chat).tolist() == list(text.render_chat(chat).encode())
    with pytest.raises(ValueError, match="takes no options"):
        text.render_chat(chat, enable_thinking=False)
    with pytest.raises(ValueError, match="at least 258"):
        TextIO.synthetic(100)


def test_a_chat_request_is_read_from_a_json_file(tmp_path):
    from model_samples import load_messages

    chat = [{"role": "user", "content": "hi"}]
    (tmp_path / "list.json").write_text(json.dumps(chat))
    assert load_messages(str(tmp_path / "list.json")) == (chat, None, {})
    body = {"messages": chat, "tools": [WEATHER], "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 5}
    (tmp_path / "object.json").write_text(json.dumps(body))
    assert load_messages(str(tmp_path / "object.json")) == (chat, [WEATHER], {"enable_thinking": False}), "what the file holds beyond these three is ignored"
    for name, wrong, sentence in (
        ("empty.json", [], "needs a list of messages"),
        ("text.json", "hi", "needs a list of messages"),
        ("norole.json", [{"content": "hi"}], 'every message has a "role"'),
        ("tools.json", {"messages": chat, "tools": {"name": "x"}}, '"tools" is a list'),
        ("options.json", {"messages": chat, "chat_template_kwargs": [1]}, '"chat_template_kwargs" is an object'),
        ("own.json", {"messages": chat, "chat_template_kwargs": {"tools": [WEATHER]}}, '"chat_template_kwargs" may not hold tools; each has its own entry'),
        ("both.json", {"messages": chat, "chat_template_kwargs": {"tools": [], "messages": []}}, "may not hold messages or tools"),
        ("malformed.json", None, "not a JSON file (Expecting"),
        ("missing.json", None, "cannot be read (No such file or directory)"),
    ):
        if wrong is not None:
            (tmp_path / name).write_text(json.dumps(wrong))
        elif name == "malformed.json":
            (tmp_path / name).write_text('{"messages": [')
        with pytest.raises(ValueError, match=re.escape(sentence)) as refusal:
            load_messages(str(tmp_path / name))
        assert f"--messages {name}:" in str(refusal.value) and str(tmp_path) not in str(refusal.value), "the file's own name, never a path"
    (tmp_path / "latin.json").write_bytes(json.dumps(chat).encode() + b"\xff")
    with pytest.raises(ValueError, match="--messages latin.json: not a JSON file"):
        load_messages(str(tmp_path / "latin.json"))


def test_a_checkpoints_chat_template_renders_tools_reasoning_and_several_turns(tmp_path):
    """The stored strings below are what the template in this file must give; they hold the constructs a checkpoint's
    template uses and the settings of the environment: blocks trimmed, a ``tojson`` that keeps ``<`` and ``&``."""
    pytest.importorskip("jinja2")
    from model_samples import TextIO

    text = TextIO.from_checkpoint(str(_text_directory(tmp_path / "checkpoint")))
    assert not text.synthetic_text and "tokenizers" in text.name and "jinja2" in text.name and str(tmp_path) not in text.name
    user = [{"role": "user", "content": "Why is the sky blue?"}]
    assert text.render_chat(user) == "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert text.render_chat(user, enable_thinking=False).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "an option reaches the template"
    assert text.render_chat(user, add_generation_prompt=False) == "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
    turns = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "Air scatters blue light.", "reasoning_content": "Rayleigh scattering."},
        {"role": "user", "content": "And at sunset?"},
    ]
    assert text.render_chat(turns) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nRayleigh scattering.\n</think>\n\nAir scatters blue light.<|im_end|>\n"
        "<|im_start|>user\nAnd at sunset?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    ), "several turns, with the reasoning of an earlier reply put back in front of its content"
    call = {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris", "days": 2}}}
    agent = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": '{"sky": "clear"}'},
        {"role": "tool", "content": '{"wind": "calm"}'},
    ]
    assert text.render_chat(agent, tools=[WEATHER]) == (
        "<|im_start|>system\n# Tools\n\n<tools>\n"
        '{"type": "function", "function": {"name": "weather", "description": "Weather of a <city> & its sky", "parameters": {"type": "object"}}}\n'
        "</tools>\n\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        '<|im_start|>user\n<tool_response>\n{"sky": "clear"}\n</tool_response>\n<tool_response>\n{"wind": "calm"}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n<think>\n"
    ), "tool schemas, a tool call and two tool results in one turn: all of it is text by the time it is encoded"
    with pytest.raises(ValueError, match="chat template refused the messages: Unexpected message role"):
        text.render_chat([{"role": "narrator", "content": "x"}])
    with pytest.raises(ValueError, match="refused the messages: No messages provided"):
        text.render_chat([])
    named = _text_directory(tmp_path / "named", template="{{ eos_token }}|{{ pad_token }}|{{ messages[0].content }}")
    assert TextIO.from_checkpoint(str(named)).render_chat(user) == "<|im_end|>|<|endoftext|>|Why is the sky blue?", "a template may name the special tokens"
    # a tool call as a serving API writes it, with its arguments as one JSON string: the template wants a mapping and says so
    as_text = {"type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
    with pytest.raises(ValueError, match="chat template refused the messages: "):
        text.render_chat(agent[:2] + [{"role": "assistant", "content": "", "tool_calls": [as_text]}])


def test_the_template_environment_trims_blocks_and_is_an_immutable_sandbox(tmp_path):
    """The settings of the template environment, each held by a template that needs it: a template written without
    whitespace-control dashes renders as the reference library renders it (the line break after a block tag and the indent
    before one are dropped), and a template cannot reach Python through an object it is handed or change what it is given."""
    pytest.importorskip("jinja2")
    from model_samples import TextIO

    plain = (
        "{% for message in messages %}\n"
        "<{{ message.role }}>{{ message.content }}\n"
        "{% endfor %}\n"
        "    {% if add_generation_prompt %}\n"
        "<assistant>\n"
        "    {% endif %}\n"
    )
    user = [{"role": "user", "content": "hi"}]
    text = TextIO.from_checkpoint(str(_text_directory(tmp_path / "plain", template=plain)))
    assert text.render_chat(user) == "<user>hi\n<assistant>\n", "blocks trimmed, their indent stripped"
    assert text.render_chat(user, add_generation_prompt=False) == "<user>hi\n"
    for name, template in (
        ("globals", "{{ raise_exception.__globals__['__builtins__'] }}"),  # a function's globals lead to every module
        ("class", "{{ ''.__class__.__mro__[1].__subclasses__() }}"),
        ("append", "{{ messages.append({'role': 'user', 'content': 'more'}) }}{{ messages | length }}"),  # a change to what it was given
        ("update", "{{ messages[0].update({'content': 'other'}) }}{{ messages[0].content }}"),
    ):
        text = TextIO.from_checkpoint(str(_text_directory(tmp_path / name, template=template)))
        with pytest.raises(ValueError, match="chat template refused the messages: "):
            text.render_chat(user)
    assert user == [{"role": "user", "content": "hi"}]


def test_a_checkpoints_tokenizer_encodes_decodes_and_streams(tmp_path):
    pytest.importorskip("jinja2")
    from model_samples import TextIO

    directory = _text_directory(tmp_path / "checkpoint")
    text = TextIO.from_checkpoint(str(directory))
    start, end = (int(text.encode(marker)[0]) for marker in ("<|im_start|>", "<|im_end|>"))
    assert text.eos_ids[0] == end and len(text.eos_ids) == 2 and text.vocab_size >= 320
    prompt = text.render_chat([{"role": "user", "content": "why is the sky blue"}])
    ids = text.encode(prompt)
    assert ids.dtype == torch.int32 and ids.dim() == 1
    assert int(ids[0]) == start and ids.tolist().count(start) == 2 and ids.tolist().count(end) == 1, "a turn marker is one id, written by the template"
    assert text.encode_chat([{"role": "user", "content": "why is the sky blue"}]).tolist() == ids.tolist()
    assert text.encode("why is the sky blue").numel() < len("why is the sky blue"), "the trained merges are used"
    assert (
        text.decode(ids.tolist()) == "user\nwhy is the sky blue\nassistant\n<think>\n"
    ), "special tokens are left out of the text, the reasoning marker is text"
    for sample in ("the sky is blue", "naïve café 日本 🙂", "<think>\nscatter\n</think>\n\nblue"):
        produced = text.encode(sample).tolist()
        stream = text.decoder(ids)
        streamed = "".join(stream.push(i) for i in produced) + stream.close()
        assert streamed == text.decode(produced) == sample, "the streamed pieces add up to the full decode"
    stream = text.decoder()
    pieces = [stream.push(i) for i in text.encode("é").tolist()]
    assert pieces == ["", "é"], "an open multi-byte character gives no text yet"
    stream = text.decoder()
    pieces = [stream.push(end), stream.push(int(text.encode("blue")[0]))]
    assert pieces[0] == "" and pieces[1] != "", "the end token adds nothing to the text"
    cut = text.encode("sky 🙂").tolist()[:-1]  # generation stopped inside a character: its last byte never came
    stream = text.decoder()
    pushed = "".join(stream.push(i) for i in cut)
    rest, again = stream.close(), stream.close()
    assert pushed == "sky " and rest != "" and rest == again and pushed + rest == text.decode(cut), "close() hands over what the full decode writes for it"
    marked = TextIO.from_checkpoint(str(_text_directory(tmp_path / "marked", opens_with_a_marker=True)))
    assert marked._tokenizer.encode("the sky").ids[0] == start, "this tokenizer adds a turn marker by itself when asked to"
    assert marked.encode("the sky").tolist() == text.encode("the sky").tolist(), "a rendered prompt is encoded as it is: the template wrote its markers"
    saved = _text_directory(tmp_path / "saved", saved_cut_and_padded=True)
    settings = json.loads((saved / "tokenizer.json").read_text())
    assert settings["truncation"]["max_length"] == 3 and settings["padding"]["strategy"] == {"Fixed": 8}, "this tokenizer.json cuts and pads when asked nothing"
    whole = TextIO.from_checkpoint(str(saved))
    for sample in ("why is the sky blue", "the sky"):  # longer than the saved limit, shorter than the saved padding
        assert whole.encode(sample).tolist() == text.encode(sample).tolist(), "a prompt is encoded whole and unpadded, whatever the file was saved with"
    assert whole.encode_chat([{"role": "user", "content": "why is the sky blue"}]).tolist() == ids.tolist()
    (directory / "generation_config.json").unlink()
    (directory / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 7}}))
    assert TextIO.from_checkpoint(str(directory)).eos_ids == (7,), "without a generation config the end token is the model config's"
    for wrong in ("248046", True, 7.0, [], [7, "8"], -1):  # a string would be read letter by letter, true as the id 1
        (directory / "config.json").write_text(json.dumps({"eos_token_id": wrong}))
        with pytest.raises(ValueError, match=r"--model checkpoint: eos_token_id in config.json is .* where a token id or a list of token ids is expected"):
            TextIO.from_checkpoint(str(directory))
        (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": wrong}))
        with pytest.raises(ValueError, match="--model checkpoint: eos_token_id in generation_config.json is "):
            TextIO.from_checkpoint(str(directory))
        (directory / "generation_config.json").unlink()
    (directory / "config.json").write_text(json.dumps({"text_config": [1]}))
    with pytest.raises(ValueError, match="no end-token id there"):
        TextIO.from_checkpoint(str(directory))
    (directory / "config.json").write_text(json.dumps({"eos_token_id": 7}))
    (directory / "chat_template.jinja").write_bytes(b"{{ messages[0].content }}\xff")
    with pytest.raises(ValueError, match="--model checkpoint: chat_template.jinja is not UTF-8 text") as refusal:
        TextIO.from_checkpoint(str(directory))
    assert str(tmp_path) not in str(refusal.value)


def test_a_directory_without_its_text_files_and_a_missing_package_are_told_apart(tmp_path, monkeypatch):
    pytest.importorskip("jinja2")
    from model_samples import TextIO

    directory = _text_directory(tmp_path / "checkpoint")
    (directory / "chat_template.jinja").unlink()
    with pytest.raises(ValueError, match="--model checkpoint: no chat template there") as refusal:
        TextIO.from_checkpoint(str(directory))
    assert str(tmp_path) not in str(refusal.value), "the directory's own name, never a path"
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}!"}))
    assert TextIO.from_checkpoint(str(directory)).render_chat([{"role": "user", "content": "hi"}]) == "hi!", "the template may live in the tokenizer's config"
    (directory / "chat_template.jinja").write_text("{{ messages[0].content }}?")
    assert (
        TextIO.from_checkpoint(str(directory)).render_chat([{"role": "user", "content": "hi"}]) == "hi?"
    ), "the template file comes first where both are there"
    for content in ("{% if %}", "{% generation %}x{% endgeneration %}"):  # a syntax error; a tag this environment does not have
        (directory / "chat_template.jinja").write_text(content)
        with pytest.raises(ValueError, match=re.escape("--model checkpoint: the chat template does not compile (")):
            TextIO.from_checkpoint(str(directory))
    (directory / "chat_template.jinja").unlink()
    (directory / "tokenizer_config.json").write_text("{not json")
    with pytest.raises(ValueError, match=re.escape("--model checkpoint: tokenizer_config.json is not a JSON file (")):
        TextIO.from_checkpoint(str(directory))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}!"}))
    (directory / "generation_config.json").unlink()
    with pytest.raises(ValueError, match="no end-token id there"):
        TextIO.from_checkpoint(str(directory))
    (directory / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match=r"--model checkpoint: tokenizer.json is not a tokenizer the tokenizers package reads \(") as refusal:
        TextIO.from_checkpoint(str(directory))
    assert str(tmp_path) not in str(refusal.value)
    (directory / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="--model checkpoint: no tokenizer.json there; a checkpoint directory holds config.json, tokenizer.json, a chat"):
        TextIO.from_checkpoint(str(directory))
    for missing in (tmp_path / "nowhere", directory / "generation_config.json", "some-org/Some-Model"):
        with pytest.raises(
            ValueError, match=r"not a directory on local disk; pass a checkpoint directory \(a hub name is not fetched: nothing is downloaded; "
        ) as refusal:
            TextIO.from_checkpoint(str(missing))
        assert str(tmp_path) not in str(refusal.value) and f"--model {Path(str(missing)).name}:" in str(refusal.value)
        assert str(refusal.value).endswith("the README beside the sample says where its checkpoint comes from)")
    monkeypatch.setitem(sys.modules, "tokenizers", None)  # as if the package were not installed
    with pytest.raises(NotImplementedError, match=r"tokenizers and jinja2 packages .*: pip install -r samples/models/requirements.txt"):
        TextIO.from_checkpoint(str(directory))


def test_the_requirement_file_has_lower_bounds_a_validated_list_and_an_update_entry():
    lines = (MODELS / "requirements.txt").read_text().splitlines()
    assert lines[0] == "# Sample-only dependencies; not part of cudnn-frontend's runtime package."
    required = {}
    for line in lines:
        if line and not line.startswith("#"):
            name, bound = re.fullmatch(r"([a-z0-9_-]+)>=([0-9][0-9.]*)", line).groups()
            required[name] = tuple(int(n) for n in bound.split("."))
    assert set(required) == {"tokenizers", "jinja2", "safetensors"}, "lower bounds only: PyTorch itself requires jinja2, an exact pin would fight it"
    validated = next(line for line in lines if line.startswith("# Validated with:"))
    for name, bound in required.items():
        version = re.search(rf"{name} ([0-9.]+)", validated)
        assert version, f"the validated list names {name}"
        assert tuple(int(n) for n in version.group(1).split(".")) >= bound, f"{name}: the validated version is not below the lower bound"
    dependabot = (REPO / ".github" / "dependabot.yml").read_text()
    assert re.search(r'package-ecosystem: "?pip"?\n\s+directory: "?/samples/models"?\n', dependabot), "dependency updates watch samples/models/requirements.txt"
    for readme in (MODELS / "README.md", MODELS / "qwen_dense" / "README.md"):
        assert "samples/models/requirements.txt" in readme.read_text(), f"{readme.name} names the requirement file"


# --------------------------------------------------------------------------- coming from a serving engine

CHAT = [{"role": "user", "content": "hi"}]
#: a request body as a chat client sends it: the fields review of the first design named, and one no sample knows
CLIENT_BODY = {
    "model": "some-org/tiny-checkpoint",
    "messages": CHAT,
    "stream": True,
    "max_completion_tokens": 6,
    "seed": 7,
    "n": 1,
    "top_p": 1,
}
#: for every field of ``serving_options.NEUTRAL`` one value that would change the output: each is refused by name
REFUSED = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "n": 2,
    "best_of": 4,
    "use_beam_search": True,
    "length_penalty": 2.0,
    "frequency_penalty": 0.5,
    "presence_penalty": -1,
    "repetition_penalty": 1.1,
    "logit_bias": {"17": -100},
    "logits_processors": [{"qualname": "my.Processor"}],
    "bad_words": ["sky"],
    "allowed_token_ids": [1, 2, 3],
    "watermarking": True,
    "stop": ["\n\n"],
    "include_stop_str_in_output": True,
    "stop_token_ids": [7],
    "ignore_eos": True,
    "min_tokens": 8,
    "repetition_detection": {"max_pattern_size": 4},
    "thinking_token_budget": 128,
    "include_reasoning": False,
    "logprobs": True,
    "top_logprobs": 5,
    "prompt_logprobs": 3,
    "logprob_token_ids": [5, 9],
    "echo": True,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
    "response_format": {"type": "json_object"},
    "structured_outputs": {"regex": "a+"},
    "structural_tag": '{"type": "structural_tag"}',
    "guided_json": {"type": "object"},
    "guided_regex": "a+",
    "guided_choice": ["yes", "no"],
    "guided_grammar": "root ::= answer",
    "tool_choice": "required",
    "function_call": {"name": "weather"},
    "functions": [{"name": "weather"}],
    "add_generation_prompt": False,
    "continue_final_message": True,
    "add_special_tokens": True,
    "chat_template": "{{ messages[0].content }}",
    "documents": [{"text": "a page"}],
    "truncate_prompt_tokens": 128,
    "reasoning_effort": "low",
    "extra_body": {"top_k": 20, "temperature": 0.7},
}
#: What the area guide lists as left out of a model sample, and the fields of a request that ask for it: the names of the
#: public chat request and of the sampling parameters serving engines add to it.  Each is a row of the table.
LEFT_OUT = {
    "sampling": ("temperature", "top_p", "top_k", "min_p", "n", "best_of", "use_beam_search", "length_penalty", "watermarking"),
    "stop strings": ("stop", "include_stop_str_in_output", "stop_token_ids", "repetition_detection"),
    "constrained output": ("response_format", "structured_outputs", "structural_tag", "guided_json", "guided_regex", "guided_choice", "guided_grammar"),
    "logit processors": ("logits_processors", "logit_bias", "bad_words", "allowed_token_ids", "thinking_token_budget"),
    "parsing of the output": ("tool_choice", "function_call", "functions", "include_reasoning"),
    "another prompt than the template's": ("add_generation_prompt", "continue_final_message", "add_special_tokens", "chat_template", "documents"),
    "fields a client library did not put into the body": ("extra_body",),
}


def test_a_request_body_as_a_client_sends_it_is_answered_field_by_field():
    """Every field of a body lands in exactly one class -- read, stated, not used, neutral, or not known -- and the line
    a run prints names each; a list of messages has nothing to answer."""
    import serving_options

    classes = [set(serving_options.READ), {"model"}, set(serving_options.UNUSED), set(serving_options.NEUTRAL)]
    assert sum(len(c) for c in classes) == len(set().union(*classes)), "no field is in two classes"
    assert set(serving_options.LIMITS) <= set(serving_options.READ) and {"seed", "stream"} <= set(serving_options.UNUSED)
    request = serving_options.parse_request(dict(CLIENT_BODY), "--messages body.json", checkpoint="tiny-checkpoint")
    assert (request.messages, request.tools, request.options) == (CHAT, None, {})
    assert (request.max_tokens, request.limit_field, request.shown) == (6, "max_completion_tokens", "--messages body.json")
    assert request.line() == (
        "--messages body.json: a chat request body. Read: messages (1), max_completion_tokens 6. "
        'model ".../tiny-checkpoint": names the checkpoint directory. '
        "Not used: stream true (the text goes where the command line says, as it is generated unless --no-stream); "
        "seed 7 (token choice is greedy, nothing is drawn). "
        "At the value that changes nothing: n 1, top_p 1."
    )
    other = serving_options.parse_request({"model": "gpt-4o", "messages": CHAT, "max_tokens": 9}, "--messages -", checkpoint="tiny-checkpoint")
    assert (other.max_tokens, other.limit_field) == (9, "max_tokens")
    assert 'model "gpt-4o": not the name of the checkpoint directory; the run decodes the one it was given, tiny-checkpoint.' in other.line()
    full = {
        "messages": CHAT,
        "tools": [WEATHER],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 5,
        "max_completion_tokens": 5,
        "user": "u-1",
        "service_tier": "auto",
        "parallel_tool_calls": False,
        "frobnicate": [1, 2],
        "audio": None,
        "seed": None,
    }
    request = serving_options.parse_request(full, "--messages full.json")
    assert (request.tools, request.options, request.max_tokens, request.limit_field) == ([WEATHER], {"enable_thinking": False}, 5, "max_tokens")
    line = request.line()
    assert "Read: messages (1), tools (1), chat_template_kwargs (enable_thinking), max_tokens 5, max_completion_tokens 5." in line
    assert (
        "Not used: user (one process: there is no account); service_tier (one process: there is no queue); "
        "parallel_tool_calls (the output is text: tool calls in it are not parsed)."
    ) in line
    assert "u-1" not in line and '"auto"' not in line, "a field that is not used is named; what somebody wrote into it is not printed back"
    assert line.endswith("Not known to this sample and ignored: frobnicate."), "a field no sample knows is printed, not refused and not swallowed"
    assert "audio" not in line and "seed" not in line and "model" not in line, "null is the field left out"
    for messages_only in (CHAT, {"messages": CHAT}):
        request = serving_options.parse_request(messages_only, "--messages chat.json")
        assert request.messages == CHAT and request.max_tokens is None
        assert request.line() == ("" if isinstance(messages_only, list) else "--messages chat.json: a chat request body. Read: messages (1).")
    prompt = serving_options.prompt_request("Why?", "--prompt 1")
    assert (prompt.messages, prompt.tools, prompt.options, prompt.max_tokens, prompt.line()) == ([{"role": "user", "content": "Why?"}], None, {}, None, "")


def test_the_test_names_a_refused_value_for_every_field_that_has_a_neutral_one():
    import serving_options

    assert set(REFUSED) == set(serving_options.NEUTRAL), "a field added to the table comes with a value this module refuses it at"
    for name, (values, reason) in serving_options.NEUTRAL.items():
        assert isinstance(values, tuple) and reason and reason[0].islower() and not reason.endswith("."), name


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_field_that_would_change_the_output_is_accepted_at_its_neutral_values_and_refused_by_name_elsewhere(name):
    import serving_options

    values, reason = serving_options.NEUTRAL[name]
    for value in (*values, None):
        request = serving_options.parse_request({"messages": CHAT, name: value}, "--messages body.json")
        said = request.line()
        if value is None:
            assert said == "--messages body.json: a chat request body. Read: messages (1).", "null is the field left out"
        else:
            assert f"At the value that changes nothing: {name} {json.dumps(value)}." in said
    accepted = ", ".join([json.dumps(value) for value in values] + ["null"])
    sentence = (
        f'--messages body.json: "{name}": {json.dumps(REFUSED[name])} would change the output and is not served: {reason}. '
        f"Accepted: {accepted}, or the field left out"
    )
    with pytest.raises(ValueError) as refusal:
        serving_options.parse_request({"messages": CHAT, name: REFUSED[name]}, "--messages body.json")
    assert str(refusal.value) == sentence


def test_a_neutral_value_is_held_by_its_kind():
    """``true`` is not 1 and "0" is not 0: a value of another kind than the neutral one is refused; a whole number and its
    floating-point spelling are the same value."""
    import serving_options

    for name, value in (
        ("temperature", 0.0),
        ("top_p", 1.0),
        ("n", 1.0),
        ("top_k", -1),
        ("logit_bias", {}),
        ("stop", []),
        ("response_format", {"type": "text"}),
    ):
        assert serving_options.parse_request({"messages": CHAT, name: value}, "--messages b.json").line().endswith(f"{name} {json.dumps(value)}.")
    for name, value in (
        ("n", True),
        ("temperature", False),
        ("temperature", "0"),
        ("top_p", "1"),
        ("stop", ""),
        ("stop", "</s>"),
        ("logprobs", 0),
        ("ignore_eos", 0),
        ("response_format", {"type": "text", "strict": True}),
        ("tool_choice", "none"),
        ("tool_choice", {"type": "function", "function": {"name": "weather"}}),
    ):
        with pytest.raises(ValueError, match=re.escape(f'"{name}": {json.dumps(value)} would change the output and is not served: ')):
            serving_options.parse_request({"messages": CHAT, name: value}, "--messages b.json")
    long = {"messages": CHAT, "logit_bias": {str(i): -100 for i in range(40)}}
    with pytest.raises(ValueError) as refusal:
        serving_options.parse_request(long, "--messages b.json")
    assert '"logit_bias": {"0": -100, ' in str(refusal.value) and "... would change the output" in str(refusal.value), "a long value is cut where it is shown"


def test_a_field_that_asks_for_what_a_sample_leaves_out_is_a_row_of_the_table_and_the_guide_says_what_holds():
    """Stop strings, constrained output, logit processors, parsing of the output: the area guide lists what a model
    sample leaves out.  A request field that asks for one of them is a row of the table, under every name a client
    sends it, and is refused by name; a field no row names is printed and ignored, and the guide says both halves."""
    import serving_options

    for feature, names in LEFT_OUT.items():
        for name in names:
            assert name in serving_options.NEUTRAL, f"{name} asks for {feature}: a row of the table, not a field the sample does not know"
            with pytest.raises(ValueError, match=re.escape(f'--messages body.json: "{name}": ')):
                serving_options.parse_request({"messages": CHAT, name: REFUSED[name]}, "--messages body.json")
    # a client library's keyword for the fields it adds: what stands in it would be decoded without, so it is refused whole
    with pytest.raises(ValueError) as refusal:
        serving_options.parse_request({"messages": CHAT, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, "--messages body.json")
    assert str(refusal.value).endswith(
        "would change the output and is not served: it is how a client library is handed the fields it adds to a request, "
        "not a field of the body it sends: its fields go to the top level of the body. Accepted: {}, null, or the field left out"
    )
    assert serving_options.parse_request({"messages": CHAT, "extra_body": {}}, "--messages body.json").line().endswith("extra_body {}.")
    # outside the table: printed, by its name, and ignored -- the sample cannot say what such a field would change
    line = serving_options.parse_request({"messages": CHAT, "frobnicate": 1, "cache_salt": "x"}, "--messages body.json").line()
    assert line.endswith("Not known to this sample and ignored: frobnicate, cache_salt.")
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    out = guide.split("Also out: ", 1)[1].split(" ## Rules", 1)[0]
    assert all(word in out for word in ("stop strings", "constrained output", "logit processors", "tool-call parsing"))
    assert "is refused there by name" in out and "is printed as not known and ignored" in out, "the guide says what the code does, both halves"
    assert "never ignored" not in out, "a field outside the table is ignored: the tripwire does not say otherwise"


def test_the_line_of_a_request_body_prints_names_and_the_values_the_table_knows_and_no_path(tmp_path):
    """What a run prints of a body: of ``model`` the last part (an engine that serves a directory is asked for it by its
    path); of a field that is not used its name, and its value only where that is a number or a boolean; a name that is
    not a plain word as a JSON string, so that a line break in it cannot put a line of its own into the run."""
    import serving_options

    def said(body, checkpoint="tiny-checkpoint"):
        return serving_options.parse_request({"messages": CHAT, **body}, "--messages body.json", checkpoint=checkpoint).line()

    for model in (
        "/data/models/secret-project/tiny-checkpoint",
        "/data/models/secret-project/tiny-checkpoint/",
        "  /data/models/secret-project/tiny-checkpoint  ",
        "C:\\models\\secret-project\\tiny-checkpoint",
        "some-org/tiny-checkpoint",
    ):
        line = said({"model": model})
        assert line.endswith('model ".../tiny-checkpoint": names the checkpoint directory.'), model
        assert "secret-project" not in line and "some-org" not in line, "what stands in front of the last part is not printed"
    for whole in ("tiny-checkpoint", "tiny-checkpoint/", " tiny-checkpoint "):
        assert said({"model": whole}).endswith('model "tiny-checkpoint": names the checkpoint directory.'), whole
    other = said({"model": "/srv/weights/other-model"})
    assert other.endswith('model ".../other-model": not the name of the checkpoint directory; the run decodes the one it was given, tiny-checkpoint.')
    assert "/srv" not in other
    assert said({"model": "/srv/" + "m" * 80}).count("m" * 51 + "...:") == 1, "a long last part is cut where it is shown"
    for no_name in ("/", "//", " / ", "\\"):
        with pytest.raises(ValueError, match=re.escape(f'--messages body.json: "model": {json.dumps(no_name)} is not a name')):
            said({"model": no_name})
    # a field that is not used: its name and the reason; the value only of the two the table knows as a number and a boolean
    line = said(
        {
            "user": "alice@example.com",
            "metadata": {"ticket": "T-1234", "folder": str(tmp_path)},
            "store": True,
            "service_tier": "flex",
            "stream_options": {"include_usage": True},
            "parallel_tool_calls": True,
            "seed": 1234,
            "stream": False,
        }
    )
    assert line.endswith(
        "Not used: user (one process: there is no account); metadata (one process: nothing is stored); store (one process: nothing is stored); "
        "service_tier (one process: there is no queue); stream_options (one process: there is no response to shape); "
        "parallel_tool_calls (the output is text: tool calls in it are not parsed); seed 1234 (token choice is greedy, nothing is drawn); "
        "stream false (the text goes where the command line says, as it is generated unless --no-stream)."
    )
    for written in ("alice@example.com", "T-1234", str(tmp_path), "flex", "include_usage"):
        assert written not in line, written
    assert said({"seed": "a\n[00] PASS"}).count("\n") == 0, "a value that is shown is shown as JSON: a line break in it is escaped"
    # names: a plain word as it stands, anything else as a JSON string
    forged = "x\n[00] PASS"
    line = said({forged: 1, "frobnicate": 2, "a b": 3, "chat_template_kwargs": {forged: 4, "enable_thinking": False}, "n" * 61: 5})
    assert "\n" not in line, "one line, whatever a field is called"
    assert 'chat_template_kwargs (enable_thinking, "x\\n[00] PASS")' in line
    assert line.endswith('Not known to this sample and ignored: "x\\n[00] PASS", frobnicate, "a b", "' + "n" * 56 + "....")


def test_what_else_a_request_body_can_get_wrong_each_with_its_own_sentence():
    import serving_options

    def refused(body):
        with pytest.raises(ValueError) as refusal:
            serving_options.parse_request(body, "--messages body.json", checkpoint="tiny-checkpoint")
        return str(refusal.value)

    for wrong in (0, -3, 2.5, "8", True, [8]):
        for name in serving_options.LIMITS:
            said = refused({"messages": CHAT, name: wrong})
            assert said == f'--messages body.json: "{name}": {json.dumps(wrong)} is not a limit of new tokens; it takes a positive whole number'
    assert refused({"messages": CHAT, "max_tokens": 5, "max_completion_tokens": 9}) == (
        '--messages body.json: "max_tokens": 5 and "max_completion_tokens": 9 are two limits of new tokens that differ; a request has one'
    )
    for wrong in (3, "", "  ", ["a"], {"id": "a"}):
        assert refused({"messages": CHAT, "model": wrong}) == f'--messages body.json: "model": {json.dumps(wrong)} is not a name'
    for wrong in (1, 0, "true", "yes"):
        assert refused({"messages": CHAT, "stream": wrong}) == f'--messages body.json: "stream": {json.dumps(wrong)} is neither true nor false'
    # the shape of the request itself: the helper layer's sentences, under the name the run shows
    assert refused({"max_tokens": 5}) == '--messages body.json: needs a list of messages, or an object with "messages"; every message has a "role"'
    assert refused("hi").startswith("--messages body.json: needs a list of messages")
    assert refused({"messages": CHAT, "tools": {"name": "x"}}) == '--messages body.json: "tools" is a list of JSON tool schemas'
    assert refused({"messages": CHAT, "chat_template_kwargs": [1]}) == '--messages body.json: "chat_template_kwargs" is an object'
    # a refusal comes before anything else is said: the first field that would change the output ends the reading
    assert '"temperature": 1' in refused({"messages": CHAT, "seed": 3, "temperature": 1, "n": 4}) and '"n": 4' in refused({"messages": CHAT, "n": 4})


def test_a_request_is_read_from_a_file_or_from_standard_input(tmp_path, monkeypatch):
    import serving_options
    from model_samples import load_messages

    (tmp_path / "body.json").write_text(json.dumps(CLIENT_BODY))
    from_file = serving_options.read_request(str(tmp_path / "body.json"), "tiny-checkpoint")
    assert from_file.shown == "--messages body.json" and from_file.max_tokens == 6 and str(tmp_path) not in from_file.line()
    assert load_messages(str(tmp_path / "body.json")) == (CHAT, None, {}), "the helper layer reads the same file for what a template is given"
    from_stdin = serving_options.read_request("-", "tiny-checkpoint", stdin=io.StringIO(json.dumps(CLIENT_BODY)))
    assert from_stdin.shown == "--messages -" and from_stdin.line() == from_file.line().replace("--messages body.json", "--messages -")
    assert serving_options.read_request("-", stdin=io.StringIO(json.dumps(CHAT))).messages == CHAT, "a list of messages on standard input"
    for content, sentence in (
        ("", "--messages -: standard input is empty; pipe a request body or a list of messages into the sample"),
        ("  \n", "--messages -: standard input is empty; pipe a request body or a list of messages into the sample"),
        ('{"messages": [', "--messages -: standard input does not hold JSON (Expecting value: line 1 column 15 (char 14))"),
        ('{"messages": [{"role": "user", "content": "hi"}], "n": 3}', '--messages -: "n": 3 would change the output and is not served: '),
    ):
        with pytest.raises(ValueError, match=re.escape(sentence)):
            serving_options.read_request("-", stdin=io.StringIO(content))

    class Closed:
        def read(self):
            raise OSError(9, "Bad file descriptor")

    with pytest.raises(ValueError, match=re.escape("--messages -: standard input cannot be read (Bad file descriptor)")):
        serving_options.read_request("-", stdin=Closed())
    with monkeypatch.context() as patched:  # closed before the process started (<&-): the interpreter has None in its place
        patched.setattr(sys, "stdin", None)
        with pytest.raises(ValueError) as refusal:
            serving_options.read_request("-")
    assert str(refusal.value) == "--messages -: standard input is closed; pipe a request body or a list of messages into the sample"
    (tmp_path / "malformed.json").write_text('{"messages": [')
    for name, sentence in (("missing.json", "cannot be read (No such file or directory)"), ("malformed.json", "not a JSON file (Expecting")):
        with pytest.raises(ValueError, match=re.escape(f"--messages {name}: {sentence}")) as refusal:
            serving_options.read_request(str(tmp_path / name))
        assert str(tmp_path) not in str(refusal.value), "the file's own name, never a path"
    (tmp_path / "sampled.json").write_text(json.dumps({"messages": CHAT, "temperature": 0.6, "top_p": 0.95}))
    with pytest.raises(ValueError, match=re.escape('--messages sampled.json: "temperature": 0.6 would change the output and is not served: ')):
        serving_options.read_request(str(tmp_path / "sampled.json"))


def test_the_limit_of_new_tokens_of_a_request_is_held_against_the_context_before_anything_is_loaded():
    """Which limit applies -- ``--force-tokens``, the request's own, ``--max-tokens``, the default -- and the context a
    prompt and its new tokens share: a limit somebody asked for that does not fit is an error, the default is cut."""
    import serving_options

    assert (serving_options.MAX_TOKENS, serving_options.MAX_MODEL_LEN) == (256, 2051)
    plain = serving_options.prompt_request("hi", "--prompt 1")
    body = serving_options.parse_request({"messages": CHAT, "max_completion_tokens": 40}, "--messages body.json")
    assert serving_options.new_tokens(plain, 58, 2051) == (256, "")
    assert serving_options.new_tokens(plain, 58, 2051, flag=12) == (12, "")
    assert serving_options.new_tokens(body, 58, 2051) == (40, "")
    assert serving_options.new_tokens(body, 58, 2051, flag=40) == (40, ""), "the flag and the request agree: nothing to say"
    assert serving_options.new_tokens(body, 58, 2051, flag=12) == (
        40,
        '--messages body.json: its own "max_completion_tokens": 40 applies, not --max-tokens 12: the flag is the limit of a request that names none',
    )
    assert "not --max-new-tokens 12:" in serving_options.new_tokens(body, 58, 2051, flag=12, flag_name="--max-new-tokens")[1], "the spelling that was typed"
    assert serving_options.new_tokens(plain, 58, 2051, forced=32) == (32, "")
    assert serving_options.new_tokens(body, 58, 2051, flag=None, forced=32) == (
        32,
        '--messages body.json: its "max_completion_tokens": 40 does not apply beside --force-tokens 32, which decodes exactly 32 tokens',
    )
    # the context: positions 1 .. max_model_len hold the prompt and the new tokens
    assert serving_options.new_tokens(plain, 1795, 2051) == (256, ""), "1,795 + 256 = 2,051: the default fits exactly"
    assert serving_options.new_tokens(plain, 1796, 2051) == (
        255,
        "--prompt 1: up to 255 new tokens, not the default 256: 1,796 prompt tokens leave that many positions under --max-model-len 2,051",
    )
    assert serving_options.new_tokens(plain, 2050, 2051)[0] == 1
    assert serving_options.new_tokens(plain, 58, 2051, flag=1993) == (1993, ""), "58 + 1,993 = 2,051"
    for arguments, source in (
        ({"flag": 1994}, "--max-tokens 1994"),
        ({"flag": 1994, "flag_name": "--max-new-tokens"}, "--max-new-tokens 1994"),
        ({"forced": 1994}, "--force-tokens 1994"),
    ):
        with pytest.raises(ValueError) as refusal:
            serving_options.new_tokens(plain, 58, 2051, **arguments)
        assert str(refusal.value) == (
            f"--prompt 1: 58 prompt tokens + 1,994 new tokens ({source}) are 2,052 positions and --max-model-len is 2,051; "
            "ask for at most 1,993 new tokens, or raise --max-model-len"
        )
    with pytest.raises(ValueError) as refusal:
        serving_options.new_tokens(body, 58, 64)
    assert str(refusal.value) == (
        '--messages body.json: 58 prompt tokens + 40 new tokens ("max_completion_tokens": 40 of the request) are 98 positions and '
        "--max-model-len is 64; ask for at most 6 new tokens, or raise --max-model-len"
    )
    for prompt_tokens in (2051, 2052, 4000):
        with pytest.raises(ValueError) as refusal:
            serving_options.new_tokens(plain, prompt_tokens, 2051, flag=1)
        assert str(refusal.value) == (
            f"--prompt 1: the prompt is {prompt_tokens:,} tokens and --max-model-len is 2,051: a prompt and its new tokens share the context, "
            "and no position is left for a new token; raise --max-model-len"
        )


def test_the_default_context_length_is_the_one_the_census_of_the_target_kernel_set_was_recorded_with():
    import serving_options

    census = json.loads((MODELS / "model_samples" / "qwen_dense" / "kernel_census.json").read_text())
    recorded = [section["recorded"]["max_model_len"] for sections in census["configurations"].values() for section in sections.values()]
    assert recorded and set(recorded) == {serving_options.MAX_MODEL_LEN}, "--max-model-len defaults to the context length of the recorded sections"


def test_the_decoding_rule_is_said_beside_the_checkpoints_own_generation_settings(tmp_path):
    pytest.importorskip("jinja2")
    import serving_options
    from model_samples import TextIO
    from model_samples.text_io import SAMPLING_SETTINGS

    assert TextIO.synthetic(300).generation_defaults == {}
    directory = _text_directory(tmp_path / "checkpoint")
    assert TextIO.from_checkpoint(str(directory)).generation_defaults == {"do_sample": True}, "what the file names, and nothing it does not"
    settings = {"eos_token_id": 7, "do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "pad_token_id": 0, "transformers_version": "5.0"}
    (directory / "generation_config.json").write_text(json.dumps(settings))
    defaults = TextIO.from_checkpoint(str(directory)).generation_defaults
    assert defaults == {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95} and set(defaults) <= set(SAMPLING_SETTINGS)
    assert list(defaults) == ["do_sample", "temperature", "top_k", "top_p"], "in the order of SAMPLING_SETTINGS, whatever the file's"
    assert serving_options.greedy_line(defaults) == (
        "greedy decoding: the token of the largest logit at every step; the checkpoint's own generation settings "
        "(do_sample true, temperature 1.0, top_k 20, top_p 0.95 in generation_config.json) are not applied"
    )
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 7, "temperature": None, "repetition_penalty": 1.05}))
    defaults = TextIO.from_checkpoint(str(directory)).generation_defaults
    assert defaults == {"repetition_penalty": 1.05}, "null is a setting the file does not make"
    assert "(repetition_penalty 1.05 in generation_config.json) are not applied" in serving_options.greedy_line(defaults)
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 7}))
    assert serving_options.greedy_line(TextIO.from_checkpoint(str(directory)).generation_defaults) == (
        "greedy decoding: the token of the largest logit at every step (the checkpoint's generation_config.json names no sampling setting)"
    )


def test_the_first_token_line_counts_decode_steps_and_states_no_time():
    import serving_options

    one = serving_options.first_token_line([("--prompt 1", 58)])
    assert one == (
        "first token: a prompt goes through the decode step, one step per prompt token (no stack of this sample has a prefill kernel): "
        "58 decode steps before the first new token of --prompt 1, as long as 58 new tokens take. "
        "No text is written until then; the first token line of the report has the time"
    )
    several = serving_options.first_token_line([("--prompt 1", 58), ("--messages body.json", 1204), ("--messages -", 7)])
    assert (
        "58 decode steps before the first new token of --prompt 1, as long as 58 new tokens take; 1,204 of --messages body.json; 7 of --messages -. " in several
    )
    assert not re.search(r"\d\s*(?:s|ms|seconds)\b", one + several), "a count, never a time: the report measures it"


class _Recorder:
    """A stand-in for standard output that records what it is handed and when it is flushed.  Its ``fail_at``-th write
    fails with ``error``, once: a write after it would be recorded again."""

    def __init__(self, events=None, fail_at=None, error=None):
        self.events = [] if events is None else events
        self.fail_at, self.error, self.writes = fail_at, error, 0

    def write(self, text):
        self.writes += 1
        if self.writes == self.fail_at:
            raise self.error
        self.events.append(("text", text))

    def flush(self):
        self.events.append(("flush",))


def test_where_the_generated_text_goes(tmp_path, monkeypatch):
    """Standard output or a file, piece by piece or once; a destination that stops taking text is remembered and not
    written to again; a later sequence count of a run writes nothing and says whether its text is the first count's."""
    import serving_options

    out = _Recorder()
    sink = serving_options.TextSink(stream=out)
    assert (sink.name, sink.streamed, sink.error) == ("stdout", True, "")
    assert sink.announced() == "the text of sequence 0 goes to stdout as it is generated, the report follows"
    for piece in ("The", "", " sky", "\n"):
        sink.write(piece)
    assert out.events == [
        ("text", "The"),
        ("flush",),
        ("text", " sky"),
        ("flush",),
        ("text", "\n"),
        ("flush",),
    ], "every piece is flushed; an empty one is not written"
    assert sink.reported() == "written to stdout as it was generated"
    once = serving_options.TextSink(streamed=False, stream=_Recorder())
    assert once.announced() == "the report follows, then the text of sequence 0 goes to stdout; the text is not streamed"
    assert once.reported() == "goes to stdout after this report"
    target = tmp_path / "answers" / "answer.txt"
    target.parent.mkdir()
    file = serving_options.TextSink(str(target))
    assert file.name == "answer.txt" and "answer.txt as it is generated" in file.announced() and str(tmp_path) not in file.announced() + file.reported()
    assert not target.exists(), "nothing is created before there is text"
    target.write_text("the answer of an earlier run\n")  # what the file held is replaced, not added to
    file.write("naïve ")
    assert target.read_text(encoding="utf-8") == "naïve ", "a piece is on disk when write() returns"
    file.write("café 🙂\n")
    file.close()
    file.close()
    assert target.read_bytes() == "naïve café 🙂\n".encode("utf-8") and file.error == ""
    for error, said in (
        (OSError(errno.EPIPE, PIPE), PIPE),
        (OSError(28, "No space left on device"), "No space left on device"),
        (
            UnicodeEncodeError("ascii", "é", 0, 1, "ordinal not in range(128)"),
            "'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)",
        ),
    ):
        out = _Recorder(fail_at=2, error=error)
        sink = serving_options.TextSink(stream=out)
        for piece in ("a", "b", "c"):
            sink.write(piece)
        assert sink.error == said and out.events == [
            ("text", "a"),
            ("flush",),
        ], "after a failed write nothing more is written, although the stream would take it"
        assert sink.reported() == f"could not be written to stdout ({said})"
    gone = serving_options.TextSink(str(tmp_path / "no-such-folder" / "answer.txt"))
    gone.write("text")
    gone.close()
    assert gone.error == "No such file or directory" and gone.reported() == "could not be written to answer.txt (No such file or directory)"
    unnamed = serving_options.TextSink("")  # a file flag that was given an empty name is a file all the same, never standard output
    unnamed.write("text")
    assert unnamed.error == "No such file or directory" and unnamed.name != "stdout" and "stdout" not in unnamed.reported()
    # a standard output that was closed before the process started (>&-): the interpreter has None in its place
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdout", None)
        closed = serving_options.TextSink()
        for piece in ("a", "b"):
            closed.write(piece)
    assert closed.error == "standard output is closed" and closed.reported() == "could not be written to stdout (standard output is closed)"
    shut = io.StringIO()
    shut.close()  # and a stream object that was closed under the run
    closed = serving_options.TextSink(stream=shut)
    closed.write("a")
    assert closed.error.startswith("I/O operation on closed file")

    # a request gets one text: the sink of a later sequence count takes nothing, and says whose text the destination holds
    out = _Recorder()
    first = serving_options.TextSink(stream=out)
    first.write("The sky")
    first.write("\n")
    assert first.reported("The sky") == "written to stdout as it was generated"
    first.write("Rayleigh\n")
    assert first.reported("Rayleigh") == "written to stdout as it was generated"
    written = list(out.events)
    later = first.later()
    assert (later.name, later.streamed, later.path) == ("stdout", True, None)
    assert later.announced() == "the report follows; the text of this prompt went to stdout with the first sequence count and is not written again"
    for piece in ("The sky", "\n"):
        later.write(piece)
    assert later.reported("The sky") == "not written again: the same text as with the first sequence count, whose text is the one that went to stdout"
    assert later.reported("Scattering") == "not written again: ANOTHER text than with the first sequence count, whose text is the one that went to stdout"
    assert later.reported("") == "not written again: ANOTHER text than with the first sequence count, whose text is the one that went to stdout"
    assert out.events == written and later.error == "", "a later sequence count writes nothing"
    filed = serving_options.TextSink(str(target), streamed=False).later()
    assert "went to answer.txt with the first sequence count" in filed.announced() and "that went to answer.txt" in filed.reported("x")


def test_a_reader_of_standard_output_that_goes_away_ends_in_no_traceback():
    """``sample | head``: the reader closes the pipe while the run still writes.  The sink remembers it and points the
    descriptor at the null device, so the interpreter's flush at exit has nothing to fail on."""
    code = f"""
import sys
sys.path.insert(0, {str(FAMILY)!r})
import serving_options
sink = serving_options.TextSink()
for piece in ("a", "b", "c"):
    sink.write(piece * 100000)
print("left over in the buffer")
sys.stderr.write("SINK ERROR=" + sink.error + "\\n")
"""
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        done = subprocess.run([sys.executable, "-c", code], stdout=write_end, stderr=subprocess.PIPE, text=True, timeout=120)
    finally:
        os.close(write_end)
    assert done.returncode == 0 and done.stderr == f"SINK ERROR={PIPE}\n", done.stderr


def test_silencing_standard_output_keeps_no_descriptor_of_its_own(monkeypatch):
    """The null device is opened to be put where standard output was, and its own descriptor is closed again: where
    there is no descriptor to take it (a replaced stream, here), and where there is one (a child process whose reader
    went away: standard output is the null device afterwards, and the process holds as many descriptors as before)."""
    import serving_options

    def held():
        return len(os.listdir("/proc/self/fd"))

    monkeypatch.setattr(sys, "stdout", io.StringIO())  # a stream with no descriptor behind it
    before = held()
    for _ in range(3):
        serving_options._silence_stdout()
    assert held() == before, "the descriptor of the null device was left open"

    code = f"""
import os, sys
sys.path.insert(0, {str(FAMILY)!r})
import serving_options
before = len(os.listdir("/proc/self/fd"))
serving_options._silence_stdout()
serving_options._silence_stdout()
null = os.stat(os.devnull)
now = os.fstat(sys.stdout.fileno())
sys.stderr.write(f"HELD {{len(os.listdir('/proc/self/fd')) - before}} NULL {{(now.st_dev, now.st_ino) == (null.st_dev, null.st_ino)}}\\n")
"""
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        done = subprocess.run([sys.executable, "-c", code], stdout=write_end, stderr=subprocess.PIPE, text=True, timeout=120)
    finally:
        os.close(write_end)
    assert done.returncode == 0 and done.stderr == "HELD 0 NULL True\n", done.stderr


def test_a_standard_stream_that_was_closed_before_the_process_started_ends_in_no_traceback():
    """``sample >&-``, ``<&-``, ``2>&-``: the interpreter then starts with ``None`` where the stream would be.  Three child
    processes, each with one descriptor closed before it starts: the sink remembers a standard output that is not there,
    a read of standard input is a ``ValueError`` in the sample's words, and a line of a run goes nowhere -- not to
    standard output, which is where ``print`` sends what it is handed no stream for."""
    head = f"import sys\nsys.path.insert(0, {str(FAMILY)!r})\nimport serving_options\n"
    children = {
        1: head + "sink = serving_options.TextSink()\nsink.write('a')\nsink.write('b')\n"
        "sys.stderr.write(f'stdout={sys.stdout!r} error={sink.error} reported={sink.reported()}\\n')\n",
        0: head + "try:\n    serving_options.read_request('-')\nexcept ValueError as error:\n    sys.stderr.write(f'stdin={sys.stdin!r} {error}\\n')\n",
        2: "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sample_with_standard_error_closed', {str(FAMILY / '00_decode.py')!r})\n"
        "sample = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(sample)\n"
        "sample._say('[00] a line of the run', 'and a table', flush=True)\nprint(f'stderr={sys.stderr!r}: the text')\n",
    }
    done = {}
    for descriptor, code in children.items():
        done[descriptor] = subprocess.run(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            preexec_fn=lambda descriptor=descriptor: os.close(descriptor),  # in the child, before the interpreter starts
        )
    assert (done[1].returncode, done[1].stdout, done[1].stderr) == (
        0,
        "",
        "stdout=None error=standard output is closed reported=could not be written to stdout (standard output is closed)\n",
    )
    assert (done[0].returncode, done[0].stdout, done[0].stderr) == (
        0,
        "",
        "stdin=None --messages -: standard input is closed; pipe a request body or a list of messages into the sample\n",
    )
    assert (done[2].returncode, done[2].stdout, done[2].stderr) == (0, "stderr=None: the text\n", ""), "no line of the run in the text"


def _sample_script(monkeypatch, script="00_decode.py"):
    """A sample script of the family as a module, for its own functions; what loading it changes in the process leaves with
    the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models and its own folder on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location(f"model_sample_qwen_dense_{script[:2]}_serving_options", FAMILY / script)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


def test_the_serving_style_flags_of_the_sample(monkeypatch, capsys):
    """``--max-tokens`` and its older spelling, ``--max-model-len``, ``--temperature``, ``--messages -``, ``--text-out``,
    ``--no-stream``: their defaults, what each applies to, and every refusal by its sentence."""
    import serving_options

    sample = _sample_script(monkeypatch)
    assert sample.serving_options is serving_options, "the script imports the module beside it"
    seeded = sample._arguments([])
    assert (seeded.max_new_tokens, seeded.limit_given, seeded.limit_flag, seeded.max_model_len) == (256, False, "--max-tokens", 2051)
    assert (seeded.temperature, seeded.text_out, seeded.no_stream, seeded.messages) == (None, None, False, [])
    assert sample._arguments(["--temperature", "0"]).temperature == 0 and sample._arguments(["--model", "d", "--temperature", "0.0"]).temperature == 0
    real = sample._arguments(["--model", "d", "--max-tokens", "8", "--max-model-len", "4096", "--messages", "-", "--messages", "chat.json"])
    assert (real.max_new_tokens, real.limit_given, real.limit_flag, real.max_model_len, real.messages) == (8, True, "--max-tokens", 4096, ["-", "chat.json"])
    for spelling in (["--max-new-tokens", "8"], ["--max-new-tokens=8"]):
        older = sample._arguments(["--model", "d", *spelling])
        assert (older.max_new_tokens, older.limit_given, older.limit_flag) == (8, True, "--max-new-tokens"), "one flag, two spellings"
    written = sample._arguments(["--model", "d", "--text-out", "answer.txt", "--no-stream"])
    assert (written.text_out, written.no_stream) == ("answer.txt", True)
    for wrong, sentence in (
        (["--max-tokens", "8"], "--max-tokens is for --model; the seeded run decodes --force-tokens N (default 32)"),
        (["--max-new-tokens", "8"], "--max-new-tokens is for --model; the seeded run decodes --force-tokens N (default 32)"),
        (["--model", "d", "--force-tokens", "4", "--max-tokens", "8"], "--force-tokens N decodes exactly N tokens; --max-tokens does not apply beside it"),
        (["--model", "d", "--max-tokens", "0"], "--sequences, --layers, --max-tokens, --max-model-len and --force-tokens take positive integers"),
        (["--max-model-len", "0"], "--sequences, --layers, --max-tokens, --max-model-len and --force-tokens take positive integers"),
        (["--max-model-len", "many"], "argument --max-model-len: invalid int value: 'many'"),
        (
            ["--model", "d", "--temperature", "0.7"],
            "--temperature 0.7: token choice is greedy, the token of the largest logit at every step; only 0 is accepted",
        ),
        (["--temperature", "1"], "--temperature 1: token choice is greedy"),
        (["--model", "d", "--messages", "-", "--messages", "-"], "--messages - reads one request from standard input; it was given 2 times"),
        (["--text-out", "answer.txt"], "--text-out and --no-stream need --model: the seeded model's output is not language and is written nowhere"),
        (["--text-out", ""], "--text-out and --no-stream need --model: the seeded model's output is not language and is written nowhere"),
        (["--no-stream"], "--text-out and --no-stream need --model: the seeded model's output is not language and is written nowhere"),
        (["--messages", "-"], "--prompt and --messages need --model"),
        (["--model", "d", "--port", "8000"], "unrecognized arguments: --port 8000"),
    ):
        with pytest.raises(SystemExit) as stop:
            sample._arguments(wrong)
        assert stop.value.code == 2 and sentence in capsys.readouterr().err, wrong
    with pytest.raises(SystemExit) as stop:
        sample._arguments(["--help"])
    shown = " ".join(capsys.readouterr().out.split())  # argparse wraps its lines by the terminal's width
    assert stop.value.code == 0
    for part in (
        # Python 3.13 prints the metavar of an option with two spellings once: "--max-tokens, --max-new-tokens N"
        "--max-tokens N, --max-new-tokens N" if sys.version_info < (3, 13) else "--max-tokens, --max-new-tokens N",
        "the limit of a request that names none of its own",
        "--max-model-len N",
        "the positions a prompt and its new tokens share (default 2051)",
        "--temperature 0",
        "from a file or from standard input (-)",
        "a field that would change the output is an error that names it",
        "--text-out FILE",
        "--no-stream",
        "The generated text goes to standard output; every other line of a run goes to standard error.",
    ):
        assert part in shown, part
    with pytest.raises(ValueError, match=r"^--text-out answer.txt: cannot be written there$"):
        sample._writable("--text-out", "/nowhere/at/all/answer.txt")
    for flag in ("--text-out", "--dump-tokens", "--dump-census"):  # --text-out "$OUT" with the variable unset
        for empty in ("", "  "):
            with pytest.raises(ValueError, match=rf"^{flag}: needs a file name; it was given an empty one$"):
                sample._writable(flag, empty)


# --------------------------------------------------------------------------- the sample on the CPU, on a checkpoint generated here

WORDS = 271  # the words of the generated tokenizer; the end token is the id after them, and the model's vocabulary holds both


def _tiny_model():
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=("linear_attention", "full_attention"),
        hidden=64,
        vocab=WORDS + 1,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=2,
        attn_kv_heads=1,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _save_safetensors(path: Path, tensors: dict) -> None:
    """One shard written from the format's definition: eight bytes of little-endian header length, the JSON header padded
    with spaces to a multiple of eight bytes, the tensors' bytes in header order."""
    spelling = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name in sorted(tensors):
        tensor = tensors[name].contiguous()
        raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
        header[name] = {"dtype": spelling[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def _checkpoint_directory(directory: Path, generation=None, drafter=False) -> Path:
    """A whole checkpoint directory of the tiny geometry, generated here: seeded weights in the layout a quantised
    checkpoint stores (the language model under one prefix, the output head under none; 8-bit codes with their scale,
    packed 4-bit codes with block scales and a global scale), ``config.json``, and its text files -- a tokenizer of
    ``WORDS`` words ``w0 w1 ...`` with the end token after them, a chat template that writes the messages' contents, and
    ``generation_config.json`` with the checkpoint's own sampling settings (``generation``).  ``drafter``: the checkpoint
    carries a drafter -- its own tensors in bf16 under ``mtp.`` and the two entries of ``config.json`` that describe it."""
    pytest.importorskip("safetensors")
    pytest.importorskip("jinja2")
    tokenizers = pytest.importorskip("tokenizers")
    from model_samples import Checkpoint
    from tokenizers import models, pre_tokenizers

    model = _tiny_model()
    geometry, source, tensors = model.geometry, Checkpoint.synthetic(model, seed=5), {}
    for spec in model.tensor_specs():
        key = spec.name if spec.name.startswith("lm_head.") else "model.language_model." + spec.name
        stored = source.load(spec.name, "cpu")
        if spec.fmt == "scalar":
            tensors[key] = torch.tensor(stored, dtype=torch.float32)
        elif spec.fmt == "fp8":
            tensors[key], tensors[key + "_scale"] = stored.codes.view(torch.float8_e4m3fn), torch.tensor(stored.scale, dtype=torch.float32)
        elif spec.fmt == "nvfp4":
            tensors[key], tensors[key + "_scale"] = stored.codes, stored.block_scales.view(torch.float8_e4m3fn)
            tensors[key + "_scale_2"] = torch.tensor(stored.scale, dtype=torch.float32)
        else:
            tensors[key] = stored
    if drafter:
        from model_samples.qwen_dense import QwenDenseDrafter

        own = QwenDenseDrafter.seeded(model, seed=5)
        drawn = Checkpoint.synthetic(own, seed=5)
        tensors.update({"mtp." + spec.name[len("drafter.") :]: drawn.load(spec.name, "cpu") for spec in own.tensor_specs() if spec.name.startswith("drafter.")})
    directory.mkdir(parents=True, exist_ok=True)
    _save_safetensors(directory / "model.safetensors", tensors)
    config = {
        "model_type": "qwen3_5_text",
        "layer_types": list(geometry.layer_kinds),
        "hidden_size": geometry.hidden,
        "vocab_size": geometry.vocab,
        "intermediate_size": geometry.mlp,
        "linear_num_key_heads": geometry.la_key_heads,
        "linear_num_value_heads": geometry.la_value_heads,
        "linear_key_head_dim": geometry.la_head_dim,
        "linear_value_head_dim": geometry.la_head_dim,
        "linear_conv_kernel_dim": geometry.la_conv_width,
        "num_attention_heads": geometry.attn_heads,
        "num_key_value_heads": geometry.attn_kv_heads,
        "head_dim": geometry.attn_head_dim,
        "rope_parameters": {"partial_rotary_factor": geometry.rotary_dim / geometry.attn_head_dim, "rope_theta": geometry.rope_theta},
        "rms_norm_eps": geometry.norm_eps,
        "attn_output_gate": True,
        "hidden_act": "silu",
        **({"mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False} if drafter else {}),
    }
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": config}))
    tokenizer = tokenizers.Tokenizer(models.WordLevel({f"w{index}": index for index in range(WORDS)}, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|im_end|>"])
    assert tokenizer.token_to_id("<|im_end|>") == WORDS
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "chat_template.jinja").write_text("{% for message in messages %}{{ message.content + ' ' }}{% endfor %}")
    settings = {"do_sample": True, "temperature": 0.6, "top_p": 0.95} if generation is None else generation
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": WORDS, **settings}))
    return directory


def _sample_on_the_cpu(monkeypatch, loads=True, script="00_decode.py"):
    """A sample script as a module whose runs stay on the CPU: stated device facts (compute capability 10.0), every
    session on the CPU, and the environment check answered.  With ``loads`` false a load is an ``AssertionError``: what
    such a test looks at comes before it."""
    pytest.importorskip("cudnn")
    from model_samples import blocks

    model_samples = importlib.import_module("model_samples")  # the package as an object: its names are patched below
    real_session = model_samples.Session

    def session(model, text, **options):
        return real_session(model, text, **dict(options, device="cpu"))

    def no_load(self):
        raise AssertionError("reached a load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (80 * GIB, 180 * GIB))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 180 * GIB, 80 * GIB)))
    monkeypatch.setattr(model_samples, "check_environment", lambda: model_samples.Environment("1.33.0", "1.33.0", True, "0123456789ab", "2.13.0", "4.8.0"))
    if not loads:
        monkeypatch.setattr(real_session, "load", no_load)
    monkeypatch.setattr(model_samples, "Session", session)
    return _sample_script(monkeypatch, script)


def _lines(text, start):
    return [line for line in text.splitlines() if line.startswith(start)]


def test_the_text_is_written_between_the_steps_of_the_loop_and_every_other_line_goes_to_standard_error(tmp_path, monkeypatch, capsys):
    """The sample's own loop function on a CPU session with a checkpoint's text side: the pieces of sequence 0 reach the
    sink while the loop still steps, the line break after the last one, and the report after that; not streamed, the
    text is handed over once, after the report; seeded weights are written nowhere."""
    import serving_options
    from model_samples import Session, StopRule, TextIO, tie_out

    sample = _sample_script(monkeypatch)
    model = _tiny_model()
    text = TextIO.from_checkpoint(str(_checkpoint_directory(tmp_path / "tiny-checkpoint")))
    session = Session(model, text, sequences=2, stack="reference", device="cpu", capacity=32)
    session.plan_table()
    session.load()
    prompts = [text.encode("w3 w1 w4")] * 2
    events = []
    real_step, real_say = session.step, sample._say

    def step(ids):
        events.append(("step",))
        return real_step(ids)

    def say(*lines, flush=False):
        events.append(("line", " ".join(str(line) for line in lines).split("\n")[0][:24]))
        real_say(*lines, flush=flush)

    monkeypatch.setattr(session, "step", step)
    monkeypatch.setattr(sample, "_say", say)
    capsys.readouterr()
    forced = StopRule(text.eos_ids, max_new_tokens=256, force_tokens=6)
    ok, judged, produced = sample._decode(
        session, text, forced, prompts, tie_out, check=False, checked_steps=16, sink=serving_options.TextSink(stream=_Recorder(events))
    )
    captured = capsys.readouterr()
    assert (
        ok and judged == 0 and [len(ids) for ids in produced] == [6, 6] and captured.out == ""
    ), "the sink is the stand-in: nothing else writes to standard output"
    kinds = [event[0] for event in events if event[0] != "flush"]
    written = [event[1] for event in events if event[0] == "text"]
    assert kinds[0] == "line" and events[0][1].startswith("[00] decoding: 3 prompt"), "said before the loop"
    assert kinds[1 : 1 + 12] == ["text", "step"] * 5 + ["text", "text"], "a piece after every host read, before the next launch; then the line break"
    assert kinds[13:] and set(kinds[13:]) == {"line"}, "the report follows the text"
    assert "".join(written) == text.decode(produced[0]) + "\n" and written[-1] == "\n" and all(written[:-1]), "sequence 0, piece by piece, then one line break"
    assert all(kind != "text" or events[index + 1] == ("flush",) for index, kind in enumerate(event[0] for event in events)), "every piece is flushed"
    err = captured.err
    assert _lines(err, "[00] decoding: ") == [
        "[00] decoding: 3 prompt tokens through the decode step, then 6 new tokens and a traced pass; "
        "the text of sequence 0 goes to stdout as it is generated, the report follows"
    ]
    assert _lines(err, "text out ") == ["text out              sequence 0, 6 tokens, stopped after 6 tokens: written to stdout as it was generated"]
    assert "  | " not in err and text.decode(produced[0]) not in err, "the text is not repeated among the lines of the run"

    # not streamed: the same text, once, after the report
    session.reset(fresh=True)
    session.new_report()
    del events[:]
    again = sample._decode(
        session, text, forced, prompts, tie_out, check=False, checked_steps=16, sink=serving_options.TextSink(streamed=False, stream=_Recorder(events))
    )
    kinds = [event[0] for event in events if event[0] != "flush"]
    assert again[2] == produced and kinds.count("text") == 1 and kinds[-1] == "text" and events[-2] == ("text", text.decode(produced[0]) + "\n")
    assert kinds[:6] == ["line"] + ["step"] * 5 and events[-3][1].startswith("text out "), "nothing is written inside the loop"
    err = capsys.readouterr().err
    assert err.splitlines()[0].endswith("; the report follows, then the text of sequence 0 goes to stdout; the text is not streamed")
    assert _lines(err, "text out ") == ["text out              sequence 0, 6 tokens, stopped after 6 tokens: goes to stdout after this report"]

    # without a sink of the caller's: standard output itself
    session.reset(fresh=True)
    session.new_report()
    monkeypatch.setattr(sample, "_say", real_say)
    sample._decode(session, text, forced, prompts, tie_out, check=False, checked_steps=16)
    captured = capsys.readouterr()
    assert captured.out == text.decode(produced[0]) + "\n" and "[00] decoding: " in captured.err and "[00]" not in captured.out

    # a destination that stopped taking text: the loop goes on, and the report says what is missing
    session.reset(fresh=True)
    session.new_report()
    closed = serving_options.TextSink(stream=_Recorder(fail_at=3, error=OSError(errno.EPIPE, PIPE)))
    still = sample._decode(session, text, forced, prompts, tie_out, check=False, checked_steps=16, sink=closed)
    assert still[2] == produced and closed.error == PIPE
    assert _lines(capsys.readouterr().err, "text out ") == [
        f"text out              sequence 0, 6 tokens, stopped after 6 tokens: could not be written to stdout ({PIPE})"
    ]

    # what a sequence that stopped inside a character still owes: written before the line break, streamed or not
    class Owing:
        def __init__(self, stream):
            self._stream = stream

        def push(self, token):
            return self._stream.push(token)

        def close(self):
            return "<owed>"

    real_decoder = text.decoder
    monkeypatch.setattr(text, "decoder", lambda prompt_ids=None: Owing(real_decoder(prompt_ids)))
    for streamed in (True, False):
        session.reset(fresh=True)
        session.new_report()
        owed = _Recorder()
        sample._decode(session, text, forced, prompts, tie_out, check=False, checked_steps=16, sink=serving_options.TextSink(streamed=streamed, stream=owed))
        assert "".join(event[1] for event in owed.events if event[0] == "text") == text.decode(produced[0]) + "<owed>\n", streamed
    monkeypatch.setattr(text, "decoder", real_decoder)
    capsys.readouterr()

    # a later sequence count of a run: the loop writes nothing, and its line says whether the text is the first count's
    kept = _Recorder()
    first = serving_options.TextSink(stream=kept)
    session.reset(fresh=True)
    session.new_report()
    sample._decode(session, text, forced, prompts, tie_out, check=False, checked_steps=16, sink=first)
    written = list(kept.events)
    capsys.readouterr()
    for again, said in ((prompts, "the same text as"), ([text.encode("w9 w9 w9")] * 2, "ANOTHER text than")):
        session.reset(fresh=True)
        session.new_report()
        other = sample._decode(session, text, forced, again, tie_out, check=False, checked_steps=16, sink=first.later())
        assert (other[2] == produced) == (again is prompts), "another prompt, other tokens"
        err = capsys.readouterr().err
        assert err.splitlines()[0].endswith(
            "; the report follows; the text of this prompt went to stdout with the first sequence count and is not written again"
        )
        assert _lines(err, "text out ") == [
            f"text out              sequence 0, 6 tokens, stopped after 6 tokens: not written again: {said} with the first sequence count, "
            "whose text is the one that went to stdout"
        ]
    assert kept.events == written, "nothing more reached the destination"

    # seeded weights: not language; the same loop writes it nowhere, whatever sink it is handed
    seeded_text = TextIO.synthetic(model.geometry.vocab)
    seeded = Session(model, seeded_text, sequences=1, stack="reference", device="cpu", capacity=32)
    seeded.plan_table()
    seeded.load()
    unused = _Recorder()
    sample._decode(
        seeded, seeded_text, forced, [seeded_text.encode("abc")], tie_out, check=False, checked_steps=None, sink=serving_options.TextSink(stream=unused)
    )
    captured = capsys.readouterr()
    assert unused.events == [] and captured.out == ""
    assert captured.err.splitlines()[0].endswith("; the report and the text follow, the text is not streamed")
    assert _lines(captured.err, "text out ")[0].startswith("text out              sequence 0, 6 tokens (seeded weights: not language): ")


def test_a_client_body_runs_to_the_last_line_with_the_text_on_standard_output(tmp_path, monkeypatch, capsys):
    """``main()`` on the CPU, on a checkpoint directory generated here, on the reference stack: a request body as a
    client sends it, from a file and from standard input.  Standard output holds the generated text and nothing else;
    the lines of the run -- among them the decoding rule, the body's line and the first-token line, each before the
    load -- are on standard error; the body's limit is the loop's bound."""
    directory = _checkpoint_directory(tmp_path / "tiny-checkpoint")
    sample = _sample_on_the_cpu(monkeypatch)
    body = dict(CLIENT_BODY, messages=[{"role": "user", "content": "w3 w1 w4 w1 w5"}])
    (tmp_path / "body.json").write_text(json.dumps(body))
    arguments = ["--model", str(directory), "--stack", "reference", "--dump-tokens", str(tmp_path / "ids.json")]
    capsys.readouterr()
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json")])
    out, err = capsys.readouterr()
    assert code == 0 and err.splitlines()[-1] == "[00] RAN reference stack (no provider checked)", err[-3000:]
    assert str(tmp_path) not in out + err
    ids = json.loads((tmp_path / "ids.json").read_text())["runs"][0]
    assert ids["prompt_ids"] == [[3, 1, 4, 1, 5]] and 1 <= len(ids["ids"][0]) <= 6, "the body's max_completion_tokens bounds the loop"
    words = " ".join(f"w{token}" for token in ids["ids"][0] if token != WORDS)
    assert out == words + "\n" and out.strip(), "standard output: the text of sequence 0 and one line break"
    lines = err.splitlines()
    rule = "[00] greedy decoding: the token of the largest logit at every step; the checkpoint's own generation settings "
    rule += "(do_sample true, temperature 0.6, top_p 0.95 in generation_config.json) are not applied"
    said = (
        "[00] --messages body.json: a chat request body. Read: messages (1), max_completion_tokens 6. "
        'model ".../tiny-checkpoint": names the checkpoint directory. '
        "Not used: stream true (the text goes where the command line says, as it is generated unless --no-stream); "
        "seed 7 (token choice is greedy, nothing is drawn). At the value that changes nothing: n 1, top_p 1."
    )
    first = "[00] first token: a prompt goes through the decode step, one step per prompt token (no stack of this sample has a prefill kernel): "
    first += "5 decode steps before the first new token of --messages body.json, as long as 5 new tokens take. "
    first += "No text is written until then; the first token line of the report has the time"
    table = next(index for index, line in enumerate(lines) if line.startswith("served by (decided before the load"))
    load = next(index for index, line in enumerate(lines) if line.startswith("[00] stage "))
    assert lines.index(rule) < lines.index(said) < lines.index(first) < table < load, "each of the three before the table, and so before the load"
    assert any(
        line.startswith(
            "[00] decoding: 5 prompt tokens through the decode step, then up to 6 new tokens and a traced pass; the text of sequence 0 goes to stdout"
        )
        for line in lines
    )
    assert any(line.startswith("text out              sequence 0, ") and line.endswith(": written to stdout as it was generated") for line in lines)
    assert "[00] checkpoint weights, greedy decoding; ids hash " in lines[-2]

    # the same body on standard input: the same tokens, the same text
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(body)))
    code = sample.main([*arguments, "--messages", "-"])
    again_out, again_err = capsys.readouterr()
    assert code == 0 and again_out == out and json.loads((tmp_path / "ids.json").read_text())["runs"][0] == ids
    assert again_err.splitlines()[0] == "[00] --messages -: reading the request from standard input"
    assert said.replace("--messages body.json", "--messages -") in again_err.splitlines()

    # --text-out: the file holds what standard output would have held; --no-stream: once, after the report
    target = tmp_path / "answer.txt"
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json"), "--text-out", str(target)])
    filed_out, filed_err = capsys.readouterr()
    assert code == 0 and filed_out == "" and target.read_text(encoding="utf-8") == out
    assert any(line.endswith(": written to answer.txt as it was generated") for line in filed_err.splitlines())
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json"), "--no-stream"])
    late_out, late_err = capsys.readouterr()
    assert code == 0 and late_out == out
    assert any(line.endswith("; the report follows, then the text of sequence 0 goes to stdout; the text is not streamed") for line in late_err.splitlines())
    assert any(line.endswith(": goes to stdout after this report") for line in late_err.splitlines())

    # two requests in one run, each with its own limit; the flag is the limit of the one that names none
    (tmp_path / "chat.json").write_text(json.dumps([{"role": "user", "content": "w9 w9"}]))
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json"), "--messages", str(tmp_path / "chat.json"), "--max-tokens", "3"])
    both_out, both_err = capsys.readouterr()
    runs = json.loads((tmp_path / "ids.json").read_text())["runs"]
    assert code == 0 and [len(run["prompt_ids"][0]) for run in runs] == [5, 2] and runs[0]["ids"] == ids["ids"] and len(runs[1]["ids"][0]) <= 3
    assert both_out.count("\n") == 2 and both_out.startswith(out), "one text and one line break per request, in order"
    note = '[00] --messages body.json: its own "max_completion_tokens": 6 applies, not --max-tokens 3: the flag is the limit of a request that names none'
    assert note in both_err.splitlines() and "then up to 3 new tokens" in both_err and "then up to 6 new tokens" in both_err
    assert "5 decode steps before the first new token of --messages body.json, as long as 5 new tokens take; 2 of --messages chat.json. " in both_err

    # the reader of the text went away: the run ends in one ERROR line that keeps its verdict
    class Gone:
        def write(self, text):
            raise OSError(errno.EPIPE, PIPE)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", Gone())
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json")])
    monkeypatch.undo()
    last = capsys.readouterr().err.splitlines()[-1]
    assert (
        code == 1
        and last == f"[00] ERROR the generated text could not be written to stdout: {PIPE} (the run itself: RAN reference stack (no provider checked))"
    )


def test_a_request_gets_one_text_whatever_the_sequence_counts_and_no_line_of_a_run_reaches_standard_output(tmp_path, monkeypatch, capsys):
    """``main()`` on the CPU.  A list of sequence counts decodes every prompt once per count: the text of a prompt is
    written once, with the first count, and the report of a later count says whether it produced the same text.  The
    lines of a run never reach standard output, also where standard error was closed before the process started.  And
    the line of a request body names no path and prints back nothing somebody wrote into the body."""
    directory = _checkpoint_directory(tmp_path / "tiny-checkpoint")
    sample = _sample_on_the_cpu(monkeypatch)
    model = ["--model", str(directory), "--stack", "reference"]
    body = {"messages": [{"role": "user", "content": "w3 w1 w4 w1 w5"}], "max_tokens": 4}
    (tmp_path / "body.json").write_text(json.dumps(body))
    capsys.readouterr()
    assert sample.main([*model, "--messages", str(tmp_path / "body.json")]) == 0
    text = capsys.readouterr().out
    assert text.count("\n") == 1 and text.strip()

    # two sequence counts: one text on standard output, and the later count says whose it is
    assert sample.main([*model, "--messages", str(tmp_path / "body.json"), "--sequences", "1,2"]) == 0
    out, err = capsys.readouterr()
    assert out == text, "one request, one text: not one per sequence count"
    assert [line.split(": ", 1)[1] for line in _lines(err, "text out ")] == [
        "written to stdout as it was generated",
        "not written again: the same text as with the first sequence count, whose text is the one that went to stdout",
    ]
    decoding = _lines(err, "[00] decoding: ")
    assert decoding[0].endswith("; the text of sequence 0 goes to stdout as it is generated, the report follows")
    assert decoding[1].endswith("; the report follows; the text of this prompt went to stdout with the first sequence count and is not written again")
    # two prompts at two counts into a file, not streamed: the file holds each prompt's text once, in order
    target = tmp_path / "answers.txt"
    arguments = [*model, "--prompt", "w1 w2", "--prompt", "w7 w8", "--max-tokens", "3", "--sequences", "2,1"]
    assert sample.main([*arguments, "--text-out", str(target), "--no-stream", "--dump-tokens", str(tmp_path / "ids.json")]) == 0
    out, err = capsys.readouterr()
    runs = json.loads((tmp_path / "ids.json").read_text())["runs"]
    assert [(run["sequences"], run["prompt"]) for run in runs] == [(2, 0), (2, 1), (1, 0), (1, 1)], "every count decodes every prompt"
    texts = [" ".join(f"w{token}" for token in run["ids"][0] if token != WORDS) + "\n" for run in runs]
    assert out == "" and target.read_text(encoding="utf-8") == texts[0] + texts[1], "the texts of the first count, each once"
    assert [line.split(": ", 1)[1] for line in _lines(err, "text out ")] == [
        "goes to answers.txt after this report",
        "goes to answers.txt after this report",
        "not written again: the same text as with the first sequence count, whose text is the one that went to answers.txt",
        "not written again: the same text as with the first sequence count, whose text is the one that went to answers.txt",
    ]

    # standard error closed before the process started (2>&-): the lines of the run go nowhere, never into the text
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stderr", None)
        code = sample.main([*model, "--messages", str(tmp_path / "body.json")])
    out, err = capsys.readouterr()
    assert code == 0 and out == text and err == "", "print() sends a line to standard output where it is handed no stream: the sample hands it none"

    # the line of a body: no path, nothing somebody wrote into the body, no line of its own from a field's name
    forged = "x\n[00] PASS"
    odd = {**body, "model": str(directory), "user": "alice@example.com", "metadata": {"ticket": "T-1234"}, forged: 1, "chat_template_kwargs": {forged: 2}}
    (tmp_path / "odd.json").write_text(json.dumps(odd))
    assert sample.main([*model, "--messages", str(tmp_path / "odd.json")]) == 0
    out, err = capsys.readouterr()
    said = _lines(err, "[00] --messages odd.json: a chat request body. ")
    assert len(said) == 1 and 'model ".../tiny-checkpoint": names the checkpoint directory.' in said[0]
    assert str(tmp_path) not in err and "alice@example.com" not in err and "T-1234" not in err
    assert not [line for line in err.splitlines() if line.startswith("[00] PASS")] and err.splitlines()[-1] == "[00] RAN reference stack (no provider checked)"


def test_a_request_beyond_the_context_is_an_error_before_the_load(tmp_path, monkeypatch, capsys):
    """``--max-model-len``: a prompt and the new tokens somebody asked for that do not fit are one ``[00] ERROR`` line with
    exit code 1 before anything is planned or loaded -- from the flag, from the body, from ``--force-tokens``, and in the
    seeded run; the default limit is cut to the room that is left, with a line; a field that would change the output and
    a file that cannot be written end the same way."""
    directory = _checkpoint_directory(tmp_path / "tiny-checkpoint", generation={})
    sample = _sample_on_the_cpu(monkeypatch, loads=False)
    model = ["--model", str(directory), "--stack", "reference"]

    def last(arguments):
        code = sample.main(arguments)
        out, err = capsys.readouterr()
        assert out == "" and str(tmp_path) not in err and "served by" not in err, err
        return code, err.splitlines()[-1]

    capsys.readouterr()
    beyond = (
        "3 prompt tokens + 2,049 new tokens ({}) are 2,052 positions and --max-model-len is 2,051; ask for at most 2,048 new tokens, or raise --max-model-len"
    )
    assert last([*model, "--prompt", "w1 w2 w3", "--max-tokens", "2049"]) == (1, "[00] ERROR --prompt 1: " + beyond.format("--max-tokens 2049"))
    assert last([*model, "--prompt", "w1 w2 w3", "--max-new-tokens", "2049"]) == (1, "[00] ERROR --prompt 1: " + beyond.format("--max-new-tokens 2049"))
    assert last([*model, "--prompt", "w1 w2 w3", "--force-tokens", "2049"]) == (1, "[00] ERROR --prompt 1: " + beyond.format("--force-tokens 2049"))
    (tmp_path / "long.json").write_text(json.dumps({"messages": [{"role": "user", "content": "w1 w2 w3"}], "max_tokens": 30}))
    small = '[00] ERROR --messages long.json: 3 prompt tokens + 30 new tokens ("max_tokens": 30 of the request) are 33 positions and --max-model-len is 32; '
    small += "ask for at most 29 new tokens, or raise --max-model-len"
    assert last([*model, "--messages", str(tmp_path / "long.json"), "--max-model-len", "32"]) == (1, small)
    assert last([*model, "--prompt", "w1 w2", "--prompt", "w1 w2 w3", "--max-model-len", "3"]) == (
        1,
        "[00] ERROR --prompt 2: the prompt is 3 tokens and --max-model-len is 3: a prompt and its new tokens share the context, "
        "and no position is left for a new token; raise --max-model-len",
    )
    assert last(["--force-tokens", "2026"]) == (
        1,
        "[00] ERROR the seeded prompt: 26 prompt tokens + 2,026 new tokens (--force-tokens 2026) are 2,052 positions and --max-model-len is 2,051; "
        "ask for at most 2,025 new tokens, or raise --max-model-len",
    )
    (tmp_path / "sampled.json").write_text(json.dumps({"messages": [{"role": "user", "content": "w1"}], "temperature": 0.6}))
    code, line = last([*model, "--messages", str(tmp_path / "sampled.json")])
    assert code == 1 and line.startswith(
        '[00] ERROR --messages sampled.json: "temperature": 0.6 would change the output and is not served: token choice is greedy'
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert last([*model, "--messages", "-"]) == (
        1,
        "[00] ERROR --messages -: standard input is empty; pipe a request body or a list of messages into the sample",
    )
    assert last([*model, "--text-out", str(tmp_path / "no-such-folder" / "answer.txt")]) == (1, "[00] ERROR --text-out answer.txt: cannot be written there")
    # a file flag that was given an empty name (--text-out "$OUT", the variable unset) is not "no flag"
    for flag in ("--text-out", "--dump-tokens"):
        assert last([*model, "--prompt", "w1", flag, ""]) == (1, f"[00] ERROR {flag}: needs a file name; it was given an empty one")
    assert last(["--model", str(directory), "--prompt", "w1", "--dump-census", ""]) == (
        1,
        "[00] ERROR --dump-census: needs a file name; it was given an empty one",
    )
    # a standard stream that was closed before the process started: the interpreter has None in its place
    (tmp_path / "chat.json").write_text(json.dumps([{"role": "user", "content": "w1"}]))
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdin", None)
        closed_input = sample.main([*model, "--messages", "-"])
    assert (closed_input, capsys.readouterr().err.splitlines()[-1]) == (
        1,
        "[00] ERROR --messages -: standard input is closed; pipe a request body or a list of messages into the sample",
    )
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdout", None)
        closed_output = sample.main([*model, "--messages", str(tmp_path / "chat.json")])
        with pytest.raises(AssertionError, match="reached a load"):  # with a file for the text, or without language, nothing is missing
            sample.main([*model, "--messages", str(tmp_path / "chat.json"), "--text-out", str(tmp_path / "answer.txt")])
        with pytest.raises(AssertionError, match="reached a load"):
            sample.main(["--stack", "reference"])
    err = capsys.readouterr().err
    assert closed_output == 1 and err.count("[00] ERROR ") == 1 and "served by" not in err.split("[00] ERROR ", 1)[0], "said before anything is planned"
    assert "[00] ERROR standard output is closed and the generated text has nowhere to go; redirect it, or name a file with --text-out" in err.splitlines()

    # what fits reaches the load: the context holds the prompt and the limit exactly; the default is cut to the room, and says so
    for arguments in (
        [*model, "--prompt", "w1 w2 w3", "--max-tokens", "2048"],
        [*model, "--messages", str(tmp_path / "long.json"), "--max-model-len", "33"],
        ["--stack", "reference", "--force-tokens", "2025"],
    ):
        with pytest.raises(AssertionError, match="reached a load"):
            sample.main(arguments)
        assert "ERROR" not in capsys.readouterr().err
    with pytest.raises(AssertionError, match="reached a load"):
        sample.main([*model, "--prompt", "w1 w2 w3", "--max-model-len", "100"])
    err = capsys.readouterr().err
    assert "[00] --prompt 1: up to 97 new tokens, not the default 256: 3 prompt tokens leave that many positions under --max-model-len 100" in err.splitlines()
    assert (
        "[00] greedy decoding: the token of the largest logit at every step (the checkpoint's generation_config.json names no sampling setting)"
        in err.splitlines()
    )
    assert "served by (decided before the load" in err and err.index("[00] first token: ") < err.index("served by (decided before the load")


def test_the_family_readme_says_of_a_request_what_the_data_and_the_run_say():
    """The section "Coming from a serving engine": the table of fields that would change the output is the data, row for
    row; every other field the sample answers is named; the lines the README quotes are the lines a run prints; and the
    flags of the section are rows of the flag table."""
    import serving_options

    readme = (FAMILY / "README.md").read_text()
    section = readme.split("## Coming from a serving engine", 1)[1].split("\n## ", 1)[0]
    table = section.split("| field | accepted | another value is refused because |\n|---|---|---|\n", 1)[1].split("\n\n", 1)[0].splitlines()
    rows = []
    for name, (values, reason) in serving_options.NEUTRAL.items():
        accepted = ", ".join([f"`{json.dumps(value)}`" for value in values] + ["`null`"])
        rows.append(f"| `{name}` | {accepted} | {reason} |")
    assert table == rows, "one row per field of serving_options.NEUTRAL, in its order, with its values and its reason"
    for name in (*serving_options.READ, "model", *serving_options.UNUSED):
        assert f"`{name}`" in section, name
    example = json.loads(section.split("```json\n", 1)[1].split("```", 1)[0])
    said = serving_options.parse_request(example, "--messages -", checkpoint="Qwen3.8-27B-NVFP4").line()
    assert f"[00] {said}\n" in section, "the line the README quotes for its example body is the line a run prints for it"
    settings = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    assert f"[00] {serving_options.greedy_line(settings)}\n" in section
    assert f"[00] {serving_options.first_token_line([('the default prompt', 58)])}\n" in section
    with pytest.raises(ValueError) as refusal:
        serving_options.parse_request({"messages": CHAT, "temperature": 0.6}, "--messages request.json")
    assert f"[00] ERROR {refusal.value}\n" in section
    limits = serving_options.parse_request({"messages": CHAT, "max_tokens": 64}, "--messages request.json")
    assert serving_options.new_tokens(limits, 58, 2051, flag=32)[1].split(": ", 1)[1] in " ".join(section.split())
    for flag, default in (
        ("--messages FILE", "--"),
        ("--max-tokens N", f"`{serving_options.MAX_TOKENS}`"),
        ("--max-model-len N", f"`{serving_options.MAX_MODEL_LEN}`"),
        ("--temperature 0", "--"),
        ("--text-out FILE", "--"),
        ("--no-stream", "off"),
    ):
        assert f"| `{flag}` | {default} | " in readme, flag
    for document in (
        readme,
        (MODELS / "README.md").read_text(),
        (REPO / "samples" / "README.md").read_text(),
        (REPO / "docs" / "model-samples.mdx").read_text(),
    ):
        commands = [line for line in document.splitlines() if line.startswith("python samples/models/qwen_dense/00_decode.py")]
        assert commands and not any("--max-new-tokens" in line for line in commands), "the commands of the documents use the flag's first spelling"


# --------------------------------------------------------------------------- the speculative script: its configuration, its flags, its last line

SPECULATIVE = {"method": "mtp", "num_speculative_tokens": 5, "use_local_argmax_reduction": True}
#: every key of a speculative configuration the sample holds, with one value it is refused at; a refused key is refused at any
SPECULATIVE_KEYS = {
    "method": "eagle3",
    "num_speculative_tokens": 0,
    "use_local_argmax_reduction": False,
    "rejection_sample_method": "synthetic",
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": True,
    "disable_padded_drafter_batch": True,
    "parallel_drafting": True,
    "use_heterogeneous_vocab": True,
    "model": "/data/somebody/another-drafter",
    "quantization": "fp8",
    "revision": "a-branch-of-somebody",
    "code_revision": "a-branch-of-somebody",
    "draft_load_config": {"download_dir": "/data/somebody"},
    "draft_lm_head_quantization": "nvfp4",
    "max_model_len": 4096,
    "synthetic_acceptance_rates": [0.9, 0.8, 0.7, 0.6, 0.5],
    "synthetic_acceptance_length": 4.5,
    "num_speculative_tokens_per_batch_size": [[1, 8, 5]],
    "enforce_eager": True,
    "tensor_parallel_size": 2,
    "draft_tensor_parallel_size": 1,
    "moe_backend": "auto",
    "attention_backend": "auto",
    "kv_cache_dtype": "auto",
    "index_share_for_mtp_iteration": True,
    "disable_eagle_block_drop": False,
    "disable_lookahead_block_hashes": False,
    "prompt_lookup_max": 4,
    "prompt_lookup_min": 2,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000,
    "suffix_decoding_max_spec_factor": 1.0,
    "suffix_decoding_min_token_prob": 0.1,
    "dspark_draft_topk": 4,
    "xpress_topc": 2,
    "fly_window_size": 4,
    "fly_entropy_threshold": 0.3,
    "fly_entropy_top_k": 3,
}
SHOWN = "--speculative-config"


def _configured(**keys):
    """The line and the count of the default object with ``keys`` set (``None``: as JSON ``null``)."""
    import serving_options

    return serving_options.parse_speculative_config(json.dumps({**SPECULATIVE, **keys}), SHOWN)


def test_the_speculative_configuration_is_one_json_object_answered_key_by_key():
    """Every key of the object lands in exactly one table -- read, neutral, refused -- and a key no table holds is refused
    by its name: a key of this object selects what is launched, so none is printed and ignored.  The line a run prints
    holds the names of the keys and the values the tables know; a boolean is compared with a boolean only."""
    import serving_options

    read, neutral, refused = serving_options.SPECULATIVE_READ, serving_options.SPECULATIVE_NEUTRAL, serving_options.SPECULATIVE_REFUSED
    assert len(read) + len(neutral) + len(refused) == len({*read, *neutral, *refused}) == 39, "no key is in two tables"
    assert set(SPECULATIVE_KEYS) == {*read, *neutral, *refused}, "a key added to a table comes with a value this module refuses it at"
    assert (list(read), list(neutral)[:2]) == (list(SPECULATIVE), ["rejection_sample_method", "draft_sample_method"])
    for reason in (*(why for _, why in read.values()), *(why for _, why in neutral.values()), *refused.values()):
        assert reason and reason[0].islower() and not reason.endswith("."), reason
    drafted, line = serving_options.parse_speculative_config(serving_options.SPECULATIVE_DEFAULT, f"{SHOWN} (the default)")
    said = f'{SHOWN} (the default): the speculative configuration of a serving engine. Read: method "mtp", num_speculative_tokens 5, '
    assert (drafted, line) == (5, said + "use_local_argmax_reduction true.") and json.loads(serving_options.SPECULATIVE_DEFAULT) == SPECULATIVE
    # the count is any whole number from 1: which counts a stack serves is its rows' answer, at the plan
    assert [_configured(num_speculative_tokens=count)[0] for count in (1, 3, 7, 12)] == [1, 3, 7, 12]
    full = _configured(draft_sample_method="greedy", rejection_sample_method="standard", parallel_drafting=False, use_heterogeneous_vocab=None, model=None)
    assert full[1].endswith(
        'use_local_argmax_reduction true. At the value that changes nothing: draft_sample_method "greedy", rejection_sample_method "standard", '
        "parallel_drafting false."
    ), "null is the key left out, for a neutral key and for a refused one"
    # true is not 1, and a count is a whole number: nothing is converted
    for name, value in (
        ("num_speculative_tokens", True),
        ("num_speculative_tokens", 5.0),
        ("num_speculative_tokens", "5"),
        ("num_speculative_tokens", -1),
        ("use_local_argmax_reduction", 1),
        ("use_local_argmax_reduction", "true"),
        ("method", "MTP"),
        ("method", ["mtp"]),
    ):
        with pytest.raises(ValueError, match=re.escape(f'{SHOWN}: "{name}": {json.dumps(value)} is not served: ')):
            _configured(**{name: value})
    for name, value in (("enable_adaptive_verification", 0), ("parallel_drafting", "false"), ("draft_sample_method", "Greedy")):
        with pytest.raises(ValueError, match=re.escape(f'{SHOWN}: "{name}": {json.dumps(value)} would change what is launched and is not served: ')):
            _configured(**{name: value})
    # a key no table holds: refused by its name, as a JSON string -- a name is whatever the object's author wrote
    unknown = (("frobnicate", '"frobnicate"', 1), ("num_speculative_token", '"num_speculative_token"', None), ("a\n[02] PASS", '"a\\n[02] PASS"', 1))
    for name, shown, value in unknown:  # ... whatever it holds: null is a key left out for a key a table holds, not for a mistyped name
        with pytest.raises(ValueError) as refusal:
            _configured(**{name: value})
        assert str(refusal.value) == (
            f"{SHOWN}: {shown} is not a key this sample holds: a key of a speculative configuration selects what is launched, so none is ignored. "
            "Leave the key out"
        )
    # a reason is true of its key: "another speculative method" is said of no option of the method mtp or of a way to keep tokens
    other = {name for name, reason in refused.items() if reason.startswith("it is an option of another speculative method")}
    kept = {"synthetic_acceptance_rates", "synthetic_acceptance_length", "fly_window_size", "fly_entropy_threshold", "fly_entropy_top_k"}
    assert len(other) == 10 and not other & {"index_share_for_mtp_iteration", *kept} and len({refused[name] for name in kept}) == 1
    assert refused["index_share_for_mtp_iteration"].startswith("it overrides a field of the drafter's own configuration: ")
    # text that is not a JSON object
    for text, sentence in (
        ("{'method': 'mtp'}", f"{SHOWN}: not JSON ("),
        ("", f"{SHOWN}: not JSON ("),
        ("[1, 2]", f"{SHOWN}: [1, 2] is not a JSON object; a speculative configuration is one, with its keys"),
        ('"mtp"', f'{SHOWN}: "mtp" is not a JSON object; a speculative configuration is one, with its keys'),
        ("5", f"{SHOWN}: 5 is not a JSON object; a speculative configuration is one, with its keys"),
        ("null", f"{SHOWN}: null is not a JSON object; a speculative configuration is one, with its keys"),
    ):
        with pytest.raises(ValueError) as refusal:
            serving_options.parse_speculative_config(text, SHOWN)
        assert str(refusal.value).startswith(sentence) if sentence.endswith("(") else str(refusal.value) == sentence, text
    # the rule the guide states for such an object, beside the rule for a request body
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    assert (
        "A configuration object of a serving engine that selects what is launched is answered key by key in the same module; "
        "a key it does not hold is an `ERROR` that names it."
    ) in guide


@pytest.mark.parametrize("name", sorted(SPECULATIVE_KEYS))
def test_a_key_of_the_speculative_configuration_is_answered_by_the_table(name):
    """One case per key: a key that is read is held to what is accepted and must be named; a neutral key is accepted at its
    value and at ``null`` and refused by name at another; a key the table refuses is refused at any value, with its
    reason, and what it holds is not printed back."""
    import serving_options

    wrong = SPECULATIVE_KEYS[name]
    if name in serving_options.SPECULATIVE_READ:
        values, reason = serving_options.SPECULATIVE_READ[name]
        accepted = ", ".join(json.dumps(value) for value in values) or "a whole number from 1"
        with pytest.raises(ValueError) as refusal:
            _configured(**{name: wrong})
        assert str(refusal.value) == f'{SHOWN}: "{name}": {json.dumps(wrong)} is not served: {reason}. Accepted: {accepted}'
        without = {key: value for key, value in SPECULATIVE.items() if key != name}
        for text in (json.dumps(without), json.dumps({**without, name: None})):  # left out, or null: the same
            with pytest.raises(ValueError) as refusal:
                serving_options.parse_speculative_config(text, SHOWN)
            assert str(refusal.value) == f'{SHOWN}: "{name}" is not named: {reason}. Accepted: {accepted}'
        assert f"{name} {json.dumps(SPECULATIVE[name])}" in _configured()[1]
    elif name in serving_options.SPECULATIVE_NEUTRAL:
        values, reason = serving_options.SPECULATIVE_NEUTRAL[name]
        for value in values:
            drafted, line = _configured(**{name: value})
            assert drafted == 5 and line.endswith(f" At the value that changes nothing: {name} {json.dumps(value)}.")
        assert _configured(**{name: None}) == _configured(), "null is the key left out"
        accepted = ", ".join([json.dumps(value) for value in values] + ["null"])
        with pytest.raises(ValueError) as refusal:
            _configured(**{name: wrong})
        assert str(refusal.value) == (
            f'{SHOWN}: "{name}": {json.dumps(wrong)} would change what is launched and is not served: {reason}. Accepted: {accepted}, or the key left out'
        )
    else:
        reason = serving_options.SPECULATIVE_REFUSED[name]
        with pytest.raises(ValueError) as refusal:
            _configured(**{name: wrong})
        assert str(refusal.value) == f'{SHOWN}: "{name}" is not served: {reason}. Leave the key out'
        assert "somebody" not in str(refusal.value), "what somebody wrote into the key is not printed back"
        assert _configured(**{name: None}) == _configured(), "null is the key left out"


def test_the_line_before_the_load_of_a_speculative_run_states_a_rule_and_counts():
    """What a speculative run says before anything is loaded: per prompt the decode steps and the drafter's passes before
    the first token, then what every step is -- counts, never a time."""
    import serving_options

    one = serving_options.speculative_line([("the default prompt", 58)], 5)
    assert one == (
        "prompt of 58 tokens = 58 decode steps and 57 passes of the drafter before the first token; then every step is 5 draft passes and one "
        "verify step of 6 tokens, and keeps 1 to 6 of them"
    )
    two = serving_options.speculative_line([("--prompt 1", 1234), ("--messages chat.json", 1)], 1)
    assert two == (
        "prompt of 1,234 tokens (--prompt 1) = 1,234 decode steps and 1,233 passes of the drafter before the first token; "
        "prompt of 1 tokens (--messages chat.json) = 1 decode steps and 0 passes of the drafter before the first token; "
        "then every step is 1 draft pass and one verify step of 2 tokens, and keeps 1 to 2 of them"
    )
    assert not re.search(r"\d ?(?:ms|s)\b", one + two), "no time: the report measures it"


def test_the_speculative_scripts_flags_say_what_they_apply_to(tmp_path, monkeypatch, capsys):
    """The flags of ``02_speculative_decode.py``: its defaults, what each applies to and every refusal by its sentence -- a
    draft vocabulary without a checkpoint, a block of the drafter in ``--provider``, ``--dump-census``, each the usage
    error; ``--sequences`` other than 1, one ``ERROR`` line in the session's own sentence, before the plan.  And the two
    flags ``00_decode.py`` declares to refuse them by the name of this script."""
    import serving_options
    from model_samples import loop

    sample = _sample_on_the_cpu(monkeypatch, loads=False, script="02_speculative_decode.py")
    assert sample.serving_options is serving_options and sample.TAG == "[02]" and sample.STACKS == loop.STACKS
    seeded = sample._arguments([])
    assert (seeded.model, seeded.sequences, seeded.layers, seeded.force_tokens, seeded.stack) == (None, 1, 2, 32, "fe-cublas")
    assert (seeded.speculative_config, seeded.draft_vocab, seeded.max_new_tokens, seeded.max_model_len) == (None, None, 256, 2051)
    config = '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true}'
    real = sample._arguments(["--model", "d", "--draft-vocab", "ids.json", "--speculative-config", config, "--max-tokens", "8", "--stack", "native"])
    assert (real.draft_vocab, real.speculative_config, real.max_new_tokens, real.force_tokens, real.stack) == ("ids.json", config, 8, None, "native")
    for wrong, sentence in (
        (["--draft-vocab", "ids.json"], "--draft-vocab needs --model: the seeded run draws its draft vocabulary from the seed"),
        (
            ["--provider", "draft.head=torch_ref.linear"],
            "--provider selects a row of a block of the model; a block of the drafter has one row, a reference chain, in every stack",
        ),
        (
            ["--dump-census", "census.json"],
            "--dump-census: the launch records of a draft pass have no kernel census to be held against yet; 00_decode.py writes those of a decode step",
        ),
        (["--check-kernels", "--stack", "reference"], "--check-kernels is for a provider stack; --stack reference checks no kernel"),
        (["--sequences", "1,4"], "argument --sequences: invalid int value: '1,4'"),
        (["--model", "d", "--messages", "-", "--messages", "-"], "--messages - reads one request from standard input; it was given 2 times"),
        (["--max-tokens", "8"], "--max-tokens is for --model; the seeded run produces --force-tokens N (default 32)"),
        (["--model", "d", "--force-tokens", "40", "--max-tokens", "8"], "--force-tokens N produces exactly N tokens; --max-tokens does not apply beside it"),
        (["--model", "d", "--layers", "4"], "--layers is for the reduced model"),
        (["--prompt", "hi"], "--prompt and --messages need --model"),
        (["--no-stream"], "--text-out and --no-stream need --model"),
        (["--temperature", "0.5"], "--temperature 0.5: token choice is greedy"),
        (["--layers", "0"], "--layers, --max-tokens, --max-model-len and --force-tokens take positive integers"),
    ):
        with pytest.raises(SystemExit) as stop:
            sample._arguments(wrong)
        assert stop.value.code == 2 and sentence in capsys.readouterr().err, wrong
    with pytest.raises(SystemExit) as stop:
        sample._arguments(["--help"])
    shown = " ".join(capsys.readouterr().out.split())  # argparse wraps its lines by the terminal's width
    assert stop.value.code == 0
    for part in (
        "--speculative-config JSON",
        f"(default {serving_options.SPECULATIVE_DEFAULT})",
        "--draft-vocab FILE",
        "with --model, required: a JSON list of the token ids the drafter proposes from",
        "--sequences 1",
        "accepted at 1 only: this script serves one request",
        "The drafter's blocks are reference chains in every stack",
        "The generated text goes to standard output; every other line of a run goes to standard error.",
        "samples/models/qwen_dense/README.md",
    ):
        assert part in shown, part
    # what a user can get wrong beyond a flag's usage: one ERROR line each, before the plan, exit code 1
    for wrong, line in (
        (["--sequences", "2"], f"[02] ERROR --sequences 2: {loop.ONE_SEQUENCE}"),
        (["--sequences", "0"], f"[02] ERROR --sequences 0: {loop.ONE_SEQUENCE}"),
        (
            ["--force-tokens", "16"],
            "[02] ERROR --force-tokens 16: the seeded run schedules its steps to keep 0 to 5 drafted tokens in turn and needs 17 new tokens "
            "to keep every count once",
        ),
        (
            ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 5}'],
            '[02] ERROR --speculative-config: "use_local_argmax_reduction" is not named: the draft token is the largest of the listed rows of the '
            "head; without it an engine launches another set. Accepted: true",
        ),
        (["--dump-tokens", str(tmp_path / "nowhere" / "ids.json")], "[02] ERROR --dump-tokens ids.json: cannot be written there"),
        (["--model", "d", "--text-out", str(tmp_path / "nowhere" / "answer.txt")], "[02] ERROR --text-out answer.txt: cannot be written there"),
    ):
        capsys.readouterr()
        code = sample.main(wrong)
        assert (code, capsys.readouterr().err.splitlines()) == (1, [line]), wrong
    # a standard stream that was closed before the process started: no text without a place for it is said before the plan,
    # and a line of the run goes nowhere -- never to standard output; a file in an ERROR line is named by its own name
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdout", None)
        code = sample.main(["--model", "d"])
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stderr", None)
        sample._say("a line of the run")
    closed = "[02] ERROR standard output is closed and the generated text has nowhere to go; redirect it, or name a file with --text-out\n"
    assert (code, *capsys.readouterr()) == (1, "", closed)
    assert sample._reason(OSError(errno.ENOENT, "No such file or directory", "/data/somebody/request.json")) == "request.json: No such file or directory"
    assert loop.ONE_SEQUENCE == "a session with a drafter serves one sequence: one request, as the drafter rows are for"
    # 00_decode.py declares the two flags to refuse them: the usage error, exit code 2, with the name of this script
    plain = _sample_script(monkeypatch)
    for wrong, sentence in (
        (
            ["--speculative-config", config],
            "00_decode.py: error: --speculative-config: speculative decoding is the script 02_speculative_decode.py of this folder, "
            "which takes the same JSON",
        ),
        (
            ["--model", "d", "--draft-vocab", "ids.json"],
            "00_decode.py: error: --draft-vocab: speculative decoding is the script 02_speculative_decode.py of this folder, which takes the same list",
        ),
    ):
        monkeypatch.setattr(sys, "argv", ["00_decode.py"])  # the program's name in a usage error is the script's own
        with pytest.raises(SystemExit) as stop:
            plain._arguments(wrong)
        assert stop.value.code == 2 and sentence in " ".join(capsys.readouterr().err.split()), wrong
    assert (FAMILY / plain.SPECULATIVE).is_file(), "the script the sentence names is a script of this folder"


COMPARED = ("tokens", "kept state", "position", "prompt", "recorded", "bytes", "rejected rows", "drafter cache", "draft proposals")


def _compared(lines, name):
    """The line ``name`` of the comparison of a speculative run with decode steps (the first of that name in a run)."""
    return next(line for line in lines if line.startswith(f"  {name} ".ljust(20)) or line.startswith(f"  {name}  "))


def test_the_speculative_sample_runs_to_its_last_line_on_the_cpu(tmp_path, monkeypatch, capsys):
    """``main()`` of ``02_speculative_decode.py`` on the CPU, on the reference stack, on a checkpoint directory with a
    drafter that is generated here: a request body from a file and from standard input.  Standard output holds the
    generated text and nothing else; standard error every line of the run, in its order, to the verdict -- what a prompt
    and a step are before the load, the measured acceptance before the comparison, the drafter's lines among the checks.
    The tokens of the run are those of the README's library loop on the same checkpoint and list."""
    import serving_options
    from model_samples import Checkpoint, Session, TextIO, sampling
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    directory = _checkpoint_directory(tmp_path / "tiny-checkpoint", drafter=True)
    (tmp_path / "draft_vocab.json").write_text(json.dumps(list(range(0, WORDS + 1, 2))))
    sample = _sample_on_the_cpu(monkeypatch, script="02_speculative_decode.py")
    body = dict(CLIENT_BODY, messages=[{"role": "user", "content": "w3 w1 w4 w1 w5"}], max_completion_tokens=20)
    (tmp_path / "body.json").write_text(json.dumps(body))
    config = '{"method": "mtp", "num_speculative_tokens": 3, "use_local_argmax_reduction": true, "draft_sample_method": "greedy"}'
    arguments = ["--model", str(directory), "--stack", "reference", "--draft-vocab", str(tmp_path / "draft_vocab.json")]
    arguments += ["--speculative-config", config, "--dump-tokens", str(tmp_path / "ids.json")]
    windows, compared = [], sampling.accepted
    monkeypatch.setattr(sampling, "accepted", lambda window, greedy: compared(window, greedy) if windows.append(list(window)) is None else None)
    capsys.readouterr()
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json")])
    out, err = capsys.readouterr()
    lines = err.splitlines()
    assert code == 0 and lines[-1] == "[02] RAN reference stack (no provider checked)", err[-3000:]
    assert str(tmp_path) not in out + err
    drafts = [token for window in windows for token in window[1:]]
    assert len(drafts) == 3 * len(windows) > 0 and all(token % 2 == 0 for token in drafts), "the drafter proposes from the list --draft-vocab names"
    dumped = json.loads((tmp_path / "ids.json").read_text())
    (run,) = dumped["runs"]
    assert (dumped["sample"], dumped["stack"], dumped["draft_tokens"]) == ("qwen_dense/02_speculative_decode", "reference", 3)
    assert run["prompt_ids"] == [[3, 1, 4, 1, 5]] and 1 <= len(run["ids"][0]) <= 20, "the body's max_completion_tokens bounds the loop"
    ids, kept = run["ids"][0], run["kept"]
    assert all(0 <= count <= 3 for count in kept) and len(ids) <= 1 + sum(count + 1 for count in kept), "every step keeps 1 to 4 tokens"
    assert len(ids) > 1 + sum(count + 1 for count in kept[:-1]), "only the last step's tokens can lie behind the end token or the limit"
    words = " ".join(f"w{token}" for token in ids if token != WORDS)
    assert out == words + "\n" and out.strip(), "standard output: the text and one line break"
    # the tokens are those of the library loop of the family README on this checkpoint and this list
    checkpoint = Checkpoint.describe(str(directory))
    model = QwenDense.describe(checkpoint)
    drafter = QwenDenseDrafter.describe(checkpoint, model, draft_vocab=str(tmp_path / "draft_vocab.json"))
    text = TextIO.from_checkpoint(str(directory))
    session = Session(model, text, stack="reference", device="cpu", capacity=5 + 20 + 2, draft_tokens=3, drafter=drafter, checkpoint=checkpoint)
    session.load()
    new, counts = session.read(session.prefill([text.encode("w3 w1 w4 w1 w5 ")])), []
    while len(new) < len(ids):
        ((window, greedy),) = session.read(session.speculate())
        counts.append(sampling.accepted(window, greedy))
        session.accept(counts[-1])
        new += greedy[: counts[-1] + 1]
    assert new[: len(ids)] == ids and counts == kept, "the script's loop is the library's: the same tokens, the same counts"
    checkpoint.close()
    # before the load: the configuration key by key, the list, the rule and the counts of a prompt and a step
    said = (
        '[02] --speculative-config: the speculative configuration of a serving engine. Read: method "mtp", num_speculative_tokens 3, '
        'use_local_argmax_reduction true. At the value that changes nothing: draft_sample_method "greedy".'
    )
    listed = "[02] draft vocabulary draft_vocab.json: 136 ids of the model's 272, in ascending order (136 entries listed, none dropped)"
    before = "[02] " + serving_options.speculative_line([("--messages body.json", 5)], 3)
    load = next(index for index, line in enumerate(lines) if line.startswith("[02] stage "))
    table = next(index for index, line in enumerate(lines) if line.startswith("served by (decided before the load"))
    assert lines.index(said) < lines.index(listed) < lines.index(before) < table < load
    assert lines[0].endswith(" | the drafter runs on reference blocks") and "1 sequence, 3 drafted tokens a step | stack reference" in lines[1]
    assert any(line.startswith("[02] --messages body.json: a chat request body. Read: messages (1), max_completion_tokens 20.") for line in lines)
    assert any(
        line.startswith("[02] greedy decoding: the token of the largest logit at every step; the checkpoint's own generation settings") for line in lines
    )
    decoding = _lines(err, "[02] decoding: ")
    assert decoding == [
        "[02] decoding: 5 prompt tokens through the decode step, each but the first behind one pass of the drafter, then up to 20 new tokens in "
        "steps of 3 draft passes and one verify step, and a traced pass; the text of sequence 0 goes to stdout as it is generated, the report follows"
    ]
    # the report stands in front of the comparison, with the measured acceptance: no schedule on a checkpoint
    report = lines.index(next(line for line in lines if line.startswith("acceptance            kept tokens per step ")))
    against = lines.index(next(line for line in lines if line.startswith("speculative against decode (1 sequence x 4 tokens per sequence, ")))
    steps = len(kept)
    mean = f"kept tokens per step {1 + sum(kept) / steps:.2f} of 4 (1 + accepted drafted tokens / steps), {steps} steps; kept "
    assert lines[report].startswith("acceptance            " + mean) and lines[report + 1].startswith("                      accepted per draft: 1: ")
    assert report < against and "scheduled, not measured" not in err
    assert "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    assert any(line.startswith("bytes per step        model ") and " + drafter 3 x " in line for line in lines[:against])
    text_out = _lines(err, "text out ")
    ended = "ended at the end token" if ids[-1] == WORDS else f"stopped after {len(ids)} tokens"
    assert text_out == [f"text out              {len(ids)} tokens in {steps} steps, {ended}: written to stdout as it was generated"]
    assert lines.index(text_out[0]) < against
    # the comparison: a checkpoint declares no band, so its token line is reported; every other line holds, on this stack the bytes too
    assert _compared(lines, "tokens").endswith("    REPORTED") and " kept tokens equal to those of the decode steps of the same tokens (" in _compared(
        lines, "tokens"
    )
    for name in COMPARED[2:]:
        assert _compared(lines, name).endswith("    PASS"), _compared(lines, name)
    assert _compared(lines, "kept state").endswith(("    PASS", "    REPORTED"))
    assert not any(line.startswith(("  acceptance ", "  ids ")) for line in lines), "nothing was scheduled, and no plain run made a stream"
    assert "  replay            ids, logits and the drafter's proposals byte-identical on a second run from a fresh state    PASS" in lines
    checks = next(line for line in lines if line.startswith("[02] checks on "))
    assert checks.startswith(
        "[02] checks on the first 16 steps: the run against decode steps of the same tokens (tokens, kept state, position, prompt, recorded, bytes, "
        "rejected rows, the drafter's cache and proposals); then on the first 16 prompt steps and the first 8 steps against the fp64-accumulated reference"
    )
    # ... on the steps the header names: the comparison on what the run recorded (16 steps), the block check on the first 8 of them
    assert steps > 16, "the seeded drafter of this checkpoint is seldom accepted: twenty tokens are more steps than a run records"
    assert lines[against].startswith("speculative against decode (1 sequence x 4 tokens per sequence, 16 engine steps behind a prompt of 5 tokens; ")
    assert any(
        line.startswith("checks (1 sequence x 4 tokens per sequence, the first 8 verify steps; ") and line.endswith("; behind a prompt of 5 decode steps)")
        for line in lines
    )
    assert f"where the session's own run ended is not held (it ran {steps} verify steps and recorded 16)" in _compared(lines, "position")
    scale = next(line for line in lines if line.startswith("  tokens ") and "greedy tokens equal to the reference stack's; " in line)
    assert "for scale, the reference stack with ieee-fp32 accumulation is " in scale, "a checkpoint run prints a second reference"
    assert lines[-3].startswith("[02] checkpoint weights, greedy decoding; ids hash ")
    assert lines[-2] == "[02] the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)"
    assert "[02] token ids and kept counts of 1 prompt(s) written to ids.json" in lines

    # the same body on standard input: the same tokens, the same text
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(body)))
    code = sample.main([*arguments, "--messages", "-"])
    again_out, again_err = capsys.readouterr()
    assert code == 0 and again_out == out and json.loads((tmp_path / "ids.json").read_text())["runs"][0] == run
    assert again_err.splitlines()[0] == "[02] --messages -: reading the request from standard input"
    # two prompts in one run, the default configuration, not streamed, into a file: a session keeps nothing between two prompts
    target = tmp_path / "answers.txt"
    two = ["--model", str(directory), "--stack", "reference", "--draft-vocab", str(tmp_path / "draft_vocab.json"), "--dump-tokens", str(tmp_path / "ids.json")]
    code = sample.main([*two, "--prompt", "w3 w1 w4 w1 w5", "--prompt", "w9 w9", "--max-tokens", "9", "--text-out", str(target), "--no-stream"])
    both_out, both_err = capsys.readouterr()
    runs = json.loads((tmp_path / "ids.json").read_text())["runs"]
    assert code == 0 and both_out == "" and [(each["prompt"], len(each["prompt_ids"][0])) for each in runs] == [(0, 5), (1, 2)]
    texts = [" ".join(f"w{token}" for token in each["ids"][0] if token != WORDS) + "\n" for each in runs]
    assert target.read_text(encoding="utf-8") == texts[0] + texts[1] and all(len(each["ids"][0]) <= 9 for each in runs)
    assert "[02] --speculative-config (the default): the speculative configuration of a serving engine. Read: method" in both_err
    assert (
        "[02] prompt of 5 tokens (--prompt 1) = 5 decode steps and 4 passes of the drafter before the first token; prompt of 2 tokens (--prompt 2) = "
        "2 decode steps and 1 passes of the drafter before the first token; then every step is 5 draft passes and one verify step of 6 tokens, "
        "and keeps 1 to 6 of them"
    ) in both_err.splitlines()
    assert both_err.count("\nspeculative against decode (") == 2 and both_err.splitlines()[-1] == "[02] RAN reference stack (no provider checked)"
    # a request whose first new token ends it: no step ran, nothing is compared, and the line says so
    code = sample.main([*two, "--prompt", "w3 w1 w4 w1 w5", "--max-tokens", "1"])
    short_out, short_err = capsys.readouterr()
    assert code == 0 and short_out.count("\n") == 1 and "text out              1 tokens in 0 steps, " in short_err
    nothing = "speculative against decode: no step ran behind the prompt (its first new token ended the request): nothing to compare    REPORTED"
    assert nothing in short_err.splitlines() and "acceptance            no step ran" in short_err.splitlines()
    # a text that ends at an end token behind its steps is compared as any other: the eleventh new token of this run made the end token
    assert ids[10] not in ids[:10], "the token that ends the text is not produced before"
    ending = _checkpoint_directory(tmp_path / "ending", generation={"eos_token_id": ids[10], "do_sample": True}, drafter=True)
    code = sample.main(["--model", str(ending), *two[2:], "--prompt", "w3 w1 w4 w1 w5", "--max-tokens", "20"])
    ended = capsys.readouterr().err.splitlines()
    assert code == 0 and "text out              11 tokens in 10 steps, ended at the end token: written to stdout as it was generated" in ended
    assert any(line.startswith("speculative against decode (1 sequence x 6 tokens per sequence, 10 engine steps behind ") for line in ended)
    assert _compared(ended, "position").endswith("the session's own run ended there    PASS") and nothing not in ended

    # what a user of the script can get wrong about the drafter: one line each, before the plan
    code = sample.main(["--model", str(directory), "--stack", "reference"])
    last = capsys.readouterr().err.splitlines()[-1]
    assert code == 1 and last.startswith("[02] ERROR a checkpoint's drafter needs a draft vocabulary: a JSON list of the token ids its head is cut to.")
    assert last.endswith('samples/models/qwen_dense/README.md, "The checkpoint\'s drafter"')
    (tmp_path / "not_a_list.json").write_text('{"ids": [1, 2]}')
    code = sample.main(["--model", str(directory), "--draft-vocab", str(tmp_path / "not_a_list.json")])
    last = capsys.readouterr().err.splitlines()[-1]
    assert code == 1 and last == "[02] ERROR draft vocabulary not_a_list.json: holds an object where a JSON list of token ids is expected"
    plain = _checkpoint_directory(tmp_path / "no-drafter")
    code = sample.main(["--model", str(plain), "--draft-vocab", str(tmp_path / "draft_vocab.json")])
    last = capsys.readouterr().err.splitlines()[-1]
    assert code == 0 and last.startswith("[02] SKIP checkpoint no-drafter (local) holds no drafter"), last
    # ... and about the text side: a prompt that renders to no token; a tokenizer of more ids than the model's table has rows
    code = sample.main([*two, "--prompt", ""])
    last = capsys.readouterr().err.splitlines()[-1]
    assert code == 1 and last == "[02] ERROR --prompt 1: the checkpoint's chat template and tokenizer turn it into no token; there is nothing to decode from"
    tokenizers = pytest.importorskip("tokenizers")
    tokenizers.Tokenizer(tokenizers.models.WordLevel({f"w{index}": index for index in range(300)}, unk_token="w0")).save(str(plain / "tokenizer.json"))
    code = sample.main(["--model", str(plain), "--draft-vocab", str(tmp_path / "draft_vocab.json")])
    last = capsys.readouterr().err.splitlines()[-1]
    mismatch = "the tokenizer has 300 ids, the model's embedding table 272 rows: not the tokenizer of this checkpoint"
    assert code == 1 and last == f"[02] ERROR --model no-drafter: {mismatch}"
    # a body's tools and its template options reach the checkpoint's template: both scripts decode the prompt the body asks for
    template = "{% if tools %}w7 {% endif %}{% if flag %}w8 {% endif %}{% for message in messages %}{{ message.content + ' ' }}{% endfor %}"
    (directory / "chat_template.jinja").write_text(template)
    (tmp_path / "tools.json").write_text(json.dumps(dict(body, tools=[WEATHER], chat_template_kwargs={"flag": True}, max_completion_tokens=1)))
    for script, more in ((sample, two[2:]), (_sample_on_the_cpu(monkeypatch), [*two[2:4], *two[6:]])):
        code = script.main(["--model", str(directory), *more, "--messages", str(tmp_path / "tools.json")])
        assert code == 0 and "Read: messages (1), tools (1), chat_template_kwargs (flag), max_completion_tokens 1." in capsys.readouterr().err
        assert json.loads((tmp_path / "ids.json").read_text())["runs"][0]["prompt_ids"] == [[7, 8, 3, 1, 4, 1, 5]]

    # the reader of the text went away: the run ends in one ERROR line that keeps its verdict
    class Gone:
        def write(self, text):
            raise OSError(errno.EPIPE, PIPE)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", Gone())
    code = sample.main([*arguments, "--messages", str(tmp_path / "body.json")])
    monkeypatch.undo()
    last = capsys.readouterr().err.splitlines()[-1]
    assert (
        code == 1
        and last == f"[02] ERROR the generated text could not be written to stdout: {PIPE} (the run itself: RAN reference stack (no provider checked))"
    )


def _keeps_one_more(real):
    """A loop that keeps a drafted token the comparison rejected: the count one above the compared one."""

    def accepted(window, greedy):
        return min(real(window, greedy) + 1, len(window) - 1)

    return accepted


def _emits_the_drafted_token(real):
    """A loop that emits the first rejected drafted token where the verify step chose another: the count is the compared
    one, and the token the caller reads at the last kept row is the draft behind it."""

    def accepted(window, greedy):
        kept = real(window, greedy)
        if kept + 1 < len(window):
            greedy[kept] = window[kept + 1]
        return kept

    return accepted


def _keeps_one_fewer(real):
    """A loop that keeps one drafted token fewer than the comparison accepted: every token it emits is the model's, and no
    step keeps what it was scheduled to keep."""

    def accepted(window, greedy):
        return max(real(window, greedy) - 1, 0)

    return accepted


WRONG_LOOPS = [
    (None, ()),
    (_emits_the_drafted_token, ("ids",)),
    (_keeps_one_more, ("tokens", "drafter cache", "draft proposals", "acceptance", "ids")),
    (_keeps_one_fewer, ("acceptance",)),
]


@pytest.mark.parametrize("wrong, failing", WRONG_LOOPS)
def test_a_wrong_loop_of_the_speculative_script_fails_its_line(monkeypatch, capsys, wrong, failing):
    """The seeded default run of ``02_speculative_decode.py`` through ``main()``, on the CPU, the tiny geometry and the
    reference stack -- and the same run with ONE thing wrong in its loop.  A drafted token emitted where the verify
    step chose another reaches no line of the comparison (the next window is the session's): the script's own ``ids``
    fails.  A count kept beyond the compared one fails the token line's greedy rule, the schedule and the drafter's two
    lines (the first pass of the next step takes the greedy token of a kept row for the token that follows it); one
    kept below it fails the schedule alone.  A failed line ends the run in ``[02] FAIL``, exit code 1, on this stack too."""
    from model_samples import sampling
    from model_samples.qwen_dense import QwenDense

    sample = _sample_on_the_cpu(monkeypatch, script="02_speculative_decode.py")
    monkeypatch.setattr(QwenDense, "reduced", classmethod(lambda cls, layers=2, vocab=15520: _tiny_model()))
    if wrong is not None:
        monkeypatch.setattr(sampling, "accepted", wrong(sampling.accepted))
    capsys.readouterr()
    code = sample.main(["--stack", "reference", "--force-tokens", "24"])
    out, err = capsys.readouterr()
    lines = err.splitlines()
    assert out == "", "a seeded run writes nothing to standard output"
    assert (code, lines[-1]) == ((1, "[02] FAIL") if failing else (0, "[02] RAN reference stack (no provider checked)")), err[-3000:]
    failed = [name for name in (*COMPARED, "acceptance", "ids") if _compared(lines, name).endswith("    FAIL")]
    assert tuple(sorted(failed)) == tuple(sorted(failing)), failed
    assert "acceptance            scheduled, not measured (reduction): " in err, "the script hands force=: nothing was measured"
    report, against = lines.index(next(line for line in lines if line.startswith("bytes per step "))), lines.index(
        next(line for line in lines if line.startswith("speculative against decode ("))
    )
    assert report < against, "the report is printed before the comparison: its bytes of a step follow where the session stands"
    if not failing:
        assert _compared(lines, "acceptance").endswith("every count from 0 to 5 was kept at least once    PASS")
        ids = _compared(lines, "ids")
        assert re.fullmatch(
            r"  ids               the 24 new tokens of the speculative run against the plain run's of this process: ids hash ([0-9a-f]{12}), plain \1    PASS",
            ids,
        ), ids
        assert f"[02] seeded weights: the output is not language; ids hash {ids.split('ids hash ')[1].split(',')[0]}" in lines
        assert "  bytes             the logits of " in err and _compared(lines, "bytes").endswith("    PASS"), "gated on the reference stack"


def _speculative_section() -> str:
    """The family README's section on the speculative script, as it is written there."""
    return (FAMILY / "README.md").read_text().split("\n## Speculative decoding\n", 1)[1].split("\n## ", 1)[0]


def test_the_family_readme_says_of_the_speculative_script_what_the_data_and_the_run_say(monkeypatch, capsys):
    """The section "Speculative decoding": the table of the speculative configuration is the data, row for row; its
    commands are parsed as written; the lines it quotes are the lines a run prints; its flags are the script's; its
    counts are the schedule's arithmetic and the script's constants; the two kinds of step differ in the blocks it
    names, per stack.  The area README, the list of samples and the documentation page name the script."""
    import shlex

    import serving_options
    from model_samples import blocks, loop, tie_out
    from model_samples.qwen_dense import QwenDense

    sample = _sample_script(monkeypatch, "02_speculative_decode.py")
    section = _speculative_section()
    text = " ".join(section.split())
    # the table: one row per key that is read, one per neutral key, one per reason of the refused keys -- in the data's order
    table = section.split("| key | answer | accepted | another value is refused because |\n|---|---|---|---|\n", 1)[1].split("\n\n", 1)[0].splitlines()
    rows = []
    for name, (values, reason) in serving_options.SPECULATIVE_READ.items():
        accepted = ", ".join(f"`{json.dumps(value)}`" for value in values) or "a whole number from 1"
        rows.append(f"| `{name}` | read; must be named | {accepted} | {reason} |")
    for name, (values, reason) in serving_options.SPECULATIVE_NEUTRAL.items():
        accepted = ", ".join([f"`{json.dumps(value)}`" for value in values] + ["`null`"])
        rows.append(f"| `{name}` | accepted at the value that changes nothing | {accepted} | {reason} |")
    reasons = {}
    for name, reason in serving_options.SPECULATIVE_REFUSED.items():
        reasons.setdefault(reason, []).append(f"`{name}`")
    rows += [f"| {', '.join(names)} | refused at any value | `null` | {reason} |" for reason, names in reasons.items()]
    assert table == rows, "the three tables of serving_options.py, in their order, with their values and reasons"
    assert "The key names and the values of this table were read in the public release 0.33.0 of a serving engine (2026-10-31)." in text
    assert "**The speculative configuration of a serving engine.**" in section and "A key no row holds is an `ERROR` that names it." in text
    # the commands, as they are written: each is parsed by the script, and the object of the third is the default one
    block = section.split("```bash\n", 1)[1].split("```", 1)[0].replace("\\\n", " ")
    commands = [shlex.split(line, comments=True) for line in block.splitlines() if line.startswith("python samples/models/qwen_dense/02_speculative_decode.py")]
    assert len(commands) == 3 and all(command[:2] == ["python", "samples/models/qwen_dense/02_speculative_decode.py"] for command in commands)
    default, native, real = (sample._arguments([word for word in command[2:] if word not in (">", "answer.txt")]) for command in commands)
    assert (default.model, default.stack, native.stack, default.speculative_config) == (None, "fe-cublas", "native", None)
    assert (real.model, real.stack, real.draft_vocab, real.messages, real.max_new_tokens) == (
        "/data/Qwen3.8-27B-NVFP4",
        "native",
        "draft_vocab.json",
        ["request.json"],
        256,
    )
    assert json.loads(real.speculative_config) == json.loads(serving_options.SPECULATIVE_DEFAULT) and commands[2][-2:] == [">", "answer.txt"]
    # the lines the section quotes are the lines a run prints
    assert f"[02] {serving_options.speculative_line([('the default prompt', 58)], 5)}\n" in section
    quoted = dict(json.loads(serving_options.SPECULATIVE_DEFAULT), draft_sample_method="greedy")
    assert f"[02] {serving_options.parse_speculative_config(json.dumps(quoted), '--speculative-config')[1]}\n" in section
    with pytest.raises(ValueError) as refusal:
        serving_options.parse_speculative_config('{"method": "mtp", "num_speculative_tokens": 5}', "--speculative-config")
    assert f"[02] ERROR {refusal.value}\n" in section
    assert f'"{loop.ONE_SEQUENCE}"' in text and f'"{tie_out.DRAFTER_KERNELS}"' in text
    assert '"the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)"' in text
    source = (FAMILY / "02_speculative_decode.py").read_text()
    assert "the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)" in source
    scheduled = "acceptance is scheduled, not measured: the drafted tokens of every step are replaced by the plain run's tokens for 0, 1, .., 5 positions"
    assert f'"{scheduled} in turn and by another token behind them"' in text
    # the flags of the section's table are flags the script declares, with the defaults it has
    flags = section.split("| flag | default | meaning |\n|---|---|---|\n", 1)[1].split("\n\n", 1)[0]
    with pytest.raises(SystemExit):
        sample._arguments(["--help"])
    shown = capsys.readouterr().out
    named = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*[a-z]", flags))
    assert len(named) >= 18 and [flag for flag in sorted(named) if f"{flag} " not in shown and f"{flag}\n" not in shown] == []
    assert f"| `--speculative-config JSON` | `{serving_options.SPECULATIVE_DEFAULT}` | " in flags and "| `--sequences 1` | `1` | " in flags
    assert "| `--dump-census FILE` | -- | not in this script: a usage error." in flags

    # the counts: what a schedule keeps at the default length, what it needs, and which steps a checkpoint run checks
    def scheduled(new, drafted):  # the counts the steps of a scheduled run of `new` tokens keep: 0, 1, .., drafted in turn
        kept, done = [], 1
        while done < new:
            kept.append(len(kept) % (drafted + 1))
            done += kept[-1] + 1
        return kept

    def least(drafted):  # the fewest new tokens of a run that keeps every count once
        return next(new for new in range(1, 99) if set(scheduled(new, drafted)) == set(range(drafted + 1)))

    assert scheduled(32, 5) == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3] and (least(5), least(8)) == (17, 38)
    assert "With the default 32 new tokens that is 10 steps that keep 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 drafted tokens." in text
    assert "(17 at five drafted tokens) is an `ERROR` before the plan; left to its default, the count follows the schedule." in text
    assert "then 32 new tokens in 10 steps scheduled to keep 0, 1, 2, 3, 4, 5, 0, 1, 2 and 3 drafted tokens" in text and len(scheduled(32, 3)) == 13
    assert 'three drafted tokens (`"num_speculative_tokens": 3`) passed in 13 steps' in text
    # the three checkpoint runs it reports -- native on the cuDNN Frontend rows, fe-cublas, native with its library rows: each one's counts
    # are the counts of one loop (every step in full; the first token is the prompt's)
    runs = ((204, 63, (14, 11, 11, 9, 10, 8), (49, 38, 27, 18, 8)), (249, 74, (13, 14, 14, 10, 11, 12), (61, 47, 33, 23, 12)))
    for new, steps, kept, chain in (*runs, (231, 71, (9, 16, 20, 9, 7, 10), (62, 46, 26, 17, 10))):
        accepted = sum(index * count for index, count in enumerate(kept))
        assert sum(kept) == steps and [sum(kept[index:]) for index in range(1, 6)] == list(chain) and new <= 1 + steps + accepted
        assert f"{new} new tokens in {steps} steps: {1 + accepted / steps:.2f} kept tokens a step (1 + {accepted} " in text
        assert ", ".join(map(str, kept[:-1])) + f" and {kept[-1]} steps kept one to six tokens" in text and f" {chain[0]} of {steps}" in text
    assert (sample.TRACED_STEPS, sample.CHECKED_PROMPT_STEPS, sample.CHECKED_STEPS) == (16, 16, 8)
    assert (
        "The comparison covers the first 16 steps behind the prompt, the block check the first 16 prompt steps and the first 8 steps, the traced pass 16 steps."
        in text
    )
    paragraph = (  # about the two texts, word for word; and the blocks it names: where the two kinds of step are served by other kernels
        "Every token of a speculative run after the first is a greedy token of a verify step; the first is the greedy token of the prompt's last "
        "decode step. The drafter decides how many tokens a step keeps, never which. In the stacks `native` and `fe-entries` on compute capability "
        "10.0 a verify step and a decode step launch different kernels for two blocks (the sections `plain` and `verify` of the kernel census: the "
        "linear-attention step, and the out-projection of the linear-attention layers up to 16 rows of a step); in `fe-cublas` for one, the "
        "linear-attention step. The speculative run is compared with decode steps of the same tokens at the steps the family's declared band "
        "decides. On a checkpoint, which declares no band, the comparison is reported: there a verify step and a decode step of the same tokens do "
        "not choose the same greedy token everywhere (this page has the run), so the text of a speculative run and the text of the plain script "
        "part early; each is the greedy text of its own kind of step."
    )
    assert paragraph in text
    model = QwenDense.reduced()
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 0, 80 * GIB)

    def served(stack, sequences, tokens):
        plan = blocks.plan(model.provider_table, model.regimes, facts, sequences, tokens, stack, model=model)
        keys = {}
        for _, name, _, row, _ in plan.parts():
            keys.setdefault(name, set()).add(row.key)
        return keys

    def differing(stack, sequences, tokens):
        decode, verify = served(stack, sequences, 1), served(stack, sequences, tokens)
        return {name for name in decode if decode[name] != verify[name]}

    assert differing("native", 1, 6) == differing("fe-entries", 1, 6) == {"linear_attention.step", "projection.fp8.out"}
    assert differing("fe-cublas", 1, 6) == {"linear_attention.step"}
    assert differing("native", 2, 8) == {"linear_attention.step", "projection.fp8.out"} and differing("native", 3, 6) == {
        "linear_attention.step"
    }, "16 rows, 18 rows"
    assert set(json.loads((MODELS / "model_samples" / "qwen_dense" / "kernel_census.json").read_text())["configurations"]) >= {"plain", "verify"}
    # what a run reports, and what is not here
    for sentence in (
        "The acceptance length is a figure of this run's prompts and of this draft-vocabulary list; it is not comparable with a figure made with another list.",
        'The rate line reads "not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)"',
        "`kept tokens per step X.XX of 6 (1 + accepted drafted tokens / steps), n steps; kept 1: .., .. 6: ..` and `accepted per draft: 1: x of n, 2: y of x, ..`",
        "Every step counts in full, the last one included, whose tokens behind an end token are not written.",
        "The report is printed before the checks.",
        "ends in `[02] RAN reference stack (no provider checked)`, or in `[02] FAIL`",
        "A prompt of N tokens is N decode steps and N - 1 passes of the drafter before the first token; then every step is five draft passes and one "
        "verify step of six tokens, and keeps one to six of them.",
        "**What is not here.** The drafter on target kernels: every block of it but its embedding lookup is a reference chain,",
        "More than one request: `--sequences` is accepted at 1. A rate. A count or a position per sequence. A captured step. `--dump-census`.",
        "Any GPU class but compute capability 10.0: the script has run there only, and a class it has not run on is uncovered, not passing.",
    ):
        assert sentence in text, sentence
    assert "tokens per second" not in text and not re.search(r"\d(?:\.\d+)? (?:ms|tokens/s)\b", text), "the section states no speed"
    # what ran, as the runs printed it: a figure of a run is changed with the run, here and there
    for sentence in (
        "on the 64-layer model 451 of the 484 launches of a verify step are target kernels (130 the target's own compiled kernels + 321 cuDNN Frontend "
        "entries with byte-equal results), on the two-layer model of the default run 17 of 20 (6 the target's own compiled kernels + 11 cuDNN Frontend "
        "entries with byte-equal results); on `fe-entries`, without the library rows, 370 of the 484 (49 the target's own compiled kernels + 321 cuDNN "
        "Frontend entries with byte-equal results) and 13 of 20 (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)",
        "Nor with a figure of another stack: a greedy run follows the first token its stack chooses otherwise (this section's last runs have three).",
        # a dated run names its rows, and the run of native with the rows a library serves stands beside it with its versions
        "correctness only; `native` on the cuDNN Frontend rows -- the rows it selected when these runs were made and `fe-entries` selects today).",
        "With its library rows (2026-11-20; `native` in PyTorch 2.14.0 with FlashInfer 0.7.2.post1; the same GPU class) the default run on `native` "
        "ended in `[02] PASS` with the same 32 new tokens: ids hash `11bfb9f9bcf5`, the plain run's of its process.",
        "Every gated line of the table above passed, `acceptance` with 10 of 10 steps on their scheduled count, "
        "and the kernel check passed over 10 traced steps, "
        "with 17 of 20 launches of the verify step target kernels (6 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal "
        "results; the attention layer's out-projection among them held by membership, on the library's own CUTLASS multiply). Three drafted tokens",
        'Three drafted tokens passed in 13 steps, with "kernel identity: not stated for this run".',
        "The memory sentence of the default run asked for about 5.7 GiB. Without FlashInfer installed the same command ended in `[02] SKIP "
        "flashinfer.gemm.bmm_fp8 needs flashinfer-python 0.7.2.post1, which is not installed: ...` before anything was loaded, exit code 0.",
        "**The last run with a checkpoint on the cuDNN Frontend rows** (2026-11-19;",
        "correctness only; `native` selected those rows then, as `fe-entries` does today; the third command of this section as it is written,",
        "**The last run with a checkpoint on `native` with its library rows** (2026-11-20; PyTorch 2.14.0, FlashInfer 0.7.2.post1; the same checkpoint, "
        "command, prompt, list and GPU class), made twice with the same figures. Both runs ended in `[02] PASS` at the end token.",
        "accepted in 62 of 71 steps, the second in 46 of those 62, the third in 26 of 46, the fourth in 17 of 26, "
        'the fifth in 10 of 17 -- the counts of the library loop on `native` in the section "The verify step".',
        "`tokens` was reported -- 39 of 45 kept tokens were the decode steps' on the same tokens, the first that differs kept token 17.",
        "The kernel check passed over 16 traced steps, with "
        "451 of 484 launches of the verify step target kernels (130 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal "
        "results): 435 of them in the launch shape the census records (114 the target's own compiled kernels + 321 cuDNN Frontend entries with "
        "byte-equal results), the 16 out-projections of the attention layers by membership, on the library's own CUTLASS multiply; nothing is stated "
        "about the kernels of the passes. The memory sentence asked for about 32.2 GiB.",
        '| for `--stack native` on compute capability 10.x | also FlashInfer at a validated version (the section "The rows a library serves"): the '
        "verify step and the decode steps call its multiplies on the stored weights. Without it the run ends in `[02] SKIP` before anything is loaded, "
        "exit code 0; `--stack fe-entries` runs the cuDNN Frontend rows without the package |",
        "so the tokens a checkpoint run writes are held by no line of that run: the loop that writes them is the one the default run proves.",
        "the stack `reference` serves any count (the seeded run as far as the new tokens its schedule needs fit the context length).",
        "a prompt of 26 tokens, then 32 new tokens in 10 steps scheduled to keep 0, 1, 2, 3, 4, 5, 0, 1, 2 and 3 drafted tokens -- ended in `[02] PASS` on "
        "`fe-cublas`, `native` and `fe-entries` and in `[02] RAN reference stack (no provider checked)` on `reference`.",
        "In each of the four the 32 new tokens were the plain run's of the same process (ids hash `11bfb9f9bcf5`, which is also what `00_decode.py "
        "--sequences 1 --force-tokens 32 --stack native` printed), nothing was written to standard output, and every gated line of the table above "
        "passed, `acceptance` with 10 of 10 steps on their scheduled count. `bytes` passed on `reference` and was reported on the other three.",
        "On `native` and `fe-entries` the kernel check passed over 10 traced steps, with 13 of 20 launches of the verify step target kernels "
        "(2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results).",
        "eight ended in `[02] SKIP cudnn.GdnVerifyFused serves 8 tokens per sequence of 2-8, sequences 1-4; this run has 1 sequences x 9 tokens per "
        "sequence` before anything was loaded. The four-layer model (`--layers 4`, on the default stack) passed.",
        "asked for about 6.3 GiB on `fe-cublas` and `reference` and about 6.1 GiB on `native` and `fe-entries`; PyTorch reserved between 3.8 and 4.6 GiB "
        "in the four runs, and the device grew by about 0.2 GiB more in each.",
        "A drafted token written out where the verify step chose another failed `ids` alone; one drafted token kept beyond the compared count failed "
        "`tokens`, `drafter cache`, `draft proposals`, `acceptance` and `ids`; one kept below it failed `acceptance` alone.",
        "58 tokens through the prompt, five drafted tokens, the draft vocabulary of this README's recipe with 65,536 ids). Both runs ended in `[02] PASS`, "
        "and both stopped at the end token.",
        "the last step kept six tokens and the end token was its fourth, so the last two of them were not written",
        "the second in 38 of those 49, the third in 27 of 38, the fourth in 18 of 27, the fifth in 8 of 18",
        "on `native` 38 of 46 kept tokens were the decode steps' on the same tokens, the first that differs kept token 13; on `fe-cublas` 43 of 49, the "
        "first kept token 30.",
        "On `native` the kernel check passed over 16 traced steps, with 370 of 484 launches of the verify step target kernels "
        "(49 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results) and nothing stated",
        "asked for about 43.3 GiB on `native`, where PyTorch reserved 39.8 GiB and the device grew by 40.0 GiB, and for about 61.0 GiB on `fe-cublas`, "
        "where PyTorch reserved 58.5 GiB and the device grew by 58.7 GiB; the GPU was otherwise empty at the start of each.",
    ):
        assert sentence in text, sentence
    # the other documents name the script, and their commands run its first form
    family = (FAMILY / "README.md").read_text()
    assert "| `02_speculative_decode.py` | speculative decoding with the checkpoint's drafter, one request in one process: " in family
    assert (
        "| a speculative configuration: the JSON object that asks an engine to draft tokens | `02_speculative_decode.py --speculative-config JSON` | " in family
    )
    area = " ".join((MODELS / "README.md").read_text().split())
    assert "The second sample, `02_speculative_decode.py`, runs that step for one request:" in area and "no script drafts yet" not in area + family
    assert "No block of the drafter is on a kernel yet, and the sample prints no rate." in area
    for document in ((MODELS / "README.md").read_text(), (REPO / "samples" / "README.md").read_text(), (REPO / "docs" / "model-samples.mdx").read_text()):
        assert "python samples/models/qwen_dense/02_speculative_decode.py" in document and "`02_speculative_decode.py`" in document
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    for lesson in (
        '- A configuration that selects what is launched has no class "printed and ignored".',
        "- A script that compares where another checks nothing needs the verdict's order.",
        "- What the library's comparison cannot see, the script holds itself.",
        "- A default follows what it is the default of. A scheduled run needs enough new tokens to keep every count once: 17 at five drafted tokens, 38 at eight.",
        "- A flag that is declared in order to be refused is still a string of the script.",
        "- A refusal belongs where the object is built.",
        "- A stand-in that rounds otherwise is not given the product's band.",
    ):
        assert lesson in guide, lesson


def test_the_speculative_loop_writes_the_tokens_a_step_kept_and_ends_inside_a_step_at_an_end_token(tmp_path, monkeypatch, capsys):
    """The loop function of ``02_speculative_decode.py`` on a CPU session with a checkpoint's text side and scheduled
    drafts (its steps keep 0, 1, 2 drafted tokens): the tokens a step kept reach the sink between that step's host read
    and the next launch; an end token inside a step ends the text there, and the step still counts in full."""
    import serving_options
    from model_samples import Session, StopRule, TextIO, sampling, tie_out
    from model_samples.qwen_dense import QwenDenseDrafter

    sample = _sample_script(monkeypatch, "02_speculative_decode.py")
    model = _tiny_model()
    text = TextIO.from_checkpoint(str(_checkpoint_directory(tmp_path / "tiny-checkpoint")))  # a checkpoint's tokenizer: its text is language
    common = dict(sequences=1, stack="reference", device="cpu", capacity=5 + 12 + 4)
    plain = Session(model, text, **common)
    session = Session(model, text, share=plain, draft_tokens=5, drafter=QwenDenseDrafter.seeded(model), **common)
    plain.plan_table([session])
    plain.load()
    session.load()
    prompt = text.encode("w3 w1 w4 w1 w5")
    ids = plain.prefill([prompt])
    stream = plain.read(ids)
    for _ in range(12 + 5 - 2):
        ids = plain.step(ids)
        stream += plain.read(ids)
    forced = torch.tensor([[tie_out.forced_drafts(stream[done:], length, 5, WORDS + 1) for length in range(6)] for done in range(1, 12)], dtype=torch.int32)
    # the schedule: the first token from the prompt, then steps that emit new tokens 2 | 3, 4 | 5, 6, 7: token 6 stands inside the third step
    end = stream[5]
    assert end not in stream[:5], "the token that ends the text is not produced before"
    events = []
    real_step, real_say = session.speculate, sample._say
    monkeypatch.setattr(session, "speculate", lambda force=None: real_step(force) if events.append(("step",)) is None else None)
    monkeypatch.setattr(sample, "_say", lambda *lines, flush=False: real_say(*lines, flush=flush) if events.append(("line",)) is None else None)
    capsys.readouterr()
    sink = serving_options.TextSink(stream=_Recorder(events))
    produced, kept, scheduled = sample._speculate(session, text, StopRule([end], max_new_tokens=12), prompt, sampling, forced, sink)
    assert produced == stream[:6] and kept == scheduled == [0, 1, 2], "three steps, each counted in full; the seventh token was kept and is not written"
    kinds = [event[0] for event in events if event[0] != "flush"]
    order = "line text step text step text text step text text"
    assert " ".join(kinds[:10]) == order, "a step's tokens behind its host read, in front of the next launch"
    assert kinds[10] == "text" and set(kinds[11:]) == {"line"}, "the line break, then the report"
    written = [event[1] for event in events if event[0] == "text"]
    assert "".join(written) == text.decode(stream[:6]) + "\n" and written[-1] == "\n"
    err = capsys.readouterr().err
    assert _lines(err, "text out ") == ["text out              6 tokens in 3 steps, ended at the end token: written to stdout as it was generated"]
    assert "acceptance            scheduled, not measured (reduction): 3 steps; kept 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0" in err
    assert session.record.count == 3 and session.record.forced == 3 and session.state.position == 5 + 1 + 2 + 3, "the state stands behind the whole third step"
    # the limit of new tokens ends a text inside a step as well: five tokens are the first, one, two and one of the third step's three
    session.reset(fresh=True)
    session.new_report()
    limited = sample._speculate(session, text, StopRule([WORDS], max_new_tokens=5), prompt, sampling, forced, serving_options.TextSink(stream=_Recorder()))
    assert limited[0] == stream[:5] and limited[1] == [0, 1, 2]
    said = _lines(capsys.readouterr().err, "text out ")
    assert said == ["text out              5 tokens in 3 steps, stopped after 5 tokens: written to stdout as it was generated"]
