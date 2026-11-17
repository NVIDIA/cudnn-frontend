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
    "use_beam_search": True,
    "frequency_penalty": 0.5,
    "presence_penalty": -1,
    "repetition_penalty": 1.1,
    "logit_bias": {"17": -100},
    "stop": ["\n\n"],
    "stop_token_ids": [7],
    "ignore_eos": True,
    "min_tokens": 8,
    "logprobs": True,
    "top_logprobs": 5,
    "echo": True,
    "skip_special_tokens": False,
    "response_format": {"type": "json_object"},
    "structured_outputs": {"regex": "a+"},
    "guided_json": {"type": "object"},
    "guided_regex": "a+",
    "guided_choice": ["yes", "no"],
    "guided_grammar": "root ::= answer",
    "tool_choice": "required",
    "add_generation_prompt": False,
    "continue_final_message": True,
    "chat_template": "{{ messages[0].content }}",
    "documents": [{"text": "a page"}],
    "truncate_prompt_tokens": 128,
    "reasoning_effort": "low",
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
        'model "some-org/tiny-checkpoint": names the checkpoint directory. '
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
        "frobnicate": [1, 2],
        "audio": None,
        "seed": None,
    }
    request = serving_options.parse_request(full, "--messages full.json")
    assert (request.tools, request.options, request.max_tokens, request.limit_field) == ([WEATHER], {"enable_thinking": False}, 5, "max_tokens")
    line = request.line()
    assert "Read: messages (1), tools (1), chat_template_kwargs (enable_thinking), max_tokens 5, max_completion_tokens 5." in line
    assert 'Not used: user "u-1" (one process: there is no account); service_tier "auto" (one process: there is no queue).' in line
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


def test_a_request_is_read_from_a_file_or_from_standard_input(tmp_path):
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


def test_where_the_generated_text_goes(tmp_path):
    """Standard output or a file, piece by piece or once; a destination that stops taking text is remembered and not
    written to again."""
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


def _sample_script(monkeypatch):
    """The sample script as a module, for its own functions; what loading it changes in the process leaves with the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script puts samples/models and its own folder on the path
    monkeypatch.setenv("KINETO_LOG_LEVEL", os.environ.get("KINETO_LOG_LEVEL", "6"))  # and sets a default for the profiler's log level
    spec = importlib.util.spec_from_file_location("model_sample_qwen_dense_00_decode_serving_options", FAMILY / "00_decode.py")
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
        "--max-tokens N, --max-new-tokens N",
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


def _checkpoint_directory(directory: Path, generation=None) -> Path:
    """A whole checkpoint directory of the tiny geometry, generated here: seeded weights in the layout a quantised
    checkpoint stores (the language model under one prefix, the output head under none; 8-bit codes with their scale,
    packed 4-bit codes with block scales and a global scale), ``config.json``, and its text files -- a tokenizer of
    ``WORDS`` words ``w0 w1 ...`` with the end token after them, a chat template that writes the messages' contents, and
    ``generation_config.json`` with the checkpoint's own sampling settings (``generation``)."""
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


def _sample_on_the_cpu(monkeypatch, loads=True):
    """The sample script as a module whose runs stay on the CPU: stated device facts (compute capability 10.0), every
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
    return _sample_script(monkeypatch)


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
        'model "some-org/tiny-checkpoint": names the checkpoint directory. '
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
