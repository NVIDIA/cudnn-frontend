# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the text side of the model samples (``samples/models/model_samples/text_io.py``): the synthetic
tokenizer's round trip and its stream; chat requests read from JSON files; and, where ``jinja2`` and ``tokenizers`` are
installed, the text side of a checkpoint directory built here: a chat template with the constructs a checkpoint's template
uses (tool schemas, a reasoning block, tool calls and tool results, several turns, an option, a refusal) rendered against
stored strings, and a tokenizer trained in the test, encoded, decoded and streamed.  The requirement file of the area and
its dependency-update entry are checked as text.  No real tokenizer or template file is committed; no GPU, no weights.

The tests that need ``jinja2`` or ``tokenizers`` skip visibly where a package is missing."""

import json
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.L0]

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "samples" / "models"

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
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _text_directory(directory: Path, template: str = TEMPLATE) -> Path:
    """The text files of a checkpoint directory, built here: a byte-level BPE tokenizer trained on a few sentences, with the
    turn markers as special tokens and the reasoning markers as ordinary added tokens; a chat template; the end tokens."""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    corpus = ["why is the sky blue", "the sky is blue because air scatters blue light", "system user assistant tool", "weather in the city today"]
    tokenizer.train_from_iterator(corpus * 4, trainer)
    tokenizer.add_tokens([tokenizers.AddedToken("<think>", special=False), tokenizers.AddedToken("</think>", special=False)])
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
        pushed = [stream.push(i) for i in ids.tolist()]
        assert "".join(pushed) + stream.close() == sample, "the streamed pieces add up to the full decode"
    stream = text.decoder()
    assert [stream.push(i) for i in "é".encode("utf-8")] == ["", "é"], "an open multi-byte character gives no text yet"
    produced = [70, 260, 280, 299]  # a byte, two seeded pieces, the end token
    stream = text.decoder()
    assert "".join(stream.push(i) for i in produced) + stream.close() == text.decode(produced) and text.decode(produced).endswith("<|end|>")
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
    ):
        (tmp_path / name).write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match=re.escape(sentence)) as refusal:
            load_messages(str(tmp_path / name))
        assert f"--messages {name}:" in str(refusal.value) and str(tmp_path) not in str(refusal.value), "the file's own name, never a path"


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
        "<|im_start|>assistant\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        '<|im_start|>user\n<tool_response>\n{"sky": "clear"}\n</tool_response>\n<tool_response>\n{"wind": "calm"}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n<think>\n"
    ), "tool schemas, a tool call and two tool results in one turn: all of it is text by the time it is encoded"
    with pytest.raises(ValueError, match="chat template refused the messages: Unexpected message role"):
        text.render_chat([{"role": "narrator", "content": "x"}])
    with pytest.raises(ValueError, match="refused the messages: No messages provided"):
        text.render_chat([])
    named = _text_directory(tmp_path / "named", template="{{ eos_token }}|{{ pad_token }}|{{ messages[0].content }}")
    assert TextIO.from_checkpoint(str(named)).render_chat(user) == "<|im_end|>|<|endoftext|>|Why is the sky blue?", "a template may name the special tokens"


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
        pushed = [stream.push(i) for i in produced]
        assert "".join(pushed) + stream.close() == text.decode(produced) == sample, "the streamed pieces add up to the full decode"
    stream = text.decoder()
    assert [stream.push(i) for i in text.encode("é").tolist()] == ["", "é"], "an open multi-byte character gives no text yet"
    stream = text.decoder()
    assert stream.push(end) == "" and stream.push(int(text.encode("blue")[0])) != "", "the end token adds nothing to the text"
    (directory / "generation_config.json").unlink()
    (directory / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 7}}))
    assert TextIO.from_checkpoint(str(directory)).eos_ids == (7,), "without a generation config the end token is the model config's"


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
    (directory / "generation_config.json").unlink()
    with pytest.raises(ValueError, match="no end-token id there"):
        TextIO.from_checkpoint(str(directory))
    (directory / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="--model checkpoint: no tokenizer.json there"):
        TextIO.from_checkpoint(str(directory))
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
