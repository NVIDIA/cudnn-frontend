# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 02: speculative decoding of the dense Qwen 3.5-architecture model with the checkpoint's drafter, in one process.

One request, one sequence.  The prompt goes through decode steps, each but the first behind one pass of the drafter; then
every step is the drafter's passes -- five by default -- one verify step of the drafted tokens behind the token kept last,
and ONE host read, after which the loop of this script counts the drafted tokens the model would have chosen itself and
keeps them.  Every token after the first is a greedy token of a verify step; the drafter decides how many tokens a step
keeps, never which.  Every block of the drafter is a reference chain in every stack (its embedding lookup is PyTorch's
gather): this script states no speed and prints no rate.

The default run needs no checkpoint and no extra package: the reduced seeded model with its seeded drafter.  Seeded
weights draft nothing a model accepts, so the run schedules: a plain session decodes the token stream first, and the
drafted tokens of every step are replaced by that stream's tokens for 0, 1, .., 5 positions in turn and by another token
behind them (the drafter still runs every pass; the kept count is the comparison's, never the schedule's).  The run is held
against decode steps of the same tokens line by line -- the tokens and the greedy rule, the kept state, the position, the
prompt byte for byte, the rows behind a rejected draft, the drafter's cache and proposals, every count kept -- and then
by the checks of every run (replay, every provider block of the model, the kernels of the traced steps).  It prints
``[02] PASS``, or ``[02] SKIP <requirement>`` with exit code 0 before the first load.

With ``--model DIR`` the session with the drafter alone produces the text, to standard output as it is generated; the
plain session is fed the prompt and the kept tokens of the first steps afterwards, for the comparison.  The drafter needs
a draft vocabulary, ``--draft-vocab FILE``: a JSON list of token ids (the family README has a recipe).  The report states
how many tokens a step kept; that figure belongs to the run's prompts and to its list.  On a checkpoint a verify step and
a decode step of the same tokens do not choose the same greedy token everywhere, so this script and ``00_decode.py``
print different texts: each is the greedy text of its own kind of step, and the comparison of the two is reported.

A request and its options: ``--speculative-config`` takes the JSON object that says how tokens are drafted, key by key
(``serving_options.py``): a key is read, accepted at the value that changes nothing, or refused by name -- none is
ignored, because each selects what is launched.  ``--messages``, ``--max-tokens`` and ``--max-model-len`` are those of
``00_decode.py``.

    python samples/models/qwen_dense/02_speculative_decode.py
    python samples/models/qwen_dense/02_speculative_decode.py --stack native      # the verify step on the stack's kernels
    python samples/models/qwen_dense/02_speculative_decode.py --model /data/Qwen3.8-27B-NVFP4 --stack native \\
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 5, "use_local_argmax_reduction": true}' \\
        --draft-vocab draft_vocab.json --messages request.json --max-tokens 256 > answer.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_START = time.perf_counter()
os.environ.setdefault("KINETO_LOG_LEVEL", "6")  # the profiler's tracing library: as in 00_decode.py, unless a level is set
_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS = os.path.normpath(os.path.join(_HERE, ".."))  # samples/models: the home of the model_samples package
for _path in (_HELPERS, _HERE):  # and this folder: serving_options, the module beside the scripts of the family
    if _path not in sys.path:
        sys.path.insert(0, _path)

import serving_options  # noqa: E402  the standard library only: cudnn and PyTorch are imported in main(), in their order

TAG = "[02]"
STACKS = ("fe-cublas", "native", "fe-entries", "reference")
DEFAULT_PROMPT = "Why is the sky blue?"
CHECKED_PROMPT_STEPS = 16  # with a checkpoint: the prompt steps under the block check; the others are replayed in front of the steps
CHECKED_STEPS = 8  # ... and the steps of speculative decoding under it
TRACED_STEPS = 16  # the steps a run records for the comparison and the traced pass; the seeded run records every step


def _say(*lines, flush: bool = False) -> None:
    """A line of the run, to standard error: standard output carries the generated text and nothing else.  Where standard
    error was closed before the process started the lines go nowhere -- never to standard output."""
    if sys.stderr is not None:
        print(*lines, file=sys.stderr, flush=flush)


def _reason(error) -> str:
    """The text of an error for an ``ERROR`` line: a file is named by its own name, never by a path."""
    if isinstance(error, OSError) and error.filename:
        return f"{os.path.basename(os.path.normpath(str(error.filename)))}: {error.strerror or type(error).__name__}"
    return str(error)


def _dump(path: str, stack: str, runs: list, flag: str = "--dump-tokens", **facts) -> str:
    """Write the prompt ids, the produced ids and the kept count of every step of a run as JSON.  Returns "" or, where the
    file cannot be written after all, the reason with the file's own name."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sample": "qwen_dense/02_speculative_decode", "stack": stack, "greedy": True, **facts, "runs": runs}, f)
    except OSError as error:
        return f"{flag} {os.path.basename(os.path.normpath(path))}: {error.strerror or type(error).__name__}"
    return ""


def _writable(flag: str, path: str) -> None:
    """A file a flag asks for that cannot be written, or a flag that was given no name, is a ``ValueError`` before the run."""
    if not path.strip():
        raise ValueError(f"{flag}: needs a file name; it was given an empty one")
    target = os.path.abspath(path)
    there = os.path.exists(target)
    if os.path.isdir(target) or not os.access(target if there else os.path.dirname(target), os.W_OK):
        raise ValueError(f"{flag} {os.path.basename(os.path.normpath(path))}: cannot be written there")


def _ids_hash(ids) -> str:
    """The ids hash of ``00_decode.py``, for one list of ids: the first 12 hex digits of a SHA-256 over them, four bytes each."""
    return hashlib.sha256(b"".join(int(i).to_bytes(4, "little") for i in ids)).hexdigest()[:12]


EPILOG = """\
The default run needs no checkpoint and no extra package; it needs a GPU of compute capability 10.x or 12.x.
With --model: pip install -r samples/models/requirements.txt first, and a draft vocabulary (--draft-vocab FILE).  The
checkpoint is your own download (nothing is fetched).  With the 27B checkpoint, a list of 65,536 ids and the default
context length a run asks for about 61 GiB of free GPU memory on the stack fe-cublas (about 64 GiB on compute capability
12.x), on the stack fe-entries for about 44 GiB on compute capability 10.0, about 48 GiB on the other 10.x classes and
about 64 GiB elsewhere, and on the stack native, whose library rows need FlashInfer and its kernel-binary package
(00_decode.py --help has the command), for about 33 GiB and about 37 GiB there: the model as 00_decode.py holds it, a state of six tokens per sequence,
and the drafter with the bf16 copies its reference blocks multiply on.  Only compute capability 10.0 has run.
The generated text goes to standard output; every other line of a run goes to standard error.
What a run proves per stack, the speculative configuration key by key and every flag: samples/models/qwen_dense/README.md
"""


def _arguments(argv):
    parser = argparse.ArgumentParser(
        description="Speculative decoding of the dense Qwen model with the checkpoint's drafter on reference blocks: the reduced "
        "seeded model (self-checking), or a checkpoint from local disk.  One request, one sequence.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", metavar="DIR", help="a checkpoint directory on local disk; never a hub name, nothing is downloaded")
    parser.add_argument(
        "--speculative-config",
        default=None,
        metavar="JSON",
        help="the speculative configuration, as the text of a JSON object; every key is answered: read, accepted "
        f"at the value that changes nothing, or an error that names it (default {serving_options.SPECULATIVE_DEFAULT})",
    )
    parser.add_argument(
        "--draft-vocab",
        metavar="FILE",
        help="with --model, required: a JSON list of the token ids the drafter proposes from (the family README has a recipe that makes "
        "one from the files --dump-tokens writes); the seeded run draws its list from the seed",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="TEXT",
        help=f'with --model: one user message; repeat for several prompts, run one after the other (default "{DEFAULT_PROMPT}")',
    )
    parser.add_argument(
        "--messages",
        action="append",
        default=[],
        metavar="FILE",
        help="with --model: a chat request as JSON, from a file or from standard input (-): a list of messages, or the request body a "
        "chat client sends, answered field by field as 00_decode.py answers it; repeatable",
    )
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        metavar="N",
        help="with --model: stop after N new tokens unless the end token comes first (default 256); the limit of a request that names "
        "none of its own. --max-new-tokens is the same flag",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=serving_options.MAX_MODEL_LEN,
        metavar="N",
        help="the positions a prompt and its new tokens share (default %(default)s); a request that asks for more is an error before anything is loaded",
    )
    parser.add_argument("--temperature", type=float, default=None, metavar="0", help="accepted at 0 only: token choice is greedy")
    parser.add_argument("--text-out", metavar="FILE", help="with --model: write the generated text to FILE instead of standard output")
    parser.add_argument("--no-stream", action="store_true", help="with --model: write the text once, after the report of its prompt")
    parser.add_argument(
        "--force-tokens",
        type=int,
        default=None,
        metavar="N",
        help="produce exactly N new tokens, ignoring the end token (default without --model: 32, or as many as the scheduled steps need "
        "to keep every count of drafted tokens once; with it: off)",
    )
    parser.add_argument("--sequences", type=int, default=1, metavar="1", help="accepted at 1 only: this script serves one request")
    parser.add_argument("--layers", type=int, default=None, metavar="N", help="layers of the reduced model (default 2); not with --model")
    parser.add_argument(
        "--stack",
        default="fe-cublas",
        choices=STACKS,
        help="the stack of the MODEL's blocks, as in 00_decode.py: fe-cublas (default), native (the verify step and the prompt's decode "
        "steps on the stack's kernels, a block no kernel serves on its fallback row; the kernel check always on), fe-entries, reference "
        "(every block in PyTorch).  The drafter's blocks are reference chains in every stack",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="BLOCK=KEY",
        help="select one declared provider row of a block of the model, in both kinds of its steps; a block of the drafter has one row",
    )
    parser.add_argument("--reference-precision", default="fp64", choices=("fp64", "ieee-fp32"), help="accumulation of the reference blocks (default fp64)")
    parser.add_argument("--dump-tokens", metavar="FILE", help="write the prompt ids, the produced ids and the kept count of every step as JSON")
    parser.add_argument(
        "--check-kernels",
        action="store_true",
        help="also check the kernels of the traced steps against the plan, as in 00_decode.py; not with --stack reference; always on in the "
        "stacks native and fe-entries",
    )
    parser.add_argument("--dump-census", metavar="FILE", help="not in this script: 00_decode.py writes the launch records of a decode step")
    args = parser.parse_args(argv)
    args.limit_flag = "--max-new-tokens" if any(word.split("=")[0] == "--max-new-tokens" for word in argv) else "--max-tokens"
    args.limit_given = args.max_new_tokens is not None
    if args.dump_census is not None:
        parser.error("--dump-census: not in this script; 00_decode.py writes the launch records of a decode step")
    if any(pair.startswith("draft.") for pair in args.provider):
        parser.error("--provider selects a row of a block of the model; a block of the drafter has one row, a reference chain, in every stack")
    if args.check_kernels and args.stack == "reference":
        parser.error("--check-kernels is for a provider stack; --stack reference checks no kernel")
    if args.model is None and args.draft_vocab is not None:
        parser.error("--draft-vocab needs --model: the seeded run draws its draft vocabulary from the seed")
    if args.model is None and (args.prompt or args.messages):
        parser.error("--prompt and --messages need --model: the seeded model has no language")
    if args.model is None and (args.text_out is not None or args.no_stream):
        parser.error("--text-out and --no-stream need --model: the seeded model's output is not language and is written nowhere")
    if args.model is not None and args.layers is not None:
        parser.error("--layers is for the reduced model; a checkpoint has the depth it has")
    if args.limit_given and args.model is None:
        parser.error(f"{args.limit_flag} is for --model; the seeded run produces --force-tokens N (default 32)")
    if args.limit_given and args.force_tokens is not None:
        parser.error(f"--force-tokens N produces exactly N tokens; {args.limit_flag} does not apply beside it")
    if args.temperature is not None and args.temperature != 0:
        parser.error(f"--temperature {args.temperature:g}: token choice is greedy, the token of the largest logit at every step; only 0 is accepted")
    if args.messages.count("-") > 1:
        parser.error(f"--messages - reads one request from standard input; it was given {args.messages.count('-')} times")
    args.forced = args.force_tokens is not None  # whether the count was typed: the seeded default follows the schedule
    if args.force_tokens is None and args.model is None:
        args.force_tokens = 32
    args.max_new_tokens = 256 if args.max_new_tokens is None else args.max_new_tokens
    if (
        args.max_new_tokens < 1
        or args.max_model_len < 1
        or (args.force_tokens is not None and args.force_tokens < 1)
        or (args.layers is not None and args.layers < 1)
    ):
        parser.error("--layers, --max-tokens, --max-model-len and --force-tokens take positive integers")
    args.layers = 2 if args.layers is None else args.layers
    return args


def _verdict(stack: str, passed: bool, judged: int, unobserved: str = "") -> str:
    """The last line of a run: the verdict of ``00_decode.py``.  A failed line -- of the comparison with decode steps, of the
    checks -- is ``FAIL`` on every stack, the reference stack included: the comparison runs there too.  ``PASS`` says that
    provider blocks of the model were judged against the reference and held, and every gated line with them."""
    if not passed:
        return "FAIL"
    if stack == "reference":
        return "RAN reference stack (no provider checked)"
    if unobserved:
        return f"RAN {stack} stack (kernels not observed: {unobserved})"
    return "PASS" if judged else "RAN (no provider block was judged)"


def _speculate(session, text, rule, prompt, sampling, forced=None, sink=None):
    """One prompt through the loop of speculative decoding, the traced pass and the report.  Returns ``(the new tokens, the
    count every step kept, the count every step was scheduled to keep -- None on the drafter's own proposals)``.

    The loop: ``prefill`` (decode steps, the drafter's passes over the prompt) and the read of the first new token; then per
    step ``speculate`` -- the drafter's passes and one verify step, nothing read --, the ONE host read of the pair,
    ``sampling.accepted`` on the host, ``accept``.  The tokens of a step are the greedy tokens of its kept rows; those
    behind an end token or the limit are not emitted.  ``forced`` ``[new tokens so far - 1, scheduled count]`` -> the
    drafted tokens of a scheduled step, on the device before the loop: the step verifies them in the place of the
    drafter's (``force=``), and the count it keeps is still the comparison's.

    The text of a checkpoint goes to ``sink`` piece by piece between the host read of a step and the next launch, or
    once after the report; seeded weights give no language and their text is written nowhere.  The report is printed
    here, before any check: its bytes of a step follow where the session stands."""
    language = not getattr(text, "synthetic_text", True)
    sink = serving_options.TextSink() if sink is None else sink
    streamed = language and sink.streamed
    drafted = session.draft_tokens
    new_tokens = f"{rule.limit} new tokens" if rule.force_tokens is not None else f"up to {rule.limit} new tokens"
    where = sink.announced() if language else "the report and the text follow, the text is not streamed"
    _say(
        f"{TAG} decoding: {int(prompt.numel())} prompt tokens through the decode step, each but the first behind one pass of the drafter, then "
        f"{new_tokens} in steps of {drafted} draft passes and one verify step, and a traced pass; {where}",
        flush=True,
    )
    tokens = session.read(session.prefill([prompt]))  # the first new token: the greedy token of the prompt's last decode step
    produced, stream, pieces, scheduled, done, text_seconds = [], text.decoder(prompt), "", [], False, 0.0
    while True:
        started = time.perf_counter()
        for token in tokens:
            if not done:
                produced.append(token)
                piece = stream.push(token)
                pieces += piece
                if streamed:
                    sink.write(piece)  # as it is generated: this write is part of the turnaround between two steps
                done = rule.ends(token) or len(produced) >= rule.limit
        text_seconds += time.perf_counter() - started
        if done:
            break
        force = None
        if forced is not None:  # a scheduled step: a row of a table built before the loop -- a view by two host integers
            scheduled.append(len(scheduled) % (drafted + 1))
            force = forced[len(produced) - 1, scheduled[-1]].view(1, drafted)
        ((window, greedy),) = session.read(session.speculate(force))  # the drafter's passes, one verify step; the ONE host read of the step
        kept = sampling.accepted(window, greedy)  # the drafted tokens the model would have chosen itself: host arithmetic
        session.accept(kept)  # the state stands at the last kept token; nothing is launched or copied
        tokens = greedy[: kept + 1]
    rest = "" if not language else stream.close()  # what a text that stopped inside a character still owes
    if streamed:
        sink.write(rest + "\n")
    kept_counts = list(session.record.accepted_all)
    session.report().text_path["text_out"] = text_seconds / max(len(produced), 1)
    session.trace(TRACED_STEPS)  # the first steps again, under the profiler, behind the prompt replayed untraced
    _say(session.report().text())
    if not language:
        _say(f"text out              {len(produced)} tokens in {len(kept_counts)} steps (seeded weights: not language): {pieces[:48]!r}")
    else:
        ended = "ended at the end token" if rule.ends(produced[-1]) else f"stopped after {len(produced)} tokens"
        _say(f"text out              {len(produced)} tokens in {len(kept_counts)} steps, {ended}: {sink.reported(pieces + rest)}", flush=True)
        if not streamed:
            sink.write(pieces + rest + "\n")
    return produced, kept_counts, (scheduled if forced is not None else None)


def _check(session, plain, tie_out, produced, scheduled, stream, prompt_tokens, real):
    """The checks of one prompt: the run against decode steps of the same tokens (``tie_out.speculative_checks``), the ids
    of both arms where a plain run made a stream, and the checks of every run (``tie_out.run_checks``).  Returns ``(held,
    provider blocks judged)``."""
    steps = f"the first {TRACED_STEPS} steps" if real else "every step"
    own = f"the first {CHECKED_PROMPT_STEPS} prompt steps and the first {CHECKED_STEPS} steps" if real else "every step"
    also = "; the kernels of the traced steps against the plan" if session.check_kernels else ""
    _say(
        f"{TAG} checks on {steps}: the run against decode steps of the same tokens (tokens, kept state, position, prompt, recorded, bytes, "
        f"rejected rows, the drafter's cache and proposals); then on {own} against the {session.reference_precision}-accumulated reference "
        f"(replay with the drafter's proposals, every provider block of the model, the tokens){also}",
        flush=True,
    )
    if session.record.count:
        against = tie_out.speculative_checks(session, plain, scheduled)
        _say(against.text())
        held = against.ok
    else:  # the first new token ended the request: a line that compares nothing is not printed as a pass
        _say("speculative against decode: no step ran behind the prompt (its first new token ended the request): nothing to compare    REPORTED")
        held = True
    same = True
    if stream is not None:  # a token this loop emitted reaches no line of the comparison: the ids of both arms are held here
        new = len(produced)
        same = produced == stream[:new]
        differs = "" if same else f"; the first that differs is new token {next(i for i, (a, b) in enumerate(zip(produced, stream)) if a != b) + 1}"
        _say(
            f"  ids               the {new} new tokens of the speculative run against the plain run's of this process: ids hash {_ids_hash(produced)}, "
            f"plain {_ids_hash(stream[:new])}{differs}    {'PASS' if same else 'FAIL'}"
        )
    checks = tie_out.run_checks(
        session,
        prompt_tokens=prompt_tokens,
        max_steps=CHECKED_STEPS if real else None,
        second_reference=real,
        max_prompt_steps=CHECKED_PROMPT_STEPS if real else None,
    )
    _say(checks.text())
    return held and same and checks.ok, checks.judged


def main(argv=None) -> int:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    try:
        import cudnn  # noqa: F401  before torch: the first cuDNN library loaded wins
        import torch
    except ImportError as error:
        _say(f"{TAG} SKIP needs nvidia-cudnn-frontend and PyTorch ({error}): see samples/models/README.md")
        return 0
    from model_samples import Checkpoint, EnvironmentMismatch, Session, StopRule, TextIO, census, check_environment, loop, sampling, tie_out
    from model_samples.blocks import DeviceFacts, parse_overrides
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter

    if not torch.cuda.is_available():
        _say(f"{TAG} SKIP needs a CUDA GPU")
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    real = args.model is not None
    checkpoint, described = None, 0.0
    try:  # what a user can get wrong: one ERROR line, before the plan
        if args.sequences != 1:
            raise ValueError(f"--sequences {args.sequences}: {loop.ONE_SEQUENCE}")
        overrides = parse_overrides(args.provider)
        for flag, path in (("--dump-tokens", args.dump_tokens), ("--text-out", args.text_out)):
            if path is not None:
                _writable(flag, path)
        if real and args.text_out is None and sys.stdout is None:  # closed before the process started (>&-): known now, said now
            raise ValueError("standard output is closed and the generated text has nowhere to go; redirect it, or name a file with --text-out")
        given = args.speculative_config is not None
        shown = "--speculative-config" if given else "--speculative-config (the default)"
        drafted, config_line = serving_options.parse_speculative_config(args.speculative_config if given else serving_options.SPECULATIVE_DEFAULT, shown)
        every = 2 + drafted * (drafted + 1) // 2  # the new tokens a schedule needs to keep every count once
        if not real and not args.forced:
            args.force_tokens = max(args.force_tokens, every)
        if not real and args.force_tokens < every:
            raise ValueError(
                f"--force-tokens {args.force_tokens}: the seeded run schedules its steps to keep 0 to {drafted} drafted tokens in turn and needs "
                f"{every} new tokens to keep every count once"
            )
        if real:  # the text side and the description of the checkpoint and of its drafter: config and shard headers, no weights
            describing = time.perf_counter()
            text = TextIO.from_checkpoint(args.model)
            checkpoint = Checkpoint.describe(args.model)
            model = QwenDense.describe(checkpoint)
            name = os.path.basename(os.path.normpath(args.model)) or "."
            if text.vocab_size > model.geometry.vocab:
                raise ValueError(
                    f"--model {name}: the tokenizer has {text.vocab_size:,} ids, the model's embedding table {model.geometry.vocab:,} rows: "
                    "not the tokenizer of this checkpoint"
                )
            drafter = QwenDenseDrafter.describe(checkpoint, model, draft_vocab=args.draft_vocab)
            if "-" in args.messages:  # said before a read that waits for whoever writes to standard input
                _say(f"{TAG} --messages -: reading the request from standard input", flush=True)
            requests = [serving_options.prompt_request(prompt, f"--prompt {n + 1}") for n, prompt in enumerate(args.prompt)]
            requests += [serving_options.read_request(path, name) for path in args.messages]
            requests = requests or [serving_options.prompt_request(DEFAULT_PROMPT, "the default prompt")]
            described = time.perf_counter() - describing
        else:
            model = QwenDense.reduced(layers=args.layers)
            text = TextIO.synthetic(model.geometry.vocab)
            drafter = QwenDenseDrafter.seeded(model)
            requests = [serving_options.Request([], None, {}, "the seeded prompt")]
    except NotImplementedError as error:
        _say(f"{TAG} SKIP {error}")
        return 0
    except (ValueError, OSError) as error:
        _say(f"{TAG} ERROR {_reason(error)}")
        return 1
    facts = DeviceFacts.probe(device)
    turns = f"0, 1, .., {drafted}" if drafted > 2 else ", ".join(str(count) for count in range(drafted + 1))
    scheduled = f"acceptance is scheduled, not measured: the drafted tokens of every step are replaced by the plain run's tokens for {turns} "
    scheduled += "positions in turn and by another token behind them"
    what = checkpoint.summary() if real else "reduced model: " + "; ".join((*model.reductions, scheduled))
    _say(f"{TAG} {model.name} speculative decode | {what} | the drafter runs on reference blocks")
    plural = "s" if drafted > 1 else ""
    _say(
        f"{TAG} {facts.name} ({facts.class_text}, {facts.sm_count} SMs) | 1 sequence, {drafted} drafted token{plural} a step | stack {args.stack} | launched from Python, not captured"
    )
    if real:
        _say(f"{TAG} {text.name}")
        _say(f"{TAG} {serving_options.greedy_line(text.generation_defaults)}")
        for request in requests:  # a request body: every field of it, answered
            if request.line():
                _say(f"{TAG} {request.line()}")
    _say(f"{TAG} {config_line}")
    _say(f"{TAG} {drafter.vocabulary.text()}")

    started = time.perf_counter()
    try:  # every prompt is rendered and encoded before anything is planned: the longest request sizes the state
        sets = []  # per prompt: (prompt ids, the seconds of the text path, its stop rule)
        for request in requests:
            # the seeded prompt is the one 00_decode.py feeds its sequence 0: an open assistant turn that ends in the sequence's own token
            messages = request.messages if real else [{"role": "user", "content": "count"}, {"role": "assistant", "content": "0"}]
            options = request.options if real else {"add_generation_prompt": False}
            render = time.perf_counter()
            rendered = text.render_chat(messages, request.tools, **options)
            tokenize = time.perf_counter()
            ids = text.encode(rendered)
            seconds = {"render": tokenize - render, "tokenize": time.perf_counter() - tokenize}
            if not int(ids.numel()):
                raise ValueError(f"{request.shown}: the checkpoint's chat template and tokenizer turn it into no token; there is nothing to decode from")
            flag = args.max_new_tokens if args.limit_given else None
            # a verify step writes a row for every drafted token: the last step of a request writes drafted - 1 rows behind its last new
            # token, and the limit of new tokens is held against the context length together with them
            limit, note = serving_options.new_tokens(
                request, int(ids.numel()), args.max_model_len, flag, args.force_tokens, args.limit_flag, reserve=drafted - 1
            )
            if note:
                _say(f"{TAG} {note}")
            sets.append((ids, seconds, StopRule(text.eos_ids, max_new_tokens=limit, force_tokens=args.force_tokens)))
    except ValueError as error:  # a template that refuses its messages or renders them to nothing; a request beyond the context
        _say(f"{TAG} ERROR {error}")
        return 1
    # a step at position p needs p + drafted + 1 positions: the last step of a run can start at its last new token but one
    common = dict(
        sequences=1,
        stack=args.stack,
        device=device,
        overrides=overrides,
        reference_precision=args.reference_precision,
        capacity=max(int(ids.numel()) + rule.limit for ids, _, rule in sets) + drafted - 1,
        max_model_len=args.max_model_len,
        check_kernels=args.check_kernels,
    )
    # the plain session loads: it is handed the checkpoint.  It makes the token stream of a seeded run and is what every run is compared with
    plain = Session(model, text, checkpoint=checkpoint, record_steps=1 if real else None, **common)
    recorded = max(CHECKED_PROMPT_STEPS, TRACED_STEPS) if real else None
    session = Session(model, text, share=plain, draft_tokens=drafted, drafter=drafter, record_steps=recorded, **common)
    if args.stack != "reference":  # said before a phase that prints nothing, outside its timed interval
        _say(
            f"{TAG} planning: on a GPU an entry of this stack builds its kernel for each kind of step now, before anything is loaded "
            "(what it needs of the installed toolchain is asked here; a first build prints nothing while it runs)",
            flush=True,
        )
    planning = time.perf_counter()
    try:  # every plan of the run before the first load, and ONE memory sentence, the drafter in it: nothing skips after this
        # what a library row of the stack does at its first use beyond an entry's build, said before it is made (as 00_decode.py says it)
        for note in dict.fromkeys(note for each in (plain, session) for note in each.resolve().announcements()):
            _say(f"{TAG} planning: {note}", flush=True)
        tables = [plain.plan_table([session]), session.plan.table(), loop.PROMPT_ROWS, session.prefill_plan.table()]
        tables += [loop.DRAFTER_ROWS.format(calls=drafter.calls_per_pass()), session.draft_plan.table()]
    except NotImplementedError as error:
        _say(f"{TAG} SKIP {error}")
        return 0
    except ValueError as error:
        _say(f"{TAG} ERROR {error}")
        return 1
    plain.stages.add("import", started - _START - described)
    if real:
        plain.stages.add("describe", described)
    plain.stages.add("plan", time.perf_counter() - planning)
    try:  # after the plan: a GPU this sample does not serve skips, whatever is installed
        environment = check_environment()
    except EnvironmentMismatch as error:
        _say(f"{TAG} ERROR {error}")
        return 1
    _say(f"{TAG} {environment.header()}")
    _say(f"{TAG} clock not locked: timings are for orientation only | reference precision {args.reference_precision}")
    _say(f"{TAG} {serving_options.speculative_line([(request.shown, int(ids.numel())) for request, (ids, _, _) in zip(requests, sets)], drafted)}")
    _say()
    _say("\n".join(tables))
    if session.check_kernels:  # the kernel table of the verify plan, first part: from the plan alone, before the load
        _say(census.table(session.plan))
    plain.load()
    session.load()  # the drafter is loaded here, with the model's embedding table
    _say(f"{TAG} {plain.stages.line()}")
    _say(f"{TAG} the session with the drafter: {session.stages.line()}")
    if session.check_kernels:  # second part: what the entries, now built, state about their launches
        _say(census.stated(session.plan))

    passed, judged, digest, dumped, unobserved = True, 0, hashlib.sha256(), [], ""
    sink = serving_options.TextSink(args.text_out, streamed=not args.no_stream)
    vocab = model.geometry.vocab
    for index, (prompt, seconds, rule) in enumerate(sets):
        if index:  # a session keeps nothing between two prompts
            _say()
            session.reset(fresh=True)
        if real:
            _say(f"prompt {index + 1} of {len(sets)}        {int(prompt.numel())} tokens")
        stream, forced = None, None
        if not real:  # the plain run first: the token stream the scheduled drafts are cut from, and what the ids are held against
            ids = plain.prefill([prompt])
            stream = plain.read(ids)
            for _ in range(rule.limit + drafted - 2):
                ids = plain.step(ids)
                stream += plain.read(ids)
            table = [[tie_out.forced_drafts(stream[done:], length, drafted, vocab) for length in range(drafted + 1)] for done in range(1, rule.limit)]
            forced = torch.tensor(table, dtype=torch.int32, device=session.device)  # every scheduled window of the run: one copy, before the first step
        session.new_report()
        session.report().text_path.update(seconds)
        produced, kept, scheduled = _speculate(session, text, rule, prompt, sampling, forced, sink)
        ok, blocks_judged = _check(session, plain, tie_out, produced, scheduled, stream, int(prompt.numel()), real)
        passed, judged = passed and ok, judged + blocks_judged
        if session.check_kernels:
            unobserved = unobserved or session.traced.unobserved
        digest.update(b"".join(int(i).to_bytes(4, "little") for i in produced))
        dumped.append({"sequences": 1, "prompt": index, "prompt_ids": [[int(i) for i in prompt]], "ids": [produced], "kept": kept})
    if checkpoint is not None:
        checkpoint.close()
    sink.close()
    gib, mib = 2.0**30, 2.0**20
    grown = (facts.free_bytes - torch.cuda.mem_get_info(device)[0]) / gib
    _say()
    peak, reserved = torch.cuda.max_memory_allocated(device) / gib, torch.cuda.max_memory_reserved(device) / gib
    _say(
        f"{TAG} GPU memory: PyTorch peak {peak:.2f} GiB allocated, {reserved:.2f} GiB reserved; in use on the device grew by {grown:.2f} GiB "
        "since the plan (PyTorch's pool, CUDA libraries and kernel reservations of this process; other processes' changes included)"
    )
    allowance = getattr(model, "first_launch_reserve", {}).get(f"cc{facts.cc[0]}.x", 0)
    for kind, plan in (("the plain decode step", plain.plan), ("the verify step", session.plan), ("the prompt's decode step", session.prefill_plan)):
        for name, block in plan.items():
            launches = block.first_launch()  # measured by the block: rows -> device bytes outside PyTorch's pool
            if launches:
                forms = " | ".join(f"{rows} row{'s' if rows > 1 else ''} {taken / mib:,.0f} MiB" for rows, taken in launches.items())
                allowed = f"the plan allowed {allowance / mib:,.0f} MiB on this class" if allowance else "the plan allows none on this class"
                _say(f"{TAG} first launch of {name} ({kind}), device memory outside PyTorch's pool: {forms} (measured in this process; {allowed})")
    unwritten = []  # what a flag asked for and is not there
    if sink.error:  # the reader of standard output went away, or the file of --text-out stopped taking text
        unwritten.append(f"the generated text could not be written to {sink.name}: {sink.error}")
    if args.dump_tokens is not None:
        error = _dump(args.dump_tokens, args.stack, dumped, draft_tokens=drafted)
        if error:
            unwritten.append(error)
        else:
            _say(f"{TAG} token ids and kept counts of {len(dumped)} prompt(s) written to {os.path.basename(args.dump_tokens)}")
    weights = "checkpoint weights, greedy decoding" if real else "seeded weights: the output is not language"
    _say(f"{TAG} {weights}; ids hash {digest.hexdigest()[:12]}")
    _say(f"{TAG} the drafter ran on reference blocks: none of its blocks was compared with a provider (no provider serves them yet)")
    verdict = _verdict(args.stack, passed, judged, unobserved)
    if unwritten:  # the run is done and its verdict stands, but what was asked for is not there
        _say(f"{TAG} ERROR {'; '.join(unwritten)} (the run itself: {verdict})")
        return 1
    _say(f"{TAG} {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
