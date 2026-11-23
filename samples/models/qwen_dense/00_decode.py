# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 00: plain decode of the dense Qwen 3.5-architecture model in one process: text in, text out.

The default run needs no checkpoint and no extra package: it builds the model at its published layer widths with two layers
and seeded weights, prints which provider serves every block, feeds one seeded prompt per sequence through the decode step,
decodes 32 tokens greedily at 1 and at 4 sequences, and checks the whole path against a plain-PyTorch reference that
accumulates in fp64 (replay, every provider block teacher-forced, the greedy tokens, each sequence alone against the same
sequence in the batch).  It prints ``[00] PASS``, or ``[00] SKIP <requirement>`` with exit code 0 on a GPU, a library version
or a free-memory budget it does not serve.  Every sequence count of a run is planned before anything is loaded, so a SKIP
comes before the first load or not at all.

With ``--model DIR`` it runs a checkpoint from local disk (nothing is downloaded): the checkpoint's own tokenizer and chat
template turn ``--prompt`` or ``--messages`` into token ids, the model decodes greedily to the end token or to
``--max-tokens``, and the same checks run on the first steps of every prompt.  That needs the packages of
``samples/models/requirements.txt``, the checkpoint on local disk (your own download) and a GPU with room for a bf16 copy
of every weight: about 57 GiB free for the 27B checkpoint.

Coming from a serving engine: ``--messages`` takes the request body a chat client sends, from a file or from standard input
(``-``).  Every field of it is answered -- read, printed, accepted at the value at which it changes nothing, or refused by
name because it would change the output (``serving_options.py`` beside this script has the rules).  ``--max-tokens`` and
``--max-model-len`` mean what they mean there: the limit of new tokens of a request, and the positions a prompt and its new
tokens share.  The generated text of sequence 0 goes to standard output as it is generated (``--no-stream``: once, after the
report; ``--text-out FILE``: into a file); every other line of a run -- the ``[00]`` lines, the tables, the report, the
checks -- goes to standard error, so the text can be piped.  A request gets one text: with a list of sequence counts it
is the text of the first count, and the report of a later count says whether its sequence 0 produced the same one.
Token choice is always greedy, and a run says so beside the checkpoint's own generation settings.  There is no socket
and no queue: one process, one fixed set of sequences.

    python samples/models/qwen_dense/00_decode.py
    python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
    python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: reports RAN, never PASS
    python samples/models/qwen_dense/00_decode.py --check-kernels          # also: every launch of the traced steps against the plan
    python samples/models/qwen_dense/00_decode.py --stack native           # the target kernel set, as far as it is served here

    pip install -r samples/models/requirements.txt                         # only for a checkpoint; brings the hf command
    hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4   # your own step: the sample fetches nothing
    python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-tokens 256
    python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages request.json > answer.txt
    python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --messages - < request.json 2> report.txt

The stack is explicit and never chosen by what is installed.  ``fe-cublas`` is a reference path, not a fast path: the
linear-attention step runs on cuDNN Frontend's fused kernel, every multiply on cuBLAS over a bf16 copy of the quantised weights,
the rest in PyTorch, and every launch is issued from Python.  Timings printed here are for orientation only.

``--stack native`` runs every block on a kernel that works on the stored weights: seven entries of cuDNN Frontend and four
calls of a library, FlashInfer, each in the form its row of the provider table names.  A block no kernel serves on this GPU
class or at this row count stays on its fallback row, and the tables say so with the reason.  It always checks its kernels:
every launch of the traced steps against the plan, an entry's kernel by name, order, count and the launch shape the entry
states, a library's kernel by name, order and count, with the kernel the library launched printed on the row's line.
``--stack fe-entries`` is every cuDNN Frontend row of ``native`` with the rest on fallback rows: no package beyond cuDNN
Frontend and PyTorch.  The four rows ``native`` has more are the library's: three multiplies on the stored weights, and
the attention core, its paged decode attention over the paged attention cache.  FlashInfer is an optional package of that stack alone
(``pip install -c samples/models/constraints-native.txt -r samples/models/requirements-flashinfer.txt``), and the kernels of
its attention are precompiled binaries of its kernel-binary package (``samples/models/requirements-flashinfer-cubin.txt``
names the command); without either, or with a version this area has not validated, on compute capability 10.x
``--stack native`` is a ``SKIP`` before anything is loaded.  The library builds the kernels of its multiplies and the
launcher of its attention at its first use: the run makes that use while it plans and says so.  A run never lets the
library fetch a kernel binary: it switches the library's download off before it reaches the library.

``--check-kernels`` adds a fifth check, against the plan instead of the reference: the run prints which kernels every block
launches (the kernel table, before the load) and what the entries state about their launches (after it), keeps the profiler's
record of every launch of the traced steps, and holds them against each other -- every launch attributed to a block, the
kernels of an entry by name, order, count, grid, block and shared memory.  Where the profiler cannot run the run ends in
``[00] RAN <stack> stack (kernels not observed: <reason>)``, never in ``PASS``.  ``--dump-census FILE`` writes those launch
records as JSON.

``--perf`` is the performance mode: the decode performance report of this GPU from one process.  One checked request
with the reference checks on it, also at four positions of the timed steps; a warm-up request (``--perf-warmup`` forced
steps, default 32); ``--perf-requests`` timed requests (default 3), each the prompt, its first new token and
``--perf-steps`` forced decode steps (default 256, the end token ignored) with nothing but the step inside; only then
the profiler, for one traced request of as many of those steps as ``--perf-trace-records`` launch records hold (default
160000), and the kernel check on it.  The report goes to standard error and, with every number, its definition, the
protocol and the environment, into a report file (``--perf-report FILE``; default ``report-<UTC time>.json`` in the
current directory); the text is written once, after the report.  Without ``--model`` the same protocol runs on the
reduced seeded model and prints no rate.  ``samples/models/README.md`` has the protocol, the statistics and the labels.
The report's ``GPU`` lines say what the run observed through the driver's ``nvidia-smi`` tool, in child processes --
the other compute processes on the GPU, its SM clock, the limit reasons -- or that it could not, with the reason.
``--perf-clock-lock MHZ`` declares a clock that an operator fixed outside this sample, which sets none: the report
holds the declaration against the clock it observed.

    python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --stack native --perf
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time

_START = time.perf_counter()
# The traced pass of the report runs under the PyTorch profiler, whose tracing library writes to stderr when it starts and
# stops and once for every kernel it cannot rate on a GPU it does not know.  Unless a level is set in the environment, keep that out of
# the sample's output; when the records are missing the report says so itself.  (The tracing library reads the level when
# the first profiler pass starts, so setting it here is early enough whether or not PyTorch is already imported.)
os.environ.setdefault("KINETO_LOG_LEVEL", "6")
_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS = os.path.normpath(os.path.join(_HERE, ".."))  # samples/models: the home of the model_samples package
for _path in (_HELPERS, _HERE):  # and this folder: serving_options, the module beside the scripts of the family
    if _path not in sys.path:
        sys.path.insert(0, _path)

import serving_options  # noqa: E402  the standard library only: cudnn and PyTorch are imported in main(), in their order

TAG = "[00]"
STACKS = ("fe-cublas", "native", "fe-entries", "reference")
DEFAULT_PROMPT = "Why is the sky blue?"
CHECKED_STEPS = 16  # with a checkpoint the checks cover the first steps of every prompt
TRACED_STEPS = 32
SPECULATIVE = "02_speculative_decode.py"  # the script of this folder that drafts
#: its two flags, each with what it takes: declared here so that each is refused by a sentence that names that script
SPECULATIVE_FLAGS = (("--speculative-config", "JSON"), ("--draft-vocab", "list"))


def _say(*lines, flush: bool = False) -> None:
    """A line of the run.  Every ``[00]`` line, table, report and check goes to standard error: standard output carries
    the generated text and nothing else, so that it can be piped.  Where standard error was closed before the process
    started (``2>&-``) the interpreter has no stream in its place and the lines go nowhere -- never to standard output,
    which is where ``print`` would send them."""
    if sys.stderr is not None:
        print(*lines, file=sys.stderr, flush=flush)


def _reason(error) -> str:
    """The text of an error for an ``ERROR`` line: a file is named by its own name, never by a path."""
    if isinstance(error, OSError) and error.filename:
        return f"{os.path.basename(os.path.normpath(str(error.filename)))}: {error.strerror or type(error).__name__}"
    return str(error)


def _seeded_messages(sequence: int) -> list:
    """The chat of the seeded prompt of one sequence: its open assistant turn ends in the sequence's own token."""
    return [{"role": "user", "content": "count"}, {"role": "assistant", "content": str(sequence % 10)}]


def _dump(path: str, stack: str, runs: list, flag: str = "--dump-tokens", **facts) -> str:
    """Write the prompt ids and the produced ids of a run as JSON -- or, for ``--dump-census``, the launch records of its
    traced steps.  Returns "" or, where the file cannot be written after all (the check before the run passed: a full disk,
    a folder that went away), the reason with the file's own name."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sample": "qwen_dense/00_decode", "stack": stack, "greedy": True, **facts, "runs": runs}, f)
    except OSError as error:
        return f"{flag} {os.path.basename(os.path.normpath(path))}: {error.strerror or type(error).__name__}"
    return ""


def _writable(flag: str, path: str) -> None:
    """A file a flag asks for that cannot be written is an error before the run, not after it: a ``ValueError`` that names the
    file by its own name.  So is a flag that was given no name at all (``--text-out "$OUT"`` with the variable unset)."""
    if not path.strip():
        raise ValueError(f"{flag}: needs a file name; it was given an empty one")
    target = os.path.abspath(path)
    there = os.path.exists(target)
    if os.path.isdir(target) or not os.access(target if there else os.path.dirname(target), os.W_OK):
        raise ValueError(f"{flag} {os.path.basename(os.path.normpath(path))}: cannot be written there")


EPILOG = """\
The default run needs no checkpoint and no extra package; it needs a GPU of compute capability 10.x or 12.x.
With --model: pip install -r samples/models/requirements.txt first.  The checkpoint is your own download (nothing is
fetched); the 27B checkpoint needs 22 GB of disk and about 57 GiB of free GPU memory on the stack fe-cublas, which holds a
bf16 copy of every weight.  The stacks native and fe-entries keep a weight their kernels multiply on its stored format
without that copy.  fe-entries: about 37 GiB on compute capability 10.0, about 41 GiB on the other 10.x classes, about 57 GiB elsewhere.
native, whose library rows need FlashInfer and its kernel-binary package (pip install -c samples/models/constraints-native.txt
-r samples/models/requirements-flashinfer.txt; the README: "The rows a library serves"): about 28 GiB and about 32 GiB there.
The generated text goes to standard output; every other line of a run goes to standard error.
The checkpoint, the memory, the checks, a request body field by field and every flag: samples/models/qwen_dense/README.md
"""


def _arguments(argv):
    parser = argparse.ArgumentParser(
        description="Plain decode of the dense Qwen model: the reduced seeded model (self-checking), or a checkpoint from local disk.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", metavar="DIR", help="a checkpoint directory on local disk; never a hub name, nothing is downloaded")
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
        "chat client sends (messages, tools, chat_template_kwargs and max_tokens are read; a field that would change the output is "
        "an error that names it); repeatable",
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
    parser.add_argument("--text-out", metavar="FILE", help="with --model: write the generated text of sequence 0 to FILE instead of standard output")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="with --model: write the text once, after the report of its prompt, instead of piece by piece as it is generated",
    )
    parser.add_argument(
        "--force-tokens",
        type=int,
        default=None,
        metavar="N",
        help="decode exactly N tokens, ignoring the end token (default without --model: 32; with it: off)",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        metavar="N[,N...]",
        help="sequences per run, fixed before the first step; a list repeats the run per count, and the text of a request is written "
        "once, with the first count (default 1,4; with --model 1)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        metavar="N",
        help="layers of the reduced model (default 2: one linear-attention + one attention layer); not with --model",
    )
    parser.add_argument(
        "--stack",
        default="fe-cublas",
        choices=STACKS,
        help="fe-cublas (default): cuDNN Frontend + cuBLAS + PyTorch, checked against the reference; native: the target kernel set of the "
        "decode step, per block the kernel a recorded census names, with the blocks not served yet on fallback rows (printed as not yet) "
        "and the kernel check always on; fe-entries: the cuDNN Frontend rows of native, everything else on fallback rows; reference: "
        "every block in PyTorch, on any CUDA GPU, nothing checked.  Explicit: never chosen by what is installed",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="BLOCK=KEY",
        help="select one declared provider row of a block (an unknown block or key prints the declared ones)",
    )
    parser.add_argument("--reference-precision", default="fp64", choices=("fp64", "ieee-fp32"), help="accumulation of the reference blocks (default fp64)")
    parser.add_argument("--dump-tokens", metavar="FILE", help="write the prompt ids and the produced ids as JSON, for a comparison made elsewhere")
    parser.add_argument(
        "--check-kernels",
        action="store_true",
        help="also check the kernels: print the kernel table, keep the profiler's record of every launch of the traced steps and hold "
        "them against the plan (every launch attributed to a block; an entry's kernel by name, order, count and launch shape). "
        "Where the profiler cannot run the run reports RAN (kernels not observed), never PASS; not with --stack reference; always on "
        "in the stacks native and fe-entries",
    )
    parser.add_argument(
        "--dump-census",
        metavar="FILE",
        help="write the launch records of the traced steps as JSON (kernel, grid, block, shared memory, start and end per launch), for a "
        "comparison made elsewhere; implies --check-kernels",
    )
    _perf_flags(parser)
    for flag, takes in SPECULATIVE_FLAGS:
        parser.add_argument(flag, metavar=takes.upper(), help=f"not this script: speculative decoding is {SPECULATIVE}, which takes the same {takes}")
    args = parser.parse_args(argv)
    for flag, takes in SPECULATIVE_FLAGS:
        if getattr(args, flag[2:].replace("-", "_")) is not None:
            parser.error(f"{flag}: speculative decoding is the script {SPECULATIVE} of this folder, which takes the same {takes}")
    args.check_kernels = args.check_kernels or args.dump_census is not None
    # the limit of new tokens has two spellings; a sentence about it names the one that was typed
    args.limit_flag = "--max-new-tokens" if any(word.split("=")[0] == "--max-new-tokens" for word in argv) else "--max-tokens"
    args.limit_given = args.max_new_tokens is not None
    _perf_arguments(parser, args)
    if args.check_kernels and args.stack == "reference":
        parser.error("--check-kernels and --dump-census are for a provider stack; --stack reference checks nothing")
    if args.model is None and (args.prompt or args.messages):
        parser.error("--prompt and --messages need --model: the seeded model has no language")
    if args.model is None and (args.text_out is not None or args.no_stream):
        parser.error("--text-out and --no-stream need --model: the seeded model's output is not language and is written nowhere")
    if args.model is not None and args.layers is not None:
        parser.error("--layers is for the reduced model; a checkpoint has the depth it has")
    if args.limit_given and args.model is None:  # the seeded run decodes a fixed count: the flag would be ignored
        parser.error(f"{args.limit_flag} is for --model; the seeded run decodes --force-tokens N (default 32)")
    if args.limit_given and args.force_tokens is not None:
        parser.error(f"--force-tokens N decodes exactly N tokens; {args.limit_flag} does not apply beside it")
    if args.temperature is not None and args.temperature != 0:
        parser.error(f"--temperature {args.temperature:g}: token choice is greedy, the token of the largest logit at every step; only 0 is accepted")
    if args.messages.count("-") > 1:
        parser.error(f"--messages - reads one request from standard input; it was given {args.messages.count('-')} times")
    try:
        args.sequences = [int(n) for n in str(args.sequences or ("1" if args.model else "1,4")).split(",")]
    except ValueError:
        parser.error("--sequences takes N or N,N,...")
    if args.force_tokens is None and args.model is None:
        args.force_tokens = 32
    args.max_new_tokens = 256 if args.max_new_tokens is None else args.max_new_tokens
    if (
        min(args.sequences) < 1
        or args.max_new_tokens < 1
        or args.max_model_len < 1
        or (args.force_tokens is not None and args.force_tokens < 1)
        or (args.layers is not None and args.layers < 1)
    ):
        parser.error("--sequences, --layers, --max-tokens, --max-model-len and --force-tokens take positive integers")
    args.layers = 2 if args.layers is None else args.layers
    return args


#: the counts of the performance mode: flag -> (default, what it counts)
PERF_COUNTS = {
    "--perf-warmup": (32, "forced decode steps of the warm-up request"),
    "--perf-steps": (256, "timed decode steps of a request, behind its first new token; at the default 257 new tokens and 256 intervals"),
    "--perf-requests": (3, "timed requests"),
    "--perf-trace-records": (160_000, "the most launch records the traced request may hold; where the timed steps do not fit, it traces the first that do"),
}


def _perf_flags(parser) -> None:
    """The flags of the performance mode.  A count that is left out is ``None`` here, so that one given without ``--perf``
    can be told from a default."""
    parser.add_argument(
        "--perf",
        action="store_true",
        help="the performance mode: one checked request, a warm-up request, timed requests with nothing but the step inside, then one traced "
        "request and the kernel check on it; prints the decode performance report and writes a report file. Implies one sequence count, the "
        "kernel check, forced steps with the end token ignored, and the text written once, after the report; not with --stack reference",
    )
    for flag, (default, what) in PERF_COUNTS.items():
        parser.add_argument(flag, type=int, default=None, metavar="N", help=f"with --perf: {what} (default {default})")
    parser.add_argument(
        "--perf-report",
        metavar="FILE",
        help="with --perf: the report file (default report-<UTC time>.json in the current directory); a file that exists, or that another flag of "
        "the run writes, is an error before the load",
    )
    parser.add_argument(
        "--perf-clock-lock",
        type=int,
        default=None,
        metavar="MHZ",
        help="with --perf: the SM clock in MHz that an operator fixed outside this sample: a declaration, written into the report and held "
        "against the clock the run observes. The sample never sets a clock",
    )


def _perf_arguments(parser, args) -> None:
    """The flags of the performance mode against each other and against the flags of an ordinary run: every refusal is a
    usage error, before anything is read or loaded.  With ``--perf`` the run decodes the mode's own fixed count (the first
    new token and ``--perf-steps`` forced steps), at one sequence count, and checks its kernels."""
    counts = {flag: getattr(args, flag[2:].replace("-", "_")) for flag in PERF_COUNTS}
    given = [flag for flag, value in counts.items() if value is not None] + (["--perf-report"] if args.perf_report is not None else [])
    given += ["--perf-clock-lock"] if args.perf_clock_lock is not None else []
    if not args.perf:
        if given:
            parser.error(f"{', '.join(given)}: a flag of the performance mode; it needs --perf")
        return
    if args.stack == "reference":
        parser.error("--perf times a provider stack and holds it against the reference; --stack reference has no provider to time")
    if args.limit_given or args.force_tokens is not None:
        parser.error(
            "--perf decodes a fixed count: the first new token and --perf-steps forced steps behind it, the end token ignored; "
            f"{args.limit_flag if args.limit_given else '--force-tokens'} does not apply beside it"
        )
    if args.sequences is not None and "," in str(args.sequences):
        parser.error("--perf times one sequence count; --sequences was given a list")
    if len(args.prompt) + len(args.messages) > 1:
        parser.error(f"--perf times one prompt; {len(args.prompt) + len(args.messages)} were given (--prompt, --messages)")
    for flag, (default, _) in PERF_COUNTS.items():
        setattr(args, flag[2:].replace("-", "_"), default if counts[flag] is None else counts[flag])
    if args.perf_warmup < 1 or args.perf_steps < 2 or args.perf_requests < 1 or args.perf_trace_records < 1:
        parser.error("--perf-warmup, --perf-requests and --perf-trace-records take positive integers, --perf-steps an integer from 2")
    if args.perf_clock_lock is not None and args.perf_clock_lock < 1:
        parser.error("--perf-clock-lock takes the declared SM clock in MHz, a positive integer")
    args.sequences = "1" if args.sequences is None else args.sequences  # one count: the seeded default of an ordinary run is a list
    args.force_tokens = args.perf_steps + 1  # a request: the first new token and that many forced steps
    args.check_kernels = True  # the kernels of the traced steps are one of the three statements of the report's checks


def _perf_longest(args) -> tuple:
    """``(forced steps, flag, name)`` of the longest request of a run of the performance mode.  The checked request and
    every timed request run ``--perf-steps`` forced steps behind the first new token and the traced request at most as
    many; the warm-up request runs ``--perf-warmup``, which may be the larger count.  One session serves every request of
    the run: it is built for the longest of them, and ``--max-model-len`` is held against that one."""
    if args.perf_warmup > args.perf_steps:
        return args.perf_warmup, "--perf-warmup", "the warm-up request"
    return args.perf_steps, "--perf-steps", "a timed request"


def _positions(args, prompt_tokens: int, rule) -> int:
    """The positions a sequence of one prompt set reaches, which its session is built for: the prompt and its new tokens.
    In the performance mode the new tokens of the longest request of the run (``_perf_longest``): its first new token and
    its forced steps."""
    return prompt_tokens + (_perf_longest(args)[0] + 1 if args.perf else rule.limit)


def _perf_limit(args, request, prompt_tokens: int) -> list:
    """The longest request of a run of the performance mode (``_perf_longest``) -- the prompt, the first new token and that
    request's forced steps -- held against ``--max-model-len`` in the mode's own words (``ValueError``: which request is
    the longest, and every count to take lower).  Returns the line a run prints for a request body whose own limit does
    not apply, in a list, or no line."""
    steps, flag, longest = _perf_longest(args)
    room = args.max_model_len - prompt_tokens - 1
    if room < steps:
        over = [name for name, count in (("--perf-steps", args.perf_steps), ("--perf-warmup", args.perf_warmup)) if count > room]
        fewer = f"take {' and '.join(over)} {room:,} or fewer, or " if room >= 2 else ""  # a timed request has at least 2 steps
        raise ValueError(
            f"{request.shown}: {prompt_tokens:,} prompt tokens, the first new token and {flag} {steps:,} ({longest}, the longest request of "
            f"this run) are {prompt_tokens + 1 + steps:,} positions and --max-model-len is {args.max_model_len:,}; {fewer}raise --max-model-len"
        )
    beside = f"does not apply beside --perf, which decodes --perf-steps {args.perf_steps:,} forced steps behind the first new token"
    return [] if request.max_tokens is None else [f'{request.shown}: its "{request.limit_field}": {request.max_tokens} {beside}']


def _record_steps(real: bool, perf: bool):
    """The steps a session records for its checks: with a checkpoint the first steps an ordinary run checks and traces;
    every step (``None``) of a seeded run and of the performance mode, whose checked request the checks replay whole."""
    return max(CHECKED_STEPS, TRACED_STEPS) if real and not perf else None


def _verdict(stack: str, passed: bool, judged: int, unobserved: str = "") -> str:
    """The last line of a run.  ``PASS`` says that provider blocks were judged against the reference and held.  A run on the
    reference stack, or one whose ``--provider`` flags put every judged block on its reference row, compared the reference
    with itself: it reports ``RAN``.  So does a run that was asked to check its kernels and could not observe them
    (``unobserved``: why): its other checks held, the one it was asked for did not run.  A failed check is ``FAIL`` whatever
    else is true, on the reference stack too (no run of this script checks anything there; a script of this folder that
    does shares this answer)."""
    if not passed:
        return "FAIL"
    if stack == "reference":
        return "RAN reference stack (no provider checked)"
    if unobserved:
        return f"RAN {stack} stack (kernels not observed: {unobserved})"
    return "PASS" if judged else "RAN (no provider block was judged)"


def _decode(session, text, rule, prompts, tie_out, check, checked_steps, sink=None):
    """One prompt set: prefill, the decode loop (the one host read of a step is here), the traced pass, the report and the
    checks.  Returns ``(passed, provider blocks judged, tokens per sequence)``.  Whether the kernels are checked is the
    session's decision (``session.check_kernels``: the flag, or a stack that declares a target kernel set) and is read
    there, in one place: the traced pass then keeps its launch records and the checks include the kernel check; its
    outcome stays on the session (``session.traced``).

    The text of sequence 0 of a checkpoint goes to ``sink`` (``serving_options.TextSink``; default: standard output, as
    it is generated): piece by piece inside the loop, between the host read of a step and the next launch, or once after
    the report.  The sink of a later sequence count of a run (``TextSink.later()``) takes nothing: a request gets one
    text, and the ``text out`` line then says whether this count produced the same one.  Seeded weights give no
    language, and their text is written nowhere.  Every other line goes to standard error.

    Two lines say what comes next before the phases that print nothing for a while (the prompt and the decode loop; the
    checks).  They are printed outside every timed interval, and flushed, so a run through a pipe shows them too."""
    language = not getattr(text, "synthetic_text", True)
    sink = serving_options.TextSink() if sink is None else sink
    streamed = language and sink.streamed
    new_tokens = f"{rule.limit} new tokens" if rule.force_tokens is not None else f"up to {rule.limit} new tokens"
    where = sink.announced() if language else "the report and the text follow, the text is not streamed"
    _say(f"{TAG} decoding: {int(prompts[0].numel())} prompt tokens through the decode step, then {new_tokens} and a traced pass; {where}", flush=True)
    next_ids = session.prefill(prompts)
    tokens = session.read(next_ids)  # the first new token of every sequence
    produced = [[] for _ in prompts]
    streams = [text.decoder(p) for p in prompts]
    pieces = ["" for _ in prompts]
    done = [False] * len(prompts)
    text_seconds = 0.0
    while True:
        started = time.perf_counter()
        for s, token in enumerate(tokens):
            if not done[s]:
                produced[s].append(token)
                piece = streams[s].push(token)
                pieces[s] += piece
                if streamed and s == 0:
                    sink.write(piece)  # as it is generated: this write is part of the turnaround between two steps
                done[s] = rule.ends(token) or len(produced[s]) >= rule.limit
        text_seconds += time.perf_counter() - started
        if all(done):
            break
        next_ids = session.step(next_ids)  # embedding, the layers, the output head, token choice: ids stay on the GPU
        tokens = session.read(next_ids)  # the one host read of a step: 4 bytes per sequence
    rest = "" if not language else streams[0].close()  # what a sequence that stopped inside a character still owes
    if streamed:
        sink.write(rest + "\n")
    session.report().text_path["text_out"] = text_seconds / max(sum(len(ids) for ids in produced), 1)
    kernels = session.check_kernels
    session.trace(TRACED_STEPS)  # the first steps again, under the profiler: GPU busy, idle, launches per step; launch records where the kernels are checked
    _say(session.report().text())
    if not language:
        _say(f"text out              sequence 0, {len(produced[0])} tokens (seeded weights: not language): {pieces[0][:48]!r}")
    else:
        ended = "ended at the end token" if rule.ends(produced[0][-1]) else f"stopped after {len(produced[0])} tokens"
        _say(f"text out              sequence 0, {len(produced[0])} tokens, {ended}: {sink.reported(pieces[0] + rest)}", flush=True)
        if not streamed:
            sink.write(pieces[0] + rest + "\n")
    if not check:
        return True, 0, produced
    steps = "every step" if checked_steps is None else f"the first {checked_steps} steps"
    also = "; the kernels of the traced steps against the plan" if kernels else ""
    _say(
        f"{TAG} checks on {steps} against the {session.reference_precision}-accumulated reference (replay, every provider block, the tokens){also}",
        flush=True,
    )
    checks = tie_out.run_checks(session, prompt_tokens=int(prompts[0].numel()), max_steps=checked_steps, second_reference=checked_steps is not None)
    _say(checks.text())
    return checks.ok, checks.judged, produced


def main(argv=None) -> int:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    try:
        import cudnn  # noqa: F401  before torch: the first cuDNN library loaded wins
        import torch
    except ImportError as error:
        _say(f"{TAG} SKIP needs nvidia-cudnn-frontend and PyTorch ({error}): see samples/models/README.md")
        return 0
    import model_samples
    from model_samples import census, tie_out
    from model_samples.blocks import DeviceFacts, parse_overrides
    from model_samples.qwen_dense import QwenDense

    if not torch.cuda.is_available():
        _say(f"{TAG} SKIP needs a CUDA GPU")
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    real = args.model is not None
    checkpoint, requests, described = None, [], 0.0
    try:
        overrides = parse_overrides(args.provider)
        for flag, path in (("--dump-tokens", args.dump_tokens), ("--dump-census", args.dump_census), ("--text-out", args.text_out)):
            if path is not None:  # a file that cannot be written, or a flag without a name, is an error before the run, not after it
                _writable(flag, path)
        if args.perf:  # the report file's name is fixed now: one that exists is never written over, and that is known before the load
            perf = importlib.import_module("model_samples.perf")  # the mode's module: an ordinary run does not import it
            report = perf.default_report_name() if args.perf_report is None else args.perf_report
            _writable("--perf-report", report)
            if os.path.exists(report):
                raise ValueError(f"--perf-report {os.path.basename(os.path.normpath(report))}: the file exists, and a report file is never written over")
            for flag, path in (("--dump-tokens", args.dump_tokens), ("--dump-census", args.dump_census), ("--text-out", args.text_out)):
                if path is not None and os.path.realpath(path) == os.path.realpath(report):  # written behind the report, over it
                    raise ValueError(
                        f"--perf-report and {flag} name one file, {os.path.basename(os.path.normpath(report))}: a report file is never written over"
                    )
        if real and args.text_out is None and sys.stdout is None:  # closed before the process started (>&-): known now, said now
            raise ValueError("standard output is closed and the generated text has nowhere to go; redirect it, or name a file with --text-out")
        if real:  # the text side and the description of the checkpoint: config and shard headers, no weights
            describing = time.perf_counter()
            text = model_samples.TextIO.from_checkpoint(args.model)
            checkpoint = model_samples.Checkpoint.describe(args.model)
            model = QwenDense.describe(checkpoint)
            if text.vocab_size > model.geometry.vocab:
                raise ValueError(
                    f"--model {os.path.basename(os.path.normpath(args.model)) or '.'}: the tokenizer has {text.vocab_size:,} ids, the model's "
                    f"embedding table {model.geometry.vocab:,} rows: not the tokenizer of this checkpoint"
                )
            name = os.path.basename(os.path.normpath(args.model)) or "."
            if "-" in args.messages:  # said before a read that waits for whoever writes to standard input
                _say(f"{TAG} --messages -: reading the request from standard input", flush=True)
            requests = [serving_options.prompt_request(prompt, f"--prompt {n + 1}") for n, prompt in enumerate(args.prompt)]
            requests += [serving_options.read_request(path, name) for path in args.messages]
            requests = requests or [serving_options.prompt_request(DEFAULT_PROMPT, "the default prompt")]
            described = time.perf_counter() - describing
        else:
            model = QwenDense.reduced(layers=args.layers)
            text = model_samples.TextIO.synthetic(model.geometry.vocab)
    except NotImplementedError as error:
        _say(f"{TAG} SKIP {error}")
        return 0
    except (ValueError, OSError) as error:
        _say(f"{TAG} ERROR {_reason(error)}")
        return 1
    facts = DeviceFacts.probe(device)
    counts = ",".join(str(n) for n in args.sequences)
    what = checkpoint.summary() if real else "reduced model: " + "; ".join(model.reductions)
    _say(f"{TAG} {model.name} decode | {what}")
    _say(f"{TAG} {facts.name} ({facts.class_text}, {facts.sm_count} SMs) | sequences {counts} | stack {args.stack} | launched from Python, not captured")
    if real:
        _say(f"{TAG} {text.name}")
        _say(f"{TAG} {serving_options.greedy_line(text.generation_defaults)}")
        for request in requests:  # a request body: every field of it, answered
            if request.line():
                _say(f"{TAG} {request.line()}")

    def stop_rule(request, prompt_tokens):
        """The stop rule of one request: its limit of new tokens, held against ``--max-model-len`` before anything is
        planned (``ValueError``); a limit that was given and does not apply is said in a line."""
        if args.perf:  # the mode's fixed count, held against the context and said in the mode's own words
            for note in _perf_limit(args, request, prompt_tokens):
                _say(f"{TAG} {note}")
            return model_samples.StopRule(text.eos_ids, max_new_tokens=args.force_tokens, force_tokens=args.force_tokens)
        flag = args.max_new_tokens if args.limit_given else None
        limit, note = serving_options.new_tokens(request, prompt_tokens, args.max_model_len, flag, args.force_tokens, args.limit_flag)
        if note:
            _say(f"{TAG} {note}")
        return model_samples.StopRule(text.eos_ids, max_new_tokens=limit, force_tokens=args.force_tokens)

    started = time.perf_counter()
    try:  # every prompt of the run is rendered and encoded before anything is planned: the longest request sizes the state
        sets = []  # per prompt set: (prompt ids, the seconds of the text path, its stop rule)
        for request in requests:
            render = time.perf_counter()
            rendered = text.render_chat(request.messages, request.tools, **request.options)
            tokenize = time.perf_counter()
            ids = text.encode(rendered)
            seconds = {"render": tokenize - render, "tokenize": time.perf_counter() - tokenize}
            if not int(ids.numel()):  # before the plan and the load, not as an exception of the loop after both
                raise ValueError(f"{request.shown}: the checkpoint's chat template and tokenizer turn it into no token; there is nothing to decode from")
            again = []  # the same two calls repeated: the first call of a process pays for caches the libraries build
            for _ in range(3):
                render = time.perf_counter()
                text.render_chat(request.messages, request.tools, **request.options)
                tokenize = time.perf_counter()
                text.encode(rendered)
                again.append((tokenize - render, time.perf_counter() - tokenize))
            seconds.update(render_repeated=min(a for a, _ in again), tokenize_repeated=min(b for _, b in again))
            sets.append((ids, seconds, stop_rule(request, int(ids.numel()))))
        seeded = []  # the seeded run: per sequence count one prompt per sequence, all of one length
        for sequences in () if real else args.sequences:
            render = time.perf_counter()
            # the open assistant turn of every prompt ends in the sequence's own token
            rendered = [text.render_chat(_seeded_messages(s), add_generation_prompt=False) for s in range(sequences)]
            tokenize = time.perf_counter()
            prompts = [text.encode(r) for r in rendered]
            seconds = {"render": (tokenize - render) / sequences, "tokenize": (time.perf_counter() - tokenize) / sequences}
            seeded.append((prompts, seconds, stop_rule(serving_options.Request([], None, {}, "the seeded prompt"), int(prompts[0].numel()))))
    except ValueError as error:  # a template that refuses its messages or renders them to nothing; a request beyond the context
        _say(f"{TAG} ERROR {error}")
        return 1
    runs = []  # (session, prompt sets) per sequence count; the later sessions run on the first one's weights
    for index, sequences in enumerate(args.sequences):
        if real:
            prompt_sets = [([ids] * sequences, seconds, rule) for ids, seconds, rule in sets]  # every sequence of a set gets the set's prompt
        else:
            prompt_sets = [seeded[index]]
        session = model_samples.Session(
            model,
            text,
            sequences=sequences,
            stack=args.stack,
            device=device,
            overrides=overrides,
            reference_precision=args.reference_precision,
            capacity=max(_positions(args, int(prompts[0].numel()), rule) for prompts, _, rule in prompt_sets),
            max_model_len=args.max_model_len,
            checkpoint=checkpoint,
            share=runs[0][0] if runs else None,
            record_steps=_record_steps(real, args.perf),
            check_kernels=args.check_kernels,
        )
        runs.append((session, prompt_sets))
    # Every sequence count is planned before the first load, and the memory sentence covers the whole run: a GPU class, a
    # library version or a free-memory budget this sample does not serve is a SKIP here, and nothing skips after it.
    first, later = runs[0][0], [session for session, _ in runs[1:]]
    if args.stack != "reference":  # said before a phase that prints nothing, outside its timed interval
        _say(
            f"{TAG} planning: on a GPU an entry of this stack builds its kernel for each sequence count now, before anything is loaded "
            "(what it needs of the installed toolchain is asked here; a first build prints nothing while it runs)",
            flush=True,
        )
    planning = time.perf_counter()
    try:
        # what a library row of the stack does at its first use beyond an entry's build, said before it is made
        for note in dict.fromkeys(note for session, _ in runs for note in session.resolve().announcements()):
            _say(f"{TAG} planning: {note}", flush=True)
        tables = [first.plan_table(later, references=2 if real else 1)] + [session.resolve().table() for session in later]
    except NotImplementedError as error:
        _say(f"{TAG} SKIP {error}")
        return 0
    except ValueError as error:
        _say(f"{TAG} ERROR {error}")
        return 1
    first.stages.add("import", started - _START - described)
    if real:  # the tokenizer, the chat template, config.json, the shard headers and the request files: no weights
        first.stages.add("describe", described)
    first.stages.add("plan", time.perf_counter() - planning)
    try:  # after the plan: a GPU this sample does not serve skips, whatever is installed
        environment = model_samples.check_environment()
    except model_samples.EnvironmentMismatch as error:
        _say(f"{TAG} ERROR {error}")
        return 1
    _say(f"{TAG} {environment.header()}")
    clock = "clock not locked: timings are for orientation only"
    if args.perf_clock_lock is not None:  # a declaration of the performance mode: its report holds it against the clock it observes
        clock = f"clock lock declared: {args.perf_clock_lock} MHz, set outside this sample, which sets no clock"
    _say(f"{TAG} {clock} | reference precision {args.reference_precision}")
    if real:  # before the load: what a prompt costs before its first new token
        _say(f"{TAG} {serving_options.first_token_line([(request.shown, int(ids.numel())) for request, (ids, _, _) in zip(requests, sets)])}")

    if args.perf:  # before the load: what the mode will run, a rule and counts
        protocol = perf.Protocol(args.perf_warmup, args.perf_steps, args.perf_requests, args.perf_trace_records)
        _say(f"{TAG} {protocol.line(int(runs[0][1][0][0][0].numel()))}")
        observer = perf.observe.Observer(device, args.perf_clock_lock)  # the GPU and the host before the load: the driver is asked, nothing is set
        observer.read("before the load")

    passed, judged, digest, dumped = True, 0, hashlib.sha256(), []
    outcome = None  # of the performance mode
    unobserved, censuses = "", []  # of the kernel check: why the kernels were not observed; the launch records per prompt set
    sink = serving_options.TextSink(args.text_out, streamed=not args.no_stream and not args.perf)  # where the text of sequence 0 goes
    for number, ((session, prompt_sets), table) in enumerate(zip(runs, tables)):
        text_to = sink.later() if number else sink  # a request gets one text: a later sequence count writes none
        _say()
        _say(table)
        if session.check_kernels:  # the kernel table, first part: from the plan alone, before the load
            _say(census.table(session.plan))
        session.load()
        _say(f"{TAG} {session.stages.line()}")
        if session.check_kernels:  # second part: what the entries, now built, state about their launches
            _say(census.stated(session.plan))
        for index, (prompts, seconds, rule) in enumerate(prompt_sets):
            if index:  # a session keeps nothing between two prompt sets
                _say()
                session.reset(fresh=True)
            if real:
                each = f"the same prompt for each of the {session.sequences} sequence(s)"
                _say(f"prompt {index + 1} of {len(prompt_sets)}        {int(prompts[0].numel())} tokens, {each}")
            session.new_report()
            session.report().text_path.update(seconds)
            if args.perf:  # the performance mode: its requests, its checks and its report in the place of the plain loop

                def tokenize(sequences=session.sequences):
                    """The text path of a request: render and tokenize, one prompt per sequence."""
                    if not real:
                        return [text.encode(text.render_chat(_seeded_messages(s), add_generation_prompt=False)) for s in range(sequences)]
                    return [text.encode(text.render_chat(requests[0].messages, requests[0].tools, **requests[0].options))] * sequences

                outcome = perf.run(
                    session,
                    text,
                    tokenize,
                    protocol,
                    _say,
                    name=name if real else f"{model.name}, reduced model",
                    facts=facts,
                    environment=environment,
                    sink=text_to,
                    report=report,
                    verdict=lambda held, blocks, missing: _verdict(args.stack, held, blocks, missing),
                    tag=TAG,
                    real=real,
                    source=requests[0].shown if real else "the seeded prompt",
                    text_path=seconds,
                    observer=observer,
                )
                ok, blocks_judged, produced = outcome.passed, outcome.judged, outcome.produced
            else:
                ok, blocks_judged, produced = _decode(
                    session,
                    text,
                    rule,
                    prompts,
                    tie_out,
                    check=args.stack != "reference",
                    checked_steps=CHECKED_STEPS if real else None,
                    sink=text_to,
                )
            passed, judged = passed and ok, judged + blocks_judged
            if session.check_kernels and session.traced is not None:  # None: a run of the performance mode that ended at its first checks
                unobserved = unobserved or session.traced.unobserved
                censuses.append({"sequences": session.sequences, "prompt": index, **session.traced.dump()})
            for ids in produced:
                digest.update(b"".join(int(i).to_bytes(4, "little") for i in ids))
            # one prompt and one list of produced ids per sequence: in the seeded run every sequence has a prompt of its own
            dumped.append({"sequences": session.sequences, "prompt": index, "prompt_ids": [[int(i) for i in p] for p in prompts], "ids": produced})
    if checkpoint is not None:
        checkpoint.close()
    sink.close()
    gib = 2.0**30
    grown = (facts.free_bytes - torch.cuda.mem_get_info(device)[0]) / gib
    _say()
    peak, reserved = torch.cuda.max_memory_allocated(device) / gib, torch.cuda.max_memory_reserved(device) / gib
    if outcome is not None:  # the performance mode resets PyTorch's peaks behind its checks: the process's peak is the larger of the two
        peak, reserved = max(peak, outcome.peaks[0] / gib), max(reserved, outcome.peaks[1] / gib)
    _say(
        f"{TAG} GPU memory: PyTorch peak {peak:.2f} GiB allocated, {reserved:.2f} GiB reserved; in use on the device grew by {grown:.2f} GiB "
        "since the plan (PyTorch's pool, CUDA libraries and kernel reservations of this process; other processes' changes included)"
    )
    mib = 2.0**20
    allowance = getattr(model, "first_launch_reserve", {}).get(f"cc{facts.cc[0]}.x", 0)
    for name, block in first.plan.items():
        launches = block.first_launch()  # measured by the block: rows -> device bytes outside PyTorch's pool
        if launches:
            forms = " | ".join(f"{rows} row{'s' if rows > 1 else ''} {taken / mib:,.0f} MiB" for rows, taken in launches.items())
            allowed = f"the plan allowed {allowance / mib:,.0f} MiB on this class" if allowance else "the plan allows none on this class"
            measured = f"measured in this process, other processes' changes included; {allowed}"
            _say(f"{TAG} first launch of {name}, device memory outside PyTorch's pool: {forms} ({measured})")
    unwritten = []  # what a flag asked for and is not there
    if sink.error:  # the reader of standard output went away, or the file of --text-out stopped taking text
        unwritten.append(f"the generated text could not be written to {sink.name}: {sink.error}")
    if outcome is not None and outcome.unwritten:  # the report file of the performance mode
        unwritten.append(outcome.unwritten)
    if args.dump_tokens is not None:
        error = _dump(args.dump_tokens, args.stack, dumped)
        if error:
            unwritten.append(error)
        else:
            _say(f"{TAG} token ids of {len(dumped)} prompt set(s) written to {os.path.basename(args.dump_tokens)}")
    if args.dump_census is not None:
        if not any(run["steps"] for run in censuses):  # no launch record was kept: there is nothing to write
            why = f"kernels not observed: {unobserved}" if unobserved else "the kernels line of the checks says why"
            error = f"--dump-census {os.path.basename(os.path.normpath(args.dump_census))}: no launch records to write ({why})"
        else:
            gpu = {"name": facts.name, "compute_capability": list(facts.cc), "sm_count": facts.sm_count}
            traced_steps = TRACED_STEPS if outcome is None else outcome.data["traced"]["steps"]
            error = _dump(args.dump_census, args.stack, censuses, "--dump-census", schema=3, gpu=gpu, traced_steps=traced_steps)
        if error:
            unwritten.append(error)
        else:
            _say(f"{TAG} launch records of {len(censuses)} traced pass(es) written to {os.path.basename(args.dump_census)}")
    if real:
        _say(f"{TAG} checkpoint weights, greedy decoding; ids hash {digest.hexdigest()[:12]}")
    else:
        _say(f"{TAG} seeded weights: the output is not language; ids hash {digest.hexdigest()[:12]}")
    verdict = _verdict(args.stack, passed, judged, unobserved)
    if unwritten:  # the run is done and its verdict stands, but what was asked for is not there
        _say(f"{TAG} ERROR {'; '.join(unwritten)} (the run itself: {verdict})")
        return 1
    _say(f"{TAG} {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
