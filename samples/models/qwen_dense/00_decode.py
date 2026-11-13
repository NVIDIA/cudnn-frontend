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
``--max-new-tokens``, the text is printed, and the same checks run on the first steps of every prompt.  That needs the
packages of ``samples/models/requirements.txt``, the checkpoint on local disk (your own download) and a GPU with room for a
bf16 copy of every weight: about 57 GiB free for the 27B checkpoint.

    python samples/models/qwen_dense/00_decode.py
    python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
    python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: reports RAN, never PASS
    python samples/models/qwen_dense/00_decode.py --check-kernels          # also: every launch of the traced steps against the plan

    pip install -r samples/models/requirements.txt                         # only for a checkpoint; brings the hf command
    hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir /data/Qwen3.8-27B-NVFP4   # your own step: the sample fetches nothing
    python samples/models/qwen_dense/00_decode.py --model /data/Qwen3.8-27B-NVFP4 --prompt "Why is the sky blue?" --max-new-tokens 256

The stack is explicit and never chosen by what is installed.  ``fe-cublas`` is a reference path, not a fast path: the
linear-attention step runs on cuDNN Frontend's fused kernel, every multiply on cuBLAS over a bf16 copy of the quantised weights,
the rest in PyTorch, and every launch is issued from Python.  Timings printed here are for orientation only.

``--check-kernels`` adds a fifth check, against the plan instead of the reference: the run prints which kernels every block
launches (the kernel table, before the load) and what the entries state about their launches (after it), keeps the profiler's
record of every launch of the traced steps, and holds them against each other -- every launch attributed to a block, the
kernels of an entry by name, order, count, grid, block and shared memory.  Where the profiler cannot run the run ends in
``[00] RAN <stack> stack (kernels not observed: <reason>)``, never in ``PASS``.  ``--dump-census FILE`` writes those launch
records as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
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
if _HELPERS not in sys.path:
    sys.path.insert(0, _HELPERS)

TAG = "[00]"
STACKS = ("fe-cublas", "reference")
DEFAULT_PROMPT = "Why is the sky blue?"
CHECKED_STEPS = 16  # with a checkpoint the checks cover the first steps of every prompt
TRACED_STEPS = 32


def _reason(error) -> str:
    """The text of an error for an ``ERROR`` line: a file is named by its own name, never by a path."""
    if isinstance(error, OSError) and error.filename:
        return f"{os.path.basename(os.path.normpath(str(error.filename)))}: {error.strerror or type(error).__name__}"
    return str(error)


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
    file by its own name."""
    target = os.path.abspath(path)
    there = os.path.exists(target)
    if os.path.isdir(target) or not os.access(target if there else os.path.dirname(target), os.W_OK):
        raise ValueError(f"{flag} {os.path.basename(os.path.normpath(path))}: cannot be written there")


EPILOG = """\
The default run needs no checkpoint and no extra package; it needs a GPU of compute capability 10.x or 12.x.
With --model: pip install -r samples/models/requirements.txt first.  The checkpoint is your own download (nothing is
fetched); the 27B checkpoint needs 22 GB of disk and about 57 GiB of free GPU memory on the stack fe-cublas.
The checkpoint, the memory, the checks and every flag: samples/models/qwen_dense/README.md
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
        help="with --model: chat messages as JSON (a list, or an object with messages, tools, chat_template_kwargs); repeatable",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, metavar="N", help="with --model: stop after N new tokens unless the end token comes first (default 256)"
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
        help="sequences per run, fixed before the first step; a list repeats the run per count (default 1,4; with --model 1)",
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
        help="fe-cublas (default): cuDNN Frontend + cuBLAS + PyTorch, checked against the reference; reference: every block in PyTorch, "
        "on any CUDA GPU, nothing checked.  Explicit: never chosen by what is installed",
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
        "Where the profiler cannot run the run reports RAN (kernels not observed), never PASS; not with --stack reference",
    )
    parser.add_argument(
        "--dump-census",
        metavar="FILE",
        help="write the launch records of the traced steps as JSON (kernel, grid, block, shared memory per launch), for a comparison "
        "made elsewhere; implies --check-kernels",
    )
    args = parser.parse_args(argv)
    args.check_kernels = args.check_kernels or args.dump_census is not None
    if args.check_kernels and args.stack == "reference":
        parser.error("--check-kernels and --dump-census are for a provider stack; --stack reference checks nothing")
    if args.model is None and (args.prompt or args.messages):
        parser.error("--prompt and --messages need --model: the seeded model has no language")
    if args.model is not None and args.layers is not None:
        parser.error("--layers is for the reduced model; a checkpoint has the depth it has")
    if args.max_new_tokens is not None and args.model is None:  # the seeded run decodes a fixed count: the flag would be ignored
        parser.error("--max-new-tokens is for --model; the seeded run decodes --force-tokens N (default 32)")
    if args.max_new_tokens is not None and args.force_tokens is not None:
        parser.error("--force-tokens N decodes exactly N tokens; --max-new-tokens does not apply beside it")
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
        or (args.force_tokens is not None and args.force_tokens < 1)
        or (args.layers is not None and args.layers < 1)
    ):
        parser.error("--sequences, --layers, --max-new-tokens and --force-tokens take positive integers")
    args.layers = 2 if args.layers is None else args.layers
    return args


def _verdict(stack: str, passed: bool, judged: int, unobserved: str = "") -> str:
    """The last line of a run.  ``PASS`` says that provider blocks were judged against the reference and held.  A run on the
    reference stack, or one whose ``--provider`` flags put every judged block on its reference row, compared the reference
    with itself: it reports ``RAN``.  So does a run that was asked to check its kernels and could not observe them
    (``unobserved``: why): its other checks held, the one it was asked for did not run.  A failed check is ``FAIL`` whatever
    else is true."""
    if stack == "reference":
        return "RAN reference stack (no provider checked)"
    if not passed:
        return "FAIL"
    if unobserved:
        return f"RAN {stack} stack (kernels not observed: {unobserved})"
    return "PASS" if judged else "RAN (no provider block was judged)"


def _decode(session, text, rule, prompts, tie_out, check, checked_steps, kernels=False):
    """One prompt set: prefill, the decode loop (the one host read of a step is here), the traced pass, the report and the
    checks.  Returns ``(passed, provider blocks judged, tokens per sequence)``.  With ``kernels`` the traced pass keeps its
    launch records and the checks include the kernel check; its outcome stays on the session (``session.traced``).

    Two lines say what comes next before the phases that print nothing for a while (the prompt and the decode loop; the
    checks).  They are printed outside every timed interval, and flushed, so a run through a pipe shows them too."""
    new_tokens = f"{rule.limit} new tokens" if rule.force_tokens is not None else f"up to {rule.limit} new tokens"
    print(
        f"{TAG} decoding: {int(prompts[0].numel())} prompt tokens through the decode step, then {new_tokens} and a traced pass; "
        "the report and the text follow, the text is not streamed",
        flush=True,
    )
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
                pieces[s] += streams[s].push(token)
                done[s] = rule.ends(token) or len(produced[s]) >= rule.limit
        text_seconds += time.perf_counter() - started
        if all(done):
            break
        next_ids = session.step(next_ids)  # embedding, the layers, the output head, token choice: ids stay on the GPU
        tokens = session.read(next_ids)  # the one host read of a step: 4 bytes per sequence
    session.report().text_path["text_out"] = text_seconds / max(sum(len(ids) for ids in produced), 1)
    session.trace(TRACED_STEPS, records=kernels)  # the first steps again, under the profiler: GPU busy, idle, launches per step
    print(session.report().text())
    if getattr(text, "synthetic_text", True):
        print(f"text out              sequence 0, {len(produced[0])} tokens (seeded weights: not language): {pieces[0][:48]!r}")
    else:
        ended = "ended at the end token" if rule.ends(produced[0][-1]) else f"stopped after {len(produced[0])} tokens"
        print(f"text out              sequence 0, {len(produced[0])} tokens, {ended}:")
        print("\n".join("  | " + line for line in (pieces[0] + streams[0].close()).splitlines()))
    if not check:
        return True, 0, produced
    steps = "every step" if checked_steps is None else f"the first {checked_steps} steps"
    also = "; the kernels of the traced steps against the plan" if kernels else ""
    print(
        f"{TAG} checks on {steps} against the {session.reference_precision}-accumulated reference (replay, every provider block, the tokens){also}",
        flush=True,
    )
    checks = tie_out.run_checks(
        session, prompt_tokens=int(prompts[0].numel()), max_steps=checked_steps, second_reference=checked_steps is not None, kernels=kernels
    )
    print(checks.text())
    return checks.ok, checks.judged, produced


def main(argv=None) -> int:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    try:
        import cudnn  # noqa: F401  before torch: the first cuDNN library loaded wins
        import torch
    except ImportError as error:
        print(f"{TAG} SKIP needs nvidia-cudnn-frontend and PyTorch ({error}): see samples/models/README.md")
        return 0
    import model_samples
    from model_samples import census, text_io, tie_out
    from model_samples.blocks import DeviceFacts, parse_overrides
    from model_samples.qwen_dense import QwenDense

    if not torch.cuda.is_available():
        print(f"{TAG} SKIP needs a CUDA GPU")
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    real = args.model is not None
    checkpoint, requests, described = None, [], 0.0
    try:
        overrides = parse_overrides(args.provider)
        for flag, path in (("--dump-tokens", args.dump_tokens), ("--dump-census", args.dump_census)):
            if path:  # a file that cannot be written is an error before the run, not after it
                _writable(flag, path)
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
            requests = [([{"role": "user", "content": prompt}], None, {}, f"--prompt {n + 1}") for n, prompt in enumerate(args.prompt)]
            requests += [(*text_io.load_messages(path), f"--messages {os.path.basename(path)}") for path in args.messages]
            requests = requests or [([{"role": "user", "content": DEFAULT_PROMPT}], None, {}, "the default prompt")]
            described = time.perf_counter() - describing
        else:
            model = QwenDense.reduced(layers=args.layers)
            text = model_samples.TextIO.synthetic(model.geometry.vocab)
    except NotImplementedError as error:
        print(f"{TAG} SKIP {error}")
        return 0
    except (ValueError, OSError) as error:
        print(f"{TAG} ERROR {_reason(error)}")
        return 1
    facts = DeviceFacts.probe(device)
    rule = model_samples.StopRule(text.eos_ids, max_new_tokens=args.max_new_tokens, force_tokens=args.force_tokens)
    counts = ",".join(str(n) for n in args.sequences)
    what = checkpoint.summary() if real else "reduced model: " + "; ".join(model.reductions)
    print(f"{TAG} {model.name} decode | {what}")
    print(f"{TAG} {facts.name} ({facts.class_text}, {facts.sm_count} SMs) | sequences {counts} | stack {args.stack} | launched from Python, not captured")
    if real:
        print(f"{TAG} {text.name}")

    started = time.perf_counter()
    try:  # every prompt of the run is rendered and encoded before anything is planned: the longest one sizes the state
        sets = []  # per prompt set: (prompt ids, the seconds of the text path)
        for messages, tools, options, shown in requests:
            render = time.perf_counter()
            rendered = text.render_chat(messages, tools, **options)
            tokenize = time.perf_counter()
            ids = text.encode(rendered)
            seconds = {"render": tokenize - render, "tokenize": time.perf_counter() - tokenize}
            if not int(ids.numel()):  # before the plan and the load, not as an exception of the loop after both
                raise ValueError(f"{shown}: the checkpoint's chat template and tokenizer turn it into no token; there is nothing to decode from")
            again = []  # the same two calls repeated: the first call of a process pays for caches the libraries build
            for _ in range(3):
                render = time.perf_counter()
                text.render_chat(messages, tools, **options)
                tokenize = time.perf_counter()
                text.encode(rendered)
                again.append((tokenize - render, time.perf_counter() - tokenize))
            seconds.update(render_repeated=min(a for a, _ in again), tokenize_repeated=min(b for _, b in again))
            sets.append((ids, seconds))
    except ValueError as error:  # a template that refuses its messages, or renders them to nothing
        print(f"{TAG} ERROR {error}")
        return 1
    longest = max((int(ids.numel()) for ids, _ in sets), default=0)
    runs = []  # (session, prompt sets) per sequence count; the later sessions run on the first one's weights
    for sequences in args.sequences:
        if real:
            prompt_sets = [([ids] * sequences, seconds) for ids, seconds in sets]  # every sequence of a set gets the set's prompt
        else:
            render = time.perf_counter()
            # one seeded prompt per sequence, all of one length; the open assistant turn ends in the sequence's own token
            rendered = [
                text.render_chat([{"role": "user", "content": "count"}, {"role": "assistant", "content": str(s % 10)}], add_generation_prompt=False)
                for s in range(sequences)
            ]
            tokenize = time.perf_counter()
            prompts = [text.encode(r) for r in rendered]
            prompt_sets = [(prompts, {"render": (tokenize - render) / sequences, "tokenize": (time.perf_counter() - tokenize) / sequences})]
            longest = int(prompts[0].numel())
        session = model_samples.Session(
            model,
            text,
            sequences=sequences,
            stack=args.stack,
            device=device,
            overrides=overrides,
            reference_precision=args.reference_precision,
            capacity=longest + rule.limit,
            checkpoint=checkpoint,
            share=runs[0][0] if runs else None,
            record_steps=max(CHECKED_STEPS, TRACED_STEPS) if real else None,
            check_kernels=args.check_kernels,
        )
        runs.append((session, prompt_sets))
    # Every sequence count is planned before the first load, and the memory sentence covers the whole run: a GPU class, a
    # library version or a free-memory budget this sample does not serve is a SKIP here, and nothing skips after it.
    first, later = runs[0][0], [session for session, _ in runs[1:]]
    planning = time.perf_counter()
    try:
        tables = [first.plan_table(later)] + [session.resolve().table() for session in later]
    except NotImplementedError as error:
        print(f"{TAG} SKIP {error}")
        return 0
    except ValueError as error:
        print(f"{TAG} ERROR {error}")
        return 1
    first.stages.add("import", started - _START - described)
    if real:  # the tokenizer, the chat template, config.json, the shard headers and the request files: no weights
        first.stages.add("describe", described)
    first.stages.add("plan", time.perf_counter() - planning)
    try:  # after the plan: a GPU this sample does not serve skips, whatever is installed
        environment = model_samples.check_environment()
    except model_samples.EnvironmentMismatch as error:
        print(f"{TAG} ERROR {error}")
        return 1
    print(f"{TAG} {environment.header()}")
    print(f"{TAG} clock not locked: timings are for orientation only | reference precision {args.reference_precision}")

    passed, judged, digest, dumped = True, 0, hashlib.sha256(), []
    unobserved, censuses = "", []  # of the kernel check: why the kernels were not observed; the launch records per prompt set
    for (session, prompt_sets), table in zip(runs, tables):
        print()
        print(table)
        if args.check_kernels:  # the kernel table, first part: from the plan alone, before the load
            print(census.table(session.plan))
        session.load()
        print(f"{TAG} {session.stages.line()}")
        if args.check_kernels:  # second part: what the entries, now built, state about their launches
            print(census.stated(session.plan))
        for index, (prompts, seconds) in enumerate(prompt_sets):
            if index:  # a session keeps nothing between two prompt sets
                print()
                session.reset(fresh=True)
            if real:
                each = f"the same prompt for each of the {session.sequences} sequence(s)"
                print(f"prompt {index + 1} of {len(prompt_sets)}        {int(prompts[0].numel())} tokens, {each}")
            session.new_report()
            session.report().text_path.update(seconds)
            ok, blocks_judged, produced = _decode(
                session,
                text,
                rule,
                prompts,
                tie_out,
                check=args.stack != "reference",
                checked_steps=CHECKED_STEPS if real else None,
                kernels=args.check_kernels,
            )
            passed, judged = passed and ok, judged + blocks_judged
            if args.check_kernels:
                unobserved = unobserved or session.traced.unobserved
                censuses.append({"sequences": session.sequences, "prompt": index, **session.traced.dump()})
            for ids in produced:
                digest.update(b"".join(int(i).to_bytes(4, "little") for i in ids))
            # one prompt and one list of produced ids per sequence: in the seeded run every sequence has a prompt of its own
            dumped.append({"sequences": session.sequences, "prompt": index, "prompt_ids": [[int(i) for i in p] for p in prompts], "ids": produced})
    if checkpoint is not None:
        checkpoint.close()
    gib = 2.0**30
    grown = (facts.free_bytes - torch.cuda.mem_get_info(device)[0]) / gib
    print()
    peak, reserved = torch.cuda.max_memory_allocated(device) / gib, torch.cuda.max_memory_reserved(device) / gib
    print(
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
            print(f"{TAG} first launch of {name}, device memory outside PyTorch's pool: {forms} ({measured})")
    unwritten = []  # what a flag asked for and is not there
    if args.dump_tokens:
        error = _dump(args.dump_tokens, args.stack, dumped)
        if error:
            unwritten.append(error)
        else:
            print(f"{TAG} token ids of {len(dumped)} prompt set(s) written to {os.path.basename(args.dump_tokens)}")
    if args.dump_census:
        if not any(run["steps"] for run in censuses):  # no launch record was kept: there is nothing to write
            why = f"kernels not observed: {unobserved}" if unobserved else "the kernels line of the checks says why"
            error = f"--dump-census {os.path.basename(os.path.normpath(args.dump_census))}: no launch records to write ({why})"
        else:
            gpu = {"name": facts.name, "compute_capability": list(facts.cc), "sm_count": facts.sm_count}
            error = _dump(args.dump_census, args.stack, censuses, "--dump-census", schema=1, gpu=gpu, traced_steps=TRACED_STEPS)
        if error:
            unwritten.append(error)
        else:
            print(f"{TAG} launch records of {len(censuses)} traced pass(es) written to {os.path.basename(args.dump_census)}")
    if real:
        print(f"{TAG} checkpoint weights, greedy decoding; ids hash {digest.hexdigest()[:12]}")
    else:
        print(f"{TAG} seeded weights: the output is not language; ids hash {digest.hexdigest()[:12]}")
    verdict = _verdict(args.stack, passed, judged, unobserved)
    if unwritten:  # the run is done and its verdict stands, but what was asked for is not there
        print(f"{TAG} ERROR {'; '.join(unwritten)} (the run itself: {verdict})")
        return 1
    print(f"{TAG} {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
