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
packages of ``samples/models/requirements.txt`` and a GPU with room for a bf16 copy of every weight.

    python samples/models/qwen_dense/00_decode.py
    python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
    python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: reports RAN, never PASS
    python samples/models/qwen_dense/00_decode.py --model /data/checkpoint --prompt "Why is the sky blue?" --max-new-tokens 256

The stack is explicit and never chosen by what is installed.  ``fe-cublas`` is a reference path, not a fast path: the
linear-attention step runs on cuDNN Frontend's fused kernel, every multiply on cuBLAS over a bf16 copy of the quantised weights,
the rest in PyTorch, and every launch is issued from Python.  Timings printed here are for orientation only.
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


def _dump(path: str, stack: str, runs: list) -> str:
    """Write the prompt ids and the produced ids of a run as JSON.  Returns "" or, where the file cannot be written after
    all (the check before the run passed: a full disk, a folder that went away), the reason with the file's own name."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sample": "qwen_dense/00_decode", "stack": stack, "greedy": True, "runs": runs}, f)
    except OSError as error:
        return f"--dump-tokens {os.path.basename(os.path.normpath(path))}: {error.strerror or type(error).__name__}"
    return ""


def _arguments(argv):
    parser = argparse.ArgumentParser(
        description="Plain decode of the dense Qwen model: the reduced seeded model (self-checking), or a checkpoint from local disk."
    )
    parser.add_argument("--model", metavar="DIR", help="a checkpoint directory on local disk; never a hub name, nothing is downloaded")
    parser.add_argument("--prompt", action="append", default=[], metavar="TEXT", help="one user message; repeat for several prompts, run one after the other")
    parser.add_argument(
        "--messages",
        action="append",
        default=[],
        metavar="FILE",
        help="chat messages as JSON (a list, or an object with messages, tools, chat_template_kwargs)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, metavar="N", help="stop after N new tokens unless the end token comes first (default 256)")
    parser.add_argument("--force-tokens", type=int, default=None, metavar="N", help="decode exactly N tokens, ignoring the end token (the seeded run: 32)")
    parser.add_argument(
        "--sequences", default=None, help="sequences per run, fixed before the first step; a list repeats the run per count (default 1,4; with --model 1)"
    )
    parser.add_argument("--layers", type=int, default=None, help="layers of the reduced model (default 2: one linear-attention + one attention layer)")
    parser.add_argument("--stack", default="fe-cublas", choices=STACKS, help="explicit; never chosen by what is installed")
    parser.add_argument("--provider", action="append", default=[], metavar="BLOCK=KEY", help="select one declared provider row of a block")
    parser.add_argument("--reference-precision", default="fp64", choices=("fp64", "ieee-fp32"), help="accumulation of the reference blocks")
    parser.add_argument("--dump-tokens", metavar="FILE", help="write the prompt ids and the produced ids as JSON, for a comparison made elsewhere")
    args = parser.parse_args(argv)
    if args.model is None and (args.prompt or args.messages):
        parser.error("--prompt and --messages need --model: the seeded model has no language")
    if args.model is not None and args.layers is not None:
        parser.error("--layers is for the reduced model; a checkpoint has the depth it has")
    try:
        args.sequences = [int(n) for n in str(args.sequences or ("1" if args.model else "1,4")).split(",")]
    except ValueError:
        parser.error("--sequences takes N or N,N,...")
    if args.force_tokens is None and args.model is None:
        args.force_tokens = 32
    if (
        min(args.sequences) < 1
        or args.max_new_tokens < 1
        or (args.force_tokens is not None and args.force_tokens < 1)
        or (args.layers is not None and args.layers < 1)
    ):
        parser.error("--sequences, --layers, --max-new-tokens and --force-tokens take positive integers")
    args.layers = 2 if args.layers is None else args.layers
    return args


def _verdict(stack: str, passed: bool, judged: int) -> str:
    """The last line of a run.  ``PASS`` says that provider blocks were judged against the reference and held.  A run on the
    reference stack, or one whose ``--provider`` flags put every judged block on its reference row, compared the reference
    with itself: it reports ``RAN``."""
    if stack == "reference":
        return "RAN reference stack (no provider checked)"
    if not passed:
        return "FAIL"
    return "PASS" if judged else "RAN (no provider block was judged)"


def _decode(session, text, rule, prompts, tie_out, check, checked_steps):
    """One prompt set: prefill, the decode loop (the one host read of a step is here), the traced pass, the report and the
    checks.  Returns ``(passed, provider blocks judged, tokens per sequence)``.

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
    session.trace(TRACED_STEPS)  # the first steps again, under the profiler: GPU busy, idle, launches per step
    print(session.report().text())
    if getattr(text, "synthetic_text", True):
        print(f"text out              sequence 0, {len(produced[0])} tokens: {pieces[0][:48]!r}")
    else:
        ended = "ended at the end token" if rule.ends(produced[0][-1]) else f"stopped after {len(produced[0])} tokens"
        print(f"text out              sequence 0, {len(produced[0])} tokens, {ended}:")
        print("\n".join("  | " + line for line in (pieces[0] + streams[0].close()).splitlines()))
    if not check:
        return True, 0, produced
    steps = "every step" if checked_steps is None else f"the first {checked_steps} steps"
    print(f"{TAG} checks on {steps} against the {session.reference_precision}-accumulated reference (replay, every provider block, the tokens)", flush=True)
    checks = tie_out.run_checks(session, prompt_tokens=int(prompts[0].numel()), max_steps=checked_steps, second_reference=checked_steps is not None)
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
    from model_samples import text_io, tie_out
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
        if args.dump_tokens:  # a file that cannot be written is an error before the run, not after it
            target = os.path.abspath(args.dump_tokens)
            there = os.path.exists(target)
            if os.path.isdir(target) or not os.access(target if there else os.path.dirname(target), os.W_OK):
                raise ValueError(f"--dump-tokens {os.path.basename(os.path.normpath(args.dump_tokens))}: cannot be written there")
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
    for (session, prompt_sets), table in zip(runs, tables):
        print()
        print(table)
        session.load()
        print(f"{TAG} {session.stages.line()}")
        for index, (prompts, seconds) in enumerate(prompt_sets):
            if index:  # a session keeps nothing between two prompt sets
                print()
                session.reset(fresh=True)
            if real:
                print(
                    f"prompt {index + 1} of {len(prompt_sets)}        {int(prompts[0].numel())} tokens, the same prompt for each of the {session.sequences} sequence(s)"
                )
            session.new_report()
            session.report().text_path.update(seconds)
            ok, blocks_judged, produced = _decode(
                session, text, rule, prompts, tie_out, check=args.stack != "reference", checked_steps=CHECKED_STEPS if real else None
            )
            passed, judged = passed and ok, judged + blocks_judged
            for ids in produced:
                digest.update(b"".join(int(i).to_bytes(4, "little") for i in ids))
            dumped.append({"sequences": session.sequences, "prompt": index, "prompt_ids": [int(i) for i in prompts[0]], "ids": produced})
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
            print(
                f"{TAG} first launch of {name}, device memory outside PyTorch's pool: {forms} (measured in this process, other processes' changes included; {allowed})"
            )
    unwritten = _dump(args.dump_tokens, args.stack, dumped) if args.dump_tokens else ""
    if args.dump_tokens and not unwritten:
        print(f"{TAG} token ids of {len(dumped)} prompt set(s) written to {os.path.basename(args.dump_tokens)}")
    if real:
        print(f"{TAG} checkpoint weights, greedy decoding; ids hash {digest.hexdigest()[:12]}")
    else:
        print(f"{TAG} seeded weights: the output is not language; ids hash {digest.hexdigest()[:12]}")
    verdict = _verdict(args.stack, passed, judged)
    if unwritten:  # the run is done and its verdict stands, but what was asked for is not there
        print(f"{TAG} ERROR {unwritten} (the run itself: {verdict})")
        return 1
    print(f"{TAG} {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
