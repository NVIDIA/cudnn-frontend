# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 00: plain decode of the dense Qwen 3.5-architecture model, token ids in and token ids out, in one process.

The default run needs no checkpoint and no extra package: it builds the model at its published layer widths with two layers
and seeded weights, prints which provider serves every block, feeds one seeded prompt per sequence through the decode step,
decodes 32 tokens greedily at 1 and at 4 sequences, and checks the whole path against a plain-PyTorch reference that
accumulates in fp64 (replay, every provider block teacher-forced, the greedy tokens, sequence 0 alone against sequence 0 in the
batch).  It prints ``[00] PASS``, or ``[00] SKIP <requirement>`` with exit code 0 on a GPU, a library version or a free-memory
budget it does not serve.

    python samples/models/qwen_dense/00_decode.py
    python samples/models/qwen_dense/00_decode.py --sequences 4 --layers 4
    python samples/models/qwen_dense/00_decode.py --stack reference        # every block on the reference: reports RAN, never PASS

The stack is explicit and never chosen by what is installed.  ``fe-cublas`` is a reference path, not a fast path: the
linear-attention step runs on cuDNN Frontend's fused kernel, every multiply on cuBLAS over a bf16 copy of the quantised weights,
the rest in PyTorch, and every launch is issued from Python.  Timings printed here are for orientation only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

_START = time.perf_counter()
_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS = os.path.normpath(os.path.join(_HERE, ".."))  # samples/models: the home of the model_samples package
if _HELPERS not in sys.path:
    sys.path.insert(0, _HELPERS)

TAG = "[00]"
STACKS = ("fe-cublas", "reference")


def _arguments(argv):
    parser = argparse.ArgumentParser(description="Plain decode of the dense Qwen model on the reduced seeded model (self-checking).")
    parser.add_argument("--sequences", default="1,4", help="sequences per run, fixed before the first step; a list repeats the run per count (default 1,4)")
    parser.add_argument("--layers", type=int, default=2, help="layers of the reduced model (default 2: one linear-attention + one attention layer)")
    parser.add_argument("--stack", default="fe-cublas", choices=STACKS, help="explicit; never chosen by what is installed")
    parser.add_argument("--provider", action="append", default=[], metavar="BLOCK=KEY", help="select one declared provider row of a block")
    parser.add_argument("--force-tokens", type=int, default=32, metavar="N", help="decode exactly N tokens, ignoring the end token (default 32)")
    parser.add_argument("--reference-precision", default="fp64", choices=("fp64", "ieee-fp32"), help="accumulation of the reference blocks")
    args = parser.parse_args(argv)
    try:
        args.sequences = [int(n) for n in str(args.sequences).split(",")]
    except ValueError:
        parser.error("--sequences takes N or N,N,...")
    if not args.sequences or min(args.sequences) < 1 or args.force_tokens < 1:
        parser.error("--sequences and --force-tokens take positive integers")
    return args


def _decode(session, text, rule, prompts, tie_out, check):
    """One prompt set: prefill, the decode loop (the one host read of a step is here), the report and the checks.
    Returns ``(passed, tokens per sequence)``."""
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
    print(session.report().text())
    print(f"text out              sequence 0, {len(produced[0])} tokens: {pieces[0][:48]!r}")
    if not check:
        return True, produced
    checks = tie_out.run_checks(session, prompt_tokens=int(prompts[0].numel()))
    print(checks.text())
    return checks.ok, produced


def main(argv=None) -> int:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    try:
        import cudnn  # noqa: F401  before torch: the first cuDNN library loaded wins
        import torch
    except ImportError as error:
        print(f"{TAG} SKIP needs nvidia-cudnn-frontend and PyTorch ({error}): see samples/models/README.md")
        return 0
    import model_samples
    from model_samples import tie_out
    from model_samples.blocks import DeviceFacts, parse_overrides
    from model_samples.qwen_dense import QwenDense

    try:
        environment = model_samples.check_environment()
    except model_samples.EnvironmentMismatch as error:
        print(f"{TAG} ERROR {error}")
        return 1
    if not torch.cuda.is_available():
        print(f"{TAG} SKIP needs a CUDA GPU")
        return 0
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        model = QwenDense.reduced(layers=args.layers)
        overrides = parse_overrides(args.provider)
    except ValueError as error:
        print(f"{TAG} ERROR {error}")
        return 1
    facts = DeviceFacts.probe(device)
    text = model_samples.TextIO.synthetic(model.geometry.vocab)
    rule = model_samples.StopRule(text.eos_ids, force_tokens=args.force_tokens)
    counts = ",".join(str(n) for n in args.sequences)
    print(f"{TAG} {model.name} decode | reduced model: {'; '.join(model.reductions)}")
    print(f"{TAG} {facts.name} ({facts.class_text}, {facts.sm_count} SMs) | sequences {counts} | stack {args.stack} | launched from Python, not captured")
    print(f"{TAG} {environment.header()}")
    print(f"{TAG} clock not locked: timings are for orientation only | reference precision {args.reference_precision}")

    shared, passed, digest = None, True, hashlib.sha256()
    for sequences in args.sequences:
        started = time.perf_counter()
        render = time.perf_counter()
        # one seeded prompt per sequence, all of one length; the open assistant turn ends in the sequence's own token
        rendered = [
            text.render_chat([{"role": "user", "content": "count"}, {"role": "assistant", "content": str(s % 10)}], add_generation_prompt=False)
            for s in range(sequences)
        ]
        tokenize = time.perf_counter()
        prompts = [text.encode(r) for r in rendered]
        encoded = time.perf_counter()
        session = model_samples.Session(
            model,
            text,
            sequences=sequences,
            stack=args.stack,
            device=device,
            overrides=overrides,
            reference_precision=args.reference_precision,
            capacity=int(prompts[0].numel()) + rule.limit,
            share=shared,
        )
        try:
            table = session.plan_table()
        except NotImplementedError as error:
            print(f"{TAG} SKIP {error}")
            return 0
        except ValueError as error:
            print(f"{TAG} ERROR {error}")
            return 1
        print()
        print(table)
        if shared is None:
            session.stages.add("import", started - _START)
        session.stages.add("plan", time.perf_counter() - encoded)
        session.load()
        print(f"{TAG} {session.stages.line()}")
        report = session.report()
        report.text_path.update(render=(tokenize - render) / sequences, tokenize=(encoded - tokenize) / sequences)
        ok, produced = _decode(session, text, rule, prompts, tie_out, check=args.stack != "reference")
        passed = passed and ok
        for ids in produced:
            digest.update(b"".join(int(i).to_bytes(4, "little") for i in ids))
        shared = shared or session
    gib = 2.0**30
    grown = (facts.free_bytes - torch.cuda.mem_get_info(device)[0]) / gib
    print()
    print(
        f"{TAG} GPU memory: PyTorch peak {torch.cuda.max_memory_allocated(device) / gib:.2f} GB allocated, {torch.cuda.max_memory_reserved(device) / gib:.2f} GB reserved; "
        f"in use on the device grew by {grown:.2f} GB since the plan (PyTorch's pool, CUDA libraries and kernel reservations of this process; other processes' changes included)"
    )
    print(f"{TAG} seeded weights: the output is not language; ids hash {digest.hexdigest()[:12]}")
    if args.stack == "reference":
        print(f"{TAG} RAN reference stack (no provider checked)")
        return 0
    print(f"{TAG} {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
