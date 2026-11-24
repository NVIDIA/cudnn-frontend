# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Session``: plan, load, prefill, step for a fixed set of sequences.

A session decides everything before it loads anything: ``plan_table()`` resolves the family's provider table for this GPU, the
number of sequences and the stack, and states the memory the run needs -- for this session and for the sessions that will
follow it on its weights, so that nothing can decline once the first of them has loaded.  Only a run that fits then lets its
entries build their kernels on scratch operands (their first use, where an entry asks what it needs of the installed
toolchain): the memory sentence comes before the first allocation on the GPU.  ``load()`` then reads (or draws) and
merges the weights and keeps each in the forms the blocks of these plans read: the stored form of a weight that a block
multiplies on (``Block.stored_weights``; in the layout of that block's provider where it has one of its own:
``Block.stored_layout``), the bf16 copy of a weight that a block reads as one -- so no copy of a weight
that every plan of the run multiplies on its stored format.  A block that would read a form the load did not keep fails
there, by name, before the first step.  ``load()`` then compiles what a provider compiles (a compiled form is launched
once) and runs one warm-up step, each as a timed stage.  The reference session of a run's checks reads bf16 copies; where
the load kept none it gets one per call (``on_demand``).  ``prefill`` feeds the prompts and ``step`` advances every
sequence by one token; the chosen ids stay on the GPU, and the one host read of a step is the caller's (``read``), in the
sample's own loop -- there is no ``generate()`` here.  ``trace`` replays the first steps of the recorded run under the profiler for the report's GPU-side
lines (``timing``); it leaves the run's state and record as they were.  Asked for launch records, the same pass also keeps
what the kernel check needs (``census.Traced``): the sites of a step in wiring order, the log of every block call of the
traced steps and the profiler's record of every launch.  ``trace_request`` is the other form of a traced pass, for a request that is timed and
then traced (``model_samples.perf``): it replays nothing and traces steps of the request that is running, issued by the
caller's own loop body; such a request starts from ``reset(fresh=False, recorded=False)`` -- the state zeroed in place,
nothing recorded.

A session's plan is for one number of tokens per sequence (``tokens``).  At one, its step is the decode step (``prefill``,
``step``).  At more, its step is the verify step of speculative decoding (``verify``): every sequence is fed that many
tokens at once -- the token the run kept last and the tokens drafted behind it -- the recurrent state after each of them
is kept in a slot of its own (``state.TokenSlots``), and the caller says how many of them the run keeps.  The decode step
is a row of another plan: a run that needs both has two sessions on one load (``share=``), planned together.  Two such
sessions share weights, never a state.

A session built with ``draft_tokens=k`` is the one that feeds a prompt and verifies behind it, on ONE state.  It has two
plans: its own, the verify plan of ``k + 1`` tokens per sequence (``plan``), and the decode plan its prompt goes through
(``prefill_plan``).  ``prefill`` runs decode steps on token 0 of the state of ``k + 1`` tokens per sequence
(``state.DecodeState.decode_layers``: the same memory in the layout of one token), so the first verify step reads what the
prompt left and nothing is copied.  ``verify`` then feeds, per sequence, the token the run kept last and the ``k`` tokens
its caller drafted behind it; the caller reads the greedy ids (``read``) and says how many drafted tokens it keeps
(``accept``; ``sampling.accepted`` counts them by the greedy rule).  The method that is called selects the plan of a step,
never the state and never a test at run time.  Without a drafter the drafted tokens are the caller's.

Handed a drafter (``drafter=``; one sequence), such a session runs the whole step of speculative decoding:
``speculate`` is the drafter's ``k`` passes, one verify step and the pair the caller reads -- the window that was
verified and the greedy token after each of its tokens -- in one host read.  The passes read what the step before left
on the state (``state.hidden``, ``state.greedy``, the kept count) and write the drafter's own cache (``state.draft``);
every prompt step but the first begins with one pass of the drafter over the position in front of it, at one row, so
the cache holds the prompt when the first step drafts.  The drafter has a plan of its own, at one row per sequence;
its table stands under the session's two, its load takes the model's embedding table, its memory is in the one
sentence.  ``speculate`` reads nothing and compares nothing: the count is the caller's (``accept``).

A session built with ``capture=True`` records its decode step once as a launch sequence and replays it for every token
(:class:`Recorded`): ``prefill`` and ``step`` then issue one replay where they issue every block's launch from Python, and
Python runs nothing between two tokens but the caller's one host read.  Whether a step can be recorded is the plan's data
(``recordable``: every block of it says so, and the step is the plain decode step -- one token per sequence, no draft
tokens, no drafter, no copies made per call); a session whose step cannot be launches from Python and says why
(``python_reason``).  The step is recorded at the first replay of a state (``record_step`` brings that forward), on that
state: a recorded launch keeps the addresses of its operands, so the token and the position rows a step reads stand at
fixed addresses (``state.Cursor``) and the step's own last launches write the chosen ids and move the rows on.  A
replayed step returns that ids buffer, so what its caller hands back is what the next lookup reads, however it was
changed in between; any other tensor is copied into the buffer.  Four things an eager step has for free are code
here, in front of every replay: the capacity test, the ids the caller handed over, the cursor at the state's
position, and the state the sequence was recorded on.  The record of a replayed step holds the ids its caller handed
over, and the checks of a run launch the recorded steps again from Python (``tie_out.run_checks``): what a replay
computed is held to that, ids and logits, byte for byte.  Every step with a tap or a call log, the traced pass and
the checks' own passes are launched from Python.

The set of sequences is fixed before the first step: a sequence that reaches its end token stays in the batch, so the number of
rows, and with it the kernel path, does not change inside a run.  A session keeps no state between two prompt sets other than
what ``reset`` clears.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from . import blocks, census
from .sampling import GREEDY
from .state import Cursor, DecodeState
from .timing import Report, Stages, now, trace_steps

STACKS = ("fe-cublas", "native", "fe-entries", "reference")
REFERENCE = "reference"  # the stack of the plain-PyTorch reference blocks: what the checks of a run compare it with
PREFILL_ROUTES = ("decode-step",)
#: how a stored weight is laid out for a block that multiplies on it and names no layout of its provider's own
CHECKPOINT_ORDER = "the checkpoint's row order"
#: the line between the two tables of a session built with ``draft_tokens=``: its verify steps' rows, then its prompt's
PROMPT_ROWS = "the prompt of this session goes through decode steps, one per prompt token, on these rows:"
#: the line above the drafter's table of a session with a drafter (``{calls}``: the block calls of one pass)
DRAFTER_ROWS = (
    "the drafter: {calls} block calls a pass; the first pass of a step behind a verify step runs at S x T rows, every other pass "
    "and every pass over the prompt at S rows; every row of this table serves any row count"
)
ONE_SEQUENCE = "a session with a drafter serves one sequence: one request, as the drafter rows are for"
#: what the memory sentence adds for PyTorch's pool to what the checks hold for every sequence beyond the first, as a share of
#: that part: the pool keeps blocks beyond the tensors that are alive, and a run whose pool is capped at the tensors' own
#: figure fails to allocate inside its checks.  The smallest of a tenth, a quarter and four tenths with which capped runs at
#: 256 and at 1,024 sequences both passed (the family README has the runs); an allowance from runs, not a bound
CHECK_POOL = 0.4


#: the steps of a traced pass that a session which replays its step runs once more as replays, behind those launched from Python
REPLAYED_TRACED = 2
#: why a session's steps are launched from Python where recording was not asked for
NOT_ASKED = "recording the step was not asked for (Session(capture=True))"
#: ... and where the step is not the plain decode step: the kinds of session whose step is not recorded
NOT_PLAIN = {
    "tokens": "a step of several tokens per sequence (a verify step) is not recorded: the count it keeps is the host's, between two steps",
    "draft_tokens": "a session that verifies behind its prompt is not recorded: its steps are verify steps, whose kept count is the host's",
    "drafter": "a step of speculative decoding is not recorded: the drafter's passes run on reference blocks, which a recorded launch sequence cannot hold",
    "on_demand": "the reference session of a run's checks makes a copy of a weight per call, which a recorded launch sequence cannot hold",
    "device": "a launch sequence is recorded on a CUDA device",
}


class LaunchGraph:
    """The launches of one pass of a step's Python, recorded on PyTorch's capture stream (``torch.cuda.graph``) and issued
    again as one launch (``replay``).  ``record(body)`` runs ``body`` once with the capture stream current: every launch
    of it lands in the sequence and none runs.  A pass the installation refuses (a host read, a copy from pageable
    host memory, a set-up inside the step) raises what PyTorch raised -- and where the refusal leaves the thread on the
    capture stream, the stream that was current is put back first: nothing computes on a side stream afterwards."""

    @staticmethod
    def serves(device) -> bool:
        """Whether a launch sequence can be recorded on ``device``: a CUDA device."""
        return torch.device(device).type == "cuda"

    def __init__(self, device):
        self.device = device
        try:  # where the build keeps the recorded sequence itself, the CUDA runtime can be asked what it holds (``nodes``)
            self.graph, self.kept = torch.cuda.CUDAGraph(keep_graph=True), True
        except TypeError:
            self.graph, self.kept = torch.cuda.CUDAGraph(), False

    def record(self, body):
        before = torch.cuda.current_stream(self.device)
        try:
            with torch.cuda.graph(self.graph):
                return body()
        except BaseException:
            torch.cuda.set_stream(before)  # the context's exit raised before it put the stream back, or never ran
            raise

    def replay(self) -> None:
        self.graph.replay()

    def reset(self) -> None:
        """Release the sequence and the memory it kept, now: a sequence left to the collector can be destroyed inside a
        later recording, which that invalidates."""
        self.graph.reset()

    def nodes(self) -> Optional[list]:
        """``(kind, kernel name)`` of every node of the recorded sequence as the CUDA runtime states it -- kind ``"kernel"``,
        ``"copy"``, ``"fill"`` or the runtime's own word -- or ``None`` where this installation does not state it (a
        PyTorch that keeps no sequence to ask, or no binding of the runtime's reader)."""
        if not self.kept:
            return None
        try:
            from cuda.bindings import driver, runtime

            def checked(result):
                if int(result[0]) != 0:
                    raise RuntimeError(f"the CUDA runtime declined: {result[0]!r}")
                return result[1:] if len(result) > 2 else result[1]

            raw = self.graph.raw_cuda_graph()
            _, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=0))
            handles, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=count))
            kinds = {
                runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
                runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "copy",
                runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset: "fill",
            }
            found = []
            for index in range(count):
                kind = checked(runtime.cudaGraphNodeGetType(handles[index]))
                name = ""
                if kinds.get(kind) == "kernel":
                    error, parameters = driver.cuGraphKernelNodeGetParams(driver.CUgraphNode(init_value=int(handles[index])))
                    if int(error) == 0 and int(parameters.func):
                        error, symbol = driver.cuFuncGetName(driver.CUfunction(init_value=int(parameters.func)))
                        if int(error) == 0:
                            name = symbol.decode() if isinstance(symbol, bytes) else str(symbol)
                found.append((kinds.get(kind, getattr(kind, "name", str(kind)).replace("cudaGraphNodeType", "").lower()), name))
            return found
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None


class Recorded:
    """One decode step of a session recorded as a launch sequence, with what its replays read and write at fixed
    addresses: the cursor of the state it was recorded on (the ids buffer the lookup reads, the position rows), and the
    tensors the recorded pass returned -- the chosen ids, the logits, the final-norm rows -- which every replay writes
    again where the pass allocated them.  ``calls``: the block calls of the step; ``launches``: what the sequence holds,
    as the CUDA runtime states it and held against the plan (``census.recorded_launches``), or ``None`` where the
    installation does not state it; ``seconds`` and ``pool_bytes``: what recording took, and what PyTorch reserved
    for the sequence's own pool, in which the working memory of one step stays for as long as the sequence lives.

    The ids buffer is also what a replayed step hands back to its caller (``Session.step``): the sequence's own last
    copy has written the chosen ids into it, and handed back it is read as it then is.  The chosen ids as the token
    choice left them (``next_ids``) stay the sequence's own: the state and the record read those."""

    def __init__(self, graph, state: DecodeState, cursor: Cursor, next_ids, logits, hidden, calls: int, launches=None, seconds=0.0, pool_bytes=0, planned=0):
        self.graph, self.state, self.cursor = graph, state, cursor
        self.next_ids, self.logits, self.hidden = next_ids, logits, hidden
        self.calls, self.launches, self.seconds, self.pool_bytes = calls, launches, seconds, pool_bytes
        self.planned = planned  # the launches of the step's blocks on exact rows, as the plan has them
        self.replays = 0
        #: whether a replay was the last to write the ids buffer for this session -- the step's own copy of the ids it
        #: chose -- and nothing has been fed since: what the tail of a request is read on (``Session.read_tail``)
        self.chosen = False

    def feed(self, ids: torch.Tensor) -> None:
        """The ids the next replay's lookup reads are the ids its caller hands over: one copy into the ids buffer --
        unless the caller hands over that buffer itself, the tensor a replayed step returns.  The lookup then reads
        what the buffer holds, whatever has written it since: no copy, and nothing to compare."""
        if ids is not self.cursor.ids:
            self.cursor.ids.copy_(ids)
        self.chosen = False

    def replay(self, state: DecodeState) -> None:
        """One replay on ``state``, which stands where the step is to run.  A sequence writes the memory of the state it was
        recorded on: another state is a ``ValueError`` and launches nothing.  The cursor is held to the state's
        position by its host integer and set from the tables where the two differ -- behind steps launched from
        Python, a reset in place, a restored snapshot."""
        if state is not self.state:
            raise ValueError(
                "a recorded step is replayed on the state it was recorded on: this session's state is another one, and a replay would write the first"
            )
        if self.cursor.position != state.position:
            self.cursor.set(state.position)
        self.graph.replay()
        self.cursor.position = state.position + 1  # the sequence's own last launches moved the rows on
        self.replays += 1
        self.chosen = True

    def text(self) -> str:
        """What the recorded step holds, for the line a run prints behind its load.  Where the installation states the
        sequence (``launches``) the count is the CUDA runtime's, with what of it was held to the plan kernel by kernel;
        where it does not, the count is worded as the plan's: ``planned`` launches of its blocks."""
        tail = f"{self.cursor.launches()} position updates"
        if self.launches is None:
            return (
                f"step recorded: by the plan {self.planned} launches of its blocks, then the token choice, 1 copy of the chosen ids and {tail} "
                "(this installation does not state what the recorded sequence holds); replayed once per token"
            )
        held = self.launches
        return (
            f"step recorded: {held['launches']} launches, as the CUDA runtime states the recorded sequence -- {held['held']} of the plan's block "
            f"launches held kernel by kernel, the lookup, {held['token_choice']} of the token choice, {held['copies']} copy of the chosen ids "
            f"and {tail} -- replayed once per token"
        )

    def release(self) -> None:
        self.graph.reset()


class StopRule:
    """When a sequence ends: at one of the end tokens or after ``max_new_tokens``; ``force_tokens`` ignores the end token and
    runs exactly that many steps (the protocol of a timed run)."""

    def __init__(self, eos_ids: Sequence[int], max_new_tokens: int = 256, force_tokens: Optional[int] = None):
        if max_new_tokens < 1 or (force_tokens is not None and force_tokens < 1):
            raise ValueError("a stop rule needs at least one new token")
        self.eos_ids = frozenset(int(i) for i in eos_ids)
        self.max_new_tokens, self.force_tokens = max_new_tokens, force_tokens

    @property
    def limit(self) -> int:
        return self.force_tokens if self.force_tokens is not None else self.max_new_tokens

    def ends(self, token: int) -> bool:
        return self.force_tokens is None and int(token) in self.eos_ids


class Record:
    """The ids fed to, the logits of and the ids chosen by the first steps of a run, kept on the GPU for the checks: one
    row per row of a step (sequences x tokens per sequence).  ``accepted`` is the host's own list of how many tokens
    behind the first each of those steps kept (0 for a decode step).  ``proposals`` (a session with a drafter:
    ``drafted`` = (sequences, drafted tokens)) holds per step the drafter's own proposals, whatever the step verified."""

    def __init__(self, steps: int, rows: int, vocab: int, device, drafted: Optional[tuple] = None):
        self.proposals = None if drafted is None else torch.zeros((steps, *drafted), dtype=torch.int32, device=device)
        self.inputs = torch.zeros((steps, rows), dtype=torch.int32, device=device)
        self.outputs = torch.zeros((steps, rows), dtype=torch.int32, device=device)
        self.logits = torch.zeros((steps, rows, vocab), dtype=torch.bfloat16, device=device)
        self.accepted: List[int] = []
        self.count = 0
        #: of EVERY verify step of the run, recorded or not, what ``accepted`` holds of the recorded ones -- the drafted tokens
        #: it kept --, and how many steps verified drafted tokens of the caller's in the place of a drafter's
        #: (``Session.speculate(force=)``): the report's acceptance lines
        self.accepted_all: List[int] = []
        self.forced = 0
        #: the ids of a whole prompt ``[sequences, tokens]``, kept by a session that replays its prompt in front of its steps
        self.prompt: Optional[torch.Tensor] = None

    def add(self, ids: torch.Tensor, logits: torch.Tensor, next_ids: torch.Tensor, accept: int = 0, proposals: Optional[torch.Tensor] = None) -> None:
        if self.count < self.inputs.shape[0]:
            if proposals is not None:
                self.proposals[self.count].copy_(proposals)
            self.inputs[self.count].copy_(ids)
            self.outputs[self.count].copy_(next_ids)
            self.logits[self.count].copy_(logits)
            self.accepted.append(accept)
        self.count += 1


class Session:
    def __init__(
        self,
        model,
        text,
        sequences: int = 1,
        sampling=GREEDY,
        stack: str = "fe-cublas",
        prefill: str = "decode-step",
        device="cuda:0",
        overrides: Optional[dict] = None,
        reference_precision: str = "fp64",
        capacity: int = 256,
        checkpoint=None,
        seed: int = 0,
        merge_rule: str = "largest_scale",
        share: Optional["Session"] = None,
        record_steps: Optional[int] = None,
        check_kernels: bool = False,
        on_demand=None,
        tokens: int = 1,
        draft_tokens: Optional[int] = None,
        drafter=None,
        max_model_len: Optional[int] = None,
        capture: bool = False,
    ):
        """``capacity`` is the number of positions (prompt + new tokens) a sequence may reach.  ``checkpoint`` defaults to the
        model's seeded weights.  ``share`` reuses another session's loaded weights (the reference session of the tie-out).
        ``record_steps`` bounds the steps whose ids and logits are kept on the GPU for the checks (default: every step).
        ``check_kernels`` says that the run will check its kernels: what that needs of a provider is then part of the plan
        (a decline is a SKIP before the load), and the traced pass keeps its launch records.  A stack whose rows are kernels
        on the stored formats (the family's ``checked_stacks``) always checks its kernels.
        ``on_demand`` (``weights.OnDemand``) is for the reference session of a run's checks, which runs on the weights of the
        session it checks: where that load kept a weight on its stored format alone, the blocks of this session get a bf16
        copy of it made for the one call and dropped after it, and the object counts what that cost.  It is refused for
        any other session (``ValueError``): a copy made per call is a cast and two synchronisations in front of a block,
        which no step of a run that is timed, traced or checked may carry.
        ``tokens`` is the number of tokens a step of this session feeds per sequence: 1 for the decode step; more for the
        verify step (``verify``), whose rows are sequences x tokens and whose plan is resolved for that pair.
        ``draft_tokens`` builds the session that feeds its prompt through decode steps and then runs verify steps behind
        it on the same state: the number of drafted tokens a step verifies behind the token it kept last, so a step
        feeds ``draft_tokens + 1`` tokens per sequence (``tokens`` is then that number and is left out here).  The
        drafted tokens are the caller's -- or, with ``drafter`` (the checkpoint's drafter, or the seeded model's:
        ``QwenDenseDrafter``), the drafter's: the session then runs its passes (``speculate``).  A drafter needs
        ``draft_tokens`` and serves one sequence.  It drafts for this session's weights and widths: a checkpoint's
        drafter is refused where the session that loads was handed none (its weights are seeded), one of another
        model's widths by the first width that differs -- each a ``ValueError`` here, before anything is planned.
        ``max_model_len`` is the positions a sequence of the run may reach at most, its context length: a constant of
        the run, which every block of the session's plans is handed in its options, and
        the state where a block reads the page table and the lengths of a paged cache (``attended_tables``).  Left
        out it is ``capacity``; below ``capacity`` it is a ``ValueError``: no step of the session writes a position past
        it.
        ``capture`` asks for the decode step to be recorded once as a launch sequence and replayed for every token
        (``prefill``, ``step``).  It is recorded where the plan says it can be (``recordable``); a session whose step
        cannot be, or whose recording the installation declines, launches every step from Python as a session without
        the flag does, and says why (``python_reason``).  Under replay the tensor ``prefill`` and ``step`` return is
        the session's ids buffer: the next step reads the ids it is fed from it and writes the next ids into it."""
        if sampling is not GREEDY:
            raise ValueError("this stack chooses tokens greedily; pass sampling=GREEDY")
        if on_demand is not None and (share is None or stack != REFERENCE):
            raise ValueError(
                f"on_demand is for the reference session of a run's checks (stack {REFERENCE}, on the weights of the session it checks: share=); "
                f"a session of stack {stack}{'' if share is None else ' on shared weights'} reads the forms its load kept and never dequantises a weight per call"
            )
        if prefill not in PREFILL_ROUTES:
            raise ValueError(f"prefill route must be one of {PREFILL_ROUTES}; got {prefill!r}")
        if sequences < 1:
            raise ValueError(f"a session needs at least one sequence; got {sequences}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(f"a session's step feeds at least one token per sequence; got tokens={tokens!r}")
        if max_model_len is not None:
            if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
                raise ValueError(f"max_model_len is the positions a sequence of the run may reach at most: a whole number from 1; got {max_model_len!r}")
            if max_model_len < int(capacity):
                raise ValueError(
                    "max_model_len is the positions a sequence of the run may reach at most, and this session is built for "
                    f"capacity={int(capacity)} positions per sequence; got max_model_len={max_model_len}"
                )
        if draft_tokens is not None:
            if isinstance(draft_tokens, bool) or not isinstance(draft_tokens, int) or draft_tokens < 1:
                raise ValueError(
                    "draft_tokens is the number of drafted tokens a step verifies behind the token it kept last: a whole number from 1; "
                    f"got {draft_tokens!r}"
                )
            if tokens != 1:
                raise ValueError(
                    "a session is built with tokens= (verify steps from a zero state) or with draft_tokens= (a prompt through decode steps, "
                    "then verify steps of draft_tokens + 1 tokens per sequence), not with both"
                )
            tokens = draft_tokens + 1
        if drafter is not None and draft_tokens is None:
            raise ValueError("a drafter drafts the tokens a session's verify steps check: drafter= needs draft_tokens=, the number it drafts a step")
        if drafter is not None and sequences != 1:
            raise ValueError(f"{ONE_SEQUENCE}; got sequences={sequences}")
        if drafter is not None:  # ... and it drafts for the weights and the widths this session's steps run on: said here, before any plan
            loaded = checkpoint if share is None else share._weights_owner().checkpoint  # of the session that loads: None draws seeded weights
            if getattr(drafter, "keys", None) and (loaded is None or getattr(loaded, "synthetic_weights", False)):
                raise ValueError(
                    "the drafter was described from a checkpoint and the weights of this session are seeded: hand the session that loads "
                    "that checkpoint (checkpoint=), or build the drafter of the seeded model"
                )
            ours, theirs = vars(model.geometry), vars(drafter.geometry)
            width = next((name for name in ours if name != "layer_kinds" and theirs.get(name) != ours[name]), None)
            if width is not None:
                raise ValueError(
                    f"the drafter was described for a model of other widths than this session's: its {width} is {theirs.get(width)!r}, "
                    f"the model's {ours[width]!r}"
                )
        self.model, self.text, self.sequences, self.stack, self.prefill_route = model, text, int(sequences), stack, prefill
        self.tokens = tokens  # per sequence, of a step of this session's plan
        self.draft_tokens = draft_tokens  # of a session that feeds its prompt through decode steps and verifies behind it; else None
        self.drafter = drafter  # what drafts those tokens; None: the caller does
        self.draft_plan: Optional[blocks.Plan] = None  # the drafter's plan, at one row per sequence
        self.draft_weights = None
        self.device = torch.device(device)
        self.overrides, self.reference_precision, self.capacity = dict(overrides or {}), reference_precision, int(capacity)
        self.max_model_len = self.capacity if max_model_len is None else max_model_len  # of the run: what a kernel that is told a maximum length is told
        self.checkpoint, self.seed, self.merge_rule, self._share = checkpoint, seed, merge_rule, share
        self.record_steps = self.capacity if record_steps is None else max(1, min(int(record_steps), self.capacity))
        self.check_kernels = bool(check_kernels) or stack in getattr(model, "checked_stacks", ())
        self.on_demand = on_demand  # the maker of the bf16 copies this session's blocks read and its weights do not hold
        self.traced: Optional[census.Traced] = None  # of the last traced pass that kept its launch records
        self._call_log: Optional[list] = None  # the block calls of the traced step that is running
        # the stage clock synchronises the session's device and is handed the device, never the session: a closure over the
        # session would close a reference cycle, and a session in a cycle is released when Python next collects cycles, not
        # when its last holder lets go -- the reference session of a run's checks would outlive them with its state and record
        device = self.device
        self.stages = Stages(sync=lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None)
        self.plan: Optional[blocks.Plan] = None
        self.prefill_plan: Optional[blocks.Plan] = None  # the decode plan of the prompt of a session built with draft_tokens=
        self.weights = None
        self.state: Optional[DecodeState] = None
        self.record: Optional[Record] = None
        self.prompt_record: Optional[Record] = None  # the prompt's decode steps of a session built with draft_tokens=
        self.tap = None  # tie_out's hook: tap(block name, layer, block, operands) -> the block's result
        self._followers: tuple = ()  # the sessions that will run on this one's weights, as ``plan_table`` was told
        self._report = self._new_report()
        self._calls = 0
        self._prefill_start = 0.0
        self.capture = bool(capture)  # whether the step is to be recorded and replayed
        self.recorded: Optional[Recorded] = None  # the recorded step of the session's present state, once it has one
        #: why the steps of this session are launched from Python; "" while they are replayed, or will be at the first replay
        self.python_reason = "" if self.capture else NOT_ASKED
        #: of a request that was replayed, read behind it (``read_tail``): whether the ids buffer held the ids the last
        #: replay chose, whether the position rows stood where the request left the state, and that position
        self.tail: Optional[tuple] = None
        self.pool_bytes: List[int] = []  # of every recording of this session: what PyTorch reserved for the sequence's own pool
        self._running: Optional[blocks.Plan] = None  # the plan of the step that is running, and its rows
        self._rows = self.sequences * self.tokens
        self._open = False  # a verify step has run and has not been told how many of its tokens are kept

    # ------------------------------------------------------------------ plan
    def resolve(self) -> blocks.Plan:
        """The plan of this session's step.  A session built with ``draft_tokens=`` resolves two: the verify plan (returned,
        ``plan``) and the decode plan of its prompt (``prefill_plan``), with the same overrides -- an override selects its
        row in both, and a key whose rows do not serve one of the two kinds of step declines there by that row's
        sentence.  The decode plan is told the tokens per sequence of the state its steps run on (``state_tokens``)."""
        if self.plan is None:
            self.plan = self._resolve(self.tokens)
        if self.draft_tokens is not None and self.prefill_plan is None:
            self.prefill_plan = self._resolve(1, state_tokens=self.tokens)
        if self.drafter is not None and self.draft_plan is None:  # one plan, at one row per sequence; the overrides are the model's
            self.draft_plan = self._resolve(1, self.drafter, {})
        return self.plan

    def _resolve(self, tokens: int, model=None, overrides: Optional[dict] = None, **more) -> blocks.Plan:
        """A plan of ``model`` (default: the session's model, with the session's overrides) at ``tokens`` tokens per sequence."""
        model = self.model if model is None else model
        options = {
            "reference_precision": self.reference_precision,
            "device": str(self.device),
            "check_kernels": self.check_kernels,
            "merge_rule": self.merge_rule,
            "max_model_len": self.max_model_len,
            **more,
        }
        facts = self.device if self.device.type == "cuda" else blocks.DeviceFacts((0, 0), "cpu")
        return blocks.plan(
            model.provider_table,
            model.regimes,
            facts,
            self.sequences,
            tokens,
            self.stack,
            self.overrides if overrides is None else overrides,
            model=model,
            options=options,
            source=model.table_source,
        )

    def _kinds(self) -> list:
        """``(plan, tokens per sequence)`` of every kind of step this session runs, in the order it runs them: the decode
        steps of its prompt (a session built with ``draft_tokens=``), then its own step."""
        self.resolve()
        return [*([(self.prefill_plan, 1)] if self.prefill_plan is not None else []), (self.plan, self.tokens)]

    def recordable(self) -> str:
        """Whether this session's step can be recorded once as a launch sequence and replayed: ``""``, or why not, in one
        sentence.  Decided before anything is loaded, from what the session is and from its plan's data: only the
        plain decode step is recorded -- one token per sequence, no draft tokens, no drafter, no copies made per call,
        on a CUDA device -- and only where every block of the plan says a recorded sequence can hold its calls
        (``Plan.recordable``)."""
        kinds = (("drafter", self.drafter is not None), ("draft_tokens", self.draft_tokens is not None), ("tokens", self.tokens != 1))
        for kind, is_one in (*kinds, ("on_demand", self.on_demand is not None), ("device", not LaunchGraph.serves(self.device))):
            if is_one:
                return NOT_PLAIN[kind]
        return self.resolve().recordable()

    def _plans(self, own: bool = False) -> list:
        """The plans of the run this session's load serves: its own -- its step's and, where it has one, its prompt's --
        and those of the sessions that will run on its weights.  ``own``: this session's alone."""
        self.resolve()
        sessions = (self,) if own else (self, *self._followers)
        return [plan for session in sessions for plan in (session.resolve(), *([session.prefill_plan] if session.prefill_plan is not None else []))]

    def _on_stored(self, every: bool, own: bool = False) -> tuple:
        """The weights that a plan of the run -- ``every``: that every plan of the run -- multiplies on the stored format,
        named as a family's loader is asked for them: a block (all of its calls), or ``<block> at <call site>``
        (``blocks.scoped``) for the calls at one site of a block whose calls several rows serve in some plan.  A plan
        that serves the block with one row answers for every site of it.  ``own``: this session's plan alone."""
        plans = self._plans(own)
        readers = {}  # block -> per plan: call site -> whether the block's part there multiplies on the stored format ("": every call)
        for plan in plans:
            reads = {}
            for _, name, site, _, block in plan.parts():
                reads.setdefault(name, {})[site] = bool(getattr(block, "stored_weights", False))
            for name, sites in reads.items():
                readers.setdefault(name, []).append(sites)
        held = all if every else any
        named = []
        # in the order the plans name them: the first plan's blocks on a stored format, then what a later plan adds
        order = dict.fromkeys(name for plan in plans for _, name, _, _, block in plan.parts() if getattr(block, "stored_weights", False))
        for name in order:
            by_plan = readers[name]
            if every and len(by_plan) < len(plans):  # a plan without the block: no statement about every plan
                continue
            sites = tuple(dict.fromkeys(site for sites in by_plan for site in sites if site))  # the call sites some plan tells apart
            at = [site for site in sites or ("",) if held(reads[""] if "" in reads else reads.get(site, False) for reads in by_plan)]
            named += [name] if len(at) == len(sites or ("",)) else [blocks.scoped(name, site) for site in at]
        return tuple(named)

    def stored_weights(self, own: bool = False) -> tuple:
        """The weights this session's load keeps on their stored format: those of every block that a provider multiplying
        on the stored format serves (``Block.stored_weights``) in this session's plan or in the plan of a session that
        will run on its weights (``plan_table(later)``) -- named by block, or, where such a provider serves the calls at
        some call sites of a block only, as ``<block> at <call site>``.  ``own``: the blocks of this session's plan
        alone -- the weights its own steps read on the stored format."""
        return self._on_stored(every=False, own=own)

    def uncopied_weights(self) -> tuple:
        """The weights this session's load keeps on their stored format ALONE, without a bf16 copy (named as
        ``stored_weights`` names them): a provider multiplying on the stored format serves their calls in this session's
        plan and in the plan of every session that will run on its weights (``plan_table(later)``), so no block of the
        run reads a copy.  The reference blocks of the checks do, and get theirs per call (``on_demand``)."""
        return self._on_stored(every=True)

    def stored_layouts(self) -> dict:
        """Block -> the layout its provider multiplies the stored weight on (``Block.stored_layout``), for the blocks of
        ``stored_weights()`` whose provider has one of its own: what the family's loader is handed beside the block names.
        A load keeps ONE stored form of a weight, so two plans of a run that multiply on it in two row orders are a
        ``ValueError``, a wrong table: two providers' own layouts, and as much a provider's own layout in one plan beside
        the checkpoint's row order in another (a block that says ``stored_weights`` and names no layout).  A plan whose
        block reads the bf16 copy is no reader of the stored form and asks for nothing here."""
        layouts, owners = {}, {}
        for plan in self._plans():
            for _, name, _, _, block in plan.parts():  # of every call site, for a block whose calls several rows serve
                if not getattr(block, "stored_weights", False):
                    continue
                layout = getattr(block, "stored_layout", None)
                owner = CHECKPOINT_ORDER if layout is None else type(block)  # whose row order this plan multiplies on
                if owners.setdefault(name, owner) is not owner:
                    orders = ", ".join(each if isinstance(each, str) else each.__name__ for each in (owners[name], owner))
                    raise ValueError(
                        f"the provider table is wrong: two plans of this run multiply on two layouts of the stored weight of {name} ({orders}); a load keeps one"
                    )
                if layout is not None:
                    layouts.setdefault(name, layout)
        return layouts

    def cache_page(self) -> Optional[int]:
        """The positions of a page of this session's attention cache, or ``None`` for the plain cache: what the blocks of its
        plans ask for (``Block.cache_page``) -- of its step's plan and, where it has one, of its prompt's.  A session has
        ONE state, so two blocks that name two pages are a ``ValueError``, a wrong table, named with their plans -- as is a
        page that is no whole number of positions from 1; a block that names none reads either layout.  Two sessions on
        one load share weights, never a state: each is built in the layout its own plans ask for.  The drafter's plan is
        not asked: its cache is the drafter's own."""
        asked = {}  # page -> the first block that asks for it, with its plan
        for plan, _ in self._kinds():
            for label, _, _, _, block in plan.parts():
                page = getattr(block, "cache_page", None)
                if page is None:
                    continue
                where = f"{label} in the plan of {plan.sequences} sequence{'s' if plan.sequences > 1 else ''}{plan.tokens_text()}"
                if isinstance(page, bool) or not isinstance(page, int) or page < 1:  # what a state asks of its page, asked here: before a byte is read
                    raise ValueError(
                        f"the provider table is wrong: {where} asks for pages of {page!r} positions; the page of an attention cache is a whole number of positions from 1"
                    )
                asked.setdefault(page, where)
        if len(asked) > 1:
            (one, first), (other, second) = list(asked.items())[:2]
            raise ValueError(
                "the provider table is wrong: the blocks of one session ask for two layouts of its attention cache -- "
                f"{first} asks for pages of {one} positions, {second} for pages of {other}; a session has one state, in one layout"
            )
        return next(iter(asked), None)

    def attended_tables(self) -> bool:
        """Whether a block of this session's plans reads its attention cache through the state's page table and lengths
        (``Block.attended_tables``): of its step's plan and, where it has one, of its prompt's, every part of a block
        that several rows serve included.  The state is then built with both, for the run's maximum length
        (``max_model_len``).  They are tables of a pool of pages: such a block in a session whose blocks ask for no
        paged cache is a ``ValueError``, a wrong table, named with its plan."""
        for plan, _ in self._kinds():
            for label, _, _, _, block in plan.parts():
                if not getattr(block, "attended_tables", False):
                    continue
                if self.cache_page() is None:
                    where = f"{label} in the plan of {plan.sequences} sequence{'s' if plan.sequences > 1 else ''}{plan.tokens_text()}"
                    raise ValueError(
                        f"the provider table is wrong: {where} reads the page table and the lengths of a paged attention cache, "
                        "and no block of this session asks for one (cache_page)"
                    )
                return True
        return False

    def _paged(self) -> dict:
        """The layout of this session's attention cache as keywords of its state and of a family's memory arithmetic: the
        page, where a block of its plans asks for a paged cache, and with it the run's maximum length where a block
        reads the page table and the lengths (a family whose states are plain is never asked).  Both wrong tables of
        that layout are named here (``cache_page``, ``attended_tables``)."""
        page, tables = self.cache_page(), self.attended_tables()
        return {} if page is None else {"page": page, **({"max_model_len": self.max_model_len} if tables else {})}

    def _kept(self) -> dict:
        """What the plans of the run ask the family's loader to keep, as its keyword arguments: nothing for a run whose every
        multiply reads a bf16 copy (a family without such a block is never asked)."""
        stored, uncopied = self.stored_weights(), self.uncopied_weights()
        return {**({"stored": stored} if stored else {}), **({"uncopied": uncopied} if uncopied else {})}

    def _per_token(self) -> dict:
        """The tokens per sequence as a keyword of a family's memory arithmetic, where a step of this session feeds several
        (a family whose steps all feed one token is never asked)."""
        return {"tokens": self.tokens} if self.tokens > 1 else {}

    def _check_forms(self) -> None:
        """Before the first step: every form of a weight a block of this session's plan reads is one the load of its weights
        kept, and a stored weight is in the row order its block multiplies on.  A session that runs on another session's
        weights was not necessarily planned with it; a ``ValueError`` then names the block and the form."""
        kept, uncopied = getattr(self.weights, "stored", None), getattr(self.weights, "uncopied", ())
        if kept is None:  # a family whose loader holds one form of every weight
            return
        laid_out = getattr(self.weights, "laid_out", ())
        other = "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"

        def names(among, name, site) -> bool:
            """Whether a load names the weights of these calls among those it keeps in one form: the block, or this call
            site of it.  For a block this plan serves with one row a load that names some call sites of it alone has
            the form for some of the block's weights and not for others, which is the load of another plan as well."""
            return name in among or (bool(site) and blocks.scoped(name, site) in among)

        parts = [part for plan in self._plans(own=True) for part in plan.parts()]  # of both kinds of step, where the session has two
        for label, name, site, _, block in parts:  # per call site, for a block whose calls several rows serve
            on_stored = getattr(block, "stored_weights", False)
            if on_stored and not names(kept, name, site):
                raise ValueError(f"{block.entry} multiplies on the stored format of the weight of {label}, and the load kept none: {other}")
            if on_stored and (getattr(block, "stored_layout", None) is not None) != (name in laid_out):
                wants, has = ("the checkpoint's row order", "a provider's own") if name in laid_out else ("its own row order", "the checkpoint's")
                raise ValueError(f"{block.entry} multiplies on the stored weight of {name} in {wants}, and the load kept it in {has}: {other}")
            partly = not site and any(kept_alone.startswith(name + " at ") for kept_alone in uncopied)  # some call sites of this block
            if not on_stored and (names(uncopied, name, site) or partly) and self.on_demand is None:
                raise ValueError(f"{block.entry} reads a bf16 copy of the weight of {label}, and the load kept its stored format alone: {other}")

    def plan_table(self, later: Sequence["Session"] = (), references: int = 1) -> str:
        """The served-by table and the memory sentence; raises ``NotImplementedError`` (a SKIP) before anything is loaded.

        ``later`` are the sessions that will run after this one in the same process on its weights (``share=self``).  Their
        plans are resolved here too, and the memory sentence then covers the whole run -- this session's need plus their
        state -- against what is free now: every decline of a run comes before its first load, none after it.

        The sentence counts what a session that records its step keeps in the pool of its recorded sequence
        (``_recorded_bytes``: the family's arithmetic), and what the checks of the run hold for every sequence beyond the first (``_check_rows``:
        the family's arithmetic, nothing at one sequence and nothing on the reference stack, which runs no checks),
        names that part, and adds a share of it for PyTorch's pool (``CHECK_POOL``).  ``references``: how many reference
        stacks the checks of the run replay (2 for a run that checks a second one).

        In this order: the tables of every session are resolved, which allocates nothing; the memory sentence says whether
        the whole run fits; and only then do the entries of each plan make their first use (``Plan.first_use``: an entry
        builds its kernel on scratch operands as large as the run's, and asks there what it needs of the installed
        toolchain).  A run that does not fit is a SKIP with its estimate, and nothing was allocated for it.

        A session built with ``draft_tokens=`` prints the table of its verify steps, one line, and the table of the
        decode steps of its prompt; the memory sentence counts ONE state for it -- its prompt runs on the state of its
        verify steps -- and, for a run that has such a session, names the kinds of step where it named sequence counts.
        A session with a drafter prints the drafter's table under one more line, and the sentence counts the drafter:
        what its load holds besides the model's embedding table, its copies, its cache and its working memory
        (``_drafter_bytes``)."""
        plan = self.resolve()
        for session in later:
            session.resolve()
        self._followers = tuple(later)
        if self._share is None:
            self.stored_layouts()  # one weight in two row orders is a wrong table: named here, with the plans, before anything is built or read
        for session in (self, *later):
            session._paged()  # ... and so are two layouts of one session's attention cache, and tables of a pool that no block asks for
        sentence = ""
        if self.device.type == "cuda":
            # what the load of the first session holds follows the plans: the stored format of the weights a block multiplies
            # on, and no bf16 copy of those no block reads as one (asked of a family's model only where a plan has such a block)
            held = self._kept() if self._share is None else {}
            need = self.model.memory_bytes(self.sequences, self.capacity, **held, **self._per_token(), **self._paged()) - (
                self.model.copy_bytes() if self._share is not None else 0
            )
            # a session whose blocks ask for a paged attention cache holds a pool and its tables where a plain one holds two tensors a layer
            need += sum(session.model.state_bytes(session.sequences, session.capacity, **session._per_token(), **session._paged()) for session in later)
            need += sum(session._drafter_bytes() for session in (self, *later))
            need += sum(session._recorded_bytes() for session in (self, *later))  # what a recorded step keeps in a pool of its own
            # one state per session: a session that feeds its prompt through decode steps runs them on the state of its verify
            # steps, so it is counted once.  Nothing is added for its second plan either: what a first launch of the fused step
            # reserves outside the pool is reserved once in a process, and the sentence carries it once (``Plan.memory``)
            counts = [self.sequences, *(session.sequences for session in later)]
            what = f" for {', '.join(str(n) for n in counts[:-1])} and {counts[-1]} sequences, one count after the other" if later else ""
            kinded = any(session.draft_tokens is not None for session in (self, *later))
            if kinded:  # a run with two kinds of step names them
                kinds = [session._kind() for session in (self, *later)]
                what = f" for {', '.join(kinds[:-1])} and {kinds[-1]}, one after the other" if later else f" for {kinds[0]}"
            # the part of the need that is the checks': the reference's copy of a weight the load keeps no copy of, one at a time,
            # and what they hold for every sequence beyond the first -- with a share of that for PyTorch's pool, which is not named
            checks = self.model.check_bytes(held["uncopied"]) if "uncopied" in held else 0
            rows = self._check_rows(later, references)
            need += rows + int(rows * CHECK_POOL)
            beyond = f"the {self.sequences - 1} sequence{'s' if self.sequences > 2 else ''} beyond the first"
            if later:
                beyond = "the sequences beyond the first of a " + ("session" if kinded else "count")
            sentence = "\n  " + plan.memory(need, what, checks, rows, beyond)  # before anything is built on the GPU: here a run that does not fit ends
        for session in (self, *later):
            for each in session._first_used():
                each.first_use()
        return self._tables() + sentence

    def _check_rows(self, later: Sequence["Session"], references: int) -> int:
        """What the checks of this run hold for every sequence beyond the first, in bytes: of each session what its family
        counts at its sequences less what it counts at one (``check_row_bytes``; a session of the reference stack runs
        no checks, and a family that counts nothing is not asked).  The checks of one session run at a time and drop
        what they held when they return -- the reference session they build is released there with its state and its
        record, by its last reference: its stage clock holds the device, not the session (``__init__``), and nothing
        but the checks holds it --, so the run takes the largest such part -- and of every other session its record,
        which stays allocated while another count runs."""
        parts, records = [], []
        for session in (self, *later):
            count = getattr(session.model, "check_row_bytes", None) if session.stack != REFERENCE else None
            if count is None:
                parts.append(0)
                records.append(0)
                continue
            shape = (session.capacity, session.tokens, session.record_steps)
            parts.append(count(session.sequences, *shape, references) - count(1, *shape, references))
            records.append(session.model.check_row_terms(session.sequences, *shape)["record"] - session.model.check_row_terms(1, *shape)["record"])
        most = max(range(len(parts)), key=parts.__getitem__)
        return parts[most] + sum(record for index, record in enumerate(records) if index != most)

    def _first_used(self) -> list:
        """The plans whose blocks make their first use for this session: its own and, where it has one, its drafter's."""
        return [*self._plans(own=True), *([self.draft_plan] if self.draft_plan is not None else [])]

    def _drafter_bytes(self) -> int:
        """What a drafter adds to the need of a run, in bytes; 0 without one: what its load and its cache hold
        (``memory_bytes``: the codes, the bf16 copies its reference blocks multiply on, the cache) and the working memory
        of a pass on reference blocks (``working_bytes``)."""
        if self.drafter is None:
            return 0
        return self.drafter.memory_bytes(self.sequences, self.capacity) + self.drafter.working_bytes()

    def _recorded_bytes(self) -> int:
        """What this session's recorded step adds to the need of a run, in bytes; 0 for a session that records none
        (``recordable``) and for a family that states no such figure (``recorded_step_bytes``)."""
        count = getattr(self.model, "recorded_step_bytes", None)
        return count(self.sequences) if self.capture and count is not None and not self.recordable() else 0

    def _tables(self) -> str:
        """The served-by table of this session's step; for a session built with ``draft_tokens=`` the table of its verify
        steps, one line, and the table of the decode steps of its prompt."""
        table = self.resolve().table()
        if self.prefill_plan is not None:
            table = f"{table}\n{PROMPT_ROWS}\n{self.prefill_plan.table()}"
        if self.draft_plan is not None:
            table = f"{table}\n{DRAFTER_ROWS.format(calls=self.drafter.calls_per_pass())}\n{self.draft_plan.table()}"
        return table

    def _kind(self) -> str:
        """This session in the memory sentence of a run that has two kinds of step."""
        sequences = f"{self.sequences} sequence{'s' if self.sequences > 1 else ''}"
        if self.draft_tokens is not None:
            return f"a session of {sequences} x {self.tokens} tokens behind its prompt" + (" with its drafter" if self.drafter is not None else "")
        return f"a session of {sequences} x {self.tokens} tokens" if self.tokens > 1 else f"a plain session of {sequences}"

    # ------------------------------------------------------------------ load
    def _weights_owner(self) -> "Session":
        """The session that loaded the weights this session runs on: itself, or the end of its chain of ``share=`` -- a
        session may share one that shares in turn.  What describes those weights is there: the checkpoint they were
        read or drawn from, which a session that shares never holds, and for seeded weights the seed of that checkpoint
        (``Checkpoint.seed``; a session's own ``seed=`` is only what it makes its checkpoint with when it is handed none)."""
        owner = self
        while owner._share is not None:
            owner = owner._share
        return owner

    def load(self) -> None:
        plan = self.resolve()
        # the layout of a stored weight a provider multiplies on, where it has its own: asked first, so that plans that ask for
        # two row orders of one weight end here -- before an entry builds anything, before a byte is read
        layouts = self.stored_layouts() if self._share is None else {}
        self._paged()  # two layouts of this session's attention cache, or its tables without a pool: the same kind of wrong table, named at the same place
        for each in self._first_used():
            each.first_use()  # made with the table (``plan_table``); a session that is loaded without one makes it here, before the weights
        if self.reference_precision == "ieee-fp32" and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if self._share is not None:
            self.weights = self._share.weights
        else:
            if self.checkpoint is None:
                from .weights import Checkpoint

                self.checkpoint = Checkpoint.synthetic(self.model, self.seed)
            # the forms the plans of the run read, asked of a family's loader only where a block multiplies on a stored format
            forms = {**self._kept(), **({"layouts": layouts} if layouts else {})}  # ... and the layouts asked above
            self.weights = self.model.load(self.checkpoint, self.device, self.stages, self.merge_rule, **forms)
        self._check_forms()
        if self.drafter is not None:  # with the table the model loaded: the drafter holds no second copy of it
            checkpoint = self._weights_owner().checkpoint  # the one the weights were loaded from: a session that shares holds none
            if getattr(checkpoint, "synthetic_weights", False):  # a seeded model: the drafter's tensors are drawn by name from the seed of ITS checkpoint
                from .weights import Checkpoint

                checkpoint = Checkpoint.synthetic(self.drafter, checkpoint.seed)
            self.draft_weights = self.drafter.load(checkpoint, self.device, self.stages, embedding=self.weights.embedding)
        for each, tokens in self._kinds():  # every kind of step this session runs: each block's compile is a stage named with its rows
            for name, block in each.items():
                if getattr(block, "compiles", False):
                    rows = self.sequences * tokens
                    with self.stages.stage(f"compile and first launch {name} ({rows} row{'s' if rows > 1 else ''})"):
                        block.compile(self.sequences, tokens)
        self.reset(fresh=True)
        if self.prefill_plan is not None:  # one decode step of a prompt, on token 0 of the state the verify steps run on
            with self.stages.stage("warm-up, the prompt's decode step"):
                self._run(self.prefill_plan, 1, torch.zeros(self.sequences, dtype=torch.int32, device=self.device))
            if self.drafter is None:
                self.reset(fresh=False)
        with self.stages.stage("warm-up"):
            if self.drafter is None:
                self._run(plan, self.tokens, torch.zeros(self.sequences * self.tokens, dtype=torch.int32, device=self.device))
            else:  # the step of such a session, behind that prompt step: the drafter's passes, the window, the verify step, the pair
                self._calls = 0
                self._speculative_step(None, 0)
        self._report.calls_per_step = self._calls
        self.reset(fresh=True)
        if self.capture:  # the plan's data decides; the state the first request runs on is recorded here, as a stage of its own
            self.python_reason = self.recordable()
            if not self.python_reason:
                with self.stages.stage("capture"):
                    self.record_step()

    def reset(self, fresh: bool = True, recorded: bool = True) -> None:
        """Back to the start of a prompt set: zero state at position 0 and an empty record (both records of a session built
        with ``draft_tokens=``: its prompt's and its verify steps'); ``fresh`` allocates them anew.  What a drafter reads
        of the step before it -- the final-norm rows, the greedy ids, the kept count -- and the drafter's cache are
        part of the state and go with it: a second prompt set starts from nothing of the first.

        ``recorded=False`` is for a request that is timed (the performance mode: ``reset(fresh=False, recorded=False)``,
        the state zeroed in place): its steps keep nothing for the checks.  The record is then one of 0 steps, the form
        the traced pass runs on -- a step copies no ids and no logits into it -- until the next reset that records."""
        if fresh or self.state is None:
            self._release()  # a recorded step writes the state it was recorded on: it goes with that state
            self.state = self._fresh_state()
            self.record = self._record(self.record_steps)
            self.prompt_record = self._prompt_record(self.record_steps)
        else:
            self.state.clear()
            self.record.count = self.record.forced = 0
            self.record.accepted.clear()
            self.record.accepted_all.clear()
            if self.prompt_record is not None:
                self.prompt_record.count, self.prompt_record.prompt = 0, None
        self.tail = None
        steps = self.record_steps if recorded else 0
        if self.record.inputs.shape[0] != steps:  # a request that keeps nothing, or the first one behind it that records again
            self.record, self.prompt_record = self._record(steps), self._prompt_record(steps)
        self._open = False
        self._calls = 0
        self._report.clock.launch.clear()
        self._report.clock.ids.clear()

    def _fresh_state(self) -> DecodeState:
        """A zero state at position 0 for this session's sequences and tokens per sequence, in allocations of its own; its
        attention caches in the layout the blocks of the session's plans ask for (``cache_page``), with the page table
        and the lengths of the run's maximum length where a block reads them (``attended_tables``)."""
        shapes = self.model.state_shapes()
        state = DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, shapes, self.device, self.tokens, **self._paged())
        if self.drafter is not None:  # the drafter's own cache: as many positions as the model's
            state.draft = self.drafter.state(self.sequences, self.capacity, self.device)
        return state

    def _record(self, steps: int) -> Record:
        """An empty record of this session's steps; with a drafter it holds the proposals of every step too."""
        drafted = None if self.drafter is None else (self.sequences, self.draft_tokens)
        return Record(steps, self.sequences * self.tokens, self.model.geometry.vocab, self.device, drafted)

    def _prompt_record(self, steps: int) -> Optional[Record]:
        """An empty record of the prompt's decode steps, one row per sequence, for a session built with ``draft_tokens=``."""
        return Record(steps, self.sequences, self.model.geometry.vocab, self.device) if self.draft_tokens is not None else None

    # ------------------------------------------------------------------ the recorded step
    def _release(self) -> None:
        """Drop the recorded step, if there is one, and what it kept."""
        if self.recorded is not None:
            self.recorded.release()
            self.recorded = None

    def close(self) -> None:
        """Release the recorded step of a session that is done: the sequence and its pool go now, not when the session is
        collected (a sequence the collector destroys inside a later recording invalidates that recording).  The session
        stays usable: its next replay records the step again."""
        self._release()

    def record_step(self) -> str:
        """Record the decode step on the session's present state, where that state has no recorded step yet: one pass of
        the step's Python on PyTorch's capture stream, in which nothing runs.  The first replay of a state does this
        itself; a caller brings it forward to keep it out of a timed interval.  Returns ``python_reason``: ``""`` where
        the step is recorded, else why this session's steps are launched from Python -- recording was not asked for,
        the plan or the kind of session says it cannot be (``recordable``), or this installation declined the pass.

        A declined pass is a net under the plan's data, never the mechanism: the session is left as it was -- the host
        position, the count of block calls, the state's bytes, PyTorch's current stream -- and launches from Python
        from then on, with the first sentence of what was raised as its reason.  A recorded sequence that does not
        hold the plan's launches is not a decline: it is a ``ValueError``, a wrong sequence, named kernel by kernel."""
        if self.python_reason or (self.recorded is not None and self.recorded.state is self.state):
            return self.python_reason
        self.python_reason = self.recordable()
        if self.python_reason:
            return self.python_reason
        self._release()  # of another state: a session holds one recorded step
        state = self.state
        cursor = state.step_cursor()
        cursor.set(state.position)
        position, hidden, greedy, calls = state.position, state.hidden, state.greedy, self._calls
        cuda = self.device.type == "cuda"
        if cuda:  # what the sequence's own pool takes is read off PyTorch's reservation, the cache emptied on both sides
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(self.device) if cuda else 0
        graph = LaunchGraph(self.device)

        def step():
            next_ids, logits = self._run(self.plan, 1, cursor.ids)
            cursor.ids.copy_(next_ids)  # the step's own copy of the ids it chose: the next replay feeds on them
            cursor.advance()  # ... and its own move to the next position: an add per row
            return next_ids, logits, state.hidden

        started, cursor.on = now(), True
        try:
            next_ids, logits, normed = graph.record(step)
        except (RuntimeError, NotImplementedError, ValueError) as error:
            if cuda:
                torch.cuda.synchronize(self.device)
            try:
                graph.reset()
            except RuntimeError:
                pass  # a recording that never began holds nothing to release
            words = (str(error).splitlines() or [type(error).__name__])[0].split(". ")[0].strip().rstrip(".")
            self.python_reason = f"the capture of the step was declined on this installation ({words[:160] or type(error).__name__})"
            return self.python_reason
        finally:  # the pass ran the step's Python and no kernel: the host's side of it is put back, the rows stand where they stood
            cursor.on = False
            made = self._calls - calls
            state.position, state.hidden, state.greedy, self._calls, cursor.position = position, hidden, greedy, calls, position
        seconds = now() - started
        pool = max(torch.cuda.memory_reserved(self.device) - reserved, 0) if cuda else 0
        launches, differs = None, []
        sites, nodes = census.expected(self.plan, self.model), graph.nodes()
        if nodes is not None:
            launches, differs = census.recorded_launches(nodes, sites, copies=1, updates=cursor.launches())
        if differs:
            graph.reset()
            raise ValueError("the recorded step does not hold the launches of the plan: " + "; ".join(differs))
        planned = sum(kernel.launches for site in sites for kernel in site.kernels if kernel.relation == "exact")
        self.recorded = Recorded(graph, state, cursor, next_ids, logits, normed, made, launches, seconds, pool, planned)
        self.pool_bytes.append(pool)
        return ""

    def _replays(self) -> bool:
        """Whether the next decode step is a replay: recording was asked for and stands, and nothing asks for Python
        between two blocks -- no tap of a check, no call log of a traced step.  Records the step where the present
        state has none (the first replay of a state)."""
        if self.python_reason or self.tap is not None or self._call_log is not None:
            return False
        return not self.record_step()

    def _replay(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step as one replay of the recorded step: ``ids`` are fed, the next ids returned -- in the session's
        ids buffer, into which the sequence's own last copy has written them and which the next replay's lookup
        reads.  Handed back, that tensor is fed as it then is; any other tensor is copied into it first.  In front
        of the replay what a step launched from Python has for free: the capacity test (a replay past the capacity
        is the same ``ValueError`` and launches nothing), the ids the caller handed over, the cursor at the state's
        position, the state the sequence was recorded on.  The record holds, for a recorded step, the ids the caller
        HANDED OVER -- copied before the replay, which writes the buffer -- and the logits and the ids the replay
        left: the ids as the token choice wrote them, the tensor the state keeps too (``state.greedy``), never the
        buffer."""
        self._fits(1)
        recorded, record, state = self.recorded, self.record, self.state
        kept = record.count < record.inputs.shape[0]
        recorded.feed(ids)
        if kept:
            record.inputs[record.count].copy_(ids)
        recorded.replay(state)
        state.position += 1
        state.hidden, state.greedy = recorded.hidden, recorded.next_ids  # what the step leaves on the state, as from Python
        if kept:
            record.outputs[record.count].copy_(recorded.next_ids)
            record.logits[record.count].copy_(recorded.logits)
            record.accepted.append(0)
        record.count += 1
        self._calls += recorded.calls
        self._report.replayed, self._report.launches = True, None if recorded.launches is None else recorded.launches["launches"]
        return recorded.cursor.ids

    @property
    def replays(self) -> bool:
        """Whether the steps of the request this session has run were replays of its recorded step."""
        return self._report.replayed

    def read_tail(self) -> Optional[tuple]:
        """Behind a replayed request, the step's own tail, which no recomputation from Python sees: ``(ids, rows,
        position)`` -- whether the ids buffer holds the ids the last replay chose (its own copy of them), whether the
        position rows stand at the state's position (its own adds), and that position.  Kept on the session
        (``tail``) for the checks' line; ``None`` where the request was not replayed.  Host reads: behind a request,
        in front of anything that replays again."""
        recorded = self.recorded
        if not self.replays or recorded is None or recorded.state is not self.state:
            return None
        cursor = recorded.cursor
        ids = bool(recorded.chosen) and cursor.ids.tolist() == recorded.next_ids.tolist()
        self.tail = (ids, cursor.position == self.state.position and cursor.holds(), self.state.position)
        return self.tail

    # ------------------------------------------------------------------ run
    def _call(self, name: str, layer: int, *operands):
        self._calls += 1
        block = (self.plan if self._running is None else self._running)[name]  # of the plan of the step that is running
        if self.on_demand is not None:  # a weight the load kept without a bf16 copy: this session's blocks get one for this call
            operands = self.on_demand.operands(operands)
        out = self.tap(name, layer, block, operands) if self.tap is not None else block(*operands)
        if self._call_log is not None:  # a traced step: the call site, and what the adapter says served the call it has just made
            self._call_log.append(census.Call(name, layer, self._rows, *block.serving()))
        return out

    def _run(self, plan: blocks.Plan, tokens: int, ids: torch.Tensor) -> tuple:
        """One step of ``plan`` at ``tokens`` tokens per sequence on the session's state: ``(next ids, logits)``.  The caller
        names the plan -- the method that was called decides the kind of step -- and the family's step is told the tokens
        only where they are not the state's own (the decode steps of a prompt on a state of several tokens)."""
        self._fits(tokens)
        self._running, self._rows = plan, self.sequences * tokens
        if tokens == self.state.tokens:
            return self.model.step(self._call, self.weights, self.state, ids)
        return self.model.step(self._call, self.weights, self.state, ids, tokens=tokens)

    def _fits(self, tokens: int) -> None:
        """A step of ``tokens`` tokens per sequence from where the state stands: a ``ValueError`` past the session's capacity."""
        if self.state.position + tokens > self.capacity:
            raise ValueError(f"the session was built for {self.capacity} positions per sequence; build it with a larger capacity")

    def _advance(self, ids: torch.Tensor, accept: Optional[int] = 0, proposals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One step of this session's plan on the ids of its rows: the decode step, or -- at several tokens per sequence --
        the verify step, after which the state stands ``accept`` tokens behind the first (``None``: the count is told
        later, ``accept``, and the record's entry stays open until then).  With a drafter the step is behind the
        drafter's passes: ``proposals`` are theirs, or -- left out, a replay of recorded ids -- the passes run here."""
        if self.drafter is not None and proposals is None:
            proposals = self._draft()
        next_ids, logits = self._run(self.plan, self.tokens, ids)
        self.record.add(ids, logits, next_ids, accept, proposals)
        if self.tokens > 1 and accept is not None:
            self.state.accept(accept)
        return next_ids

    def _feed(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step of the prompt of a session built with ``draft_tokens=``: a step of the decode plan on token 0 of
        the state, recorded with the prompt; with a drafter, behind the drafter's pass over the position in front."""
        self._fill(ids)
        next_ids, logits = self._run(self.prefill_plan, 1, ids)
        self.prompt_record.add(ids, logits, next_ids)
        return next_ids

    # ------------------------------------------------------------------ the drafter's passes
    def _draft_call(self, name: str, index: int, *operands):
        """One block call of a draft pass, served by the drafter's plan; ``index`` is the pass, where a step has the layer.
        Not handed to ``tap``: every block of the drafter but its embedding lookup is a reference chain, and the tap of a
        run's checks is built on the model's plans."""
        self._calls += 1
        block = self.draft_plan[name]
        out = block(*operands)
        if self._call_log is not None:  # a traced step: at the rows of the call (a quantised operand follows a call at its row count)
            if isinstance(operands[0], torch.Tensor):
                self._rows = int(operands[0].shape[0])
            self._call_log.append(census.Call(name, index, self._rows, *block.serving()))
        return out

    def _fill(self, ids: torch.Tensor) -> None:
        """In front of a prompt step of a session with a drafter: one pass of the drafter (its body) over the position in
        front of this one -- the final-norm row the step before left, and this step's token, which is the token that
        follows it -- at one row per sequence.  The first prompt step has no position in front of it; the pass over the
        last prompt position is the first pass of the first step, which needs the first new token."""
        state = self.state
        if self.drafter is not None and state.hidden is not None:
            self.drafter.body(self._draft_call, self.draft_weights, state.draft, state.hidden, ids, state.position - 1)

    def _draft(self) -> torch.Tensor:
        """The drafter's passes of a step -> its proposals ``[sequences, draft_tokens]`` int32 on the device.  It is handed
        what the step before left: every final-norm row of that step and the greedy id after each (one row behind the
        prompt, ``tokens`` behind a verify step), the position of that step's first row, and how many drafted tokens the
        run kept of it -- one host integer, the row the passes go on from."""
        state, kept = self.state, self.state.accepted
        self._behind_prompt()
        return self.drafter.draft(
            self._draft_call, self.draft_weights, state.draft, state.hidden, state.greedy, state.position - (kept + 1), kept, self.draft_tokens
        )

    def _behind_prompt(self) -> None:
        """A step of a session with a drafter stands behind its prompt: a ``ValueError`` in front of it, where no step has
        left a final-norm row."""
        if self.state.hidden is None:
            raise ValueError("the first step of a session with a drafter drafts behind the last prompt position: feed the prompt first (Session.prefill)")

    def _speculative_step(self, force: Optional[torch.Tensor], accept: Optional[int]) -> torch.Tensor:
        """One step of speculative decoding: the drafter's passes; the window -- the greedy token the run kept last, then
        the proposals (``force``: these drafted tokens in their place) --; the verify step on it; -> the pair ``[sequences,
        2, tokens]`` of the window and the greedy id after each of its tokens.  The window and the pair are one
        concatenation each; nothing is read."""
        proposals = self._draft()
        last = self.state.greedy.view(self.sequences, -1)[:, self.state.accepted : self.state.accepted + 1]  # a view by a host integer
        window = torch.cat([last, proposals if force is None else force], dim=1)
        next_ids = self._advance(window.reshape(-1), accept, proposals)
        return torch.stack([window, next_ids.view(self.sequences, self.tokens)], dim=1)

    def _closed(self, what: str) -> None:
        """``what`` replays steps with their kept counts: a ``ValueError`` while the last verify step waits for its count."""
        if self._open:
            raise ValueError(
                f"{what} replays every step with its kept count; the last verify step of this session has not been told its count (Session.accept)"
            )

    def _decode_only(self, what: str) -> None:
        if self.tokens > 1:
            raise ValueError(
                f"{what} feeds one token per sequence; a step of this session feeds {self.tokens} (Session.verify).  The decode step is a "
                "row of another plan: a second session on these weights (share=), planned with this one"
            )

    def prefill(self, prompts: List[torch.Tensor]) -> torch.Tensor:
        """Feed one prompt per sequence and return the first new ids ``[sequences]`` on the GPU.  Route ``decode-step``: every
        prompt token goes through the decode step, one step per token, so the prompts of one run have one length.

        On a session built with ``draft_tokens=`` these are steps of its decode plan on token 0 of its state: they come
        before its first verify step (a ``ValueError`` behind one), and the prompt is kept whole for the replays of the
        checks and of the traced pass (``prompt_record``)."""
        if self.draft_tokens is None:
            self._decode_only("Session.prefill")
        elif self.state.verify_steps:
            raise ValueError("the prompt of a session built with draft_tokens= is fed before its first verify step")
        if len(prompts) != self.sequences:
            raise ValueError(f"this session has {self.sequences} sequences; got {len(prompts)} prompts")
        lengths = {int(p.numel()) for p in prompts}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError(f"route decode-step feeds the prompts in lockstep: they need one non-zero length; got {sorted(lengths)}")
        self._prefill_start = now()
        ids = torch.stack([p.to(torch.int32) for p in prompts]).to(self.device)
        self._report.text_path["to_device"] = now() - self._prefill_start
        self._report.prompt_tokens = ids.shape[1]
        self._report.prefill_route = "decode step"
        feed = self._advance
        if self.draft_tokens is not None:
            fed = self.prompt_record.prompt  # a prompt fed in parts is one prompt: the replays feed all of it
            self.prompt_record.prompt = ids if fed is None else torch.cat([fed, ids], dim=1)
            self._report.prompt_tokens = self.prompt_record.prompt.shape[1]
            feed = self._feed
        elif self._replays():  # one copy of the token into the ids buffer and one replay per prompt token
            feed = self._replay
        next_ids = None
        for t in range(ids.shape[1]):
            next_ids = feed(ids[:, t].contiguous())
        return next_ids

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step for every sequence: the ids to feed ``[sequences]`` -> the next ids ``[sequences]``, on the GPU.
        The ids handed over are fed whatever they are.  On a session that records its step this is one replay, and
        the tensor returned is the session's ids buffer: the next step reads the ids it is fed from it and writes
        the ids it chooses into it.  Handed back, it is fed as it then is, changed in place or not; any other tensor
        is copied into it.  A caller that keeps the ids of a step past the next one copies them."""
        if self.drafter is not None:
            raise ValueError(
                "behind its prompt a session with a drafter runs steps of speculative decoding (Session.speculate); a decode step behind a keep is not built"
            )
        if self.draft_tokens is not None:
            raise ValueError(
                "behind its prompt a session built with draft_tokens= runs verify steps (Session.verify); a decode step behind a keep is not built"
            )
        self._decode_only("Session.step")
        replays = self._replays()  # in front of the clock: the first replay of a state records the step
        self._report.clock.mark_launch()
        return self._replay(ids) if replays else self._advance(ids)

    def verify(self, ids: torch.Tensor, accept: Optional[int] = None) -> torch.Tensor:
        """One verify step for every sequence: ``ids`` ``[sequences, tokens]`` on the GPU -- per sequence the token the run
        kept last and the tokens drafted behind it -- -> the next ids ``[sequences, tokens]``: at ``[i, t]`` the greedy
        token after token ``t`` of sequence ``i``, given the tokens in front of it.  One launch sequence for all rows
        (sequences x tokens); the recurrent state after every token is written to a slot of its own.

        ``accept`` says how many of the tokens behind the first the run keeps, 0 .. tokens - 1, one number for the fixed
        set of sequences (so the attention caches keep one position): the state then stands at that token, and the next
        step goes on ``accept + 1`` positions later.  Which tokens are kept is the caller's rule (a speculative loop
        keeps a drafted token while it equals the greedy token in front of it); this method compares nothing.  Nothing
        is read to the host and nothing is copied for it: the one host read of a step is the caller's (``read``).

        On a session built with ``draft_tokens=`` the count may be left out: the step runs, the caller reads its greedy
        ids and says how many drafted tokens it keeps with ``accept``; the next ``verify`` before that is a ``ValueError``.
        On a session built with ``tokens=`` the count is part of the step.  A session with a drafter refuses this method
        (a ``ValueError``): its step is ``speculate``, which runs the drafter's passes in front of the verify step."""
        if self.tokens == 1:
            raise ValueError("Session.verify is the step of a session built for several tokens per sequence (tokens=); this session's step is Session.step")
        if self.drafter is not None:
            raise ValueError(
                "the step of a session with a drafter is Session.speculate (force= verifies the caller's drafted tokens): "
                "a verify step without the drafter's passes in front of it would leave the drafter's cache without its positions"
            )
        if tuple(ids.shape) != (self.sequences, self.tokens):
            raise ValueError(
                f"a verify step of this session takes ids [{self.sequences}, {self.tokens}] (sequences, tokens per sequence); got {tuple(ids.shape)}"
            )
        waits = accept is None and self.draft_tokens is not None  # the count follows the host read (``accept``)
        if not waits and (isinstance(accept, bool) or not isinstance(accept, int) or not 0 <= accept < self.tokens):
            raise ValueError(f"a verify step of {self.tokens} tokens per sequence keeps 0 to {self.tokens - 1} tokens behind the first; got accept={accept!r}")
        if self._open:
            raise ValueError("the step before this one has not been told how many of its tokens are kept (Session.accept)")
        self._report.clock.mark_launch()
        next_ids = self._advance(ids.to(torch.int32).reshape(-1), accept).view(self.sequences, self.tokens)
        self._open = waits
        if not waits:
            self.record.accepted_all.append(accept)
        return next_ids

    def speculate(self, force: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One step of speculative decoding on a session with a drafter -> ``[sequences, 2, tokens]`` int32 on the GPU:
        ``[:, 0]`` the window that was verified -- the token the run kept last, then the drafted tokens -- and ``[:, 1]``
        the greedy token after each of them.  In order: the drafter's ``draft_tokens`` passes (the first over the rows
        of the step before, the others on the drafter's own row and proposal), the window, one verify step, the pair.

        ``force`` ``[sequences, draft_tokens]`` int32 on the GPU: the drafted tokens to verify in the place of the
        drafter's.  The drafter runs all the same, and its own proposals are in the record (``record.proposals``).

        It reads nothing and compares nothing: the caller reads the pair (``read``: the ONE host read of the step),
        counts the kept drafted tokens (``sampling.accepted``) and says so (``accept``); a second step before that is a
        ``ValueError``.  So is a step in front of the prompt, and a session without a drafter: its caller builds the
        window and calls ``verify``."""
        if self.drafter is None:
            raise ValueError(
                "Session.speculate drafts the tokens of its step: it needs a drafter (drafter=); without one the caller builds the window and calls Session.verify"
            )
        if self._open:
            raise ValueError("the step before this one has not been told how many of its tokens are kept (Session.accept)")
        if force is not None and (tuple(force.shape) != (self.sequences, self.draft_tokens) or force.dtype != torch.int32):
            raise ValueError(
                f"force is the drafted tokens of the step, int32 [{self.sequences}, {self.draft_tokens}] (sequences, draft tokens); "
                f"got {force.dtype} {tuple(force.shape)}"
            )
        self._behind_prompt()  # a refused step runs nothing and leaves nothing: no pass, no entry of the record, no mark on the report's clock
        self._fits(self.tokens)  # before the passes: a step past the capacity writes nothing, the drafter's cache included
        self._report.clock.mark_launch()
        pair = self._speculative_step(force, None)
        self._open = True
        self.record.forced += force is not None
        return pair

    def accept(self, count: int) -> None:
        """Say how many drafted tokens the verify step that has just run keeps -- the step of ``verify`` given no count, or of
        ``speculate`` --: one whole number, 0 .. tokens - 1, for the fixed set of sequences.  The state then stands at
        that token and the record of the step holds the count.  It launches nothing and copies nothing.

        One count for several sequences is exactly greedy for every sequence only if it is the smallest count of the
        set -- a larger one keeps a token some sequence did not accept -- and that is the count ``sampling.accepted``
        returns from the host read of the step.  ``ValueError`` when no step waits for a count (``verify`` was given
        one, or no step has run)."""
        if not self._open and self.drafter is not None:  # the step such a session was asked for is speculate
            raise ValueError("no step of this session waits for its kept count: Session.accept follows a Session.speculate (a session with a drafter)")
        if not self._open:
            raise ValueError(
                "no verify step of this session waits for its kept count: Session.accept follows a Session.verify that was given none "
                "(a session built with draft_tokens=)"
            )
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < self.tokens:
            raise ValueError(f"a verify step of {self.tokens} tokens per sequence keeps 0 to {self.tokens - 1} tokens behind the first; got accept={count!r}")
        if self.record.count <= len(self.record.accepted):  # the step is among the recorded ones: its entry was left open
            self.record.accepted[self.record.count - 1] = count
        self.state.accept(count)
        self._open = False
        self.record.accepted_all.append(count)

    def read(self, ids: torch.Tensor) -> List[int]:
        """The one host read of a step: the chosen ids, 4 bytes per sequence -- or the pair of a step of speculative decoding
        (``speculate``), 8 bytes per token of a sequence."""
        tokens = ids.tolist()
        if not self._report.clock.ids and not self._report.clock.launch:
            self._report.prefill_seconds = now() - self._prefill_start
        else:
            self._report.clock.mark_ids()
        return tokens

    def trace(self, steps: int = 32, records: Optional[bool] = None) -> None:
        """The traced pass of the report: replay the first ``steps`` recorded inputs of this run from a zero state under the
        profiler, each step ending in the host read of its ids, as in the loop.  The run's own state and record are put
        back afterwards.  Where kernel activity records cannot be had the report says why.

        With ``records`` (default: the session's ``check_kernels``) the pass also keeps ``self.traced`` for the kernel check:
        the sites of a step with what the entries state about their launches, read before the first traced step; the log
        of every block call of every traced step; and the launch records of the pass, cut into steps -- or why there are
        none.

        On a session built with ``draft_tokens=`` the traced steps are its verify steps: the pass first replays the whole
        prompt from a zero state WITHOUT the profiler, up to the host read that ends a prompt, then the first ``steps``
        recorded verify steps with their kept counts under it.  The launch records are those of verify steps alone, held
        against the verify plan; the kernels of the prompt's decode steps are not observed.

        On a session that replayed its request the pass runs ``REPLAYED_TRACED`` steps more: the first traced steps
        again as replays of the recorded step, on the run's own state from zero (saved before, written back after).
        The report then has the GPU-busy figure and the launches of both forms (``Trace.replayed``), and the launch
        records of the replays are kept beside those of the steps launched from Python (``Traced.replayed``).

        With a drafter a traced step is the step ``speculate`` runs, fed the recorded drafted tokens: the passes, the
        window, the verify step and the pair, ended by the host read of the pair.  The sites of such a step are, for
        every pass, the drafter's wiring with the pass's number in the place of the layer, then the model's.  The passes
        over the prompt run in front of the window, untraced, with the prompt."""
        if self.on_demand is not None:  # the reference of a run's checks: its calls carry the copies it makes, which no step of a run does
            raise ValueError(
                "a session that makes bf16 copies of its weights per call (on_demand) is the reference of a run's checks; its steps are not traced"
            )
        records = self.check_kernels if records is None else bool(records)
        passes = census.passes(self.draft_plan, self.drafter, self.draft_tokens) if records and self.drafter is not None else []
        sites = passes + census.expected(self.plan, self.model) if records else []
        self._closed("a traced pass")
        count = min(steps, self.record.count, self.record.inputs.shape[0])
        if count < 1:
            self._report.trace_missing = "needs a recorded step on a CUDA device"
            if self.draft_tokens is not None:  # the prompt's decode steps are replayed in front of the pass, never traced
                self._report.trace_missing = "needs a recorded verify step behind the prompt on a CUDA device"
            if records:
                self.traced = census.Traced(sites, [], missing=self._report.trace_missing, rows=self.sequences * self.tokens)
            return
        inputs, accepted, calls = self.record.inputs[:count], list(self.record.accepted[:count]), self._calls
        kept = (self.state, self.record, self.prompt_record)
        prompt = None if self.prompt_record is None else self.prompt_record.prompt
        # a session that replayed its request: the first of the traced steps once more as replays, in the same pass, on the
        # run's own state -- the one the sequence was recorded on -- from zero; what that state held is saved and written back
        own, live = self.state, self.recorded
        tail = min(REPLAYED_TRACED, count) if self.replays and live is not None and live.state is own else 0
        saved = own.snapshot() if tail else None
        if tail:  # in front of the profiler: the traced replays then hold a step's launches and nothing of this
            own.clear()
            live.cursor.set(0)
        self.state = self._fresh_state()
        self.record = self._record(0)
        self.prompt_record = self._prompt_record(0)
        call_log: List[list] = []
        launches: list = []

        def step(i: int) -> None:
            if i >= count:  # the tail: step ``i - count`` again, by replay, ended by its host read
                self.state, self._call_log = own, None
                self._replay(inputs[i - count]).tolist()
                return
            if records:
                self._call_log = []
                call_log.append(self._call_log)
            if self.drafter is None:
                self._advance(inputs[i], accepted[i]).tolist()
            else:  # the step as ``speculate`` runs it, on the recorded drafted tokens; the host read is the pair's
                self._speculative_step(inputs[i].view(self.sequences, self.tokens)[:, 1:], accepted[i]).tolist()

        try:  # the device is trace_steps' to decline: what is swapped here is put back whatever it does
            if prompt is not None:  # the prompt of the traced steps, from the zero state: in front of the profiler, in no record of it
                for t in range(prompt.shape[1]):
                    next_ids = self._feed(prompt[:, t].contiguous())
                    # ... ended as the loop ends a prompt, by the host read of its last ids: none of its kernels is in flight when the window opens
                    if t + 1 == prompt.shape[1]:
                        next_ids.tolist()
            more = {"tail": tail} if tail else {}  # a pass without replays is asked for as it always was
            if records:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device, records=launches, **more)
            else:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device, **more)
        finally:
            (self.state, self.record, self.prompt_record), self._calls, self._call_log = kept, calls, None
            if saved is not None:
                own.restore(saved)
        if records:
            self.traced = census.Traced.of(sites, call_log, launches, count, self._report.trace_missing, self.sequences * self.tokens, **more)
            if tail:  # what the recorded step launches behind its blocks: the copy of the chosen ids, the position updates
                self.traced.own_tail = (1, live.cursor.launches())

    def trace_request(self, step, steps: int) -> None:
        """The traced steps of a request that is running (the performance mode's traced request): ``step(i)`` for ``i`` in
        ``range(steps)`` under the profiler, with launch records.  ``step`` is the caller's own loop body -- one
        ``Session.step`` from where the state stands and its host read (``read``) -- so a traced step is launched as the
        untraced steps of the request were.  Unlike ``trace`` nothing is replayed, swapped or put back: the state, its
        allocations and the ids are the request's own, and the prompt in front of the traced steps is the caller's, fed
        untraced and closed by its host read.  It keeps what ``trace`` keeps for the kernel check (``self.traced``: the
        sites of a step, one call log per traced step, the launch records cut into steps, or why there are none)."""
        self._decode_only("Session.trace_request")
        if self.on_demand is not None or self.draft_tokens is not None:  # the reference of a run's checks, as in ``trace``; a session behind a prompt
            raise ValueError("Session.trace_request traces decode steps of a session that runs on its own weights (no on_demand, no draft_tokens)")
        sites = census.expected(self.plan, self.model)
        call_log: List[list] = []
        launches: list = []

        def logged(i: int) -> None:
            self._call_log = []
            call_log.append(self._call_log)
            step(i)

        try:
            self._report.trace, self._report.trace_missing = trace_steps(logged, steps, self.device, records=launches)
        finally:
            self._call_log = None
        self.traced = census.Traced.of(sites, call_log, launches, steps, self._report.trace_missing, self.sequences)

    def trace_replays(self, step, steps: int) -> tuple:
        """The replayed steps of a request that is running, under the profiler: ``step(i)`` for ``i`` in ``range(steps)``, the
        caller's own loop body -- one ``Session.step``, which is one replay, and its host read.  For a caller that has
        traced the same steps launched from Python (``trace_request``): the launch records of the replays are kept
        beside those (``Traced.replayed``), so the kernel check holds the recorded step to the step launched from
        Python, launch by launch.  Returns ``(timing.Trace, "")`` of the replayed steps, or ``(None, reason)`` where
        their records cannot be had.  ``ValueError`` on a session whose present state has no recorded step, or that
        has traced no request."""
        self._decode_only("Session.trace_replays")
        if self.recorded is None or self.recorded.state is not self.state or self.python_reason:
            raise ValueError("Session.trace_replays traces replays of the recorded step: this session's present state has none")
        if self.traced is None:
            raise ValueError("Session.trace_replays follows a traced request of steps launched from Python (Session.trace_request), which it is held against")
        launches: list = []
        trace, missing = trace_steps(step, steps, self.device, records=launches)
        cut, _ = census.split_steps(launches)
        self.traced.replayed = cut if launches and len(cut) == steps else []
        self.traced.own_tail = (1, self.recorded.cursor.launches())
        return trace, missing

    def new_report(self) -> None:
        """A fresh report for the next prompt set of this session; the block calls per step stay the session's."""
        calls = self._report.calls_per_step
        self._report = self._new_report()
        self._report.calls_per_step = calls

    def _new_report(self) -> Report:
        reduced = bool(getattr(self.model, "reductions", ()))
        return Report(self.model.name, reduced=reduced, tokens=self.tokens, draft_tokens=self.draft_tokens, drafter=self.drafter is not None)

    def report(self) -> Report:
        """The report of this session's run.  The bytes of a step are the family's arithmetic for a decode step (one embedding
        row, one recurrent state and three history columns per sequence): a session of several tokens per sequence states
        none, and its report says so, as it does of the first-token line and of the rate (``timing.Report``).  A session
        with a drafter states the bytes of its step -- the verify step's arithmetic at its tokens per sequence and a draft
        pass's, ``draft_tokens`` times -- and the counts its steps kept (``Report.accepted``; ``Report.scheduled``: some step
        verified drafted tokens of the caller's), as the record of the run has them."""
        if self.drafter is not None and self.state is not None:
            # the weights a block of the verify plan multiplies on the stored format are read there, not as a copy
            stored = tuple(label for label, _, _, _, block in self.plan.parts() if getattr(block, "stored_weights", False))
            read = {"stored": stored} if stored else {}
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position, tokens=self.tokens, **read)
            self._report.pass_bytes = self.drafter.bytes_per_pass(self.state.position)
            self._report.accepted, self._report.scheduled = list(self.record.accepted_all), bool(self.record.forced)
        if hasattr(self.model, "bytes_per_step") and self.state is not None and self.tokens == 1:
            # a block of this session's plan that multiplies on the stored format reads its weight there, not as a copy
            # (asked of a family's model only where the plan has such a block)
            stored = self.stored_weights(own=True)
            read = {"stored": stored} if stored else {}
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position, **read)
        return self._report
