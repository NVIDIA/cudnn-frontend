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
traced steps and the profiler's record of every launch -- and, for a stack that declares a target kernel set, what the run can
say about that set (``claim``).

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

The set of sequences is fixed before the first step: a sequence that reaches its end token stays in the batch, so the number of
rows, and with it the kernel path, does not change inside a run.  A session keeps no state between two prompt sets other than
what ``reset`` clears.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from . import blocks, census
from .sampling import GREEDY
from .state import DecodeState
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
        #: of EVERY verify step of the run, recorded or not: the drafted tokens it kept, and how many steps verified drafted
        #: tokens of the caller's in the place of a drafter's (``Session.speculate(force=)``) -- the report's acceptance lines
        self.kept: List[int] = []
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
    ):
        """``capacity`` is the number of positions (prompt + new tokens) a sequence may reach.  ``checkpoint`` defaults to the
        model's seeded weights.  ``share`` reuses another session's loaded weights (the reference session of the tie-out).
        ``record_steps`` bounds the steps whose ids and logits are kept on the GPU for the checks (default: every step).
        ``check_kernels`` says that the run will check its kernels: what that needs of a provider is then part of the plan
        (a decline is a SKIP before the load), and the traced pass keeps its launch records.  A stack that declares a target
        kernel set (the family's ``target_stacks``) always checks its kernels: its statement is about them.
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
        ``draft_tokens`` and serves one sequence."""
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
        self.model, self.text, self.sequences, self.stack, self.prefill_route = model, text, int(sequences), stack, prefill
        self.tokens = tokens  # per sequence, of a step of this session's plan
        self.draft_tokens = draft_tokens  # of a session that feeds its prompt through decode steps and verifies behind it; else None
        self.drafter = drafter  # what drafts those tokens; None: the caller does
        self.draft_plan: Optional[blocks.Plan] = None  # the drafter's plan, at one row per sequence
        self.draft_weights = None
        self.device = torch.device(device)
        self.overrides, self.reference_precision, self.capacity = dict(overrides or {}), reference_precision, int(capacity)
        self.checkpoint, self.seed, self.merge_rule, self._share = checkpoint, seed, merge_rule, share
        self.record_steps = self.capacity if record_steps is None else max(1, min(int(record_steps), self.capacity))
        self.check_kernels = bool(check_kernels) or stack in getattr(model, "target_stacks", ())
        self.on_demand = on_demand  # the maker of the bf16 copies this session's blocks read and its weights do not hold
        self.traced: Optional[census.Traced] = None  # of the last traced pass that kept its launch records
        self._call_log: Optional[list] = None  # the block calls of the traced step that is running
        self.stages = Stages(sync=lambda: torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None)
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

    def claim(self) -> Optional[census.Claim]:
        """What this session's run can say about the target kernel set of its stack -- the identity of every block call, the
        three numbers, the reasons its GPU class is not claimed (``census.Claim``) -- or ``None`` for a stack that declares no
        target.  Decided from the plan and the family's census file: before anything is loaded.  ``ValueError`` when that
        file is not a census of this model."""
        if self.stack not in getattr(self.model, "target_stacks", ()):
            return None
        plan = self.resolve()
        return self.model.target().claim(plan, self.model, census.configuration_of(plan))

    def plan_table(self, later: Sequence["Session"] = ()) -> str:
        """The served-by table and the memory sentence; raises ``NotImplementedError`` (a SKIP) before anything is loaded.

        ``later`` are the sessions that will run after this one in the same process on its weights (``share=self``).  Their
        plans are resolved here too, and the memory sentence then covers the whole run -- this session's need plus their
        state -- against what is free now: every decline of a run comes before its first load, none after it.

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
        sentence = ""
        if self.device.type == "cuda":
            # what the load of the first session holds follows the plans: the stored format of the weights a block multiplies
            # on, and no bf16 copy of those no block reads as one (asked of a family's model only where a plan has such a block)
            held = self._kept() if self._share is None else {}
            need = self.model.memory_bytes(self.sequences, self.capacity, **held, **self._per_token()) - (
                self.model.copy_bytes() if self._share is not None else 0
            )
            need += sum(session.model.state_bytes(session.sequences, session.capacity, **session._per_token()) for session in later)
            need += sum(session._drafter_bytes() for session in (self, *later))
            # one state per session: a session that feeds its prompt through decode steps runs them on the state of its verify
            # steps, so it is counted once.  Nothing is added for its second plan either: what a first launch of the fused step
            # reserves outside the pool is reserved once in a process, and the sentence carries it once (``Plan.memory``)
            counts = [self.sequences, *(session.sequences for session in later)]
            what = f" for {', '.join(str(n) for n in counts[:-1])} and {counts[-1]} sequences, one count after the other" if later else ""
            if any(session.draft_tokens is not None for session in (self, *later)):  # a run with two kinds of step names them
                kinds = [session._kind() for session in (self, *later)]
                what = f" for {', '.join(kinds[:-1])} and {kinds[-1]}, one after the other" if later else f" for {kinds[0]}"
            # the part of the need that is the checks': the reference's copy of a weight the load keeps no copy of, one at a time
            checks = self.model.check_bytes(held["uncopied"]) if "uncopied" in held else 0
            sentence = "\n  " + plan.memory(need, what, checks)  # before anything is built on the GPU: this is where a run that does not fit ends
        for session in (self, *later):
            for each in session._first_used():
                each.first_use()
        return self._tables() + sentence

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
    def load(self) -> None:
        plan = self.resolve()
        # the layout of a stored weight a provider multiplies on, where it has its own: asked first, so that plans that ask for
        # two row orders of one weight end here -- before an entry builds anything, before a byte is read
        layouts = self.stored_layouts() if self._share is None else {}
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
            source = self if self._share is None else self._share
            checkpoint = source.checkpoint
            if getattr(checkpoint, "synthetic_weights", False):  # a seeded model: the drafter's tensors are drawn by name from the same seed
                from .weights import Checkpoint

                checkpoint = Checkpoint.synthetic(self.drafter, source.seed)
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
                self._engine(None, 0)
        self._report.calls_per_step = self._calls
        self.reset(fresh=True)

    def reset(self, fresh: bool = True) -> None:
        """Back to the start of a prompt set: zero state at position 0 and an empty record (both records of a session built
        with ``draft_tokens=``: its prompt's and its verify steps'); ``fresh`` allocates them anew.  What a drafter reads
        of the step before it -- the final-norm rows, the greedy ids, the kept count -- and the drafter's cache are
        part of the state and go with it: a second prompt set starts from nothing of the first."""
        if fresh or self.state is None:
            self.state = self._fresh_state()
            self.record = self._record(self.record_steps)
            self.prompt_record = self._prompt_record(self.record_steps)
        else:
            self.state.clear()
            self.record.count = self.record.forced = 0
            self.record.accepted.clear()
            self.record.kept.clear()
            if self.prompt_record is not None:
                self.prompt_record.count, self.prompt_record.prompt = 0, None
        self._open = False
        self._calls = 0
        self._report.clock.launch.clear()
        self._report.clock.ids.clear()

    def _fresh_state(self) -> DecodeState:
        """A zero state at position 0 for this session's sequences and tokens per sequence, in allocations of its own."""
        shapes = self.model.state_shapes()
        state = DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, shapes, self.device, self.tokens)
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
        Not handed to ``tap``: every block of the drafter is the reference chain, and the tap of a run's checks is built
        on the model's plans."""
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
        if state.hidden is None:
            raise ValueError("the first step of a session with a drafter drafts behind the last prompt position: feed the prompt first (Session.prefill)")
        return self.drafter.draft(
            self._draft_call, self.draft_weights, state.draft, state.hidden, state.greedy, state.position - (kept + 1), kept, self.draft_tokens
        )

    def _engine(self, force: Optional[torch.Tensor], accept: Optional[int]) -> torch.Tensor:
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
        next_ids = None
        for t in range(ids.shape[1]):
            next_ids = feed(ids[:, t].contiguous())
        return next_ids

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """One decode step for every sequence: the ids to feed ``[sequences]`` -> the next ids ``[sequences]``, on the GPU."""
        if self.draft_tokens is not None:
            raise ValueError(
                "behind its prompt a session built with draft_tokens= runs verify steps (Session.verify); a decode step behind a keep is not built"
            )
        self._decode_only("Session.step")
        self._report.clock.mark_launch()
        return self._advance(ids)

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
        On a session built with ``tokens=`` the count is part of the step."""
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
            self.record.kept.append(accept)
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
        if self.state.hidden is not None:  # (in front of the prompt the passes refuse the step by name)
            self._fits(self.tokens)  # before the passes: a step past the capacity writes nothing, the drafter's cache included
        self._report.clock.mark_launch()
        pair = self._engine(force, None)
        self._open = True
        self.record.forced += force is not None
        return pair

    def accept(self, count: int) -> None:
        """Say how many drafted tokens the verify step that has just run keeps: one whole number, 0 .. tokens - 1, for the
        fixed set of sequences.  The state then stands at that token and the record of the step holds the count.  It
        launches nothing and copies nothing.

        One count for several sequences is exactly greedy for every sequence only if it is the smallest count of the
        set -- a larger one keeps a token some sequence did not accept -- and that is the count ``sampling.accepted``
        returns from the host read of the step.  ``ValueError`` when no step waits for a count (``verify`` was given
        one, or has not run)."""
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
        self.record.kept.append(count)

    def read(self, ids: torch.Tensor) -> List[int]:
        """The one host read of a step: the chosen ids, 4 bytes per sequence."""
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

        With a drafter a traced step is the step ``speculate`` runs, fed the recorded drafted tokens: the passes, the
        window, the verify step and the pair, ended by the host read of the pair.  The sites of such a step are, for
        every pass, the drafter's wiring with the pass's number in the place of the layer, then the model's; the claim
        is the verify plan's and speaks about the model's sites (``census.Traced.first``).  The passes over the prompt
        run in front of the window, untraced, with the prompt."""
        if self.on_demand is not None:  # the reference of a run's checks: its calls carry the copies it makes, which no step of a run does
            raise ValueError(
                "a session that makes bf16 copies of its weights per call (on_demand) is the reference of a run's checks; its steps are not traced"
            )
        records = self.check_kernels if records is None else bool(records)
        passes = census.passes(self.draft_plan, self.drafter, self.draft_tokens) if records and self.drafter is not None else []
        sites = passes + census.expected(self.plan, self.model) if records else []
        claim = self.claim() if records else None
        self._closed("a traced pass")
        count = min(steps, self.record.count, self.record.inputs.shape[0])
        if count < 1:
            self._report.trace_missing = "needs a recorded step on a CUDA device"
            if self.draft_tokens is not None:  # the prompt's decode steps are replayed in front of the pass, never traced
                self._report.trace_missing = "needs a recorded verify step behind the prompt on a CUDA device"
            if records:
                self.traced = census.Traced(sites, [], missing=self._report.trace_missing, rows=self.sequences * self.tokens, claim=claim, first=len(passes))
            return
        inputs, accepted, calls = self.record.inputs[:count], list(self.record.accepted[:count]), self._calls
        kept = (self.state, self.record, self.prompt_record)
        prompt = None if self.prompt_record is None else self.prompt_record.prompt
        self.state = self._fresh_state()
        self.record = self._record(0)
        self.prompt_record = self._prompt_record(0)
        call_log: List[list] = []
        launches: list = []

        def step(i: int) -> None:
            if records:
                self._call_log = []
                call_log.append(self._call_log)
            if self.drafter is None:
                self._advance(inputs[i], accepted[i]).tolist()
            else:  # the step as ``speculate`` runs it, on the recorded drafted tokens; the host read is the pair's
                self._engine(inputs[i].view(self.sequences, self.tokens)[:, 1:], accepted[i]).tolist()

        try:  # the device is trace_steps' to decline: what is swapped here is put back whatever it does
            if prompt is not None:  # the prompt of the traced steps, from the zero state: in front of the profiler, in no record of it
                for t in range(prompt.shape[1]):
                    next_ids = self._feed(prompt[:, t].contiguous())
                    # ... ended as the loop ends a prompt, by the host read of its last ids: none of its kernels is in flight when the window opens
                    if t + 1 == prompt.shape[1]:
                        next_ids.tolist()
            if records:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device, records=launches)
            else:
                self._report.trace, self._report.trace_missing = trace_steps(step, count, self.device)
        finally:
            (self.state, self.record, self.prompt_record), self._calls, self._call_log = kept, calls, None
        if records:
            self.traced = census.Traced.of(sites, call_log, launches, count, self._report.trace_missing, self.sequences * self.tokens, claim, len(passes))

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
        pass's, ``draft_tokens`` times -- and the counts its steps kept (``Report.kept``; ``Report.scheduled``: some step
        verified drafted tokens of the caller's), as the record of the run has them."""
        if self.drafter is not None and self.state is not None:
            # the weights a block of the verify plan multiplies on the stored format are read there, not as a copy
            stored = tuple(label for label, _, _, _, block in self.plan.parts() if getattr(block, "stored_weights", False))
            read = {"stored": stored} if stored else {}
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position, tokens=self.tokens, **read)
            self._report.pass_bytes = self.drafter.bytes_per_pass(self.state.position)
            self._report.kept, self._report.scheduled = list(self.record.kept), bool(self.record.forced)
        if hasattr(self.model, "bytes_per_step") and self.state is not None and self.tokens == 1:
            # a block of this session's plan that multiplies on the stored format reads its weight there, not as a copy
            # (asked of a family's model only where the plan has such a block)
            stored = self.stored_weights(own=True)
            read = {"stored": stored} if stored else {}
            self._report.bytes = self.model.bytes_per_step(self.sequences, self.state.position, **read)
        return self._report
