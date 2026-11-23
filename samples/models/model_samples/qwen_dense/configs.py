# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provider table and the regime table of the dense Qwen family, as data.

For every block the rows are ordered; the first row whose GPU classes, sequence range, tokens-per-sequence range and stack
contain the run serves the block.  A row whose entry serves some layer widths only names them (``widths``): a model of the
family with another width is not that row's, and the next row is asked.  The whole table is printed before every run.  A row with a ``fallback`` reason is a
reference chain or a multiply over a dequantised copy; ``band`` is the tie-out band of the row against the reference block
(``model_samples.tie_out.BANDS``) and is never widened in place.  ``form`` is what the row's adapter passes to its entry: fields
of the entry's own form, which the kernel check compares with the entry's answer.  A row whose entry a step calls in several
forms names them by call site (``sites``; the call sites of this family's wiring: ``QwenDense.sites``).  A row may serve
some call sites of its block only: it names every other site without fields, and the calls there are served by the next
row of the stack that serves the run -- the plan then prints the block once per call site.  A stack adds its rows here;
it removes none.

The stacks:

* ``fe-cublas`` -- the reference path: the fused linear-attention step on its entry's own pick, every multiply on cuBLAS over
  a bf16 copy, the rest in PyTorch;
* ``native`` -- every block on a kernel that works on the stored formats: an entry of cuDNN Frontend, or a call of a
  library (``LIBRARY_ROWS``: FlashInfer's multiplies on the stored weights and its attention over the paged attention
  cache).  A block no kernel serves for the run stays on its fallback row.  On the classes of the library's rows the
  stack needs the library's package and, for the attention, its kernel-binary package;
* ``fe-entries`` -- every cuDNN Frontend row of ``native``, with every block a library serves there on its fallback row:
  no package beyond cuDNN Frontend and PyTorch, so the rows of the entries stay verified wherever the area's GPU test
  runs;
* ``reference`` -- every block in plain PyTorch.

A plan is resolved for a number of tokens per sequence.  One is the decode step.  More is the verify step of speculative
decoding (``Session(tokens=T)``): the linear-attention block is then the fused verify entry's row of that count, and every
other block is the row its own ranges give at ``sequences x T`` rows -- a row that does not hold the count falls to the
next one, as at any other row count.  One block has rows for such a step alone: the out-projection, whose
linear-attention call is the FP8 projection entry's in a verify step, in a split that follows the rows of the step
(``VERIFY_OUT_ROWS``).

``readme_table()`` renders the tables of the decode step as the family README shows them; a host test keeps the two in
agreement.

The checkpoint's drafter has a table of its own, ``DRAFTER_PROVIDERS``: one row per block of its draft pass, in every
stack, at any row count.  Every block of it is a reference chain -- no kernel serves a drafter here yet -- and its
embedding lookup is PyTorch's gather.  A pass calls its multiply at five sites and its norm at two
(``DRAFT_PROJECTION_SITES``, ``DRAFT_NORM_SITES``: the names ``QwenDenseDrafter.sites()`` hands the blocks); the rows
name no form for them, because a reference chain is called in none.
"""

from __future__ import annotations

from ..blocks import GIB, ProviderRow, Regime
from ..census import form_text

ANY = (("any",), (1, None), (1, None))
BOTH = ("fe-cublas", "reference")
#: the stacks whose rows are kernels on the stored formats; a run of one always checks its kernels
CHECKED_STACKS = ("native", "fe-entries")
#: every stack with provider rows (all but the reference), and every stack
PROVIDER_STACKS = ("fe-cublas", *CHECKED_STACKS)
#: the positions of one page of the paged attention cache that the attention core of ``native`` reads.  Where the
#: attention preparation runs on its entry (``PREPARE_FORM``: ``native`` and ``fe-entries`` on compute capability 10.x)
#: the state's attention cache is that pool of pages; everywhere else the attention blocks read one
#: contiguous cache.  The performance mode of a sample judges the two positions on either side of the first page boundary
#: inside its timed steps on every stack, so that its judged positions are the same whichever cache a stack has
ATTENTION_PAGE = 64
EVERY = (*BOTH, *CHECKED_STACKS)

#: the forms the add + RMSNorm + quantise kernel is called in, by call site of the wiring: the residual mode (the first
#: call of a step starts the residual stream from its operand, every other call adds to it in place) and the outputs the
#: next block reads.  A layer's input norm writes the 8-bit operand of the in-projection, and the normed bf16 rows too where
#: the mixer reads them (the linear-attention step does, the attention layer does not); the norm in front of an MLP writes
#: 4-bit codes alone; the final norm writes bf16 rows alone, and the output head's 4-bit operand is quantised from them by a
#: launch of its own (the block ``output_head.quantize``).  Six forms; a model of 48 + 16 layers calls five of them,
#: 1 / 47 / 16 / 64 / 1 times a step
NORM_FORMS = (
    ("linear_attention.first", (("ADD", False), ("STORE_RESIDUAL", True), ("BF16_OUT", True), ("QUANT", "e4m3_static"))),
    ("full_attention.first", (("ADD", False), ("STORE_RESIDUAL", True), ("BF16_OUT", False), ("QUANT", "e4m3_static"))),
    ("linear_attention", (("ADD", True), ("STORE_RESIDUAL", False), ("BF16_OUT", True), ("QUANT", "e4m3_static"))),
    ("full_attention", (("ADD", True), ("STORE_RESIDUAL", False), ("BF16_OUT", False), ("QUANT", "e4m3_static"))),
    ("mlp", (("ADD", True), ("STORE_RESIDUAL", False), ("BF16_OUT", False), ("QUANT", "nvfp4"))),
    ("output_head", (("ADD", True), ("STORE_RESIDUAL", False), ("BF16_OUT", True), ("QUANT", None))),
)

#: the in-projection of a layer, by call site of the wiring: the shape of its merged weight (out features x in features) --
#: ``qkv`` and ``z`` of a linear-attention layer; the query with its gate half, the key and the value projection of an
#: attention layer.  The FP8 projection entry is one weight shape per instance, so a row of it names both; the form of
#: the kernel is the row's and the same at both sites.  A model of 48 + 16 layers calls them 48 and 16 times a step
PROJECTION_SITES = (
    ("linear_attention.qkvz", (("OUT_FEATURES", 16384), ("IN_FEATURES", 5120))),
    ("full_attention.qkv", (("OUT_FEATURES", 14336), ("IN_FEATURES", 5120))),
)

#: the out-projection of a layer, by call site of the wiring: the layer's kind.  Both kinds multiply by a weight of one
#: shape, 5120 x 6144.  In a step of several tokens per sequence the FP8 projection kernel serves the linear-attention
#: layers' call and a library multiply the attention layers': a row of that kernel names the weight's
#: shape at the first site and hands the second on to the next row of its stack (``None`` in place of the fields)
OUT_PROJECTION_SITES = (
    ("linear_attention.out", (("OUT_FEATURES", 5120), ("IN_FEATURES", 6144))),
    ("full_attention.out", None),
)

#: the out-projection of a linear-attention layer in a step of several tokens per sequence, by the rows of the step
#: (sequences x tokens): (GPU classes, first row count, last row count, K slices, weight tile, ring stages, token tile).
#: The split follows the row count on compute capability 10.0 -- two K slices and a ring of 11 stages up to 8 rows, four
#: slices and 7 stages from 9 to 16, and the library multiply from 17 -- and is four slices at every row count from 2 to
#: 32 on the other 10.x classes.  These limits are kept as they are until a row is run beyond them: widening one is
#: kernel work of its own.  A split is a numerics attribute of the entry and fixed per
#: entry instance, so each range is a row of its own, and the rows of two splits are two byte classes: the bytes of a
#: step change where its rows cross from one range into the next, by design.  The token tile follows the row count in
#: the entry (the smallest of 8 / 16 / 32 that holds the rows); the entry takes it as a floor of its
#: form, so a row names the tile of its range and the call log of a step shows it for the row count of each call
VERIFY_OUT_ROWS = (
    (("cc10.0",), 2, 8, 2, 64, 11, 8),
    (("cc10.0",), 9, 16, 4, 64, 7, 16),
    (("cc10.3", "cc10.7"), 2, 8, 4, 64, 7, 8),
    (("cc10.3", "cc10.7"), 9, 16, 4, 64, 7, 16),
    (("cc10.3", "cc10.7"), 17, 32, 4, 64, 7, 32),
)

#: the form the attention preparation kernel is called in, stated whole: three position rows in the sections of the
#: published model's rotary width (a text position is one number, so the three rows the state's table hands over are equal),
#: pages of 64 rows in both pools, 64-bit positions and slots, an ordinary launch -- no wait for its predecessor and no
#: early release of its successor.  The entry has no default for the first five; the last two are its own defaults, named
PREPARE_FORM = (
    ("POSITIONS", "3d"),
    ("SECTION", (11, 11, 10)),
    ("PAGE_SIZE", ATTENTION_PAGE),
    ("POSITION_DTYPE", "int64"),
    ("SLOT_DTYPE", "int64"),
    ("PDL", False),
    ("PDL_RELEASE", False),
)

#: the form the 4-bit quantise in front of the output head is called in, stated whole: the row width (the model's
#: hidden width), launched as a dependent of its predecessor, releasing its successor directly behind its own wait.  The
#: three fields are the entry's whole config; its kernel's symbol carries them
HEAD_QUANTIZE_FORM = (("WIDTH", 5120), ("PDL", True), ("PDL_RELEASE", True))

#: the fused gate / up kernel of the MLP by row count: a step runs the narrowest of its four token tiles that holds its
#: rows, so the token tile is a field of the form and the block has one row per tile: (first row count, last
#: row count, token tile).  A launch record does not show the tile -- the grid and the block are the same at all four,
#: the shared memory at three -- so it is held on the objects and in the kernel's symbol
GATE_UP_TILES = ((1, 8, 8), (9, 16, 16), (17, 32, 32), (33, 64, 64))

#: the fused verify step of a linear-attention layer by tokens per sequence: the step shape is a compile-time fact of its
#: kernel and a field of the form (``TOKENS``), so the block has one row per count: (tokens per sequence, last sequence
#: count).  The rows hold up to 16 sequences, and up to 4 from six tokens per sequence -- limits kept as they are until a
#: wider step shape is run; eight tokens is what every form of the entry holds on the classes it serves.  A step shape
#: beyond these has no row: the plan declines it by name
VERIFY_STEPS = tuple((tokens, 16 if tokens < 6 else 4) for tokens in range(2, 9))


#: the blocks of ``native`` a library serves: block -> (provider key, what the row serves, the layer widths the operand
#: layout of the call was run with -- the published model's; another width keeps the block on its fallback row).  The
#: rows are for every 10.x class, at every row count.  Nothing here is gated on batch invariance: the
#: library states none for these calls, the kernel of the 4-bit multiply follows the row count, and the kernel of the
#: attention the sequences, the tokens per sequence and the run's maximum length
LIBRARY_ROWS = {
    "mlp.down": ("flashinfer.mm_fp4", "any row count, hidden 5120, intermediate 17408", (("hidden", 5120), ("mlp", 17408))),
    "output_head": ("flashinfer.mm_fp4_head", "any row count, hidden 5120", (("hidden", 5120),)),
    "projection.fp8.out": (
        "flashinfer.bmm_fp8",
        "any row count, weight 5120 x 6144",
        (("hidden", 5120), ("la_value_heads", 48), ("la_head_dim", 128), ("attn_heads", 24), ("attn_head_dim", 256)),
    ),
    "attention.core": (
        "flashinfer.decode_attention",
        "rows 1-1,024, 24 heads of width 256 on 4 key / value heads, a cache in pages of 64",
        (("attn_heads", 24), ("attn_kv_heads", 4), ("attn_head_dim", 256)),
    ),
}
#: the form of a row of the library's 4-bit multiply: its CuTe DSL backend, called outside any tuning context -- the
#: kernel class and the launch shape are then the library's untuned pick for the call, which a run prints -- and
#: of its 8-bit multiply: the backend the library chooses itself (its tuner's pick where one has run: never a row)
MM_FP4_FORM = (("BACKEND", "cute-dsl"), ("TUNED", False))
BMM_FP8_FORM = (("BACKEND", "auto"),)
#: the form of the row of the library's paged decode attention: on pools whose pages hold the
#: heads first.  The call names the library's own backend for these GPU classes, whose kernels are precompiled binaries:
#: a constant of the adapter, no field of the row.  The row claims this call, never a launch shape: which kernel of the
#: set a call launches is the library's function of the run's maximum length, the sequences, the tokens per sequence and
#: the GPU, and a run prints the kernel that was launched
DECODE_ATTENTION_FORM = (("KV_LAYOUT", "HND"),)


def _library_row(block: str, form, band: str = "bf16_rounding", last=None) -> ProviderRow:
    """The row of ``native`` that serves ``block`` with a library call on the stored weight (:data:`LIBRARY_ROWS`).  Its launch
    order: the library launches its own kernels as dependents of the launch before them (the adapters pass nothing that changes
    it), so behind a launch that releases early -- the fused step or the gate in front of the out-projection, the gate / up
    kernel in front of the down multiply -- the kernel starts early and must wait before it reads that launch's codes.  The
    kernels of the validated version wait in front of their first load.  The 4-bit multiply reads its one scalar in front of
    that wait (both kernel classes of the validated version on compute capability 10.0): the fp32 product of the two global
    scales, which the adapter forms on the host once per pair of values -- a constant no launch writes.  A launch behind one
    that carries the attribute must wait before it reads the product (the norm: its row).  In front of the head stands the
    quantise of its operand, a block of its own: where its entry serves, a launch that releases early, and the head's kernel
    waits before it reads the codes and the block scales; on its reference chain PyTorch's launches, none of which releases
    early.
    ``band``: the row's band against the reference; ``last``: the last row count the row claims (left out: any)."""
    key, serves, widths = LIBRARY_ROWS[block]
    # no test of this repository's own lanes runs a row that needs the library's package: printed as unverified in CI
    return ProviderRow(("cc10.x",), (1, last), (1, None), key, "native", serves=serves, band=band, ci=False, form=form, rows=(1, last), widths=widths)


def _verify_out_rows(stack):
    """The rows of the FP8 projection entry for the out-projection of a step of several tokens per sequence in ``stack``, one
    per range of :data:`VERIFY_OUT_ROWS`: the linear-attention layers' call on the entry, the attention layers' call
    handed on to the next row."""
    return [
        ProviderRow(
            classes,
            (1, None),
            (2, None),  # a step of several tokens per sequence: at one token a library multiply serves this block
            "fe.fp8_projection_decode",
            stack,
            # every row of a class names the range of all of them, so a decline says what the entry serves there, not one range of it
            serves=f"rows {first}-{last} of 2-{max(end for group, _, end, *_ in VERIFY_OUT_ROWS if group == classes)} of a step of several tokens per sequence, "
            "the linear-attention layers' weight 5120 x 6144",
            band="bf16_rounding",
            # the class this geometry of the entry has run on (its determinism row: verified on compute capability 10.0,
            # admitted and unverified on the other 10.x classes)
            batch_invariant=tuple(c for c in classes if c == "cc10.0"),
            # the kernel of this call, stated whole and never left to the entry's pick: the K slices and the ring
            # depth of the row range, the weight tile of 64 rows, the token tile of the range, the weight stream issued
            # ahead of the wait; launched as a dependent of its predecessor -- the fused verify step, which releases its
            # successor at entry: this kernel starts under it, streams its weight and waits for it right before it reads the
            # 8-bit codes -- and releasing its successor after its last weight issue.  The two performance fields of the
            # entry are stated like the others: 32 tensor-memory columns, released after the values read from them are
            # consumed.  The launch before it must not write the
            # weight or the scale: both are constants of the layer.  The launch after it reads its rows: the norm in
            # front of the MLP carries the attribute and waits first (its row says so)
            form=(
                ("SPLIT_K", split),
                ("MMA_M", tile),
                ("STAGES", stages),
                ("TOKEN_TILE", token_tile),
                ("TMEM_COLS", 32),
                ("PREWAIT_WEIGHTS", True),
                ("HOLD_TMEM", True),
                ("PDL", True),
                ("PDL_RELEASE", True),
            ),
            rows=(first, last),
            sites=OUT_PROJECTION_SITES,
            # the layer widths the weight's shape follows from (the published model's).  A checkpoint of the family with
            # another width keeps the block on the multiply over a copy below
            widths=(("hidden", 5120), ("la_value_heads", 48), ("la_head_dim", 128)),
        )
        for classes, first, last, split, tile, stages, token_tile in VERIFY_OUT_ROWS
    ]


def _verify_rows(stack, form, batch_invariant=()):
    """The rows of the fused verify entry for ``stack``, one per tokens count of :data:`VERIFY_STEPS`, in the launch form
    ``form`` (the two launch knobs; the row adds its ``TOKENS``)."""
    return [
        ProviderRow(
            ("cc10.x", "cc12.x"),
            (1, last),
            (tokens, tokens),
            "fe.gdn_verify_fused",
            stack,
            serves=f"{tokens} tokens per sequence of 2-8, sequences 1-{last}",
            band="gdn_step",
            # no class is listed: the entry states replay-identical bytes and no batch invariance between one sequence
            # and several (its adapter carries the statement, and a run prints it)
            batch_invariant=batch_invariant,
            form=(*form, ("TOKENS", tokens)),
        )
        for tokens, last in VERIFY_STEPS
    ]


PROVIDERS = {
    "embedding": [ProviderRow(*ANY, "torch.index_select", EVERY, serves="any")],
    "add_norm_quantize": [
        ProviderRow(
            ("cc10.x",),
            (1, 1024),
            (1, None),
            "fe.add_norm_quant_decode",
            CHECKED_STACKS,
            serves="rows 1-1,024, hidden 5120",
            band="add_norm_quant",
            batch_invariant=("cc10.0",),
            # the launch form at every call site: the kernel waits for its predecessor after it has loaded the norm
            # weight and the scale and before it reads the rows, and releases its successor directly behind that wait.  The
            # launch after a layer's input norm is the in-projection's: where that row is the FP8 projection entry it carries
            # the dependent-launch attribute, starts on this release and waits for this kernel before it reads the 8-bit
            # operand; every other launch after a norm in this stack carries none, and the release changes nothing there.
            # The launch before it must not write the norm weight or the scale, which are constants of the layer
            form=(("PDL", True), ("PDL_RELEASE", True)),
            rows=(1, 1024),
            sites=NORM_FORMS,
            # the row width the entry serves (its page: 5120, the published model's).  A checkpoint of the family with another
            # hidden width keeps the block on the reference chain below
            widths=(("hidden", 5120),),
        ),
        ProviderRow(*ANY, "torch_ref.add_norm_quantize", EVERY, serves="any", fallback="reference chain"),
    ],
    "projection.fp8.in": [
        ProviderRow(
            ("cc10.0",),  # the row's limit, kept as it is until the row is run on another class; elsewhere the block is on its multiply over a copy
            (1, 32),
            (1, None),
            "fe.fp8_projection_decode",
            CHECKED_STACKS,
            serves="rows 1-32, weights 16384 x 5120 and 14336 x 5120",
            band="bf16_rounding",
            batch_invariant=("cc10.0",),
            # the kernel, stated whole and never left to the entry's pick: one K slice per weight tile of 128 rows, a
            # ring of 8 stages, the weight stream issued ahead of the wait; launched as a dependent of its predecessor (it
            # waits for the norm's kernel right before it reads the 8-bit operand) and releasing its successor after its last
            # weight issue.  The two performance fields of the entry are stated like the others: 32 tensor-memory
            # columns, released after the values read from them are consumed.  The launch before it
            # must not write the weight or the scale: both are constants of the layer.  The launch after it reads its rows:
            # the fused linear-attention step of this stack carries the attribute and waits first (its row says so), and
            # the chain behind it in an attention layer carries none
            form=(
                ("SPLIT_K", 1),
                ("MMA_M", 128),
                ("STAGES", 8),
                ("TMEM_COLS", 32),
                ("PREWAIT_WEIGHTS", True),
                ("HOLD_TMEM", True),
                ("PDL", True),
                ("PDL_RELEASE", True),
            ),
            rows=(1, 32),
            sites=PROJECTION_SITES,
            # the layer widths the two weight shapes follow from (the published model's).  A checkpoint of the family with
            # another width keeps the block on the multiply over a copy below
            widths=(
                ("hidden", 5120),
                ("la_key_heads", 16),
                ("la_value_heads", 48),
                ("la_head_dim", 128),
                ("attn_heads", 24),
                ("attn_kv_heads", 4),
                ("attn_head_dim", 256),
            ),
        ),
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "linear_attention.step": [
        ProviderRow(
            ("cc10.x", "cc12.x"),
            (1, 65535),
            (1, 1),
            "fe.gdn_decode_fused",
            "fe-cublas",
            serves="rows 1-65,535",
            measured=("cc 10.0, 1 row",),
            band="gdn_step",
            batch_invariant=("cc10.0", "cc10.3", "cc12.0"),
            # the entry's own pick per rows, launched without a wait on its predecessor and without an early release of its
            # successor.  Both are what the entry does when nothing is passed; the row says so, and the kernel check holds it
            form=(("PDL", False), ("PDL_RELEASE", False)),
        ),
        ProviderRow(
            ("cc10.x", "cc12.x"),
            (1, 65535),
            (1, 1),
            "fe.gdn_decode_fused",
            CHECKED_STACKS,
            serves="rows 1-65,535",
            measured=("cc 10.0, 1 row",),
            band="gdn_step",
            batch_invariant=("cc10.0",),  # the class this form has run on; the row above lists what its form has
            # the form of these stacks: the entry's own pick per rows, launched as a dependent of its predecessor (it waits
            # for it only before it reads the in-projection's rows) and releasing its successor at entry.  The launch before
            # it must not write this layer's state: in this stack that launch is the in-projection's
            form=(("PDL", True), ("PDL_RELEASE", True)),
        ),
        # a step of several tokens per sequence (the verify step of speculative decoding): the fused verify entry, in the
        # launch form of the stack's decode row above -- launched without a wait and without an early release in the
        # reference path; as a dependent of the in-projection that releases its successor at entry in the form of the
        # other two stacks (the launch before it must not write this layer's state, its history or the slot table: in
        # these stacks it is the in-projection's).  The reference chain below serves both kinds of step
        *_verify_rows("fe-cublas", (("PDL", False), ("PDL_RELEASE", False))),
        *_verify_rows(CHECKED_STACKS, (("PDL", True), ("PDL_RELEASE", True))),
        ProviderRow(*ANY, "torch_ref.gdn_decode_step", "reference", serves="any", fallback="reference"),
    ],
    "projection.fp8.out": [
        # a step of several tokens per sequence (the verify step of speculative decoding): the linear-attention layers'
        # call on the FP8 projection entry, in the split of the step's rows; the attention layers' call is the next
        # row's.  A step of one token per sequence, a step of more rows than the ranges hold, another class and another
        # width are the multiply over a copy below, at both call sites
        *_verify_out_rows(CHECKED_STACKS),
        # the library's 8-bit multiply on the stored codes: both call sites of a step of one token
        # per sequence, and in a step of several the attention layers' site and every call beyond the rows above.
        # Launch order: a dependent of the fused step or of the gate, which release early; it waits before its first load
        _library_row("projection.fp8.out", BMM_FP8_FORM),
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 8-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "mlp.gate_up": [
        *(
            ProviderRow(
                ("cc10.x",),
                (1, last),
                (1, None),
                "fe.swiglu_fp4_decode",
                CHECKED_STACKS,
                serves=f"rows {first}-{last} of 1-64, hidden 5120, intermediate 17408",
                band="swiglu_fp4",
                batch_invariant=("cc10.0",),
                # the kernel, stated whole and never left to the entry's default: the token tile of the row count, two
                # weight stages issued ahead of the wait, the weight stream loaded with the evict-first L2 policy, at most 148
                # thread blocks, launched as a dependent of its predecessor.  It waits for that launch -- the norm in front of
                # the MLP, which releases at its own wait -- before it reads the 4-bit operand, its block scales and the two
                # scalars; only weight tiles are read ahead of the wait.  The launch before it must not write the weight or
                # its block scales: both are constants of the layer.  It signals the launch after it before its last output
                # tile is written: a launch behind it that carries the attribute must wait before it reads the codes; the
                # multiply behind it carries it where it is the library's (``native``: its row), a multiply over a copy none
                form=(("TOKEN_TILE", tile), ("PREWAIT_STAGES", 2), ("L2_EVICT_FIRST", True), ("MAX_CLUSTERS", 148), ("PDL", True)),
                rows=(first, last),
                # the widths the entry serves (its page: hidden 5120, intermediate 17408, the published model's).  A checkpoint of
                # the family with another width keeps the block on the multiply over a copy below
                widths=(("hidden", 5120), ("mlp", 17408)),
            )
            for first, last, tile in GATE_UP_TILES
        ),
        ProviderRow(
            *ANY,
            "cublas.bf16_dequantized+torch_ref.silu_mul_quantize",
            PROVIDER_STACKS,
            serves="any",
            fallback="bf16 copy of 4-bit weights",
            band="bf16_rounding",
        ),
        ProviderRow(*ANY, "torch_ref.linear+torch_ref.silu_mul_quantize", "reference", serves="any", fallback="reference"),
    ],
    "mlp.down": [
        # the library's 4-bit multiply on the stored codes.  The operand is the fused gate / up kernel's output, in the layout
        # this kernel reads: nothing is launched between the two.  Launch order: a dependent of that kernel, which releases
        # early; it waits before its first load
        _library_row("mlp.down", MM_FP4_FORM),
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "attention.prepare": [
        ProviderRow(
            ("cc10.x",),  # the classes of the row; the entry has run on a device of compute capability 10.0
            (1, 1024),
            (1, None),
            "fe.attention_prepare_decode",
            CHECKED_STACKS,
            serves="rows 1-1,024, 24 heads of width 256 on 4 key / value heads, rotary width 64 in the sections 11 + 11 + 10",
            band="attention_prepare",
            batch_invariant=("cc10.0",),
            # the form of the row (``PREPARE_FORM``).  The kernel is an ordinary launch: it starts when the in-projection's kernel
            # has ended, reads that kernel's rows through four views and writes the queries, the gates and the two halves of
            # the layer's pool; nothing is launched between the two.  The launch after it is the attention core's, which reads
            # the pool and the queries: the library's kernel in ``native`` (its row), a reference chain in ``fe-entries``.
            # The entry writes a pool of pages, so a session whose plan has this row builds its attention cache as one
            # (the adapter says so: ``cache_page``), with the cos / sin table and the position and slot rows the kernel reads
            form=PREPARE_FORM,
            rows=(1, 1024),
            # the widths the entry serves (its page: the published model's attention widths) and the rotary sections the row's
            # form names.  A checkpoint of the family with another one keeps the block on the reference chain below, on a
            # plain cache
            widths=(("attn_heads", 24), ("attn_kv_heads", 4), ("attn_head_dim", 256), ("rotary_dim", 64), ("rotary_section", (11, 11, 10))),
        ),
        ProviderRow(*ANY, "torch_ref.attention_prepare", EVERY, serves="any", fallback="reference chain"),
    ],
    # the chain reads the layer's cache through its accessors.  On the two reference stacks that is one run of positions per
    # sequence; on ``native`` and ``fe-entries`` it is the pool of pages the preparation's kernel writes wherever the row above serves,
    # and the reason of their row holds for either layout
    "attention.core": [
        ProviderRow(*ANY, "torch_ref.attention", BOTH, serves="any", fallback="reference chain over a plain per-layer cache"),
        # the library's paged decode attention on the pool the preparation's kernel wrote, through the page table and the
        # lengths the state holds for the run's maximum length; the rows of the preparation's row.  The library documents no
        # numerics contract for this kernel, so the band is the bound of a stated model against the reference chain, and
        # nothing is gated on batch invariance: the kernel, and with it the bytes of a sequence, follow the batch.
        # Launch order: with the library's default the launch is a dependent of the launch before it, the preparation's,
        # which is an ordinary launch and releases nothing early; the kernel waits before it reads the queries and the
        # pool.  A wait in front of its read of the page table and the lengths is not established, so those two are
        # constants of the state that no launch writes.  The launch after it, the gate's, is an ordinary launch
        _library_row("attention.core", DECODE_ATTENTION_FORM, band="attention_bf16_weights", last=1024),
        ProviderRow(*ANY, "torch_ref.attention", CHECKED_STACKS, serves="any", fallback="reference chain over the attention cache"),
    ],
    "attention.gate_quantize": [
        ProviderRow(
            ("cc10.x",),
            (1, 160),
            (1, None),
            "fe.attention_gate_quant_decode",
            CHECKED_STACKS,
            serves="rows 1-160",
            band="gate_fp8",
            batch_invariant=("cc10.0",),
            # the form of the row: 256 threads per thread block, an ordinary launch that releases its successor at entry.  The
            # launch after it carries the attribute where it is the library's out-projection (``native``: its row), else none
            form=(("THREADS", 256), ("PDL_RELEASE", True)),
            rows=(1, 160),
        ),
        ProviderRow(*ANY, "torch_ref.gate_quantize", EVERY, serves="any", fallback="reference chain"),
    ],
    # the 4-bit operand of the output head, from the final norm's bf16 rows: formed by a launch of its own
    "output_head.quantize": [
        ProviderRow(
            ("cc10.x",),  # the classes of the row; the entry has run on a device of compute capability 10.0
            (1, 1024),
            (1, None),
            "fe.row_quantize_fp4_decode",
            CHECKED_STACKS,
            serves="rows 1-1,024, hidden 5120",
            band="row_quantize_fp4",
            batch_invariant=("cc10.0",),
            # the form of the row (``HEAD_QUANTIZE_FORM``).  Launch order: the launch in front is the final norm's, which waits and
            # releases directly behind its own wait, so this kernel can start while the norm still runs: it loads its one
            # scale, waits, and only then reads the rows.  The launch before it must not write the scale: the adapter forms
            # it on the host once per value, a constant no launch writes.  The kernel releases directly behind its own wait,
            # so the launch behind it can start before the codes are written: the library's head multiply in ``native``, a
            # dependent launch that waits in front of its first load of the codes and the block scales (its row); in
            # ``fe-entries`` PyTorch's launches of the multiply over a copy, which carry no attribute and start when this
            # kernel has ended
            form=HEAD_QUANTIZE_FORM,
            rows=(1, 1024),
            # the row width the entry serves (its page: 5120, the published model's).  A checkpoint of the family with another
            # hidden width keeps the block on the reference chain below
            widths=(("hidden", 5120),),
        ),
        ProviderRow(*ANY, "torch_ref.quantize", EVERY, serves="any", fallback="reference chain"),
    ],
    "output_head": [
        # the same multiply, on the vocabulary's weight.  Its operand is what the quantise block in front of it hands over, in
        # the layout that block wrote.  Launch order: a dependent of that block's last launch; it waits before its first load
        # of the codes, the block scales and the weight.  In front of that wait it reads its one scalar, which the launch
        # before it must not write: the adapter forms it on the host once per pair of values, a constant no launch writes
        _library_row("output_head", MM_FP4_FORM),
        ProviderRow(*ANY, "cublas.bf16_dequantized", PROVIDER_STACKS, serves="any", fallback="bf16 copy of 4-bit weights", band="bf16_rounding"),
        ProviderRow(*ANY, "torch_ref.linear", "reference", serves="any", fallback="reference"),
    ],
    "token_choice": [ProviderRow(*ANY, "torch.argmax", EVERY, serves="any")],
}

#: the call sites of the drafter's multiply, by the loader's name of the weight (``drafter.MULTIPLIES``), each with the shape
#: of that weight at the published widths (out features x in features).  A kernel that is one weight shape per instance
#: takes them as the sites of its row; the reference chain of today is one function at all five
DRAFT_PROJECTION_SITES = (
    ("fc", (("OUT_FEATURES", 5120), ("IN_FEATURES", 10240))),
    ("qkv", (("OUT_FEATURES", 14336), ("IN_FEATURES", 5120))),
    ("o", (("OUT_FEATURES", 5120), ("IN_FEATURES", 6144))),
    ("gate_up", (("OUT_FEATURES", 34816), ("IN_FEATURES", 5120))),
    ("down", (("OUT_FEATURES", 5120), ("IN_FEATURES", 17408))),
)
#: the call sites of the drafter's norm + quantise, by what the codes feed: the layer's in-projection -- the rows it norms
#: ARE the residual stream of the pass, nothing is added -- and the MLP, in front of which the attention branch is added
#: to the stream.  (The final norm of a pass adds too and writes bf16 rows and no codes: a block of its own.)
DRAFT_NORM_SITES = (
    ("input", (("ADD", False),)),
    ("mlp", (("ADD", True),)),
)

#: the blocks of one pass of the checkpoint's drafter.  Every row serves every stack, every GPU class and any row count:
#: the first pass of a step behind a verify step runs at sequences x tokens rows and every other pass at one row per
#: sequence, on the same rows of this table
DRAFTER_PROVIDERS = {
    "draft.embedding": [ProviderRow(*ANY, "torch.index_select", EVERY, serves="any")],
    "draft.pre_fc_norm": [ProviderRow(*ANY, "torch_ref.pre_fc_norm_quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.projection": [ProviderRow(*ANY, "torch_ref.linear", EVERY, serves="any", fallback="reference chain")],
    "draft.norm_quantize": [ProviderRow(*ANY, "torch_ref.add_norm_quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.attention.prepare": [ProviderRow(*ANY, "torch_ref.attention_prepare", EVERY, serves="any", fallback="reference chain")],
    "draft.attention.core": [ProviderRow(*ANY, "torch_ref.attention", EVERY, serves="any", fallback="reference chain over a plain cache of the drafter's own")],
    "draft.gate_quantize": [ProviderRow(*ANY, "torch_ref.gate_row_quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.silu_mul": [ProviderRow(*ANY, "torch_ref.silu_mul", EVERY, serves="any", fallback="reference chain")],
    "draft.activation_quantize": [ProviderRow(*ANY, "torch_ref.quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.final_norm": [ProviderRow(*ANY, "torch_ref.add_norm_quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.head.quantize": [ProviderRow(*ANY, "torch_ref.quantize", EVERY, serves="any", fallback="reference chain")],
    "draft.head": [ProviderRow(*ANY, "torch_ref.linear", EVERY, serves="any", fallback="reference chain")],
    "draft.token_choice": [ProviderRow(*ANY, "torch.argmax+lookup", EVERY, serves="any", fallback="reference chain")],
}

REGIMES = [
    Regime(
        ("any",),
        (1, None),
        "launch-bound",
        "every block of these stacks is launched from Python, one launch or more per block; no byte-bound form is declared, "
        "and the sequence count at which one would win is not measured on any class",
    ),
]

#: how each block is named in text
BLOCK_TITLES = {
    "embedding": "token embedding lookup",
    "add_norm_quantize": "residual add + RMSNorm + quantise",
    "projection.fp8.in": "8-bit in-projection (linear attention QKVZ / attention QKV)",
    "linear_attention.step": "linear-attention step",
    "projection.fp8.out": "8-bit out-projection",
    "mlp.gate_up": "MLP gate / up + SiLU x up + quantise",
    "mlp.down": "MLP down",
    "attention.prepare": "attention preparation",
    "attention.core": "attention core",
    "attention.gate_quantize": "gate + quantise",
    "output_head.quantize": "4-bit quantise of the output head's operand",
    "output_head": "output head",
    "token_choice": "token choice",
}

#: how each block of the drafter's pass is named in text (``BLOCK_TITLES`` stays the model's)
DRAFTER_BLOCK_TITLES = {
    "draft.embedding": "token embedding lookup",
    "draft.pre_fc_norm": "RMSNorm of the embedding rows and of the hidden rows + one quantisation per row over both",
    "draft.projection": "8-bit multiply, a scale per weight row and per row of the step",
    "draft.norm_quantize": "(residual add +) RMSNorm + quantise per row",
    "draft.attention.prepare": "attention preparation",
    "draft.attention.core": "attention core",
    "draft.gate_quantize": "gate + quantise per row",
    "draft.silu_mul": "SiLU x up",
    "draft.activation_quantize": "quantise per row",
    "draft.final_norm": "residual add + RMSNorm",
    "draft.head.quantize": "4-bit quantisation of the head's operand",
    "draft.head": "output head over the listed rows",
    "draft.token_choice": "token choice over the listed rows",
}

#: device memory the first launch of the fused linear-attention step reserves outside PyTorch's allocator, per GPU class: an
#: allowance for the memory sentence of the plan, which is printed before anything is launched.  The run then measures the
#: launch and prints what it took (this sample on one compute capability 12.0 GPU: 3,124 MiB at the first form, none after)
FIRST_LAUNCH_RESERVE = {"cc12.x": 3 * GIB}


#: the logits band of the token check, declared here and not taken from a run: the logits of the provider stack and of the
#: reference stack, each run end to end on its own state, may differ by this many bf16 units of the run's largest reference
#: logit.  It is declared for the reduced seeded model (measured by this sample: 1.0 unit with two layers, 1.2 with four, 1.4
#: with eight); a real checkpoint declares its own
LOGITS_BAND_UNITS = 4.0
#: the logits band for a checkpoint read from local disk (the published depth, trained weights): none.  The token line of
#: such a run is REPORTED and gated on finite logits only (``tie_out.token_check`` with ``band_units=None``).
#: The model quantises its MLP inputs to 4 bits per element and its projection inputs to 8, so a difference of one bf16
#: rounding in a multiply becomes a different code in the next quantiser, 64 layers deep.  Measured by this sample on the
#: published 27B-class checkpoint: two REFERENCE stacks that differ only in their accumulation precision (fp64 against IEEE
#: fp32) were up to 52 units (13.0 logits) apart over 80 steps and chose another greedy token at 13 of them; the provider
#: stack was up to 54 units (13.5 logits) from the fp64 reference over the same steps.  In three 16-step windows of the
#: kind a run checks, the provider stack was 8.7 to 10.3 logits from the reference and a second reference 7.5 to 11.4.
#: No band separates a correct stack from a failed one here.  In bf16 units of the largest logit a band is 8, 16 or 32
#: logits wide by that logit's binade, and the distance of two correct stacks does not follow the binade: 128 units --
#: beyond which a stack that returns zeros passes -- are 8 logits where a window's largest logit is below 16, less than
#: two correct stacks were apart.  And a greedy token is decidable only where its margin exceeds twice the band, which no
#: step of those runs reached.  What verifies a provider on a checkpoint is the per-block check, teacher-forced; the run
#: prints the distance, the count of equal greedy tokens and the same for a second reference, to be read side by side
CHECKPOINT_LOGITS_BAND_UNITS = None


def readme_table(calls: dict, stack: str = "fe-cublas", cc=(10, 0), table=None, titles=None, per: str = "step (64 layers)") -> str:
    """The block table of the family README (markdown) for ``stack`` on compute capability ``cc``, with ``calls`` per step of
    the full model: per block the first row of the stack that serves the class in a step of one token per sequence -- the
    decode step -- with the form its entry is called in.  ``table`` and ``titles`` name another provider table of the
    family and its block titles (the drafter's: ``DRAFTER_PROVIDERS``, ``DRAFTER_BLOCK_TITLES``), and ``per`` what its
    calls are counted over (``"pass"``)."""
    from ..providers import ENTRY_NAMES

    table, titles = PROVIDERS if table is None else table, BLOCK_TITLES if titles is None else titles
    lines = [f"| block | calls per {per} | provider key | entry [form] |", "|---|---|---|---|"]
    for block, rows in table.items():
        row = next(r for r in rows if stack in r.stacks and r.serves_class(cc) and r.tokens[0] <= 1)
        forms = f" + one of {len(row.served)} forms, by call site" if row.sites else ""
        entry = ENTRY_NAMES[row.key] + form_text(row.form) + forms + (f", fallback: {row.fallback}" if row.fallback else "")
        lines.append(f"| {titles[block]} | {calls[block]} | `{row.key}` | {entry} |")
    return "\n".join(lines)


def readme_draft_sites() -> str:
    """The call-site table of the drafter in the family README (markdown): the five sites of its multiply with the shape
    of the weight there at the published widths, and the two of its norm + quantise with whether the call adds the
    branch to the residual stream."""
    lines = ["| block | call site | what differs there |", "|---|---|---|"]
    for site, fields in DRAFT_PROJECTION_SITES:
        shape = dict(fields)
        lines.append(f"| {DRAFTER_BLOCK_TITLES['draft.projection']} | `{site}` | weight {shape['OUT_FEATURES']} x {shape['IN_FEATURES']} |")
    for site, fields in DRAFT_NORM_SITES:
        added = "the branch is added to the residual stream first" if dict(fields)["ADD"] else "no add: its rows are the residual stream"
        lines.append(f"| {DRAFTER_BLOCK_TITLES['draft.norm_quantize']} | `{site}` | {added} |")
    return "\n".join(lines)


def readme_sites(model, stack: str = "native", cc=(10, 0)) -> str:
    """The call-site table of the family README (markdown): for every row of ``stack`` on ``cc`` that names call sites in a
    step of one token per sequence, the form of each site and how many calls of a step of ``model`` are made there."""
    from collections import Counter

    calls = Counter(zip((block for block, _ in model.wiring()), model.sites()))
    lines = ["| block | call site | calls per step (64 layers) | form of the entry there |", "|---|---|---|---|"]
    for block, rows in PROVIDERS.items():
        row = next(r for r in rows if stack in r.stacks and r.serves_class(cc) and r.tokens[0] <= 1)
        for site, fields in row.sites:
            if fields is None:  # a call site the row hands on to the next row: no form of this row there
                continue
            lines.append(f"| {BLOCK_TITLES[block]} | `{site}` | {calls[(block, site)]} | {form_text(fields).strip()} |")
    return "\n".join(lines)


def readme_regimes() -> str:
    lines = ["| GPU classes | sequences | regime | note |", "|---|---|---|---|"]
    for r in REGIMES:
        upper = "and more" if r.sequences[1] is None else f"to {r.sequences[1]}"
        lines.append(f"| {', '.join(r.classes)} | {r.sequences[0]} {upper} | {r.name} | {r.note} |")
    return "\n".join(lines)
