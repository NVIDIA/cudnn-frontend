# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The drafter of a dense Qwen 3.5-architecture checkpoint: which tensors it is, what a load makes of them, the rows of the
model's output head it proposes tokens from, and its draft pass on reference blocks.

A checkpoint of this family can carry a drafter beside the model: a small network that reads the model's last hidden row
and the embedding of the last token and proposes the next tokens, which the model then verifies in one step (speculative
decoding).  This module is its loader and its draft pass; no session runs a pass yet.  The loader answers three
questions, each from the checkpoint itself:

* **Which tensors.**  The drafter's own tensors sit under one prefix of the checkpoint, found in its shard headers: two
  norms in front of a ``[hidden, 2 x hidden]`` multiply, one attention layer with its MLP at the model's widths, a final
  norm -- all stored in bf16.  It has no embedding table and no output head of its own: it reads the model's.  The whole
  list with what a load makes of each tensor is the manifest (``QwenDenseDrafter.manifest``); a checkpoint whose drafter
  is another network is declined by the first tensor that does not fit, before anything is read.
* **8-bit weights at load.**  The weights of the drafter's five multiplies are quantised when they are loaded, each row
  with its own scale (``weights.quantize_fp8_per_channel``); the parts that feed one multiply are concatenated first:
  query (with its gate half), key and value; gate and up.
* **The draft vocabulary.**  A drafter proposes from a subset of the vocabulary: a JSON list of token ids (``DraftVocab``).
  The ids are taken once each, in ascending order, and only those that are ids of the model; the drafter's head is then
  the rows of the model's 4-bit output head at those ids, byte for byte as the checkpoint stores them, with the head's own
  global scale.  The family README has a recipe that makes such a list from the files ``--dump-tokens`` writes.

``QwenDenseDrafter.seeded(model)`` is the drafter of the reduced seeded model: the same tensor list drawn from a seed, and a
draft vocabulary drawn from that seed instead of a file.

**The draft pass.**  One pass is 18 block calls (``wiring()``; the provider table ``configs.DRAFTER_PROVIDERS``): the body,
15 calls that end in the final norm (``body``), and the head, 3 calls (``propose``).  ``rows`` rows of a step, sequence
by sequence:

========================= ================================================== ===========================================
block (call site)         operands                                           result
========================= ================================================== ===========================================
draft.embedding           ids ``[rows]`` int32, the model's table            ``[rows, hidden]`` bf16
draft.pre_fc_norm         embedding rows, hidden rows, two norm weights,     8-bit ``[rows, 2 hidden]``, the embedding
                          ``Quant``                                          half first, one scale per row over both
draft.projection (fc)     8-bit operand, weight copy, site                   ``[rows, hidden]`` bf16: the residual stream
draft.norm_quantize       branch, residual, norm weight, ``Quant``, site     (8-bit operand, normed bf16, new residual)
 (input, mlp)
draft.projection (qkv)    8-bit operand, weight copy, site                   ``[rows, 2 x 6144 + 2 x 1024]`` bf16
draft.attention.prepare   QKV row, cache, position, params                   (q, gate); key and value inserted
draft.attention.core      q, cache, position                                 ``[rows, 6144]`` bf16
draft.gate_quantize       attention output, gate, ``Quant``                  8-bit ``[rows, 6144]``
draft.projection (o)      8-bit operand, weight copy, site                   ``[rows, hidden]`` bf16
draft.projection (gate_up) 8-bit operand, weight copy, site                  ``[rows, 2 x 17408]`` bf16: gate, then up
draft.silu_mul            gate / up rows                                     ``[rows, 17408]`` bf16
draft.activation_quantize those rows, ``Quant``                              8-bit ``[rows, 17408]``
draft.projection (down)   8-bit operand, weight copy, site                   ``[rows, hidden]`` bf16
draft.final_norm          branch, residual, norm weight, no ``Quant``        the output rows ``[rows, hidden]`` bf16
draft.head.quantize       one row per sequence, the head's ``Quant``         4-bit ``[sequences, hidden]``
draft.head                4-bit operand, copy of the listed head rows        logits ``[sequences, listed ids]`` bf16
draft.token_choice        logits, the token id of every listed row           ids ``[sequences]`` int32
========================= ================================================== ===========================================

Every 8-bit operand of a pass is quantised per row of the step, at a scale formed from the row
(``weights.quantize_fp8_per_row``, ``providers.ROW_QUANT``), and every multiply applies the scale of the operand's row
and the scale of the weight's row.  The first multiply reads the embedding of a token and a hidden row side by side;
its result is the residual stream of the pass, which the layer's input norm reads without an add.  The attention layer
has a cache of the drafter's own (``state``): a pass writes its keys and values at the positions of its rows and is told
where they stand; the cache has no position.  The model's caches and state are never written by a pass.

``draft`` is the passes of one step of speculative decoding: the first pass over the rows it is handed -- the final-norm
rows of the model's last step and the token that follows each -- then the row of the last kept token, the head, and
every later pass on the pass's own output row and its own last proposal, one position further each.  ``call(block,
pass, *operands)`` runs one block, as a family's step takes it; the pass, counted from 0, stands where a step hands
over the layer.

Every block of a pass is a reference chain in every stack, so nothing judges it against another implementation of
itself: each function is held to its definition by a host test, and a pass over several rows to passes of one row.
The reference blocks multiply on bf16 copies of the five weights and of the listed head rows, which a load keeps beside
the codes (``load(copies=True)``).
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch

from ..providers import ROW_QUANT, AttentionParams, Quant
from ..state import KvCache
from ..weights import BLOCK, BlockScaledTensor, Dequantized, ScaledTensor, TensorSpec, dequantize, quantize_fp8_per_channel
from . import configs
from .model import BRANCH_GAIN, Geometry

#: this family's names of the drafter's own tensors start with this; the checkpoint's prefix for them is found in its headers
OWN = "drafter."
#: how that prefix is found: the tensor every drafter of this kind has, and a second one that must sit beside it
ANCHOR, BESIDE = "pre_fc_norm_hidden.weight", "fc.weight"
#: the model's tensors a drafter reads: the embedding table, the 4-bit output head and the static scale of the head's operand
SHARED = ("embed_tokens.weight", "lm_head.weight", "lm_head.input_scale")
#: the weights of the drafter's five multiplies, as a loaded drafter names them, each with the checkpoint's parts in the
#: order they are concatenated along the output dimension
MULTIPLIES = (
    ("fc", ("fc",)),
    ("qkv", ("layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj")),
    ("o", ("layers.0.self_attn.o_proj",)),
    ("gate_up", ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj")),
    ("down", ("layers.0.mlp.down_proj",)),
)
#: its norm weights, as a loaded drafter names them
NORMS = (
    ("pre_fc_norm_embedding", "pre_fc_norm_embedding"),
    ("pre_fc_norm_hidden", "pre_fc_norm_hidden"),
    ("input_norm", "layers.0.input_layernorm"),
    ("q_norm", "layers.0.self_attn.q_norm"),
    ("k_norm", "layers.0.self_attn.k_norm"),
    ("post_norm", "layers.0.post_attention_layernorm"),
    ("final_norm", "norm"),
)
#: the block calls of one pass in the order ``body`` and ``propose`` make them, each with its call site (``""`` for a block
#: that is called once): the body ends in the final norm, the head runs on one row per sequence
BODY = (
    ("draft.embedding", ""),
    ("draft.pre_fc_norm", ""),
    ("draft.projection", "fc"),
    ("draft.norm_quantize", "input"),
    ("draft.projection", "qkv"),
    ("draft.attention.prepare", ""),
    ("draft.attention.core", ""),
    ("draft.gate_quantize", ""),
    ("draft.projection", "o"),
    ("draft.norm_quantize", "mlp"),
    ("draft.projection", "gate_up"),
    ("draft.silu_mul", ""),
    ("draft.activation_quantize", ""),
    ("draft.projection", "down"),
    ("draft.final_norm", ""),
)
HEAD = (("draft.head.quantize", ""), ("draft.head", ""), ("draft.token_choice", ""))
READS = "it reads a drafter stored in bf16, whose multiplies it quantises to 8 bits when it loads"
RECIPE = 'samples/models/qwen_dense/README.md, "The checkpoint\'s drafter"'
_JSON_KINDS = {dict: "an object", str: "a string", bool: "true or false", int: "a number", float: "a number", type(None): "null"}


@dataclass(frozen=True)
class DraftVocab:
    """The token ids a drafter proposes from: each once, in ascending order, every one an id of the model.  ``listed`` is
    the number of entries of the list as it was given; ``repeated`` and ``outside`` count the entries that were dropped
    because their id was listed before, or is not an id of the model."""

    ids: Tuple[int, ...]
    vocab: int
    source: str  # the file's own name, never a path; or what drew the list
    listed: int
    repeated: int = 0
    outside: int = 0

    @classmethod
    def read(cls, path: str, vocab: int) -> "DraftVocab":
        """The list of a JSON file, for a model of ``vocab`` ids.  A file that cannot be read, is not JSON, is not a list,
        holds an entry that is not a whole number (nothing is converted), or lists no id of the model is a ``ValueError``
        that names the file by its own name."""
        shown = os.path.basename(os.path.normpath(str(path))) or "."
        try:
            with open(path, encoding="utf-8") as f:
                listed = json.load(f)
        except OSError as error:
            raise ValueError(f"draft vocabulary {shown}: cannot be read ({error.strerror or type(error).__name__})") from None
        except ValueError as error:  # not JSON, or not UTF-8
            raise ValueError(f"draft vocabulary {shown}: not a JSON file ({error})") from None
        if not isinstance(listed, list):
            raise ValueError(f"draft vocabulary {shown}: holds {_JSON_KINDS[type(listed)]} where a JSON list of token ids is expected")
        for position, entry in enumerate(listed):
            if type(entry) is not int:  # a fraction, a string, true: a list that someone has to look at
                raise ValueError(f"draft vocabulary {shown}: entry {position} is {repr(entry)[:40]} where a token id (a whole number) is expected")
        inside = [entry for entry in listed if 0 <= entry < vocab]
        ids = tuple(sorted(set(inside)))
        if not ids:
            raise ValueError(f"draft vocabulary {shown}: none of its {len(listed):,} entries is a token id of this model (0 to {vocab - 1:,})")
        return cls(ids, int(vocab), shown, len(listed), repeated=len(inside) - len(ids), outside=len(listed) - len(inside))

    @classmethod
    def seeded(cls, vocab: int, rows: Optional[int] = None, seed: int = 0) -> "DraftVocab":
        """A list drawn on the CPU generator from ``seed``: ``rows`` distinct ids of the model (default: a quarter of its
        vocabulary), for the seeded model, which has no file to read one from."""
        rows = vocab // 4 if rows is None else rows
        if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= vocab:
            raise ValueError(f"a seeded draft vocabulary holds between 1 and {vocab:,} ids; got {rows!r}")
        generator = torch.Generator(device="cpu").manual_seed((int(seed) * 1000003 + zlib.crc32(b"draft vocabulary")) % (2**63))
        ids = sorted(torch.randperm(vocab, generator=generator)[:rows].tolist())
        return cls(tuple(ids), int(vocab), f"seeded list (seed {int(seed)})", rows)

    def text(self) -> str:
        dropped = [f"{count:,} {why}" for count, why in ((self.repeated, "listed before"), (self.outside, "outside the vocabulary")) if count]
        entries = f"{self.listed:,} entries listed; dropped: {', '.join(dropped)}" if dropped else f"{self.listed:,} entries listed, none dropped"
        return f"draft vocabulary {self.source}: {len(self.ids):,} ids of the model's {self.vocab:,}, in ascending order ({entries})"


@dataclass(frozen=True)
class ManifestRow:
    """One tensor of the drafter: this family's name, the checkpoint's own (the family's name for seeded weights), how it is
    stored, and what a load makes of it."""

    name: str
    key: str
    stored: str
    loaded: str


@dataclass
class DrafterWeights:
    """A loaded drafter.  The five weights are 8-bit codes with one fp32 scale per row; the norm weights are the
    checkpoint's bf16 tensors; ``head`` is the rows of the model's 4-bit output head at the draft vocabulary's ids and
    ``head_ids`` the token id of each of those rows; ``embedding`` is the model's table.  ``copies`` is what the
    reference blocks of a pass multiply on: per weight (``fc``, ``qkv``, ``o``, ``gate_up``, ``down``) and for the
    listed head rows (``head``) the bf16 copy of the code values with the scale a multiply applies and the stored
    tensor it was made from (``weights.Dequantized``); ``None`` for a load that kept none."""

    embedding: object  # [vocabulary, hidden] bf16
    pre_fc_norm_embedding: object  # [hidden] bf16
    pre_fc_norm_hidden: object  # [hidden] bf16
    fc: ScaledTensor  # [hidden, 2 hidden]
    input_norm: object  # [hidden] bf16
    qkv: ScaledTensor  # [2 attention width + 2 key/value width, hidden]: query (with its gate half), key, value
    q_norm: object  # [head dim] bf16
    k_norm: object  # [head dim] bf16
    o: ScaledTensor  # [hidden, attention width]
    post_norm: object  # [hidden] bf16
    gate_up: ScaledTensor  # [2 MLP width, hidden]: the gate rows, then the up rows
    down: ScaledTensor  # [hidden, MLP width]
    final_norm: object  # [hidden] bf16
    head: BlockScaledTensor  # [listed ids, hidden]
    head_quant: Quant  # how the head's operand is quantised: 4-bit, the head's static input scale
    head_ids: object  # [listed ids] int32
    copies: Optional[Dict[str, Dequantized]] = None  # fc, qkv, o, gate_up, down, head: what a reference block multiplies on


@dataclass
class QwenDenseDrafter:
    """The drafter of one model: its tensor list, its draft vocabulary and its load.  ``describe`` makes it from a described
    checkpoint, ``seeded`` for the reduced seeded model."""

    geometry: Geometry  # the model's widths; its one layer is an attention layer
    vocabulary: DraftVocab
    shared: Tuple[TensorSpec, ...]  # the model's own specs of the tensors the drafter reads from it
    label: str = "seeded weights"  # what it was described from
    prefix: str = ""  # the checkpoint's prefix of the drafter's own tensors
    keys: Dict[str, str] = field(default_factory=dict)  # this family's tensor names -> the checkpoint's own
    name: str = "qwen_dense drafter"
    # what a plan asks of a model (``blocks.plan(drafter.provider_table, drafter.regimes, .., model=drafter)``).  The drafter
    # names no target stacks: no kernel census holds a draft pass, and a plan of it states no kernel identity
    provider_table = configs.DRAFTER_PROVIDERS
    regimes = configs.REGIMES
    table_source = "model_samples/qwen_dense/configs.py"

    @classmethod
    def seeded(cls, model, rows: Optional[int] = None, seed: int = 0) -> "QwenDenseDrafter":
        """The drafter of a seeded model: its tensors are drawn by name like the model's (``Checkpoint.synthetic(drafter,
        seed)``; with the model's seed the head it cuts its rows from is the model's), its draft vocabulary from ``seed``."""
        return cls(_one_layer(model.geometry), DraftVocab.seeded(model.geometry.vocab, rows, seed), _shared(model))

    @classmethod
    def describe(cls, checkpoint, model, draft_vocab: Optional[str] = None) -> "QwenDenseDrafter":
        """The drafter of a checkpoint on local disk, for the ``model`` that was described from the same checkpoint
        (``QwenDense.describe``).  Its tensor list is bound beside the model's from the shard headers; nothing is read.

        ``NotImplementedError`` (a SKIP) when the checkpoint holds no drafter or one this sample does not read: more than
        one layer, an embedding table of its own, a tensor that is missing, in another format or of another shape, or one
        under the drafter's prefix that is not on the list.  ``ValueError`` (an ERROR) when an entry of ``config.json``
        holds another kind than it stands for, when no draft vocabulary is given (the sentence names the recipe), and for
        everything ``DraftVocab.read`` refuses.  The checkpoint is judged first, the list after it."""
        config = checkpoint.config.get("text_config", checkpoint.config)
        layers = config.get("mtp_num_hidden_layers", 1)
        if isinstance(layers, bool) or not isinstance(layers, int):
            raise ValueError(f"{checkpoint.label}: config.json has mtp_num_hidden_layers = {layers!r} where a whole number is expected")
        own_table = config.get("mtp_use_dedicated_embeddings", False)
        if not isinstance(own_table, bool):
            raise ValueError(f"{checkpoint.label}: config.json has mtp_use_dedicated_embeddings = {own_table!r} where true or false is expected")
        prefix = _prefix(checkpoint)
        if layers != 1:
            raise NotImplementedError(f"{checkpoint.label} has a drafter of {layers} layers; this sample reads a drafter of one layer")
        if own_table:
            raise NotImplementedError(f"{checkpoint.label} has a drafter with an embedding table of its own; this sample reads one that shares the model's")
        shared = _shared(model)
        specs = _own_specs(model.geometry) + list(shared)
        of_model = model.checkpoint_keys(checkpoint)
        keys = {spec.name: (prefix + spec.name[len(OWN) :] if spec.name.startswith(OWN) else of_model[spec.name]) for spec in specs}
        read = set(keys.values())
        unread = sorted(key for key in checkpoint.tensors if key.startswith(prefix) and key not in read)
        if unread:
            more = f" (and {len(unread) - 1} more)" if len(unread) > 1 else ""
            raise NotImplementedError(f"{checkpoint.label} holds a drafter tensor this sample does not read, {unread[0]!r}{more}: {READS}")
        checkpoint.bind(specs, keys, reads=READS)
        if draft_vocab is None:
            raise ValueError(
                "a checkpoint's drafter needs a draft vocabulary: a JSON list of the token ids its head is cut to. "
                f"A recipe that makes one from the files --dump-tokens writes: {RECIPE}"
            )
        return cls(_one_layer(model.geometry), DraftVocab.read(draft_vocab, model.geometry.vocab), shared, checkpoint.label, prefix, keys)

    # ------------------------------------------------------------------ what a plan asks of a model
    @property
    def norm_eps(self) -> float:
        return self.geometry.norm_eps

    def calls(self) -> Dict[str, int]:
        """Calls per pass of every block: five for the multiply, two for the norm + quantise, one for every other block."""
        counts: Dict[str, int] = {}
        for block, _ in (*BODY, *HEAD):
            counts[block] = counts.get(block, 0) + 1
        return counts

    def calls_per_pass(self) -> int:
        return len(BODY) + len(HEAD)

    def calls_text(self) -> Dict[str, str]:
        return {block: str(count) for block, count in self.calls().items()}

    def wiring(self) -> Tuple[Tuple[str, int], ...]:
        """The block calls of ONE pass in the order ``body`` and ``propose`` make them, as ``(block, pass)``: the pass stands
        where a model's wiring has the layer, and this is pass 0.  The passes of a step are this list once per pass, with
        the pass's number (``draft`` hands it to ``call``)."""
        return tuple((block, 0) for block, _ in (*BODY, *HEAD))

    def sites(self) -> Tuple[str, ...]:
        """The call site of every call of ``wiring()``, in its order: the weight's name for the multiply (``fc``, ``qkv``,
        ``o``, ``gate_up``, ``down``: ``configs.DRAFT_PROJECTION_SITES``), what the codes feed for the norm + quantise
        (``input``, ``mlp``: ``configs.DRAFT_NORM_SITES``), ``""`` for a block that is called once."""
        return tuple(site for _, site in (*BODY, *HEAD))

    # ------------------------------------------------------------------ the draft pass
    def state(self, sequences: int, capacity: int, device) -> KvCache:
        """The drafter's own attention cache for ``sequences`` sequences of up to ``capacity`` positions: keys and values
        ``[sequences, kv_heads, capacity, head_dim]`` bf16, zero.  It has no position of its own: every pass is told
        where its rows stand."""
        if sequences < 1 or capacity < 1:
            raise ValueError(f"a drafter's cache needs at least one sequence and one position; got sequences={sequences}, capacity={capacity}")
        g = self.geometry
        shape = (sequences, g.attn_kv_heads, capacity, g.attn_head_dim)
        return KvCache(torch.zeros(shape, dtype=torch.bfloat16, device=device), torch.zeros(shape, dtype=torch.bfloat16, device=device))

    def _copies(self, weights: DrafterWeights) -> Dict[str, Dequantized]:
        if weights.copies is None:
            raise ValueError(
                "the reference blocks of a draft pass multiply on bf16 copies of the drafter's weights, and this load kept none: "
                "load the drafter with copies=True"
            )
        return weights.copies

    def body(self, call, weights: DrafterWeights, cache: KvCache, hidden, ids, position: int, index: int = 0):
        """The body of one pass, 15 block calls: ``hidden`` ``[rows, hidden]`` bf16 and ``ids`` ``[rows]`` int32 on the device
        -> the final-norm rows ``[rows, hidden]`` bf16.  The rows stand sequence by sequence, ``rows / sequences`` per
        sequence (the sequences are the cache's); row ``t`` of a sequence reads hidden row ``t`` and the embedding of id
        ``t`` and stands at ``position + t``, where its key and value are written into ``cache``.  ``call(block, pass,
        *operands)`` runs one block; ``index`` is the pass this is, counted from 0, which ``call`` is handed in the
        place a step hands over the layer.  Nothing is read from the device.

        The attention blocks are handed the cache as a view of their own (``KvCache(k, v, rows / sequences)``): how many
        positions a pass inserts is the pass's, not a property of the cache.  ``ValueError`` for rows that are not a
        whole number per sequence, for ids that are not one per row, and for a pass that would write past the cache."""
        sequences, rows = cache.k.shape[0], hidden.shape[0]
        if rows < sequences or rows % sequences or ids.shape[0] != rows:
            raise ValueError(
                f"a draft pass takes a whole number of rows per sequence of its cache ({sequences}) and one id per row; "
                f"got {rows} hidden rows and {ids.shape[0]} ids"
            )
        tokens = rows // sequences
        if position < 0 or position + tokens > cache.k.shape[2]:
            raise ValueError(
                f"the drafter's cache was built for {cache.k.shape[2]} positions per sequence; this pass writes {position} to {position + tokens - 1}"
            )
        copies, g = self._copies(weights), self.geometry
        view = KvCache(cache.k, cache.v, tokens)
        attention = AttentionParams(g.attn_heads, g.attn_kv_heads, g.attn_head_dim, g.rotary_dim, g.rope_theta, weights.q_norm, weights.k_norm)
        embedded = call("draft.embedding", index, ids, weights.embedding)
        q = call("draft.pre_fc_norm", index, embedded, hidden, weights.pre_fc_norm_embedding, weights.pre_fc_norm_hidden, ROW_QUANT)
        stream = call("draft.projection", index, q, copies["fc"], "fc")  # the residual stream of the pass
        q, _, residual = call("draft.norm_quantize", index, stream, None, weights.input_norm, ROW_QUANT, "input")
        row = call("draft.projection", index, q, copies["qkv"], "qkv")
        qh, gate = call("draft.attention.prepare", index, row, view, position, attention)
        attended = call("draft.attention.core", index, qh, view, position)
        q = call("draft.gate_quantize", index, attended, gate, ROW_QUANT)
        branch = call("draft.projection", index, q, copies["o"], "o")
        q, _, residual = call("draft.norm_quantize", index, branch, residual, weights.post_norm, ROW_QUANT, "mlp")
        gate_up = call("draft.projection", index, q, copies["gate_up"], "gate_up")
        q = call("draft.activation_quantize", index, call("draft.silu_mul", index, gate_up), ROW_QUANT)
        branch = call("draft.projection", index, q, copies["down"], "down")
        _, normed, _ = call("draft.final_norm", index, branch, residual, weights.final_norm, None)
        return normed

    def propose(self, call, weights: DrafterWeights, rows, index: int = 0):
        """The head of one pass, 3 block calls: the final-norm rows of one row per sequence ``[sequences, hidden]`` bf16 ->
        the proposed token of every sequence ``[sequences]`` int32, on the device: the head's static 4-bit quantisation,
        the multiply over the listed rows of the model's output head, the column of the largest logit (the first of
        equal ones) and its token id.  ``index``: as for ``body``."""
        q = call("draft.head.quantize", index, rows, weights.head_quant)
        logits = call("draft.head", index, q, self._copies(weights)["head"])
        return call("draft.token_choice", index, logits, weights.head_ids)

    def draft(self, call, weights: DrafterWeights, cache: KvCache, hidden, ids, position: int, kept: int, passes: int):
        """The ``passes`` passes of one step of speculative decoding -> the proposed tokens ``[sequences, passes]`` int32, on
        the device.

        Pass 0 is ``body`` on what it is handed: per sequence the final-norm rows of the model's last step and, for each
        of them, the token that follows it; ``position`` is where the first row of a sequence stands.  Of its output the
        row ``kept`` of every sequence is taken -- the row of the last token the run kept; a view by a host integer,
        nothing is launched for it -- and ``propose`` gives the first proposal.  Pass ``j`` = 1, 2, .. is ``body`` on
        the output row of pass ``j - 1`` and the proposal of pass ``j - 1``, one row per sequence at position ``position
        + kept + j``, then ``propose``.  The rows of pass 0 behind ``kept`` write cache positions that the later passes
        and the first pass of the next step write again before a row reads them.

        ``ValueError`` for a ``kept`` that is not a row of a sequence, and for no pass."""
        sequences = cache.k.shape[0]
        tokens = hidden.shape[0] // sequences
        if isinstance(kept, bool) or not isinstance(kept, int) or not 0 <= kept < max(tokens, 1):
            raise ValueError(f"a draft step continues from one of the {tokens} rows per sequence it is handed, 0 to {tokens - 1}; got kept={kept!r}")
        if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
            raise ValueError(f"a draft step has at least one pass; got passes={passes!r}")
        row = self.body(call, weights, cache, hidden, ids, position).unflatten(0, (sequences, tokens))[:, kept]
        drafts = [self.propose(call, weights, row)]
        for index in range(1, passes):
            row = self.body(call, weights, cache, row, drafts[-1], position + kept + index, index)
            drafts.append(self.propose(call, weights, row, index))
        return torch.stack(drafts, dim=1)

    # ------------------------------------------------------------------ the tensor list
    def tensor_specs(self) -> List[TensorSpec]:
        """The drafter's own tensors, all bf16, with the recipe ``Checkpoint.synthetic`` draws them by (the model's:
        weights at the unit scale of their input width, the two that write back into the residual stream at a quarter
        of it), then the model's tensors it reads, as the model specifies them."""
        return _own_specs(self.geometry) + list(self.shared)

    def manifest(self) -> Tuple[ManifestRow, ...]:
        """Every tensor of the drafter with what a load makes of it: its norm weights as stored, the weights of its five
        multiplies as 8-bit codes with one scale per row (a part of a merged weight with its rows there), the model's
        three as the drafter uses them."""
        specs = {spec.name: spec for spec in self.tensor_specs()}

        def row(name, loaded):
            spec = specs[name]
            fmt = {"bf16": "bf16", "nvfp4": "4-bit", "scalar": "fp32"}[spec.fmt]
            return ManifestRow(name, self.keys.get(name, name), f"{fmt} {list(spec.shape)}" if spec.shape else f"{fmt} scalar", loaded)

        rows = [row(OWN + part + ".weight", f"{loaded}: as stored") for loaded, part in NORMS]
        for loaded, parts in MULTIPLIES:
            start = 0
            for part in parts:
                name = OWN + part + ".weight"
                stop = start + specs[name].shape[0]
                where = f"rows {start:,} to {stop - 1:,} of {loaded}" if len(parts) > 1 else loaded
                rows.append(row(name, f"{where}: 8-bit codes, one scale per row"))
                start = stop
        ids = f"{len(self.vocabulary.ids):,}"
        rows.append(row(SHARED[0], "embedding: the model's table"))
        rows.append(row(SHARED[1], f"head: its rows at the {ids} ids of the draft vocabulary, as stored"))
        rows.append(row(SHARED[2], "head_quant: the static scale of the head's 4-bit operand"))
        return tuple(rows)

    def load_bytes(self) -> Dict[str, int]:
        """What a loaded drafter holds on the device besides the model's embedding table, in bytes: arithmetic on the tensor
        list.  ``weights``: one byte per element of the five weights and one fp32 scale per row; ``norms``: 2 bytes per
        element; ``head``: half a byte per element of the listed rows, one block-scale byte per 16, and an int32 id per row."""
        specs = {spec.name: spec for spec in self.tensor_specs()}
        weights = norms = 0
        for _, parts in MULTIPLIES:
            for part in parts:
                spec = specs[OWN + part + ".weight"]
                weights += spec.numel + 4 * spec.shape[0]
        for _, part in NORMS:
            norms += 2 * specs[OWN + part + ".weight"].numel
        rows, hidden = len(self.vocabulary.ids), self.geometry.hidden
        return {"weights": weights, "norms": norms, "head": rows * (hidden // 2 + hidden // BLOCK + 4)}

    def copy_bytes(self) -> Dict[str, int]:
        """What the bf16 copies of a load with ``copies=True`` hold, in bytes: 2 per element of the five weights
        (``weights``) and of the listed head rows (``head``).  The reference blocks of a pass multiply on them."""
        specs = {spec.name: spec for spec in self.tensor_specs()}
        weights = sum(2 * specs[OWN + part + ".weight"].numel for _, parts in MULTIPLIES for part in parts)
        return {"weights": weights, "head": 2 * len(self.vocabulary.ids) * self.geometry.hidden}

    def memory_bytes(self, sequences: int, capacity: int, copies: bool = True) -> int:
        """What a loaded drafter and its cache hold on the device besides the model's embedding table, in bytes: what the
        load holds (``load_bytes``), the bf16 copies of a load with ``copies`` (``copy_bytes``), and the cache of
        ``sequences`` sequences of ``capacity`` positions (keys and values, 2 bytes per element).  No working memory:
        a run that states what it needs adds its own allowance."""
        g = self.geometry
        cache = 2 * sequences * g.attn_kv_heads * capacity * g.attn_head_dim * 2
        return sum(self.load_bytes().values()) + (sum(self.copy_bytes().values()) if copies else 0) + cache

    def manifest_text(self) -> str:
        """The manifest as a run prints it: no path, a file by its own name."""
        rows = self.manifest()
        own = sum(1 for row in rows if row.name.startswith(OWN))
        under = f" under {self.prefix!r}" if self.prefix else ""
        width = max(len(row.key) for row in rows)
        stored = max(len(row.stored) for row in rows)
        lines = [f"drafter of {self.label}: {own} tensors of its own{under}, {len(rows) - own} of the model's"]
        lines.append(f"  {'tensor':<{width}}  {'stored':<{stored}}  at load")
        lines += [f"  {row.key:<{width}}  {row.stored:<{stored}}  {row.loaded}" for row in rows]
        held = self.load_bytes()
        gib = 2.0**30
        lines.append(f"  {self.vocabulary.text()}")
        lines.append(
            f"  loaded, besides the model's embedding table: {sum(held.values()) / gib:.2f} GiB "
            f"(8-bit weights {held['weights'] / gib:.2f}, norm weights {held['norms'] / gib:.2f}, head rows {held['head'] / gib:.2f})"
        )
        copied = self.copy_bytes()
        lines.append(
            f"  and the bf16 copies its reference blocks multiply on (a load keeps them unless told copies=False): {sum(copied.values()) / gib:.2f} GiB "
            f"(of the five weights {copied['weights'] / gib:.2f}, of the head rows {copied['head'] / gib:.2f})"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ load
    def load(self, checkpoint, device, stages=None, embedding=None, copies: bool = True) -> DrafterWeights:
        """Read (or draw) the drafter's tensors, concatenate the parts of its merged weights, quantise the five weights to
        8 bits per row, and cut the head.  The head is read whole on the CPU and only its listed rows go to ``device``.
        ``embedding``: the table the model loaded, so that the drafter holds no second copy; without it the table is read.
        ``copies``: beside the codes, the bf16 copies the reference blocks of a pass multiply on -- of the five weights
        and of the listed head rows, each made from the codes the load keeps (``DrafterWeights.copies``; exact:
        ``weights.dequantize``).  A pass on a load without them is a ``ValueError``.
        ``stages`` (``timing.Stages``) accumulates the seconds of read (or draw), merge, quantise and dequantise."""
        from ..timing import Stages

        stages = Stages() if stages is None else stages
        read = "draw" if getattr(checkpoint, "synthetic_weights", False) else "read"

        def stored(name):
            with stages.stage(read):
                return checkpoint.load(name, device)

        loaded = {name: stored(OWN + part + ".weight") for name, part in NORMS}
        for name, parts in MULTIPLIES:
            weight = [stored(OWN + part + ".weight") for part in parts]
            if len(weight) > 1:
                with stages.stage("merge"):
                    weight = [torch.cat(weight)]
            with stages.stage("quantise"):
                loaded[name] = quantize_fp8_per_channel(weight[0])
            del weight
        with stages.stage(read):
            whole = checkpoint.load(SHARED[1], "cpu")
            index = torch.tensor(self.vocabulary.ids, dtype=torch.int64)
            head = BlockScaledTensor(whole.codes[index].to(device), whole.block_scales[index].to(device), whole.scale)
            del whole
        copied = None
        if copies:  # what a reference block multiplies on: the code values in bf16, with the scale and the stored tensor beside them
            copied = {}
            with stages.stage("dequantise"):
                for name, held in (*((name, loaded[name]) for name, _ in MULTIPLIES), ("head", head)):
                    copied[name] = dequantize(held)
                    copied[name].stored = held
        return DrafterWeights(
            embedding=stored(SHARED[0]) if embedding is None else embedding,
            head=head,
            head_quant=Quant("nvfp4", checkpoint.load(SHARED[2], device)),
            head_ids=index.to(torch.int32).to(device),
            copies=copied,
            **loaded,
        )


def _prefix(checkpoint) -> str:
    """The checkpoint's prefix of the drafter's own tensors, from its shard headers: the one prefix that has both ``ANCHOR``
    and ``BESIDE`` under it."""
    found = sorted({key[: -len(ANCHOR)] for key in checkpoint.tensors if key.endswith("." + ANCHOR)})
    found = [prefix for prefix in found if prefix + BESIDE in checkpoint.tensors]
    if not found:
        raise NotImplementedError(f"{checkpoint.label} holds no drafter (no tensor ends in {'.' + ANCHOR!r} with {BESIDE!r} beside it)")
    if len(found) > 1:
        raise NotImplementedError(f"{checkpoint.label}: the drafter's tensors are not under one prefix (found: {', '.join(repr(p) for p in found)})")
    return found[0]


def _one_layer(geometry: Geometry) -> Geometry:
    """The geometry of a model's drafter: the model's widths, and one layer, an attention layer."""
    return replace(geometry, layer_kinds=("full_attention",))


def _own_specs(g: Geometry) -> List[TensorSpec]:
    """The drafter's own tensors for a model of geometry ``g``: seven norm weights, then the parts of its five weights."""
    kv = g.attn_kv_heads * g.attn_head_dim
    shapes = {
        "fc": (g.hidden, 2 * g.hidden),
        "layers.0.self_attn.q_proj": (2 * g.attn_out, g.hidden),
        "layers.0.self_attn.k_proj": (kv, g.hidden),
        "layers.0.self_attn.v_proj": (kv, g.hidden),
        "layers.0.self_attn.o_proj": (g.hidden, g.attn_out),
        "layers.0.mlp.gate_proj": (g.mlp, g.hidden),
        "layers.0.mlp.up_proj": (g.mlp, g.hidden),
        "layers.0.mlp.down_proj": (g.hidden, g.mlp),
    }
    branch = ("layers.0.self_attn.o_proj", "layers.0.mlp.down_proj")  # these write back into the residual stream
    specs = []
    for _, part in NORMS:
        width = g.attn_head_dim if part.endswith(("q_norm", "k_norm")) else g.hidden
        specs.append(TensorSpec(OWN + part + ".weight", (width,), "bf16", std=0.02))
    for _, parts in MULTIPLIES:
        for part in parts:
            gain = BRANCH_GAIN if part in branch else 1.0
            specs.append(TensorSpec(OWN + part + ".weight", shapes[part], "bf16", std=gain * shapes[part][1] ** -0.5))
    return specs


def _shared(model) -> Tuple[TensorSpec, ...]:
    """The model's own specs of the tensors a drafter reads from it, in the order of ``SHARED``."""
    specs = {spec.name: spec for spec in model.tensor_specs()}
    return tuple(specs[name] for name in SHARED)
