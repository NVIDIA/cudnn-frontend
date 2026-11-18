# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The drafter of a dense Qwen 3.5-architecture checkpoint: which tensors it is, what a load makes of them, and the rows of
the model's output head it proposes tokens from.

A checkpoint of this family can carry a drafter beside the model: a small network that reads the model's last hidden row
and the embedding of the last token and proposes the next tokens, which the model then verifies in one step (speculative
decoding).  This module is its LOADER and nothing more: no draft step, no block, no provider row.  It answers three
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
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..providers import Quant
from ..weights import BLOCK, BlockScaledTensor, ScaledTensor, TensorSpec, quantize_fp8_per_channel
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
    ``head_ids`` the token id of each of those rows; ``embedding`` is the model's table."""

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


@dataclass
class QwenDenseDrafter:
    """The drafter of one model: its tensor list, its draft vocabulary and its load.  ``describe`` makes it from a described
    checkpoint, ``seeded`` for the reduced seeded model."""

    geometry: Geometry
    vocabulary: DraftVocab
    shared: Tuple[TensorSpec, ...]  # the model's own specs of the tensors the drafter reads from it
    label: str = "seeded weights"  # what it was described from
    prefix: str = ""  # the checkpoint's prefix of the drafter's own tensors
    keys: Dict[str, str] = field(default_factory=dict)  # this family's tensor names -> the checkpoint's own
    name: str = "qwen_dense drafter"

    @classmethod
    def seeded(cls, model, rows: Optional[int] = None, seed: int = 0) -> "QwenDenseDrafter":
        """The drafter of a seeded model: its tensors are drawn by name like the model's (``Checkpoint.synthetic(drafter,
        seed)``; with the model's seed the head it cuts its rows from is the model's), its draft vocabulary from ``seed``."""
        return cls(model.geometry, DraftVocab.seeded(model.geometry.vocab, rows, seed), _shared(model))

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
        return cls(model.geometry, DraftVocab.read(draft_vocab, model.geometry.vocab), shared, checkpoint.label, prefix, keys)

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
        return "\n".join(lines)

    # ------------------------------------------------------------------ load
    def load(self, checkpoint, device, stages=None, embedding=None) -> DrafterWeights:
        """Read (or draw) the drafter's tensors, concatenate the parts of its merged weights, quantise the five weights to
        8 bits per row, and cut the head.  The head is read whole on the CPU and only its listed rows go to ``device``.
        ``embedding``: the table the model loaded, so that the drafter holds no second copy; without it the table is read.
        ``stages`` (``timing.Stages``) accumulates the seconds of read (or draw), merge and quantise."""
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
        return DrafterWeights(
            embedding=stored(SHARED[0]) if embedding is None else embedding,
            head=head,
            head_quant=Quant("nvfp4", checkpoint.load(SHARED[2], device)),
            head_ids=index.to(torch.int32).to(device),
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
