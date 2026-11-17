# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense Qwen 3.5-architecture model: geometry, the layer list in order and the block wiring of one decode step.

A layer is a linear-attention (gated delta-rule) layer or an attention layer, three of the first to one of the second, each
followed by an MLP.  One decode step of ``rows`` sequences, block by block (the operands of every block kind):

====================== ================================================= ==============================================
block                  operands                                          result
====================== ================================================= ==============================================
embedding              ids ``[rows]`` int32, table                       ``[rows, hidden]`` bf16
add_norm_quantize      branch, residual, norm weight, ``Quant``, site    (quantised operand, normed bf16, new residual)
projection.fp8.in      8-bit operand, weight copy, site                  ``[rows, 16384 | 14336]`` bf16
linear_attention.step  x, z, normed hidden, slot ids, params, slots      8-bit ``[rows, 6144]``; state updated in place
attention.prepare      QKV row, cache, position, params                  (q, gate); key and value inserted in the cache
attention.core         q, cache, position                                ``[rows, 6144]`` bf16
attention.gate_quantize attention output, gate, ``Quant``                8-bit ``[rows, 6144]``
projection.fp8.out     8-bit operand, weight copy                        ``[rows, hidden]`` bf16
mlp.gate_up            4-bit operand, merged gate / up copy, ``Quant``   4-bit ``[rows, 17408]``
mlp.down               4-bit operand, weight copy                        ``[rows, hidden]`` bf16
output_head            4-bit operand, weight copy                        logits ``[rows, vocabulary]`` bf16
token_choice           logits                                            ids ``[rows]`` int32
====================== ================================================= ==============================================

The norm block is called at several sites of a step, and a kernel that serves it is called in another form at each: ``site``
names the call by what the normed rows feed (``NORM_SITES``; ``sites()`` lists the site of every call of the wiring).  The
in-projection is called with another weight shape in each kind of layer, and its site says which
(``IN_PROJECTION_SITES``).  A reference chain and a multiply over a copy ignore the site.

The weights that feed one multiply are merged at load: the query (with its gate half), key and value projections of an
attention layer; ``qkv`` and ``z`` of a linear-attention layer; its ``b`` and ``a`` rows, b first; the MLP's gate and up.
The loader then keeps each multiplied weight in the forms the plans of the run read it in (``STORED_WEIGHTS`` names the
blocks): a bf16 copy where a block reads one -- every fallback row, the reference stack -- and the merged stored tensor
where a block multiplies on it (8-bit codes with their one scale; packed 4-bit codes with their block scales and their
global scale).  A weight that every plan of the run reads on its stored format gets no copy; the reference blocks of the
checks then make theirs for one call at a time (``weights.OnDemand``), and the memory sentence counts the largest.

``QwenDense.reduced()`` keeps every layer width of the published model and reduces the depth and the vocabulary; every
reduction is listed in ``reductions`` and printed by a sample.  ``QwenDense.describe(checkpoint)`` takes the geometry from
the ``config.json`` of a described checkpoint and binds this family's tensor list to it: the checkpoint's own tensor names
are found in its shard headers (the language model's tensors share one prefix, the one of its token embedding), and every
tensor is checked for its storage format and shape before anything is read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..blocks import GIB
from ..providers import AttentionParams, GdnParams, Quant, split_qkvz
from ..weights import BLOCK, E2M1_MAX, E4M3_MAX, TensorSpec
from . import configs

PATTERN = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
#: seeded weights only.  The projections that write a branch back into the residual stream are drawn at a quarter of the unit
#: scale, as a trained model's branches are small against its stream: with unit branches the 4-bit quantisers of a
#: random-weight model amplify a one-unit rounding difference between two stacks to tens of percent of the logits within one
#: step.  The output head is a seeded permutation of the embedding table, so the greedy token is decided by a wide margin and
#: the produced ids are a walk through that permutation: the layers move the logits, not the choice.
BRANCH_GAIN = 0.25
#: the call sites of the residual add + RMSNorm + quantise block, by what the normed rows feed: the mixer of a layer (the
#: layer's kind), its MLP, the output head.  The first call of a step has no residual stream yet: ``FIRST`` behind its site
NORM_SITES = ("linear_attention", "full_attention", "mlp", "output_head")
FIRST = ".first"
#: the blocks that multiply a weight.  For each the loader keeps what the plans of a run read (``QwenDense.load``): the
#: bf16 copy, and -- for a provider that multiplies on the stored format (``Block.stored_weights``) -- the merged stored
#: tensor: 8-bit codes with one scale for the two projections, packed 4-bit codes with block scales and a global scale
#: for the MLP's two multiplies and the output head (``QwenDense.weight_shapes`` has the format and the shapes)
STORED_WEIGHTS = ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up", "mlp.down", "output_head")
#: the call sites of the in-projection, by the layer's kind: what the merged weight holds there (its shape differs).  A
#: site name belongs to one block: a provider reads the sites of its row out of ``sites()`` by name
IN_PROJECTION_SITES = {"linear_attention": "linear_attention.qkvz", "full_attention": "full_attention.qkv"}


@dataclass(frozen=True)
class Geometry:
    """Layer widths of the published dense 27B-class model; ``layer_kinds`` in order."""

    layer_kinds: Tuple[str, ...] = PATTERN * 16
    hidden: int = 5120
    vocab: int = 248320
    mlp: int = 17408
    la_key_heads: int = 16
    la_value_heads: int = 48
    la_head_dim: int = 128
    la_conv_width: int = 4
    attn_heads: int = 24
    attn_kv_heads: int = 4
    attn_head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1.0e7
    norm_eps: float = 1.0e-6

    @property
    def qkv_channels(self) -> int:
        return (2 * self.la_key_heads + self.la_value_heads) * self.la_head_dim

    @property
    def la_out(self) -> int:
        return self.la_value_heads * self.la_head_dim

    @property
    def attn_out(self) -> int:
        return self.attn_heads * self.attn_head_dim

    def in_projection(self, kind: str) -> Tuple[int, int]:
        """``(out features, in features)`` of the merged in-projection weight of a layer of ``kind``: ``qkv`` and ``z`` of a
        linear-attention layer; the query (with its gate half), the key and the value projection of an attention layer."""
        if kind == "linear_attention":
            return self.qkv_channels + self.la_out, self.hidden
        return 2 * self.attn_out + 2 * self.attn_kv_heads * self.attn_head_dim, self.hidden


@dataclass
class LayerWeights:
    kind: str
    input_norm: object
    in_quant: Quant
    in_proj: object
    out_proj: object
    post_norm: object
    mlp_quant: Quant
    gate_up: object
    down_quant: Quant
    down: object
    gdn: Optional[GdnParams] = None
    attn: Optional[AttentionParams] = None
    out_quant: Optional[Quant] = None  # attention layers: the out-projection's static 8-bit input scale


@dataclass
class Weights:
    """The weights of one load.  ``stored`` and ``uncopied`` say what that load kept of the multiplied weights, by block
    (``STORED_WEIGHTS``): the blocks whose weights it holds on their stored format, and those of them it holds on that
    format alone, without a bf16 copy.  A session reads them before its first step: a block must find the form it reads.
    ``alone`` lists the weights of the blocks ``uncopied`` one by one -- ``(block, layer, weight)``, the layer ``None``
    for the output head -- each with the checksum of its stored bytes at this load (``weights.Dequantized.checksum``):
    the checks of a run hold these bytes to it, because their reference of such a weight is made from them."""

    embedding: object
    layers: List[LayerWeights]
    final_norm: object
    head_quant: Quant
    head: object
    copy_bytes: int = 0
    stored: Tuple[str, ...] = ()
    uncopied: Tuple[str, ...] = ()
    alone: Tuple[tuple, ...] = ()


@dataclass
class QwenDense:
    """The model as a sample sees it: geometry, tables, the synthetic weight recipe, the load and the step."""

    geometry: Geometry = field(default_factory=Geometry)
    reductions: Tuple[str, ...] = ()
    name: str = "qwen_dense"
    provider_table = configs.PROVIDERS
    regimes = configs.REGIMES
    first_launch_reserve = configs.FIRST_LAUNCH_RESERVE
    logits_band_units = configs.LOGITS_BAND_UNITS
    table_source = "model_samples/qwen_dense/configs.py"
    target_stacks = configs.TARGET_STACKS  # the stacks a run of which is held against the census of the target kernel set
    _target = None  # that census, read once per process

    @classmethod
    def target(cls):
        """The family's census of the target kernel set of a decode step (``kernel_census.json`` beside ``configs.py``):
        ``model_samples.census.Target``.  ``ValueError`` when the file is not such a census."""
        if cls._target is None:
            from ..census import Target

            cls._target = Target.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_census.json"))
        return cls._target

    @classmethod
    def reduced(cls, layers: int = 2, vocab: int = 15520) -> "QwenDense":
        """The published layer widths with ``layers`` layers (the last ones of the published order, so the model ends in an
        attention layer) and a vocabulary of ``vocab`` ids."""
        full = Geometry()
        if not 1 <= layers <= len(full.layer_kinds):
            raise ValueError(f"layers must be between 1 and {len(full.layer_kinds)}; got {layers}")
        if vocab % 16 or vocab < 272:
            raise ValueError(f"the reduced vocabulary must be a multiple of 16 and at least 272; got {vocab}")
        kinds = full.layer_kinds[-layers:]
        la = kinds.count("linear_attention")
        reductions = (
            f"{layers} of {len(full.layer_kinds)} layers ({la} linear-attention + {layers - la} attention)",
            f"vocabulary {vocab:,} of {full.vocab:,}",
            "seeded weights",
            "synthetic tokenizer",
        )
        return cls(Geometry(layer_kinds=kinds, vocab=vocab), reductions)

    @classmethod
    def describe(cls, checkpoint) -> "QwenDense":
        """The model of a checkpoint on local disk (``Checkpoint.describe``): the geometry from its ``config.json``, this
        family's tensor list bound to its tensors.  Nothing is loaded.  ``NotImplementedError`` (a SKIP) when the checkpoint
        is not of this family or not in the storage formats this sample reads; ``ValueError`` (an ERROR) when an entry of
        its ``config.json`` holds something else than the number, list or object it stands for."""
        config = checkpoint.config.get("text_config", checkpoint.config)
        if not isinstance(config, dict):
            raise ValueError(f"{checkpoint.label}: config.json has a text_config that is not a JSON object")
        kind = config.get("model_type")
        if kind not in ("qwen3_5", "qwen3_5_text"):
            raise NotImplementedError(f"{checkpoint.label} is a model of type {kind!r}; this sample reads the dense Qwen 3.5 architecture")
        rope = config.get("rope_parameters") or {}
        if not isinstance(rope, dict):
            raise ValueError(f"{checkpoint.label}: config.json has rope_parameters that are not a JSON object")

        def number(key, kind=int, source=config, **default):
            """One entry of the config as a number; a ``ValueError`` that names the entry where it holds something else."""
            value = source.get(key, default["default"]) if default else source[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"{checkpoint.label}: config.json has {key} = {value!r} where {'a whole number' if kind is int else 'a number'} is expected")
            return kind(value)

        try:
            kinds = config["layer_types"]
            if not isinstance(kinds, list) or not all(isinstance(name, str) for name in kinds):
                raise ValueError(f"{checkpoint.label}: config.json has layer_types = {kinds!r} where a list of layer names is expected")
            kinds = tuple(kinds)
            head_dim = number("head_dim")
            geometry = Geometry(
                layer_kinds=kinds,
                hidden=number("hidden_size"),
                vocab=number("vocab_size"),
                mlp=number("intermediate_size"),
                la_key_heads=number("linear_num_key_heads"),
                la_value_heads=number("linear_num_value_heads"),
                la_head_dim=number("linear_value_head_dim"),
                la_conv_width=number("linear_conv_kernel_dim"),
                attn_heads=number("num_attention_heads"),
                attn_kv_heads=number("num_key_value_heads"),
                attn_head_dim=head_dim,
                rotary_dim=int(head_dim * number("partial_rotary_factor", float, rope, default=config.get("partial_rotary_factor", 1.0))),
                rope_theta=number("rope_theta", float, rope, default=config.get("rope_theta", 10000.0)),
                norm_eps=number("rms_norm_eps", float),
            )
        except KeyError as error:
            raise NotImplementedError(f"{checkpoint.label}: config.json has no {error.args[0]!r}; not a checkpoint this sample reads") from None
        key_head_dim = number("linear_key_head_dim", default=geometry.la_head_dim)
        unread = [
            ("layer types other than linear_attention and full_attention", set(kinds) - set(PATTERN)),
            ("a key head width that differs from the value head width", key_head_dim != geometry.la_head_dim),
            ("an attention layer without its output gate", not config.get("attn_output_gate", True)),
            (f"activation {config.get('hidden_act')!r}", config.get("hidden_act", "silu") != "silu"),
        ]
        for what, present in unread:
            if present:
                raise NotImplementedError(f"{checkpoint.label} has {what}; this sample's blocks do not model that")
        model = cls(geometry, ())
        model.logits_band_units = configs.CHECKPOINT_LOGITS_BAND_UNITS
        checkpoint.bind(model.tensor_specs(), model.checkpoint_keys(checkpoint))
        return model

    def checkpoint_keys(self, checkpoint) -> Dict[str, str]:
        """This family's tensor names -> the checkpoint's own, read from its shard headers.  The language model's tensors
        share one prefix, found as the prefix of the token embedding that also has a first layer under it; the output head is
        stored under that prefix or under none."""
        anchor, first = "embed_tokens.weight", "layers.0.input_layernorm.weight"
        prefixes = sorted({key[: -len(anchor)] for key in checkpoint.tensors if key.endswith(anchor)})
        prefixes = [p for p in prefixes if (p == "" or p.endswith(".")) and p + first in checkpoint.tensors]
        if len(prefixes) != 1:
            found = ", ".join(repr(p) for p in prefixes) or "none"
            raise NotImplementedError(f"{checkpoint.label}: the language model's tensors are not under one prefix (found: {found})")
        prefix = prefixes[0]
        return {
            s.name: (s.name if prefix + s.name not in checkpoint.tensors and s.name in checkpoint.tensors else prefix + s.name) for s in self.tensor_specs()
        }

    @property
    def norm_eps(self) -> float:
        return self.geometry.norm_eps

    # ------------------------------------------------------------------ tables
    def calls(self) -> Dict[str, Tuple[int, ...]]:
        """Calls per step of every block: one number, or (linear-attention layers, attention layers) for the projections."""
        kinds = self.geometry.layer_kinds
        la, at, n = kinds.count("linear_attention"), kinds.count("full_attention"), len(kinds)
        return {
            "embedding": (1,),
            "add_norm_quantize": (2 * n + 1,),
            "projection.fp8.in": (la, at),
            "linear_attention.step": (la,),
            "projection.fp8.out": (la, at),
            "mlp.gate_up": (n,),
            "mlp.down": (n,),
            "attention.prepare": (at,),
            "attention.core": (at,),
            "attention.gate_quantize": (at,),
            "output_head": (1,),
            "token_choice": (1,),
        }

    def calls_per_step(self) -> int:
        return sum(sum(c) for c in self.calls().values())

    def wiring(self) -> Tuple[Tuple[str, int], ...]:
        """The block calls of one decode step in the order ``step`` makes them, as ``(block, layer)``: the wiring order the
        kernel check holds a step's launches against.  A block outside the layers has layer -1; the final norm is the call
        after the last layer."""
        kinds = self.geometry.layer_kinds
        calls = [("embedding", -1)]
        for i, kind in enumerate(kinds):
            calls += [("add_norm_quantize", i), ("projection.fp8.in", i)]
            if kind == "linear_attention":
                calls.append(("linear_attention.step", i))
            else:
                calls += [("attention.prepare", i), ("attention.core", i), ("attention.gate_quantize", i)]
            calls += [("projection.fp8.out", i), ("add_norm_quantize", i), ("mlp.gate_up", i), ("mlp.down", i)]
        calls += [("add_norm_quantize", len(kinds)), ("output_head", -1), ("token_choice", -1)]
        return tuple(calls)

    def sites(self) -> Tuple[str, ...]:
        """The call site of every call of ``wiring()``, in its order: for the norm block what its normed rows feed (the layer's
        kind for a layer's input norm, ``"mlp"``, ``"output_head"``), with ``".first"`` behind the first call of a step, which
        has no residual stream to add to; for the in-projection the layer's kind and what its merged weight holds
        (``IN_PROJECTION_SITES``: the weight has another shape in each kind of layer); ``""`` for a block that is called in
        one form everywhere.  A site name belongs to one block.  ``step`` hands the same names to the two blocks; a
        provider row names the form of each (``ProviderRow.sites``)."""
        kinds = self.geometry.layer_kinds
        sites, seen = [], set()
        for block, layer in self.wiring():
            if block == "projection.fp8.in":
                sites.append(IN_PROJECTION_SITES[kinds[layer]])
            elif block != "add_norm_quantize":
                sites.append("")
            elif layer == len(kinds):
                sites.append("output_head")
            elif layer in seen:  # a layer's second call: the norm in front of its MLP
                sites.append("mlp")
            else:
                seen.add(layer)
                sites.append(kinds[layer] + (FIRST if layer == 0 else ""))
        return tuple(sites)

    def calls_text(self) -> Dict[str, str]:
        return {block: " / ".join(str(c) for c in count) for block, count in self.calls().items()}

    # ------------------------------------------------------------------ synthetic weights
    def tensor_specs(self) -> List[TensorSpec]:
        """The tensors of a checkpoint of this geometry, with the recipe ``Checkpoint.synthetic`` draws them by.  Static
        activation scales are constants chosen for normed activations; a 4-bit weight's global scale is fixed so that the gate
        and the up projection share it; an 8-bit weight's scale follows its own largest value, so merged parts differ."""
        g = self.geometry
        H, I, D = g.hidden, g.mlp, g.la_head_dim
        act8, act4 = 8.0 / E4M3_MAX, 8.0 / (E4M3_MAX * E2M1_MAX)

        def w4(name, shape, gain=1.0):
            std = gain * shape[1] ** -0.5
            return [TensorSpec(name + ".weight", shape, "nvfp4", std=std, scale=6.5 * std / (E4M3_MAX * E2M1_MAX))]

        def w8(name, shape, in_scale, gain=1.0):
            return [TensorSpec(name + ".weight", shape, "fp8", std=gain * shape[1] ** -0.5), TensorSpec(name + ".input_scale", (), "scalar", mean=in_scale)]

        specs = [TensorSpec("embed_tokens.weight", (g.vocab, H), "bf16")]
        for i, kind in enumerate(g.layer_kinds):
            p = f"layers.{i}."
            specs.append(TensorSpec(p + "input_layernorm.weight", (H,), "bf16", std=0.02))
            if kind == "linear_attention":
                a = p + "linear_attn."
                specs += w8(a + "in_proj_qkv", (g.qkv_channels, H), act8) + w8(a + "in_proj_z", (g.la_out, H), act8)
                specs += [
                    TensorSpec(a + "in_proj_b.weight", (g.la_value_heads, H), "bf16", std=H**-0.5),
                    TensorSpec(a + "in_proj_a.weight", (g.la_value_heads, H), "bf16", std=H**-0.5),
                    TensorSpec(a + "conv1d.weight", (g.qkv_channels, 1, g.la_conv_width), "bf16", std=0.3),
                    TensorSpec(a + "A_log", (g.la_value_heads,), "bf16", std=0.5),
                    TensorSpec(a + "dt_bias", (g.la_value_heads,), "bf16", std=0.5),
                    TensorSpec(a + "norm.weight", (D,), "bf16", mean=1.0, std=0.2),
                ]
                specs += w8(a + "out_proj", (H, g.la_out), 16.0 / E4M3_MAX, BRANCH_GAIN)
            else:
                a = p + "self_attn."
                kv = g.attn_kv_heads * g.attn_head_dim
                specs += w8(a + "q_proj", (2 * g.attn_out, H), act8) + w8(a + "k_proj", (kv, H), act8) + w8(a + "v_proj", (kv, H), act8)
                specs += [
                    TensorSpec(a + "q_norm.weight", (g.attn_head_dim,), "bf16", std=0.02),
                    TensorSpec(a + "k_norm.weight", (g.attn_head_dim,), "bf16", std=0.02),
                ]
                specs += w8(a + "o_proj", (H, g.attn_out), act8, BRANCH_GAIN)
            specs.append(TensorSpec(p + "post_attention_layernorm.weight", (H,), "bf16", std=0.02))
            m = p + "mlp."
            specs += w4(m + "gate_proj", (I, H)) + w4(m + "up_proj", (I, H)) + w4(m + "down_proj", (H, I), BRANCH_GAIN)
            specs += [TensorSpec(m + "gate_proj.input_scale", (), "scalar", mean=act4), TensorSpec(m + "up_proj.input_scale", (), "scalar", mean=act4)]
            specs.append(TensorSpec(m + "down_proj.input_scale", (), "scalar", mean=4 * act4))
        specs.append(TensorSpec("norm.weight", (H,), "bf16", std=0.02))
        specs += [
            TensorSpec("lm_head.weight", (g.vocab, H), "nvfp4", std=H**-0.5, scale=6.5 * H**-0.5 / (E4M3_MAX * E2M1_MAX), tied="embed_tokens.weight"),
            TensorSpec("lm_head.input_scale", (), "scalar", mean=act4),
        ]
        return specs

    def weight_shapes(self, block: str) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
        """``(stored format, shapes)`` of the weight a block of ``STORED_WEIGHTS`` multiplies: ``"fp8"`` or ``"nvfp4"``, and
        ``(out features, in features)`` of the merged weight of every call of a step, in layer order."""
        g = self.geometry
        if block == "projection.fp8.in":
            return "fp8", tuple(g.in_projection(kind) for kind in g.layer_kinds)
        if block == "projection.fp8.out":
            return "fp8", tuple((g.hidden, g.la_out if kind == "linear_attention" else g.attn_out) for kind in g.layer_kinds)
        if block == "mlp.gate_up":  # the gate and the up projection, merged
            return "nvfp4", ((2 * g.mlp, g.hidden),) * len(g.layer_kinds)
        if block == "mlp.down":
            return "nvfp4", ((g.hidden, g.mlp),) * len(g.layer_kinds)
        if block == "output_head":
            return "nvfp4", ((g.vocab, g.hidden),)
        raise ValueError(f"block {block} multiplies no weight this family's loader keeps; those are {', '.join(STORED_WEIGHTS)}")

    def _elements(self, blocks: Sequence[str]) -> int:
        """The weight elements the blocks ``blocks`` multiply in a step."""
        return sum(out * inner for block in blocks for out, inner in self.weight_shapes(block)[1])

    def copy_bytes(self, uncopied: Sequence[str] = ()) -> int:
        """Bytes of the bf16 tensors a load holds: 2 per weight element -- without the copies of the blocks ``uncopied``,
        whose weights the load keeps on their stored format alone."""
        return sum(2 * s.numel for s in self.tensor_specs() if s.fmt != "scalar") - 2 * self._elements(uncopied)

    def state_bytes(self, sequences: int, capacity: int) -> int:
        """Bytes of the decode state of ``sequences`` sequences: the slot pools of the linear-attention layers (one null slot
        each) and the key/value caches of the attention layers."""
        g = self.geometry
        state = 0
        for kind in g.layer_kinds:
            if kind == "linear_attention":
                state += (sequences + 1) * (g.la_value_heads * g.la_head_dim**2 * 4 + (g.la_conv_width - 1) * g.qkv_channels * 2)
            else:
                state += 2 * sequences * g.attn_kv_heads * capacity * g.attn_head_dim * 2
        return state

    def stored_bytes(self, stored: Sequence[str] = ()) -> int:
        """Bytes of the weights the loader keeps on their stored format for the blocks ``stored`` (``STORED_WEIGHTS``): one
        per element of a merged 8-bit weight; half a byte per element of a packed 4-bit weight and one block-scale byte per
        16 elements.  The scales themselves (one float per weight) are not device memory."""
        total = 0
        for block in stored:
            fmt, shapes = self.weight_shapes(block)
            total += sum(out * inner if fmt == "fp8" else out * (inner // 2) + out * (inner // BLOCK) for out, inner in shapes)
        return total

    def check_bytes(self, uncopied: Sequence[str] = ()) -> int:
        """Bytes the checks of a run add to what its load holds: the reference blocks read bf16 copies, and for a weight of
        the blocks ``uncopied`` -- kept on its stored format alone -- the copy is made for one call and dropped after it
        (``weights.OnDemand``).  One such copy exists at a time: the largest of them, 2 bytes per element; 0 where every
        weight has its copy."""
        return max((2 * out * inner for block in uncopied for out, inner in self.weight_shapes(block)[1]), default=0)

    def memory_bytes(self, sequences: int, capacity: int, stored: Sequence[str] = (), uncopied: Sequence[str] = ()) -> int:
        """What a run needs on the GPU besides a kernel's own reservation: the weights held -- the bf16 copies (not those
        of the blocks ``uncopied``) and the stored tensors of the blocks ``stored`` (``stored_bytes``) -- the state,
        working memory, and the one copy the checks make on demand at a time (``check_bytes``).

        Working memory is load-time temporaries, activations, the reference's fp64 temporaries during the checks and the
        allocator's slack: 2 GiB, or an eighth of the weights held where that is more, and a quarter of the stored
        tensors kept.  An allowance from runs, not a bound.  Of the published checkpoint (one sequence, the default
        prompt), what PyTorch reserved and what the device grew by, against the weights held:

        * every weight as a copy, 50.1 GiB held: 55.2 GiB reserved, 55.4 GiB grown -- 5.3 GiB beyond the weights;
        * the 4.8 GiB of stored in-projections beside every copy, 54.9 GiB held: 61.7 and 61.9 GiB -- 6.9 GiB beyond;
        * the stored in-projections without their copies, 45.3 GiB held: 51.7 and 51.9 GiB -- 6.6 GiB beyond.

        So the room beyond the weights does not shrink with the copies a run leaves out, and it is larger where merged
        stored tensors stay in the pool among the temporaries of the load: an eighth of what is held covered the first
        two and was 0.5 GiB short of the third.

        On the reduced models the 2 GiB apply, and they are an estimate that runs miss by a few tenths of a GiB in
        either direction, with and without their copies: over runs of one to four layers PyTorch reserved between
        1.7 and 2.3 GiB beyond the weights, the state and the checks' copy (the family README has the runs).  The
        room there does not follow the copies left out -- 1.7 GiB with every multiplied weight of the two-layer model
        on its stored format alone, 2.3 GiB with the in-projections of the four-layer model on theirs, 2.0 GiB with
        those copies kept -- so nothing is added for them."""
        kept = self.stored_bytes(stored)
        held = self.copy_bytes(uncopied) + kept
        working = max(2 * GIB, held // 8) + kept // 4
        return held + self.state_bytes(sequences, capacity) + working + self.check_bytes(uncopied)

    def bytes_per_step(self, sequences: int, positions: int, stored: Sequence[str] = ()) -> Dict[str, int]:
        """What one decode step of ``sequences`` sequences reads and writes, in bytes: arithmetic on the tensor list, not a
        measurement.

        * ``weights``: every weight the step reads in full, in its stored format (8-bit: 1 byte per element; 4-bit: 0.5 byte
          per element plus one block-scale byte per 16; bf16: 2), and one row of the embedding table per sequence;
        * ``copies``: the same weights as a stack of this family reads them: a bf16 copy at 2 bytes per element -- but the
          weights of the blocks ``stored`` (the blocks of the stack's plan whose provider multiplies on the stored format:
          ``STORED_WEIGHTS``) on that format, as the loader kept them;
        * ``stored``: the part of ``copies`` that is read on a stored format (``stored_bytes``); 0 for a stack whose every
          multiply reads a copy;
        * ``state``: the recurrent state and the conv history of every linear-attention layer, read and written, and the
          cached keys and values of every attention layer up to ``positions`` positions, read."""
        g = self.geometry
        on_disk = copies = 0
        for spec in self.tensor_specs():
            if spec.fmt == "scalar" or spec.name == "embed_tokens.weight":
                continue
            on_disk += spec.stored_bytes
            copies += 2 * spec.numel
        embedding = 2 * sequences * g.hidden
        state = 0
        for kind in g.layer_kinds:
            if kind == "linear_attention":
                state += 2 * sequences * (g.la_value_heads * g.la_head_dim**2 * 4 + (g.la_conv_width - 1) * g.qkv_channels * 2)
            else:
                state += 2 * sequences * g.attn_kv_heads * positions * g.attn_head_dim * 2
        # a block that multiplies on the stored format reads its weight there (1 byte per element of an 8-bit weight, 0.5625 of
        # a 4-bit one), and the 2 bytes per element of a copy of it are not read by a step
        kept = self.stored_bytes(stored)
        return {"weights": on_disk + embedding, "copies": copies - 2 * self._elements(stored) + kept + embedding, "stored": kept, "state": state}

    def state_shapes(self) -> Dict[str, tuple]:
        g = self.geometry
        return {
            "state": (g.la_value_heads, g.la_head_dim, g.la_head_dim),
            "conv": (g.la_conv_width - 1, g.qkv_channels),
            "kv": (g.attn_kv_heads, g.attn_head_dim),
        }

    # ------------------------------------------------------------------ load
    def load(self, checkpoint, device, stages, merge_rule: str = "largest_scale", stored: Sequence[str] = (), uncopied: Sequence[str] = ()) -> Weights:
        """Read (or draw) and merge every weight, and keep each multiplied weight in the forms the plans of the run read
        (``weights.Dequantized``); each stage's seconds accumulate in ``stages``.

        ``stored`` names the blocks (``STORED_WEIGHTS``) a provider of the plans multiplies on the stored format: their
        merged stored tensor is kept (``Dequantized.stored``) -- the 8-bit codes with their one scale, or the packed 4-bit
        codes with their block scales and their global scale, exactly what a copy is made from.  ``uncopied`` names those
        of them that NO block of the plans reads as a bf16 copy: for them no copy is made, and a checksum of the stored
        bytes is taken in its place (stage ``checksum``; ``Weights.alone``) -- the reference of the checks will be made
        from these bytes, so the checks hold them to what they are now.  Every other multiplied weight is dequantised to
        its bf16 copy, as every stack of this family did.  The returned weights say what was kept."""
        import torch

        from ..weights import Dequantized, dequantize, merge_block_scaled, merge_scaled, stored_checksum

        unknown = sorted(set(stored) - set(STORED_WEIGHTS))
        if unknown:
            raise ValueError(
                f"this family's loader keeps the stored weights of {', '.join(STORED_WEIGHTS)}; a provider of this run multiplies on those of {', '.join(unknown)}"
            )
        bare = sorted(set(uncopied) - set(stored))
        if bare:  # a weight is held in one form at least: without its copy it needs its stored form
            raise ValueError(f"a load that keeps no bf16 copy of the weights of {', '.join(bare)} must keep their stored format, and was not asked to")
        kept, copyless = tuple(stored), tuple(uncopied)
        alone = []  # (block, layer, weight) of every weight kept on its stored format alone
        g = self.geometry
        read = "draw" if getattr(checkpoint, "synthetic_weights", False) else "read"

        def stored(name):
            with stages.stage(read):
                return checkpoint.load(name, device)

        def copy(parts, block, merge=None, layer=None):
            """The weight of ``block`` (in layer ``layer``) in the forms the plans read: its bf16 copy unless no block reads
            one, its merged stored tensor where a block multiplies on it."""
            if merge is not None:
                with stages.stage("merge"):
                    parts = [merge(parts)]
            whole = parts[0]
            if block in copyless:  # every plan of the run multiplies this weight on its stored format: no copy is made
                with stages.stage("checksum"):  # of the bytes the block multiplies on and the checks' reference is made from
                    bare = Dequantized(None, whole.scale, whole, stored_checksum(whole))
                alone.append((block, layer, bare))
                return bare
            with stages.stage("dequantise"):
                copied = dequantize(whole)
            if block in kept:  # a block of the plans multiplies on the stored format, another reads the copy: both stay
                copied.stored = whole
            return copied

        def scale(*names):  # the static input scale of merged parts collapses to the largest
            return max(checkpoint.load(n + ".input_scale", device) for n in names)

        layers = []
        for i, kind in enumerate(g.layer_kinds):
            p = f"layers.{i}."
            m = p + "mlp."
            common = dict(
                input_norm=stored(p + "input_layernorm.weight"),
                post_norm=stored(p + "post_attention_layernorm.weight"),
                mlp_quant=Quant("nvfp4", scale(m + "gate_proj", m + "up_proj")),
                gate_up=copy([stored(m + "gate_proj.weight"), stored(m + "up_proj.weight")], "mlp.gate_up", merge_block_scaled, layer=i),
                down_quant=Quant("nvfp4", scale(m + "down_proj")),
                down=copy([stored(m + "down_proj.weight")], "mlp.down", layer=i),
            )
            if kind == "linear_attention":
                a = p + "linear_attn."
                out_scale = scale(a + "out_proj")
                gdn = GdnParams(
                    g.la_key_heads,
                    g.la_value_heads,
                    g.la_head_dim,
                    stored(a + "conv1d.weight").reshape(g.qkv_channels, g.la_conv_width).contiguous(),
                    stored(a + "A_log").float(),
                    stored(a + "dt_bias"),
                    stored(a + "norm.weight"),
                    torch.cat([stored(a + "in_proj_b.weight"), stored(a + "in_proj_a.weight")]),
                    torch.tensor([out_scale], dtype=torch.float32, device=device),
                    out_scale,
                )
                parts = [stored(a + "in_proj_qkv.weight"), stored(a + "in_proj_z.weight")]
                layer = LayerWeights(
                    kind,
                    in_quant=Quant("fp8", scale(a + "in_proj_qkv", a + "in_proj_z")),
                    in_proj=copy(parts, "projection.fp8.in", lambda ps: merge_scaled(ps, merge_rule), layer=i),
                    out_proj=copy([stored(a + "out_proj.weight")], "projection.fp8.out", layer=i),
                    gdn=gdn,
                    **common,
                )
            else:
                a = p + "self_attn."
                attn = AttentionParams(
                    g.attn_heads, g.attn_kv_heads, g.attn_head_dim, g.rotary_dim, g.rope_theta, stored(a + "q_norm.weight"), stored(a + "k_norm.weight")
                )
                parts = [stored(a + "q_proj.weight"), stored(a + "k_proj.weight"), stored(a + "v_proj.weight")]
                layer = LayerWeights(
                    kind,
                    in_quant=Quant("fp8", scale(a + "q_proj", a + "k_proj", a + "v_proj")),
                    in_proj=copy(parts, "projection.fp8.in", lambda ps: merge_scaled(ps, merge_rule), layer=i),
                    out_proj=copy([stored(a + "o_proj.weight")], "projection.fp8.out", layer=i),
                    attn=attn,
                    out_quant=Quant("fp8", scale(a + "o_proj")),
                    **common,
                )
            del parts
            layers.append(layer)
        return Weights(
            embedding=stored("embed_tokens.weight"),
            layers=layers,
            final_norm=stored("norm.weight"),
            head_quant=Quant("nvfp4", scale("lm_head")),
            head=copy([stored("lm_head.weight")], "output_head"),
            copy_bytes=self.copy_bytes(copyless),
            stored=kept,
            uncopied=copyless,
            alone=tuple(alone),  # read after the head's weight was made, two lines up
        )

    # ------------------------------------------------------------------ one decode step
    def step(self, call, weights: Weights, state, ids):
        """One decode step: ``ids`` ``[rows]`` int32 on the device -> ``(next ids [rows] int32, logits [rows, vocabulary])``.
        ``call(block, layer, *operands)`` runs one block; a session passes the plan's blocks, a check wraps them."""
        position = state.position
        branch, residual = call("embedding", -1, ids, weights.embedding), None
        for i, layer in enumerate(weights.layers):
            site = layer.kind if residual is not None else layer.kind + FIRST  # the first norm of a step starts the residual stream
            q, normed, residual = call("add_norm_quantize", i, branch, residual, layer.input_norm, layer.in_quant, site)
            row = call("projection.fp8.in", i, q, layer.in_proj, IN_PROJECTION_SITES[layer.kind])
            if layer.kind == "linear_attention":
                x, z = split_qkvz(row, layer.gdn.qkv_channels, layer.gdn.value_heads, layer.gdn.head_dim)
                mixed = call("linear_attention.step", i, x, z, normed, state.slot_ids, layer.gdn, state.layers[i])
            else:
                qh, gate = call("attention.prepare", i, row, state.layers[i], position, layer.attn)
                attended = call("attention.core", i, qh, state.layers[i], position)
                mixed = call("attention.gate_quantize", i, attended, gate, layer.out_quant)
            branch = call("projection.fp8.out", i, mixed, layer.out_proj)
            q, _, residual = call("add_norm_quantize", i, branch, residual, layer.post_norm, layer.mlp_quant, "mlp")
            branch = call("mlp.down", i, call("mlp.gate_up", i, q, layer.gate_up, layer.down_quant), layer.down)
        q, _, _ = call("add_norm_quantize", len(weights.layers), branch, residual, weights.final_norm, weights.head_quant, "output_head")
        logits = call("output_head", -1, q, weights.head)
        next_ids = call("token_choice", -1, logits)
        state.position = position + 1
        return next_ids, logits
