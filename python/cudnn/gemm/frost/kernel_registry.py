# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-template registry — six-dimension support funnel.

Whether a kernel runs depends on: gpu arch · kernel template · tile config
(geometry) · graph type · mma type · other graph info. Tile configs are PURE
GEOMETRY; the template supplies execution strategy (cta_group,
pipeline, graph type, mainloop), so one geometry runs on several templates.

:meth:`KernelTemplate.accepts` funnels cheapest-first:
  1. graph type + pipeline-family + mainloop axis
  2. mma type per TEMPLATE FAMILY (:data:`MMA_TYPE_SUPPORT` existence sets) +
     the rare family×combo×GPU exceptions (:data:`MMA_GPU_ARCH_SPECIAL_CASES`)
  3. tile config — accepted geometries by PREDICATE over CATALOG (cta_group
     constraints), never a hand-maintained list
  4. other graph info — mainloop op scope + TMA alignment

Capability checks are reused from ``compiler._check_*`` (lazy import; compiler
doesn't import this module). :func:`select_template` drives the compiler's
template-file selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .fusion_ir import BINARY_OPS, UNARY_OPS, FusionChain
from .tile_config import CATALOG, TileConfig, config_class_for_pipeline


def _pipeline_from_file(template_file: str) -> str:
    """Leading ``sm<NNN>`` pipeline-family token of a template filename — the
    template pairs with the config family of the same pipeline (config_sm<NNN>)
    and runs on that family's SM ranges (:data:`PIPELINE_ARCH_RANGES`)."""
    m = re.match(r"(sm\d+)_", template_file)
    if m is None:
        raise ValueError(f"cannot determine pipeline family from template {template_file!r} " f"(expected a leading 'sm<NNN>_' token)")
    return m.group(1)


# Active-GPU SM ranges per template pipeline — half-open [lo, hi) segments,
# SM = major*10 + minor. A family may support several DISJOINT segments
PIPELINE_ARCH_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "sm100": ((100, 120),),
    "sm103": ((103, 110),),
    "sm107": ((107, 110),),
}

# Pointwise ops a mainloop-fusion template can transform in SMEM.
_SUPPORTED_MAINLOOP_OPS: frozenset[str] = frozenset(UNARY_OPS) | frozenset(BINARY_OPS)


# Dimension 1: graph type


class GraphType(Enum):
    """The kind of computation the graph expresses; each template supports
    exactly one. ``CONVOLUTION`` is a placeholder (no template yet)."""

    MATMUL = "matmul"
    BLOCK_SCALE_MATMUL = "block_scale_matmul"
    MOE = "moe"
    MOE_BLOCK_SCALE = "moe_block_scale"  # MoE grouped matmul, block-scaled inputs
    CONVOLUTION = "convolution"  # placeholder — no template yet


def classify_graph_type(chain: FusionChain) -> GraphType:
    """Stage-1 classifier: which graph type this chain is."""
    if chain.has_moe and chain.has_block_scale:
        return GraphType.MOE_BLOCK_SCALE
    if chain.has_block_scale:
        return GraphType.BLOCK_SCALE_MATMUL
    if chain.has_moe:
        return GraphType.MOE
    return GraphType.MATMUL


# Dimension 2: mma type × arch (graph-type-level, config/template independent).
# Unified MMA-type × arch support — SINGLE source of truth (merges the old
# compiler matmul table + block-scale per-side case list). Indexed by graph
# type; value = {mma_type_key -> supported arch ranges}. Key shape differs per
# graph type (matmul = (a, b, accum); block-scale = full per-side spec) but the
# lookup is one code path.


def _matmul_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    return (mm.a_dtype, mm.b_dtype, mm.accum_dtype)


def _block_scale_mma_type(chain: FusionChain) -> tuple:
    mm = chain.matmul
    bs = chain.block_scale
    assert bs is not None
    return (
        mm.a_dtype,
        bs.sf_dtype_a,
        bs.block_size_a,
        bs.sfa_reorder,
        bs.dequant_compute_a,
        bs.dequant_out_a,
        mm.b_dtype,
        bs.sf_dtype_b,
        bs.block_size_b,
        bs.sfb_reorder,
        bs.dequant_compute_b,
        bs.dequant_out_b,
        mm.accum_dtype,
    )


def _bs_key(a: str, sfa: str, b: str, sfb: str, kblk: int) -> tuple:
    """Block-scale mma-type key from the varying parts (data + SF dtypes,
    K-block). All cases share reorder F8_128x4, fp32 dequant compute/out, fp32
    accumulate, A block=(1,kblk) / B block=(kblk,1)."""
    return (
        a,
        sfa,
        (1, kblk),
        "F8_128x4",
        "fp32",
        "fp32",
        b,
        sfb,
        (kblk, 1),
        "F8_128x4",
        "fp32",
        "fp32",
        "fp32",
    )


_BLOCK_SCALE_CASES = frozenset(
    {
        _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16),
        _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16),
        _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
        _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16),
        _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32),
        _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
        _bs_key("fp8_e4m3", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
        _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e4m3", "fp8_e8m0", 32),
        _bs_key("fp8_e5m2", "fp8_e8m0", "fp8_e5m2", "fp8_e8m0", 32),
    }
)

_MATMUL_CASES = frozenset(
    {
        ("bf16", "bf16", "fp32"),
        ("fp16", "fp16", "fp32"),
        ("int8", "int8", "int32"),
        ("fp8_e4m3", "fp8_e4m3", "fp32"),
        ("fp8_e4m3", "fp8_e5m2", "fp32"),
        ("fp8_e5m2", "fp8_e4m3", "fp32"),
        ("fp8_e5m2", "fp8_e5m2", "fp32"),
    }
)

# MoE grouped matmul reuses its base pipeline's MMA machinery verbatim — combo
# lookups fold to the base graph type, so MMA_TYPE_SUPPORT never carries MoE rows.
_MMA_BASE_GRAPH_TYPE: dict[GraphType, GraphType] = {
    GraphType.MOE: GraphType.MATMUL,
    GraphType.MOE_BLOCK_SCALE: GraphType.BLOCK_SCALE_MATMUL,
}

# Per-graph-type mma-type key extractors (key SHAPES differ per graph type, so
# the key spaces never collide).
_MMA_KEY_FNS: dict[GraphType, object] = {
    GraphType.MATMUL: _matmul_mma_type,
    GraphType.BLOCK_SCALE_MATMUL: _block_scale_mma_type,
}

# {template_pipeline: {GraphType: frozenset of supported mma-type keys}} — which
# dtype combos each TEMPLATE FAMILY implements, arch-FREE (whether the active
# GPU can execute a supported combo is the family's stage-0 SM gate, except
# the rare MMA_GPU_ARCH_SPECIAL_CASES below). A new family (e.g. an sm120
# mxfp8 pipeline) adds its own entry; it never inherits another family's set.
MMA_TYPE_SUPPORT: dict[str, dict[GraphType, frozenset]] = {
    "sm100": {
        GraphType.MATMUL: _MATMUL_CASES,
        GraphType.BLOCK_SCALE_MATMUL: _BLOCK_SCALE_CASES,
    },
    "sm103": {
        GraphType.BLOCK_SCALE_MATMUL: frozenset(
            {
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 16),
                _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 16),
                _bs_key("fp4_e2m1", "fp8_e8m0", "fp4_e2m1", "fp8_e8m0", 32),
                _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16),
                _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32),
            }
        ),
    },
    "sm107": {
        GraphType.BLOCK_SCALE_MATMUL: _BLOCK_SCALE_CASES,
    },
}

# The ONE home for checks that need template SM family × mma dtype × ACTUAL
# GPU arch together: (family, combo) pairs whose MMA instruction exists only
# on specific family members — NARROWER than the family's SM ranges. Most
# combos never appear here (stage-0 family gate + the existence sets decide).
# Values are half-open [lo, hi) segments, same shape as PIPELINE_ARCH_RANGES.
MMA_GPU_ARCH_SPECIAL_CASES: dict[tuple[str, tuple], tuple[tuple[int, int], ...]] = {
    ("sm100", ("int8", "int8", "int32")): ((100, 101), (110, 111)),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16)): ((107, 110),),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32)): ((107, 110),),
    ("sm100", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32)): ((107, 110),),
    ("sm103", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((107, 110),),
    ("sm107", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 16)): ((107, 110),),
    ("sm107", _bs_key("fp4_e2m1", "fp8_e5m3", "fp4_e2m1", "fp8_e5m3", 32)): ((107, 110),),
    ("sm107", _bs_key("fp4_e2m1", "fp8_e4m3", "fp4_e2m1", "fp8_e4m3", 32)): ((107, 110),),
}


def mma_arch_reject(chain: FusionChain, graph_type: GraphType, template_pipeline: str) -> str | None:
    """Stage 2: does the ``template_pipeline`` family's pipeline support the
    graph's MMA type, and — for the rare :data:`MMA_GPU_ARCH_SPECIAL_CASES`
    (family, combo) pairs — does the ACTUAL GPU have the instruction?
    ``None`` = yes. Independent of tile config / cta_group."""
    from . import compiler as C

    base_type = _MMA_BASE_GRAPH_TYPE.get(graph_type, graph_type)
    key_fn = _MMA_KEY_FNS.get(base_type)
    if key_fn is None:
        return f"graph type {graph_type.value!r} has no kernel pipeline yet"
    cases = MMA_TYPE_SUPPORT.get(template_pipeline, {}).get(base_type)
    if cases is None:
        return f"the {template_pipeline} family has no {graph_type.value} pipeline"
    key = key_fn(chain)
    if key not in cases:
        if base_type is GraphType.MATMUL:
            mm = chain.matmul
            return f"the {template_pipeline} {graph_type.value} pipeline does not support " f"input/acc dtype combo {mm.a_dtype}x{mm.b_dtype}->{mm.accum_dtype}"
        return f"the {template_pipeline} {graph_type.value} pipeline does not support " f"this configuration: mma type {key}"
    special = MMA_GPU_ARCH_SPECIAL_CASES.get((template_pipeline, key))
    if special is not None:
        arch = C._current_arch()
        if arch is not None and not any(lo <= arch < hi for lo, hi in special):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in special)
            return f"the {template_pipeline} {key[0]}x{key[1]} MMA instruction exists only " f"on {spans}, but the active GPU is sm_{arch}"
    return None


# Kernel template — owns the execution-strategy axes (cta_group / pipeline)


@dataclass(frozen=True)
class KernelTemplate:
    """One kernel template. Carries the execution-strategy axes the pure-geometry
    config does NOT (pipeline, cta_group, graph_type, mainloop).
    ``accepts`` runs the funnel using ``self.cta_group`` for cta_group gates."""

    file: str  # template filename under kernel_templates/
    pipeline: str  # pipeline family from the filename; pairs with config_<pipeline>
    cta_group: int  # 1 or 2 (1-CTA vs 2-CTA MMA)
    graph_type: GraphType  # the single graph type this template supports
    mainloop: bool  # mainloop-fusion variant (transform A/B before MMA)
    # Multi-GEMM support (templates without it reject multi-GEMM chains).
    supports_multi_gemm: bool = True
    # A CTA tile spanning several MMA instructions along M (num_mma_m > 1).
    supports_multi_mma_m: bool = True

    @property
    def block_scale(self) -> bool:
        """True iff this template consumes block-scaled (FP4/FP8 + SF) inputs."""
        return self.graph_type in (
            GraphType.BLOCK_SCALE_MATMUL,
            GraphType.MOE_BLOCK_SCALE,
        )

    # stage 0: active-GPU SM ranges (from the template's pipeline-family prefix)

    def arch_active_reject(self) -> str | None:
        """``None`` if the active GPU's SM is in one of the template pipeline
        family's ranges (:data:`PIPELINE_ARCH_RANGES`), or no GPU is visible
        (render-only / CI). Per-ARCH-FAMILY gate (vs :func:`mma_arch_reject`,
        which gates the graph's MMA type)."""
        from . import compiler as C

        arch = C._current_arch()
        ranges = PIPELINE_ARCH_RANGES[self.pipeline]
        if arch is not None and not any(lo <= arch < hi for lo, hi in ranges):
            spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)
            return f"template {self.file} runs only on {spans}, " f"but the active GPU is sm_{arch}"
        return None

    # stage 1: pipeline / graph-type / mainloop axes

    def _axis_reject(self, chain: FusionChain, config: TileConfig, graph_type: GraphType) -> str | None:
        cfg_cls = config_class_for_pipeline(self.pipeline)
        if not isinstance(config, cfg_cls):
            return f"config {config.name} is not a {cfg_cls.__name__} " f"(template {self.file} pairs with config_{self.pipeline})"
        if graph_type is not self.graph_type:
            return f"graph_type {graph_type.value} != " f"template graph_type {self.graph_type.value}"
        if chain.has_mainloop_fusion != self.mainloop:
            return f"graph mainloop_fusion={chain.has_mainloop_fusion} != " f"template mainloop={self.mainloop}"
        if chain.is_multi_gemm and not self.supports_multi_gemm:
            return f"template {self.file} does not support multi-GEMM " f"({chain.num_gemms} parallel GEMMs); only the 1ctamma CLC " "template does this pass"
        return None

    # stage 3: tile-config gates (this template's cta_group on the geometry)

    def _config_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        from . import compiler as C

        # cta_group constraints (the template's, not the config's).
        try:
            C._check_cta_group_geometry(config, self.cta_group)
            C._check_mma_n_dim(chain, config, self.cta_group)
        except NotImplementedError as e:
            return str(e)
        reason = self.multi_mma_m_reject(config)
        if reason is not None:
            return reason
        try:
            if self.block_scale:
                from .tile_config import validate_block_scale_config

                bs = chain.block_scale
                assert bs is not None
                data_elem_bits = 4 if bs.is_fp4 else 8
                cta_k_elems = config.cta_tile_k_bytes * 8 // data_elem_bits
                validate_block_scale_config(config, bs.block_size, cta_k_elems)
            else:
                C._check_dtype_config_compat(chain, config, self.cta_group)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # stage 4a: per-template capability hook

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        """Template-SPECIFIC constraints beyond the shared gates (base = none;
        mainloop overrides). Future variable-MMA / per-template caps attach here."""
        return None

    # stage 4: other graph info

    def _other_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        extra = self._extra_reject(chain, config)
        if extra is not None:
            return extra
        from . import compiler as C

        try:
            C._check_input_alignment(chain)
            if not self.block_scale:
                # Block-scale skips only the OUTPUT vec-bytes gate here (its
                # jit path applies it with block-scale-specific handling); the
                # TMA input-alignment rule above is universal.
                C._compute_output_vec_bytes(chain)
        except (ValueError, NotImplementedError) as e:
            return str(e)
        return None

    # full accept/reject: the four-stage funnel

    def accepts(self, chain: FusionChain, config: TileConfig) -> str | None:
        """``None`` if this template can compile (chain, config); else the first
        stage's rejection reason. Cheapest-first (short-circuits):
        pipeline/graph-type/mainloop → mma-type×arch → tile-config → other."""
        gt = classify_graph_type(chain)
        return (
            self.arch_active_reject()
            or self._axis_reject(chain, config, gt)
            or mma_arch_reject(chain, gt, self.pipeline)
            or self._config_reject(chain, config)
            or self._other_reject(chain, config)
        )

    def multi_mma_m_reject(self, config: TileConfig) -> str | None:
        """``None`` unless the config splits the CTA tile across several MMA
        instructions along M and this template has not been adapted to it."""
        if config.num_mma_m > 1 and not self.supports_multi_mma_m:
            return (
                f"{self.file} does not support a CTA tile spanning several MMA "
                f"instructions along M (num_mma_m={config.num_mma_m}); use a "
                f"cta_tile_m of {config.mma_inst_m}"
            )
        return None

    def active_reject(self, config: TileConfig) -> str | None:
        """The gates a JIT path applies once it has picked this template: the
        active GPU's SM range, then capabilities a pure-geometry config can ask
        for that this template does not implement."""
        return self.arch_active_reject() or self.multi_mma_m_reject(config)

    def candidate_configs(self, chain: FusionChain) -> tuple[TileConfig, ...]:
        """Catalog geometries this template accepts for ``chain`` — by
        predicate filter, never hand-maintained."""
        return tuple(c for c in CATALOG if self.accepts(chain, c) is None)


class MainloopKernelTemplate(KernelTemplate):
    """A mainloop-fusion template. Every pre-MMA op must be in
    :data:`_SUPPORTED_MAINLOOP_OPS` (can't be tripped by constructible input
    today, but keeps the contract on the template)."""

    def _extra_reject(self, chain: FusionChain, config: TileConfig) -> str | None:
        for side, ops in (("A", chain.mainloop_a_ops), ("B", chain.mainloop_b_ops)):
            for op in ops:
                if op.op not in _SUPPORTED_MAINLOOP_OPS:
                    return f"{self.file} cannot fuse mainloop op {op.op!r} on " f"operand {side} (supported: unary + scalar-aux binary)"
        return None


# Registry — one entry per template file (20 today). A geometry config expands
# across these via `candidates`. cta_group / mainloop live HERE.


def _mm(
    file: str,
    *,
    cta_group: int,
    mainloop: bool = False,
    graph_type: GraphType = GraphType.MATMUL,
    supports_multi_gemm: bool = True,
    supports_multi_mma_m: bool = True,
) -> KernelTemplate:
    pipeline = _pipeline_from_file(file)
    if pipeline not in PIPELINE_ARCH_RANGES:
        raise KeyError(
            f"template {file!r}: pipeline family {pipeline!r} has no SM-range entry in " f"PIPELINE_ARCH_RANGES — add one when introducing a new family"
        )
    cls = MainloopKernelTemplate if mainloop else KernelTemplate
    return cls(
        file=file,
        pipeline=pipeline,
        cta_group=cta_group,
        graph_type=graph_type,
        mainloop=mainloop,
        supports_multi_gemm=supports_multi_gemm,
        supports_multi_mma_m=supports_multi_mma_m,
    )


TEMPLATES: tuple[KernelTemplate, ...] = (
    # plain matmul
    _mm("sm100_matmul_1ctamma.py", cta_group=1),
    _mm("sm100_matmul_2ctamma.py", cta_group=2),
    # block-scaled matmul
    _mm(
        "sm100_block_scale_matmul_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
    ),
    _mm(
        "sm100_block_scale_matmul_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
    ),
    # sm103 block-scaled matmul: fp4-only (nvfp4/mxfp4), K=48B UTCOMMA
    # (K-tile 384 B, 8 MMAs over 3× 128-B chunks via circular SMEM descs).
    # num_mma_m > 1 is NOT adapted here: the chunk pipeline miscomputes (A reads
    # unwritten SMEM in K, period 192 B) and the ab_stages budget under-counts,
    # so cta_tile_m=256 also overruns the SMEM cap. Both are silent-wrong /
    # launch-fail, hence the gate. See CLAUDE.md for what was ruled out.
    # Multi-GEMM has never been validated on this pipeline either — same gate.
    _mm(
        "sm103_block_scale_matmul_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
        supports_multi_gemm=False,
        supports_multi_mma_m=False,
    ),
    _mm(
        "sm103_block_scale_matmul_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
        supports_multi_gemm=False,
        supports_multi_mma_m=False,
    ),
    _mm(
        "sm107_block_scale_matmul_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
    ),
    _mm(
        "sm107_block_scale_matmul_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.BLOCK_SCALE_MATMUL,
    ),
    # mainloop-fusion matmul (no block-scale variant yet)
    # The mainloop templates have no per-GEMM operand indexing (no gemm_a_idx /
    # gemm_b_idx in the MMA warp), so a second GEMM's accumulator would never be
    # written. The analyzer also only detects mainloop at len(matmuls) == 1.
    _mm("sm100_matmul_mainloop_1ctamma.py", cta_group=1, mainloop=True, supports_multi_gemm=False),
    _mm("sm100_matmul_mainloop_2ctamma.py", cta_group=2, mainloop=True, supports_multi_gemm=False),
    # MoE grouped matmul fwd (own grouped persistent scheduler).
    _mm(
        "sm100_moe_grouped_matmul_fwd_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.MOE,
    ),
    _mm(
        "sm100_moe_grouped_matmul_fwd_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.MOE,
    ),
    # MoE grouped matmul with block-scaled (FP4/FP8 + SF) inputs.
    _mm(
        "sm100_moe_grouped_block_scale_matmul_fwd_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.MOE_BLOCK_SCALE,
    ),
    _mm(
        "sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.MOE_BLOCK_SCALE,
    ),
    _mm(
        "sm107_moe_grouped_block_scale_matmul_fwd_1ctamma.py",
        cta_group=1,
        graph_type=GraphType.MOE_BLOCK_SCALE,
    ),
    _mm(
        "sm107_moe_grouped_block_scale_matmul_fwd_2ctamma.py",
        cta_group=2,
        graph_type=GraphType.MOE_BLOCK_SCALE,
    ),
)


# Pipeline families the AUTO path (``tile_config.select_config``) may build
# with, best first. sm103 is deliberately absent: its 384-byte K-tile is outside
# select_config's geometry ladder, so it stays an explicit-config pipeline.
_AUTO_PIPELINE_ORDER: tuple[str, ...] = ("sm107", "sm100")


def preferred_pipeline(chain: FusionChain) -> str:
    """Pipeline family the auto path should build ``chain`` with: the first
    :data:`_AUTO_PIPELINE_ORDER` entry that has a template for this graph type
    and whose SM range covers the active GPU. A graph type the newer family
    does not implement (plain matmul, MoE) falls through to sm100 by itself."""
    gt = classify_graph_type(chain)
    for pipeline in _AUTO_PIPELINE_ORDER:
        if any(t.pipeline == pipeline and t.graph_type is gt and t.arch_active_reject() is None for t in TEMPLATES):
            return pipeline
    return _AUTO_PIPELINE_ORDER[-1]


def select_template(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
) -> KernelTemplate:
    """The single template that renders (chain, config) under the requested
    execution strategy (``cta_group``; mainloop + graph_type derived from the
    chain). Capability gates are NOT applied — this renders even unsupported
    configs for deliberate single-point probing."""
    gt = classify_graph_type(chain)
    matches = [
        t
        for t in TEMPLATES
        if isinstance(config, config_class_for_pipeline(t.pipeline))
        and t.graph_type is gt
        and t.mainloop == chain.has_mainloop_fusion
        and t.cta_group == cta_group
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no kernel template for graph_type={gt.value}, "
            f"mainloop={chain.has_mainloop_fusion}, cta_group={cta_group}, "
            f"pipeline={config.pipeline}. "
            "E.g. mainloop fusion has no block-scale template variant yet."
        )
    raise ValueError(f"ambiguous template match (registry bug): {[t.file for t in matches]}")


def candidates(chain: FusionChain) -> list[tuple[KernelTemplate, TileConfig]]:
    """Traversal-mode candidate set for ``chain`` via the funnel. Each accepted
    (template, geometry) is a JIT-able point; one geometry expands across the
    templates that accept it ({1,2}ctamma, etc.)."""
    gt = classify_graph_type(chain)
    tmpls = [t for t in TEMPLATES if t.graph_type is gt]  # stage 1
    if not tmpls:
        return []
    out: list[tuple[KernelTemplate, TileConfig]] = []
    for tmpl in tmpls:
        for cfg in tmpl.candidate_configs(chain):  # stages 0 + 2 + 3 + 4
            out.append((tmpl, cfg))
    return out
