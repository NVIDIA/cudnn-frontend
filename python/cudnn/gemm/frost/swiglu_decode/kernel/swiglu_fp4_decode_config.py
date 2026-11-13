# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the block-scaled FP4 gate / up kernel with a SwiGLU and block-quantising epilogue for decode rows
(``swiglu_fp4_decode_nvfp4.py``): the geometry, the performance config, the token-tile rule, the stage rule's host mirror, the
shared-memory and tensor-memory ledgers, the barrier constants and the validator.  Imports neither torch nor the CuTe DSL, so the host
entry (``swiglu_decode/swiglu_fp4_decode.py``) validates on a host that has neither.

NO NUMERICS KNOB.  The kernel has no split, no reduction across rows and no association choice: an output element is one fp32 chain in
ascending K, and a block scale is a function of its own row.  Every field of ``Cfg`` is therefore performance-only -- which token tile a
row count runs, how many weight stages are issued ahead of the dependent-launch wait, the L2 policy of the weight stream, the cap of the
persistent grid, the launch attribute -- or one of the part's readings, which only decline.  A host test pins that list.

ONE TOKEN TILE PER CALL.  The kernel puts the token rows on the narrow N tile of the tensor-core instruction: 8, 16, 32 or 64 tokens.
A call's rows must fit one tile (:func:`check_rows_fit`); :func:`token_tile_for` picks the narrowest that holds them, not below a floor.

THE STAGE RULE (:func:`stages`).  Two staging stages are reserved first, the operand ring takes what the capacity then holds, the
remainder goes back to staging stages.  Its capacity input is the CONSTANT ``STAGE_RULE_CAPACITY_BYTES`` on every part -- never a
device reading -- so one token tile is one kernel everywhere; the part's readings in ``Cfg`` decide only whether the part can hold it.
The kernel module checks at trace that the traced kernel's stage counts, shared-memory total and transaction bytes equal this mirror.
"""

from dataclasses import dataclass, replace
from typing import Dict, Tuple

NAME = "the SwiGLU FP4 decode kernel"

MAX_ROWS = 64  # the served row ceiling: one token tile per call
TOKEN_TILES = (8, 16, 32, 64)  # token rows on the N tile of the MMA instruction
WEIGHT_TILE_ROWS = 128  # weight rows per CTA tile: 64 gate channels, then the same 64 up channels
CHANNELS_PER_TILE = WEIGHT_TILE_ROWS // 2
SF_VEC = 16  # elements per E4M3 block scale (operands and output)
SF_ATOM_ROWS = 128  # rows of one scale atom band (the 128 x 4 atom order)
MMA_K = 64  # K elements (4 bits each) per tcgen05 block-scaled MMA instruction
K_TILES = 8  # MMA instructions per operand-ring stage: a constant of the kernel, not a knob
K_TILE = MMA_K * K_TILES  # K elements per operand-ring stage
PREWAIT_CHOICES = (0, 2)  # weight stages issued ahead of the dependent-launch wait
ACC_STAGES = 2  # accumulator stages in tensor memory
TMEM_ALLOC_COLS = 512  # the allocation (and the deallocation) of a CTA: the whole part's columns
SF_TMEM_COLS = (WEIGHT_TILE_ROWS // 32) * K_TILES  # tensor-memory columns of one operand's scales (weight: 32; activation: 32)
EPILOGUE_TILE_MAX = 32  # token rows of one epilogue sub-tile

STAGE_RULE_CAPACITY_BYTES = 232448  # the stage rule's input on EVERY part; never a device reading
STAGE_RULE_INITIAL_STAGING = 2
BUFFER_ALIGN_BYTES = 1024
SFB_PAD_BYTES = 512  # activation-scale ring pad below the 64-token tile
MBARRIER_BYTES = 8
TMEM_POINTER_BYTES = 4

# Warp roles and named barriers (the barrier table of the kernel module).  SUM(issuing lanes) == init for every mbarrier.
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
THREADS_PER_CTA = 32 * (len(EPILOGUE_WARPS) + 2)
EPILOGUE_THREADS = 32 * len(EPILOGUE_WARPS)
TMEM_PTR_SYNC_THREADS = 32 * (len(EPILOGUE_WARPS) + 1)  # the MMA warp and the epilogue warps read the allocation's base word
CTA_SYNC_BARRIER = 0
EPILOG_SYNC_BARRIER = 1
TMEM_PTR_SYNC_BARRIER = 2
AB_FULL_ARRIVERS = 1  # one elected lane of the TMA warp, with the stage's transaction bytes
AB_EMPTY_ARRIVERS = 1  # one elected tcgen05.commit of the MMA warp per stage
ACC_FULL_ARRIVERS = 1  # one elected tcgen05.commit of the MMA warp per work tile
ACC_EMPTY_ARRIVERS = len(EPILOGUE_WARPS)  # one elected lane of each epilogue warp per work tile

# Operand alignment the compiled entry assumes (bytes of the base address).
PACKED_ALIGN_BYTES = 32  # weight, activation
FLAT_ALIGN_BYTES = 16  # the scale blobs and the two outputs
SCALAR_ALIGN_BYTES = 4

QUALIFIED_MIN_COMPILER = (13, 4)  # the CUDA compiler of the CuTe DSL the kernel's bytes were qualified with
SM100_SMEM_OPTIN_BYTES, SM100_SMEM_PER_SM_BYTES, SM100_SM_COUNT = 227 * 1024, 228 * 1024, 148  # an SM100-class reference reading

#: the DSL names the kernel module binds at import or trace time, per module: probed before the kernel module is imported, so a DSL
#: that clears the version floor but moved one of them declines by name instead of raising from inside the DSL
DSL_NAMES = (
    ("cutlass.pipeline", ("PipelineTmaUmma", "PipelineUmmaAsync", "CooperativeGroup", "Agent", "make_pipeline_state", "PipelineUserType")),
    ("cutlass.utils", ("StaticPersistentTileScheduler", "PersistentTileSchedulerParams")),
    (
        "cutlass.utils.blackwell_helpers",
        ("make_blockscaled_trivial_tiled_mma", "compute_epilogue_tile_shape", "make_smem_layout_a", "make_smem_layout_b", "make_smem_layout_epi")
        + ("cluster_shape_to_tma_atom_A", "cluster_shape_to_tma_atom_B", "cluster_shape_to_tma_atom_SFB", "get_tmem_load_op", "get_smem_store_op"),
    ),
    (
        "cutlass.utils.blockscaled_layout",
        ("tile_atom_to_shape_SF", "make_smem_layout_sfa", "make_smem_layout_sfb", "make_tmem_layout_sfa", "make_tmem_layout_sfb"),
    ),
    ("cutlass.cute.nvgpu.tcgen05", ("Cp4x32x128bOp", "make_s2t_copy", "get_s2t_smem_desc_tensor", "make_tmem_copy", "CtaGroup", "Field")),
    ("cutlass.cute.nvgpu", ("make_tiled_tma_atom_A", "make_tiled_tma_atom_B")),
    ("cutlass.cute.nvgpu.cpasync", ("tma_partition", "prefetch_descriptor")),
    (
        "cutlass.cute.arch",
        ("alloc_tmem", "dealloc_tmem", "relinquish_tmem_alloc_permit", "retrieve_tmem_ptr", "griddepcontrol_wait", "griddepcontrol_launch_dependents")
        + ("shuffle_sync_bfly", "elect_one", "fence_proxy", "barrier", "mbarrier_init_fence"),
    ),
    ("cutlass.cutlass_dsl", ("dsl_user_op",)),
    # the tree's shim over the DSL names that moved between releases; a DSL on which the shim itself does not import declines by name
    ("cudnn._cutlass_compat", ("LayoutEnum", "SmemAllocator", "get_smem_capacity_in_bytes")),
)


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class Geometry:
    """``(HIDDEN, INTERMEDIATE)`` of one gate / up product: the weight is ``[2 x INTERMEDIATE, HIDDEN]`` FP4 (gate rows and up rows,
    interleaved in tiles of 64 + 64), a row of the activation ``[HIDDEN]`` FP4, a row of the output ``[INTERMEDIATE]`` FP4."""

    HIDDEN: int
    INTERMEDIATE: int

    def __post_init__(self):
        for name in ("HIDDEN", "INTERMEDIATE"):
            v = getattr(self, name)
            if isinstance(v, bool) or not hasattr(v, "__index__") or int(v) < 1:
                raise ValueError(f"{NAME}: Geometry.{name} must be a positive int, got {v!r}")
            object.__setattr__(self, name, int(v))

    @property
    def WEIGHT_ROWS(self) -> int:
        return 2 * self.INTERMEDIATE

    @property
    def WEIGHT_TILES(self) -> int:
        """Work tiles of one call: 128 weight rows each, one token tile."""
        return self.WEIGHT_ROWS // WEIGHT_TILE_ROWS

    @property
    def K_BLOCKS(self) -> int:
        """Operand-ring stages one work tile streams (K elements / ``K_TILE``, rounded up)."""
        return (self.HIDDEN + K_TILE - 1) // K_TILE

    def describe(self) -> str:
        return f"hidden {self.HIDDEN}, intermediate {self.INTERMEDIATE}"


DEFAULT_GEOMETRY = Geometry(5120, 17408)
#: the geometries the kernel was validated on; the host entry declines any other by name
SERVED_GEOMETRIES = (Geometry(5120, 17408), Geometry(5120, 128))


@dataclass(frozen=True)
class Cfg:
    """One compiled form.  Every field is PERFORMANCE-ONLY or a part reading: no field changes an output byte.

    ``TOKEN_TILE``: the token tile of the form; in a constructor config it is a FLOOR (a row count runs the narrowest tile that holds it,
    not below the floor).  ``PREWAIT_STAGES``: weight stages the loader issues ahead of the dependent-launch wait, 2 or 0 (the one axis of
    the kernel body besides the tile and the L2 policy).  ``L2_EVICT_FIRST``: the weight stream is loaded with the evict-first L2 policy.
    ``MAX_CLUSTERS``: cap of the persistent grid, any value >= 1 (above the part's SM count the launch is a second wave).  ``PDL``: the
    programmatic-stream-serialization attribute of the launch (a launch fact; the kernel body is the same).  ``SMEM_OPTIN_BYTES`` /
    ``SMEM_PER_SM_BYTES`` / ``SM_COUNT``: the part's readings, stamped by the host entry; they only decline."""

    TOKEN_TILE: int = TOKEN_TILES[0]
    PREWAIT_STAGES: int = 2
    L2_EVICT_FIRST: bool = True
    MAX_CLUSTERS: int = 148
    PDL: bool = False
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    SM_COUNT: int = SM100_SM_COUNT
    GEOM: Geometry = DEFAULT_GEOMETRY

    def __str__(self) -> str:
        """The form as a compact text: the head of the kernel symbol, so that two forms of one entry never share a symbol."""
        g = self.GEOM
        geometry = f"h{g.HIDDEN}_i{g.INTERMEDIATE}" if isinstance(g, Geometry) else "h0_i0"
        policy = "evf" if self.L2_EVICT_FIRST else "dflt"
        return f"{geometry}_t{self.TOKEN_TILE}_pw{self.PREWAIT_STAGES}_{policy}_c{self.MAX_CLUSTERS}_{'pdl' if self.PDL else 'nopdl'}"


#: the performance fields of ``Cfg`` and the part readings: together they are every field (a host test pins it)
PERFORMANCE_FIELDS = ("TOKEN_TILE", "PREWAIT_STAGES", "L2_EVICT_FIRST", "MAX_CLUSTERS", "PDL")
READING_FIELDS = ("SMEM_OPTIN_BYTES", "SMEM_PER_SM_BYTES", "SM_COUNT")


def token_tile_for(rows: int, floor: int = TOKEN_TILES[0]) -> int:
    """The token tile of a row count: the narrowest of 8 / 16 / 32 / 64 that holds it, not below ``floor``.  ``ValueError`` for a
    malformed row count (not an ``int``, a ``bool``, below one) or floor, ``NotImplementedError`` past the served ceiling."""
    if not _is_int(rows) or rows < 1:
        raise ValueError(f"{NAME}: rows must be a positive int, got {rows!r}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call (decode-time row counts); got {rows}")
    if not _is_int(floor) or floor not in TOKEN_TILES:
        raise ValueError(f"{NAME}: Cfg.TOKEN_TILE must be one of {TOKEN_TILES}, got {floor!r}")
    return max(floor, next(t for t in TOKEN_TILES if rows <= t))


def check_rows_fit(cfg: Cfg, rows: int) -> int:
    """``rows`` when the form's token tile holds them.  The kernel runs ONE token tile per call: a second one is not a served path."""
    if not _is_int(rows) or rows < 1:
        raise ValueError(f"{NAME}: rows must be a positive int, got {rows!r}")
    if rows > cfg.TOKEN_TILE:
        raise ValueError(f"{NAME}: rows {rows} exceed the token tile {cfg.TOKEN_TILE}; the kernel runs one token tile per call")
    return rows


def form_of(cfg: Cfg, token_tile: int) -> Cfg:
    """``cfg`` as the form of one token tile (every other field kept)."""
    return replace(cfg, TOKEN_TILE=token_tile)


def epilogue_tile(cfg: Cfg) -> int:
    """Token rows of one epilogue sub-tile: the token tile up to 32, two sub-tiles of 32 at the 64-token tile."""
    return min(cfg.TOKEN_TILE, EPILOGUE_TILE_MAX)


def tx_bytes(cfg: Cfg) -> int:
    """Bytes one operand-ring stage receives (the transaction count of its full barrier): the weight tile, its scales, the activation
    tile and its scale atom."""
    weight = WEIGHT_TILE_ROWS * K_TILE // 2
    activation = cfg.TOKEN_TILE * K_TILE // 2
    weight_scales = WEIGHT_TILE_ROWS * K_TILE // SF_VEC
    activation_scales = SF_ATOM_ROWS * K_TILE // SF_VEC
    return weight + activation + weight_scales + activation_scales


def staging_stage_bytes(cfg: Cfg) -> int:
    """Bytes of one staging stage: 128 weight rows x one epilogue sub-tile of bf16."""
    return WEIGHT_TILE_ROWS * epilogue_tile(cfg) * 2


def sfb_pad_bytes(cfg: Cfg) -> int:
    return SFB_PAD_BYTES if cfg.TOKEN_TILE < 64 else 0


def stages(cfg: Cfg) -> Tuple[int, int]:
    """``(operand stages, staging stages)`` of a form by the stage rule on ``STAGE_RULE_CAPACITY_BYTES``."""
    ab, c = tx_bytes(cfg), staging_stage_bytes(cfg)
    reserved = BUFFER_ALIGN_BYTES + sfb_pad_bytes(cfg) + STAGE_RULE_INITIAL_STAGING * c
    operand = (STAGE_RULE_CAPACITY_BYTES - reserved) // ab
    staging = STAGE_RULE_INITIAL_STAGING + (STAGE_RULE_CAPACITY_BYTES - operand * ab - reserved) // c
    return operand, staging


def smem_table(cfg: Cfg) -> Tuple[Tuple[str, int, int], ...]:
    """The shared-memory buffers of a form in allocation order: ``(name, offset, bytes)``.  The kernel's own allocation is the source;
    this mirror is what the host validates against a part and what the kernel module compares with the traced kernel."""
    operand, staging = stages(cfg)
    rows = []
    offset = 0
    for name, size, align in (
        ("ab_full_mbar", operand * MBARRIER_BYTES, MBARRIER_BYTES),
        ("ab_empty_mbar", operand * MBARRIER_BYTES, MBARRIER_BYTES),
        ("acc_full_mbar", ACC_STAGES * MBARRIER_BYTES, MBARRIER_BYTES),
        ("acc_empty_mbar", ACC_STAGES * MBARRIER_BYTES, MBARRIER_BYTES),
        ("tmem_dealloc_mbar", MBARRIER_BYTES, MBARRIER_BYTES),
        ("tmem_holding_buf", TMEM_POINTER_BYTES, TMEM_POINTER_BYTES),
        ("sC", staging * staging_stage_bytes(cfg), BUFFER_ALIGN_BYTES),
        ("sA", operand * WEIGHT_TILE_ROWS * K_TILE // 2, BUFFER_ALIGN_BYTES),
        ("sB", operand * cfg.TOKEN_TILE * K_TILE // 2, BUFFER_ALIGN_BYTES),
        ("sSFA", operand * WEIGHT_TILE_ROWS * K_TILE // SF_VEC, BUFFER_ALIGN_BYTES),
        ("sSFB", operand * SF_ATOM_ROWS * K_TILE // SF_VEC + sfb_pad_bytes(cfg), BUFFER_ALIGN_BYTES),
    ):
        offset = _align_up(offset, align)
        rows.append((name, offset, size))
        offset += size
    return tuple(rows)


def smem_bytes(cfg: Cfg) -> int:
    """Shared memory one CTA of a form holds = what the launch reserves (the struct's size, rounded to its alignment)."""
    name, offset, size = smem_table(cfg)[-1]
    return _align_up(offset + size, BUFFER_ALIGN_BYTES)


def tmem_cols_used(cfg: Cfg) -> int:
    """Tensor-memory columns a form uses of the ``TMEM_ALLOC_COLS`` it allocates: two accumulator stages of one token tile, the weight
    scales and the activation scales of one stage."""
    return cfg.TOKEN_TILE * ACC_STAGES + 2 * SF_TMEM_COLS


def grid(cfg: Cfg) -> Tuple[int, int, int]:
    """The launch grid: persistent CTAs along z, one per work tile up to ``MAX_CLUSTERS``."""
    return (1, 1, min(cfg.GEOM.WEIGHT_TILES, cfg.MAX_CLUSTERS))


def barrier_ledger() -> Dict[str, Dict[str, int]]:
    """Init count and issuing lanes per phase of every mbarrier ring, and participants and reaching threads of every named barrier."""
    return {
        "ab_full": dict(init=AB_FULL_ARRIVERS, issuing_lanes=1),
        "ab_empty": dict(init=AB_EMPTY_ARRIVERS, issuing_lanes=1),
        "acc_full": dict(init=ACC_FULL_ARRIVERS, issuing_lanes=1),
        "acc_empty": dict(init=ACC_EMPTY_ARRIVERS, issuing_lanes=len(EPILOGUE_WARPS) * 1),
        "cta_sync": dict(init=THREADS_PER_CTA, issuing_lanes=32 * (len(EPILOGUE_WARPS) + 2)),
        "tmem_ptr_sync": dict(init=TMEM_PTR_SYNC_THREADS, issuing_lanes=32 * (len(EPILOGUE_WARPS) + 1)),
        "epilog_sync": dict(init=EPILOGUE_THREADS, issuing_lanes=32 * len(EPILOGUE_WARPS)),
    }


def check_geometry(geometry) -> Geometry:
    """``geometry`` as a served ``Geometry``: ``TypeError`` for another type, ``NotImplementedError`` (by name) for one not validated."""
    if not isinstance(geometry, Geometry):
        raise TypeError(f"{NAME}: geometry must be a swiglu_fp4_decode_config.Geometry, got {type(geometry).__name__}")
    if geometry not in SERVED_GEOMETRIES:
        raise NotImplementedError(f"{NAME} serves {' or '.join(g.describe() for g in SERVED_GEOMETRIES)} (validated); got {geometry.describe()}")
    return geometry


def part_readings(smem_per_sm: int, smem_optin: int, sm_count: int) -> Dict[str, int]:
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in, multiprocessors)``, validated."""
    for name, value in (("smem_per_sm", smem_per_sm), ("smem_optin", smem_optin), ("sm_count", sm_count)):
        if not _is_int(value) or value <= 0:
            raise ValueError(f"{NAME}: {name} must be a positive int (a device reading), got {value!r}")
    if smem_optin > smem_per_sm:
        raise ValueError(f"{NAME}: the per-CTA shared-memory opt-in ({smem_optin} B) cannot exceed the SM's ({smem_per_sm} B)")
    return dict(SMEM_PER_SM_BYTES=smem_per_sm, SMEM_OPTIN_BYTES=smem_optin, SM_COUNT=sm_count)


def validate_cfg(cfg: Cfg) -> None:
    """Structural validation of one form.  ``TypeError`` for a config that is not a ``Cfg``; ``ValueError`` for a knob the kernel
    cannot run (the message names the knob and its domain) and for a part reading below the form's footprint (the host entry turns the
    latter into its part decline).  The served-geometry and row-count declines are :func:`check_geometry` / :func:`token_tile_for`."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a swiglu_fp4_decode_config.Cfg, got {type(cfg).__name__}")
    if not isinstance(cfg.GEOM, Geometry):
        raise TypeError(f"{NAME}: Cfg.GEOM must be a swiglu_fp4_decode_config.Geometry, got {type(cfg.GEOM).__name__}")
    g = cfg.GEOM
    if g.WEIGHT_ROWS % WEIGHT_TILE_ROWS or g.HIDDEN % MMA_K:
        raise ValueError(
            f"{NAME}: the weight is cut into tiles of {WEIGHT_TILE_ROWS} rows ({CHANNELS_PER_TILE} gate + {CHANNELS_PER_TILE} up channels) and K "
            f"into instructions of {MMA_K}; got {g.describe()}"
        )
    if not _is_int(cfg.TOKEN_TILE) or cfg.TOKEN_TILE not in TOKEN_TILES:
        raise ValueError(f"{NAME}: Cfg.TOKEN_TILE must be one of {TOKEN_TILES}, got {cfg.TOKEN_TILE!r}")
    operand, staging = stages(cfg)
    if not _is_int(cfg.PREWAIT_STAGES) or cfg.PREWAIT_STAGES not in PREWAIT_CHOICES or cfg.PREWAIT_STAGES > operand:
        raise ValueError(
            f"{NAME}: Cfg.PREWAIT_STAGES must be one of {PREWAIT_CHOICES} and at most the form's {operand} operand stages, got {cfg.PREWAIT_STAGES!r}"
        )
    for name in ("L2_EVICT_FIRST", "PDL"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{NAME}: Cfg.{name} must be a bool, got {getattr(cfg, name)!r}")
    if not _is_int(cfg.MAX_CLUSTERS) or cfg.MAX_CLUSTERS < 1:
        raise ValueError(f"{NAME}: Cfg.MAX_CLUSTERS must be an int >= 1 (the cap of the persistent grid), got {cfg.MAX_CLUSTERS!r}")
    part_readings(cfg.SMEM_PER_SM_BYTES, cfg.SMEM_OPTIN_BYTES, cfg.SM_COUNT)
    if operand < 1 or staging < STAGE_RULE_INITIAL_STAGING:
        raise ValueError(f"{NAME}: the stage rule leaves {operand} operand stages and {staging} staging stages at token tile {cfg.TOKEN_TILE}")
    if tmem_cols_used(cfg) > TMEM_ALLOC_COLS:
        raise ValueError(f"{NAME}: token tile {cfg.TOKEN_TILE} needs {tmem_cols_used(cfg)} tensor-memory columns of the {TMEM_ALLOC_COLS} a CTA allocates")
    footprint = smem_bytes(cfg)
    if footprint > cfg.SMEM_OPTIN_BYTES:
        raise ValueError(
            f"{NAME}: the form at token tile {cfg.TOKEN_TILE} holds {footprint} B of shared memory per thread block; "
            f"the part's per-block maximum is {cfg.SMEM_OPTIN_BYTES} B"
        )
