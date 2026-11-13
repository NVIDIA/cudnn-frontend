# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the FP8 split-K projection kernel for decode rows (``fp8_projection_decode_e4m3.py``): the geometry, the numerics / form split,
the measured picks, the shared-memory and tensor-memory ledgers and the validator.  Imports neither torch nor the CuTe DSL, so the host
entry (``projection_decode/fp8_projection_decode.py``) validates and picks on a host that has neither.

TWO KINDS OF KNOBS, kept in two types so that one can never reach the other:

* ``Numerics(GEOM, SPLIT_K)`` -- what the BYTES depend on.  ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` with fp32
  accumulation: the K axis is cut into ``SPLIT_K`` equal slices (one CTA of a cluster each), every slice is one tensor-core chain over its
  128-element K tiles in ascending K, the slice sums are added in slice order, the sum is scaled and cast once.  ``SPLIT_K`` therefore
  changes the fp32 association: it is fixed for the life of a host entry and is never a function of the row count.
* ``Form(MMA_M, STAGES, TMEM_COLS, TOKEN_TILE, PREWAIT_WEIGHTS, HOLD_TMEM)`` -- performance only: the weight rows per CTA, the depth of
  the operand ring, the tensor-memory columns a CTA allocates, the token tile, whether the weight stream runs ahead of the
  dependent-launch wait and when the epilogue releases the tensor memory.  None of them enters a sum.  :func:`pick` returns a ``Form``
  and nothing else, so a pick cannot change the split.

``Cfg`` is one compiled rendering: a ``Numerics``, a ``Form`` resolved for one row count, the launch knobs and the part's readings.

TWO KINDS OF SERVED CLASSES, by how their pick was obtained.  The classes of ``ONE_WAVE_CLASSES`` have a measured table and the rule
that goes with it: the ring is lowered until the grid fits one wave on the part.  Every other served class runs its table row AS WRITTEN
-- the row was not measured with this entry, the ring is lowered only until a CTA fits the part's per-CTA shared memory, and the grid may
take more than one wave -- and its pick releases the tensor memory late (``HOLD_TMEM_CLASSES``), so that a CTA's accumulator registers
are consumed before another CTA of the same launch can be given its columns.  ``Cfg.SYMBOL_TEXT`` is the text a rendering of those
classes carries in its kernel symbol (the classes of ``SHAPE_SYMBOL_CLASSES`` keep the symbol they were validated with).

LEDGERS.  Shared memory: :func:`smem_bytes` mirrors the kernel's allocation order (the buffer table of the kernel module) and is what the
launch reserves; ``Cfg.SMEM_RESIDENT`` = how many CTAs of that footprint shared memory lets the launcher place on one SM (an upper
bound of what a part places).  Tensor memory: an SM has 512 columns and a CTA of this kernel holds ``TMEM_COLS`` of them from its first
instruction to its last; :func:`validate_cfg` declines ``SMEM_RESIDENT x TMEM_COLS > 512`` by name -- a CTA placed by shared memory but
without columns would park inside ``tcgen05.alloc`` for a neighbour's whole lifetime, and under dependent launch it could have released
its successor before it owns its columns.  The ledger cuts both ways: a very shallow ring is declined too (a one-stage CTA of 11,520 B
would sit 18 to an SM), because 17 CTAs of 32 columns do not fit either.
``Cfg.RUNNING`` = ``min(SMEM_RESIDENT, 512 // TMEM_COLS)`` is what the one-wave rule of the pick counts.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

NAME = "the FP8 projection decode kernel"

K_TILE = 128  # K elements (one byte each) per operand-ring stage
MMA_K = 32  # K elements per tcgen05 FP8 MMA instruction
MMAS_PER_K_TILE = K_TILE // MMA_K  # the kernel asserts this at trace: it sets the accumulation order
THREADS_PER_CTA = 256  # warp 0 weight loader, warp 1 activation loader, warp 2 MMA, warp 3 idle, warps 4-7 epilogue
EPILOGUE_THREADS = 128
MMA_WARP_LANES = 32
MAX_ROWS = 32  # the served row ceiling: one token tile per CTA
TOKEN_TILES = (8, 16, 32)  # rows <= 8 / <= 16 / <= 32
WEIGHT_TILES = (64, 128)  # MMA_M: weight rows per CTA
SPLITS = (1, 2, 4)
TMEM_COLS_PER_SM = 512
TMEM_COLS_CHOICES = (32, 64, 128, 256)  # tcgen05.alloc takes a power of two >= 32
FP32_BYTES = 4
REMOTE_STORE_VALUES = 4  # one st.async carries four fp32 values (16 bytes)
MBARRIER_BYTES = 8
TMEM_POINTER_BYTES = 4
OPERAND_RING_ALIGN_BYTES = 1024
MAILBOX_ALIGN_BYTES = 128
SMEM_ALIGN_BYTES = 128
SMEM_CTA_RESERVE_BYTES = 1024  # per-CTA shared memory the launcher accounts beside the kernel's own reservation
OPERAND_ALIGN_BYTES = 32  # every tensor operand's base address (the compiled entry assumes it)

# Barrier init counts: SUM(issuing lanes) == init (the barrier table of the kernel module).
FULL_BAR_ARRIVERS = 2  # one elected lane of the weight loader + one elected lane of the activation loader, each with its expect-tx
EMPTY_BAR_ARRIVERS = 1  # one elected tcgen05.commit of the MMA warp per stage
MMA_DONE_BAR_ARRIVERS = 1  # one elected tcgen05.commit after the last K tile
TMEM_BAR_ARRIVERS = MMA_WARP_LANES + EPILOGUE_THREADS  # a rendezvous: every lane of the MMA warp and every epilogue thread, per phase
REDUCE_BAR_ARRIVERS = 1  # epilogue thread 0 of slice 0, with the peers' byte count as its expect-tx

# The SM100-class reference readings a host entry without a CUDA device keeps (compute capability 10.3; cc 10.0 has 148 SMs).
SM100_SM_COUNT = 152
SM100_SMEM_PER_SM_BYTES = 228 * 1024
SM100_SMEM_OPTIN_BYTES = 227 * 1024


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class Geometry:
    """``(OUT_FEATURES, IN_FEATURES)`` of one projection: the weight is ``[OUT_FEATURES, IN_FEATURES]``, a row of the activation
    ``[IN_FEATURES]``, a row of the output ``[OUT_FEATURES]``."""

    OUT_FEATURES: int
    IN_FEATURES: int

    def __post_init__(self):
        for name in ("OUT_FEATURES", "IN_FEATURES"):
            v = getattr(self, name)
            if isinstance(v, bool) or not hasattr(v, "__index__") or int(v) < 1:
                raise ValueError(f"{NAME}: Geometry.{name} must be a positive int, got {v!r}")
            object.__setattr__(self, name, int(v))

    @property
    def K_TILES(self) -> int:
        return self.IN_FEATURES // K_TILE

    def describe(self) -> str:
        return f"out features {self.OUT_FEATURES}, in features {self.IN_FEATURES}"


#: the geometries the kernel was validated on; the host entry declines any other by name (the structural validator admits more, unmeasured)
SERVED_GEOMETRIES = (Geometry(4608, 8192), Geometry(8192, 2048), Geometry(5120, 6144), Geometry(16384, 5120), Geometry(14336, 5120))


@dataclass(frozen=True)
class Numerics:
    """What the output bytes are a function of (on one runtime): the geometry and the number of K slices."""

    GEOM: Geometry
    SPLIT_K: int

    def describe(self) -> str:
        return f"{self.GEOM.describe()}, split_k {self.SPLIT_K}"


#: the validated byte classes (a reference check each; a measured pick at the first three, a table row run as written at the others);
#: every other ``(geometry, split_k)`` is declined by name
SERVED_CLASSES = (
    Numerics(Geometry(4608, 8192), 4),
    Numerics(Geometry(4608, 8192), 2),
    Numerics(Geometry(8192, 2048), 2),
    Numerics(Geometry(5120, 6144), 4),
    Numerics(Geometry(5120, 6144), 2),
    Numerics(Geometry(16384, 5120), 1),
    Numerics(Geometry(14336, 5120), 1),
)

#: the classes whose pick is lowered to ONE wave (rule 3 of :func:`pick`): the classes that rule was measured on.  Every other served
#: class runs its ``PICKS`` row as written.
ONE_WAVE_CLASSES = frozenset(SERVED_CLASSES[:3])
#: the classes whose pick releases the tensor memory late (``Form.HOLD_TMEM``): every class that runs its row as written.  Their grids
#: can take more than one wave, so columns a CTA deallocates can be given to another CTA of the same launch; with the late release the
#: registers read from those columns are consumed (the remote stores, the output store) before the release.  A performance pick like
#: any other: no byte depends on it.
HOLD_TMEM_CLASSES = frozenset(SERVED_CLASSES[3:])
#: the classes whose kernel symbol names the weight's shape only: the symbol their renderings were validated and recorded with (a
#: compiled kernel's bytes depend on its symbol).  A rendering of every other class carries ``Cfg.SYMBOL_TEXT`` in its symbol.
SHAPE_SYMBOL_CLASSES = frozenset(SERVED_CLASSES[:3])


@dataclass(frozen=True)
class Form:
    """The performance form of the kernel; no field of it enters a sum.

    ``MMA_M``: weight rows per CTA (64 or 128).  ``STAGES``: depth of the operand ring in K tiles, 1 .. the K tiles of one slice.
    ``TMEM_COLS``: tensor-memory columns a CTA allocates (0 = the smallest granule holding the token tile, which is 32 at every served
    tile).  ``TOKEN_TILE``: 0 = the smallest of 8 / 16 / 32 holding the call's rows; 8 / 16 / 32 = a floor (a wider tile than the rows
    need).  ``PREWAIT_WEIGHTS``: under dependent launch the weight loader does not wait for the predecessor (the weights are never a
    predecessor's output); False = every loader waits, the control arm.  ``HOLD_TMEM``: False = the epilogue releases the tensor memory
    right after its accumulator read and that read's ``tcgen05.wait`` (the deallocation then overlaps the reduction); True = it holds the
    columns until every register read from them is consumed and releases them as the last instruction of its role, which moves the
    deallocation behind the output store (the CTA ends that much later)."""

    MMA_M: int = 64
    STAGES: int = 7
    TMEM_COLS: int = 0
    TOKEN_TILE: int = 0
    PREWAIT_WEIGHTS: bool = True
    HOLD_TMEM: bool = False


def token_tile_for(rows: int, floor: int = 0) -> int:
    """The token tile of a row count (the smallest of 8 / 16 / 32 holding it, not below ``floor``): ``ValueError`` for a count below one,
    ``NotImplementedError`` past the served ceiling."""
    if not _is_int(rows) or rows < 1:
        raise ValueError(f"{NAME}: rows must be a positive int, got {rows!r}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call (decode-time row counts); got {rows}")
    if floor not in (0, *TOKEN_TILES):
        raise ValueError(f"{NAME}: Form.TOKEN_TILE must be 0 (by the row count) or one of {TOKEN_TILES}, got {floor!r}")
    return max(floor, next(t for t in TOKEN_TILES if rows <= t))


def tmem_cols_for(token_tile: int) -> int:
    """The tensor-memory columns the accumulator of a token tile needs: one column per token row, in the allocation's 32-column granule."""
    return max(TMEM_COLS_CHOICES[0], token_tile)


def split_k_for(geometry: Geometry) -> int:
    """The default number of K slices of a geometry: the largest of 4 / 2 / 1 that divides its K tiles and leaves at least eight K tiles per
    slice.  A function of the geometry ALONE -- never of the row count or the part (``split_k`` is a numerics attribute)."""
    for split in sorted(SPLITS, reverse=True):
        if geometry.K_TILES % split == 0 and geometry.K_TILES // split >= 8:
            return split
    return 1


def smem_bytes(mma_m: int, token_tile: int, split_k: int, stages: int) -> int:
    """The kernel's shared-memory footprint, in its allocation order (the buffer table of the kernel module): the weight ring, the
    activation ring, the two stage-barrier arrays and the two single barriers, the tensor-memory base pointer and, for more than one K
    slice, the reduction mailbox and its barrier.  The launch reserves exactly this."""
    size = _align_up(mma_m * K_TILE * stages, OPERAND_RING_ALIGN_BYTES)
    size += token_tile * K_TILE * stages
    size = _align_up(size, MBARRIER_BYTES) + 2 * stages * MBARRIER_BYTES + 2 * MBARRIER_BYTES
    size = _align_up(size, TMEM_POINTER_BYTES) + TMEM_POINTER_BYTES
    if split_k > 1:
        size = _align_up(size, MAILBOX_ALIGN_BYTES)
        size += (split_k - 1) * mma_m * token_tile * FP32_BYTES
        size = _align_up(size, MBARRIER_BYTES) + MBARRIER_BYTES
    return _align_up(size, SMEM_ALIGN_BYTES)


@dataclass(frozen=True)
class Cfg:
    """One compiled rendering: the numerics, the form resolved for ``ROWS``, the launch knobs (``PDL``: the launch carries the
    programmatic-stream-serialization attribute and the activation loader waits for the predecessor; ``PDL_RELEASE``: the weight loader
    releases the successor after its last issue) and the part's readings.  Hashable: it is the host entry's compile key."""

    GEOM: Geometry
    SPLIT_K: int
    ROWS: int
    MMA_M: int = 64
    STAGES: int = 7
    TOKEN_TILE: int = 8
    TMEM_COLS: int = 32
    PREWAIT_WEIGHTS: bool = True
    HOLD_TMEM: bool = False
    PDL: bool = False
    PDL_RELEASE: bool = False
    SMEM_PER_SM_BYTES: int = SM100_SMEM_PER_SM_BYTES
    SMEM_OPTIN_BYTES: int = SM100_SMEM_OPTIN_BYTES
    SM_COUNT: int = SM100_SM_COUNT

    @property
    def NUMERICS(self) -> Numerics:
        return Numerics(self.GEOM, self.SPLIT_K)

    @property
    def FORM(self) -> Form:
        return Form(self.MMA_M, self.STAGES, self.TMEM_COLS, self.TOKEN_TILE, self.PREWAIT_WEIGHTS, self.HOLD_TMEM)

    @property
    def N(self) -> int:
        return self.GEOM.OUT_FEATURES

    @property
    def K(self) -> int:
        return self.GEOM.IN_FEATURES

    @property
    def K_TILES_PER_SLICE(self) -> int:
        """K tiles one CTA walks: the slice of its cluster rank."""
        return self.GEOM.K_TILES // self.SPLIT_K

    @property
    def CTAS(self) -> int:
        """Grid: one cluster of ``SPLIT_K`` CTAs per ``MMA_M`` weight rows."""
        return (self.N // self.MMA_M) * self.SPLIT_K

    @property
    def VALUES_PER_THREAD(self) -> int:
        """Accumulator elements one epilogue thread owns."""
        return self.MMA_M * self.TOKEN_TILE // EPILOGUE_THREADS

    @property
    def MAILBOX_ELEMENTS(self) -> int:
        """fp32 slots of the reduction mailbox on slice 0: one accumulator tile per peer slice."""
        return (self.SPLIT_K - 1) * self.MMA_M * self.TOKEN_TILE

    @property
    def REDUCE_TX_BYTES(self) -> int:
        """Bytes the peers' remote stores complete on slice 0's reduction barrier = its expect-tx."""
        return self.MAILBOX_ELEMENTS * FP32_BYTES

    @property
    def WEIGHT_STAGE_BYTES(self) -> int:
        return self.MMA_M * K_TILE

    @property
    def ACTIVATION_STAGE_BYTES(self) -> int:
        return self.TOKEN_TILE * K_TILE

    @property
    def SMEM_BYTES(self) -> int:
        return smem_bytes(self.MMA_M, self.TOKEN_TILE, self.SPLIT_K, self.STAGES)

    @property
    def SMEM_RESIDENT(self) -> int:
        """CTAs of this footprint the launcher places on one SM (shared memory alone decides the placement)."""
        return self.SMEM_PER_SM_BYTES // (self.SMEM_BYTES + SMEM_CTA_RESERVE_BYTES)

    @property
    def RUNNING(self) -> int:
        """CTAs of one SM that hold their tensor-memory columns at the same time."""
        return min(self.SMEM_RESIDENT, TMEM_COLS_PER_SM // self.TMEM_COLS)

    @property
    def WAVES(self) -> int:
        return -(-self.CTAS // (self.SM_COUNT * max(self.RUNNING, 1)))

    @property
    def SYMBOL_TEXT(self) -> str:
        """The text of this rendering in its kernel symbol, right behind the kernel's name: ``n<OUT_FEATURES>k<IN_FEATURES>s<SPLIT_K>
        r<ROWS>_m<MMA_M>g<STAGES>t<TOKEN_TILE>c<TMEM_COLS>_w<PREWAIT_WEIGHTS>h<HOLD_TMEM>p<PDL>q<PDL_RELEASE>`` -- every field the
        rendered kernel depends on (the part's readings are not among them), at most 41 characters, so two renderings that differ in
        one of them never share a symbol.  Empty for a class of ``SHAPE_SYMBOL_CLASSES``, whose symbol names the weight's shape only."""
        if self.NUMERICS in SHAPE_SYMBOL_CLASSES:
            return ""
        flags = f"w{int(self.PREWAIT_WEIGHTS)}h{int(self.HOLD_TMEM)}p{int(self.PDL)}q{int(self.PDL_RELEASE)}"
        return f"n{self.N}k{self.K}s{self.SPLIT_K}r{self.ROWS}_m{self.MMA_M}g{self.STAGES}t{self.TOKEN_TILE}c{self.TMEM_COLS}_{flags}"

    def describe(self) -> str:
        return (
            f"{self.GEOM.describe()}, split_k {self.SPLIT_K}, rows {self.ROWS}: weight tile {self.MMA_M}, {self.STAGES} stages, token tile "
            f"{self.TOKEN_TILE}, {self.TMEM_COLS} tensor-memory columns"
        )


def check_geometry(geometry) -> Geometry:
    """``geometry`` as a served ``Geometry``: ``TypeError`` for another type, ``NotImplementedError`` (by name) for one not validated."""
    if not isinstance(geometry, Geometry):
        raise TypeError(f"{NAME}: geometry must be a fp8_projection_decode_config.Geometry, got {type(geometry).__name__}")
    if geometry not in SERVED_GEOMETRIES:
        raise NotImplementedError(f"{NAME} serves {'; '.join(g.describe() for g in SERVED_GEOMETRIES)} (validated); got {geometry.describe()}")
    return geometry


def check_numerics(geometry, split_k: Optional[int] = None) -> Numerics:
    """The byte class of ``(geometry, split_k)`` (``split_k`` None = :func:`split_k_for`): ``ValueError`` for a split that is not 1 / 2 / 4
    or does not cut the geometry's K tiles evenly, ``NotImplementedError`` (by name) for a legal pair that was not validated -- the
    message names the splits the geometry IS served at, and says so when the declined split is the geometry's default (the default
    is a function of the geometry alone and is not a served class at every geometry: such an entry is built with an explicit split)."""
    geometry = check_geometry(geometry)
    defaulted = split_k is None
    if split_k is None:
        split_k = split_k_for(geometry)
    if not _is_int(split_k) or split_k not in SPLITS:
        raise ValueError(f"{NAME}: split_k must be one of {SPLITS}, got {split_k!r}")
    if geometry.K_TILES % split_k:
        raise ValueError(f"{NAME}: split_k {split_k} does not cut the {geometry.K_TILES} K tiles of {geometry.describe()} into equal slices")
    numerics = Numerics(geometry, split_k)
    if numerics not in SERVED_CLASSES:
        splits = [str(c.SPLIT_K) for c in SERVED_CLASSES if c.GEOM == geometry]
        if not splits:
            raise NotImplementedError(f"{NAME} serves {'; '.join(c.describe() for c in SERVED_CLASSES)} (validated); got {numerics.describe()}")
        how = " -- the default of the geometry; pass split_k" if defaulted else ""
        raise NotImplementedError(f"{NAME} serves split_k {' or '.join(splits)} at {geometry.describe()} (validated); got split_k {split_k}{how}")
    return numerics


def part_readings(smem_per_sm: int, smem_optin: int, sm_count: int) -> Dict[str, int]:
    """The ``Cfg`` fields of a part reading ``(shared memory per SM, per-CTA opt-in, multiprocessors)``, validated."""
    for name, value in (("smem_per_sm", smem_per_sm), ("smem_optin", smem_optin), ("sm_count", sm_count)):
        if not _is_int(value) or value <= 0:
            raise ValueError(f"{NAME}: {name} must be a positive int (a device reading), got {value!r}")
    if smem_optin > smem_per_sm:
        raise ValueError(f"{NAME}: the per-CTA shared-memory opt-in ({smem_optin} B) cannot exceed the SM's ({smem_per_sm} B)")
    return dict(SMEM_PER_SM_BYTES=smem_per_sm, SMEM_OPTIN_BYTES=smem_optin, SM_COUNT=sm_count)


def validate_cfg(cfg: Cfg) -> None:
    """Structural validation of one rendering.  ``ValueError``: a knob (or a knob combination) the kernel cannot run; the message names
    the knob and its bound.  The served-class and row-count declines are :func:`check_numerics` / :func:`token_tile_for`."""
    if not isinstance(cfg.GEOM, Geometry):
        raise TypeError(f"{NAME}: Cfg.GEOM must be a Geometry, got {type(cfg.GEOM).__name__}")
    if cfg.K % K_TILE:
        raise ValueError(f"{NAME}: in features must be a multiple of {K_TILE}, got {cfg.K}")
    if cfg.SPLIT_K not in SPLITS or cfg.GEOM.K_TILES % cfg.SPLIT_K:
        raise ValueError(f"{NAME}: split_k must be one of {SPLITS} and cut the {cfg.GEOM.K_TILES} K tiles evenly, got {cfg.SPLIT_K!r}")
    if not _is_int(cfg.ROWS) or not 1 <= cfg.ROWS <= MAX_ROWS:
        raise ValueError(f"{NAME}: rows must be 1 .. {MAX_ROWS}, got {cfg.ROWS!r}")
    if cfg.MMA_M not in WEIGHT_TILES or cfg.N % cfg.MMA_M:
        raise ValueError(f"{NAME}: MMA_M must be one of {WEIGHT_TILES} and divide the {cfg.N} out features, got {cfg.MMA_M!r}")
    if cfg.TOKEN_TILE not in TOKEN_TILES or cfg.TOKEN_TILE < cfg.ROWS:
        raise ValueError(f"{NAME}: the token tile must be one of {TOKEN_TILES} and hold the {cfg.ROWS} rows, got {cfg.TOKEN_TILE!r}")
    if not _is_int(cfg.STAGES) or cfg.STAGES < 1:
        raise ValueError(f"{NAME}: STAGES must be a positive int, got {cfg.STAGES!r}")
    if cfg.STAGES > cfg.K_TILES_PER_SLICE:
        raise ValueError(f"{NAME}: {cfg.STAGES} stages exceed the {cfg.K_TILES_PER_SLICE} K tiles of one slice of K {cfg.K} at split_k {cfg.SPLIT_K}")
    if cfg.TMEM_COLS not in TMEM_COLS_CHOICES or cfg.TMEM_COLS < cfg.TOKEN_TILE:
        raise ValueError(f"{NAME}: TMEM_COLS must be one of {TMEM_COLS_CHOICES} and hold the token tile's {cfg.TOKEN_TILE} columns, got {cfg.TMEM_COLS!r}")
    if cfg.VALUES_PER_THREAD % REMOTE_STORE_VALUES:
        raise ValueError(
            f"{NAME}: a ({cfg.MMA_M}, {cfg.TOKEN_TILE}) tile gives {cfg.VALUES_PER_THREAD} values per epilogue thread; the cluster reduction "
            f"ships them {REMOTE_STORE_VALUES} at a time"
        )
    for name in ("PREWAIT_WEIGHTS", "HOLD_TMEM", "PDL", "PDL_RELEASE"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{NAME}: {name} must be a bool, got {getattr(cfg, name)!r}")
    part_readings(cfg.SMEM_PER_SM_BYTES, cfg.SMEM_OPTIN_BYTES, cfg.SM_COUNT)
    if cfg.SMEM_BYTES > cfg.SMEM_OPTIN_BYTES:
        raise ValueError(
            f"{NAME}: weight tile {cfg.MMA_M} with {cfg.STAGES} stages at token tile {cfg.TOKEN_TILE} needs {cfg.SMEM_BYTES} B of shared memory per "
            f"CTA; the part's opt-in is {cfg.SMEM_OPTIN_BYTES} B"
        )
    if cfg.SMEM_RESIDENT * cfg.TMEM_COLS > TMEM_COLS_PER_SM:
        raise ValueError(
            f"{NAME}: the tensor-memory ledger does not close: the launcher places {cfg.SMEM_RESIDENT} CTAs of {cfg.SMEM_BYTES} B on one SM and each "
            f"allocates {cfg.TMEM_COLS} columns ({cfg.SMEM_RESIDENT * cfg.TMEM_COLS} > {TMEM_COLS_PER_SM}); lower TMEM_COLS or raise the footprint"
        )


# (first row, last row, MMA_M, STAGES) per served class.  The first three: measured on compute capability 10.3 (152 SMs) with 32
# tensor-memory columns.
# A deeper ring pays while the grid still fits one wave AND the kernel starts ahead of its predecessor; from five rows the 7-stage form
# is level or ahead at (4608, 8192), and past the 8-row token tile an 11-stage CTA no longer shares an SM.
PICKS: Dict[Numerics, Tuple[Tuple[int, int, int, int], ...]] = {
    Numerics(Geometry(4608, 8192), 4): ((1, 4, 64, 11), (5, 32, 64, 7)),
    Numerics(Geometry(4608, 8192), 2): ((1, 32, 64, 8),),
    Numerics(Geometry(8192, 2048), 2): ((1, 32, 64, 8),),
    # The rows below are forms of the classes run AS WRITTEN: not measured with this entry; the ring depth is lowered only to fit a
    # part's per-CTA shared-memory limit, and these classes release the tensor memory late (``HOLD_TMEM_CLASSES``).
    Numerics(Geometry(5120, 6144), 4): ((1, 32, 64, 7),),
    Numerics(Geometry(5120, 6144), 2): ((1, 32, 64, 11),),
    Numerics(Geometry(16384, 5120), 1): ((1, 32, 128, 8),),
    Numerics(Geometry(14336, 5120), 1): ((1, 32, 128, 8),),
}


def resolve_form(numerics: Numerics, rows: int, form: Form, readings: Dict[str, int], pdl: bool = False, pdl_release: bool = False) -> Cfg:
    """``form`` as the rendering of ``rows`` on a part (the token tile and the tensor-memory columns resolved), NOT validated."""
    token_tile = token_tile_for(rows, form.TOKEN_TILE)
    return Cfg(
        GEOM=numerics.GEOM,
        SPLIT_K=numerics.SPLIT_K,
        ROWS=rows,
        MMA_M=form.MMA_M,
        STAGES=form.STAGES,
        TOKEN_TILE=token_tile,
        TMEM_COLS=form.TMEM_COLS if form.TMEM_COLS else tmem_cols_for(token_tile),
        PREWAIT_WEIGHTS=form.PREWAIT_WEIGHTS,
        HOLD_TMEM=form.HOLD_TMEM,
        PDL=pdl,
        PDL_RELEASE=pdl_release,
        **readings,
    )


def pick(numerics: Numerics, rows: int, sm_count: int, smem_per_sm: int, smem_optin: int) -> Form:
    """The performance form of ``rows`` for a served class on a part's readings -- a ``Form``, so it cannot name a split: the measured
    performance form for a class of ``ONE_WAVE_CLASSES``, the table row as written for every other class.

    1. the token tile is the smallest of 8 / 16 / 32 holding ``rows`` and the tensor-memory columns are what that tile needs;
    2. ``(MMA_M, STAGES)`` is the row of ``PICKS`` whose band holds ``rows``;
    3. a class of ``ONE_WAVE_CLASSES``: the stages are lowered until the grid fits ONE wave on this part (``CTAS <= SM count x
       RUNNING``) -- a deeper ring never pays for a second wave -- and a grid that fits one wave at no depth keeps one stage.  Every
       other class: the stages are lowered only until a CTA fits the part's per-CTA shared memory; the grid may take more than one
       wave.  Either way a part whose opt-in cannot hold a one-stage CTA is a ``NotImplementedError`` (a part fact);
    4. a class of ``HOLD_TMEM_CLASSES`` releases the tensor memory late."""
    token_tile_for(rows)
    readings = part_readings(smem_per_sm, smem_optin, sm_count)
    if numerics not in PICKS:
        raise NotImplementedError(f"{NAME} has no measured form for {numerics.describe()}")
    mma_m, stages = next((m, s) for lo, hi, m, s in PICKS[numerics] if lo <= rows <= hi)
    stages = min(stages, numerics.GEOM.K_TILES // numerics.SPLIT_K)
    one_wave = numerics in ONE_WAVE_CLASSES
    while True:
        cfg = resolve_form(numerics, rows, Form(MMA_M=mma_m, STAGES=stages), readings)
        if cfg.SMEM_BYTES <= smem_optin and (cfg.WAVES == 1 or not one_wave):
            break
        if stages == 1:
            if cfg.SMEM_BYTES > smem_optin:
                raise NotImplementedError(
                    f"{NAME} does not serve this part: a one-stage CTA of {cfg.describe()} needs {cfg.SMEM_BYTES} B of shared memory, the "
                    f"per-CTA opt-in is {smem_optin} B"
                )
            break
        stages -= 1
    return Form(MMA_M=cfg.MMA_M, STAGES=cfg.STAGES, TMEM_COLS=cfg.TMEM_COLS, HOLD_TMEM=numerics in HOLD_TMEM_CLASSES)
