# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-callable CuTe DSL requirement gates of the ``tile_dsl`` primitives (AGENTS.md Rule 7).

THIS MODULE IMPORTS NOTHING FROM ``cutlass`` -- that is its whole point.  The primitive modules (``tma.py``, ``mask.py``,
...) import ``cutlass.experimental`` at module level, so on a DSL below the library floor (the package floor
``nvidia-cutlass-dsl==4.6.2`` has no ``cutlass.experimental``) ``from cudnn.frost.tile_dsl.tma import ...`` dies with a
``ModuleNotFoundError`` before any function in it can run.  A host-side check (an adapter's ``check_support``, a route
check) therefore imports its gate from HERE, before it imports the primitive or its kernel module, and gets a message
naming the installed and the required version instead of the raw import failure.

The version half reads ``cudnn.frost.buffers.cutedsl_state()`` (package metadata, no DSL import); the arch half
(``cutedsl_arch_requirement_error``) imports ``cutlass.base_dsl`` lazily, and only once the version half is satisfied, on
a cc 10.7 part.  ``tile_dsl`` never queries the device: the cc is the API layer's fact, passed in.
"""

from ..buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error


def requirement_error(what, device_cc=None):
    """Message refusing ``what`` (a primitive's dotted name) on the installed DSL, else ``None``.

    Below ``CUTEDSL_MIN_VERSION`` the message names the installed and the required version (``cutedsl_requirement_error``);
    at or above it, when ``device_cc`` is given, a DSL without the part's target is refused by name
    (``cutedsl_arch_requirement_error``: an SM107 part needs the public 4.8.0 wheel, the first whose ``Arch`` knows
    ``sm_107a``).  Host Python only: no DSL import on the version path, no device query.
    """
    msg = cutedsl_requirement_error(what)
    if msg is None and device_cc is not None:
        msg = cutedsl_arch_requirement_error(tuple(device_cc))
    return msg


def tma_gather4_requirement_error(device_cc=None):
    """The host half of the Rule-7 gate of :func:`tile_dsl.tma.tma_gather4`; ``tma_gather4`` runs the version half again at
    trace time, before it imports the inline-asm atom.  Import it from THIS module on a host path -- ``tma.py`` re-exports
    the name for callers that already hold the DSL, but importing ``tma`` below the floor is the failure this gate exists
    to pre-empt."""
    return requirement_error("tile_dsl.tma.tma_gather4", device_cc)
