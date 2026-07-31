# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-id namespace shared by cuDNN and Python backends.

Execution engines live in one flat integer id space, exactly like cuDNN's own
backend engines (which have small ids 0..N, each with knobs). Python engines
occupy a reserved high region so the two never collide and a heuristics query
can return a single ranked list mixing both, e.g.:

    [(engine_id=1048576, knobs), (engine_id=1, knobs), (engine_id=5, knobs), ...]

Dispatch is a single predicate on the id: ``is_python_engine(id)`` -> run via the
Python engine registry; otherwise lower to the cuDNN C++ backend.

Each Python engine declares a *stable* ``engine_id`` in this range (it owns its
id, the way a cuDNN engine does), so ids don't shift with registration order —
autotune results and pinned plans stay reproducible across runs.
"""

# Start of the reserved Python-engine id region. 1<<20 (~1.05M) is far above any
# plausible cuDNN engine count, so the two id spaces can never collide without
# having to know cuDNN's actual maximum.
PYTHON_ENGINE_ID_BASE = 1 << 20

# The backend side of the plan list: "delegate to the loaded backend's own
# heuristics". Deliberately ONE entry — the backend's engine set varies by
# backend version and is only discoverable per graph at plan time, never
# statically enumerable by the frontend.
BACKEND_HEURISTIC_ENGINE_ID = -1


def is_python_engine(engine_id: int) -> bool:
    """True iff ``engine_id`` names a Python engine (vs a cuDNN backend engine)."""
    return engine_id >= PYTHON_ENGINE_ID_BASE
