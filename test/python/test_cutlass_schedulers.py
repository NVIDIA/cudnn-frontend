# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler metadata must survive the host/device MLIR argument boundary."""

import importlib.util
from pathlib import Path
import sys

import pytest

cutlass = pytest.importorskip("cutlass", minversion="4.6.2")
from cutlass._mlir import ir
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values

pytestmark = [pytest.mark.L0, pytest.mark.filterwarnings("error::DeprecationWarning")]

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "cudnn"


def _load_scheduler(name, monkeypatch):
    relative_path = {
        "static": "_cutlass_helpers/static_persistent_tile_scheduler.py",
        "grouped": "gemm/cutedsl/grouped/utils.py",
    }[name]
    # Load just the scheduler; importing the grouped package also imports every
    # GEMM API, which is unnecessary for this metadata-only regression.
    spec = importlib.util.spec_from_file_location(f"_test_cutlass_{name}_scheduler", _PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scheduler_name", ["static", "grouped"])
@pytest.mark.parametrize("swizzle", [1, 2, 4])
@pytest.mark.parametrize("raster_along_m", [True, False])
def test_scheduler_divisors_survive_mlir_roundtrip(scheduler_name, swizzle, raster_along_m, monkeypatch):
    scheduler = _load_scheduler(scheduler_name, monkeypatch)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            params = scheduler.PersistentTileSchedulerParams(
                (cutlass.Int32(16), cutlass.Int32(8), cutlass.Int32(2)),
                (2, 1, 1),
                swizzle_size=swizzle,
                raster_along_m=raster_along_m,
            )
            values = extract_mlir_values(params)
            restored = new_from_mlir_values(params, values)
            assert len(extract_mlir_values(restored)) == len(values)

            divisor_names = [name for name in vars(params) if name.endswith("_fdd")]
            assert divisor_names
            for name in divisor_names:
                divisor = getattr(params, name)
                restored_divisor = getattr(restored, name)
                if divisor is None:
                    assert restored_divisor is None
                else:
                    # Preserve both the encoded divisor and its scalar SSA value.
                    assert extract_mlir_values(restored_divisor) == extract_mlir_values(divisor)


@pytest.mark.parametrize("static_coordinates", [False, True])
def test_work_tile_roundtrip_preserves_static_coordinates(static_coordinates, monkeypatch):
    scheduler = _load_scheduler("static", monkeypatch)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            coords = (cutlass.Int32(3), 4, 0) if static_coordinates else (cutlass.Int32(3), cutlass.Int32(4), cutlass.Int32(0))
            tile = scheduler.WorkTileInfo(coords, cutlass.Boolean(True))
            values = extract_mlir_values(tile)
            restored = new_from_mlir_values(tile, values)
            assert extract_mlir_values(restored) == values
