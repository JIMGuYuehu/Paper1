#!/usr/bin/env python3
"""Synthetic regression tests for Paper 1 preprocessing gates.

INPUT
    No climate files. Small in-memory stand-ins for NetCDF metadata/coordinates.
OUTPUT
    unittest status only; no files are created or modified.
ACTION
    Exercise native WACCM grid/level checks, hindcast coverage through 31 May,
    MLS calendar-date uniqueness, and the requirement that MERRA-2 height
    covers every 23-level NAM pressure.
"""

from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import netCDF4  # noqa: F401
except ModuleNotFoundError:
    # The STREAM jimnew environment has netCDF4. This minimal local fallback
    # lets the metadata-only tests run in a workstation Python without it.
    sys.modules["netCDF4"] = types.SimpleNamespace(Dataset=None)

import paper1_source_audit as audit


class FakeDimension:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class FakeVariable:
    def __init__(self, values: object, dimensions: tuple[str, ...], units: str = ""):
        self.values = np.asarray(values)
        self.dimensions = dimensions
        self.units = units

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)


class FakeDataset:
    def __init__(
        self,
        variables: dict[str, FakeVariable],
        dimensions: dict[str, int],
        attributes: dict[str, object] | None = None,
    ):
        self.variables = variables
        self.dimensions = {
            name: FakeDimension(size) for name, size in dimensions.items()
        }
        self.attributes = attributes or {}

    def __enter__(self) -> "FakeDataset":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def ncattrs(self) -> list[str]:
        return list(self.attributes)

    def getncattr(self, name: str) -> object:
        return self.attributes[name]


def fake_waccm_dataset(lev_count: int = 66) -> FakeDataset:
    dimensions = {"time": 2, "lat": 96, "lon": 144, "lev": lev_count, "ilev": 67}
    variables = {
        "time": FakeVariable([0.0, 1.0], ("time",)),
        "date": FakeVariable([10101, 10102], ("time",)),
        "datesec": FakeVariable([0, 0], ("time",)),
        "lat": FakeVariable(np.linspace(-90.0, 90.0, 96), ("lat",), "degrees_north"),
        "lon": FakeVariable(np.arange(144) * 2.5, ("lon",), "degrees_east"),
        "lev": FakeVariable(np.geomspace(1.0e-5, 995.0, lev_count), ("lev",), "hPa"),
        "ilev": FakeVariable(np.geomspace(5.0e-6, 1000.0, 67), ("ilev",), "hPa"),
        "hyam": FakeVariable(np.linspace(0.0, 1.0, lev_count), ("lev",)),
        "hybm": FakeVariable(np.linspace(1.0, 0.0, lev_count), ("lev",)),
        "hyai": FakeVariable(np.linspace(0.0, 1.0, 67), ("ilev",)),
        "hybi": FakeVariable(np.linspace(1.0, 0.0, 67), ("ilev",)),
        "P0": FakeVariable(100000.0, (), "Pa"),
    }
    for name in ("U", "V", "T", "Z3", "O3"):
        variables[name] = FakeVariable([0.0], ("time", "lev", "lat", "lon"))
    variables["PS"] = FakeVariable([0.0], ("time", "lat", "lon"))
    return FakeDataset(
        variables,
        dimensions,
        {"model": "WACCM4", "case": "synthetic", "version": "test"},
    )


def fake_merra_height_dataset(levels: np.ndarray) -> FakeDataset:
    variables = {
        "lat": FakeVariable(np.linspace(-90.0, 90.0, 361), ("lat",), "degrees_north"),
        "lon": FakeVariable(np.arange(576) * 0.625, ("lon",), "degrees_east"),
        "lev": FakeVariable(levels, ("lev",), "hPa"),
        "time": FakeVariable([0.0], ("time",), "hours since 2000-01-01 00:00:00"),
        "H": FakeVariable([0.0], ("time", "lev", "lat", "lon"), "m"),
    }
    return FakeDataset(
        variables,
        {"time": 1, "lat": 361, "lon": 576, "lev": len(levels)},
    )


def daily_model_dates(start: datetime, end: datetime) -> np.ndarray:
    values: list[int] = []
    current = start
    while current <= end:
        if not (current.month == 2 and current.day == 29):
            values.append(current.year * 10000 + current.month * 100 + current.day)
        current += timedelta(days=1)
    return np.asarray(values, dtype=np.int64)


class SourceContractTests(unittest.TestCase):
    def test_native_waccm_grid_and_66_levels_pass(self) -> None:
        dataset = fake_waccm_dataset()
        with mock.patch.object(audit.netCDF4, "Dataset", return_value=dataset):
            result = audit.inspect_waccm_native_file(
                Path(__file__), ("U", "V", "T", "Z3", "O3", "PS")
            )
        self.assertEqual(result["lat_count"], 96)
        self.assertEqual(result["lon_count"], 144)
        self.assertEqual(result["lev_count"], 66)
        self.assertIn("WACCM4", str(result["model_attributes"]))

    def test_waccm_65_levels_fail(self) -> None:
        dataset = fake_waccm_dataset(lev_count=65)
        with mock.patch.object(audit.netCDF4, "Dataset", return_value=dataset):
            with self.assertRaisesRegex(RuntimeError, "lev=65; expected 66"):
                audit.inspect_waccm_native_file(
                    Path(__file__), ("U", "V", "T", "Z3", "O3", "PS")
                )

    def test_hindcasts_require_may_31_endpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "0008-05-31"):
            audit.hindcast_window_contract(
                "0008-01",
                "member",
                daily_model_dates(datetime(8, 1, 1), datetime(8, 5, 30)),
            )
        january = daily_model_dates(datetime(8, 1, 1), datetime(8, 5, 31))
        self.assertEqual(
            audit.hindcast_window_contract("0008-01", "member", january),
            (151, 80101, 80531),
        )
        with self.assertRaisesRegex(RuntimeError, "0008-05-31"):
            audit.hindcast_window_contract(
                "0008-02",
                "member",
                daily_model_dates(datetime(8, 2, 1), datetime(8, 5, 30)),
            )
        february = daily_model_dates(datetime(8, 2, 1), datetime(8, 5, 31))
        self.assertEqual(
            audit.hindcast_window_contract("0008-02", "member", february),
            (120, 80201, 80531),
        )

    def test_mls_duplicate_calendar_date_fails(self) -> None:
        decoded = [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 12)]
        with self.assertRaisesRegex(RuntimeError, "repeats calendar dates"):
            audit.unique_calendar_day_coverage(decoded, 2020, "synthetic MLS")

    def test_merra_height_rejects_ep_only_18_levels(self) -> None:
        dataset = fake_merra_height_dataset(audit.MERRA_PROCESSED_18_HPA)
        with self.assertRaisesRegex(RuntimeError, "full 23-level NAM grid"):
            audit.validate_merra_grid(dataset, Path("synthetic.nc"), ("H",))

    def test_merra_height_accepts_native_42_levels(self) -> None:
        dataset = fake_merra_height_dataset(audit.MERRA_NATIVE_42_HPA)
        audit.validate_merra_grid(dataset, Path("synthetic.nc"), ("H",))


if __name__ == "__main__":
    unittest.main()
