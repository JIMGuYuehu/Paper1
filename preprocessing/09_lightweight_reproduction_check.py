#!/usr/bin/env python3
"""Run small raw-to-existing-product numerical checks from a representative input subset.

INPUT
    Immutable WACCM, MERRA-2, and EXTR ClOx inputs configured in config.sh,
    plus the established processed products at PAPER1_REFERENCE_ROOT.
OUTPUT
    ${PAPER1_DERIVED_ROOT}/validation/lightweight_reproduction.json and one
    rebuilt 365-day ClOx sample file. Temporary pressure-interpolation files
    remain below the same validation directory and are removed automatically.
ACTION
    Rebuild the small zonal-mean ClOx year 0001 with the production no-leap
    function and compare it with the established year file. For large WACCM
    and MERRA-2 files, compare representative full spatial records rather than
    rewriting multi-GB annual products. Recompute three real-data 30--70 hPa
    WACCM partial-column values with the cleaned diagnostic kernel, and run
    the production CDO ml2pl operator for one WACCM day. Coordinates, masks
    and numerical tolerances are checked explicitly. No source or established
    product is ever opened for writing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER1_ROOT = SCRIPT_DIR.parent
DIAGNOSTIC_LIB = PAPER1_ROOT / "analysis" / "lib"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DIAGNOSTIC_LIB))

from paper1_diagnostics import (  # noqa: E402
    DU_PER_PA_MOLE_FRACTION,
    waccm_partial_ozone,
)
from noleap import encoding_for, normalize_dataset  # noqa: E402


ARCHIVE_ROOT = Path(
    os.environ.get("PAPER1_ARCHIVE_ROOT", str(PAPER1_ROOT / "data"))
).expanduser().resolve()
RAW_MODEL_ROOT = Path(
    os.environ.get("PAPER1_RAW_MODEL_ROOT", str(ARCHIVE_ROOT / "raw_model"))
).expanduser().resolve()
REFERENCE_ROOT = Path(
    os.environ.get("PAPER1_REFERENCE_ROOT", str(ARCHIVE_ROOT / "reference"))
).expanduser().resolve()
DERIVED_ROOT = Path(
    os.environ.get("PAPER1_DERIVED_ROOT", str(PAPER1_ROOT / "work"))
).expanduser().resolve()
OUTPUT_ROOT = DERIVED_ROOT / "validation"
REPORT = OUTPUT_ROOT / "lightweight_reproduction.json"
CDO_BIN = Path(os.environ.get("PAPER1_CDO_BIN", "cdo"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_safe_output() -> None:
    """Confine writes to PAPER1_DERIVED_ROOT and keep inputs read-only."""

    require(DERIVED_ROOT != Path(DERIVED_ROOT.anchor), "invalid output root")
    require(
        OUTPUT_ROOT == DERIVED_ROOT or DERIVED_ROOT in OUTPUT_ROOT.parents,
        "validation output escapes PAPER1_DERIVED_ROOT",
    )
    for protected in (RAW_MODEL_ROOT, REFERENCE_ROOT):
        require(
            OUTPUT_ROOT != protected and protected not in OUTPUT_ROOT.parents,
            f"validation output overlaps source root: {protected}",
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, object]:
    """Compare shape, finite/missing mask and numerical values."""

    left = np.asarray(np.ma.filled(np.ma.asarray(actual), np.nan), dtype=np.float64)
    right = np.asarray(np.ma.filled(np.ma.asarray(expected), np.nan), dtype=np.float64)
    require(left.shape == right.shape, f"{label}: shape {left.shape} != {right.shape}")
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    require(np.array_equal(left_finite, right_finite), f"{label}: finite masks differ")
    finite = left_finite & right_finite
    difference = np.abs(left[finite] - right[finite])
    maximum = float(difference.max()) if difference.size else 0.0
    scale = np.maximum(np.abs(right[finite]), np.finfo(np.float64).tiny)
    relative = difference / scale
    maximum_relative = float(relative.max()) if relative.size else 0.0
    require(
        np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True),
        f"{label}: values differ (max_abs={maximum:.6g}, max_rel={maximum_relative:.6g})",
    )
    return {
        "label": label,
        "shape": list(left.shape),
        "finite_count": int(finite.sum()),
        "max_abs_difference": maximum,
        "max_relative_difference": maximum_relative,
        "rtol": rtol,
        "atol": atol,
        "status": "PASS",
    }


def require_coordinates_equal(
    source: netCDF4.Dataset, reference: netCDF4.Dataset, names: tuple[str, ...], label: str
) -> None:
    for name in names:
        require(name in source.variables, f"{label}: source lacks coordinate {name}")
        require(name in reference.variables, f"{label}: reference lacks coordinate {name}")
        left = np.asarray(source.variables[name][:])
        right = np.asarray(reference.variables[name][:])
        require(
            np.array_equal(left, right),
            f"{label}: coordinate {name} differs",
        )


def compare_waccm_chunk(
    raw_file: Path,
    reference_file: Path,
    variables: tuple[str, ...],
    label: str,
) -> list[dict[str, object]]:
    """Compare three full spatial records carried through NCO concatenation."""

    require(raw_file.is_file(), f"missing raw sample: {raw_file}")
    require(reference_file.is_file(), f"missing established product: {reference_file}")
    results: list[dict[str, object]] = []
    with netCDF4.Dataset(raw_file) as source, netCDF4.Dataset(reference_file) as reference:
        require_coordinates_equal(source, reference, ("lev", "lat", "lon"), label)
        raw_dates = np.asarray(source.variables["date"][:], dtype=np.int64)
        reference_dates = np.asarray(reference.variables["date"][:], dtype=np.int64)
        chosen = np.unique(np.asarray([0, raw_dates.size // 2, raw_dates.size - 1]))
        source_dates = raw_dates[chosen]
        reference_indices = []
        for date_value in source_dates:
            found = np.flatnonzero(reference_dates == date_value)
            require(found.size == 1, f"{label}: date {date_value} is not unique in reference")
            reference_indices.append(int(found[0]))
        for variable in variables:
            require(variable in source.variables, f"{label}: raw source lacks {variable}")
            require(variable in reference.variables, f"{label}: reference lacks {variable}")
            results.append(
                comparison(
                    source.variables[variable][chosen, ...],
                    reference.variables[variable][reference_indices, ...],
                    label=f"{label}:{variable}:three_full_records",
                )
            )
    return results


def load_clox_module():
    """Load script 08 so this check reuses its archive-specific validator."""

    path = SCRIPT_DIR / "08_extr_clox_200yr.py"
    spec = importlib.util.spec_from_file_location("paper1_extr_clox", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild_and_compare_clox() -> list[dict[str, object]]:
    """Rebuild one complete small annual file with production functions."""

    module = load_clox_module()
    source_path = module.SOURCE
    reference_path = (
        REFERENCE_ROOT
        / "CO2x1SmidEmin_yBWCN_timefixed"
        / "CLOX"
        / "CO2x1SmidEmin_yBWCN.cam.h1.0001.CLOX.nc"
    )
    sample_path = OUTPUT_ROOT / "extr_clox_target0001_rebuilt.nc"
    with xr.open_dataset(source_path, decode_times=False, engine="netcdf4") as source:
        source_year_values, missing_counts = module.validate_archive_calendar(source)
        indices = np.flatnonzero(source_year_values == 101)
        annual = source[[name for name in ("CLOX", "date", "time_bnds", "gw") if name in source]].isel(
            time=indices
        )
        rebuilt, missing = normalize_dataset(annual, "CLOX", 1)
        require(missing.size == missing_counts[101] == 0, "ClOx year 0101 is incomplete")
        rebuilt.load()
    temporary = sample_path.with_name(sample_path.name + f".tmp.{os.getpid()}")
    try:
        rebuilt.to_netcdf(
            temporary, format="NETCDF4", engine="netcdf4", encoding=encoding_for(rebuilt)
        )
        os.replace(temporary, sample_path)
    finally:
        rebuilt.close()
        temporary.unlink(missing_ok=True)

    results: list[dict[str, object]] = []
    with netCDF4.Dataset(sample_path) as actual, netCDF4.Dataset(reference_path) as expected:
        require_coordinates_equal(actual, expected, ("date", "lev", "lat"), "ClOx year0001")
        results.append(
            comparison(
                actual.variables["CLOX"][:],
                expected.variables["CLOX"][:],
                label="ClOx:rebuilt_full_year0001",
            )
        )
    results[-1]["rebuilt_output"] = str(sample_path)
    results[-1]["reference"] = str(reference_path)
    return results


def compare_merra_daily() -> list[dict[str, object]]:
    """Compare one real daily granule with its record in the annual product."""

    results: list[dict[str, object]] = []
    cases = (
        (
            sorted(REFERENCE_ROOT.glob("MERRA2M2I6NPANA/MERRA2_*.20200101.SUB.nc"))[0],
            REFERENCE_ROOT / "MERRA2_Processed" / "O3" / "MERRA2.O3.2020.nc",
            "O3",
            "O3",
        ),
        (
            sorted(REFERENCE_ROOT.glob("MERRA2M2I6NPANA/Z/MERRA2_*.20200101.SUB.nc"))[0],
            REFERENCE_ROOT / "MERRA2_Processed" / "Z3" / "MERRA2.Z3.2020.nc",
            "H",
            "Z3",
        ),
    )
    for raw_path, reference_path, raw_name, reference_name in cases:
        with netCDF4.Dataset(raw_path) as source, netCDF4.Dataset(reference_path) as reference:
            require_coordinates_equal(
                source, reference, ("lev", "lat", "lon"), f"MERRA2:{reference_name}"
            )
            values = np.ma.asarray(source.variables[raw_name][:])
            if values.shape[0] == 1:
                daily = values[0]
                mode = "GES_DISC_daily_mean"
            elif values.shape[0] == 4:
                daily = np.ma.mean(values, axis=0)
                mode = "local_four_record_mean"
            else:
                raise RuntimeError(f"{raw_path}: expected one or four records")
            result = comparison(
                daily,
                reference.variables[reference_name][0, ...],
                label=f"MERRA2:{reference_name}:2020-01-01",
                rtol=2.0e-7,
                atol=1.0e-12,
            )
            result["source_mode"] = mode
            results.append(result)
    return results


def compare_partial_ozone() -> list[dict[str, object]]:
    """Recompute three BWCN year-0008 partial columns with the cleaned kernel."""

    source_path = REFERENCE_ROOT / "BWCN" / "O3" / "BWCN.cam.h3.0008.O3.nc"
    reference_path = REFERENCE_ROOT / "BWCN" / "partial_O3" / "BWCN_partial_O3_all_ranges.nc"
    wanted_dates = np.asarray([80301, 80401, 80501], dtype=np.int64)
    with xr.open_dataset(source_path, decode_times=False, engine="netcdf4") as source:
        dates = np.asarray(source["date"].values, dtype=np.int64)
        indices = [int(np.flatnonzero(dates == date)[0]) for date in wanted_dates]
        subset = source.isel(time=indices).load()
    calculated = waccm_partial_ozone(subset)
    subset.close()
    with xr.open_dataset(reference_path, decode_times=False, engine="netcdf4") as reference:
        dates = np.asarray(reference["date"].values, dtype=np.int64)
        indices = [int(np.flatnonzero(dates == date)[0]) for date in wanted_dates]
        expected = np.asarray(
            reference["O3_partial_60_90N_30_70hPa"].isel(time=indices).values
        )
    # The established notebook used NA=6.02214e23 and M_air=28.964e-3;
    # Methods-V7 uses the exact SI Avogadro constant and M_air=28.9647e-3.
    # Correct the established DU values by that known constant ratio before
    # testing the integration, mask and averaging implementation.
    legacy_du_factor = 6.02214e23 / (9.80665 * 28.964e-3 * 2.687e20)
    constant_ratio = DU_PER_PA_MOLE_FRACTION / legacy_du_factor
    converted_expected = expected * constant_ratio
    result = comparison(
        calculated.values,
        converted_expected,
        label="diagnostic:WACCM_partial_O3_DU:three_dates:V7_constants",
        rtol=3.0e-7,
        atol=2.0e-5,
    )
    raw_difference = np.asarray(calculated.values) - expected
    result.update(
        legacy_du_factor=float(legacy_du_factor),
        methods_v7_du_factor=float(DU_PER_PA_MOLE_FRACTION),
        legacy_to_v7_constant_ratio=float(constant_ratio),
        unconverted_legacy_max_abs_difference_du=float(
            np.max(np.abs(raw_difference))
        ),
        difference_note=(
            "established product used NA=6.02214e23 and M_air=28.964e-3; "
            "comparison applies only the explicit constant-factor conversion"
        ),
    )
    return [result]


def compare_pressure_interpolation() -> list[dict[str, object]]:
    """Run the exact script-07 CDO operator for one day and compare it."""

    require(CDO_BIN.is_file(), f"CDO not found: {CDO_BIN}")
    script_text = (SCRIPT_DIR / "07_waccm_pressure_level_products.sh").read_text()
    match = re.search(r'^PLEV_PA="([0-9,]+)"$', script_text, flags=re.MULTILINE)
    require(match is not None, "cannot read the production pressure-level contract")
    plev_pa = match.group(1)
    source = (
        REFERENCE_ROOT
        / "B2000WCN001002_timefixed"
        / "U"
        / "B2000WCN.sample.cam.h3.0001.U.nc"
    )
    reference = (
        REFERENCE_ROOT
        / "B2000WCN001002_timefixed"
        / "interpolated"
        / "U"
        / "B2000WCN.sample.cam.h3.0001.U.nc"
    )
    sample = OUTPUT_ROOT / "waccm_longrun_0001_day1_U_ml2pl.nc"
    with tempfile.TemporaryDirectory(prefix="ml2pl.", dir=OUTPUT_ROOT) as temporary_text:
        temporary = Path(temporary_text)
        one_day = temporary / "hybrid_day.nc"
        with_bounds = temporary / "hybrid_day_with_bounds.nc"
        subprocess.run(
            ["ncks", "-4", "-L", "1", "-O", "-d", "time,0,0", str(source), str(one_day)],
            check=True,
        )
        subprocess.run(
            ["ncatted", "-O", "-a", "bounds,lev,c,c,ilev", str(one_day), str(with_bounds)],
            check=True,
        )
        subprocess.run(
            [
                str(CDO_BIN), "-L", "-s", "-O", "-f", "nc4", "-z", "zip_1",
                f"-ml2pl,{plev_pa}", "-select,name=U", str(with_bounds), str(sample),
            ],
            check=True,
        )
    with netCDF4.Dataset(sample) as actual, netCDF4.Dataset(reference) as expected:
        require_coordinates_equal(actual, expected, ("lat", "lon"), "CDO ml2pl")
        actual_pressure = np.asarray(actual.variables["plev"][:], dtype=np.float64)
        expected_pressure = np.asarray(expected.variables["plev"][:], dtype=np.float64)
        common_pressure = np.intersect1d(actual_pressure, expected_pressure)
        new_only_pressure = np.setdiff1d(actual_pressure, expected_pressure)
        require(
            common_pressure.size == 31,
            f"CDO ml2pl: expected 31 legacy/common levels; found {common_pressure.size}",
        )
        require(
            np.array_equal(
                new_only_pressure,
                np.asarray([10.0, 50.0, 400.0, 4000.0, 92500.0]),
            ),
            f"CDO ml2pl: unexpected Methods-V7-only levels {new_only_pressure.tolist()}",
        )
        actual_indices = [
            int(np.flatnonzero(actual_pressure == level)[0]) for level in common_pressure
        ]
        expected_indices = [
            int(np.flatnonzero(expected_pressure == level)[0]) for level in common_pressure
        ]
        result = comparison(
            actual.variables["U"][0, actual_indices, ...],
            expected.variables["U"][0, expected_indices, ...],
            label="preprocessing:CDO_ml2pl:WACCM_day1_U:31_common_levels",
            rtol=3.0e-6,
            atol=2.0e-4,
        )
    result["rebuilt_output"] = str(sample)
    result["pressure_levels_pa"] = plev_pa
    result["legacy_common_level_count"] = int(common_pressure.size)
    result["methods_v7_additional_levels_pa"] = new_only_pressure.tolist()
    result["additional_level_note"] = (
        "legacy 31-level product has no numerical reference at 925, 40, 4, "
        "0.5 or 0.1 hPa; the cleaned 36-level coordinate is validated exactly"
    )
    return [result]


def atomic_json(payload: dict[str, object]) -> None:
    temporary = REPORT.with_name(REPORT.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, REPORT)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    assert_safe_output()
    results: list[dict[str, object]] = []
    results.extend(rebuild_and_compare_clox())

    long_raw = sorted(
        (RAW_MODEL_ROOT / "extr_2000" / "extr_2000" / "B2000WCN.e122.f19_g16.001" / "atm" / "hist").glob(
            "B2000WCN.e122.f19_g16.001.cam.h3.0001-*.nc.extr.nc"
        )
    )[0]
    results.extend(
        compare_waccm_chunk(
            long_raw,
            REFERENCE_ROOT / "B2000WCN001002_timefixed" / "O3" / "B2000WCN.sample.cam.h3.0001.O3.nc",
            ("O3",),
            "WACCM_LONGRUN:run001:year0001:first_chunk",
        )
    )
    results.extend(
        compare_waccm_chunk(
            long_raw,
            REFERENCE_ROOT / "B2000WCN001002_timefixed" / "U" / "B2000WCN.sample.cam.h3.0001.U.nc",
            ("U",),
            "WACCM_LONGRUN:run001:year0001:first_chunk",
        )
    )

    bwcn_raw = sorted(
        (RAW_MODEL_ROOT / "LENS" / "BWCN.e122.f19_g16.002" / "atm" / "hist").glob(
            "BWCN.e122.f19_g16.002.cam.h3.0008-*.nc"
        )
    )[0]
    results.extend(
        compare_waccm_chunk(
            bwcn_raw,
            REFERENCE_ROOT / "BWCN" / "O3" / "BWCN.cam.h3.0008.O3.nc",
            ("O3",),
            "WACCM_BWCN:year0008:first_chunk",
        )
    )

    hindcast_files = sorted((RAW_MODEL_ROOT / "lens" / "0008-01").glob("*.cam.h3.*.nc*"))
    require(bool(hindcast_files), "no January hindcast raw files found")
    prefix = re.sub(r"\.[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{5}\.nc.*$", "", str(hindcast_files[0]))
    first_member_files = sorted(Path(prefix).parent.glob(Path(prefix).name + ".*.nc*"))
    member_name = Path(prefix).name
    results.extend(
        compare_waccm_chunk(
            first_member_files[0],
            REFERENCE_ROOT / "Hindcast" / "0008-01" / "O3" / f"{member_name}.O3.nc",
            ("O3",),
            "WACCM_HINDCAST:0008-01:first_member:first_chunk",
        )
    )

    results.extend(compare_merra_daily())
    results.extend(compare_pressure_interpolation())
    results.extend(compare_partial_ozone())

    payload = {
        "validation_kind": "lightweight representative raw/existing-product comparison",
        "production_reconstruction": False,
        "source_files_modified": False,
        "established_products_modified": False,
        "checks": results,
        "status": "PASS",
    }
    atomic_json(payload)
    print(f"lightweight reproduction PASS: {len(results)} checks")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
