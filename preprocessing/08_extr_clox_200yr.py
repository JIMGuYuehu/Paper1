#!/usr/bin/env python3
"""Build the 200-year EXTR ClOx climatology input used by Figure 3.

INPUT
    Combined WACCM ClOx file set by PAPER1_EXTR_CLOX_SOURCE.
    CO2x1SmidEmin_yBWCN.cam.h1.0101-0300.CLO.isobar.zm.nc
OUTPUT
    ${PAPER1_DERIVED_ROOT}/CO2x1SmidEmin_yBWCN_timefixed/CLOX/
    CO2x1SmidEmin_yBWCN.cam.h1.YYYY.CLOX.nc for target years 0001-0200,
    plus ${PAPER1_DERIVED_ROOT}/manifests/extr_clox_200yr.tsv.
ACTION
    Verify source years 0101-0300 and variable CLOX. The archived combined
    file has one documented corruption signature: 17 consecutive all-zero
    placeholder records between 0188-09-05 and 0188-09-23, plus known gaps in
    source years 0188, 0219 and 0220. Reject any departure from that exact
    signature, exclude only the proven placeholders, split the remaining data
    into 200 target years, map each to a 365-day no-leap calendar, fill absent
    source days with NaN (no temporal interpolation), retain an explicit
    source-to-target-year mapping, and write every product atomically.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import xarray as xr

from noleap import encoding_for, full_dates, normalize_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER1_ROOT = SCRIPT_DIR.parent
ARCHIVE_ROOT = Path(
    os.environ.get("PAPER1_ARCHIVE_ROOT", str(PAPER1_ROOT / "data"))
).expanduser().resolve(strict=False)
SOURCE = Path(
    os.environ.get(
        "PAPER1_EXTR_CLOX_SOURCE",
        str(
            ARCHIVE_ROOT
            / "WACCM"
            / "clox_climatology"
            / "CO2x1SmidEmin_yBWCN.cam.h1.0101-0300.CLO.isobar.zm.nc"
        ),
    )
).expanduser().resolve(strict=False)
DERIVED_ROOT_TEXT = os.environ.get(
    "PAPER1_DERIVED_ROOT", str(PAPER1_ROOT / "work")
)
if not DERIVED_ROOT_TEXT.strip():
    raise RuntimeError("PAPER1_DERIVED_ROOT is empty")
DERIVED_ROOT = Path(DERIVED_ROOT_TEXT).expanduser().resolve(strict=False)
OUTPUT_ROOT = DERIVED_ROOT / "CO2x1SmidEmin_yBWCN_timefixed" / "CLOX"
MANIFEST = DERIVED_ROOT / "manifests" / "extr_clox_200yr.tsv"
AUDIT_ONLY_TEXT = os.environ.get("PAPER1_AUDIT_ONLY", "0")
if AUDIT_ONLY_TEXT not in {"0", "1"}:
    raise RuntimeError("PAPER1_AUDIT_ONLY must be 0 or 1")
AUDIT_ONLY = AUDIT_ONLY_TEXT == "1"


# This is an archive-specific integrity contract, not a generic instruction to
# discard year zero. Direct inspection on 2026-08-24 established that the 17
# records are consecutive, carry date=time=0, contain only zero-valued CLOX,
# and replace 0188-09-06--0188-09-22. The combined file also contains only the
# first ten days of 0219 and lacks the first five days of 0220. Every absent
# calendar day is preserved as NaN by normalize_dataset.
EXPECTED_MISSING_BY_SOURCE_YEAR = {
    188: np.arange(1880906, 1880923, dtype=np.int64),
    219: full_dates(219)[10:].astype(np.int64),
    220: full_dates(220)[:5].astype(np.int64),
}
EXPECTED_ZERO_PLACEHOLDER_COUNT = 17


def _is_within(candidate: Path, parent: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def assert_safe_destination() -> None:
    if DERIVED_ROOT == Path(DERIVED_ROOT.anchor):
        raise RuntimeError("refusing filesystem root as PAPER1_DERIVED_ROOT")
    if _is_within(DERIVED_ROOT, ARCHIVE_ROOT) or _is_within(ARCHIVE_ROOT, DERIVED_ROOT):
        raise RuntimeError("input and output roots must not overlap")
    for candidate in (OUTPUT_ROOT, MANIFEST):
        assert_output_path(candidate)


def assert_output_path(candidate: Path) -> None:
    """Reject writes outside PAPER1_DERIVED_ROOT."""

    resolved = candidate.resolve(strict=False)
    if not _is_within(resolved, DERIVED_ROOT):
        raise RuntimeError(
            f"output escapes PAPER1_DERIVED_ROOT: {candidate} -> {resolved}"
        )


def valid_output(path: Path, target_year: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path, decode_times=False, engine="netcdf4") as dataset:
            return bool(
                "CLOX" in dataset
                and dataset.sizes.get("time") == 365
                and "date" in dataset
                and np.array_equal(dataset["date"].values, full_dates(target_year))
            )
    except Exception:
        return False


def atomic_dataset(dataset: xr.Dataset, output: Path) -> None:
    assert_output_path(output)
    assert_output_path(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_output_path(output.parent)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    assert_output_path(temporary)
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        dataset.to_netcdf(
            temporary,
            format="NETCDF4",
            engine="netcdf4",
            encoding=encoding_for(dataset),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_manifest(rows: list[dict[str, object]]) -> None:
    assert_output_path(MANIFEST)
    assert_output_path(MANIFEST.parent)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    assert_output_path(MANIFEST.parent)
    temporary = MANIFEST.with_name(MANIFEST.name + f".tmp.{os.getpid()}")
    assert_output_path(temporary)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "target_year", "source_year", "source_valid_days",
                    "missing_days", "archive_zero_placeholder_records",
                    "source_path", "source_size_bytes", "source_mtime_ns",
                    "output",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)


def validate_archive_calendar(source: xr.Dataset) -> tuple[np.ndarray, dict[int, int]]:
    """Validate the exact known date/gap corruption signature of this archive."""

    raw_dates = np.asarray(source["date"].values, dtype=np.int64).reshape(-1)
    if raw_dates.size != source.sizes.get("time"):
        raise RuntimeError("EXTR ClOx date length differs from time length")

    zero_indices = np.flatnonzero(raw_dates == 0)
    if (
        zero_indices.size != EXPECTED_ZERO_PLACEHOLDER_COUNT
        or not np.array_equal(
            zero_indices,
            np.arange(zero_indices[0], zero_indices[0] + zero_indices.size),
        )
        or zero_indices[0] == 0
        or zero_indices[-1] == raw_dates.size - 1
        or raw_dates[zero_indices[0] - 1] != 1880905
        or raw_dates[zero_indices[-1] + 1] != 1880923
    ):
        raise RuntimeError(
            "EXTR ClOx zero-date records do not match the documented 17-record "
            "0188-09-06--0188-09-22 placeholder block"
        )
    if "time" not in source:
        raise RuntimeError("EXTR ClOx source lacks time")
    zero_times = np.asarray(source["time"].isel(time=zero_indices).values)
    if not np.all(np.isfinite(zero_times)) or not np.all(zero_times == 0):
        raise RuntimeError("EXTR ClOx zero-date placeholders do not have time=0")
    zero_clox = np.asarray(source["CLOX"].isel(time=zero_indices).values)
    if not np.all(np.isfinite(zero_clox)) or not np.all(zero_clox == 0):
        raise RuntimeError(
            "EXTR ClOx zero-date placeholders are not finite all-zero fields"
        )

    valid_dates = raw_dates[raw_dates > 0]
    if np.unique(valid_dates).size != valid_dates.size:
        raise RuntimeError("EXTR ClOx positive calendar dates are not unique")
    expected_dates = np.concatenate(
        [full_dates(year).astype(np.int64) for year in range(101, 301)]
    )
    unexpected = np.setdiff1d(valid_dates, expected_dates)
    if unexpected.size:
        raise RuntimeError(
            f"EXTR ClOx contains unexpected positive dates: {unexpected[:10].tolist()}"
        )

    missing_counts: dict[int, int] = {}
    for source_year in range(101, 301):
        actual = valid_dates[valid_dates // 10000 == source_year]
        missing = np.setdiff1d(full_dates(source_year), actual).astype(np.int64)
        expected_missing = EXPECTED_MISSING_BY_SOURCE_YEAR.get(
            source_year, np.asarray([], dtype=np.int64)
        )
        if not np.array_equal(missing, expected_missing):
            raise RuntimeError(
                f"EXTR ClOx source year {source_year:04d} missing dates changed: "
                f"found {missing[:10].tolist()} (count={missing.size})"
            )
        missing_counts[source_year] = int(missing.size)

    if sum(missing_counts.values()) != 377:
        raise RuntimeError("EXTR ClOx documented missing-day total is not 377")
    return raw_dates // 10000, missing_counts


def main() -> None:
    assert_safe_destination()
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    rows: list[dict[str, object]] = []
    source_stat = SOURCE.stat()
    with xr.open_dataset(SOURCE, decode_times=False, engine="netcdf4") as source:
        if "CLOX" not in source or "date" not in source:
            raise RuntimeError(f"source must contain CLOX and date: {SOURCE}")
        source_year_values, expected_missing_counts = validate_archive_calendar(source)
        source_years = sorted(
            set(int(item) for item in source_year_values if int(item) > 0)
        )
        expected_source_years = list(range(101, 301))
        if source_years != expected_source_years:
            raise RuntimeError(
                f"expected exact EXTR years 0101-0300; found {source_years[:3]}...{source_years[-3:]} "
                f"(count={len(source_years)})"
            )

        keep_names = [
            name for name in ("CLOX", "date", "datesec", "time_bnds", "gw")
            if name in source
        ]
        for source_year in expected_source_years:
            target_year = source_year - 100
            output = OUTPUT_ROOT / (
                f"CO2x1SmidEmin_yBWCN.cam.h1.{target_year:04d}.CLOX.nc"
            )
            assert_output_path(output)
            indices = np.flatnonzero(source_year_values == source_year)
            if indices.size == 0:
                raise RuntimeError(f"source year is empty: {source_year:04d}")
            annual_source = source[keep_names].isel(time=indices)
            normalized, missing = normalize_dataset(annual_source, "CLOX", target_year)
            normalized.attrs.update(
                source_experiment="CO2x1SmidEmin_yBWCN",
                source_file=str(SOURCE),
                source_year=f"{source_year:04d}",
                target_normalized_year=f"{target_year:04d}",
                source_valid_days=int(indices.size),
                archive_zero_placeholder_records=EXPECTED_ZERO_PLACEHOLDER_COUNT,
                archive_missing_day_contract=(
                    "0188:09-06--09-22;0219:01-11--12-31;"
                    "0220:01-01--01-05;missing days are NaN"
                ),
                processing_note="365-day noleap; absent source days are NaN, never interpolated",
            )
            if int(missing.size) != expected_missing_counts[source_year]:
                raise RuntimeError(
                    f"source year {source_year:04d}: normalization missing-day "
                    f"count {missing.size} != audited count "
                    f"{expected_missing_counts[source_year]}"
                )
            if AUDIT_ONLY:
                pass
            elif output.exists():
                if not valid_output(output, target_year):
                    raise RuntimeError(f"existing output failed validation: {output}")
            else:
                normalized.load()
                atomic_dataset(normalized, output)
                if not valid_output(output, target_year):
                    raise RuntimeError(f"new output failed validation: {output}")
            normalized.close()
            rows.append(
                {
                    "target_year": f"{target_year:04d}",
                    "source_year": f"{source_year:04d}",
                    "source_valid_days": int(indices.size),
                    "missing_days": int(missing.size),
                    "archive_zero_placeholder_records": (
                        EXPECTED_ZERO_PLACEHOLDER_COUNT
                    ),
                    "source_path": str(SOURCE),
                    "source_size_bytes": int(source_stat.st_size),
                    "source_mtime_ns": int(source_stat.st_mtime_ns),
                    "output": str(output),
                }
            )
            print(
                f"EXTR CLOX {source_year:04d} -> {target_year:04d}; "
                f"missing={missing.size}; audit_only={AUDIT_ONLY}",
                flush=True,
            )
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 EXTR outputs; built {len(rows)}")
    atomic_manifest(rows)
    print(
        f"validated 200-year EXTR ClOx chain; wrote {MANIFEST}; "
        f"audit_only={AUDIT_ONLY}"
    )


if __name__ == "__main__":
    main()
