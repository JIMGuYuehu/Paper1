#!/usr/bin/env python3
"""Normalize CAM annual files and audit winter-to-spring sample completeness.

INPUT
    A yearly, single-variable CAM NetCDF file containing ``time`` and ``date``.
OUTPUT
    A 365-day no-leap NetCDF file, event-completeness TSV, or pressure-year /
    Figure-15 availability TSV.
ACTION
    Map source month/day values to the requested sample year, remove 29
    February, insert missing calendar days as NaN (never interpolate across
    time), retain ``date`` and ``datesec``, and write atomically. The manifest
    marks an ozone-event year complete only when 27 February-2 May is present.
    Those two padding days on each side are required to calculate a centered
    5-day mean whose reported minima span 1 March-30 April. This is the 207 +
    23 sample rule used by Methods V7, not the count of raw model-year labels.
    For precursor fields, separately derive complete event Y plus its existing
    Y-1 pressure source; do not discard an incomplete spring that is needed as
    a later complete event's November-December predecessor.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import netCDF4
import xarray as xr


MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
YEAR_RE = re.compile(r"\.cam\.(?:h[0-9]+\.)?(?P<year>\d{4})\.[A-Za-z0-9_]+\.nc$")


def full_dates(year: int) -> np.ndarray:
    return np.asarray(
        [year * 10000 + month * 100 + day
         for month, length in enumerate(MONTH_LENGTHS, start=1)
         for day in range(1, length + 1)],
        dtype=np.int32,
    )


def ozone_event_required_dates(year: int) -> set[int]:
    """Dates needed for centered-5-day values reported on 1 March-30 April."""
    return {
        int(item)
        for item in full_dates(year)
        if 227 <= int(item) % 10000 <= 502
    }


def time_values(year: int) -> np.ndarray:
    return (year - 1) * 365 + np.arange(365, dtype=np.float64)


def time_bounds(year: int) -> np.ndarray:
    values = time_values(year)
    bounds = np.column_stack((values - 1.0, values))
    # CAM's first model-day record has the degenerate [0, 0] bound; do not
    # manufacture a negative day before the no-leap epoch.
    if year == 1:
        bounds[0] = (0.0, 0.0)
    return bounds


def normalize_dataset(source: xr.Dataset, variable: str, year: int) -> tuple[xr.Dataset, np.ndarray]:
    if variable not in source:
        raise KeyError(f"missing variable {variable}")
    if "time" not in source.dims or "date" not in source:
        raise ValueError("source must contain a time dimension and CAM date variable")

    raw_dates = np.asarray(source["date"].values, dtype=np.int64).reshape(-1)
    if raw_dates.size != source.sizes["time"]:
        raise ValueError("date length differs from time length")
    month_day = raw_dates % 10000
    keep = month_day != 229
    source = source.isel(time=np.flatnonzero(keep))
    month_day = month_day[keep]
    mapped_dates = year * 10000 + month_day
    if np.unique(mapped_dates).size != mapped_dates.size:
        unique_dates, counts = np.unique(mapped_dates, return_counts=True)
        duplicates = unique_dates[counts > 1]
        raise ValueError(f"duplicate calendar days after normalization: {duplicates[:10]}")

    expected = full_dates(year)
    unexpected = np.setdiff1d(mapped_dates, expected)
    if unexpected.size:
        raise ValueError(f"invalid no-leap dates: {unexpected[:10].tolist()}")
    missing = np.setdiff1d(expected, mapped_dates).astype(np.int32)

    attrs = dict(source.attrs)
    time_attrs = dict(source["time"].attrs)
    date_attrs = dict(source["date"].attrs)
    datesec_attrs = dict(source["datesec"].attrs) if "datesec" in source else {}
    time_bnds_attrs = dict(source["time_bnds"].attrs) if "time_bnds" in source else {}

    # Integer YYYYMMDD values are only a temporary reindexing key. Missing days
    # become NaN in all time-dependent scientific variables.
    output = source.assign_coords(time=("time", mapped_dates)).reindex(time=expected)
    output = output.assign_coords(time=("time", time_values(year)))
    output["time"].attrs = time_attrs
    output["time"].attrs.update(
        units="days since 0001-01-01 00:00:00",
        calendar="noleap",
        bounds="time_bnds",
    )
    output["date"] = xr.DataArray(expected, dims=("time",), attrs=date_attrs)

    datesec = np.zeros(365, dtype=np.int32)
    if "datesec" in source:
        original = np.asarray(source["datesec"].values).reshape(-1)
        if original.size:
            datesec[:] = int(original[0])
    output["datesec"] = xr.DataArray(datesec, dims=("time",), attrs=datesec_attrs)

    bounds_dim = "nbnd"
    if "time_bnds" in source and source["time_bnds"].ndim == 2:
        bounds_dim = source["time_bnds"].dims[1]
    output["time_bnds"] = xr.DataArray(
        time_bounds(year), dims=("time", bounds_dim), attrs=time_bnds_attrs
    )
    output.attrs = attrs
    output.attrs.update(
        paper1_calendar_normalization="365_day noleap; missing source days are NaN",
        paper1_missing_days_filled_with_nan=int(missing.size),
        paper1_missing_dates=",".join(str(int(item)) for item in missing),
        paper1_target_year=f"{year:04d}",
    )
    return output, missing


def encoding_for(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, array in dataset.variables.items():
        options: dict[str, object] = {}
        if name in dataset.data_vars:
            options.update(zlib=True, complevel=1, shuffle=True)
        if np.issubdtype(array.dtype, np.floating):
            options["_FillValue"] = np.nan
        if options:
            encoding[name] = options
    return encoding


def normalize_file(args: argparse.Namespace) -> None:
    if args.output.exists():
        if args.output.stat().st_size <= 0:
            raise RuntimeError(f"existing output is empty: {args.output}")
        with xr.open_dataset(args.output, decode_times=False, engine="netcdf4") as existing:
            if (
                args.variable not in existing
                or existing.sizes.get("time") != 365
                or not np.array_equal(existing["date"].values, full_dates(args.year))
                or "paper1_missing_dates" not in existing.attrs
            ):
                raise RuntimeError(f"existing output failed validation: {args.output}")
        print(f"retained valid output: {args.output}")
        return

    with xr.open_dataset(args.input, decode_times=False, engine="netcdf4") as source:
        output, missing = normalize_dataset(source, args.variable, args.year)
        output.load()
    output.attrs.update(
        paper1_source_file=str(args.input),
        paper1_source_run=args.source_run,
        paper1_source_year=f"{args.source_year:04d}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        output.to_netcdf(
            temporary, format="NETCDF4", engine="netcdf4", encoding=encoding_for(output)
        )
        os.replace(temporary, args.output)
    finally:
        output.close()
        temporary.unlink(missing_ok=True)
    print(f"wrote {args.output}; missing source days={missing.size}")


def parse_year(path: Path) -> int:
    match = YEAR_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse output year: {path.name}")
    return int(match.group("year"))


def missing_dates(path: Path, variable: str) -> set[int]:
    with xr.open_dataset(path, decode_times=False, engine="netcdf4") as dataset:
        if variable not in dataset or dataset.sizes.get("time") != 365:
            raise ValueError(f"invalid annual {variable} file: {path}")
        expected = full_dates(parse_year(path))
        if "date" not in dataset or not np.array_equal(dataset["date"].values, expected):
            raise ValueError(f"invalid annual date coordinate: {path}")
        if "paper1_missing_dates" not in dataset.attrs:
            raise ValueError(f"annual file lacks gap provenance: {path}")
        text = str(dataset.attrs["paper1_missing_dates"]).strip()
        return {int(token) for token in text.split(",") if token}


def event_manifest(args: argparse.Namespace) -> None:
    variables = tuple(token.strip() for token in args.variables.split(",") if token.strip())
    if not variables:
        raise ValueError("at least one variable is required")
    files: dict[str, dict[int, Path]] = {}
    for variable in variables:
        found = {parse_year(path): path for path in sorted((args.root / variable).glob(f"*.{variable}.nc"))}
        if not found:
            raise FileNotFoundError(f"no annual {variable} files under {args.root / variable}")
        files[variable] = found

    common_years = sorted(set.intersection(*(set(item) for item in files.values())))
    if len(common_years) != args.expected_years:
        raise RuntimeError(
            f"expected {args.expected_years} common annual outputs; found {len(common_years)}"
        )
    rows: list[dict[str, object]] = []
    for event_year in common_years:
        problems: list[str] = []
        for variable in variables:
            current_missing = missing_dates(files[variable][event_year], variable)
            required = ozone_event_required_dates(event_year)
            seasonal_missing = sorted(item for item in current_missing if item in required)
            if seasonal_missing:
                problems.append(f"{variable}:{','.join(map(str, seasonal_missing))}")
        rows.append(
            {
                "event_year": event_year,
                "complete_centered5_o3_window": int(not problems),
                "reason": ";".join(problems),
            }
        )

    complete_count = sum(int(row["complete_centered5_o3_window"]) for row in rows)
    if complete_count != args.expected_complete:
        raise RuntimeError(
            f"expected {args.expected_complete} complete Feb27-May2 events; found {complete_count}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("event_year", "complete_centered5_o3_window", "reason"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {args.output}; complete Feb27-May2 events={complete_count}")


def source_event_manifest(args: argparse.Namespace) -> None:
    """Audit event-window dates directly in raw chunks, without concatenating fields."""
    grouped: dict[int, list[tuple[int, Path]]] = {}
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target_year = int(row["target_year"])
            source_year = int(row["source_year"])
            grouped.setdefault(target_year, []).append((source_year, Path(row["path"])))
    if not grouped:
        raise RuntimeError(f"raw inventory is empty: {args.inventory}")
    if len(grouped) != args.expected_years:
        raise RuntimeError(
            f"expected {args.expected_years} raw target years; found {len(grouped)}"
        )

    rows: list[dict[str, object]] = []
    for target_year, source_items in sorted(grouped.items()):
        observed: set[int] = set()
        for source_year, path in sorted(source_items, key=lambda item: str(item[1])):
            # This audit needs only CAM's small integer date vector.  Opening
            # through xarray initialises every coordinate/backend index and is
            # prohibitively slow over the archive mount; direct netCDF4 access
            # reads the identical variable without touching scientific fields.
            with netCDF4.Dataset(path, mode="r") as dataset:
                if "date" not in dataset.variables:
                    raise RuntimeError(f"raw source lacks CAM date: {path}")
                raw_dates = np.asarray(
                    dataset.variables["date"][:], dtype=np.int64
                ).reshape(-1)
            if raw_dates.size == 0:
                raise RuntimeError(f"raw chunk has no CAM dates: {path}")
            raw_years = set(int(item) // 10000 for item in raw_dates)
            # Inventory rows are intentionally annual. Crossing a CAM year in
            # one chunk would make the source->target mapping ambiguous, so it
            # is rejected rather than silently borrowing boundary records.
            if raw_years != {source_year}:
                raise RuntimeError(
                    f"raw date year does not match inventory source_year={source_year:04d}: "
                    f"{path} has {sorted(raw_years)}"
                )
            allowed = set(int(item) for item in full_dates(source_year))
            unexpected = sorted(set(int(item) for item in raw_dates) - allowed)
            if unexpected:
                raise RuntimeError(f"invalid CAM calendar dates in {path}: {unexpected[:10]}")
            for raw_date in raw_dates:
                month_day = int(raw_date) % 10000
                if month_day != 229:
                    observed.add(target_year * 10000 + month_day)
        required = ozone_event_required_dates(target_year)
        missing = sorted(required - observed)
        rows.append(
            {
                "event_year": target_year,
                "complete_centered5_o3_window": int(not missing),
                "reason": "missing:" + ",".join(map(str, missing)) if missing else "",
                "source_files": len(source_items),
            }
        )

    complete_count = sum(int(row["complete_centered5_o3_window"]) for row in rows)
    if complete_count != args.expected_complete:
        raise RuntimeError(
            f"expected {args.expected_complete} raw-source-complete Feb27-May2 events; "
            f"found {complete_count}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "event_year", "complete_centered5_o3_window", "reason", "source_files"
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"wrote {args.output}; raw-source-complete Feb27-May2 events={complete_count}"
    )


def pressure_year_manifest(args: argparse.Namespace) -> None:
    """Derive pressure-source years from ranking events and Y-1 predecessors.

    Figure 15 ranks complete spring event Y, but its November-December field
    belongs to Y-1. An incomplete spring can therefore still be a required
    pressure source (for example, year 0096 supplies event 0097). The first
    event in a contiguous segment has no predecessor in that segment; it is
    recorded as unavailable for Figure 15 rather than treated as a raw-data
    preprocessing failure.
    """
    with args.raw_years.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"raw-year manifest has no header: {args.raw_years}")
        year_column = "target_year" if "target_year" in reader.fieldnames else "source_year"
        raw_values = [int(row[year_column]) for row in reader]
    if not raw_values:
        raise RuntimeError(f"raw-year manifest is empty: {args.raw_years}")
    if len(raw_values) != len(set(raw_values)):
        raise RuntimeError(f"raw-year manifest contains duplicate {year_column} values")
    raw_years = set(raw_values)
    if len(raw_years) != args.expected_raw_years:
        raise RuntimeError(
            f"{args.segment}: raw years={len(raw_years)}, expected={args.expected_raw_years}"
        )
    ordered_raw = sorted(raw_years)
    contiguous = list(range(ordered_raw[0], ordered_raw[-1] + 1))
    if ordered_raw != contiguous:
        raise RuntimeError(
            f"{args.segment}: raw target-year axis has internal gaps; refusing ambiguous Y-1 mapping"
        )

    event_flags: dict[int, int] = {}
    with args.events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            year = int(row["event_year"])
            if year in event_flags:
                raise RuntimeError(f"duplicate event year in {args.events}: {year}")
            flag = int(row["complete_centered5_o3_window"])
            if flag not in {0, 1}:
                raise RuntimeError(f"invalid complete flag for {year}: {flag}")
            event_flags[year] = flag
    if set(event_flags) != raw_years:
        missing = sorted(raw_years - set(event_flags))
        extra = sorted(set(event_flags) - raw_years)
        raise RuntimeError(
            f"{args.segment}: raw/event year sets differ; missing_events={missing}, extra={extra}"
        )

    ranking_events = {year for year, flag in event_flags.items() if flag == 1}
    predecessor_for: dict[int, list[int]] = {}
    unavailable_events: list[int] = []
    for event_year in sorted(ranking_events):
        predecessor = event_year - 1
        if predecessor in raw_years:
            predecessor_for.setdefault(predecessor, []).append(event_year)
        else:
            unavailable_events.append(event_year)
    internal_unavailable = [year for year in unavailable_events if year != ordered_raw[0]]
    if internal_unavailable:
        raise RuntimeError(
            f"{args.segment}: ranking events lack internal predecessors: {internal_unavailable}"
        )

    pressure_years = ranking_events | set(predecessor_for)
    field_events = ranking_events - set(unavailable_events)
    observed = {
        "ranking": len(ranking_events),
        "pressure": len(pressure_years),
        "field": len(field_events),
        "unavailable": len(unavailable_events),
    }
    expected = {
        "ranking": args.expected_ranking_events,
        "pressure": args.expected_pressure_years,
        "field": args.expected_field_events,
        "unavailable": args.expected_unavailable_events,
    }
    if observed != expected:
        raise RuntimeError(f"{args.segment}: count contract differs: {observed} != {expected}")

    rows: list[dict[str, object]] = []
    for year in ordered_raw:
        is_ranking = year in ranking_events
        predecessor_year = year - 1 if is_ranking else None
        field_available = is_ranking and predecessor_year in raw_years
        rows.append(
            {
                "segment": args.segment,
                "pressure_year": year,
                "raw_available": 1,
                "ranking_event_complete": int(is_ranking),
                "needed_as_predecessor": int(year in predecessor_for),
                "pressure_source_required": int(year in pressure_years),
                "figure15_field_available": int(field_available) if is_ranking else "",
                "predecessor_year": predecessor_year if is_ranking else "",
                "predecessor_for_events": ",".join(
                    f"{item:04d}" for item in predecessor_for.get(year, [])
                ),
                "availability_reason": (
                    "segment_start_has_no_Y_minus_1" if is_ranking and not field_available else ""
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "segment", "pressure_year", "raw_available", "ranking_event_complete",
                    "needed_as_predecessor", "pressure_source_required",
                    "figure15_field_available", "predecessor_year",
                    "predecessor_for_events", "availability_reason",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    unavailable_text = ",".join(f"{year:04d}" for year in unavailable_events)
    print(
        f"wrote {args.output}; segment={args.segment}; ranking={len(ranking_events)}; "
        f"pressure_sources={len(pressure_years)}; Figure15_fields={len(field_events)}; "
        f"unavailable_boundary_events={unavailable_text}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize-year")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--variable", required=True)
    normalize.add_argument("--year", type=int, required=True)
    normalize.add_argument("--source-year", type=int, required=True)
    normalize.add_argument("--source-run", required=True)
    normalize.set_defaults(run=normalize_file)

    manifest = commands.add_parser("event-manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--variables", required=True)
    manifest.add_argument("--expected-years", type=int, required=True)
    manifest.add_argument("--expected-complete", type=int, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(run=event_manifest)

    source_manifest = commands.add_parser("source-event-manifest")
    source_manifest.add_argument("--inventory", type=Path, required=True)
    source_manifest.add_argument("--expected-years", type=int, required=True)
    source_manifest.add_argument("--expected-complete", type=int, required=True)
    source_manifest.add_argument("--output", type=Path, required=True)
    source_manifest.set_defaults(run=source_event_manifest)

    pressure_manifest = commands.add_parser("pressure-year-manifest")
    pressure_manifest.add_argument("--raw-years", type=Path, required=True)
    pressure_manifest.add_argument("--events", type=Path, required=True)
    pressure_manifest.add_argument("--segment", required=True)
    pressure_manifest.add_argument("--expected-raw-years", type=int, required=True)
    pressure_manifest.add_argument("--expected-ranking-events", type=int, required=True)
    pressure_manifest.add_argument("--expected-pressure-years", type=int, required=True)
    pressure_manifest.add_argument("--expected-field-events", type=int, required=True)
    pressure_manifest.add_argument("--expected-unavailable-events", type=int, default=1)
    pressure_manifest.add_argument("--output", type=Path, required=True)
    pressure_manifest.set_defaults(run=pressure_year_manifest)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
