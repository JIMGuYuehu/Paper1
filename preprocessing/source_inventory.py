#!/usr/bin/env python3
"""Audit Paper 1 raw-source schemas without modifying source data.

INPUT
    WACCM restart chunks/pressure products, March restart members, MERRA-2 SUB or
    annual files, or Aura MLS Level-3 zonal-mean files.
OUTPUT
    A compact TSV inventory written atomically to the requested staging path.
ACTION
    Enforce exact member/date coverage, required variables, pressure levels,
    MLS group names, and MERRA-2 daily-mean provenance before/after processing.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np


MERRA_RE = re.compile(
    r"^MERRA2_(?P<collection>\d+)\.inst6_3d_ana_Np\."
    r"(?P<date>\d{8})\.SUB\.nc$"
)
MERRA_PROCESSED_18_HPA = np.asarray(
    [350, 300, 250, 200, 150, 100, 70, 50, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1],
    dtype=float,
)
MERRA_NATIVE_42_HPA = np.asarray(
    [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
        650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50,
        40, 30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1,
    ],
    dtype=float,
)
MERRA_NAM_23_HPA = np.asarray(
    [
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70,
        50, 30, 20, 10, 5, 3, 2, 1, 0.5, 0.1,
    ],
    dtype=float,
)
WACCM_PLEV_PA = np.asarray(
    [
        100000, 95000, 92500, 90000, 85000, 80000, 75000, 70000, 60000,
        55000, 50000, 45000, 40000, 35000, 30000, 25000, 22500, 20000,
        17500, 15000, 12500, 10000, 7000, 5000, 4000, 3000, 2000, 1000,
        700, 500, 400, 300, 200, 100, 50, 10,
    ],
    dtype=float,
)
MARCH_REQUIRED_LEVELS_PA = np.asarray(
    [1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 100000.0]
)
MARCH_CANONICAL_23_PA = np.asarray(
    [
        10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 10000,
        15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000,
        92500, 100000,
    ],
    dtype=float,
)
HINDCAST_CHUNK_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<stamp>\d{4}-\d{2}-\d{2}-\d{5})\.nc.*$"
)
HINDCAST_VARIABLES = ("U", "V", "T", "Z3", "O3", "PS")
WACCM_NATIVE_COUNTS = {"lat": 96, "lon": 144, "lev": 66, "ilev": 67}
WACCM_REQUIRED_COORDINATES = (
    "time", "date", "datesec", "lat", "lon", "lev", "ilev", "P0",
    "hyam", "hybm", "hyai", "hybi",
)


def atomic_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_evidence(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def compact_units(value: object) -> str:
    return re.sub(r"[\s_*^]", "", str(value)).lower()


def printable_attribute(value: object) -> str:
    """Return a one-line representation suitable for a TSV JSON field."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    text = " ".join(str(value).split())
    return text if len(text) <= 2000 else text[:1997] + "..."


def waccm_model_attributes(dataset: netCDF4.Dataset) -> str:
    """Record available model/case provenance without manufacturing metadata."""
    keywords = (
        "model", "case", "version", "source", "title", "experiment",
        "institution", "convention",
    )
    selected = {
        name: printable_attribute(dataset.getncattr(name))
        for name in dataset.ncattrs()
        if any(keyword in name.lower() for keyword in keywords)
    }
    if not selected:
        return "MISSING"
    return json.dumps(selected, ensure_ascii=True, sort_keys=True)


def inspect_waccm_native_file(
    path: Path,
    required_variables: tuple[str, ...],
    reference_coordinates: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Validate one native WACCM4 history chunk without loading data fields."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"WACCM source chunk is missing or empty: {path}")
    with netCDF4.Dataset(path) as dataset:
        missing = [
            name
            for name in (*required_variables, *WACCM_REQUIRED_COORDINATES)
            if name not in dataset.variables
        ]
        if missing:
            raise RuntimeError(f"{path} is missing WACCM fields: {missing}")
        for dimension, expected_count in WACCM_NATIVE_COUNTS.items():
            if dimension not in dataset.dimensions:
                raise RuntimeError(f"{path} is missing WACCM dimension {dimension}")
            actual_count = len(dataset.dimensions[dimension])
            if actual_count != expected_count:
                raise RuntimeError(
                    f"{path} has {dimension}={actual_count}; expected {expected_count}"
                )
        coordinates: dict[str, np.ndarray] = {}
        for name in ("lat", "lon", "lev", "ilev", "hyam", "hybm", "hyai", "hybi"):
            values = np.asarray(dataset.variables[name][:], dtype=float).reshape(-1)
            if values.size != WACCM_NATIVE_COUNTS["ilev" if name in {"ilev", "hyai", "hybi"} else "lev"] \
                    and name not in {"lat", "lon"}:
                raise RuntimeError(f"{path}:{name} has an invalid coordinate length")
            if name == "lat" and values.size != WACCM_NATIVE_COUNTS["lat"]:
                raise RuntimeError(f"{path}:lat has an invalid coordinate length")
            if name == "lon" and values.size != WACCM_NATIVE_COUNTS["lon"]:
                raise RuntimeError(f"{path}:lon has an invalid coordinate length")
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{path}:{name} contains non-finite coordinates")
            coordinates[name] = values

        latitude = coordinates["lat"]
        longitude = coordinates["lon"]
        levels = coordinates["lev"]
        if not np.allclose(
            latitude, np.linspace(-90.0, 90.0, 96), rtol=0.0, atol=1e-4
        ):
            raise RuntimeError(
                f"{path} is not the native WACCM ~1.9 degree 96-latitude grid"
            )
        if not (
            np.all(np.diff(longitude) > 0.0)
            and np.allclose(np.diff(longitude), 2.5, rtol=0.0, atol=1e-5)
            and np.isclose(longitude[-1] - longitude[0], 357.5, rtol=0.0, atol=1e-5)
        ):
            raise RuntimeError(f"{path} is not the native WACCM 2.5 degree global longitude grid")
        lat_units = compact_units(getattr(dataset.variables["lat"], "units", ""))
        lon_units = compact_units(getattr(dataset.variables["lon"], "units", ""))
        if lat_units not in {"degreesnorth", "degreenorth", "degreesn"}:
            raise RuntimeError(f"{path} has invalid latitude units: {lat_units!r}")
        if lon_units not in {"degreeseast", "degreeeast", "degreese"}:
            raise RuntimeError(f"{path} has invalid longitude units: {lon_units!r}")
        if not (np.all(np.diff(levels) > 0.0) or np.all(np.diff(levels) < 0.0)):
            raise RuntimeError(f"{path} WACCM lev coordinate is not strictly monotonic")

        for name, expected_dimension in (
            ("lat", "lat"), ("lon", "lon"), ("lev", "lev"), ("ilev", "ilev"),
            ("hyam", "lev"), ("hybm", "lev"), ("hyai", "ilev"), ("hybi", "ilev"),
        ):
            if tuple(dataset.variables[name].dimensions) != (expected_dimension,):
                raise RuntimeError(
                    f"{path}:{name} must use only {expected_dimension}: "
                    f"{dataset.variables[name].dimensions}"
                )
        ntime = len(dataset.dimensions.get("time", ()))
        if ntime <= 0:
            raise RuntimeError(f"{path} has no WACCM time records")
        for name in ("time", "date", "datesec"):
            if tuple(dataset.variables[name].dimensions) != ("time",):
                raise RuntimeError(f"{path}:{name} must use only the time dimension")
            if len(dataset.variables[name]) != ntime:
                raise RuntimeError(f"{path}:{name} length does not match time")
        for variable in required_variables:
            expected_dimensions = (
                ("time", "lat", "lon")
                if variable == "PS"
                else ("time", "lev", "lat", "lon")
            )
            if tuple(dataset.variables[variable].dimensions) != expected_dimensions:
                raise RuntimeError(
                    f"{path}:{variable} dimensions are {dataset.variables[variable].dimensions}; "
                    f"expected {expected_dimensions}"
                )
        if reference_coordinates is not None:
            for name, reference in reference_coordinates.items():
                if not np.allclose(coordinates[name], reference, rtol=0.0, atol=1e-6):
                    raise RuntimeError(f"{path}:{name} differs from its case probe grid")

        evidence: dict[str, object] = {
            "ntime_probe": ntime,
            "lat_count": latitude.size,
            "lon_count": longitude.size,
            "lev_count": levels.size,
            "ilev_count": coordinates["ilev"].size,
            "lat_units": str(getattr(dataset.variables["lat"], "units", "MISSING")),
            "lon_units": str(getattr(dataset.variables["lon"], "units", "MISSING")),
            "lev_units": str(getattr(dataset.variables["lev"], "units", "MISSING")),
            "lat_values": ",".join(f"{item:.8g}" for item in latitude),
            "lon_values": ",".join(f"{item:.8g}" for item in longitude),
            "lev_values": ",".join(f"{item:.8g}" for item in levels),
            "model_attributes": waccm_model_attributes(dataset),
            "coordinates": {name: values.copy() for name, values in coordinates.items()},
        }
    return evidence


def waccm_schema(args: argparse.Namespace) -> None:
    """Audit one probe per WACCM case and, on request, every raw chunk."""
    if bool(args.case_column) == bool(args.fixed_case):
        raise RuntimeError("provide exactly one of --case-column or --fixed-case")
    required_variables = tuple(
        item.strip() for item in args.required_vars.split(",") if item.strip()
    )
    if not required_variables:
        raise RuntimeError("--required-vars must not be empty")
    grouped: dict[str, list[Path]] = defaultdict(list)
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or args.path_column not in reader.fieldnames:
            raise RuntimeError(f"inventory lacks path column {args.path_column!r}")
        if args.case_column and args.case_column not in reader.fieldnames:
            raise RuntimeError(f"inventory lacks case column {args.case_column!r}")
        for row in reader:
            case = args.fixed_case or row[args.case_column].strip()
            path_text = row[args.path_column].strip()
            if not case or not path_text:
                raise RuntimeError("WACCM schema inventory contains an empty case/path")
            grouped[case].append(Path(path_text))
    if not grouped:
        raise RuntimeError("WACCM schema inventory is empty")

    rows: list[dict[str, object]] = []
    for case, paths in sorted(grouped.items()):
        unique_paths = sorted(set(paths))
        if len(unique_paths) != len(paths):
            raise RuntimeError(f"WACCM schema inventory has duplicate paths for {case}")
        probe = unique_paths[0]
        probe_evidence = inspect_waccm_native_file(probe, required_variables)
        references = probe_evidence.pop("coordinates")
        audited_paths = unique_paths if args.all_files else [probe]
        for path in audited_paths[1:]:
            inspected = inspect_waccm_native_file(path, required_variables, references)
            inspected.pop("coordinates")
        rows.append(
            {
                "case": case,
                "required_variables": ",".join(required_variables),
                "audit_scope": "all_chunks" if args.all_files else "case_probe",
                "total_file_count": len(unique_paths),
                "audited_file_count": len(audited_paths),
                "probe_path": str(probe),
                **file_evidence(probe),
                **probe_evidence,
            }
        )
    atomic_tsv(
        args.output,
        (
            "case", "required_variables", "audit_scope", "total_file_count",
            "audited_file_count", "probe_path", "size_bytes", "mtime_ns",
            "ntime_probe", "lat_count", "lon_count", "lev_count", "ilev_count",
            "lat_units", "lon_units", "lev_units", "lat_values", "lon_values",
            "lev_values", "model_attributes",
        ),
        rows,
    )
    scope = "every chunk" if args.all_files else "one strict probe per case"
    print(f"validated native WACCM 96x144, 66-level schema for {scope}; wrote {args.output}")


def pressure_file(args: argparse.Namespace) -> None:
    """Validate one CDO ml2pl product against the exact Methods grid."""
    if not args.path.is_file() or args.path.stat().st_size <= 0:
        raise RuntimeError(f"pressure-level output is missing or empty: {args.path}")
    with netCDF4.Dataset(args.path) as dataset:
        if args.variable not in dataset.variables:
            raise RuntimeError(f"{args.path} lacks {args.variable}")
        coordinate_name = next(
            (name for name in ("plev", "lev") if name in dataset.variables), None
        )
        if coordinate_name is None:
            raise RuntimeError(f"{args.path} lacks plev/lev coordinate")
        coordinate = dataset.variables[coordinate_name]
        values = np.asarray(coordinate[:], dtype=float).reshape(-1)
        units = compact_units(getattr(coordinate, "units", ""))
        if units in {"hpa", "mb", "millibar", "millibars"}:
            values_pa = values * 100.0
        elif units in {"pa", "pascal", "pascals"}:
            values_pa = values
        else:
            raise RuntimeError(f"{args.path} has unsupported pressure units {units!r}")
        if values_pa.shape != WACCM_PLEV_PA.shape or not np.allclose(
            values_pa, WACCM_PLEV_PA, rtol=0.0, atol=1e-5
        ):
            raise RuntimeError(
                f"{args.path} pressure grid/order differs from exact 36-level Methods grid: "
                f"{values_pa.tolist()}"
            )
        dimensions = dataset.variables[args.variable].dimensions
        if coordinate_name not in dimensions:
            raise RuntimeError(
                f"{args.path}:{args.variable} does not use {coordinate_name}: {dimensions}"
            )
    print(f"validated exact 36-level pressure grid: {args.path}")


def noleap_day_number(raw_date: int) -> int:
    year, month_day = divmod(int(raw_date), 10000)
    month, day = divmod(month_day, 100)
    lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if month < 1 or month > 12 or day < 1 or day > lengths[month - 1]:
        raise RuntimeError(f"invalid noleap YYYYMMDD value: {raw_date}")
    return (year - 1) * 365 + sum(lengths[: month - 1]) + day - 1


def hindcast_prefixes(directory: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(directory.glob("*.cam.h3.*.nc*")):
        match = HINDCAST_CHUNK_RE.match(path.name)
        if match is not None:
            grouped[match.group("prefix")].append(path)
    return grouped


def inspect_hindcast_chunks(paths: list[Path]) -> np.ndarray:
    dates: list[int] = []
    for path in paths:
        with netCDF4.Dataset(path) as dataset:
            missing = [
                name for name in (*HINDCAST_VARIABLES, "date", "datesec", "time")
                if name not in dataset.variables
            ]
            if missing:
                raise RuntimeError(f"{path} is missing required fields: {missing}")
            ntime = len(dataset.dimensions.get("time", ()))
            chunk_dates = np.asarray(dataset.variables["date"][:], dtype=np.int64).reshape(-1)
            if ntime <= 0 or chunk_dates.size != ntime:
                raise RuntimeError(f"{path} has an empty/mismatched time/date axis")
            datesec = np.asarray(dataset.variables["datesec"][:]).reshape(-1)
            time_values = np.asarray(dataset.variables["time"][:], dtype=float).reshape(-1)
            if datesec.size != ntime or time_values.size != ntime or not np.all(np.isfinite(time_values)):
                raise RuntimeError(f"{path} has mismatched datesec/time records")
            calendar_name = str(
                getattr(dataset.variables["time"], "calendar", "")
            ).lower()
            if calendar_name not in {"noleap", "365_day"}:
                raise RuntimeError(f"{path} has wrong/missing noleap calendar")
            for variable in HINDCAST_VARIABLES:
                if "time" not in dataset.variables[variable].dimensions:
                    raise RuntimeError(f"{path}:{variable} lacks the time dimension")
            dates.extend(int(item) for item in chunk_dates)
    if not dates:
        raise RuntimeError("hindcast member contains no raw dates")
    array = np.asarray(dates, dtype=np.int64)
    ordinals = np.asarray([noleap_day_number(item) for item in array], dtype=np.int64)
    if not np.all(np.diff(ordinals) == 1):
        raise RuntimeError("hindcast member chunks are not strictly continuous daily records")
    if set(int(item) // 10000 for item in array) != {8}:
        raise RuntimeError("hindcast raw dates must remain in model year 0008")
    return array


def hindcast_window_contract(
    case: str, prefix_name: str, dates: np.ndarray
) -> tuple[int, int, int]:
    """Require initialization and every retained plot's latest source date."""
    expected_start = 80101 if case == "0008-01" else 80201
    if int(dates[0]) != expected_start:
        raise RuntimeError(
            f"{case}/{prefix_name} initializes {dates[0]:08d}, "
            f"expected {expected_start:08d}"
        )
    if int(dates[-1]) < 80531:
        raise RuntimeError(
            f"{case}/{prefix_name} ends before 0008-05-31, required by the "
            f"event-day-150 display products: {dates[-1]:08d}"
        )
    if case == "0008-01":
        figure7 = dates[(dates >= 80101) & (dates <= 80530)]
        if (
            figure7.size != 150
            or int(figure7[0]) != 80101
            or int(figure7[-1]) != 80530
        ):
            raise RuntimeError(
                f"{case}/{prefix_name} does not provide Figure 7's exact "
                "150 daily records from 0008-01-01 through 0008-05-30"
            )
    return dates.size, int(dates[0]), int(dates[-1])


def hindcast(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    ids_by_case: dict[str, set[str]] = {}
    contract_by_case: dict[str, tuple[int, int, int]] = {}
    for case in ("0008-01", "0008-02"):
        directory = args.source_root / case
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        prefixes = hindcast_prefixes(directory)
        if len(prefixes) != 30:
            raise RuntimeError(f"{case} must contain exactly 30 raw member prefixes")
        normalized: set[str] = set()
        for prefix_name, paths in sorted(prefixes.items()):
            member_id = prefix_name.replace(case, "CASE", 1)
            if member_id in normalized:
                raise RuntimeError(f"duplicate normalized member ID in {case}: {member_id}")
            normalized.add(member_id)
            dates = inspect_hindcast_chunks(paths)
            contract = hindcast_window_contract(case, prefix_name, dates)
            if case in contract_by_case and contract_by_case[case] != contract:
                raise RuntimeError(
                    f"{case} member end/length differs: {contract_by_case[case]} vs {contract}"
                )
            contract_by_case[case] = contract
            rows.append(
                {
                    "case": case,
                    "normalized_member_id": member_id,
                    "prefix": str(directory / prefix_name),
                    "chunk_count": len(paths),
                    "ntime": dates.size,
                    "start_date": f"{int(dates[0]):08d}",
                    "end_date": f"{int(dates[-1]):08d}",
                }
            )
        ids_by_case[case] = normalized
    if ids_by_case["0008-01"] != ids_by_case["0008-02"]:
        raise RuntimeError("January/February hindcasts do not share the exact 30 member IDs")
    atomic_tsv(
        args.output,
        (
            "case", "normalized_member_id", "prefix", "chunk_count", "ntime",
            "start_date", "end_date",
        ),
        rows,
    )
    print(f"validated every raw chunk for 30+30 hindcast members; wrote {args.output}")


def hindcast_outputs(args: argparse.Namespace) -> None:
    source_contract: dict[tuple[str, str], tuple[int, int, int, str]] = {}
    with args.source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            prefix_name = Path(row["prefix"]).name
            source_contract[(row["case"], prefix_name)] = (
                int(row["ntime"]), int(row["start_date"]), int(row["end_date"]),
                row["normalized_member_id"],
            )
    if len(source_contract) != 60:
        raise RuntimeError("hindcast source manifest must contain exactly 30+30 members")
    rows: list[dict[str, object]] = []
    for (case, prefix_name), (ntime, start, end, member_id) in sorted(source_contract.items()):
        for variable in HINDCAST_VARIABLES:
            path = args.root / case / variable / f"{prefix_name}.{variable}.nc"
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing/empty hindcast output: {path}")
            with netCDF4.Dataset(path) as dataset:
                missing = [name for name in (variable, "date", "datesec", "time") if name not in dataset.variables]
                if missing:
                    raise RuntimeError(f"{path} is missing {missing}")
                dates = np.asarray(dataset.variables["date"][:], dtype=np.int64).reshape(-1)
                datesec = np.asarray(dataset.variables["datesec"][:]).reshape(-1)
                time_values = np.asarray(dataset.variables["time"][:], dtype=float).reshape(-1)
                ordinals = np.asarray([noleap_day_number(item) for item in dates], dtype=np.int64)
                if (
                    dates.size != ntime
                    or datesec.size != ntime
                    or time_values.size != ntime
                    or not np.all(np.isfinite(time_values))
                    or int(dates[0]) != start
                    or int(dates[-1]) != end
                    or not np.all(np.diff(ordinals) == 1)
                ):
                    raise RuntimeError(f"{path} does not match the raw member date contract")
                if "time" not in dataset.variables[variable].dimensions:
                    raise RuntimeError(f"{path}:{variable} lacks the time dimension")
                calendar_name = str(getattr(dataset.variables["time"], "calendar", "")).lower()
                if calendar_name not in {"noleap", "365_day"}:
                    raise RuntimeError(f"{path} has wrong/missing noleap calendar")
            rows.append(
                {
                    "case": case,
                    "normalized_member_id": member_id,
                    "variable": variable,
                    "path": str(path),
                    "ntime": ntime,
                    "start_date": f"{start:08d}",
                    "end_date": f"{end:08d}",
                    **file_evidence(path),
                }
            )
    if len(rows) != 360:
        raise RuntimeError(f"expected 360 hindcast outputs; validated {len(rows)}")
    atomic_tsv(
        args.output,
        (
            "case", "normalized_member_id", "variable", "path", "size_bytes",
            "mtime_ns", "ntime", "start_date", "end_date",
        ),
        rows,
    )
    print(f"validated all 360 retained/new hindcast outputs; wrote {args.output}")


def march_id(path: Path, case: str) -> str:
    marker = f".{case}."
    if marker not in path.name or not path.name.endswith(".nc"):
        raise ValueError(f"unexpected March-restart filename: {path.name}")
    return path.name.split(marker, 1)[1][:-3]


def march(args: argparse.Namespace) -> None:
    case_dirs = {"0008-02": args.source_root / "Feb", "0008-03": args.source_root / "Mar"}
    paths_by_case: dict[str, dict[str, Path]] = {}
    for case, directory in case_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        found = sorted(directory.glob(f"BWCN.e122.f19_g16.002.{case}.*.nc"))
        if len(found) != 30:
            raise RuntimeError(f"expected 30 March-restart {case} files; found {len(found)}")
        mapped = {march_id(path, case): path for path in found}
        if len(mapped) != 30:
            raise RuntimeError(f"duplicate March-restart member IDs in {case}")
        paths_by_case[case] = mapped
    common = sorted(set(paths_by_case["0008-02"]) & set(paths_by_case["0008-03"]))
    if len(common) != 30 or set(paths_by_case["0008-02"]) != set(paths_by_case["0008-03"]):
        raise RuntimeError("February and March restart cases do not share the same 30 member IDs")

    rows: list[dict[str, object]] = []
    required_variables = ("U", "V", "T", "Z3", "O3")
    exact_time_contracts = {
        "0008-02": (121, "0008-02-01", "0008-06-01"),
        "0008-03": (123, "0008-03-01", "0008-07-01"),
    }
    reference_levels_pa: np.ndarray | None = None
    for case in ("0008-02", "0008-03"):
        expected_start = f"0008-{case[-2:]}-01"
        for member_id in common:
            path = paths_by_case[case][member_id]
            with netCDF4.Dataset(path) as dataset:
                missing = [name for name in required_variables if name not in dataset.variables]
                if missing:
                    raise RuntimeError(f"{path} is missing variables: {missing}")
                if "plev" not in dataset.variables:
                    raise RuntimeError(f"{path} has no plev coordinate")
                levels = np.asarray(dataset.variables["plev"][:], dtype=float).reshape(-1)
                units = str(getattr(dataset.variables["plev"], "units", "")).lower()
                normalized_units = compact_units(units)
                if normalized_units in {"hpa", "mb", "millibar", "millibars"}:
                    levels_pa = levels * 100.0
                elif normalized_units in {"pa", "pascal", "pascals"}:
                    levels_pa = levels
                else:
                    raise RuntimeError(f"{path} has invalid plev units: {units!r}")
                if levels_pa.shape != MARCH_CANONICAL_23_PA.shape or not np.allclose(
                    levels_pa, MARCH_CANONICAL_23_PA, rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError(
                        f"{path} must use the exact ordered canonical 23-level grid"
                    )
                if reference_levels_pa is None:
                    reference_levels_pa = levels_pa.copy()
                elif not np.allclose(levels_pa, reference_levels_pa, rtol=0.0, atol=1e-6):
                    raise RuntimeError(f"{path} does not use the common canonical 23-level grid")
                absent = [
                    float(level)
                    for level in MARCH_REQUIRED_LEVELS_PA
                    if not np.any(np.isclose(levels_pa, level, rtol=0.0, atol=1.0))
                ]
                if absent:
                    raise RuntimeError(f"{path} is missing required plev values (Pa): {absent}")
                for variable in required_variables:
                    dimensions = dataset.variables[variable].dimensions
                    if "time" not in dimensions or "plev" not in dimensions:
                        raise RuntimeError(
                            f"{path}:{variable} must use time and plev dimensions: {dimensions}"
                        )
                if "time" not in dataset.variables or "time" not in dataset.dimensions:
                    raise RuntimeError(f"{path} has no time coordinate")
                time_variable = dataset.variables["time"]
                ntime = len(dataset.dimensions["time"])
                time_values = np.asarray(time_variable[:], dtype=float).reshape(-1)
                time_units = str(getattr(time_variable, "units", ""))
                time_calendar = str(getattr(time_variable, "calendar", "")).lower()
                if ntime <= 0 or time_values.size != ntime or not np.all(np.isfinite(time_values)):
                    raise RuntimeError(f"{path} has invalid time records")
                if time_calendar not in {"noleap", "365_day"}:
                    raise RuntimeError(f"{path} has wrong calendar: {time_calendar!r}")
                if expected_start not in time_units:
                    raise RuntimeError(
                        f"{path} time units must be based on {expected_start}: {time_units!r}"
                    )
                if ntime > 1 and not np.allclose(np.diff(time_values), 1.0, rtol=0.0, atol=1e-8):
                    raise RuntimeError(f"{path} time coordinate is not strictly daily")
                decoded = netCDF4.num2date(
                    time_values,
                    units=time_units,
                    calendar=time_calendar,
                    only_use_cftime_datetimes=True,
                )
                decoded_dates = np.asarray(
                    [
                        int(item.year) * 10000 + int(item.month) * 100 + int(item.day)
                        for item in decoded
                    ],
                    dtype=np.int64,
                )
                decoded_ordinals = np.asarray(
                    [noleap_day_number(int(item)) for item in decoded_dates],
                    dtype=np.int64,
                )
                if decoded_dates.size != ntime or not np.all(np.diff(decoded_ordinals) == 1):
                    raise RuntimeError(f"{path} decoded dates are not strictly continuous daily")
                start_date = decoded[0].strftime("%Y-%m-%d")
                end_date = decoded[-1].strftime("%Y-%m-%d")
                if start_date != expected_start:
                    raise RuntimeError(f"{path} starts {start_date}, expected {expected_start}")
                contract = (ntime, start_date, end_date)
                if contract != exact_time_contracts[case]:
                    raise RuntimeError(
                        f"{path} time coverage is {contract}, expected "
                        f"{exact_time_contracts[case]}"
                    )
                if 80531 not in decoded_dates:
                    raise RuntimeError(
                        f"{path} does not cover 0008-05-31 required by common displays"
                    )
                rows.append(
                    {
                        "case": case,
                        "member_id": member_id,
                        "path": str(path),
                        "ntime": ntime,
                        "plev_count": levels.size,
                        "plev_units": units,
                        "plev_values_pa": ",".join(f"{item:g}" for item in levels_pa),
                        "calendar": time_calendar,
                        "time_units": time_units,
                        "start_date": start_date,
                        "end_date": end_date,
                        "includes_0008_05_31": 1,
                        **file_evidence(path),
                    }
                )
    atomic_tsv(
        args.output,
        (
            "case", "member_id", "path", "size_bytes", "mtime_ns", "ntime",
            "start_date", "end_date", "calendar", "time_units", "plev_count", "plev_units",
            "plev_values_pa", "includes_0008_05_31",
        ),
        rows,
    )
    print(f"validated restart cases Feb=30, Mar=30, common=30; wrote {args.output}")


def expected_dates(year: int) -> list[date]:
    start = date(year, 1, 1)
    return [start + timedelta(days=offset) for offset in range(366 if calendar.isleap(year) else 365)]


def merra_records(root: Path, years: range) -> dict[date, Path]:
    records: dict[date, Path] = {}
    for path in sorted(root.glob("MERRA2_*.inst6_3d_ana_Np.*.SUB.nc")):
        match = MERRA_RE.match(path.name)
        if not match:
            continue
        text = match.group("date")
        item = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        if item.year not in years:
            continue
        if item in records:
            raise RuntimeError(f"duplicate MERRA-2 date {item}: {records[item]} and {path}")
        records[item] = path
    return records


def validate_merra_grid(dataset: netCDF4.Dataset, path: Path, variables: tuple[str, ...]) -> None:
    if "lat" not in dataset.variables or "lon" not in dataset.variables:
        raise RuntimeError(f"{path} lacks lat/lon coordinates")
    latitude = np.asarray(dataset.variables["lat"][:], dtype=float).reshape(-1)
    longitude = np.asarray(dataset.variables["lon"][:], dtype=float).reshape(-1)
    if latitude.size != 361 or longitude.size != 576:
        raise RuntimeError(
            f"{path} is not the native 0.5 x 0.625 grid: "
            f"lat={latitude.size}, lon={longitude.size}"
        )
    if not (
        np.isclose(latitude[0], -90.0)
        and np.isclose(latitude[-1], 90.0)
        and np.allclose(np.diff(latitude), 0.5, rtol=0.0, atol=1e-6)
    ):
        raise RuntimeError(f"{path} latitude is not -90..90 at 0.5 degree spacing")
    if not np.allclose(np.diff(longitude), 0.625, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"{path} longitude is not spaced by 0.625 degree")
    if not (
        np.isclose(longitude[-1] - longitude[0], 359.375, atol=1e-6)
        and (
            np.isclose(longitude[0], -180.0, atol=1e-6)
            or np.isclose(longitude[0], 0.0, atol=1e-6)
        )
    ):
        raise RuntimeError(f"{path} longitude range is not a native global grid")
    if compact_units(getattr(dataset.variables["lat"], "units", "")) not in {
        "degreesnorth", "degreenorth"
    }:
        raise RuntimeError(f"{path} has invalid latitude units")
    if compact_units(getattr(dataset.variables["lon"], "units", "")) not in {
        "degreeseast", "degreeeast"
    }:
        raise RuntimeError(f"{path} has invalid longitude units")
    if "lev" not in dataset.variables or len(dataset.variables["lev"]) <= 0:
        raise RuntimeError(f"{path} lacks pressure levels")
    if compact_units(getattr(dataset.variables["lev"], "units", "")) not in {"hpa", "mb"}:
        raise RuntimeError(f"{path} pressure units are not hPa/mb")
    pressure = np.asarray(dataset.variables["lev"][:], dtype=float).reshape(-1)
    is_height = variables in {("H",), ("Z3",)}
    if is_height:
        ordered_unique = (
            np.unique(pressure).size == pressure.size
            and (
                np.all(np.diff(pressure) < 0.0)
                or np.all(np.diff(pressure) > 0.0)
            )
        )
        missing_nam = [
            float(level)
            for level in MERRA_NAM_23_HPA
            if not np.any(np.isclose(pressure, level, rtol=0.0, atol=1e-5))
        ]
        if not ordered_unique or missing_nam:
            raise RuntimeError(
                f"{path} Z3/H grid must be strictly ordered and contain the full "
                f"23-level NAM grid; missing_hPa={missing_nam}"
            )
    elif not (
        pressure.shape == MERRA_PROCESSED_18_HPA.shape
        and np.allclose(pressure, MERRA_PROCESSED_18_HPA, rtol=0.0, atol=1e-5)
    ):
        raise RuntimeError(f"{path} is not the exact processed 18-level EP grid")
    if "time" not in dataset.variables:
        raise RuntimeError(f"{path} lacks a time coordinate variable")
    time_units = str(getattr(dataset.variables["time"], "units", ""))
    if "since" not in time_units.lower():
        raise RuntimeError(f"{path} time coordinate lacks CF units: {time_units!r}")

    expected_units = {
        "U": {"ms-1", "m/s"},
        "V": {"ms-1", "m/s"},
        "T": {"k", "kelvin"},
        "O3": {"kgkg-1", "kg/kg"},
        "H": {"m", "meter", "metre"},
        "Z3": {"m", "meter", "metre"},
    }
    for variable in variables:
        array = dataset.variables[variable]
        if tuple(array.dimensions) != ("time", "lev", "lat", "lon"):
            raise RuntimeError(f"{path}:{variable} has unexpected dimensions {array.dimensions}")
        units = compact_units(getattr(array, "units", ""))
        if units not in expected_units[variable]:
            raise RuntimeError(f"{path}:{variable} has unexpected units {units!r}")


def source_mode(
    path: Path, variables: tuple[str, ...], *, check_grid: bool = False
) -> tuple[str, str]:
    with netCDF4.Dataset(path) as dataset:
        missing = [name for name in variables if name not in dataset.variables]
        if missing:
            raise RuntimeError(f"{path} is missing variables: {missing}")
        if "time" not in dataset.dimensions:
            raise RuntimeError(f"{path} has no time dimension")
        ntime = len(dataset.dimensions["time"])
        attrs = {name: str(dataset.getncattr(name)) for name in dataset.ncattrs()}
        metadata = " | ".join(attrs.values())
        short_name = next(
            (value for name, value in attrs.items() if name.lower() == "shortname"), ""
        )
        if short_name.strip().upper() != "M2I6NPANA":
            raise RuntimeError(f"{path} has wrong or missing ShortName: {short_name!r}")
        match = MERRA_RE.match(path.name)
        if match is None:
            raise RuntimeError(f"unexpected MERRA-2 filename: {path.name}")
        granule_key = (
            f"MERRA2_{match.group('collection')}.inst6_3d_ana_Np.{match.group('date')}"
        )
        if granule_key not in metadata:
            raise RuntimeError(f"{path} lacks collection/granule metadata for {granule_key}")
        filename_day = date(
            int(match.group("date")[:4]),
            int(match.group("date")[4:6]),
            int(match.group("date")[6:8]),
        )
        time_variable = dataset.variables["time"]
        time_values = np.asarray(time_variable[:], dtype=float).reshape(-1)
        time_units = str(getattr(time_variable, "units", ""))
        time_calendar = str(getattr(time_variable, "calendar", "standard"))
        if time_values.size != ntime or not np.all(np.isfinite(time_values)):
            raise RuntimeError(f"{path} has invalid time values")
        decoded = netCDF4.num2date(
            time_values,
            units=time_units,
            calendar=time_calendar,
            only_use_cftime_datetimes=True,
        )
        if any(
            (int(item.year), int(item.month), int(item.day))
            != (filename_day.year, filename_day.month, filename_day.day)
            for item in decoded
        ):
            raise RuntimeError(f"{path} time coordinate does not match filename date")
        if "time_bnds" not in dataset.variables:
            raise RuntimeError(f"{path} lacks time_bnds")
        bounds = np.asarray(dataset.variables["time_bnds"][:], dtype=float)
        if bounds.shape != (ntime, 2) or not np.all(np.isfinite(bounds)):
            raise RuntimeError(f"{path} has invalid time_bnds shape/values: {bounds.shape}")
        if not np.all(bounds[:, 0] <= time_values) or not np.all(time_values <= bounds[:, 1]):
            raise RuntimeError(f"{path} time records are outside time_bnds")
        if check_grid:
            validate_merra_grid(dataset, path, variables)
        if ntime == 1:
            daily_evidence = "daily mean" in metadata.lower()
            begin = attrs.get("RangeBeginningTime", "")
            end = attrs.get("RangeEndingTime", "")
            range_evidence = begin.startswith("00") and end.startswith("18")
            if not (daily_evidence and range_evidence):
                raise RuntimeError(
                    f"one-record granule lacks Daily mean / 00-18 UTC provenance: {path}"
                )
            # GES DISC L34RS daily-subset files may retain the explicit
            # global 00/18 UTC provenance but collapse ``time_bnds`` to the
            # single daily label (for example [[0, 0]]).  Accept that exact
            # representation only when the Daily-mean history and Range attrs
            # above are both present; otherwise require literal 00--18 bounds.
            collapsed_label = (
                np.allclose(bounds[:, 0], time_values, rtol=0.0, atol=1e-6)
                and np.allclose(bounds[:, 1], time_values, rtol=0.0, atol=1e-6)
            )
            if collapsed_label:
                return (
                    "gesdisc_daily_mean",
                    "Daily mean; RangeBeginningTime=00, RangeEndingTime=18; "
                    "L34RS time_bnds collapsed to daily label",
                )
            decoded_bounds = netCDF4.num2date(
                bounds.reshape(-1),
                units=time_units,
                calendar=time_calendar,
                only_use_cftime_datetimes=True,
            )
            bound_hours = [item.hour + item.minute / 60.0 for item in decoded_bounds]
            bound_dates = [
                (int(item.year), int(item.month), int(item.day)) for item in decoded_bounds
            ]
            if (
                bound_dates != [
                    (filename_day.year, filename_day.month, filename_day.day),
                    (filename_day.year, filename_day.month, filename_day.day),
                ]
                or not np.allclose(bound_hours, [0.0, 18.0], rtol=0.0, atol=1e-6)
            ):
                raise RuntimeError(f"{path} daily-mean bounds are not 00-18 UTC")
            return "gesdisc_daily_mean", "Daily mean; explicit 00-18 UTC time_bnds"
        if ntime == 4:
            hours = [item.hour + item.minute / 60.0 for item in decoded]
            if not np.allclose(hours, [0.0, 6.0, 12.0, 18.0], rtol=0.0, atol=1e-6):
                raise RuntimeError(f"{path} four records are not 00/06/12/18 UTC: {hours}")
            return "four_inst6_records", "four 6-hourly time records; local daymean required"
        raise RuntimeError(f"expected 1 daily-mean or 4 inst6 records in {path}; found {ntime}")


def merra(args: argparse.Namespace) -> None:
    years = range(args.start_year, args.end_year + 1)
    core_records = merra_records(args.core_root, years)
    z_records = merra_records(args.z_root, years)
    expected_all = {item for year in years for item in expected_dates(year)}
    for label, records in (("core", core_records), ("Z/H", z_records)):
        missing = sorted(expected_all - set(records))
        extra = sorted(set(records) - expected_all)
        if missing or extra:
            raise RuntimeError(
                f"MERRA-2 {label} coverage mismatch: missing={len(missing)}, extra={len(extra)}, "
                f"first_missing={[item.isoformat() for item in missing[:10]]}"
            )
    if set(core_records) != set(z_records):
        raise RuntimeError("MERRA-2 core and Z/H source dates differ")

    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for year in years:
        dates = expected_dates(year)
        core_modes: set[str] = set()
        z_modes: set[str] = set()
        core_evidence: set[str] = set()
        z_evidence: set[str] = set()
        for index, item in enumerate(dates):
            mode, evidence = source_mode(
                core_records[item], ("U", "V", "T", "O3"), check_grid=index == 0
            )
            core_modes.add(mode)
            core_evidence.add(evidence)
            core_match = MERRA_RE.match(core_records[item].name)
            raw_rows.append(
                {
                    "family": "U,V,T,O3",
                    "date": item.isoformat(),
                    "collection": core_match.group("collection") if core_match else "",
                    "mode": mode,
                    "path": str(core_records[item]),
                    **file_evidence(core_records[item]),
                }
            )
            mode, evidence = source_mode(z_records[item], ("H",), check_grid=index == 0)
            z_modes.add(mode)
            z_evidence.add(evidence)
            z_match = MERRA_RE.match(z_records[item].name)
            raw_rows.append(
                {
                    "family": "H_to_Z3",
                    "date": item.isoformat(),
                    "collection": z_match.group("collection") if z_match else "",
                    "mode": mode,
                    "path": str(z_records[item]),
                    **file_evidence(z_records[item]),
                }
            )
        if len(core_modes) != 1 or len(z_modes) != 1:
            raise RuntimeError(f"mixed MERRA-2 source modes in {year}: {core_modes}, {z_modes}")
        rows.append(
            {
                "year": year,
                "expected_days": len(dates),
                "core_files": len(dates),
                "z_files": len(dates),
                "core_mode": next(iter(core_modes)),
                "z_mode": next(iter(z_modes)),
                "core_evidence": "; ".join(sorted(core_evidence)),
                "z_evidence": "; ".join(sorted(z_evidence)),
                "core_first": str(core_records[dates[0]]),
                "z_first": str(z_records[dates[0]]),
            }
        )
    atomic_tsv(
        args.output,
        (
            "year", "expected_days", "core_files", "z_files", "core_mode", "z_mode",
            "core_evidence", "z_evidence", "core_first", "z_first",
        ),
        rows,
    )
    raw_output = args.output.with_name(
        f"merra2_raw_files_{args.start_year}_{args.end_year}.tsv"
    )
    atomic_tsv(
        raw_output,
        ("family", "date", "collection", "mode", "path", "size_bytes", "mtime_ns"),
        raw_rows,
    )
    print(
        f"validated MERRA-2 {args.start_year}-{args.end_year}; "
        f"wrote {args.output} and {raw_output}"
    )


def merra_yearly(args: argparse.Namespace) -> None:
    """Validate every retained/new annual MERRA-2 staging product."""
    specifications = (
        ("U", "MERRA2.U.{year}.nc"),
        ("V", "MERRA2.V.{year}.nc"),
        ("T", "MERRA2.T.{year}.nc"),
        ("O3", "MERRA2.O3.{year}.nc"),
        ("Z3", "MERRA2.Z3.{year}.nc"),
    )
    reference_z3_levels: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for variable, pattern in specifications:
        for year in range(args.start_year, args.end_year + 1):
            path = args.root / variable / pattern.format(year=year)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing/empty annual MERRA-2 output: {path}")
            with netCDF4.Dataset(path) as dataset:
                if variable not in dataset.variables:
                    raise RuntimeError(f"{path} lacks {variable}")
                validate_merra_grid(dataset, path, (variable,))
                pressure = np.asarray(dataset.variables["lev"][:], dtype=float).reshape(-1)
                pressure_units = compact_units(getattr(dataset.variables["lev"], "units", ""))
                if pressure.size == 0 or np.unique(pressure).size != pressure.size:
                    raise RuntimeError(f"{path} has an empty/duplicate pressure grid")
                if not (np.all(np.diff(pressure) < 0.0) or np.all(np.diff(pressure) > 0.0)):
                    raise RuntimeError(f"{path} pressure grid is not strictly ordered")
                if variable != "Z3":
                    if pressure.shape != MERRA_PROCESSED_18_HPA.shape or not np.allclose(
                        pressure, MERRA_PROCESSED_18_HPA, rtol=0.0, atol=1e-5
                    ):
                        raise RuntimeError(f"{path} is not the exact processed 18-level grid")
                elif reference_z3_levels is None:
                    reference_z3_levels = pressure.copy()
                elif not np.allclose(pressure, reference_z3_levels, rtol=0.0, atol=1e-5):
                    raise RuntimeError(f"{path} Z3 pressure grid differs between years")

                time_variable = dataset.variables["time"]
                time_values = np.asarray(time_variable[:], dtype=float).reshape(-1)
                units = str(getattr(time_variable, "units", ""))
                time_calendar = str(getattr(time_variable, "calendar", "standard"))
                expected = expected_dates(year)
                if time_values.size != len(expected) or not np.all(np.isfinite(time_values)):
                    raise RuntimeError(
                        f"{path} has invalid time values/count={time_values.size}; "
                        f"expected {len(expected)}"
                    )
                decoded = netCDF4.num2date(
                    time_values,
                    units=units,
                    calendar=time_calendar,
                    only_use_cftime_datetimes=True,
                )
                decoded_dates = [
                    (int(item.year), int(item.month), int(item.day)) for item in decoded
                ]
                expected_tuples = [(item.year, item.month, item.day) for item in expected]
                if decoded_dates != expected_tuples:
                    raise RuntimeError(f"{path} does not have the exact daily {year} axis")
                rows.append(
                    {
                        "variable": variable,
                        "year": year,
                        "path": str(path),
                        "ntime": len(expected),
                        "first_date": expected[0].isoformat(),
                        "last_date": expected[-1].isoformat(),
                        "nlev": pressure.size,
                        "lev_units": pressure_units,
                        "lev_values_hpa": ",".join(f"{item:g}" for item in pressure),
                        **file_evidence(path),
                    }
                )
    atomic_tsv(
        args.output,
        (
            "variable", "year", "path", "size_bytes", "mtime_ns", "ntime",
            "first_date", "last_date", "nlev", "lev_units", "lev_values_hpa",
        ),
        rows,
    )
    print(
        f"validated {len(rows)} annual MERRA-2 files for "
        f"{args.start_year}-{args.end_year}; wrote {args.output}"
    )


def one_mls_file(root: Path, product: str, year: int) -> Path:
    matches = sorted(root.glob(f"MLS-Aura_L3DZ-{product}_v05-*_{year}.nc"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one MLS {product} file for {year}; found {matches}")
    return matches[0]


def unique_calendar_day_coverage(
    decoded: list[object] | np.ndarray, year: int, context: str
) -> tuple[int, int, float]:
    """Reject repeated/out-of-year decoded dates and summarize coverage."""
    decoded_years = {int(item.year) for item in decoded}
    if decoded_years != {year}:
        raise RuntimeError(
            f"{context} decoded dates are outside filename year {year}: "
            f"{sorted(decoded_years)}"
        )
    calendar_dates = [
        f"{int(item.year):04d}-{int(item.month):02d}-{int(item.day):02d}"
        for item in decoded
    ]
    unique_calendar_dates = set(calendar_dates)
    if len(unique_calendar_dates) != len(calendar_dates):
        counts: dict[str, int] = defaultdict(int)
        for item in calendar_dates:
            counts[item] += 1
        duplicates = sorted(item for item, count in counts.items() if count > 1)
        raise RuntimeError(f"{context} repeats calendar dates: {duplicates[:10]}")
    expected_calendar_days = 366 if calendar.isleap(year) else 365
    unique_count = len(unique_calendar_dates)
    return unique_count, expected_calendar_days, unique_count / expected_calendar_days


def mls(args: argparse.Namespace) -> None:
    specifications = (
        ("ClO", args.root / "ClO", "ClO PressureZM Day"),
        ("H2O", args.root / "H2O", "H2O PressureZM"),
    )
    rows: list[dict[str, object]] = []
    for product, directory, group_name in specifications:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for year in range(args.start_year, args.end_year + 1):
            path = one_mls_file(directory, product, year)
            with netCDF4.Dataset(path) as dataset:
                if group_name not in dataset.groups:
                    raise RuntimeError(f"{path} lacks exact group {group_name!r}")
                group = dataset.groups[group_name]
                version_attrs = {
                    f"root:{name}": str(dataset.getncattr(name))
                    for name in dataset.ncattrs()
                    if "version" in name.lower()
                }
                version_attrs.update(
                    {
                        f"group:{name}": str(group.getncattr(name))
                        for name in group.ncattrs()
                        if "version" in name.lower()
                    }
                )
                version_text = "; ".join(
                    f"{name}={value}" for name, value in sorted(version_attrs.items())
                )
                if not version_attrs or not any(
                    re.search(r"(^|[^0-9])0?5(?:\.|[^0-9]|$)", value)
                    for value in version_attrs.values()
                ):
                    raise RuntimeError(f"{path} lacks MLS V5 metadata: {version_text!r}")
                for variable in ("value", "nvalues"):
                    if variable not in group.variables:
                        raise RuntimeError(f"{path}:{group_name} lacks {variable}")
                for coordinate in ("time", "lev", "lat"):
                    if coordinate not in group.variables and coordinate not in group.dimensions:
                        raise RuntimeError(f"{path}:{group_name} lacks {coordinate}")
                value = group.variables["value"]
                nvalues = group.variables["nvalues"]
                if value.dimensions != nvalues.dimensions:
                    raise RuntimeError(f"{path}:{group_name} value/nvalues dimensions differ")
                value_units = compact_units(getattr(value, "units", ""))
                if value_units not in {
                    "molmol-1", "mol/mol", "vmr", "vv", "molefraction",
                    "volumemixingratio", "1", "dimensionless",
                }:
                    raise RuntimeError(
                        f"{path}:{group_name} must store unscaled mole fraction, not "
                        f"pre-scaled ppmv/ppbv: {value_units!r}"
                    )
                time_variable = group.variables["time"]
                pressure = np.asarray(group.variables["lev"][:], dtype=float).reshape(-1)
                latitude = np.asarray(group.variables["lat"][:], dtype=float).reshape(-1)
                ntime = len(group.dimensions["time"])
                if ntime < (90 if year == 2004 else 300):
                    raise RuntimeError(f"{path}:{group_name} has implausible ntime={ntime}")
                if pressure.size == 0 or np.nanmin(pressure) > 1.0 or np.nanmax(pressure) < 100.0:
                    raise RuntimeError(f"{path}:{group_name} does not span 1-100 hPa")
                if compact_units(getattr(group.variables["lev"], "units", "")) not in {
                    "hpa", "mb"
                }:
                    raise RuntimeError(f"{path}:{group_name} pressure units are not hPa")
                if latitude.size == 0 or np.nanmin(latitude) > 60.0 or np.nanmax(latitude) < 82.0:
                    raise RuntimeError(f"{path}:{group_name} does not span 60-82 N")
                if compact_units(getattr(group.variables["lat"], "units", "")) not in {
                    "degreesnorth", "degreenorth"
                }:
                    raise RuntimeError(f"{path}:{group_name} latitude units are invalid")
                time_units = str(getattr(time_variable, "units", ""))
                if "since" not in time_units.lower():
                    raise RuntimeError(f"{path}:{group_name} time units are invalid")
                time_values = np.asarray(time_variable[:], dtype=float).reshape(-1)
                if time_values.size != ntime or not np.all(np.diff(time_values) > 0.0):
                    raise RuntimeError(f"{path}:{group_name} time is not strictly increasing")
                time_calendar = str(getattr(time_variable, "calendar", "standard"))
                decoded = netCDF4.num2date(
                    time_values,
                    units=time_units,
                    calendar=time_calendar,
                    only_use_cftime_datetimes=True,
                )
                (
                    unique_calendar_days,
                    expected_calendar_days,
                    calendar_day_coverage_fraction,
                ) = unique_calendar_day_coverage(decoded, year, f"{path}:{group_name}")
                rows.append(
                    {
                        "product": product,
                        "year": year,
                        "group": group_name,
                        "path": str(path),
                        "ntime": ntime,
                        "unique_calendar_days": unique_calendar_days,
                        "expected_calendar_days": expected_calendar_days,
                        "calendar_day_coverage_fraction": calendar_day_coverage_fraction,
                        "nlev": pressure.size,
                        "nlat": latitude.size,
                        "value_units": value_units,
                        "first_date": decoded[0].strftime("%Y-%m-%d"),
                        "last_date": decoded[-1].strftime("%Y-%m-%d"),
                        "version_metadata": version_text,
                        **file_evidence(path),
                    }
                )
    atomic_tsv(
        args.output,
        (
            "product", "year", "group", "version_metadata", "path", "size_bytes",
            "mtime_ns", "ntime", "first_date", "last_date", "nlev", "nlat",
            "value_units", "unique_calendar_days", "expected_calendar_days",
            "calendar_day_coverage_fraction",
        ),
        rows,
    )
    print(f"validated MLS Level-3 {args.start_year}-{args.end_year}; wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    waccm_schema_parser = commands.add_parser("waccm-schema")
    waccm_schema_parser.add_argument("--inventory", type=Path, required=True)
    waccm_schema_parser.add_argument("--path-column", default="path")
    waccm_schema_parser.add_argument("--case-column")
    waccm_schema_parser.add_argument("--fixed-case")
    waccm_schema_parser.add_argument("--required-vars", required=True)
    waccm_schema_parser.add_argument("--all-files", action="store_true")
    waccm_schema_parser.add_argument("--output", type=Path, required=True)
    waccm_schema_parser.set_defaults(run=waccm_schema)

    pressure_parser = commands.add_parser("pressure-file")
    pressure_parser.add_argument("--path", type=Path, required=True)
    pressure_parser.add_argument("--variable", required=True)
    pressure_parser.set_defaults(run=pressure_file)

    hindcast_parser = commands.add_parser("hindcast")
    hindcast_parser.add_argument("--source-root", type=Path, required=True)
    hindcast_parser.add_argument("--output", type=Path, required=True)
    hindcast_parser.set_defaults(run=hindcast)

    hindcast_outputs_parser = commands.add_parser("hindcast-outputs")
    hindcast_outputs_parser.add_argument("--root", type=Path, required=True)
    hindcast_outputs_parser.add_argument("--source-manifest", type=Path, required=True)
    hindcast_outputs_parser.add_argument("--output", type=Path, required=True)
    hindcast_outputs_parser.set_defaults(run=hindcast_outputs)

    marina_parser = commands.add_parser("march")
    marina_parser.add_argument("--source-root", type=Path, required=True)
    marina_parser.add_argument("--output", type=Path, required=True)
    marina_parser.set_defaults(run=march)

    merra_parser = commands.add_parser("merra")
    merra_parser.add_argument("--core-root", type=Path, required=True)
    merra_parser.add_argument("--z-root", type=Path, required=True)
    merra_parser.add_argument("--start-year", type=int, default=1980)
    merra_parser.add_argument("--end-year", type=int, default=2025)
    merra_parser.add_argument("--output", type=Path, required=True)
    merra_parser.set_defaults(run=merra)

    merra_yearly_parser = commands.add_parser("merra-yearly")
    merra_yearly_parser.add_argument("--root", type=Path, required=True)
    merra_yearly_parser.add_argument("--start-year", type=int, default=1980)
    merra_yearly_parser.add_argument("--end-year", type=int, default=2025)
    merra_yearly_parser.add_argument("--output", type=Path, required=True)
    merra_yearly_parser.set_defaults(run=merra_yearly)

    mls_parser = commands.add_parser("mls")
    mls_parser.add_argument("--root", type=Path, required=True)
    mls_parser.add_argument("--start-year", type=int, default=2004)
    mls_parser.add_argument("--end-year", type=int, default=2020)
    mls_parser.add_argument("--output", type=Path, required=True)
    mls_parser.set_defaults(run=mls)
    return root


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
