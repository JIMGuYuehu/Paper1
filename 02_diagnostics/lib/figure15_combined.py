"""Canonical Figure-15 event assembly and 5,000-composite bootstrap.

The combined ranking table is authoritative for the WACCM low-ozone threshold
and flags.  Figure 15 itself uses only the 207 LONGRUN rows of that master,
then joins those IDs to the available Nov--Mar Z300/EP fields without
re-ranking.  BWCN is an explicitly excluded field scope, not missing data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from paper1_diagnostics import (
    EP_FREE_PLEV_HPA, cosine_latitude_mean, date_int, noleap_index,
    random_same_size_composites, select_latitude, select_pressure_levels,
)

MONTHS = (11, 12, 1, 2, 3)
MONTH_NAMES = ("Nov", "Dec", "Jan", "Feb", "Mar")
MONTH_LENGTH = {11: 30, 12: 31, 1: 31, 2: 28, 3: 31}
SEASON_LENGTH = 151


def _date_parts(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(dates, dtype=np.int64)
    return values // 10000, (values // 100) % 100, values % 100


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Cannot parse canonical Boolean flag {value!r}")
        return normalized == "true"
    return bool(value)


def _require_all_finite_daily_field(values: np.ndarray, *, context: str) -> None:
    """Reject a monthly mean if any retained daily grid value is missing."""

    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or array.shape[0] < 1:
        raise ValueError(f"{context}: expected time plus spatial dimensions")
    missing = int((~np.isfinite(array)).sum())
    if missing:
        raise ValueError(f"{context}: {missing} non-finite daily 300-hPa grid values")


def _monthly_z300(path: Path) -> dict[int, xr.DataArray]:
    """Read exact staged 300 hPa and form the five natural-month means."""

    with xr.open_dataset(path, decode_times=True, chunks={"time": 16}) as source:
        field = select_pressure_levels(source["Z3"], [300.0]).sel(plev=300.0)
        field = select_latitude(field, 20.0, 90.0)
        dates = date_int(source)
        _, months, days = _date_parts(dates)
        output = {}
        for month in MONTHS:
            selected = np.flatnonzero(
                (months == month) & ~((months == 2) & (days == 29))
            )
            if selected.size != MONTH_LENGTH[month]:
                raise ValueError(
                    f"{path.name}: month {month} has {selected.size} days; "
                    f"expected {MONTH_LENGTH[month]}"
                )
            daily = field.isel(time=selected)
            _require_all_finite_daily_field(
                np.asarray(daily.values, dtype=float), context=f"{path.name} month {month}"
            )
            output[month] = daily.mean("time", skipna=False).load()
    return output


def _load_ep_source(path: Path, *, model_year: bool) -> dict[str, object]:
    with xr.open_dataset(path, decode_times=False, chunks={"time": 365}) as source:
        required_attrs = {
            "natural_month_n2": "True", "do_ubar": "True",
            "w_argument": "None", "wave": "-1",
        }
        for name, expected in required_attrs.items():
            if str(source.attrs.get(name)) != expected:
                raise ValueError(f"{path}: {name} must be {expected!r}")
        if "fz_upward" not in source:
            raise ValueError(f"{path}: canonical full-latitude fz_upward is required")
        if not np.array_equal(np.asarray(source.plev.values, dtype=float), EP_FREE_PLEV_HPA):
            raise ValueError(f"{path}: Figure-15 EP must use the exact 18-level hPa grid")
        dates = np.asarray(source.date.values, dtype=int)
        labels = (
            np.asarray(source.model_year.values, dtype=int)
            if model_year else dates // 10000
        )
        flux = cosine_latitude_mean(source["fz_upward"], 40.0, 80.0)
        flux = flux.transpose("time", "plev").load()
    return {"flux": flux, "dates": dates, "labels": labels}


def _ep_event(source: Mapping[str, object], year: int) -> xr.DataArray:
    dates = np.asarray(source["dates"], dtype=int)
    labels = np.asarray(source["labels"], dtype=int)
    month = (dates // 100) % 100
    selected = (
        (((labels == year - 1) & np.isin(month, (11, 12)))
         | ((labels == year) & np.isin(month, (1, 2, 3))))
        & (noleap_index(dates) >= 0)
    )
    indices = np.flatnonzero(selected)
    indices = indices[np.argsort(dates[indices], kind="stable")]
    expected_months = np.concatenate(
        [np.full(MONTH_LENGTH[item], item, dtype=int) for item in MONTHS]
    )
    if indices.size != SEASON_LENGTH or not np.array_equal(month[indices], expected_months):
        raise ValueError(f"EP event {year}: complete ordered 151-day Nov--Mar season required")
    return (
        source["flux"].isel(time=indices).rename(time="season_day")
        .assign_coords(season_day=np.arange(SEASON_LENGTH))
        .transpose("plev", "season_day")
    )


def _z_event(cache, segment: str, year: int) -> xr.DataArray:
    values = [
        cache[(segment, year - 1)][11], cache[(segment, year - 1)][12],
        cache[(segment, year)][1], cache[(segment, year)][2],
        cache[(segment, year)][3],
    ]
    return xr.concat(values, dim=pd.Index(MONTH_NAMES, name="month"))


def prepare_figure15_source(
    *, label: str, rankings: pd.DataFrame,
    sources: Mapping[str, Mapping[str, object]], master_ranking_path: Path,
    require_all_ranked: bool, classification_master: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Join one declared field scope to fields while preserving master flags."""

    required = {
        "event_id", "source_segment", "event_year", "model_year",
        "minimum_du", "rank", "sample_size", "low25_count", "is_low25",
        "low25_threshold_du",
    }
    master = rankings if classification_master is None else classification_master
    for name, table in (("field-scope ranking", rankings), ("classification master", master)):
        if required - set(table):
            raise ValueError(f"{label}: {name} columns missing {sorted(required - set(table))}")
        if table.event_id.astype(str).duplicated().any():
            raise ValueError(f"{label}: {name} event_id must be unique")
    master_index = master.assign(event_id=master.event_id.astype(str)).set_index("event_id")
    scope_ids = rankings.event_id.astype(str).to_numpy()
    absent_from_master = sorted(set(scope_ids) - set(master_index.index))
    if absent_from_master:
        raise ValueError(f"{label}: field-scope IDs absent from master {absent_from_master[:5]}")
    inherited = master_index.loc[scope_ids]
    for column in ("source_segment", "event_year", "model_year", "rank", "is_low25"):
        left = rankings[column].astype(str).to_numpy()
        right = inherited[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"{label}: field-scope {column} differs from classification master")
    for column in (
        "minimum_du", "sample_size", "low25_count", "low25_threshold_du",
    ):
        left = rankings[column].to_numpy(dtype=float)
        right = inherited[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True):
            raise ValueError(f"{label}: field-scope {column} differs from classification master")
    if not set(rankings.source_segment.astype(str)).issubset(set(sources)):
        raise ValueError(f"{label}: field-scope ranking has a segment without source fields")
    ep_sources = {
        segment: _load_ep_source(
            Path(specification["ep_path"]),
            model_year=bool(specification["model_year"]),
        )
        for segment, specification in sources.items()
    }
    ep_available_years = {
        segment: set(np.asarray(source["labels"], dtype=int))
        for segment, source in ep_sources.items()
    }
    cache, rows, z_events, ep_events = {}, [], [], []
    unavailable_ids, unavailable_reasons = [], []
    for _, row in rankings.iterrows():
        segment, event_id = str(row.source_segment), str(row.event_id)
        specification = sources.get(segment)
        if specification is None:
            unavailable_ids.append(event_id)
            unavailable_reasons.append(f"{event_id}:no-source-spec")
            continue
        year = int(row.model_year if specification["model_year"] else row.event_year)
        z_files = specification["z_files"]
        if year not in z_files or year - 1 not in z_files:
            unavailable_ids.append(event_id)
            unavailable_reasons.append(f"{event_id}:Z300")
            continue
        if year not in ep_available_years[segment] or year - 1 not in ep_available_years[segment]:
            unavailable_ids.append(event_id)
            unavailable_reasons.append(f"{event_id}:EP")
            continue
        # Once all source years exist, an incomplete calendar or mismatched
        # grid is corruption, not ordinary field unavailability, and must fail.
        for source_year in (year - 1, year):
            key = (segment, source_year)
            if key not in cache:
                cache[key] = _monthly_z300(Path(z_files[source_year]))
        z_event = _z_event(cache, segment, year)
        ep_event = _ep_event(ep_sources[segment], year)
        if not np.isfinite(np.asarray(z_event.values, dtype=float)).all():
            raise ValueError(f"{event_id}: selected monthly Z300 field contains missing values")
        if not np.isfinite(np.asarray(ep_event.values, dtype=float)).all():
            raise ValueError(f"{event_id}: selected Nov--Mar EP field contains missing values")
        if z_events:
            for coordinate in ("lat", "lon"):
                if not np.array_equal(z_events[0][coordinate], z_event[coordinate]):
                    raise ValueError(f"{event_id}: Z300 {coordinate} grid differs")
            if not np.array_equal(ep_events[0].plev, ep_event.plev):
                raise ValueError(f"{event_id}: EP pressure grid differs")
        rows.append(row)
        z_events.append(z_event)
        ep_events.append(ep_event)
    if not rows:
        raise RuntimeError(f"{label}: no complete Figure-15 events")
    if require_all_ranked and unavailable_ids:
        raise RuntimeError(f"{label}: all ranked events required; unavailable={unavailable_reasons[:5]}")
    table = pd.DataFrame(rows).reset_index(drop=True)
    if require_all_ranked and len(table) != len(rankings):
        raise RuntimeError(f"{label}: accepted {len(table)}/{len(rankings)} events")
    if label == "WACCM":
        master_counts = master.groupby("source_segment").size().to_dict()
        scope_counts = rankings.groupby("source_segment").size().to_dict()
        if len(master) != 230 or master_counts != {"BWCN": 23, "LONGRUN": 207}:
            raise RuntimeError(
                f"WACCM ranking master must be LONGRUN:207 + BWCN:23; found {master_counts}"
            )
        if scope_counts != {"LONGRUN": 207}:
            raise RuntimeError(f"Figure 15 WACCM field scope must be LONGRUN:207; found {scope_counts}")
    event_ids = table.event_id.astype(str).to_numpy()
    z = xr.concat(z_events, dim=xr.IndexVariable("event_year", event_ids))
    ep = xr.concat(ep_events, dim=xr.IndexVariable("event_year", event_ids))
    low_mask = np.asarray([_as_bool(value) for value in table.is_low25], dtype=bool)
    if not 1 < int(low_mask.sum()) < low_mask.size:
        raise RuntimeError(f"{label}: invalid inherited low25 membership")
    manifest = ";".join(
        f"{segment}:Z3={Path(spec['z_root'])};EP={Path(spec['ep_path'])}"
        for segment, spec in sources.items()
    )
    excluded_master_ids = sorted(set(master.event_id.astype(str)) - set(scope_ids))
    return {
        "label": label, "rankings": rankings, "classification_master": master,
        "table": table,
        "event_ids": event_ids, "low_mask": low_mask, "z": z, "ep": ep,
        "master_ranking_path": Path(master_ranking_path),
        "source_manifest": manifest, "unavailable_ids": unavailable_ids,
        "unavailable_reasons": unavailable_reasons,
        "excluded_master_ids": excluded_master_ids,
    }


def compute_z300_monthly_stationary(state):
    climatology = state["z"].mean("event_year", skipna=True)
    state["z_centered"] = state["z"] - climatology
    state["z_stationary"] = climatology - climatology.mean("lon", skipna=True)
    return state


def compute_ep_calendar_standardized(state):
    mean = state["ep"].mean("event_year", skipna=True)
    standard_deviation = state["ep"].std("event_year", skipna=True, ddof=0)
    state["ep_standardized"] = (
        (state["ep"] - mean) / standard_deviation.where(standard_deviation > 0.0)
    )
    return state


def compute_low25_composites(state):
    mask = state["low_mask"]
    state["z_low25"] = np.nanmean(np.asarray(state["z_centered"].values)[mask], axis=0)
    values = np.asarray(
        state["ep_standardized"].transpose("event_year", "plev", "season_day").values
    )
    state["ep_low25"] = np.nanmean(values[mask], axis=0)
    return state


def bootstrap_and_package(state, *, repetitions=5000, seed=15500):
    z_values = np.asarray(state["z_centered"].values, dtype=float)
    z_observed, z_mean, z_std, z_significant = random_same_size_composites(
        z_values, state["low_mask"], repetitions=repetitions, seed=seed,
    )
    ep_values = np.asarray(
        state["ep_standardized"].transpose("event_year", "plev", "season_day").values,
        dtype=float,
    )
    ep_observed, ep_mean, ep_std, ep_significant = random_same_size_composites(
        ep_values, state["low_mask"], repetitions=repetitions, seed=seed + 1,
    )
    if not np.allclose(z_observed, state["z_low25"], equal_nan=True):
        raise RuntimeError("Z300 bootstrap composite differs from fixed low25 composite")
    if not np.allclose(ep_observed, state["ep_low25"], equal_nan=True):
        raise RuntimeError("EP bootstrap composite differs from fixed low25 composite")
    rankings, master, table = (
        state["rankings"], state["classification_master"], state["table"]
    )
    threshold = np.unique(np.asarray(master.low25_threshold_du, dtype=float))
    if threshold.size != 1 or not np.isfinite(threshold[0]):
        raise RuntimeError("Ranking must contain one finite low25 threshold")
    master_sizes = np.unique(np.asarray(master.sample_size, dtype=int))
    master_low_counts = np.unique(np.asarray(master.low25_count, dtype=int))
    if (
        master_sizes.size != 1 or int(master_sizes[0]) != len(master)
        or master_low_counts.size != 1
        or int(master_low_counts[0]) != sum(_as_bool(value) for value in master.is_low25)
    ):
        raise RuntimeError("Canonical master sample/low counts are internally inconsistent")
    scope_low_count = sum(_as_bool(value) for value in rankings.is_low25)
    field_scope_segments = sorted(set(rankings.source_segment.astype(str)))
    master_ids = set(master.event_id.astype(str))
    scope_ids = set(rankings.event_id.astype(str))
    available_ids = set(table.event_id.astype(str))
    unavailable_ids = set(state["unavailable_ids"])
    excluded_ids = set(state["excluded_master_ids"])
    if (
        available_ids & unavailable_ids
        or scope_ids != available_ids | unavailable_ids
        or master_ids != scope_ids | excluded_ids
    ):
        raise RuntimeError("Master/scope/availability event-ID partition is inconsistent")
    event_ids = table.event_id.astype(str).to_numpy()
    output = xr.Dataset(
        {
            "z300_low25_anomaly": (("month", "lat", "lon"), z_observed),
            "z300_stationary_climatology": (("month", "lat", "lon"), state["z_stationary"].values),
            "z300_bootstrap_mean": (("month", "lat", "lon"), z_mean),
            "z300_bootstrap_std": (("month", "lat", "lon"), z_std),
            "z300_bootstrap_significant": (("month", "lat", "lon"), z_significant.astype(np.int8)),
            "epflux_low25_std_anomaly": (("pressure", "season_day"), ep_observed),
            "epflux_bootstrap_mean": (("pressure", "season_day"), ep_mean),
            "epflux_bootstrap_std": (("pressure", "season_day"), ep_std),
            "epflux_bootstrap_significant": (("pressure", "season_day"), ep_significant.astype(np.int8)),
            "is_low25": (("event_year",), np.asarray(state["low_mask"], dtype=np.int8)),
            "o3_minimum_du": (("event_year",), np.asarray(table.minimum_du, dtype=np.float32)),
            "canonical_rank": (("event_year",), np.asarray(table["rank"], dtype=np.int32)),
        },
        coords={
            "month": list(MONTH_NAMES), "lat": state["z"].lat.values,
            "lon": state["z"].lon.values,
            "pressure": np.asarray(state["ep"].plev.values, dtype=float),
            "season_day": np.arange(SEASON_LENGTH), "event_year": event_ids,
            "event_id": ("event_year", event_ids),
            "source_segment": ("event_year", table.source_segment.astype(str).to_numpy()),
            "model_year": ("event_year", np.asarray(table.model_year, dtype=np.int32)),
            "ranking_event_year": ("event_year", np.asarray(table.event_year, dtype=np.int32)),
        },
        attrs={
            "source_label": state["label"],
            "method": "monthly Z300 and calendar-day standardized 40--80N upward EP-flux composite",
            "master_ranking_path": str(state["master_ranking_path"]),
            "source_manifest": state["source_manifest"],
            "master_sample_size": int(master_sizes[0]),
            "master_low_count": int(master_low_counts[0]),
            "field_scope_segments": ",".join(field_scope_segments),
            "scope_master_count": int(len(rankings)),
            "scope_master_low_count": int(scope_low_count),
            "excluded_master_event_ids": ",".join(state["excluded_master_ids"]),
            "available_event_count": int(len(table)),
            "available_low_count": int(np.asarray(state["low_mask"]).sum()),
            "unavailable_event_ids": ",".join(state["unavailable_ids"]),
            "unavailable_event_reasons": ";".join(state["unavailable_reasons"]),
            "low25_definition": (
                "membership inherited unchanged from canonical classification master; "
                "field scope never re-ranked"
            ),
            "low25_threshold_du": float(threshold[0]),
            "bootstrap_replicates": int(repetitions),
            "bootstrap_method": "same-size event composites sampled with replacement; two-standard-deviation mask",
            "bootstrap_seed_z300": int(seed), "bootstrap_seed_epflux": int(seed + 1),
            "epflux_method": "monthly natural-calendar N2; do_ubar=True; w=None; wave=-1; 40--80N cosine mean; upward=-ep2",
            "natural_month_n2": "True", "do_ubar": "True",
            "w_argument": "None", "wave": "-1",
            "ep_standardization_ddof": 0,
            "standardization_ddof_note": "Methods V7 omits ddof; workflow fixes population ddof=0 for reproducibility",
            "event_calendar": "Nov--Mar no-leap (151 days)",
            "z300_completeness": (
                "every retained 300-hPa grid cell finite on every required daily source; "
                "natural-month means use skipna=False"
            ),
        },
    )
    output.pressure.attrs.update(units="hPa", positive="down")
    return output
