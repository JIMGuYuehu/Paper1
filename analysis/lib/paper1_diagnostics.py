"""Numerical kernels shared by the Methods-V7 Paper 1 notebooks.

This module deliberately contains no project-specific output paths.  The
notebooks declare inputs and staging outputs; these functions implement the
methods once so January, February, March, reanalysis, and free-running WACCM
cannot silently diverge.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr


PLEV_HPA = np.array(
    [
        0.1, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200,
        250, 300, 400, 500, 600, 700, 850, 925, 1000,
    ],
    dtype=np.float64,
)
PLEV_PA = PLEV_HPA * 100.0
# EP flux is evaluated only where every free-running/reanalysis source has an
# exact staged pressure level.  Keep this grid separate from the 23-level NAM
# grid: MERRA-2 U/V/T has only these 18 levels.
EP_FREE_PLEV_HPA = np.array(
    [1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 350],
    dtype=np.float64,
)
# January/February staged restart products and March hindcast contain the full
# 23-level Methods grid (including the required 10, 50, and 100 hPa levels).
EP_RESTART_PLEV_HPA = PLEV_HPA.copy()
AVOGADRO = 6.02214076e23
GRAVITY = 9.80665
MOLAR_MASS_AIR = 28.9647e-3
MOLAR_MASS_O3 = 47.9982e-3
DU_MOLECULES_M2 = 2.687e20
DU_PER_PA_MOLE_FRACTION = AVOGADRO / (
    GRAVITY * MOLAR_MASS_AIR * DU_MOLECULES_M2
)


def _ozone_constant_attrs() -> dict[str, float]:
    """Return the exact Eq.-1 constants recorded on every O3-column product."""

    return {
        "avogadro_mol_minus1": AVOGADRO,
        "gravity_m_s_minus2": GRAVITY,
        "dry_air_molar_mass_kg_mol_minus1": MOLAR_MASS_AIR,
        "ozone_molar_mass_kg_mol_minus1": MOLAR_MASS_O3,
        "du_molecules_m_minus2": DU_MOLECULES_M2,
        "du_per_pa_mole_fraction": DU_PER_PA_MOLE_FRACTION,
    }


def _normalised_units(field: xr.DataArray) -> str:
    """Return a compact unit spelling suitable for strict source checks."""

    return re.sub(r"[\s_{}()^*/-]+", "", str(field.attrs.get("units", "")).lower())


def _require_ozone_units(field: xr.DataArray, *, mass_mixing_ratio: bool) -> None:
    """Reject an O3 source whose mixing-ratio convention is absent or ambiguous."""

    units = _normalised_units(field)
    mass_aliases = {"kgkg1", "kgkg"}
    mole_aliases = {"molmol1", "molmol", "molefraction"}
    allowed = mass_aliases if mass_mixing_ratio else mole_aliases
    if units not in allowed:
        expected = "kg/kg" if mass_mixing_ratio else "mol/mol"
        raise ValueError(
            f"O3 units must explicitly represent {expected}; found {field.attrs.get('units')!r}"
        )


def parse_year(path: Path) -> int:
    matches = re.findall(r"(?<!\d)(\d{4})(?!\d)", Path(path).name)
    if not matches:
        raise ValueError(f"Cannot parse four-digit year from {path}")
    return int(matches[-1])


def parse_member(path: Path) -> str:
    patterns = (r"\.(\d{3})(?=\.cam\.|\.[A-Z0-9_]+\.nc$)", r"(?<!\d)(\d{3})(?!\d)")
    for pattern in patterns:
        matches = re.findall(pattern, Path(path).name)
        if matches:
            return matches[-1]
    raise ValueError(f"Cannot parse member id from {path}")


def date_int(dataset: xr.Dataset) -> np.ndarray:
    """Return YYYYMMDD integers without assuming a Gregorian xarray index."""

    if "date" in dataset:
        values = np.asarray(dataset["date"].values).astype(np.int64)
        if values.ndim != 1:
            raise ValueError("date must be one-dimensional")
        return values
    if "time" not in dataset:
        raise KeyError("dataset has neither date nor time")
    result = []
    for value in np.asarray(dataset["time"].values):
        if hasattr(value, "year"):
            result.append(int(value.year) * 10000 + int(value.month) * 100 + int(value.day))
        else:
            stamp = pd.Timestamp(value)
            result.append(stamp.year * 10000 + stamp.month * 100 + stamp.day)
    return np.asarray(result, dtype=np.int32)


def date_parts(dates: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(dates, dtype=np.int64)
    return values // 10000, (values // 100) % 100, values % 100


def noleap_index(dates: Sequence[int]) -> np.ndarray:
    """Map month/day to 0..364; map 29 February to -1."""

    _, month, day = date_parts(dates)
    starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    result = starts[month - 1] + day - 1
    result[(month == 2) & (day == 29)] = -1
    return result.astype(np.int16)


def select_latitude(field: xr.DataArray, south: float, north: float) -> xr.DataArray:
    latitude = field["lat"]
    selected = field.sel(
        lat=slice(south, north) if latitude[0] < latitude[-1] else slice(north, south)
    )
    return selected.sortby("lat")


def select_pressure_levels(
    field: xr.DataArray, target_hpa: Sequence[float] = PLEV_HPA
) -> xr.DataArray:
    """Select, without re-interpolation, the required staged pressure levels."""

    pressure_name = "plev" if "plev" in field.coords else "lev"
    pressure = np.asarray(field[pressure_name].values, dtype=float)
    if np.nanmax(pressure) > 2000.0:
        pressure = pressure / 100.0
    wanted = np.asarray(target_hpa, dtype=float)
    indices = []
    for level in wanted:
        matches = np.flatnonzero(np.isclose(pressure, level, rtol=0.0, atol=1.0e-6))
        if matches.size != 1:
            raise ValueError(
                f"staged pressure-level product must contain exactly one {level:g} hPa level"
            )
        indices.append(int(matches[0]))
    selected = field.isel({pressure_name: indices}).assign_coords(
        {pressure_name: wanted}
    )
    if pressure_name != "plev":
        selected = selected.rename({pressure_name: "plev"})
    selected.plev.attrs.update(units="hPa", selection="exact staged CDO pressure levels")
    return selected


def cosine_latitude_mean(field: xr.DataArray, south: float, north: float) -> xr.DataArray:
    selected = select_latitude(field, south, north)
    weights = np.cos(np.deg2rad(selected["lat"])).clip(0.0, 1.0)
    return selected.weighted(weights).mean("lat", skipna=True)


def hybrid_mid_pressure(dataset: xr.Dataset) -> xr.DataArray:
    if "PS" not in dataset:
        raise KeyError("PS is required for CAM hybrid-pressure interpolation")
    p0 = dataset["P0"] if "P0" in dataset else xr.DataArray(100000.0)
    return dataset["hyam"] * p0 + dataset["hybm"] * dataset["PS"]


def log_pressure_interpolate(
    field: xr.DataArray,
    pressure_pa: xr.DataArray,
    target_hpa: Sequence[float] = PLEV_HPA,
) -> xr.DataArray:
    targets = np.asarray(target_hpa, dtype=np.float64) * 100.0

    def one_profile(values: np.ndarray, pressures: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values) & np.isfinite(pressures) & (pressures > 0.0)
        if valid.sum() < 2:
            return np.full(targets.shape, np.nan)
        p = pressures[valid]
        v = values[valid]
        order = np.argsort(p)
        return np.interp(
            np.log(targets), np.log(p[order]), v[order], left=np.nan, right=np.nan
        )

    result = xr.apply_ufunc(
        one_profile,
        field,
        pressure_pa,
        input_core_dims=[["lev"], ["lev"]],
        output_core_dims=[["plev"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"plev": targets.size}},
    )
    return result.assign_coords(plev=("plev", np.asarray(target_hpa, dtype=float)))


def waccm_partial_ozone(dataset: xr.Dataset) -> xr.DataArray:
    """Exact CAM hybrid-layer overlap for the 30--70 hPa O3 column."""

    _require_ozone_units(dataset["O3"], mass_mixing_ratio=False)
    o3 = select_latitude(dataset["O3"], 60.0, 90.0)
    ps = select_latitude(dataset["PS"], 60.0, 90.0)
    p0 = dataset["P0"] if "P0" in dataset else xr.DataArray(100000.0)
    interface = dataset["hyai"] * p0 + dataset["hybi"] * ps
    top = interface.isel(ilev=slice(0, -1)).rename(ilev="lev").assign_coords(lev=o3.lev)
    bottom = interface.isel(ilev=slice(1, None)).rename(ilev="lev").assign_coords(lev=o3.lev)
    lower = xr.apply_ufunc(np.minimum, top, bottom)
    upper = xr.apply_ufunc(np.maximum, top, bottom)
    overlap = (
        xr.apply_ufunc(np.minimum, upper, 7000.0)
        - xr.apply_ufunc(np.maximum, lower, 3000.0)
    ).clip(min=0.0)
    # Mask non-overlapping layers so an entirely missing 30--70 hPa profile
    # remains NaN instead of becoming an artificial zero column.
    contribution = (o3 * overlap).where(overlap > 0.0)
    column = contribution.sum("lev", skipna=True, min_count=1) * DU_PER_PA_MOLE_FRACTION
    if "lon" in column.dims:
        column = column.mean("lon", skipna=True)
    result = cosine_latitude_mean(column, 60.0, 90.0).rename("partial_o3_du")
    result = result.assign_coords(date=("time", date_int(dataset)))
    result.attrs.update(
        units="DU",
        pressure_integration="exact CAM-interface overlap with 30--70 hPa",
        latitude_average="60--90N cosine weighted",
        **_ozone_constant_attrs(),
    )
    return result


def merra2_partial_ozone(dataset: xr.Dataset) -> xr.DataArray:
    """MERRA-2 30--70 hPa column after kg/kg to mol/mol conversion."""

    _require_ozone_units(dataset["O3"], mass_mixing_ratio=True)
    field = select_latitude(dataset["O3"], 60.0, 90.0)
    pressure_name = "lev" if "lev" in field.coords else "plev"
    pressure = np.asarray(field[pressure_name].values, dtype=float)
    if np.nanmax(pressure) > 2000.0:
        pressure = pressure / 100.0
        field = field.assign_coords({pressure_name: pressure})
    field = field.sortby(pressure_name)
    inner = field[pressure_name].where(
        (field[pressure_name] > 30.0) & (field[pressure_name] < 70.0), drop=True
    )
    target = xr.DataArray(
        np.r_[30.0, inner.values, 70.0], dims="integration_level"
    )
    # Integrate every grid-cell profile first.  Averaging before integration is
    # not equivalent when missing values vary with level/latitude/longitude.
    bounded = field.interp({pressure_name: target})
    if pressure_name in bounded.coords:
        bounded = bounded.drop_vars(pressure_name)
    mole_fraction = bounded * MOLAR_MASS_AIR / MOLAR_MASS_O3
    integral = xr.apply_ufunc(
        np.trapz,
        mole_fraction,
        target * 100.0,
        input_core_dims=[["integration_level"], ["integration_level"]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
    )
    column = integral * DU_PER_PA_MOLE_FRACTION
    if "lon" in column.dims:
        column = column.mean("lon", skipna=True)
    result = cosine_latitude_mean(column, 60.0, 90.0).rename("partial_o3_du")
    result = result.assign_coords(date=("time", date_int(dataset)))
    result.attrs.update(
        units="DU",
        pressure_integration="linear-pressure trapezoid with exact 30 and 70 hPa boundaries",
        source_conversion="MERRA-2 O3 kg/kg converted to mol/mol",
        latitude_average="60--90N cosine weighted",
        **_ozone_constant_attrs(),
    )
    return result


def pressure_level_partial_ozone(
    dataset: xr.Dataset, *, mass_mixing_ratio: bool
) -> xr.DataArray:
    """Exact-boundary 30--70 hPa integration for a pressure-level O3 field."""

    _require_ozone_units(dataset["O3"], mass_mixing_ratio=mass_mixing_ratio)
    field = select_latitude(dataset["O3"], 60.0, 90.0)
    pressure_name = "lev" if "lev" in field.coords else "plev"
    pressure = np.asarray(field[pressure_name].values, dtype=float)
    if np.nanmax(pressure) > 2000.0:
        field = field.assign_coords({pressure_name: pressure / 100.0})
    field = field.sortby(pressure_name)
    inner = field[pressure_name].where(
        (field[pressure_name] > 30.0) & (field[pressure_name] < 70.0), drop=True
    )
    target = xr.DataArray(np.r_[30.0, inner.values, 70.0], dims="integration_level")
    bounded = field.interp({pressure_name: target})
    if pressure_name in bounded.coords:
        bounded = bounded.drop_vars(pressure_name)
    if mass_mixing_ratio:
        bounded = bounded * MOLAR_MASS_AIR / MOLAR_MASS_O3
    integral = xr.apply_ufunc(
        np.trapz,
        bounded,
        target * 100.0,
        input_core_dims=[["integration_level"], ["integration_level"]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
    )
    column = integral * DU_PER_PA_MOLE_FRACTION
    if "lon" in column.dims:
        column = column.mean("lon", skipna=True)
    result = cosine_latitude_mean(column, 60.0, 90.0).rename("partial_o3_du")
    result = result.assign_coords(date=("time", date_int(dataset)))
    result.attrs.update(
        units="DU",
        pressure_integration="linear-pressure trapezoid with exact 30 and 70 hPa boundaries",
        source_conversion=("kg/kg to mol/mol" if mass_mixing_ratio else "input is mol/mol"),
        latitude_average="60--90N cosine weighted",
        **_ozone_constant_attrs(),
    )
    return result


RANKING_COLUMNS = [
    "source_family", "source_segment", "event_id", "event_year", "model_year",
    "minimum_date", "minimum_du", "rank", "sample_size", "low25_count",
    "low25_threshold_du", "is_low25", "is_target",
]


def spring_minimum(
    series: xr.DataArray,
    *,
    source_family: str,
    source_segment: str,
    event_year: int,
    model_year: int,
    target: bool = False,
) -> dict[str, object]:
    """Centered-5-day minimum over the exact 1 March--30 April window."""

    dates = np.asarray(series["date"].values, dtype=np.int64)
    order = np.argsort(dates)
    sorted_series = series.isel(time=order)
    dates = dates[order]
    smoothed = sorted_series.rolling(time=5, center=True, min_periods=5).mean()
    year, month, day = date_parts(dates)
    window = (year == int(event_year)) & (
        ((month == 3) & (day >= 1)) | ((month == 4) & (day <= 30))
    )
    selected = smoothed.isel(time=np.flatnonzero(window))
    selected_dates = dates[window]
    if selected.sizes.get("time", 0) != 61:
        raise ValueError(
            f"{source_segment} year {event_year}: expected exactly 61 Mar--Apr days; "
            f"found {selected.sizes.get('time', 0)}"
        )
    values = np.asarray(selected.values, dtype=float)
    if np.isfinite(values).sum() != 61:
        raise ValueError(f"{source_segment} year {event_year}: incomplete centered-5-day window")
    index = int(np.nanargmin(values))
    return {
        "source_family": source_family,
        "source_segment": source_segment,
        "event_id": f"{source_segment}:{int(model_year):04d}",
        "event_year": int(event_year),
        "model_year": int(model_year),
        "minimum_date": f"{int(selected_dates[index]):08d}",
        "minimum_du": float(values[index]),
        "is_target": bool(target),
    }


def rank_events(
    rows: Iterable[dict[str, object]], *, expected_count: int, low_fraction: float = 0.25
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} complete events; found {len(frame)}")
    if frame["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    frame = frame.sort_values(["minimum_du", "event_id"], kind="mergesort").reset_index(drop=True)
    count = int(np.floor(low_fraction * expected_count))
    frame["rank"] = np.arange(1, expected_count + 1, dtype=int)
    frame["sample_size"] = expected_count
    frame["low25_count"] = count
    frame["low25_threshold_du"] = float(frame.loc[count - 1, "minimum_du"])
    frame["is_low25"] = frame["rank"] <= count
    return frame[RANKING_COLUMNS]


def circular_noleap_climatology(field: xr.DataArray, window: int = 21) -> xr.DataArray:
    """Calendar month-day climatology with circular 21-day smoothing."""

    indices = noleap_index(field["date"].values)
    usable = indices >= 0
    prepared = field.isel(time=np.flatnonzero(usable)).assign_coords(
        month_day=("time", indices[usable])
    )
    climatology = prepared.groupby("month_day").mean("time", skipna=True).reindex(
        month_day=np.arange(365)
    )
    if climatology.sizes["month_day"] != 365:
        raise ValueError("No-leap climatology must contain 365 month-day bins")
    half = window // 2
    padded = xr.concat(
        [climatology.isel(month_day=slice(-half, None)), climatology,
         climatology.isel(month_day=slice(0, half))],
        dim="month_day",
    )
    smooth = padded.rolling(
        month_day=window, center=True, min_periods=window
    ).mean().isel(month_day=slice(half, half + 365))
    return smooth.assign_coords(month_day=np.arange(365))


def train_fixed_nam_reference(zonal_height: xr.DataArray) -> xr.Dataset:
    """Train one fixed EOF per pressure from calendar-month mean anomalies."""

    field = select_latitude(zonal_height, 20.0, 90.0).sortby("plev")
    years, months, _ = date_parts(field["date"].values)
    year_month = years * 100 + months
    monthly = field.assign_coords(year_month=("time", year_month)).groupby(
        "year_month"
    ).mean("time", skipna=True)
    month_of_sample = np.asarray(monthly.year_month.values, dtype=int) % 100
    anomalies = []
    for month in range(1, 13):
        indices = np.flatnonzero(month_of_sample == month)
        part = monthly.isel(year_month=indices)
        anomalies.append((part - part.mean("year_month", skipna=True)).rename(year_month="sample"))
    anomaly = xr.concat(anomalies, dim="sample")
    cosine = np.cos(np.deg2rad(field.lat)).clip(0.0, 1.0)
    weights = np.sqrt(cosine)
    patterns = []
    standard_deviation = []
    variance_fraction = []
    valid_monthly_sample_count = []
    for pressure in anomaly.plev.values:
        weighted = anomaly.sel(plev=pressure).transpose("sample", "lat") * weights
        matrix = np.asarray(weighted.values, dtype=float)
        valid_rows = np.all(np.isfinite(matrix), axis=1)
        matrix = matrix[valid_rows]
        if matrix.shape[0] < 24:
            raise ValueError(f"Too few complete monthly samples at {pressure} hPa")
        matrix -= matrix.mean(axis=0, keepdims=True)
        _, singular, right = np.linalg.svd(matrix, full_matrices=False)
        pattern = right[0]
        polar = np.asarray(field.lat.values) >= 65.0
        cosine_values = np.asarray(cosine.values, dtype=float)
        weight_values = np.asarray(weights.values, dtype=float)
        usable_polar = polar & (cosine_values > 1.0e-8)
        physical_pattern = np.divide(
            pattern, weight_values,
            out=np.full(pattern.shape, np.nan), where=weight_values > 1.0e-8,
        )
        polar_loading = np.average(
            physical_pattern[usable_polar], weights=cosine_values[usable_polar]
        )
        if not np.isfinite(polar_loading) or np.isclose(polar_loading, 0.0):
            raise ValueError(f"EOF sign is undefined at {pressure} hPa")
        if polar_loading > 0.0:
            pattern = -pattern
        pc = matrix @ pattern
        patterns.append(pattern)
        standard_deviation.append(np.std(pc, ddof=0))
        variance_fraction.append(singular[0] ** 2 / np.sum(singular ** 2))
        valid_monthly_sample_count.append(int(valid_rows.sum()))
    reference = xr.Dataset(
        {
            "eof1_weighted": (("plev", "lat"), np.asarray(patterns)),
            "monthly_pc_std": ("plev", np.asarray(standard_deviation)),
            "explained_variance_fraction": ("plev", np.asarray(variance_fraction)),
            "valid_monthly_sample_count": (
                "plev", np.asarray(valid_monthly_sample_count, dtype=np.int32)
            ),
            "daily_height_climatology": circular_noleap_climatology(field, 21),
        },
        coords={"plev": field.plev, "lat": field.lat, "month_day": np.arange(365)},
        attrs={
            "method": "fixed leading EOF of calendar-month zonal-mean height anomalies north of 20N",
            "latitude_weight": "sqrt(cos(latitude))",
            "daily_climatology": "month-day no-leap climatology, circular centered 21-day mean",
            "monthly_pc_standardization_ddof": 0,
            "monthly_mean": "actual finite daily values within each natural calendar month",
            "calendar_month_sample_count": int(monthly.sizes["year_month"]),
            "valid_monthly_sample_count_definition": (
                "monthly zonal-height mean is finite at every retained 20--90N latitude"
            ),
            "sign_orientation": "negative 65--90N cosine-mean physical-height loading",
            "positive_phase": "lower polar height and stronger polar vortex",
        },
    )
    return reference


def project_nam(zonal_height: xr.DataArray, reference: xr.Dataset) -> xr.Dataset:
    """Project daily fields directly on a fixed EOF; no empirical calibration."""

    field = select_latitude(zonal_height, 20.0, 90.0).sortby("plev")
    indices = noleap_index(field.date.values)
    climatology = reference.daily_height_climatology
    baseline = []
    for index in indices:
        if index >= 0:
            baseline.append(climatology.sel(month_day=int(index)))
        else:
            baseline.append(
                0.5 * (
                    climatology.sel(month_day=58) + climatology.sel(month_day=59)
                )
            )
    baseline = xr.concat(baseline, dim="time").assign_coords(time=field.time)
    anomaly = field - baseline
    weights = np.sqrt(np.cos(np.deg2rad(field.lat)).clip(0.0, 1.0))
    usable_latitude = weights > 1.0e-8
    required_latitudes = int(usable_latitude.sum())
    complete_profile = (
        anomaly.where(usable_latitude).count("lat") == required_latitudes
    )
    projected = (
        (anomaly * weights * reference.eof1_weighted)
        .sum("lat", skipna=True, min_count=1)
        .where(complete_profile)
        / reference.monthly_pc_std
    )
    nam = projected.rename("nam").transpose("time", "plev")
    if not np.any(np.isclose(nam.plev.values, 1000.0)):
        raise ValueError("NAM pressure grid does not contain 1000 hPa exactly")
    ao = nam.sel(plev=1000.0).rename("ao")
    output = xr.Dataset({"nam": nam, "ao": ao}).assign_coords(
        date=("time", np.asarray(field.date.values, dtype=np.int32))
    )
    output.attrs.update(
        method="direct fixed-EOF projection; AO is vertical NAM at exactly 1000 hPa",
        calibration="none",
        positive_phase="stronger polar vortex",
        monthly_pc_standardization_ddof=int(
            reference.attrs.get("monthly_pc_standardization_ddof", 0)
        ),
    )
    return output


def natural_month_groups(dates: Sequence[int]) -> list[np.ndarray]:
    years, months, _ = date_parts(dates)
    keys = years * 100 + months
    return [np.flatnonzero(keys == key) for key in np.unique(keys)]


def compute_epflux_monthly(
    u: xr.DataArray,
    v: xr.DataArray,
    temperature: xr.DataArray,
    dates: Sequence[int],
    compute_epflux_div,
) -> xr.Dataset:
    """All-wave EP flux with natural-calendar-month static stability."""

    for name, field in (("U", u), ("V", v), ("T", temperature)):
        if tuple(field.dims) != ("time", "plev", "lat", "lon"):
            raise ValueError(f"{name} must have dimensions time,plev,lat,lon")
    for coordinate in ("time", "plev", "lat", "lon"):
        reference = np.asarray(u[coordinate].values)
        for name, field in (("V", v), ("T", temperature)):
            if not np.array_equal(reference, np.asarray(field[coordinate].values)):
                raise ValueError(f"U/{name} {coordinate} coordinates differ")
    if not np.array_equal(np.asarray(u.plev.values, dtype=float),
                          np.sort(np.asarray(u.plev.values, dtype=float))):
        raise ValueError("EP pressure grid must be strictly ascending in hPa")
    if np.nanmax(np.asarray(u.plev.values, dtype=float)) > 1100.0:
        raise ValueError("EP plev must be expressed in hPa")
    if len(dates) != u.sizes["time"]:
        raise ValueError("EP integer-date coordinate length differs from U/V/T time")

    components = []
    for indices in natural_month_groups(dates):
        uu = u.isel(time=indices).transpose("time", "plev", "lat", "lon")
        vv = v.isel(time=indices).transpose("time", "plev", "lat", "lon")
        tt = temperature.isel(time=indices).transpose("time", "plev", "lat", "lon")
        ep1, ep2, div1, div2 = compute_epflux_div(
            lat=np.asarray(uu.lat.values, dtype=float),
            pres=np.asarray(uu.plev.values, dtype=float),
            u=np.asarray(uu.values, dtype=float),
            v=np.asarray(vv.values, dtype=float),
            t=np.asarray(tt.values, dtype=float),
            w=None,
            do_ubar=True,
            wave=-1,
        )
        components.append(
            xr.Dataset(
                {
                    "ep1": (("time", "plev", "lat"), np.asarray(ep1, dtype=np.float32)),
                    "ep2_raw": (("time", "plev", "lat"), np.asarray(ep2, dtype=np.float32)),
                    "div1": (("time", "plev", "lat"), np.asarray(div1, dtype=np.float32)),
                    "div2": (("time", "plev", "lat"), np.asarray(div2, dtype=np.float32)),
                },
                coords={
                    "time": uu.time, "plev": uu.plev, "lat": uu.lat,
                    "date": ("time", np.asarray(dates, dtype=np.int32)[indices]),
                },
            )
        )
    output = xr.concat(components, dim="time").sortby("time")
    upward = (-output.ep2_raw).rename("fz_upward")
    upward.attrs.update(units=output.ep2_raw.attrs.get("units", "aostools EP2 units"),
                        sign_convention="upward = -aostools ep2")
    output["fz_upward"] = upward
    weighted = cosine_latitude_mean(upward, 40.0, 80.0).rename("fz_upward_40_80n")
    output["fz_upward_40_80n"] = weighted
    output["ep100_upward_40_80n"] = weighted.sel(plev=100.0).rename(
        "ep100_upward_40_80n"
    )
    output.attrs.update(
        method="aostools ComputeEPfluxDiv; natural-calendar-month N2 denominator",
        natural_month_n2="True", do_ubar="True", w_argument="None", wave="-1",
        wave_description="all resolved waves",
        latitude_average="40--80N cosine weighted",
        upward_sign="minus aostools ep2",
    )
    output.plev.attrs.update(units="hPa", positive="down")
    return output


def crps_ensemble(members: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(members, dtype=float)
    truth = np.asarray(reference, dtype=float)
    first = np.nanmean(np.abs(values - truth[None, ...]), axis=0)
    second = np.nanmean(
        np.abs(values[:, None, ...] - values[None, :, ...]), axis=(0, 1)
    )
    return first - 0.5 * second


def sign_agreement(members: np.ndarray) -> np.ndarray:
    """Fraction of available members sharing the ensemble-mean sign."""

    values = np.asarray(members, dtype=float)
    mean = np.nanmean(values, axis=0)
    target_sign = np.sign(mean)
    finite = np.isfinite(values)
    matches = finite & (np.sign(values) == target_sign[None, ...])
    denominator = finite.sum(axis=0)
    result = np.divide(
        matches.sum(axis=0), denominator,
        out=np.full(mean.shape, np.nan), where=denominator > 0,
    )
    # A zero ensemble-mean anomaly has no defined sign and therefore cannot be
    # called robust, even if individual values are also exactly zero.
    result[target_sign == 0.0] = np.nan
    return result


def bootstrap_event_profile(
    baseline: np.ndarray,
    target: np.ndarray,
    *,
    repetitions: int = 5000,
    seed: int = 15001,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Target-excluded percentile bootstrap for an event-profile anomaly."""

    pool = np.asarray(baseline, dtype=float)
    event = np.asarray(target, dtype=float)
    if pool.ndim < 2 or pool.shape[1:] != event.shape:
        raise ValueError("baseline must be event x profile and match target profile")
    if pool.shape[0] < 2:
        raise ValueError("event-profile bootstrap requires at least two baseline events")
    climatology = np.nanmean(pool, axis=0)
    anomaly = event - climatology
    flattened = pool.reshape(pool.shape[0], -1)
    finite = np.isfinite(flattened)
    filled = np.where(finite, flattened, 0.0)
    distribution = np.empty((repetitions, flattened.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    probability = np.full(pool.shape[0], 1.0 / pool.shape[0])
    completed = 0
    while completed < repetitions:
        current = min(batch_size, repetitions - completed)
        counts = rng.multinomial(pool.shape[0], probability, size=current).astype(float)
        numerator = counts @ filled
        denominator = counts @ finite
        means = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        distribution[completed:completed + current] = (
            means - climatology.reshape(1, -1)
        ).astype(np.float32)
        completed += current
    low, high = np.nanpercentile(distribution, [2.5, 97.5], axis=0)
    low = low.reshape(event.shape)
    high = high.reshape(event.shape)
    significant = (anomaly < low) | (anomaly > high)
    return anomaly.astype(np.float32), low.astype(np.float32), high.astype(np.float32), significant


def random_same_size_composites(
    events: np.ndarray,
    observed_mask: np.ndarray,
    *,
    repetitions: int = 5000,
    seed: int = 15500,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same-size low-event composite bootstrap used by Methods V7."""

    values = np.asarray(events, dtype=float)
    mask = np.asarray(observed_mask, dtype=bool)
    number = int(mask.sum())
    if values.shape[0] != mask.size or not 1 < number < mask.size:
        raise ValueError("invalid low-event mask")
    field_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    finite = np.isfinite(flat)
    filled = np.where(finite, flat, 0.0)
    observed = np.nanmean(values[mask], axis=0)
    rng = np.random.default_rng(seed)
    probability = np.full(values.shape[0], 1.0 / values.shape[0])
    total = np.zeros(flat.shape[1], dtype=np.float64)
    total_square = np.zeros(flat.shape[1], dtype=np.float64)
    valid_count = np.zeros(flat.shape[1], dtype=np.int64)
    completed = 0
    while completed < repetitions:
        current = min(batch_size, repetitions - completed)
        # A multinomial count vector is algebraically identical to drawing
        # `number` complete event years independently with replacement.
        counts = rng.multinomial(number, probability, size=current).astype(float)
        numerator = counts @ filled
        denominator = counts @ finite
        bootstrap = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        valid = np.isfinite(bootstrap)
        total += np.nansum(bootstrap, axis=0)
        total_square += np.nansum(bootstrap * bootstrap, axis=0)
        valid_count += valid.sum(axis=0)
        completed += current
    mean_flat = np.divide(
        total, valid_count, out=np.full_like(total, np.nan), where=valid_count > 0
    )
    second = np.divide(
        total_square, valid_count, out=np.full_like(total_square, np.nan),
        where=valid_count > 0,
    )
    std_flat = np.sqrt(np.maximum(second - mean_flat * mean_flat, 0.0))
    mean = mean_flat.reshape(field_shape)
    std = std_flat.reshape(field_shape)
    significant = np.abs(observed - mean) >= 2.0 * std
    return (
        observed.astype(np.float32), mean.astype(np.float32),
        std.astype(np.float32), significant,
    )
