"""Figure-ready relationship products for the current Paper 1 manuscript.

The plotting stage must only render these products.  Every minimum, window
mean, low-25 selection, Pearson statistic, p value, and OLS fit is calculated
here and installed through the atomic staging writers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

from paper1_diagnostics import date_parts, noleap_index
from workflow_io import PRODUCT_VERSION, write_csv_atomic, write_netcdf_atomic


def regression(x, y) -> dict[str, float | int]:
    """Return the one accepted Pearson/OLS summary with explicit sample size."""

    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return {"r": np.nan, "p": np.nan, "n": len(frame), "slope": np.nan, "intercept": np.nan}
    fit = linregress(frame.x.to_numpy(float), frame.y.to_numpy(float))
    return {
        "r": float(fit.rvalue), "p": float(fit.pvalue), "n": int(len(frame)),
        "slope": float(fit.slope), "intercept": float(fit.intercept),
    }


def centered_spring_detail(member, member_dates, reference, reference_dates):
    """Exact centered-5-day Mar-Apr minimum, padding only outside restart coverage."""

    member = np.asarray(member, dtype=float)
    member_dates = np.asarray(member_dates, dtype=int)
    reference = np.asarray(reference, dtype=float)
    reference_dates = np.asarray(reference_dates, dtype=int)
    ref_lookup = {int(date): float(value) for date, value in zip(reference_dates, reference)}
    member_lookup = {int(date): float(value) for date, value in zip(member_dates, member)}
    event_year = int(((member_dates // 10000)[((member_dates // 100) % 100) == 3])[0])
    month_days = (
        [(2, 27), (2, 28)]
        + [(3, day) for day in range(1, 32)]
        + [(4, day) for day in range(1, 31)]
        + [(5, 1), (5, 2)]
    )
    required_int = np.asarray(
        [event_year * 10000 + month * 100 + day for month, day in month_days], dtype=int
    )
    values, provenance = [], []
    for date in required_int:
        if int(date) in member_lookup:
            values.append(member_lookup[int(date)]); provenance.append("member")
        elif int(date) in ref_lookup:
            values.append(ref_lookup[int(date)]); provenance.append("reference_padding")
        else:
            raise RuntimeError(f"centered-5-day spring metric lacks required boundary date {date}")
    smooth = pd.Series(values).rolling(5, center=True, min_periods=5).mean().to_numpy()
    month = (required_int // 100) % 100
    day = required_int % 100
    window = ((month == 3) & (day >= 1)) | ((month == 4) & (day <= 30))
    if window.sum() != 61 or np.isfinite(smooth[window]).sum() != 61:
        raise RuntimeError("centered-5-day spring metric requires 61 finite Mar1-Apr30 values")
    indices = np.flatnonzero(window)
    selected = indices[int(np.nanargmin(smooth[window]))]
    date = int(required_int[selected])
    return {
        "minimum_du": float(smooth[selected]), "minimum_date": date,
        "minimum_doy": int(noleap_index([date])[0] + 1),
        "boundary_padding": ",".join(sorted(set(np.asarray(provenance)[~np.isin(required_int, member_dates)]))),
    }


def _reference_for_dates(values, dates, requested):
    lookup = {int(date): index for index, date in enumerate(np.asarray(dates, dtype=int))}
    return np.asarray(values, dtype=float)[[lookup[int(date)] for date in requested]]


def _histogram_table(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    edges = np.arange(60, 126, 5, dtype=int)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = frame[(frame.minimum_doy >= left) & (frame.minimum_doy < right)]
        low = selected.minimum_du <= threshold
        rows.append({
            "bin_left_doy": left, "bin_right_doy": right,
            "count_all": int(len(selected)), "count_low": int(low.sum()),
            "count_other": int((~low).sum()), "low25_threshold_du": threshold,
            "minimum_method": "centered5 Mar1-Apr30",
        })
    if sum(row["count_all"] for row in rows) != len(frame):
        raise RuntimeError("5-day minimum-date histogram did not retain every member")
    return pd.DataFrame(rows)


def _relationship_table(frame, *, case, x_name, selector, ref_x, ref_y):
    selected = frame.loc[selector(frame)].copy()
    stats = regression(selected[x_name], selected.minimum_du)
    output = pd.DataFrame({
        "case": case, "member": selected.member.astype(str),
        "x": selected[x_name].astype(float), "y": selected.minimum_du.astype(float),
        "is_low": selected.is_low.astype(bool),
        "low25_threshold_du": selected.low25_threshold_du.astype(float),
        "epflux_method": "do_ubar=True; monthly natural-calendar N2; w=None; wave=-1; upward=-ep2; 40--80N cosine",
        "minimum_method": "centered5 Mar1-Apr30; exact 30--70hPa; 60--90N cosine",
    })
    for name, value in {**stats, "ref_x": ref_x, "ref_y": ref_y}.items():
        output[name] = value
    return output


def build_hindcast_relationships(root: Path, *, overwrite: bool = False) -> pd.DataFrame:
    """Build the current Figures 8, 9, and 11 products.

    The two selected late-winter precursor windows contain exactly 20 no-leap
    days: January 21--February 9 (DOY 21--40) and February 21--March 12
    (DOY 52--71).  EP flux is always the canonical no-W, natural-month-N2
    product described in the manuscript Methods.
    """

    root = Path(root)
    ranking = pd.read_csv(root / "ozone" / "waccm_master_rankings.csv")
    threshold = float(ranking.low25_threshold_du.iloc[0])
    with xr.open_dataset(root / "ozone" / "bwcn_partial_o3.nc", decode_times=False) as ds:
        mask = np.asarray(ds.model_year.values, dtype=int) == 8
        ref_o3 = np.asarray(ds.partial_o3_du.values, dtype=float)[mask]
        ref_dates = np.asarray(ds.date.values, dtype=int)[mask]
    with xr.open_dataset(root / "dynamics" / "waccm_bwcn_year0008.nc", decode_times=False) as ds:
        ref_ep = np.asarray(ds.ep100_upward_40_80n.values, dtype=float)
        ref_ep_dates = np.asarray(ds.date.values, dtype=int)

    rows = []
    reference_details = centered_spring_detail(ref_o3, ref_dates, ref_o3, ref_dates)
    for case in ("0008-01", "0008-02", "0008-03"):
        with xr.open_dataset(root / "ozone" / f"hindcast_{case}_partial_o3.nc", decode_times=False) as ds:
            o3 = np.asarray(ds.partial_o3_du.values, dtype=float)
            dates = np.asarray(ds.date.values, dtype=int)
            members = np.asarray(ds.member.values).astype(str)
        with xr.open_dataset(root / "dynamics" / f"hindcast_{case}.nc", decode_times=False) as ds:
            ep = np.asarray(ds.ep100_upward_40_80n.values, dtype=float)
            if not np.array_equal(dates, np.asarray(ds.date.values, dtype=int)):
                raise RuntimeError(f"{case}: O3 and EP calendars differ")
        month = (dates // 100) % 100
        doy = noleap_index(dates) + 1
        for index, member in enumerate(members):
            detail = centered_spring_detail(o3[index], dates, ref_o3, ref_dates)
            rows.append({
                "case": case, "member": member, **detail,
                "epsilon_min_du": detail["minimum_du"] - reference_details["minimum_du"],
                "is_low": detail["minimum_du"] <= threshold,
                "low25_threshold_du": threshold,
                "ep100_january_mean": float(np.nanmean(ep[index, month == 1])) if np.any(month == 1) else np.nan,
                "ep100_february_mean": float(np.nanmean(ep[index, month == 2])) if np.any(month == 2) else np.nan,
                "ep100_janfeb_mean": float(np.nanmean(ep[index, np.isin(month, [1, 2])])) if np.any(np.isin(month, [1, 2])) else np.nan,
                "ep100_doy21_40_mean": float(np.nanmean(ep[index, (doy >= 21) & (doy <= 40)])),
                "ep100_doy52_71_mean": float(np.nanmean(ep[index, (doy >= 52) & (doy <= 71)])),
            })
    frame = pd.DataFrame(rows)
    frame["minimum_method"] = "centered5 Mar1-Apr30; exact 30--70hPa; 60--90N cosine"
    frame["epflux_method"] = "do_ubar=True; monthly natural-calendar N2; w=None; wave=-1; upward=-ep2; 40--80N cosine"
    frame["ranking_manifest"] = str(root / "ozone" / "waccm_master_rankings.csv")
    required = [
        "case", "member", "minimum_du", "minimum_date", "minimum_doy",
        "epsilon_min_du", "is_low", "low25_threshold_du", "ep100_january_mean",
        "ep100_february_mean", "ep100_janfeb_mean", "ep100_doy21_40_mean",
        "ep100_doy52_71_mean",
        "boundary_padding",
    ]
    write_csv_atomic(frame, root / "verification" / "precursor_metrics.csv",
                     required_columns=required, exact_rows=90, overwrite=overwrite)

    for case, filename in (("0008-01", "figure08a.csv"), ("0008-02", "figure09a.csv")):
        subset = frame[frame.case == case]
        write_csv_atomic(_histogram_table(subset, threshold), root / "relationships" / filename,
                         required_columns=["bin_left_doy", "bin_right_doy", "count_all", "count_low", "count_other"],
                         exact_rows=13, overwrite=overwrite)

    ref_month = (ref_ep_dates // 100) % 100
    ref_doy = noleap_index(ref_ep_dates) + 1
    specifications = (
        ("figure08b.csv", "0008-01", "ep100_janfeb_mean", lambda x: np.ones(len(x), dtype=bool),
         float(np.nanmean(ref_ep[np.isin(ref_month, [1, 2])]))),
    )
    for filename, case, x_name, selector, ref_x in specifications:
        subset = frame[frame.case == case]
        table = _relationship_table(subset, case=case, x_name=x_name, selector=selector,
                                    ref_x=ref_x, ref_y=reference_details["minimum_du"])
        write_csv_atomic(table, root / "relationships" / filename,
                          required_columns=["case", "member", "x", "y", "is_low", "r", "p", "n", "slope", "intercept", "ref_x", "ref_y", "low25_threshold_du", "epflux_method", "minimum_method"],
                          exact_rows=len(table), overwrite=overwrite)

    # Main-text Figure 09d: after the February initialization, skip the first
    # 20 complete forecast days and average the following 20 days (days
    # 21--40, inclusive Feb 21--Mar 12).  The year-0008 reference point is
    # display context only and is excluded from the 30-member Pearson/OLS fit.
    february = frame.loc[frame.case.eq("0008-02")].copy()
    figure09d = _relationship_table(
        february,
        case="0008-02",
        x_name="ep100_doy52_71_mean",
        selector=lambda values: np.ones(len(values), dtype=bool),
        ref_x=float(np.nanmean(ref_ep[(ref_doy >= 52) & (ref_doy <= 71)])),
        ref_y=reference_details["minimum_du"],
    )
    figure09d["window_start_doy"] = 52
    figure09d["window_end_doy"] = 71
    figure09d["window_days"] = 20
    figure09d["forecast_day_start"] = 21
    figure09d["forecast_day_end"] = 40
    write_csv_atomic(
        figure09d,
        root / "relationships" / "figure09d.csv",
        required_columns=[
            "case", "member", "x", "y", "is_low", "r", "p", "n",
            "slope", "intercept", "ref_x", "ref_y",
            "low25_threshold_du", "epflux_method", "minimum_method",
            "window_start_doy", "window_end_doy", "window_days",
            "forecast_day_start", "forecast_day_end",
        ],
        exact_rows=30,
        overwrite=overwrite,
    )

    # Figure 8h uses the January-initialized 20-day EP100 mean as predictor
    # and the already validated Jan 1--May 30 (Nt=150) partial-O3 RMSE as the
    # predictand.  The reference EP line is display context only and is never
    # included in the Pearson/OLS fit.
    member_metrics = pd.read_csv(root / "verification" / "member_metrics.csv")
    january_ep = frame.loc[
        frame.case.eq("0008-01"),
        ["case", "member", "ep100_doy21_40_mean", "epflux_method"],
    ].copy()
    january_rmse = member_metrics.loc[
        member_metrics.case.eq("0008-01"),
        ["case", "member", "o3_rmse_du", "evaluation_start", "evaluation_end", "evaluation_nt"],
    ].copy()
    figure08h = january_ep.merge(
        january_rmse, on=["case", "member"], how="inner", validate="one_to_one"
    )
    if len(figure08h) != 30 or figure08h.member.astype(str).nunique() != 30:
        raise RuntimeError("Figure 8h requires exactly 30 unique January members")
    if set(figure08h.evaluation_nt.astype(int)) != {150}:
        raise RuntimeError("Figure 8h O3 RMSE must use exactly 150 daily values")
    figure08h = figure08h.rename(
        columns={"ep100_doy21_40_mean": "x", "o3_rmse_du": "y"}
    )
    statistics = regression(figure08h.x, figure08h.y)
    for name, value in statistics.items():
        figure08h[name] = value
    figure08h["ref_x"] = float(
        np.nanmean(ref_ep[(ref_doy >= 21) & (ref_doy <= 40)])
    )
    figure08h["window_start_doy"] = 21
    figure08h["window_end_doy"] = 40
    figure08h["window_days"] = 20
    figure08h["rmse_method"] = (
        "daily partial-O3 RMSE versus BWCN year 0008; Jan1-May30; Nt=150"
    )
    write_csv_atomic(
        figure08h,
        root / "relationships" / "figure08h.csv",
        required_columns=[
            "case", "member", "x", "y", "r", "p", "n", "slope",
            "intercept", "ref_x", "window_start_doy", "window_end_doy",
            "window_days", "evaluation_start", "evaluation_end",
            "evaluation_nt", "epflux_method", "rmse_method",
        ],
        exact_rows=30,
        overwrite=overwrite,
    )

    # Figure 11: all window means and their Pearson link, with the accepted
    # safety gate that the window must end before the earliest member minimum.
    cases, lengths, skips = ("0008-01", "0008-02"), np.arange(7, 46), np.arange(1, 61)
    shape = (len(cases), len(lengths), len(skips))
    values = {name: np.full(shape, np.nan) for name in ("r", "p", "n", "window_start_doy", "window_end_doy")}
    safe = np.zeros(shape, dtype=np.int8)
    for ci, case in enumerate(cases):
        init_doy = 1 if case.endswith("01") else 32
        with xr.open_dataset(root / "dynamics" / f"hindcast_{case}.nc", decode_times=False) as ds:
            ep = np.asarray(ds.ep100_upward_40_80n.values, dtype=float)
            dates = np.asarray(ds.date.values, dtype=int)
            ep_members = np.asarray(ds.member.values).astype(str)
        day = noleap_index(dates) + 1
        member_frame = frame[frame.case == case].set_index("member").loc[ep_members].reset_index()
        earliest = int(member_frame.minimum_doy.min())
        for li, length in enumerate(lengths):
            for si, skip in enumerate(skips):
                start = int(init_doy + skip - 1); end = int(start + length - 1)
                values["window_start_doy"][ci, li, si] = start
                values["window_end_doy"][ci, li, si] = end
                window = (day >= start) & (day <= end)
                if window.sum() != length:
                    continue
                x = np.nanmean(ep[:, window], axis=1)
                stat = regression(x, member_frame.minimum_du.to_numpy(float))
                values["r"][ci, li, si], values["p"][ci, li, si], values["n"][ci, li, si] = stat["r"], stat["p"], stat["n"]
                safe[ci, li, si] = int(end < earliest)
    scan = xr.Dataset(
        {name: (("case", "length", "skip"), array) for name, array in values.items()} |
        {"safe": (("case", "length", "skip"), safe)},
        coords={"case": list(cases), "length": lengths, "skip": skips},
        attrs={
            "product_version": PRODUCT_VERSION, "method": "Pearson of member EP100 window mean vs centered5 Mar-Apr O3 minimum",
            "window_lengths_days": "7--45", "initialization_skips_days": "1--60",
            "selected_marker": "skip=21,length=20; one diamond per panel",
            "low25_threshold_du": threshold, "source_manifest": str(root / "ozone" / "waccm_master_rankings.csv"),
        },
    )
    write_netcdf_atomic(scan, root / "relationships" / "figure11.nc",
                        required_vars={name: ("case", "length", "skip") for name in (*values, "safe")},
                        required_coords=("case", "length", "skip"), exact_sizes={"case": 2, "length": 39, "skip": 60}, overwrite=overwrite)
    return frame


def _canonical_event_rows(
    dates,
    source_years,
    values: dict[str, np.ndarray],
    ranking: pd.DataFrame,
    *,
    source: str,
    segment: str,
) -> pd.DataFrame:
    """Attach each raw day to one canonical ranked Oct--Sep event.

    January--September of source year Y belongs to ranked event Y.  October--
    December of source year Y belongs only to ranked successor Y+1.  Thus an
    EP padding year can supply its successor's autumn days but can never enter
    the climatology as a separate unranked event.
    """

    dates = np.asarray(dates, dtype=int)
    source_years = np.asarray(source_years, dtype=int)
    if dates.ndim != 1 or source_years.shape != dates.shape:
        raise ValueError(f"{source}/{segment}: date and source-year coordinates differ")
    for name, array in values.items():
        if np.asarray(array).shape != dates.shape:
            raise ValueError(f"{source}/{segment}: {name} length differs from date")
    required = {"event_id", "source_segment", "event_year", "model_year"}
    if required - set(ranking):
        raise ValueError(f"{source}/{segment}: ranking lacks {sorted(required - set(ranking))}")
    selected_ranking = ranking.loc[ranking.source_segment.astype(str) == segment].copy()
    if selected_ranking.empty:
        raise ValueError(f"{source}/{segment}: no canonical ranking rows")
    if selected_ranking.event_id.duplicated().any() or selected_ranking.model_year.duplicated().any():
        raise ValueError(f"{source}/{segment}: ranking event/model years must be unique")
    by_model_year = selected_ranking.set_index(selected_ranking.model_year.astype(int))

    calendar_year, month, day = date_parts(dates)
    if not np.array_equal(calendar_year, source_years):
        raise RuntimeError(
            f"{source}/{segment}: staged model_year differs from the integer-date year"
        )
    noleap = ~((month == 2) & (day == 29))
    canonical_model_year = source_years + (month >= 10).astype(int)
    canonical_id = pd.Series(canonical_model_year).map(by_model_year.event_id)
    keep = noleap & canonical_id.notna().to_numpy()
    canonical_model_year = canonical_model_year[keep]
    rows = by_model_year.loc[canonical_model_year]
    month, day = month[keep], day[keep]
    starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    noleap_day = starts[month - 1] + day - 1
    frame = pd.DataFrame({
        "source": source, "segment": segment,
        "event_id": rows.event_id.astype(str).to_numpy(),
        "event_year": rows.event_year.astype(int).to_numpy(),
        "model_year": rows.model_year.astype(int).to_numpy(),
        "source_year": source_years[keep], "month": month, "day": day,
        "calendar_key": month * 100 + day,
        "season_day": np.where(month >= 10, noleap_day - 273, noleap_day + 92),
        "is_predecessor_padding": source_years[keep] != canonical_model_year,
    })
    for name, array in values.items():
        frame[name] = np.asarray(array)[keep]
    if frame.duplicated(["event_id", "calendar_key"]).any():
        raise RuntimeError(f"{source}/{segment}: duplicate canonical event calendar days")
    allowed = set(selected_ranking.event_id.astype(str))
    if not set(frame.event_id).issubset(allowed):
        raise RuntimeError(f"{source}/{segment}: unranked event entered canonical daily frame")
    return frame


def _source_year_coordinate(dataset: xr.Dataset) -> np.ndarray:
    if "model_year" in dataset:
        return np.asarray(dataset.model_year.values, dtype=int)
    dates = np.asarray(dataset.date.values, dtype=int)
    return dates // 10000


def _validate_relationship_rankings(rankings: dict[str, pd.DataFrame]) -> None:
    """Require the exact canonical populations before any daily mapping."""

    expected_segments = {
        "MERRA2": {"MERRA2": 46},
        "WACCM": {"LONGRUN": 207, "BWCN": 23},
    }
    required = {"event_id", "source_segment", "event_year", "model_year"}
    for source, segment_counts in expected_segments.items():
        ranking = rankings[source]
        missing = required - set(ranking)
        if missing:
            raise ValueError(f"{source}: canonical ranking lacks {sorted(missing)}")
        if ranking.event_id.astype(str).duplicated().any():
            raise RuntimeError(f"{source}: canonical event IDs are not unique")
        actual_counts = ranking.groupby(ranking.source_segment.astype(str)).size().to_dict()
        if actual_counts != segment_counts:
            raise RuntimeError(
                f"{source}: canonical segment counts {actual_counts}; expected {segment_counts}"
            )
        if not np.array_equal(
            ranking.event_year.to_numpy(dtype=int),
            ranking.model_year.to_numpy(dtype=int),
        ):
            raise RuntimeError(f"{source}: event_year/model_year mapping is not one-to-one")
        expected_ids = (
            ranking.source_segment.astype(str)
            + ":"
            + ranking.model_year.astype(int).map(lambda year: f"{year:04d}")
        )
        if not np.array_equal(ranking.event_id.astype(str).to_numpy(), expected_ids.to_numpy()):
            raise RuntimeError(f"{source}: event IDs do not encode segment:model_year")
        for segment in segment_counts:
            selected = ranking.loc[ranking.source_segment.astype(str) == segment]
            if selected.model_year.astype(int).duplicated().any():
                raise RuntimeError(f"{source}/{segment}: model years are not unique")
    merra_years = set(rankings["MERRA2"].model_year.astype(int))
    if merra_years != set(range(1980, 2026)):
        raise RuntimeError("MERRA2 canonical ranking must be exactly 1980--2025")


def _merge_canonical_daily_frames(
    ep_frame: pd.DataFrame,
    nam_frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Reject any EP/NAM calendar mismatch before combining variables."""

    keys = [
        "source", "segment", "event_id", "event_year", "model_year",
        "source_year", "month", "day", "calendar_key", "season_day",
        "is_predecessor_padding",
    ]
    for name, frame in (("EP100", ep_frame), ("NAM/AO", nam_frame)):
        missing = set(keys) - set(frame)
        if missing:
            raise ValueError(f"{label}: {name} canonical frame lacks {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label}: {name} canonical keys are not unique")
    ep_keys = set(ep_frame[keys].itertuples(index=False, name=None))
    nam_keys = set(nam_frame[keys].itertuples(index=False, name=None))
    if ep_keys != nam_keys:
        ep_only = sorted(ep_keys - nam_keys)[:3]
        nam_only = sorted(nam_keys - ep_keys)[:3]
        raise RuntimeError(
            f"{label}: EP/NAM canonical calendars differ; "
            f"EP-only={ep_only}, NAM-only={nam_only}"
        )
    merged = ep_frame.merge(nam_frame, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(ep_frame) or len(merged) != len(nam_frame):
        raise RuntimeError(f"{label}: canonical inner merge changed the daily population")
    return merged


def _daily_frame(
    nam_path: Path,
    ep_path: Path,
    ranking: pd.DataFrame,
    *,
    source: str,
    segment: str,
) -> pd.DataFrame:
    with xr.open_dataset(nam_path, decode_times=False) as nam_ds:
        if not np.any(np.isclose(nam_ds.plev.values, 50.0)):
            raise RuntimeError(f"{nam_path}: exact 50-hPa NAM is missing")
        nam_frame = _canonical_event_rows(
            np.asarray(nam_ds.date.values, dtype=int),
            _source_year_coordinate(nam_ds),
            {
                "nam50": np.asarray(nam_ds.nam.sel(plev=50.0).values, dtype=float),
                "ao": np.asarray(nam_ds.ao.values, dtype=float),
            },
            ranking, source=source, segment=segment,
        )
    with xr.open_dataset(ep_path, decode_times=False) as ep_ds:
        ep_frame = _canonical_event_rows(
            np.asarray(ep_ds.date.values, dtype=int),
            _source_year_coordinate(ep_ds),
            {"ep100": np.asarray(ep_ds.ep100_upward_40_80n.values, dtype=float)},
            ranking, source=source, segment=segment,
        )
    frame = _merge_canonical_daily_frames(
        ep_frame, nam_frame, label=f"{source}/{segment}"
    )
    canonical_ids = set(
        ranking.loc[ranking.source_segment.astype(str) == segment, "event_id"].astype(str)
    )
    for variable, variable_frame in (("EP100", ep_frame), ("NAM/AO", nam_frame)):
        core = variable_frame.loc[variable_frame.month.isin([1, 2, 3, 4])]
        core_ids = set(core.event_id)
        if core_ids != canonical_ids:
            missing = sorted(canonical_ids - core_ids)
            raise RuntimeError(f"{source}/{segment}: {variable} lacks ranked Jan--Apr events {missing[:5]}")
        counts = core.groupby("event_id").calendar_key.nunique()
        if not (counts == 120).all():
            bad = counts[counts != 120].head().to_dict()
            raise RuntimeError(f"{source}/{segment}: {variable} Jan--Apr calendars incomplete {bad}")
    return frame.sort_values(["event_id", "season_day"], kind="stable").reset_index(drop=True)


def _standardize_canonical_ep(frame: pd.DataFrame, ranking: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Population-standardize EP100 by month-day over ranked events only."""

    output = frame.copy()
    canonical_ids = set(ranking.event_id.astype(str))
    if not set(output.event_id).issubset(canonical_ids):
        raise RuntimeError(f"{label}: unranked event reached EP standardization")
    if output.dropna(subset=["ep100"]).duplicated(["event_id", "calendar_key"]).any():
        raise RuntimeError(f"{label}: duplicate event/day samples in EP standardization")
    grouped = output.groupby("calendar_key")["ep100"]
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform(
        lambda values: np.nanstd(values.to_numpy(float), ddof=0)
    )
    output["ep100_daily_z"] = (
        output.ep100 - mean
    ) / standard_deviation.replace(0.0, np.nan)
    return output


def _synthetic_event_mapping_regression() -> None:
    """Guard against treating predecessor/padding years as extra events."""

    ranking = pd.DataFrame({
        "event_id": ["TEST:0002", "TEST:0003"],
        "source_segment": ["TEST", "TEST"],
        "event_year": [2, 3], "model_year": [2, 3],
    })
    dates = np.asarray([11101, 20101, 21101, 30101, 31101, 991101], dtype=int)
    source_years = np.asarray([1, 2, 2, 3, 3, 99], dtype=int)
    frame = _canonical_event_rows(
        dates, source_years, {"ep100": np.asarray([10., 1., 20., 2., 999., 888.])},
        ranking, source="TEST", segment="TEST",
    )
    expected = [
        ("TEST:0002", 1101, True), ("TEST:0002", 101, False),
        ("TEST:0003", 1101, True), ("TEST:0003", 101, False),
    ]
    actual = list(zip(frame.event_id, frame.calendar_key, frame.is_predecessor_padding))
    if actual != expected:
        raise AssertionError(f"canonical padding regression failed: {actual}")
    standardized = _standardize_canonical_ep(frame, ranking, label="synthetic")
    grouped = standardized.groupby("calendar_key").ep100_daily_z
    for _, values in grouped:
        if not np.allclose(np.sort(values.to_numpy(float)), [-1.0, 1.0]):
            raise AssertionError("unranked padding polluted synthetic calendar-day standardization")
    nam_frame = _canonical_event_rows(
        dates, source_years, {"nam50": np.asarray([3., 4., 5., 6., 777., 666.])},
        ranking, source="TEST", segment="TEST",
    )
    merged = _merge_canonical_daily_frames(frame, nam_frame, label="synthetic")
    if len(merged) != 4 or not merged.is_predecessor_padding.iloc[[0, 2]].all():
        raise AssertionError("synthetic EP/NAM padding rows were not retained")
    try:
        _merge_canonical_daily_frames(
            frame, nam_frame.iloc[1:].copy(), label="synthetic missing NAM padding"
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing synthetic NAM predecessor padding was not rejected")
    second_segment = ranking.assign(
        event_id=["SECOND:0002", "SECOND:0003"], source_segment="SECOND"
    )
    pooled_ranking = pd.concat([ranking, second_segment], ignore_index=True)
    second = _canonical_event_rows(
        dates, source_years, {"ep100": np.arange(dates.size, dtype=float)},
        pooled_ranking, source="WACCM", segment="SECOND",
    )
    if set(second.event_id) != {"SECOND:0002", "SECOND:0003"}:
        raise AssertionError("overlapping model years leaked across WACCM segments")


def _window_mean(frame: pd.DataFrame, variable: str, months: tuple[int, ...]) -> pd.Series:
    lengths = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    expected = sum(lengths[month] for month in months)
    selected = frame[frame.month.isin(months)]
    grouped = selected.groupby("event_id")[variable]
    mean = grouped.mean()
    count = grouped.count()
    return mean.where(count >= int(np.ceil(0.75 * expected)))


def _zscore(series: pd.Series) -> pd.Series:
    values = series.to_numpy(float)
    standard_deviation = np.nanstd(values, ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        return pd.Series(np.nan, index=series.index)
    return (series - np.nanmean(values)) / standard_deviation


def _sample_mask(frame: pd.DataFrame, sample: str) -> np.ndarray:
    if sample == "all":
        return np.ones(len(frame), dtype=bool)
    if sample == "low25":
        return frame.is_low25.astype(bool).to_numpy()
    if sample == "other":
        return ~frame.is_low25.astype(bool).to_numpy()
    raise ValueError(sample)


def _standardized_regression(x, y) -> dict[str, float | int]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    frame["x"] = _zscore(frame.x)
    frame["y"] = _zscore(frame.y)
    return regression(frame.x, frame.y)


def build_free_run_relationships(root: Path, *, overwrite: bool = False) -> None:
    """Build all Figure 2c/2d/2g point, scan, and statistic products."""

    _synthetic_event_mapping_regression()
    root = Path(root)
    rankings = {
        "MERRA2": pd.read_csv(root / "ozone" / "merra2_rankings.csv"),
        "WACCM": pd.read_csv(root / "ozone" / "waccm_master_rankings.csv"),
    }
    _validate_relationship_rankings(rankings)
    relationship_rankings = {
        "MERRA2": rankings["MERRA2"].copy(),
        "WACCM": rankings["WACCM"].loc[
            rankings["WACCM"].source_segment.astype(str) == "LONGRUN"
        ].copy(),
    }
    if len(relationship_rankings["WACCM"]) != 207:
        raise RuntimeError("Figure 2 WACCM relationships require 207 ranked LONGRUN events")
    frames = {
        "MERRA2": _daily_frame(
            root / "nam" / "merra2_daily_nam_ao.nc",
            root / "epflux" / "merra2_1980_2025_epflux.nc",
            relationship_rankings["MERRA2"], source="MERRA2", segment="MERRA2",
        ),
        "WACCM": _daily_frame(
            root / "nam" / "waccm_longrun_daily_nam_ao.nc",
            root / "epflux" / "waccm_longrun_epflux.nc",
            relationship_rankings["WACCM"], source="WACCM", segment="LONGRUN",
        ),
    }
    frames["MERRA2"] = _standardize_canonical_ep(
        frames["MERRA2"], relationship_rankings["MERRA2"], label="MERRA2 canonical ranking"
    )
    # Methods V7 defines the multi-spring WACCM relationship sample from the
    # long integration.  BWCN rows remain in the 230-event classification
    # master (and retain the fixed threshold) but do not enter these moments.
    frames["WACCM"] = _standardize_canonical_ep(
        frames["WACCM"], relationship_rankings["WACCM"],
        label="WACCM 207-event LONGRUN relationship sample",
    )

    point_products, statistic_rows = [], []
    for source, daily in frames.items():
        rank = relationship_rankings[source].copy()
        indexed = rank.set_index("event_id")
        points = pd.DataFrame(index=indexed.index)
        points["source"] = source
        points["segment"] = indexed.source_segment.astype(str)
        points["event_id"] = indexed.index
        points["event_year"] = indexed.event_year.astype(int)
        points["model_year"] = indexed.model_year.astype(int)
        points["is_low25"] = indexed.is_low25.astype(bool)
        points["o3_minimum_du"] = indexed.minimum_du.astype(float)
        points["ep100_djf"] = _window_mean(daily, "ep100_daily_z", (12, 1, 2))
        points["nam50_jfm"] = _window_mean(daily, "nam50", (1, 2, 3))
        points["nam50_jfma"] = _window_mean(daily, "nam50", (1, 2, 3, 4))
        points["ao_jfma"] = _window_mean(daily, "ao", (1, 2, 3, 4))
        points["epflux_method"] = "do_ubar=True; monthly natural-calendar N2; w=None; wave=-1; upward=-ep2; 40--80N cosine"
        points["ep_standardization_population"] = (
            "calendar-day ddof=0 over canonical ranked event IDs; predecessor Oct-Dec attached only to ranked successor"
        )
        points["ep_standardization_master_count"] = len(indexed)
        points["relationship_sample_scope"] = (
            "LONGRUN ranked events only; low25 flags inherited from combined 230-event master"
            if source == "WACCM" else "MERRA2 1980--2025 ranked events"
        )
        points["nam_method"] = "calendar-month fixed EOF; true no-leap month-day centered21 climatology; AO=1000hPa NAM"
        points["minimum_method"] = "centered5 Mar1-Apr30; exact 30--70hPa; 60--90N cosine"
        points["ranking_manifest"] = str(
            root / "ozone" / ("merra2_rankings.csv" if source == "MERRA2" else "waccm_master_rankings.csv")
        )
        points = points.dropna(subset=["ep100_djf", "nam50_jfm", "nam50_jfma", "ao_jfma", "o3_minimum_du"])
        for name in ("ep100_djf", "nam50_jfm", "nam50_jfma", "ao_jfma", "o3_minimum_du"):
            points[f"{name}_z"] = _zscore(points[name])
        point_products.append(points.reset_index(drop=True))
        relations = (
            ("ep_nam", "ep100_djf_z", "nam50_jfm_z"),
            ("ep_o3", "ep100_djf_z", "o3_minimum_du"),
            ("nam_ao", "nam50_jfma_z", "ao_jfma_z"),
        )
        for relation, x_name, y_name in relations:
            for sample in ("all", "low25", "other"):
                selected = points.loc[_sample_mask(points, sample)]
                stat = regression(selected[x_name], selected[y_name])
                statistic_rows.append({
                    "source": source, "relation": relation, "sample": sample,
                    "x_metric": x_name, "y_metric": y_name, **stat,
                    "epflux_method": "do_ubar=True; monthly natural-calendar N2; w=None; wave=-1",
                    "ep_standardization_population": "canonical ranked event IDs; predecessor Oct-Dec attached only to ranked successor; ddof=0",
                    "ep_standardization_master_count": len(indexed),
                    "relationship_sample_scope": points.relationship_sample_scope.iloc[0],
                    "ranking_manifest": points.ranking_manifest.iloc[0],
                })

    point_frame = pd.concat(point_products, ignore_index=True)
    point_columns = [
        "source", "segment", "event_id", "event_year", "model_year", "is_low25",
        "ep100_djf", "nam50_jfm", "nam50_jfma", "ao_jfma", "o3_minimum_du",
        "ep100_djf_z", "nam50_jfm_z", "nam50_jfma_z", "ao_jfma_z", "o3_minimum_du_z",
        "epflux_method", "nam_method", "minimum_method", "ranking_manifest",
        "ep_standardization_population",
        "ep_standardization_master_count",
        "relationship_sample_scope",
    ]
    write_csv_atomic(point_frame, root / "relationships" / "figure02c.csv",
                     required_columns=point_columns, overwrite=overwrite)
    write_csv_atomic(pd.DataFrame(statistic_rows), root / "relationships" / "figure02c_stats.csv",
                     required_columns=["source", "relation", "sample", "x_metric", "y_metric", "r", "p", "n", "slope", "intercept", "ep_standardization_population", "ep_standardization_master_count", "relationship_sample_scope"],
                     exact_rows=18, overwrite=overwrite)

    sources, samples = ("MERRA2", "WACCM"), ("all", "low25", "other")
    lengths, starts = np.arange(15, 121, 15), np.arange(0, 151, 5)
    shape = (2, 3, len(lengths), len(starts))
    arrays = {name: np.full(shape, np.nan) for name in ("r", "p", "n")}
    valid = np.zeros(shape, dtype=np.int8)
    for source_index, source in enumerate(sources):
        daily, rank = frames[source], relationship_rankings[source].set_index("event_id")
        for length_index, length in enumerate(lengths):
            for start_index, start in enumerate(starts):
                if start + length > 151:
                    continue
                selected = daily[(daily.season_day >= start) & (daily.season_day < start + length)]
                grouped = selected.groupby("event_id").ep100_daily_z
                means = grouped.mean().where(grouped.count() >= int(np.ceil(0.75 * length)))
                relation = pd.DataFrame({"x": means, "y": rank.minimum_du, "is_low25": rank.is_low25}).dropna()
                for sample_index, sample in enumerate(samples):
                    part = relation.loc[_sample_mask(relation, sample)]
                    stat = regression(part.x, part.y)
                    for name in arrays:
                        arrays[name][source_index, sample_index, length_index, start_index] = stat[name]
                    valid[source_index, sample_index, length_index, start_index] = int(stat["n"] >= 3)
    scan = xr.Dataset(
        {name: (("source", "sample", "window_length", "start_day"), value) for name, value in arrays.items()} |
        {"valid": (("source", "sample", "window_length", "start_day"), valid)},
        coords={"source": list(sources), "sample": list(samples), "window_length": lengths, "start_day": starts},
        attrs={
            "product_version": PRODUCT_VERSION, "method": "calendar-day standardized EP100 window mean vs fixed centered5 Mar-Apr O3 minimum; Pearson",
            "start_day_origin": "0=Oct1", "window_constraint": "ends no later than Feb28",
            "source_manifests": "ozone/merra2_rankings.csv;ozone/waccm_master_rankings.csv",
            "ep_standardization_population": "canonical ranked event IDs; predecessor Oct-Dec attached only to ranked successor; ddof=0",
            "ep_standardization_master_counts": (
                f"MERRA2={len(relationship_rankings['MERRA2'])};"
                f"WACCM_LONGRUN={len(relationship_rankings['WACCM'])}"
            ),
            "waccm_relationship_scope": (
                "LONGRUN ranked events only; low25 threshold/flags inherited from combined 230-event master"
            ),
        },
    )
    write_netcdf_atomic(scan, root / "relationships" / "figure02d.nc",
                        required_vars={name: ("source", "sample", "window_length", "start_day") for name in ("r", "p", "n", "valid")},
                        required_coords=("source", "sample", "window_length", "start_day"),
                        exact_sizes={"source": 2, "sample": 3, "window_length": 8, "start_day": 31}, overwrite=overwrite)

    ep_nam = [
        ("O→N", (10,), (11,), 1), ("N→D", (11,), (12,), 1),
        ("D→J", (12,), (1,), 1), ("J→F", (1,), (2,), 1), ("F→M", (2,), (3,), 1),
        ("ON→ND", (10, 11), (11, 12), 2), ("ND→DJ", (11, 12), (12, 1), 2),
        ("DJ→JF", (12, 1), (1, 2), 2), ("JF→FM", (1, 2), (2, 3), 2),
        ("OND→NDJ", (10, 11, 12), (11, 12, 1), 3),
        ("NDJ→DJF", (11, 12, 1), (12, 1, 2), 3),
        ("DJF→JFM", (12, 1, 2), (1, 2, 3), 3),
    ]
    ep_o3 = [
        ("O", (10,), 1), ("N", (11,), 1), ("D", (12,), 1), ("J", (1,), 1), ("F", (2,), 1),
        ("ON", (10, 11), 2), ("ND", (11, 12), 2), ("DJ", (12, 1), 2), ("JF", (1, 2), 2),
        ("OND", (10, 11, 12), 3), ("NDJ", (11, 12, 1), 3), ("DJF", (12, 1, 2), 3),
    ]
    nam_ao = [("JFMA", (1, 2, 3, 4), 4), ("FMA", (2, 3, 4), 3), ("MA", (3, 4), 2), ("A", (4,), 1)]
    rows = []
    for source, daily in frames.items():
        rank = relationship_rankings[source].set_index("event_id")
        specifications = []
        for label, x_months, y_months, duration in ep_nam:
            specifications.append(("ep_nam", label, duration, label == "DJF→JFM", _window_mean(daily, "ep100_daily_z", x_months), _window_mean(daily, "nam50", y_months), "ep100", "nam50"))
        for label, x_months, duration in ep_o3:
            specifications.append(("ep_o3", label, duration, label == "DJF", _window_mean(daily, "ep100_daily_z", x_months), rank.minimum_du, "ep100", "o3_minimum_du"))
        for label, months, duration in nam_ao:
            specifications.append(("nam_ao", label, duration, label == "JFMA", _window_mean(daily, "nam50", months), _window_mean(daily, "ao", months), "nam50", "ao"))
        for relation, label, duration, selected_flag, x, y, x_metric, y_metric in specifications:
            values = pd.DataFrame({"x": x, "y": y, "is_low25": rank.is_low25}).dropna()
            for sample in ("all", "other"):
                part = values.loc[_sample_mask(values, sample)]
                stat = _standardized_regression(part.x, part.y)
                rows.append({
                    "source": source, "relation": relation, "window_label": label,
                    "duration": duration, "selected": selected_flag, "sample": sample,
                    "x_metric": x_metric, "y_metric": y_metric, **stat,
                    "epflux_method": "do_ubar=True; monthly natural-calendar N2; w=None; wave=-1; upward=-ep2; 40--80N cosine",
                    "ep_standardization_population": "canonical ranked event IDs; predecessor Oct-Dec attached only to ranked successor; ddof=0",
                    "ep_standardization_master_count": len(rank),
                    "relationship_sample_scope": (
                        "LONGRUN ranked events only; low25 flags inherited from combined 230-event master"
                        if source == "WACCM" else "MERRA2 1980--2025 ranked events"
                    ),
                    "ranking_manifest": str(root / "ozone" / ("merra2_rankings.csv" if source == "MERRA2" else "waccm_master_rankings.csv")),
                })
    discrete = pd.DataFrame(rows)
    write_csv_atomic(discrete, root / "relationships" / "figure02g.csv",
                     required_columns=["source", "relation", "window_label", "duration", "selected", "sample", "x_metric", "y_metric", "r", "p", "n", "slope", "intercept", "ep_standardization_population", "ep_standardization_master_count", "relationship_sample_scope"],
                     exact_rows=112, overwrite=overwrite)


def _event_month_days() -> list[tuple[int, int]]:
    lengths = {11: 30, 12: 31, 1: 31, 2: 28, 3: 31, 4: 30, 5: 31}
    return [(month, day) for month in (11, 12, 1, 2, 3, 4, 5) for day in range(1, lengths[month] + 1)]


def _target_event(dataset: xr.Dataset, variable: str, event_year: int) -> xr.DataArray:
    dates = np.asarray(dataset.date.values, dtype=int)
    year, month, day = date_parts(dates)
    assigned = year + (month >= 10).astype(int)
    keep = (assigned == event_year) & np.isin(month, [11, 12, 1, 2, 3, 4, 5])
    selected = dataset[variable].isel(time=np.flatnonzero(keep)).load()
    selected_dates = dates[keep]
    lookup = {int(date % 10000): index for index, date in enumerate(selected_dates)}
    expected = _event_month_days()
    indices = []
    for month_value, day_value in expected:
        key = month_value * 100 + day_value
        if key not in lookup:
            raise RuntimeError(f"{variable}: target event lacks month-day {key:04d}")
        indices.append(lookup[key])
    return selected.isel(time=indices).assign_coords(date=("time", np.asarray([
        (event_year - 1 if month_value >= 10 else event_year) * 10000 + month_value * 100 + day_value
        for month_value, day_value in expected
    ], dtype=int)))


def _target_excluded_o3_context(datasets: list[xr.Dataset], target_event_id: str) -> tuple[pd.DataFrame, float]:
    records = []
    for dataset in datasets:
        dates = np.asarray(dataset.date.values, dtype=int)
        year, month, day = date_parts(dates)
        segment = str(dataset.attrs.get("source_segment", "MERRA2"))
        event_year = year + (month >= 10).astype(int)
        prefix = "MERRA2" if segment == "MERRA2" else segment
        values = np.asarray(dataset.partial_o3_du.values, dtype=float)
        for value, yy, mm, dd in zip(values, event_year, month, day):
            if mm == 2 and dd == 29:
                continue
            records.append({
                "event_id": f"{prefix}:{int(yy):04d}", "event_year": int(yy),
                "month": int(mm), "day": int(dd), "calendar_key": int(mm * 100 + dd),
                "value": float(value),
            })
    frame = pd.DataFrame(records)
    climatology = (
        frame.loc[frame.event_id != target_event_id]
        .groupby("calendar_key").value.mean()
    )
    target = frame[(frame.event_id == target_event_id) & frame.month.isin([11, 12, 1, 2, 3, 4, 5])].copy()
    target["anomaly"] = target.value - target.calendar_key.map(climatology)
    target = target.set_index("calendar_key").loc[
        [month * 100 + day for month, day in _event_month_days()]
    ].reset_index()
    target["value_rm5"] = target.value.rolling(5, center=True, min_periods=5).mean()
    target["anomaly_rm5"] = target.anomaly.rolling(5, center=True, min_periods=5).mean()
    spring = target[target.month.isin([3, 4])].dropna(subset=["value_rm5"])
    minimum = spring.sort_values(["value_rm5", "calendar_key"], kind="mergesort").iloc[0]
    return target, float(minimum.value_rm5)


def build_figure04_context(root: Path, *, overwrite: bool = False) -> None:
    """Build the fully calculated NAM/AO/O3 event context used only for plotting."""

    root = Path(root)
    with xr.open_dataset(root / "nam" / "merra2_daily_nam_ao.nc", decode_times=False) as ds:
        merra_nam = _target_event(ds, "nam", 2020)
        merra_ao = _target_event(ds, "ao", 2020)
    with xr.open_dataset(root / "nam" / "waccm_bwcn_daily_nam_ao.nc", decode_times=False) as ds:
        waccm_nam = _target_event(ds, "nam", 8)
        waccm_ao = _target_event(ds, "ao", 8)
    if not np.array_equal(merra_nam.plev.values, waccm_nam.plev.values):
        raise RuntimeError("Figure 4 requires the identical canonical 23-level NAM grid")

    merra_ds = xr.open_dataset(root / "ozone" / "merra2_1980_2025_partial_o3.nc", decode_times=False)
    long_ds = xr.open_dataset(root / "ozone" / "longrun_partial_o3.nc", decode_times=False)
    bwcn_ds = xr.open_dataset(root / "ozone" / "bwcn_partial_o3.nc", decode_times=False)
    try:
        merra_o3, merra_minimum = _target_excluded_o3_context([merra_ds], "MERRA2:2020")
        waccm_o3, waccm_minimum = _target_excluded_o3_context([long_ds, bwcn_ds], "BWCN:0008")
    finally:
        merra_ds.close(); long_ds.close(); bwcn_ds.close()

    with xr.open_dataset(root / "ozone" / "event_profile_bootstrap5000.nc", decode_times=False) as ds:
        keep = (np.asarray(ds.season_day.values, dtype=int) >= 31) & (np.asarray(ds.season_day.values, dtype=int) <= 242)
        merra_profile = np.asarray(ds.merra2_o3_anomaly.isel(season_day=np.flatnonzero(keep)).values, dtype=float)
        waccm_profile = np.asarray(ds.waccm_o3_anomaly.isel(season_day=np.flatnonzero(keep)).values, dtype=float)
        merra_pressure = np.asarray(ds.merra2_pressure_hpa.values, dtype=float)
        waccm_pressure = np.asarray(ds.waccm_pressure_hpa.values, dtype=float)
    if not np.array_equal(merra_pressure, waccm_pressure):
        raise RuntimeError("Figure 4 event-profile pressure grids must be identical")

    date_coord = np.asarray([month * 100 + day for month, day in _event_month_days()], dtype=int)
    # Locate minima by the already ordered event table, never by plotting code.
    merra_minimum_day = int(np.nanargmin(np.where(merra_o3.month.isin([3, 4]), merra_o3.value_rm5, np.nan)))
    waccm_minimum_day = int(np.nanargmin(np.where(waccm_o3.month.isin([3, 4]), waccm_o3.value_rm5, np.nan)))
    output = xr.Dataset(
        {
            "nam": (("source", "event_day", "plev"), np.stack([merra_nam.values, waccm_nam.values])),
            "ao": (("source", "event_day"), np.stack([merra_ao.values, waccm_ao.values])),
            "partial_o3_du": (("source", "event_day"), np.stack([merra_o3.value, waccm_o3.value])),
            "partial_o3_anomaly_rm5_du": (("source", "event_day"), np.stack([merra_o3.anomaly_rm5, waccm_o3.anomaly_rm5])),
            "o3_profile_anomaly": (("source", "event_day", "profile_pressure_hpa"), np.stack([merra_profile, waccm_profile])),
            "minimum_du": ("source", [merra_minimum, waccm_minimum]),
            "minimum_anomaly_du": ("source", [float(merra_o3.anomaly_rm5.iloc[merra_minimum_day]), float(waccm_o3.anomaly_rm5.iloc[waccm_minimum_day])]),
            "minimum_event_day": ("source", [merra_minimum_day, waccm_minimum_day]),
        },
        coords={
            "source": ["MERRA2", "WACCM"], "event_day": np.arange(212),
            "date": ("event_day", date_coord), "plev": merra_nam.plev.values,
            "profile_pressure_hpa": merra_pressure,
        },
        attrs={
            "product_version": PRODUCT_VERSION, "method": "fixed EOF NAM; AO=1000hPa NAM; target-excluded month-day O3 climatology; centered5",
            "event_window": "Nov1--May31 no-leap", "ozone_column": "exact 30--70hPa then 60--90N cosine mean",
            "minimum_window": "Mar1--Apr30", "profile_source": "ozone/event_profile_bootstrap5000.nc",
            "source_manifests": "ozone/merra2_rankings.csv;ozone/waccm_master_rankings.csv",
        },
    )
    output.plev.attrs.update(units="hPa", positive="down")
    output.profile_pressure_hpa.attrs.update(units="hPa", positive="down")
    write_netcdf_atomic(
        output, root / "cases" / "figure04_context.nc",
        required_vars={
            "nam": ("source", "event_day", "plev"), "ao": ("source", "event_day"),
            "partial_o3_du": ("source", "event_day"), "partial_o3_anomaly_rm5_du": ("source", "event_day"),
            "o3_profile_anomaly": ("source", "event_day", "profile_pressure_hpa"),
            "minimum_du": ("source",), "minimum_anomaly_du": ("source",), "minimum_event_day": ("source",),
        }, required_coords=("source", "event_day", "date", "plev", "profile_pressure_hpa"),
        exact_sizes={"source": 2, "event_day": 212, "plev": 23}, overwrite=overwrite,
    )
