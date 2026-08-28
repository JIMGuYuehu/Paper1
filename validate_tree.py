"""Release validation for the cleaned Paper 1 workflow.

The default checks use only the Python standard library. They enforce the
exact selected-figure scope, notebook explain/code pairing, cleared outputs,
Python syntax, one plotting block per figure, and absence of build/runtime
debris. ``--require-scientific`` additionally imports the STREAM scientific
stack and runs deterministic kernel tests. ``--require-products`` executes all
14 plotting notebooks against an already completed staging tree and verifies
the 24 PNG/PDF pairs; it deliberately does not launch the expensive raw-data
or diagnostic reconstruction.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


# Validation must not make the source tree fail its own no-cache gate on the
# next run.  Notebook kernels inherit the environment setting below.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


ROOT = Path(__file__).resolve().parent
PREPROCESSING = ROOT / "01_preprocessing"
DIAGNOSTICS = ROOT / "02_diagnostics"
PLOTTING = ROOT / "03_plotting"
MANIFEST = ROOT / "MANUSCRIPT_FIGURE_MANIFEST.csv"
MANUSCRIPT = ROOT.parent

EXPECTED_SHELL = (
    "00_common.sh",
    "01_waccm_longrun_extract.sh",
    "02_waccm_restart_source_extract.sh",
    "03_waccm_restart_ensembles_extract.sh",
    "04_marina_march_inventory.sh",
    "05_merra2_daily_pressure_levels.sh",
    "06_mls_level3_inventory.sh",
    "07_waccm_pressure_level_products.sh",
)
EXPECTED_PREPROCESSING_PYTHON = (
    "08_extr_clox_200yr.py",
    "09_lightweight_reproduction_check.py",
    "paper1_noleap.py",
    "paper1_source_audit.py",
    "test_source_contracts.py",
)
EXPECTED_DIAGNOSTICS = {
    "01_partial_ozone_column.ipynb": 5,
    "02_nam_ao_fixed_eof.ipynb": 4,
    "03_ep_flux.ipynb": 7,
    "04_mls_clo_h2o_anomalies.ipynb": 3,
    "05_hindcast_verification.ipynb": 12,
    "06_marina_march_products.ipynb": 4,
    "07_figure15_low_o3_bootstrap.ipynb": 11,
}
EXPECTED_PLOTS = {
    "figure01_o3_event_context.ipynb": (
        "figure01a_o3_event_context",
        "figure01bc_waccm0008_merra2_2020_o3_anomaly_1to100hpa",
    ),
    "figure02_precursor_relationships.ipynb": (
        "figure2c_ubar_MERRA2NEWNAM", "figA1", "figA2",
    ),
    "figure03_chemistry_dehydration.ipynb": (
        "figure3_clo", "figure3_h2o",
    ),
    "figure04_nam_o3_context.ipynb": (
        "figure04_merra2_2020_waccm0008_nam_o3_context_MERRA2NEWNAM",
    ),
    "figure05_hindcast_evolution.ipynb": (
        "figure05a_hindcast_o3_evolution",
        "figure05b_hindcast_u60n10_tmin50_evolution",
    ),
    "figure06_hindcast_spread_timing.ipynb": (
        "figure06b",
    ),
    "figure07_rmse_pathway.ipynb": (
        "figure07a_u60n10_rmse_vs_o3_rmse",
        "figure07b_tmin50_rmse_vs_u60n10_rmse",
        "figure07c_tmin50_rmse_vs_o3_rmse",
    ),
    "figure08_january_wave_precursor.ipynb": (
        "figure08a", "figure08b_jan_wave_vs_o3minimum_raw", "figure08h",
    ),
    "figure09_february_wave_precursor.ipynb": (
        "figure09a_feb_o3_minimum_date_histogram",
        "figure09b_feb_wave_vs_o3minimum_raw",
    ),
    "figure11_wave_window_scan.ipynb": (
        "figure11b",
    ),
    "figure15_low_o3_z300_epflux.ipynb": (
        "figure15f_ubar_N2correction_bootstrap",
    ),
    "figure16_daily_spread_crps.ipynb": ("figure16b_daily",),
    "figure17_nam_evolution.ipynb": ("figure17a",),
    "figure18_ao_evolution.ipynb": ("figure18a",),
}

FORBIDDEN_SCOPE_TEXT = (
    "nocoupl", "clim2d", "clim3d", "int-3d-ncar", "spearman",
    "ubar_wcorr", "figure09c", "figure9c", "figure16c", "figure17b",
    "figure17c", "figure19", "figure20", "figure21", "figure08d",
    "figure16a", "doy51_72", "20 feb", "13 mar",
)
FORBIDDEN_PLOT_COMPUTATION = (
    ".rolling(", ".groupby(", "np.nanmean(", "np.nanstd(",
    "pearsonr(", "linregress(", "polyfit(", ".corr(", "scipy.stats",
    "crps_ensemble(", "bootstrap_event_profile(",
    "random_same_size_composites(", "rank_events(", "spring_minimum(",
    "compute_epflux", "train_fixed_nam_reference(", "project_nam(",
)
FORBIDDEN_PLOT_PATH_TEXT = (
    "layout_revision", "chat审阅", "candidate/", "candidate\\",
    "codex_layout_revision",
)
RUNTIME_SUFFIXES = {".nc", ".csv", ".parquet", ".png", ".pdf", ".pyc"}


class ValidationError(RuntimeError):
    """A deterministic release-contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    require(notebook.get("nbformat") == 4, f"{path}: expected nbformat 4")
    metadata_text = json.dumps(notebook.get("metadata", {}), sort_keys=True).lower()
    require("generated_by" not in metadata_text, f"{path}: builder metadata remains")
    return notebook


def validate_notebook(path: Path, expected_cells: int, kind: str) -> tuple[int, set[str]]:
    notebook = load_notebook(path)
    expected_stems = set(EXPECTED_PLOTS.get(path.name, ()))
    seen_stems: set[str] = set()
    code_count = 0
    all_text: list[str] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        all_text.append(source)
        if cell.get("cell_type") != "code":
            continue
        code_count += 1
        require(
            index > 0 and notebook["cells"][index - 1].get("cell_type") == "markdown",
            f"{path}: code cell {index} lacks an immediately preceding explanation",
        )
        explanation = "".join(notebook["cells"][index - 1].get("source", [])).lower()
        labels = ("inputs:", "outputs:", "plot action:") if kind == "plot" else (
            "inputs:", "outputs:", "method:",
        )
        for label in labels:
            require(label in explanation, f"{path}: explanation before cell {index} lacks {label}")
        require(cell.get("execution_count") is None, f"{path}: stored execution count")
        require(cell.get("outputs", []) == [], f"{path}: stored output in cell {index}")
        try:
            tree = ast.parse(source, filename=f"{path}:{index}")
        except SyntaxError as error:
            raise ValidationError(f"{path}: code cell {index} does not compile: {error}") from error

        if kind == "plot":
            lowered = source.lower()
            for token in FORBIDDEN_PLOT_COMPUTATION:
                require(token not in lowered, f"{path}: plotting cell recalculates {token}")
            for token in FORBIDDEN_PLOT_PATH_TEXT:
                require(token not in lowered, f"{path}: plotting cell uses old/raw path token {token}")
            absolute_mnt_strings = {
                node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("/mnt/")
            }
            require(
                absolute_mnt_strings <= {
                    "/mnt/backup_ETH",
                    "/mnt/soclim0/public_data/weiji",
                },
                f"{path}: non-staging /mnt path remains: {absolute_mnt_strings}",
            )
            require("canonical_path(" in source, f"{path}: plotting cell bypasses staging products")
            require(".png" in source and ".pdf" in source, f"{path}: PNG/PDF writer missing")
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "save_figure"
            ]
            require(len(calls) == 1, f"{path}: cell {index} must call save_figure exactly once")
            matched = {stem for stem in expected_stems if stem in source}
            require(len(matched) == 1, f"{path}: cell {index} must name exactly one selected stem")
            seen_stems.update(matched)

    require(code_count == expected_cells, f"{path}: {code_count} code cells; expected {expected_cells}")
    text = "\n".join(all_text).lower()
    if kind == "plot":
        for token in FORBIDDEN_SCOPE_TEXT:
            require(token not in text, f"{path}: out-of-scope token remains: {token}")
        require(not re.search(r"[a-z]:\\", text), f"{path}: Windows absolute path remains")
        require(seen_stems == expected_stems, f"{path}: output stems {seen_stems} != {expected_stems}")
    return code_count, seen_stems


def validate_preprocessing() -> None:
    shell_names = tuple(path.name for path in sorted(PREPROCESSING.glob("*.sh")))
    require(shell_names == EXPECTED_SHELL, f"preprocessing shell files {shell_names}")
    for name in EXPECTED_SHELL:
        path = PREPROCESSING / name
        text = path.read_text(encoding="utf-8")
        require(text.startswith("#!/usr/bin/env bash"), f"{path}: Bash shebang missing")
        require("set -euo pipefail" in text, f"{path}: strict Bash mode missing")
        for heading in ("INPUT", "OUTPUT", "ACTION"):
            require(f"# {heading}" in text, f"{path}: {heading} header missing")

    python_names = tuple(
        path.name for path in sorted(PREPROCESSING.glob("*.py"))
    )
    require(
        python_names == EXPECTED_PREPROCESSING_PYTHON,
        f"preprocessing Python files {python_names}",
    )
    for name in EXPECTED_PREPROCESSING_PYTHON:
        path = PREPROCESSING / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        docstring = (ast.get_docstring(tree) or "").upper()
        for heading in ("INPUT", "OUTPUT", "ACTION"):
            require(heading in docstring, f"{path}: module docstring lacks {heading}")

    pressure_shell = (PREPROCESSING / "07_waccm_pressure_level_products.sh").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^PLEV_PA="([0-9,]+)"$', pressure_shell, flags=re.MULTILINE)
    require(match is not None, "07 pressure-level script lacks a literal PLEV_PA contract")
    shell_grid_pa = tuple(int(value) for value in match.group(1).split(","))
    expected_grid_pa = (
        100000, 95000, 92500, 90000, 85000, 80000, 75000, 70000, 60000,
        55000, 50000, 45000, 40000, 35000, 30000, 25000, 22500, 20000,
        17500, 15000, 12500, 10000, 7000, 5000, 4000, 3000, 2000, 1000,
        700, 500, 400, 300, 200, 100, 50, 10,
    )
    require(
        shell_grid_pa == expected_grid_pa,
        "07 pressure-level grid is not the exact 36-level contract (including 40/4 hPa)",
    )
    pressure_contracts = (
        "--segment LONGRUN",
        "--expected-raw-years 210",
        "--expected-ranking-events 207",
        "--expected-pressure-years 209",
        "--expected-field-events 206",
        "--segment BWCN",
        "--expected-raw-years 24",
        "--expected-ranking-events 23",
        "--expected-pressure-years 23",
        "--expected-field-events 22",
        "--expected-unavailable-events 1",
        'expected_tasks}" -eq 1168',
        'expected=292',
    )
    for contract in pressure_contracts:
        require(contract in pressure_shell, f"07 pressure-year/task contract missing: {contract}")

    audit_tree = ast.parse(
        (PREPROCESSING / "paper1_source_audit.py").read_text(encoding="utf-8")
    )
    audit_grid: tuple[int, ...] | None = None
    for node in audit_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "WACCM_PLEV_PA"
                   for target in node.targets):
            continue
        call = node.value
        if (isinstance(call, ast.Call) and call.args
                and isinstance(call.args[0], (ast.List, ast.Tuple))):
            audit_grid = tuple(ast.literal_eval(call.args[0]))
    require(audit_grid == expected_grid_pa, "source-audit and CDO WACCM grids differ")


def validate_source_tree() -> None:
    forbidden_names = []
    runtime_files = []
    runtime_root = (ROOT / "runtime").resolve()
    for path in ROOT.rglob("*"):
        # Runtime products are intentionally confined to this ignored subtree.
        # They are validated by --require-products, not mistaken for source files.
        resolved = path.resolve()
        if resolved == runtime_root or runtime_root in resolved.parents:
            continue
        relative = path.relative_to(ROOT)
        lowered = path.name.lower()
        if path.is_dir() and lowered in {"__pycache__", ".ipynb_checkpoints"}:
            forbidden_names.append(str(relative))
        if path.is_file():
            if lowered.startswith("build_") or lowered.endswith((".tmp", ".new", ".tmp.nc")):
                forbidden_names.append(str(relative))
            if (
                path.suffix.lower() in RUNTIME_SUFFIXES
                and relative.parts[:1] != ("assets",)
                and path != MANIFEST
            ):
                runtime_files.append(str(relative))
            if path.suffix == ".py":
                source = path.read_text(encoding="utf-8")
                try:
                    compile(source, str(path), "exec")
                except SyntaxError as error:
                    raise ValidationError(f"{path}: Python syntax failure: {error}") from error
    require(not forbidden_names, f"builder/cache/temp artifacts remain: {forbidden_names}")
    require(not runtime_files, f"runtime data/figures are stored with source: {runtime_files}")


def validate_manuscript_manifest() -> None:
    """Validate the 24-row manifest and, when present, its paper snapshot."""

    require(MANIFEST.is_file(), "MANUSCRIPT_FIGURE_MANIFEST.csv is missing")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 24, f"figure manifest has {len(rows)} rows; expected 24")
    paths = [row["manuscript_path"] for row in rows]
    require(len(set(paths)) == 24, "figure manifest contains duplicate paths")
    require(
        {row["output_stem"] for row in rows}
        == {stem for stems in EXPECTED_PLOTS.values() for stem in stems},
        "manifest output stems differ from plotting scope",
    )
    for row in rows:
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                f"invalid manifest SHA-256: {row['manuscript_path']}")
        require(int(row["bytes"]) > 0,
                f"invalid byte count: {row['manuscript_path']}")

    # Standalone clones contain code and the manifest only. In the author's
    # manuscript checkout the parent directory also contains the exact assets;
    # verify their bytes, hashes and TeX references when that snapshot exists.
    if not (MANUSCRIPT / "results.tex").is_file():
        return
    for row in rows:
        relative = row["manuscript_path"]
        asset = MANUSCRIPT / relative
        require(asset.is_file(), f"manifest asset is missing: {relative}")
        require(asset.stat().st_size == int(row["bytes"]),
                f"manifest byte count changed: {relative}")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        require(digest == row["sha256"], f"manifest SHA-256 changed: {relative}")

    referenced: set[str] = set()
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}")
    for tex_name in ("results.tex", "appendix.tex"):
        tex_path = MANUSCRIPT / tex_name
        require(tex_path.is_file(), f"current manuscript file is missing: {tex_name}")
        referenced.update(match.replace("\\", "/") for match in pattern.findall(
            tex_path.read_text(encoding="utf-8")
        ))
    require(referenced == set(paths),
            f"manifest/current TeX figure sets differ: manifest-only={sorted(set(paths)-referenced)}, "
            f"TeX-only={sorted(referenced-set(paths))}")


def validate_static() -> None:
    validate_source_tree()
    validate_manuscript_manifest()
    validate_preprocessing()
    diagnostic_names = {path.name for path in DIAGNOSTICS.glob("*.ipynb")}
    plotting_names = {path.name for path in PLOTTING.glob("*.ipynb")}
    require(diagnostic_names == set(EXPECTED_DIAGNOSTICS), "diagnostic notebook set differs")
    require(plotting_names == set(EXPECTED_PLOTS), "plotting notebook set differs")

    diagnostic_cells = sum(
        validate_notebook(DIAGNOSTICS / name, count, "diagnostic")[0]
        for name, count in EXPECTED_DIAGNOSTICS.items()
    )
    plotting_cells = sum(
        validate_notebook(PLOTTING / name, len(stems), "plot")[0]
        for name, stems in EXPECTED_PLOTS.items()
    )
    require(diagnostic_cells == 46, f"diagnostic code-cell total {diagnostic_cells}")
    require(plotting_cells == 24, f"plotting code-cell total {plotting_cells}")

    library = (DIAGNOSTICS / "lib" / "paper1_diagnostics.py").read_text(encoding="utf-8")
    require("ao = nam.sel(plev=1000.0)" in library, "AO is not the exact vertical-NAM slice")
    workflow = (DIAGNOSTICS / "lib" / "workflow_io.py").read_text(encoding="utf-8")
    require(
        'PRODUCT_VERSION = "Paper1_828_repro_v1"' in workflow,
        "cleaned product version is not Paper1_828_repro_v1",
    )
    require("Methods_V7_cleaned_v1" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*DIAGNOSTICS.glob("*.ipynb"), *PLOTTING.glob("*.ipynb"))
    ), "v1 product contract remains in notebooks")
    print("static: 8 shell + 5 preprocessing Python files")
    print("static: 7 diagnostic notebooks / 46 code cells")
    relationship_source = (
        DIAGNOSTICS / "lib" / "relationship_products.py"
    ).read_text(encoding="utf-8").lower()
    for token in (
        "ep100_doy21_40_mean", "ep100_doy52_71_mean",
        '"figure08h.csv"', '"figure09b.csv"',
        '"selected_marker": "skip=21,length=20; one diamond per panel"',
    ):
        require(token in relationship_source, f"current relationship contract missing: {token}")
    for token in ("ep100_doy51_72_mean", '"figure08d.csv"'):
        require(token not in relationship_source, f"superseded relationship contract remains: {token}")
    asset = ROOT / "assets" / "figure01a_o3_event_context_source.png"
    require(asset.is_file() and asset.stat().st_size > 1_000_000,
            "accepted Figure 1a carrier asset is missing or implausibly small")
    print("static: 14 plotting notebooks / 24 one-figure code cells")


def _dates(start: dt.date, end: dt.date) -> list[int]:
    output = []
    current = start
    while current <= end:
        output.append(current.year * 10000 + current.month * 100 + current.day)
        current += dt.timedelta(days=1)
    return output


def validate_scientific() -> None:
    import unittest

    for name in (
        "numpy", "pandas", "xarray", "netCDF4", "scipy", "matplotlib",
        "cartopy", "cftime", "dask", "nbclient", "nbformat",
    ):
        importlib.import_module(name)

    sys.path.insert(0, str(DIAGNOSTICS / "lib"))
    sys.path.insert(0, str(PREPROCESSING))
    import numpy as np
    import xarray as xr
    import paper1_diagnostics as kernels
    import paper1_noleap as noleap
    import relationship_products as relationships
    import test_source_contracts
    import workflow_io

    expected_free_ep = np.array(
        [1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 350],
        dtype=float,
    )
    np.testing.assert_array_equal(kernels.EP_FREE_PLEV_HPA, expected_free_ep)
    require(kernels.PLEV_HPA.size == 23, "restart/NAM pressure grid is not 23 levels")

    nam_latitude = np.array([20.0, 40.0, 60.0, 80.0, 90.0])
    nam_field = xr.DataArray(
        np.array([
            [[np.nan, np.nan, np.nan, np.nan, np.nan]],
            [[1.0, 1.0, np.nan, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0, 1.0]],
        ]),
        dims=("time", "plev", "lat"),
        coords={
            "time": np.arange(3), "plev": [1000.0], "lat": nam_latitude,
            "date": ("time", [20010101, 20010102, 20010103]),
        },
    )
    nam_reference = xr.Dataset(
        {
            "eof1_weighted": (("plev", "lat"), np.ones((1, nam_latitude.size))),
            "monthly_pc_std": ("plev", [1.0]),
            "daily_height_climatology": (
                ("month_day", "plev", "lat"),
                np.zeros((365, 1, nam_latitude.size)),
            ),
        },
        coords={"month_day": np.arange(365), "plev": [1000.0], "lat": nam_latitude},
    )
    nam_projection = kernels.project_nam(nam_field, nam_reference)
    require(np.isnan(nam_projection.nam.values[0, 0]), "all-NaN NAM day became finite")
    require(np.isnan(nam_projection.nam.values[1, 0]), "partial-latitude NAM became finite")
    require(np.isfinite(nam_projection.nam.values[2, 0]), "complete NAM day became missing")

    pressure = np.array([20.0, 30.0, 50.0, 70.0, 100.0])
    latitude = np.array([60.0, 75.0, 90.0])
    concentration = 1.0e-6
    shape = (1, pressure.size, latitude.size, 2)
    mole = xr.Dataset(
        {"O3": (("time", "lev", "lat", "lon"), np.full(shape, concentration))},
        coords={
            "time": [0], "lev": pressure, "lat": latitude, "lon": [0.0, 180.0],
            "date": ("time", [20010301]),
        },
    )
    mole.O3.attrs["units"] = "mol/mol"
    expected_du = concentration * 4000.0 * kernels.DU_PER_PA_MOLE_FRACTION
    actual_du = kernels.pressure_level_partial_ozone(
        mole, mass_mixing_ratio=False
    ).item()
    np.testing.assert_allclose(actual_du, expected_du, rtol=2.0e-12)

    mass = mole.copy(deep=True)
    mass["O3"][:] = concentration * kernels.MOLAR_MASS_O3 / kernels.MOLAR_MASS_AIR
    mass.O3.attrs["units"] = "kg/kg"
    mass_du = kernels.pressure_level_partial_ozone(mass, mass_mixing_ratio=True).item()
    np.testing.assert_allclose(mass_du, expected_du, rtol=2.0e-12)
    wrong = mole.copy(deep=True)
    wrong.O3.attrs["units"] = "ppmv"
    try:
        kernels.pressure_level_partial_ozone(wrong, mass_mixing_ratio=False)
    except ValueError:
        pass
    else:
        raise ValidationError("unsupported ozone units were accepted")

    spring_dates = _dates(dt.date(2001, 2, 27), dt.date(2001, 5, 2))
    spring = xr.DataArray(
        np.arange(len(spring_dates), dtype=float), dims="time",
        coords={"date": ("time", spring_dates)},
    )
    minimum = kernels.spring_minimum(
        spring, source_family="test", source_segment="TEST", event_year=2001,
        model_year=1,
    )
    require(minimum["minimum_date"] == "20010301", "centered spring window shifted")

    rows = [
        {
            "source_family": "WACCM", "source_segment": "TEST",
            "event_id": f"TEST:{index:04d}", "event_year": index,
            "model_year": index, "minimum_date": "20010301",
            "minimum_du": float(index), "is_target": False,
        }
        for index in range(230)
    ]
    ranking = kernels.rank_events(rows, expected_count=230)
    require(len(ranking) == 230, "230-event ranking changed size")
    require(int(ranking.is_low25.sum()) == 57, "floor(0.25*230) is not 57")
    require(float(ranking.low25_threshold_du.iloc[0]) == 56.0, "quartile threshold drift")

    np.testing.assert_array_equal(
        kernels.noleap_index([20000228, 20000229, 20000301]), [58, -1, 59]
    )
    source = xr.Dataset(
        {
            "O3": ("time", [1.0, 3.0]),
            "date": ("time", [19990101, 19990103]),
            "datesec": ("time", [0, 0]),
        },
        coords={"time": [0.0, 2.0]},
    )
    normalized, missing = noleap.normalize_dataset(source, "O3", 2001)
    jan2 = int(np.flatnonzero(normalized.date.values == 20010102)[0])
    require(20010102 in set(missing.tolist()), "missing no-leap day not recorded")
    require(np.isnan(normalized.O3.values[jan2]), "missing day was interpolated")

    climate_dates = np.r_[noleap.full_dates(2001), noleap.full_dates(2002)]
    climate = xr.DataArray(
        np.tile(np.sin(np.arange(365) * 2.0 * np.pi / 365.0), 2), dims="time",
        coords={"date": ("time", climate_dates)},
    )
    smooth = kernels.circular_noleap_climatology(climate, window=21)
    require(smooth.sizes["month_day"] == 365, "smoothed climatology is not 365 days")
    require(np.isfinite(smooth.values).all(), "circular climatology has edge NaNs")

    ep_dates = [20010130, 20010131, 20010201, 20010202]
    ep_shape = (4, 2, 3, 2)
    coordinates = {
        "time": np.arange(4), "plev": [50.0, 100.0],
        "lat": [40.0, 60.0, 80.0], "lon": [0.0, 180.0],
    }
    u = xr.DataArray(np.zeros(ep_shape), dims=("time", "plev", "lat", "lon"), coords=coordinates)
    v = u.copy(deep=True)
    temperature = u.copy(deep=True) + 250.0
    calls: list[dict] = []

    def fake_ep(**kwargs):
        calls.append(kwargs)
        result_shape = (kwargs["u"].shape[0], kwargs["u"].shape[1], kwargs["u"].shape[2])
        zero = np.zeros(result_shape)
        return zero, np.full(result_shape, 7.0), zero, zero

    ep = kernels.compute_epflux_monthly(u, v, temperature, ep_dates, fake_ep)
    require(len(calls) == 2, "EP kernel was not called once per natural month")
    for call in calls:
        require(call["w"] is None and call["do_ubar"] is True and call["wave"] == -1,
                "EP kernel arguments differ from the manuscript Methods")
    np.testing.assert_allclose(ep.fz_upward.values, -7.0)
    for name, expected in {
        "natural_month_n2": "True", "do_ubar": "True",
        "w_argument": "None", "wave": "-1",
    }.items():
        require(
            str(ep.attrs.get(name, "")) == expected,
            f"EP machine metadata {name} does not match Methods contract",
        )

    np.testing.assert_allclose(
        kernels.crps_ensemble(np.array([[0.0], [2.0]]), np.array([1.0])), [0.5]
    )
    robust = kernels.sign_agreement(np.r_[np.ones((27, 1)), -np.ones((3, 1))])
    np.testing.assert_allclose(robust, [0.9])
    zero_mean = kernels.sign_agreement(np.r_[-np.ones((15, 1)), np.ones((15, 1))])
    require(np.isnan(zero_mean[0]), "zero ensemble mean received a robust sign")

    reference_dates = np.asarray(spring_dates)
    member_mask = reference_dates >= 20010301
    detail = relationships.centered_spring_detail(
        np.full(member_mask.sum(), 10.0), reference_dates[member_mask],
        np.full(reference_dates.size, 5.0), reference_dates,
    )
    require(
        "reference_padding" in detail["boundary_padding"],
        "March centered minimum did not use Feb 27-28 reference padding",
    )

    environment_keys = ("PAPER1_DERIVED_ROOT", "PAPER1_ARCHIVE_ROOT", "PAPER1_RAW_ROOT")
    saved = {key: os.environ.get(key) for key in environment_keys}
    validation_runtime = ROOT / "runtime"
    runtime_preexisted = validation_runtime.exists()
    validation_runtime.mkdir(parents=True, exist_ok=True)
    try:
        # Keep validation writes inside the repository's only authorised STREAM
        # scope; TemporaryDirectory removes the test tree even on assertion failure.
        with tempfile.TemporaryDirectory(
            prefix=".validation_tmp_", dir=validation_runtime
        ) as temporary:
            base = Path(temporary)
            archive = base / "archive"
            staging = base / "staging"
            os.environ["PAPER1_ARCHIVE_ROOT"] = str(archive)
            os.environ["PAPER1_DERIVED_ROOT"] = str(staging)
            require(workflow_io.derived_root() == staging.resolve(), "safe staging rejected")
            workflow_io._assert_staging_target(staging / "diagnostics" / "ok.nc")
            try:
                workflow_io._assert_staging_target(base / "escape.nc")
            except PermissionError:
                pass
            else:
                raise ValidationError("output path escaped staging")
            os.environ["PAPER1_DERIVED_ROOT"] = str(archive / "BWCN" / "new")
            try:
                workflow_io.derived_root()
            except PermissionError:
                pass
            else:
                raise ValidationError("protected legacy descendant accepted as staging")
            os.environ["PAPER1_DERIVED_ROOT"] = str(base)
            try:
                workflow_io.derived_root()
            except PermissionError:
                pass
            else:
                raise ValidationError("ancestor of protected legacy trees accepted as staging")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if not runtime_preexisted:
            validation_runtime.rmdir()

    source_suite = unittest.defaultTestLoader.loadTestsFromModule(test_source_contracts)
    source_result = unittest.TextTestRunner(verbosity=0).run(source_suite)
    require(
        source_result.wasSuccessful(),
        "preprocessing source-contract regression tests failed",
    )
    print("scientific: imports and deterministic Methods/source-contract tests passed")


def validate_products() -> None:
    import matplotlib.image as mpimg
    import nbformat
    import numpy as np
    from nbclient import NotebookClient
    sys.path.insert(0, str(DIAGNOSTICS / "lib"))
    from workflow_io import derived_root

    execution_started = time.time()
    for name in EXPECTED_PLOTS:
        path = PLOTTING / name
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
    root = derived_root()
    figure_root = Path(
        os.environ.get("PAPER1_FIGURE_ROOT", str(root / "figures"))
    ).expanduser().resolve()
    require(
        figure_root == root or root in figure_root.parents,
        f"PAPER1_FIGURE_ROOT escapes PAPER1_DERIVED_ROOT: {figure_root}",
    )
    missing = []
    for stems in EXPECTED_PLOTS.values():
        for stem in stems:
            for suffix in (".png", ".pdf"):
                path = figure_root / f"{stem}{suffix}"
                if not path.is_file() or path.stat().st_size < 1024:
                    missing.append(str(path))
                    continue
                if path.stat().st_mtime < execution_started - 2.0:
                    missing.append(f"{path} (not refreshed by this validation run)")
                    continue
                if suffix == ".png":
                    try:
                        pixels = mpimg.imread(path)
                    except Exception as error:
                        missing.append(f"{path} (PNG decode failed: {error})")
                        continue
                    if pixels.ndim < 2 or min(pixels.shape[:2]) < 100:
                        missing.append(f"{path} (implausibly small raster {pixels.shape})")
                    elif not np.isfinite(pixels).any() or float(np.nanstd(pixels)) < 1.0e-4:
                        missing.append(f"{path} (blank/constant raster)")
                else:
                    payload = path.read_bytes()
                    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
                        missing.append(f"{path} (invalid PDF signature/trailer)")
    require(not missing, f"missing/blank rendered outputs: {missing}")
    print("products: 14 clean-kernel plotting notebooks and 48 nonblank files passed")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--require-scientific", action="store_true",
        help="import the scientific stack and run deterministic kernel tests",
    )
    command.add_argument(
        "--require-products", action="store_true",
        help="execute plotting notebooks against an already completed staging tree",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    validate_static()
    if args.require_scientific or args.require_products:
        validate_scientific()
    if args.require_products:
        validate_products()
    print("Paper1 validation passed")


if __name__ == "__main__":
    main()
