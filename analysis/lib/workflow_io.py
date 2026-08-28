"""Safe I/O and schema gates for the Paper 1 diagnostics.

Every product is written below ``PAPER1_DERIVED_ROOT``.  A candidate NetCDF
or CSV is first written to a sibling temporary file, reopened, and validated;
only then is it atomically moved into place.  The legacy public-data tree is
therefore input-only.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


PRODUCT_VERSION = "Paper1_828_repro_v1"
PAPER1_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PAPER1_REPOSITORY_ROOT / "data"
DEFAULT_DERIVED_ROOT = PAPER1_REPOSITORY_ROOT / "work"


def _is_within(candidate: Path, parent: Path) -> bool:
    """Return whether *candidate* is *parent* or one of its descendants."""

    return candidate == parent or parent in candidate.parents


def archive_root() -> Path:
    """Return the read-only raw-data root."""

    return Path(
        os.environ.get("PAPER1_ARCHIVE_ROOT", os.environ.get("PAPER1_RAW_ROOT", str(DEFAULT_RAW_ROOT)))
    ).expanduser().resolve()


def derived_root() -> Path:
    """Return the working directory for generated products."""

    root = Path(
        os.environ.get("PAPER1_DERIVED_ROOT", str(DEFAULT_DERIVED_ROOT))
    ).expanduser().resolve()
    archive = archive_root()
    if root == Path(root.anchor):
        raise PermissionError("PAPER1_DERIVED_ROOT cannot be a filesystem root")
    if _is_within(root, archive) or _is_within(archive, root):
        raise PermissionError("PAPER1_DERIVED_ROOT must not overlap the raw-data root")
    return root


def preprocessed_root() -> Path:
    """Return the root containing products from preprocessing."""

    return Path(
        os.environ.get("PAPER1_PREPROCESSED_ROOT", str(derived_root()))
    ).expanduser().resolve()


def raw_root() -> Path:
    """Backward-compatible alias for :func:`archive_root`."""

    return archive_root()


def product_path(*parts: str) -> Path:
    return derived_root().joinpath(*parts)


def _assert_staging_target(path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    root = derived_root()
    if target != root and root not in target.parents:
        raise PermissionError(f"Refusing to write outside PAPER1_DERIVED_ROOT: {target}")
    return target


def validate_dataset(
    dataset: xr.Dataset,
    *,
    required_vars: Mapping[str, Sequence[str]],
    required_coords: Sequence[str] = (),
    exact_sizes: Mapping[str, int] | None = None,
    required_attrs: Mapping[str, object] | None = None,
) -> None:
    """Validate variable names/dimensions plus selected sizes and attributes."""

    for name, dimensions in required_vars.items():
        if name not in dataset:
            raise ValueError(f"Missing required variable {name!r}")
        if tuple(dataset[name].dims) != tuple(dimensions):
            raise ValueError(
                f"{name!r} dimensions {dataset[name].dims!r} != {tuple(dimensions)!r}"
            )
    for name in required_coords:
        if name not in dataset.coords and name not in dataset:
            raise ValueError(f"Missing required coordinate {name!r}")
    for name, size in (exact_sizes or {}).items():
        if int(dataset.sizes.get(name, -1)) != int(size):
            raise ValueError(f"Dimension {name!r} has {dataset.sizes.get(name)}; expected {size}")
    for name, expected in (required_attrs or {}).items():
        if dataset.attrs.get(name) != expected:
            raise ValueError(
                f"Attribute {name!r} is {dataset.attrs.get(name)!r}; expected {expected!r}"
            )
    for name in required_vars:
        if dataset[name].size == 0:
            raise ValueError(f"Required variable {name!r} is empty")


def _temporary_sibling(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def write_netcdf_atomic(
    dataset: xr.Dataset,
    target: Path,
    *,
    required_vars: Mapping[str, Sequence[str]],
    required_coords: Sequence[str] = (),
    exact_sizes: Mapping[str, int] | None = None,
    required_attrs: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> Path:
    """Schema-check a temporary NetCDF and atomically install it in staging."""

    target = _assert_staging_target(target)
    attrs = dict(dataset.attrs)
    attrs.setdefault("product_version", PRODUCT_VERSION)
    candidate = dataset.assign_attrs(attrs)
    required_attrs = {"product_version": PRODUCT_VERSION, **(required_attrs or {})}

    if target.exists() and target.stat().st_size and not overwrite:
        with xr.open_dataset(target, decode_times=False) as existing:
            validate_dataset(
                existing, required_vars=required_vars,
                required_coords=required_coords, exact_sizes=exact_sizes,
                required_attrs=required_attrs,
            )
        print(f"retained validated product: {target}")
        return target

    temporary = _temporary_sibling(target)
    encoding = {
        name: {"zlib": True, "complevel": 2}
        for name in candidate.data_vars
        if np.issubdtype(candidate[name].dtype, np.number)
    }
    try:
        candidate.to_netcdf(temporary, encoding=encoding)
        with xr.open_dataset(temporary, decode_times=False) as check:
            validate_dataset(
                check, required_vars=required_vars,
                required_coords=required_coords, exact_sizes=exact_sizes,
                required_attrs=required_attrs,
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"saved validated product: {target}")
    return target


def write_csv_atomic(
    frame: pd.DataFrame,
    target: Path,
    *,
    required_columns: Sequence[str],
    exact_rows: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Version/schema-check a temporary CSV and atomically install it.

    Every cleaned table carries the same product-version contract as NetCDF
    outputs.  This prevents a schema-compatible table from an earlier method
    revision from being silently retained after a scientific-code change.
    """

    target = _assert_staging_target(target)

    candidate_frame = frame.copy()
    if "product_version" in candidate_frame:
        versions = set(candidate_frame["product_version"].dropna().astype(str))
        if versions != {PRODUCT_VERSION}:
            raise ValueError(
                f"CSV product_version values {sorted(versions)!r}; "
                f"expected only {PRODUCT_VERSION!r}"
            )
    else:
        candidate_frame["product_version"] = PRODUCT_VERSION

    def check(candidate: pd.DataFrame) -> None:
        missing = [name for name in required_columns if name not in candidate.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        if "product_version" not in candidate.columns:
            raise ValueError("CSV is missing product_version")
        versions = set(candidate["product_version"].dropna().astype(str))
        if versions != {PRODUCT_VERSION}:
            raise ValueError(
                f"CSV product_version values {sorted(versions)!r}; "
                f"expected only {PRODUCT_VERSION!r}"
            )
        if exact_rows is not None and len(candidate) != exact_rows:
            raise ValueError(f"CSV has {len(candidate)} rows; expected {exact_rows}")

    check(candidate_frame)
    if target.exists() and target.stat().st_size and not overwrite:
        check(pd.read_csv(target))
        print(f"retained validated table: {target}")
        return target

    temporary = _temporary_sibling(target)
    try:
        candidate_frame.to_csv(temporary, index=False)
        check(pd.read_csv(temporary))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"saved validated table: {target}")
    return target
