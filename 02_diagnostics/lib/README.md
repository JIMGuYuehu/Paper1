# Shared current-manuscript diagnostic kernels

- `workflow_io.py`: separates immutable `PAPER1_ARCHIVE_ROOT` (MLS) from
  `PAPER1_PREPROCESSED_ROOT` (default: the same staging root populated by 01)
  and `PAPER1_DERIVED_ROOT` (default: repository-local `Paper1/runtime`); both
  staging roots must resolve to this checkout's `code/runtime` itself or one of
  its descendants. It prevents writes into source/raw/legacy or
  symlink-redirected paths, reopens and validates temporary NetCDF/CSV
  products, then atomically replaces targets.
- `paper1_diagnostics.py`: date/calendar handling, exact partial-column ozone,
  fixed-EOF NAM/AO, natural-month EP flux, CRPS, sign agreement, and both V7
  bootstrap definitions. It keeps the 23-level NAM grid, the 18-level
  MERRA-2/free-running EP grid, and the 23-level restart EP grid explicit.
- `figure15_combined.py`: joins canonical rankings to November–March Z300
  and EP events. WACCM threshold/ranks/flags come from the combined 207+23
  classification master, while fields are explicitly scoped to LONGRUN: BWCN
  exclusions and missing LONGRUN fields are recorded separately. It preserves
  event provenance, applies population `ddof=0` standardization, and performs
  5,000 same-size random composites.
- `relationship_products.py`: precomputes every Figure 2/4/8/9/11 minimum,
  window mean, fixed membership, histogram count, Pearson p value, and OLS fit
  so plotting notebooks perform no scientific calculation. Figure 2 uses only
  the 207 ranked LONGRUN events for WACCM moments while inheriting their flags
  from the combined 230-event classification master.
- `epflux_jucker.py`: minimal GPLv3 Methods-only subset derived from Martin
  Jucker's `aostools.climate`. It rejects any call other than
  `do_ubar=True`, `w=None`, `wave=-1`; every EP output records its source URL,
  licence, and the runtime SHA256 of this exact file.

No module in this directory writes to a hard-coded legacy public-data path.
