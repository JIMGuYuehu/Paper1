# Paper 1 code

Code for the 24 figures used by the current manuscript snapshot (2026-08-28).
Raw WACCM, MERRA-2 and MLS archives are read-only; generated data and figures
are written under `runtime/`, which is ignored by Git.

## Run

On STREAM:

```bash
conda activate jimnew
export PAPER1_ARCHIVE_ROOT=/mnt/soclim0/public_data/weiji
export PAPER1_DERIVED_ROOT="$(pwd)/runtime"
export PAPER1_PREPROCESSED_ROOT="$PAPER1_DERIVED_ROOT"
```

1. Run `01_preprocessing/01_*.sh` through `07_*.sh`, then the required Python
   scripts in the same directory. These stage WACCM LONGRUN/BWCN/restarts,
   MERRA-2 and MLS without changing the source archives.
2. Run `02_diagnostics/01_*.ipynb` through `07_*.ipynb`. These calculate
   partial ozone, NAM/AO, EP flux, chemistry anomalies, hindcast scores,
   correlations and bootstrap products.
3. Run the notebooks in `03_plotting/`. They read the processed products and
   write one PNG/PDF pair per plotting block to `runtime/figures/`.

Static validation:

```bash
python -B validate_tree.py
```

Optional checks on STREAM:

```bash
python -B validate_tree.py --require-scientific
python -B validate_tree.py --require-products
python 01_preprocessing/09_lightweight_reproduction_check.py
```

See `FIGURE_MAP.md` for figure-to-notebook mapping,
`METHODS_NOTES.md` for the calculation definitions and known version issues,
and `MANUSCRIPT_FIGURE_MANIFEST.csv` for exact manuscript-file hashes.
