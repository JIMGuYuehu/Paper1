# Plotting

The 14 notebooks render the 24 outputs listed in `../FIGURE_MAP.md`. Plotting
cells use diagnostic products and do not recompute NAM/AO, EP flux, ozone
ranking, Pearson statistics, RMSE, CRPS or bootstrap tests. Each logical figure
has one explanation block followed by one plotting block and writes a same-stem
PNG/PDF pair under `runtime/figures/`.

Reanalysis/observations are placed left or above WACCM. Relationships are
Pearson-only. Figure 6 spread and Figure 16b CRPS are daily and unsmoothed.
