# Figures

Each notebook corresponds to one manuscript figure group. Every output has an explanatory Markdown cell followed by one plotting cell. Plotting reads the calculated products from `PAPER1_DERIVED_ROOT` and writes PNG/PDF files to `PAPER1_FIGURE_ROOT`.

`paper_style.py` contains the shared final-manuscript canvas sizes and publication-scale typography. Every plotting notebook calls it immediately before saving so titles, labels, ticks, legends, and colour bars remain readable after placement in the paper.

Observations/reanalysis are normally placed left or above WACCM; Appendix B follows the accepted WACCM-left/MERRA-2-right panel order. Correlation panels report Pearson statistics only. Figure 6 spread, Figure 16b CRPS, and Appendix B ozone-profile anomalies use unsmoothed daily values.
