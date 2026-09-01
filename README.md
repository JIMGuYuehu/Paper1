# Predictability of extreme Arctic spring ozone depletion

Analysis and plotting code for the manuscript *Predictability of extreme Arctic spring ozone depletion*.

This repository contains the workflow used for the 24 figure files currently included in this release: raw-data preparation, calculation of the diagnostics described in Methods, and figure production. Raw WACCM output, MERRA-2, and MLS files are not stored in Git because of their size and distribution terms.

## Repository structure

```text
preprocessing/   prepare WACCM, MERRA-2, and MLS inputs
analysis/        calculate ozone, NAM/AO, EP flux, chemistry, and hindcast metrics
figures/         one plotting notebook for each manuscript figure group
data/README.md   input-data products and configuration
```

Generated intermediate files are written to `work/`; figures are written to `work/figures/`. Both directories are ignored by Git.

## Data

- **WACCM4:** two free-running integrations (207 and 23 complete springs), the selected year-0008 reference event, and January/February/March 30-member restart ensembles. These model outputs are not redistributed in this repository.
- **MERRA-2:** M2I6NPANA pressure-level analyses for 1980--2025, DOI [10.5067/A7S6XP56VZWS](https://doi.org/10.5067/A7S6XP56VZWS).
- **Aura MLS V5:** daytime ClO and H2O Level-3 zonal-mean products, DOIs [10.5067/AURA/MLS/DATA/3565](https://doi.org/10.5067/AURA/MLS/DATA/3565) and [10.5067/AURA/MLS/DATA/3568](https://doi.org/10.5067/AURA/MLS/DATA/3568).

The expected inputs and configuration variables are listed in [`data/README.md`](data/README.md).

## Environment

```bash
conda env create -f environment.yml
conda activate paper1-ozone
cp config.example.sh config.sh
```

Edit `config.sh` to point to the input data, then load it with `source config.sh`. The example creates all derived products inside this checkout; it contains no institution-specific paths.

## Workflow

1. Run the numbered scripts in `preprocessing/`. These scripts use Bash, CDO, NCO, and Python and never modify the source files.
2. Run the numbered notebooks in `analysis/` to calculate the scientific diagnostics and figure-ready products.
3. Run the notebooks in `figures/` to create the manuscript PNG/PDF files.

The preprocessing stage is large. If archived figure-ready products are supplied with the final paper, place them at `PAPER1_DERIVED_ROOT` and start directly from the plotting notebooks.

## Figure-to-code map

| Manuscript item | Notebook | Main input/diagnostic |
|---|---|---|
| Figure 1a; Appendix B ozone profiles | `figures/figure01_o3_event_context.ipynb` | MERRA-2/WACCM partial ozone and anomaly profiles |
| Figure 2c; Appendix A1--A2 | `figures/figure02_precursor_relationships.ipynb` | NAM, U60N10, EP100, and ozone rankings |
| Figure 3 H2O and ClO | `figures/figure03_chemistry_dehydration.ipynb` | MLS/WACCM H2O and ClO/ClOx anomalies; 195 K PSC-I reference |
| Figure 4 | `figures/figure04_nam_o3_context.ipynb` | MERRA-2 2020 and WACCM year-0008 NAM/ozone evolution |
| Figure 5a; Figure 5b | `figures/figure05_hindcast_evolution.ipynb` | restart O3, U60N10, and 50-hPa polar-cap minimum-temperature evolution |
| Figure 6b | `figures/figure06_hindcast_spread_timing.ipynb` | daily population spread of O3, EP100, U60N10, and 50-hPa polar-cap minimum temperature |
| Figure 7a; Figure 7b; Figure 7c | `figures/figure07_rmse_pathway.ipynb` | January-member RMSE relationships |
| Figure 8b; Figure 8h; Appendix C Figure 8a | `figures/figure08_january_wave_precursor.ipynb` | January EP100 predictors, ozone minimum/RMSE, and minimum dates |
| Figure 9d | `figures/figure09d_february_member_ep100.ipynb` | February days-21--40 EP100 versus centered-five-day spring ozone minimum |
| Appendix C Figure 9a | `figures/figure09a_february_o3_minimum_date.ipynb` | February-hindcast ozone-minimum dates |
| Appendix C Figure 11b | `figures/figure11_wave_window_scan.ipynb` | Pearson window scan and days-21--40 markers |
| Figure 15f | `figures/figure15_low_o3_z300_epflux.ipynb` | Z300 and upward EP-flux composites with 5,000 bootstraps |
| Appendix C Figure 16b | `figures/figure16_daily_spread_crps.ipynb` | daily unsmoothed CRPS |
| Appendix C Figure 17a | `figures/figure17_nam_evolution.ipynb` | vertical NAM and 90% sign agreement |
| Appendix C Figure 18a | `figures/figure18_ao_evolution.ipynb` | AO, defined as the 1000-hPa NAM level |

## Core definitions

- Partial ozone column: 30--70 hPa, followed by a cosine-weighted 60--90 N mean; spring minima use a centred five-day mean over 1 March--30 April.
- Appendix B event profiles: daily 60--90 N ozone anomalies relative to a target-excluded daily climatology (MERRA-2 1980--2019, N=40; the 22 complete non-target BWCN years); no temporal smoothing is applied to the climatology, anomaly, or bootstrap significance mask.
- NAM/AO: fixed leading EOFs of monthly zonal-mean geopotential-height anomalies north of 20 N; AO is the 1000-hPa NAM level.
- EP flux: Jucker pressure-coordinate formulation with all resolved waves, zonal-wind correction, no pressure-velocity covariance, monthly static stability, and a 40--80 N cosine mean of the upward component.
- Hindcast relationships: population spread (`ddof=0`), finite-member CRPS, memberwise RMSE, and two-sided Pearson correlation. No Spearman correlation is used.

## Citation and license

Use `CITATION.cff` to cite this repository. The vendored EP-flux implementation is derived from Martin Jucker's GPLv3 `aostools`; the repository is therefore prepared for a GPL-3.0-compatible release. Confirm the final license and archived release DOI before publication.
