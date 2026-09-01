# Input data

The scientific data are not committed to Git. Set the paths in `config.sh` before running the workflow.

| Variable | Required input |
|---|---|
| `PAPER1_WACCM_LONGRUN_001` | first WACCM4 free-running history directory |
| `PAPER1_WACCM_LONGRUN_002` | second WACCM4 free-running history directory |
| `PAPER1_WACCM_RESTART_SOURCE` | 23-year restart-source history directory |
| `PAPER1_HINDCAST_SOURCE` | January and February raw restart ensembles |
| `PAPER1_MARCH_HINDCAST_SOURCE` | pressure-level February/March restart files |
| `PAPER1_MERRA2_SOURCE` | MERRA-2 M2I6NPANA U/V/T/O3 files |
| `PAPER1_MERRA2_Z_SOURCE` | MERRA-2 M2I6NPANA geopotential-height files |
| `PAPER1_MLS_SOURCE` | MLS V5 Level-3 zonal-mean ClO/H2O files |
| `PAPER1_EXTR_CLOX_SOURCE` | combined 200-year WACCM ClOx file |
| `PAPER1_RAW_MODEL_ROOT` | read-only raw-model tree used by the lightweight check |
| `PAPER1_REFERENCE_ROOT` | read-only established products used by the lightweight check |

`PAPER1_ARCHIVE_ROOT` is only a convenient common parent. Every dataset path can be set independently. `PAPER1_DERIVED_ROOT` is the working directory for staged and calculated products; the default from `config.example.sh` is `work/` inside the checkout.

The preprocessing scripts create the following main product groups:

```text
work/
  B2000WCN001002_timefixed/   long free-running integration
  BWCN/                       23-year restart-source integration
  Hindcast/                   January and February restart members
  MERRA2_Processed/           daily MERRA-2 fields
  manifests/                  source inventories and coverage checks
  ozone/ nam/ epflux/         calculated diagnostics
  chemistry/ verification/    chemistry and hindcast metrics
  figures/                    PNG/PDF outputs
```

Source files are opened read-only. Missing WACCM calendar days are retained as missing values; they are not temporally interpolated.

`work/ozone/event_profile_bootstrap5000.nc` stores the daily, unsmoothed Appendix B event-profile anomaly, target-excluded daily climatology, 95% bootstrap bounds, and significance mask. Its fixed baselines are MERRA-2 1980--2019 (N=40) and the 22 complete BWCN years other than year 0008. The product is required to carry the exact baseline IDs/counts, `profile_anomaly_temporal_resolution=daily`, and `temporal_smoothing_days=0`.

`work/relationships/figure09d.csv` stores the 30 February-initialized members, the no-W EP100 mean for forecast days 21--40 (21 February--12 March), the centered-five-day spring ozone minimum, Pearson/OLS statistics, and the year-0008 reference point used only for display.
