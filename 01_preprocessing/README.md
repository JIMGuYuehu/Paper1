# Preprocessing

The numbered scripts stage read-only raw archives under `runtime/`:

- `01`: WACCM LONGRUN (207 complete springs)
- `02`: WACCM BWCN restart source (23 complete springs)
- `03`: January/February 30-member restart ensembles
- `04`: Marina February/March ensemble inventory
- `05`: MERRA-2 daily pressure-level data
- `06`: MLS Level-3 ClO/H2O inventory
- `07`: WACCM pressure-level U/V/T/Z3 products
- `08`: 200-year ClOx extraction and audited missing-date handling
- `09`: bounded real-data reproduction check

Every producer checks input schema/calendar and refuses to write outside this
repository's `runtime/` directory. Use `PAPER1_AUDIT_ONLY=1` for inventory-only
checks. On STREAM, use system NCO and
`PAPER1_CDO_BIN=/home/weiji/miniconda3/envs/cdo_tools/bin/cdo`.
