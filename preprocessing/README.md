# Preprocessing

Run the numbered scripts in order after loading `config.sh`.

1. `01_waccm_longrun_extract.sh`: 207 complete springs from the two long WACCM integrations.
2. `02_waccm_restart_source_extract.sh`: 23 complete springs from the restart-source integration.
3. `03_waccm_restart_ensembles_extract.sh`: January and February 30-member ensembles.
4. `04_waccm_march_inventory.sh`: February/March pressure-level restart inventory.
5. `05_merra2_daily_pressure_levels.sh`: daily MERRA-2 U, V, T, O3, and Z3.
6. `06_mls_level3_inventory.sh`: MLS V5 daytime ClO and H2O inventory.
7. `07_waccm_pressure_level_products.sh`: WACCM hybrid-to-pressure interpolation.
8. `08_extr_clox_200yr.py`: 200-year WACCM ClOx climatology.
9. `09_lightweight_reproduction_check.py`: representative raw-to-product comparison.

Each script states its inputs, outputs, scientific action, and required software at the top. Source files are never modified.
