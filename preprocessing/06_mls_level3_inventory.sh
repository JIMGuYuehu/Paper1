#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Aura MLS V5 Level-3 annual files under:
#   ${PAPER1_ARCHIVE_ROOT}/MLS/Level3_Zonal_v5/ClO
#   ${PAPER1_ARCHIVE_ROOT}/MLS/Level3_Zonal_v5/H2O
#   with names MLS-Aura_L3DZ-{ClO,H2O}_v05-*_YYYY.nc.
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/manifests/mls_level3_v5_2004_2020.tsv
# ACTION
#   Require exactly one annual file per product/year for 2004-2020 and verify
#   the exact NetCDF groups ``ClO PressureZM Day`` and ``H2O PressureZM`` plus
#   value/nvalues/time/lev/lat, chemical value units, V5 metadata, and decoded
#   unique calendar dates belonging to the filename year. Repeated decoded
#   dates are rejected and unique-day coverage is recorded. This prevents use
#   of night-time ClO, a wrong MLS version, or a filename-only false match.
#   Sources are unchanged.
# REQUIREMENTS
#   Bash and Python/netCDF4.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_commands python

MLS_ROOT="${PAPER1_MLS_SOURCE:-${PAPER1_ARCHIVE_ROOT}/MLS/Level3_Zonal_v5}"
INVENTORY="${PAPER1_DERIVED_ROOT}/manifests/mls_level3_v5_2004_2020.tsv"
prepare_output_file "${INVENTORY}"

python "${SCRIPT_DIR}/source_inventory.py" mls \
    --root "${MLS_ROOT}" \
    --start-year 2004 \
    --end-year 2020 \
    --output "${INVENTORY}"
