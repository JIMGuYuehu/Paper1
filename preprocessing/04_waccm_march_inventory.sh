#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Pressure-level February and March restart members under:
#   PAPER1_MARCH_HINDCAST_SOURCE/{Feb,Mar}
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/manifests/waccm_feb_mar_pressure_members.tsv
# ACTION
#   Verify 30 February and 30 March files with the same member IDs; require U,
#   V, T, Z3, and O3; require the exact ordered canonical 23-level grid; and
#   exact decoded daily noleap coverage (Feb: 121 days through Jun 1; Mar: 123
#   days through Jul 1), including 31 May in both cases. Source files stay
#   read-only; only an atomic inventory is
#   written.
# REQUIREMENTS
#   Bash and Python/netCDF4.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_commands python

SOURCE_ROOT="${PAPER1_MARCH_HINDCAST_SOURCE:-${PAPER1_ARCHIVE_ROOT}/WACCM/march_hindcast}"
INVENTORY="${PAPER1_DERIVED_ROOT}/manifests/waccm_feb_mar_pressure_members.tsv"
prepare_output_file "${INVENTORY}"

python "${SCRIPT_DIR}/source_inventory.py" march \
    --source-root "${SOURCE_ROOT}" \
    --output "${INVENTORY}"
