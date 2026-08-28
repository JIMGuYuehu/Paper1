#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Marina Friedel's pressure-level restart members under:
#   /mnt/backup_ETH/Marina/WACCM/CHEM_2000_restart/
#       BWCN.e122.f19_g16.002_0008/Feb
#       BWCN.e122.f19_g16.002_0008/Mar
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/manifests/marina_feb_mar_pressure_members.tsv
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
# shellcheck source=00_common.sh
source "${SCRIPT_DIR}/00_common.sh"
require_commands python

SOURCE_ROOT="${PAPER1_MARINA_SOURCE:-/mnt/backup_ETH/Marina/WACCM/CHEM_2000_restart/BWCN.e122.f19_g16.002_0008}"
INVENTORY="${PAPER1_DERIVED_ROOT}/manifests/marina_feb_mar_pressure_members.tsv"
prepare_output_file "${INVENTORY}"

python "${SCRIPT_DIR}/paper1_source_audit.py" marina \
    --source-root "${SOURCE_ROOT}" \
    --output "${INVENTORY}"
