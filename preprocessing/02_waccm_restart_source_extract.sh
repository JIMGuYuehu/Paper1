#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Daily WACCM4 h3 files from the BWCN restart-source integration:
#   ${PAPER1_ARCHIVE_ROOT}/WACCM/restart_source
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/BWCN/<VARIABLE>/BWCN.cam.h3.YYYY.<VARIABLE>.nc
#   ${PAPER1_DERIVED_ROOT}/manifests/waccm_bwcn_*.tsv
# ACTION
#   Require 24 raw model-year labels, verify the real h3 schema (including
#   CLOX/H2O), concatenate each year, normalize it to a 365-day no-leap axis
#   with NaN gap placeholders, retain date/datesec, and require 23 complete
#   27 February-2 May padded ozone-event windows for the restart distribution.
# REQUIREMENTS
#   Bash, Python/xarray/netCDF4, NCO, and write access to the staging root.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_commands find ncks ncrcat python stat

SRC_DIR="${PAPER1_WACCM_RESTART_SOURCE:-${PAPER1_ARCHIVE_ROOT}/WACCM/restart_source}"
OUT_ROOT="${PAPER1_DERIVED_ROOT}/BWCN"
MANIFEST_ROOT="${PAPER1_DERIVED_ROOT}/manifests"
MAX_JOBS="${PAPER1_MAX_JOBS:-6}"
CORE_VARS=(U V T Z3 O3 PS CLOX H2O)
COORD_VARS="P0,hyai,hyam,hybi,hybm,date,datesec,time,time_bnds,lat,lon,lev,ilev,gw"

[[ -d "${SRC_DIR}" ]] || die "missing BWCN source: ${SRC_DIR}"
prepare_output_dir "${OUT_ROOT}"
prepare_output_dir "${MANIFEST_ROOT}"

task_file=""
cleanup_top_level() {
    local candidate
    for candidate in "${schema_tmp:-}" "${raw_years_tmp:-}" \
        "${raw_files_tmp:-}" "${task_file:-}"; do
        [[ -z "${candidate}" ]] || rm -f -- "${candidate}"
    done
}
trap cleanup_top_level EXIT

mapfile -t YEARS < <(
    find "${SRC_DIR}" -maxdepth 1 -type f \
        -name 'BWCN.e122.f19_g16.002.cam.h3.[0-9][0-9][0-9][0-9]-*.nc' \
        -printf '%f\n' \
        | sed -E 's/.*\.cam\.h3\.([0-9]{4})-.*/\1/' \
        | sort -u
)
[[ ${#YEARS[@]} -eq 24 ]] || die "expected 24 BWCN raw year labels; found ${#YEARS[@]}"

probe_file=$(find "${SRC_DIR}" -maxdepth 1 -type f \
    -name "BWCN.e122.f19_g16.002.cam.h3.${YEARS[0]}-*.nc" -print -quit)
[[ -n "${probe_file}" ]] || die "cannot select a BWCN schema probe"
schema_tmp="${MANIFEST_ROOT}/waccm_bwcn_variable_availability.tsv.tmp.$$"
schema_manifest="${MANIFEST_ROOT}/waccm_bwcn_variable_availability.tsv"
prepare_output_file "${schema_tmp}"
prepare_output_file "${schema_manifest}"
printf 'variable\tavailable\tprobe_file\n' > "${schema_tmp}"
for variable in "${CORE_VARS[@]}"; do
    if ncks -m -v "${variable}" "${probe_file}" >/dev/null 2>&1; then
        printf '%s\t1\t%s\n' "${variable}" "${probe_file}" >> "${schema_tmp}"
    else
        printf '%s\t0\t%s\n' "${variable}" "${probe_file}" >> "${schema_tmp}"
        rm -f -- "${schema_tmp}"
        die "required BWCN variable is absent: ${variable}"
    fi
done
mv -- "${schema_tmp}" "${schema_manifest}"

raw_years_tmp="${MANIFEST_ROOT}/waccm_bwcn_raw_years.tsv.tmp.$$"
raw_years_manifest="${MANIFEST_ROOT}/waccm_bwcn_raw_years.tsv"
prepare_output_file "${raw_years_tmp}"
prepare_output_file "${raw_years_manifest}"
printf 'source_year\n' > "${raw_years_tmp}"
printf '%s\n' "${YEARS[@]}" >> "${raw_years_tmp}"
mv -- "${raw_years_tmp}" "${raw_years_manifest}"

raw_files_tmp="${MANIFEST_ROOT}/waccm_bwcn_raw_files.tsv.tmp.$$"
raw_files_manifest="${MANIFEST_ROOT}/waccm_bwcn_raw_files.tsv"
event_manifest="${MANIFEST_ROOT}/waccm_bwcn_complete_events.tsv"
prepare_output_file "${raw_files_tmp}"
prepare_output_file "${raw_files_manifest}"
prepare_output_file "${event_manifest}"
printf 'source_year\ttarget_year\tpath\tsize_bytes\tmtime_epoch\n' > "${raw_files_tmp}"
while IFS= read -r path; do
    source_year=$(basename "${path}" | sed -E 's/.*\.cam\.h3\.([0-9]{4})-.*/\1/')
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${source_year}" "${source_year}" "${path}" "$(stat -c '%s' "${path}")" \
        "$(stat -c '%Y' "${path}")" >> "${raw_files_tmp}"
done < <(find "${SRC_DIR}" -maxdepth 1 -type f \
    -name 'BWCN.e122.f19_g16.002.cam.h3.[0-9][0-9][0-9][0-9]-*.nc' | sort)
mv -- "${raw_files_tmp}" "${raw_files_manifest}"

waccm_schema_manifest="${MANIFEST_ROOT}/waccm_bwcn_source_schema.tsv"
prepare_output_file "${waccm_schema_manifest}"
waccm_schema_args=(
    waccm-schema
    --inventory "${raw_files_manifest}"
    --fixed-case BWCN
    --required-vars "$(IFS=,; echo "${CORE_VARS[*]}")"
    --output "${waccm_schema_manifest}"
)
[[ "${PAPER1_AUDIT_ONLY}" == 1 ]] && waccm_schema_args+=(--all-files)
python "${SCRIPT_DIR}/source_inventory.py" "${waccm_schema_args[@]}"

python "${SCRIPT_DIR}/noleap.py" source-event-manifest \
    --inventory "${raw_files_manifest}" \
    --expected-years 24 \
    --expected-complete 23 \
    --output "${event_manifest}"

if [[ "${PAPER1_AUDIT_ONLY}" == 1 ]]; then
    printf 'audit-only: validated every BWCN chunk native grid/schema, 24 labels, and 23 padded event windows; no concatenation run\n'
    exit 0
fi

process_year_variable() (
    local source_year="$1" variable="$2"
    local out_dir="${OUT_ROOT}/${variable}"
    local target="${out_dir}/BWCN.cam.h3.${source_year}.${variable}.nc"
    local selected_vars
    prepare_output_file "${target}"
    if [[ -e "${target}" ]]; then
        [[ -s "${target}" ]] || { echo "empty target: ${target}" >&2; return 1; }
        python "${NOLEAP_HELPER}" normalize-year \
            --input "${target}" --output "${target}" --variable "${variable}" \
            --year "$((10#${source_year}))" --source-year "$((10#${source_year}))" \
            --source-run BWCN.002
        return 0
    fi

    shopt -s nullglob
    local inputs=("${SRC_DIR}/BWCN.e122.f19_g16.002.cam.h3.${source_year}-"*.nc)
    shopt -u nullglob
    ((${#inputs[@]} > 0)) || { echo "no BWCN inputs for ${source_year}" >&2; return 1; }
    local raw_tmp="${target}.ncrcat.tmp.$$"
    assert_under_staging "${raw_tmp}"
    trap 'rm -f -- "${raw_tmp}"' EXIT
    rm -f -- "${raw_tmp}"
    selected_vars="${variable},PS,${COORD_VARS}"
    [[ "${variable}" == PS ]] && selected_vars="PS,${COORD_VARS}"
    if ! ncrcat -O -v "${selected_vars}" "${inputs[@]}" "${raw_tmp}"; then
        rm -f -- "${raw_tmp}"
        return 1
    fi
    if ! python "${NOLEAP_HELPER}" normalize-year \
        --input "${raw_tmp}" --output "${target}" --variable "${variable}" \
        --year "$((10#${source_year}))" --source-year "$((10#${source_year}))" \
        --source-run BWCN.002; then
        rm -f -- "${raw_tmp}"
        return 1
    fi
    rm -f -- "${raw_tmp}"
)

export -f process_year_variable
export SRC_DIR OUT_ROOT COORD_VARS
export NOLEAP_HELPER="${SCRIPT_DIR}/noleap.py"

task_file=$(mktemp "${MANIFEST_ROOT}/.waccm_bwcn_tasks.XXXXXX")
assert_under_staging "${task_file}"
for year in "${YEARS[@]}"; do
    for variable in "${CORE_VARS[@]}"; do
        printf 'process_year_variable %q %q\n' "${year}" "${variable}" >> "${task_file}"
    done
done
xargs -P "${MAX_JOBS}" -I TASK bash -c 'TASK' < "${task_file}"

python "${SCRIPT_DIR}/noleap.py" event-manifest \
    --root "${OUT_ROOT}" \
    --variables "$(IFS=,; echo "${CORE_VARS[*]}")" \
    --expected-years 24 \
    --expected-complete 23 \
    --output "${event_manifest}"

printf 'validated BWCN: 24 raw labels; 23 complete Feb27-May2 event windows\n'
