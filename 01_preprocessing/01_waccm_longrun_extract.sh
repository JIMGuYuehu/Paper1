#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   WACCM4 h3 ``*.extr.nc`` files from the two free-running integrations:
#   /mnt/backup_ETH/extr_2000/extr_2000/B2000WCN.e122.f19_g16.001/atm/hist
#   /mnt/backup_ETH/extr_2000/extr_2000/B2000WCN.e122.f19_g16.002/atm/hist
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/B2000WCN001002_timefixed/<VARIABLE>/*.nc
#   ${PAPER1_DERIVED_ROOT}/manifests/waccm_longrun_*.tsv
# ACTION
#   Verify 104 run-001 and 106 run-002 source years, probe rather than assume
#   the source schema, concatenate each available variable, offset run-002 by
#   104 years, normalize to 365-day no-leap files while retaining date/datesec,
#   and require exactly 207 complete 27 February-2 May windows. This padding is
#   required for centered 5-day means reported from 1 March through 30 April.
#   Missing source days are represented by NaN; values are never interpolated.
# REQUIREMENTS
#   Bash, Python/xarray/netCDF4, NCO, and write access to the staging root.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=00_common.sh
source "${SCRIPT_DIR}/00_common.sh"
require_commands find grep ncks ncrcat python stat

SRC_RUN001="${PAPER1_WACCM_LONGRUN_001:-/mnt/backup_ETH/extr_2000/extr_2000/B2000WCN.e122.f19_g16.001/atm/hist}"
SRC_RUN002="${PAPER1_WACCM_LONGRUN_002:-/mnt/backup_ETH/extr_2000/extr_2000/B2000WCN.e122.f19_g16.002/atm/hist}"
OUT_ROOT="${PAPER1_DERIVED_ROOT}/B2000WCN001002_timefixed"
MANIFEST_ROOT="${PAPER1_DERIVED_ROOT}/manifests"
RUN2_OFFSET=104
MAX_JOBS="${PAPER1_MAX_JOBS:-8}"

# U/V/T/Z3/O3/PS are needed by the retained dynamical and ozone figures.
# CLOX and H2O are probed and extracted when they actually occur in the h3
# schema; the independent EXTR ClOx climatology is built by script 08.
REQUIRED_VARS=(U V T Z3 O3 PS)
OPTIONAL_VARS=(CLOX H2O)
COORD_VARS="P0,hyai,hyam,hybi,hybm,date,datesec,time,time_bnds,lat,lon,lev,ilev,gw"

[[ -d "${SRC_RUN001}" ]] || die "missing run-001 source: ${SRC_RUN001}"
[[ -d "${SRC_RUN002}" ]] || die "missing run-002 source: ${SRC_RUN002}"
prepare_output_dir "${OUT_ROOT}"
prepare_output_dir "${MANIFEST_ROOT}"

task_file=""
cleanup_top_level() {
    local candidate
    for candidate in "${availability_tmp:-}" "${raw_years_tmp:-}" \
        "${raw_files_tmp:-}" "${task_file:-}"; do
        [[ -z "${candidate}" ]] || rm -f -- "${candidate}"
    done
}
trap cleanup_top_level EXIT

discover_years() {
    local source_dir="$1" run_id="$2"
    find "${source_dir}" -maxdepth 1 -type f \
        -name "B2000WCN.e122.f19_g16.${run_id}.cam.h3.[0-9][0-9][0-9][0-9]-*.nc.extr.nc" \
        -printf '%f\n' \
        | sed -E 's/.*\.cam\.h3\.([0-9]{4})-.*/\1/' \
        | sort -u
}

mapfile -t YEARS001 < <(discover_years "${SRC_RUN001}" 001)
mapfile -t YEARS002 < <(discover_years "${SRC_RUN002}" 002)
[[ ${#YEARS001[@]} -eq 104 ]] \
    || die "expected 104 run-001 raw years; found ${#YEARS001[@]}"
[[ ${#YEARS002[@]} -eq 106 ]] \
    || die "expected 106 run-002 raw years; found ${#YEARS002[@]}"

first_source_file() {
    local source_dir="$1" run_id="$2" year="$3"
    find "${source_dir}" -maxdepth 1 -type f \
        -name "B2000WCN.e122.f19_g16.${run_id}.cam.h3.${year}-*.nc.extr.nc" \
        -print -quit
}

availability_tmp="${MANIFEST_ROOT}/waccm_longrun_variable_availability.tsv.tmp.$$"
availability_manifest="${MANIFEST_ROOT}/waccm_longrun_variable_availability.tsv"
prepare_output_file "${availability_tmp}"
prepare_output_file "${availability_manifest}"
printf 'run\tvariable\tavailable\tprobe_file\n' > "${availability_tmp}"
AVAILABLE001=()
AVAILABLE002=()
for run_id in 001 002; do
    if [[ "${run_id}" == 001 ]]; then
        source_dir="${SRC_RUN001}"
        probe_year="${YEARS001[0]}"
    else
        source_dir="${SRC_RUN002}"
        probe_year="${YEARS002[0]}"
    fi
    probe_file=$(first_source_file "${source_dir}" "${run_id}" "${probe_year}")
    [[ -n "${probe_file}" ]] || die "cannot select schema probe for run ${run_id}"
    for variable in "${REQUIRED_VARS[@]}" "${OPTIONAL_VARS[@]}"; do
        if ncks -m -v "${variable}" "${probe_file}" >/dev/null 2>&1; then
            printf '%s\t%s\t1\t%s\n' "${run_id}" "${variable}" "${probe_file}" >> "${availability_tmp}"
            if [[ "${run_id}" == 001 ]]; then
                AVAILABLE001+=("${variable}")
            else
                AVAILABLE002+=("${variable}")
            fi
        else
            printf '%s\t%s\t0\t%s\n' "${run_id}" "${variable}" "${probe_file}" >> "${availability_tmp}"
        fi
    done
done
mv -- "${availability_tmp}" "${availability_manifest}"

for variable in "${REQUIRED_VARS[@]}"; do
    grep -q $'001\t'"${variable}"$'\t1\t' "${MANIFEST_ROOT}/waccm_longrun_variable_availability.tsv" \
        || die "required ${variable} is absent from run-001 h3 source"
    grep -q $'002\t'"${variable}"$'\t1\t' "${MANIFEST_ROOT}/waccm_longrun_variable_availability.tsv" \
        || die "required ${variable} is absent from run-002 h3 source"
done

raw_years_tmp="${MANIFEST_ROOT}/waccm_longrun_raw_years.tsv.tmp.$$"
raw_years_manifest="${MANIFEST_ROOT}/waccm_longrun_raw_years.tsv"
prepare_output_file "${raw_years_tmp}"
prepare_output_file "${raw_years_manifest}"
printf 'run\tsource_year\ttarget_year\n' > "${raw_years_tmp}"
for year in "${YEARS001[@]}"; do
    printf '001\t%s\t%04d\n' "${year}" "$((10#${year}))" >> "${raw_years_tmp}"
done
for year in "${YEARS002[@]}"; do
    printf '002\t%s\t%04d\n' "${year}" "$((10#${year} + RUN2_OFFSET))" >> "${raw_years_tmp}"
done
mv -- "${raw_years_tmp}" "${raw_years_manifest}"

raw_files_tmp="${MANIFEST_ROOT}/waccm_longrun_raw_files.tsv.tmp.$$"
raw_files_manifest="${MANIFEST_ROOT}/waccm_longrun_raw_files.tsv"
event_manifest="${MANIFEST_ROOT}/waccm_longrun_complete_events.tsv"
prepare_output_file "${raw_files_tmp}"
prepare_output_file "${raw_files_manifest}"
prepare_output_file "${event_manifest}"
printf 'run\tsource_year\ttarget_year\tpath\tsize_bytes\tmtime_epoch\n' > "${raw_files_tmp}"
for run_id in 001 002; do
    source_dir="${SRC_RUN001}"
    [[ "${run_id}" == 002 ]] && source_dir="${SRC_RUN002}"
    while IFS= read -r path; do
        source_year=$(basename "${path}" | sed -E 's/.*\.cam\.h3\.([0-9]{4})-.*/\1/')
        target_year=$((10#${source_year}))
        [[ "${run_id}" == 002 ]] && target_year=$((target_year + RUN2_OFFSET))
        printf '%s\t%s\t%04d\t%s\t%s\t%s\n' \
            "${run_id}" "${source_year}" "${target_year}" "${path}" \
            "$(stat -c '%s' "${path}")" "$(stat -c '%Y' "${path}")" >> "${raw_files_tmp}"
    done < <(find "${source_dir}" -maxdepth 1 -type f \
        -name "B2000WCN.e122.f19_g16.${run_id}.cam.h3.[0-9][0-9][0-9][0-9]-*.nc.extr.nc" | sort)
done
mv -- "${raw_files_tmp}" "${raw_files_manifest}"

waccm_schema_manifest="${MANIFEST_ROOT}/waccm_longrun_source_schema.tsv"
prepare_output_file "${waccm_schema_manifest}"
waccm_schema_args=(
    waccm-schema
    --inventory "${raw_files_manifest}"
    --case-column run
    --required-vars "$(IFS=,; echo "${REQUIRED_VARS[*]}")"
    --output "${waccm_schema_manifest}"
)
[[ "${PAPER1_AUDIT_ONLY}" == 1 ]] && waccm_schema_args+=(--all-files)
python "${SCRIPT_DIR}/paper1_source_audit.py" "${waccm_schema_args[@]}"

python "${SCRIPT_DIR}/paper1_noleap.py" source-event-manifest \
    --inventory "${raw_files_manifest}" \
    --expected-years 210 \
    --expected-complete 207 \
    --output "${event_manifest}"

if [[ "${PAPER1_AUDIT_ONLY}" == 1 ]]; then
    printf 'audit-only: validated every long-run chunk native grid/schema, 104 + 106 raw years, and 207 padded event windows; no concatenation run\n'
    exit 0
fi

extract_one() (
    local run_id="$1" source_dir="$2" source_year="$3" variable="$4" offset="$5"
    local target_year target_dir target raw_tmp selected_vars
    target_year=$((10#${source_year} + offset))
    printf -v target_year '%04d' "${target_year}"
    target_dir="${OUT_ROOT}/${variable}"
    target="${target_dir}/B2000WCN.sample.cam.h3.${target_year}.${variable}.nc"
    prepare_output_file "${target}"
    if [[ -e "${target}" ]]; then
        [[ -s "${target}" ]] || { echo "empty target: ${target}" >&2; return 1; }
        python "${NOLEAP_HELPER}" normalize-year \
            --input "${target}" --output "${target}" --variable "${variable}" \
            --year "$((10#${target_year}))" --source-year "$((10#${source_year}))" \
            --source-run "B2000WCN.${run_id}"
        return 0
    fi

    shopt -s nullglob
    local inputs=("${source_dir}/B2000WCN.e122.f19_g16.${run_id}.cam.h3.${source_year}-"*.nc.extr.nc)
    shopt -u nullglob
    ((${#inputs[@]} > 0)) || { echo "no inputs for ${run_id}/${source_year}" >&2; return 1; }
    raw_tmp="${target}.ncrcat.tmp.$$"
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
        --year "$((10#${target_year}))" --source-year "$((10#${source_year}))" \
        --source-run "B2000WCN.${run_id}"; then
        rm -f -- "${raw_tmp}"
        return 1
    fi
    rm -f -- "${raw_tmp}"
)

export -f extract_one
export OUT_ROOT COORD_VARS
export NOLEAP_HELPER="${SCRIPT_DIR}/paper1_noleap.py"

task_file=$(mktemp "${MANIFEST_ROOT}/.waccm_longrun_tasks.XXXXXX")
assert_under_staging "${task_file}"
for year in "${YEARS001[@]}"; do
    for variable in "${AVAILABLE001[@]}"; do
        printf 'extract_one 001 %q %q %q 0\n' "${SRC_RUN001}" "${year}" "${variable}" >> "${task_file}"
    done
done
for year in "${YEARS002[@]}"; do
    for variable in "${AVAILABLE002[@]}"; do
        printf 'extract_one 002 %q %q %q %q\n' \
            "${SRC_RUN002}" "${year}" "${variable}" "${RUN2_OFFSET}" >> "${task_file}"
    done
done
xargs -P "${MAX_JOBS}" -I TASK bash -c 'TASK' < "${task_file}"

python "${SCRIPT_DIR}/paper1_noleap.py" event-manifest \
    --root "${OUT_ROOT}" \
    --variables "$(IFS=,; echo "${REQUIRED_VARS[*]}")" \
    --expected-years 210 \
    --expected-complete 207 \
    --output "${event_manifest}"

printf 'validated long run: 104 + 106 raw years; 207 complete Feb27-May2 event windows\n'
