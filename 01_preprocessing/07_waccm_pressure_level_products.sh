#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Staged hybrid-coordinate U/V/T/Z3 annual and restart-member files produced
#   by scripts 01-03 under ${PAPER1_DERIVED_ROOT}.
# OUTPUT
#   <CASE>/interpolated/<VARIABLE>/<source filename> under the same staging root
#   ${PAPER1_DERIVED_ROOT}/manifests/waccm_{longrun,bwcn}_pressure_years.tsv
#   and ${PAPER1_DERIVED_ROOT}/manifests/waccm_cdo_ml2pl_products.tsv.
# ACTION
#   Add the CAM lev-to-ilev bounds hint on a temporary copy, then run CDO ml2pl
#   to the exact 36 fixed pressure levels used by the accepted NAM/EP-flux
#   workflow. Derive LONGRUN/BWCN pressure sources from raw-year and padded
#   event manifests as complete event Y UNION every existing Y-1 predecessor.
#   Thus an incomplete spring can still supply a later event's Nov-Dec field.
#   Ranking counts (207/23), pressure-year counts (209/23), and Figure-15 field
#   availability (206/22) are separate asserted contracts. Source and legacy
#   public files are never modified.
# REQUIREMENTS
#   Bash, Python/netCDF4, system NCO, and CDO 2.6.1 (default cdo_tools path).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=00_common.sh
source "${SCRIPT_DIR}/00_common.sh"
require_commands awk find ncatted ncks python

CDO_BIN="${PAPER1_CDO_BIN:-/home/weiji/miniconda3/envs/cdo_tools/bin/cdo}"
[[ -x "${CDO_BIN}" ]] || die "CDO executable not found: ${CDO_BIN}"
MAX_JOBS="${PAPER1_MAX_JOBS:-4}"
VARS=(Z3 U V T)
PLEV_PA="100000,95000,92500,90000,85000,80000,75000,70000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,4000,3000,2000,1000,700,500,400,300,200,100,50,10"
CASE_ROOTS=(
    "${PAPER1_DERIVED_ROOT}/B2000WCN001002_timefixed"
    "${PAPER1_DERIVED_ROOT}/BWCN"
    "${PAPER1_DERIVED_ROOT}/Hindcast/0008-01"
    "${PAPER1_DERIVED_ROOT}/Hindcast/0008-02"
)
MANIFEST="${PAPER1_DERIVED_ROOT}/manifests/waccm_cdo_ml2pl_products.tsv"
LONG_SELECTION="${PAPER1_DERIVED_ROOT}/manifests/waccm_longrun_pressure_years.tsv"
BWCN_SELECTION="${PAPER1_DERIVED_ROOT}/manifests/waccm_bwcn_pressure_years.tsv"
prepare_output_file "${MANIFEST}"
prepare_output_file "${LONG_SELECTION}"
prepare_output_file "${BWCN_SELECTION}"

if [[ "${PAPER1_AUDIT_ONLY}" == 1 ]]; then
    printf 'audit-only: ml2pl needs staged NetCDF inputs and is intentionally skipped\n'
    exit 0
fi

WACCM_TMP_ROOT="${PAPER1_DERIVED_ROOT}/_tmp/waccm_ml2pl"
prepare_output_dir "${WACCM_TMP_ROOT}"

python "${SCRIPT_DIR}/paper1_noleap.py" pressure-year-manifest \
    --raw-years "${PAPER1_DERIVED_ROOT}/manifests/waccm_longrun_raw_years.tsv" \
    --events "${PAPER1_DERIVED_ROOT}/manifests/waccm_longrun_complete_events.tsv" \
    --segment LONGRUN \
    --expected-raw-years 210 \
    --expected-ranking-events 207 \
    --expected-pressure-years 209 \
    --expected-field-events 206 \
    --expected-unavailable-events 1 \
    --output "${LONG_SELECTION}"

python "${SCRIPT_DIR}/paper1_noleap.py" pressure-year-manifest \
    --raw-years "${PAPER1_DERIVED_ROOT}/manifests/waccm_bwcn_raw_years.tsv" \
    --events "${PAPER1_DERIVED_ROOT}/manifests/waccm_bwcn_complete_events.tsv" \
    --segment BWCN \
    --expected-raw-years 24 \
    --expected-ranking-events 23 \
    --expected-pressure-years 23 \
    --expected-field-events 22 \
    --expected-unavailable-events 1 \
    --output "${BWCN_SELECTION}"

valid_output() {
    local path="$1" variable="$2"
    [[ -s "${path}" ]] || return 1
    python "${SCRIPT_DIR}/paper1_source_audit.py" pressure-file \
        --path "${path}" --variable "${variable}" >/dev/null 2>&1
}

process_one() (
    local source="$1" output="$2" variable="$3"
    prepare_output_file "${output}"
    if [[ -e "${output}" ]]; then
        valid_output "${output}" "${variable}" \
            || { echo "existing pressure-level output failed validation: ${output}" >&2; return 1; }
        return 0
    fi
    local temporary_dir
    temporary_dir=$(mktemp -d "${WACCM_TMP_ROOT}/job.XXXXXX")
    assert_under_staging "${temporary_dir}"
    local with_bounds="${temporary_dir}/with_lev_bounds.nc"
    local temporary_output="${output}.tmp.$$"
    assert_under_staging "${temporary_output}"
    trap 'rm -f -- "${temporary_output}" "${with_bounds}"; rmdir -- "${temporary_dir}" 2>/dev/null || true' EXIT
    if ! ncatted -O -a bounds,lev,c,c,ilev "${source}" "${with_bounds}"; then
        rm -f -- "${with_bounds}"
        rmdir -- "${temporary_dir}" 2>/dev/null || true
        return 1
    fi
    # CDO evaluates chained operators from right to left. On the verified
    # WACCM sample, select(name=VAR) retained PS and all hybrid-coordinate
    # dependencies; the following ml2pl produced VAR(time,plev,lat,lon).
    if ! "${CDO_BIN}" -L -s -O -f nc4 -z zip_1 \
        -ml2pl,"${PLEV_PA}" -select,name="${variable}" \
        "${with_bounds}" "${temporary_output}"; then
        rm -f -- "${temporary_output}"
        rm -f -- "${with_bounds}"
        rmdir -- "${temporary_dir}" 2>/dev/null || true
        return 1
    fi
    if ! valid_output "${temporary_output}" "${variable}"; then
        rm -f -- "${temporary_output}"
        rm -f -- "${with_bounds}"
        rmdir -- "${temporary_dir}" 2>/dev/null || true
        return 1
    fi
    mv -- "${temporary_output}" "${output}"
    rm -f -- "${with_bounds}"
    rmdir -- "${temporary_dir}" 2>/dev/null || true
)

export -f valid_output process_one
export CDO_BIN PLEV_PA SCRIPT_DIR WACCM_TMP_ROOT

task_file=$(mktemp "${PAPER1_DERIVED_ROOT}/manifests/.ml2pl_tasks.XXXXXX")
assert_under_staging "${task_file}"
manifest_tmp=""
cleanup() {
    rm -f -- "${task_file}"
    [[ -z "${manifest_tmp}" ]] || rm -f -- "${manifest_tmp}"
}
trap cleanup EXIT
expected_tasks=0
for case_index in "${!CASE_ROOTS[@]}"; do
    case_root="${CASE_ROOTS[case_index]}"
    [[ -d "${case_root}" ]] || die "staged case is missing; run earlier scripts first: ${case_root}"
    selection_manifest=""
    expected_count=30
    expected_ranking=30
    expected_fields=30
    case "${case_index}" in
        0) selection_manifest="${LONG_SELECTION}"; expected_count=209; expected_ranking=207; expected_fields=206 ;;
        1) selection_manifest="${BWCN_SELECTION}"; expected_count=23; expected_ranking=23; expected_fields=22 ;;
    esac
    if [[ -n "${selection_manifest}" ]]; then
        [[ -s "${selection_manifest}" ]] || die "missing/empty pressure-year manifest: ${selection_manifest}"
        ranking_in_manifest=$(awk -F '\t' 'NR>1 && $4==1 {count++} END {print count+0}' "${selection_manifest}")
        pressure_in_manifest=$(awk -F '\t' 'NR>1 && $6==1 {count++} END {print count+0}' "${selection_manifest}")
        fields_in_manifest=$(awk -F '\t' 'NR>1 && $7==1 {count++} END {print count+0}' "${selection_manifest}")
        [[ "${ranking_in_manifest}" -eq "${expected_ranking}" ]] || die \
            "${selection_manifest}: ranking events=${ranking_in_manifest}, expected=${expected_ranking}"
        [[ "${pressure_in_manifest}" -eq "${expected_count}" ]] || die \
            "${selection_manifest}: pressure sources=${pressure_in_manifest}, expected=${expected_count}"
        [[ "${fields_in_manifest}" -eq "${expected_fields}" ]] || die \
            "${selection_manifest}: Figure15 fields=${fields_in_manifest}, expected=${expected_fields}"
    fi
    for variable in "${VARS[@]}"; do
        [[ -d "${case_root}/${variable}" ]] || die "missing staged variable directory: ${case_root}/${variable}"
        variable_count=0
        while IFS= read -r source; do
            [[ -n "${source}" ]] || continue
            if [[ -n "${selection_manifest}" ]]; then
                source_name=$(basename "${source}")
                [[ "${source_name}" =~ \.([0-9]{4})\.${variable}\.nc$ ]] \
                    || die "cannot parse event year from staged source: ${source}"
                source_year=$((10#${BASH_REMATCH[1]}))
                awk -F '\t' -v year="${source_year}" \
                    'NR>1 && ($2+0)==year && $6==1 {found=1} END {exit !found}' \
                    "${selection_manifest}" || continue
            fi
            output="${case_root}/interpolated/${variable}/$(basename "${source}")"
            printf 'process_one %q %q %q\n' "${source}" "${output}" "${variable}" >> "${task_file}"
            variable_count=$((variable_count + 1))
        done < <(find "${case_root}/${variable}" -maxdepth 1 -type f -name "*.${variable}.nc" | sort)
        [[ "${variable_count}" -gt 0 ]] || die "no eligible ${variable} inputs in ${case_root}"
        [[ "${variable_count}" -eq "${expected_count}" ]] || die \
            "${case_root}/${variable}: eligible inputs=${variable_count}, expected=${expected_count}"
        expected_tasks=$((expected_tasks + variable_count))
    done
done
[[ "${expected_tasks}" -gt 0 && -s "${task_file}" ]] || die "ml2pl task manifest is empty"
[[ "${expected_tasks}" -eq 1168 ]] || die \
    "ml2pl task count=${expected_tasks}, expected 1168 (4 variables x 292 sources)"
xargs -P "${MAX_JOBS}" -I TASK bash -c 'TASK' < "${task_file}"

manifest_tmp="${MANIFEST}.tmp.$$"
assert_under_staging "${manifest_tmp}"
printf 'case\tvariable\toutput\n' > "${manifest_tmp}"
manifest_rows=0
declare -A outputs_by_variable=()
for case_index in "${!CASE_ROOTS[@]}"; do
    case_root="${CASE_ROOTS[case_index]}"
    case_name="${case_root#${PAPER1_DERIVED_ROOT}/}"
    expected_count=30
    [[ "${case_index}" -eq 0 ]] && expected_count=209
    [[ "${case_index}" -eq 1 ]] && expected_count=23
    for variable in "${VARS[@]}"; do
        output_count=0
        while IFS= read -r output; do
            valid_output "${output}" "${variable}" || die "invalid output during manifest scan: ${output}"
            printf '%s\t%s\t%s\n' "${case_name}" "${variable}" "${output}" >> "${manifest_tmp}"
            output_count=$((output_count + 1))
        done < <(find "${case_root}/interpolated/${variable}" -maxdepth 1 -type f \
            -name "*.${variable}.nc" 2>/dev/null | sort)
        [[ "${output_count}" -eq "${expected_count}" ]] || die \
            "${case_name}/${variable}: valid outputs=${output_count}, expected=${expected_count}"
        outputs_by_variable["${variable}"]=$(( ${outputs_by_variable["${variable}"]:-0} + output_count ))
        manifest_rows=$((manifest_rows + output_count))
    done
done
[[ "${manifest_rows}" -eq "${expected_tasks}" ]] || die \
    "pressure-product manifest rows=${manifest_rows}, scheduled=${expected_tasks}"
for variable in "${VARS[@]}"; do
    [[ "${outputs_by_variable["${variable}"]:-0}" -eq 292 ]] || die \
        "${variable}: total pressure outputs=${outputs_by_variable["${variable}"]:-0}, expected=292"
done
[[ "${manifest_rows}" -eq 1168 ]] || die "pressure-product total=${manifest_rows}, expected=1168"
mv -- "${manifest_tmp}" "${MANIFEST}"
manifest_tmp=""
printf 'wrote pressure-level product manifest: %s\n' "${MANIFEST}"
