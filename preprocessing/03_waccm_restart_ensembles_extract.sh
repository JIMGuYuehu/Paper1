#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Raw perfect-model restart files under ${PAPER1_ARCHIVE_ROOT}/WACCM/hindcast/0008-01 and
#   ${PAPER1_ARCHIVE_ROOT}/WACCM/hindcast/0008-02; a member may contain several h3 segments.
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/Hindcast/0008-01|0008-02/<VARIABLE>/*.nc
#   ${PAPER1_DERIVED_ROOT}/manifests/waccm_hindcast_jan_feb_members.tsv
# ACTION
#   Discover member prefixes, require 30 January and 30 February members with
#   the same normalized member IDs, open every raw chunk, verify the required
#   schema and continuous daily noleap start/end contract through at least 31
#   May for the event-day-150 figures. January must additionally contain the
#   exact 150 daily records from 1 January through 30 May used by Figure 7.
#   Then concatenate each
#   member/variable atomically. Retained/new outputs are revalidated against
#   that source contract. March files are audited separately by 04.
# REQUIREMENTS
#   Bash, NCO, and write access to the staging root.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_commands find comm ncks ncrcat python stat

INPUT_BASE="${PAPER1_HINDCAST_SOURCE:-${PAPER1_ARCHIVE_ROOT}/WACCM/hindcast}"
OUTPUT_BASE="${PAPER1_DERIVED_ROOT}/Hindcast"
MANIFEST_ROOT="${PAPER1_DERIVED_ROOT}/manifests"
MAX_JOBS="${PAPER1_MAX_JOBS:-8}"
CASES=(0008-01 0008-02)
CORE_VARS=(U V T Z3 O3 PS)
COORD_VARS="P0,hyai,hyam,hybi,hybm,date,datesec,time,time_bnds,lat,lon,lev,ilev,gw"

prepare_output_dir "${OUTPUT_BASE}"
prepare_output_dir "${MANIFEST_ROOT}"

prefixes_for_case() {
    local case_name="$1" case_dir="${INPUT_BASE}/$1"
    find "${case_dir}" -maxdepth 1 -type f -name '*.cam.h3.*.nc*' -printf '%p\n' \
        | sed -E 's/\.[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{5}\.nc.*$//' \
        | sort -u
}

normalize_member_id() {
    local case_name="$1" prefix="$2" name
    name=$(basename "${prefix}")
    printf '%s\n' "${name/${case_name}/CASE}"
}

jan_prefixes=""
feb_prefixes=""
jan_ids=""
feb_ids=""
common_ids=""
task_file=""
cleanup_top_level() {
    local candidate
    for candidate in "${jan_prefixes}" "${feb_prefixes}" "${jan_ids}" \
        "${feb_ids}" "${common_ids}" "${schema_tmp:-}" "${member_tmp:-}" \
        "${raw_files_tmp:-}" "${task_file}"; do
        [[ -z "${candidate}" ]] || rm -f -- "${candidate}"
    done
}
trap cleanup_top_level EXIT

jan_prefixes=$(mktemp "${MANIFEST_ROOT}/.hindcast_jan_prefixes.XXXXXX")
feb_prefixes=$(mktemp "${MANIFEST_ROOT}/.hindcast_feb_prefixes.XXXXXX")
jan_ids=$(mktemp "${MANIFEST_ROOT}/.hindcast_jan_ids.XXXXXX")
feb_ids=$(mktemp "${MANIFEST_ROOT}/.hindcast_feb_ids.XXXXXX")
common_ids=$(mktemp "${MANIFEST_ROOT}/.hindcast_common_ids.XXXXXX")
for temporary in "${jan_prefixes}" "${feb_prefixes}" "${jan_ids}" "${feb_ids}" "${common_ids}"; do
    assert_under_staging "${temporary}"
done

for case_name in "${CASES[@]}"; do
    [[ -d "${INPUT_BASE}/${case_name}" ]] || die "missing ensemble case: ${INPUT_BASE}/${case_name}"
done
prefixes_for_case 0008-01 > "${jan_prefixes}"
prefixes_for_case 0008-02 > "${feb_prefixes}"
[[ $(wc -l < "${jan_prefixes}") -eq 30 ]] || die "January ensemble does not have 30 members"
[[ $(wc -l < "${feb_prefixes}") -eq 30 ]] || die "February ensemble does not have 30 members"
while IFS= read -r prefix; do normalize_member_id 0008-01 "${prefix}"; done < "${jan_prefixes}" | sort > "${jan_ids}"
while IFS= read -r prefix; do normalize_member_id 0008-02 "${prefix}"; done < "${feb_prefixes}" | sort > "${feb_ids}"
comm -12 "${jan_ids}" "${feb_ids}" > "${common_ids}"
[[ $(wc -l < "${common_ids}") -eq 30 ]] \
    || die "January and February do not share the same 30 normalized member IDs"

schema_tmp="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_schema.tsv.tmp.$$"
schema_manifest="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_schema.tsv"
prepare_output_file "${schema_tmp}"
prepare_output_file "${schema_manifest}"
printf 'case\tvariable\tavailable\tprobe_file\n' > "${schema_tmp}"
for case_name in "${CASES[@]}"; do
    prefix_file="${jan_prefixes}"
    [[ "${case_name}" == 0008-02 ]] && prefix_file="${feb_prefixes}"
    first_prefix=$(head -n 1 "${prefix_file}")
    shopt -s nullglob
    segments=("${first_prefix}."*.nc*)
    shopt -u nullglob
    ((${#segments[@]} > 0)) || die "no segments for schema probe: ${first_prefix}"
    for variable in "${CORE_VARS[@]}"; do
        if ncks -m -v "${variable}" "${segments[0]}" >/dev/null 2>&1; then
            printf '%s\t%s\t1\t%s\n' "${case_name}" "${variable}" "${segments[0]}" >> "${schema_tmp}"
        else
            rm -f -- "${schema_tmp}"
            die "${case_name} probe is missing ${variable}: ${segments[0]}"
        fi
    done
done
mv -- "${schema_tmp}" "${schema_manifest}"

member_tmp="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_members.tsv.tmp.$$"
member_manifest="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_members.tsv"
prepare_output_file "${member_tmp}"
prepare_output_file "${member_manifest}"
printf 'normalized_member_id\tin_january\tin_february\n' > "${member_tmp}"
while IFS= read -r member_id; do
    printf '%s\t1\t1\n' "${member_id}" >> "${member_tmp}"
done < "${common_ids}"
mv -- "${member_tmp}" "${member_manifest}"

raw_files_tmp="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_raw_files.tsv.tmp.$$"
raw_files_manifest="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_raw_files.tsv"
prepare_output_file "${raw_files_tmp}"
prepare_output_file "${raw_files_manifest}"
printf 'case\tpath\tsize_bytes\tmtime_epoch\n' > "${raw_files_tmp}"
for case_name in "${CASES[@]}"; do
    while IFS= read -r path; do
        printf '%s\t%s\t%s\t%s\n' \
            "${case_name}" "${path}" "$(stat -c '%s' "${path}")" \
            "$(stat -c '%Y' "${path}")" >> "${raw_files_tmp}"
    done < <(find "${INPUT_BASE}/${case_name}" -maxdepth 1 -type f \
        -name '*.cam.h3.*.nc*' | sort)
done
mv -- "${raw_files_tmp}" "${raw_files_manifest}"

waccm_schema_manifest="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_source_schema.tsv"
prepare_output_file "${waccm_schema_manifest}"
waccm_schema_args=(
    waccm-schema
    --inventory "${raw_files_manifest}"
    --case-column case
    --required-vars "$(IFS=,; echo "${CORE_VARS[*]}")"
    --output "${waccm_schema_manifest}"
)
[[ "${PAPER1_AUDIT_ONLY}" == 1 ]] && waccm_schema_args+=(--all-files)
python "${SCRIPT_DIR}/source_inventory.py" "${waccm_schema_args[@]}"

SOURCE_CONTRACT="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_source_contract.tsv"
prepare_output_file "${SOURCE_CONTRACT}"
python "${SCRIPT_DIR}/source_inventory.py" hindcast \
    --source-root "${INPUT_BASE}" \
    --output "${SOURCE_CONTRACT}"

if [[ "${PAPER1_AUDIT_ONLY}" == 1 ]]; then
    printf 'audit-only: validated every chunk and daily coverage through May31 for 30+30 common members, including January Figure7 Nt=150; no concatenation run\n'
    exit 0
fi

process_member() (
    local case_name="$1" prefix="$2"
    shopt -s nullglob
    local inputs=("${prefix}."*.nc*)
    shopt -u nullglob
    ((${#inputs[@]} > 0)) || { echo "no inputs for ${prefix}" >&2; return 1; }
    local member_name
    member_name=$(basename "${prefix}")
    IFS=, read -r -a variables <<< "${CORE_VARS_STR}"
    local variable out_dir target raw_tmp="" packed_tmp="" selected_vars
    trap '[[ -z "${raw_tmp}" ]] || rm -f -- "${raw_tmp}"; [[ -z "${packed_tmp}" ]] || rm -f -- "${packed_tmp}"' EXIT
    for variable in "${variables[@]}"; do
        out_dir="${OUTPUT_BASE}/${case_name}/${variable}"
        target="${out_dir}/${member_name}.${variable}.nc"
        prepare_output_file "${target}"
        if [[ -e "${target}" ]]; then
            [[ -s "${target}" ]] || { echo "empty target: ${target}" >&2; return 1; }
            ncks -m -v "${variable},date,datesec" "${target}" >/dev/null
            continue
        fi
        raw_tmp="${target}.ncrcat.tmp.$$"
        packed_tmp="${target}.packed.tmp.$$"
        assert_under_staging "${raw_tmp}"
        assert_under_staging "${packed_tmp}"
        rm -f -- "${raw_tmp}" "${packed_tmp}"
        selected_vars="${variable},PS,${COORD_VARS}"
        [[ "${variable}" == PS ]] && selected_vars="PS,${COORD_VARS}"
        if ! ncrcat -O -v "${selected_vars}" "${inputs[@]}" "${raw_tmp}"; then
            rm -f -- "${raw_tmp}" "${packed_tmp}"
            return 1
        fi
        if ! ncks -4 -L 1 -O "${raw_tmp}" "${packed_tmp}"; then
            rm -f -- "${raw_tmp}" "${packed_tmp}"
            return 1
        fi
        ncks -m -v "${variable},date,datesec" "${packed_tmp}" >/dev/null
        mv -- "${packed_tmp}" "${target}"
        rm -f -- "${raw_tmp}"
    done
)

export -f process_member
export OUTPUT_BASE COORD_VARS
export CORE_VARS_STR="$(IFS=,; echo "${CORE_VARS[*]}")"

task_file=$(mktemp "${MANIFEST_ROOT}/.hindcast_tasks.XXXXXX")
assert_under_staging "${task_file}"
for case_name in "${CASES[@]}"; do
    prefix_file="${jan_prefixes}"
    [[ "${case_name}" == 0008-02 ]] && prefix_file="${feb_prefixes}"
    while IFS= read -r prefix; do
        printf 'process_member %q %q\n' "${case_name}" "${prefix}" >> "${task_file}"
    done < "${prefix_file}"
done
xargs -P "${MAX_JOBS}" -I TASK bash -c 'TASK' < "${task_file}"
products_manifest="${MANIFEST_ROOT}/waccm_hindcast_jan_feb_products.tsv"
prepare_output_file "${products_manifest}"
python "${SCRIPT_DIR}/source_inventory.py" hindcast-outputs \
    --root "${OUTPUT_BASE}" \
    --source-manifest "${SOURCE_CONTRACT}" \
    --output "${products_manifest}"
printf 'validated restart ensembles and every retained/new product: January=30, February=30, common=30\n'
