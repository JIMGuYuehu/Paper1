#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   MERRA-2 U/V/T/O3 daily SUB granules:
#     /mnt/soclim0/public_data/weiji/MERRA2M2I6NPANA
#   MERRA-2 geopotential-height source H granules:
#     /mnt/soclim0/public_data/weiji/MERRA2M2I6NPANA/Z
# OUTPUT
#   ${PAPER1_DERIVED_ROOT}/MERRA2_Processed/{U,V,T,O3,Z3}/MERRA2.*.YYYY.nc
#   ${PAPER1_DERIVED_ROOT}/manifests/merra2_daily_sources_1980_2025.tsv
#   ${PAPER1_DERIVED_ROOT}/manifests/merra2_yearly_products_1980_2025.tsv
# ACTION
#   Require one source granule for every Gregorian day in 1980-2025. A
#   one-record granule is accepted only when metadata proves it is the GES DISC
#   Daily mean covering 00-18 UTC; such data are not averaged a second time.
#   A genuine four-record inst6 granule is day-averaged locally. Merge each
#   source family once per year, extract U/V/T/O3, rename source H to Z3, then
#   reopen every annual product to verify its exact grid, units, and date axis.
# REQUIREMENTS
#   Bash, Python/netCDF4, system NCO, and CDO 2.6.1 (default cdo_tools path).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=00_common.sh
source "${SCRIPT_DIR}/00_common.sh"
require_commands awk find ncks ncrename python

CORE_ROOT="${PAPER1_MERRA2_SOURCE:-/mnt/soclim0/public_data/weiji/MERRA2M2I6NPANA}"
Z_ROOT="${PAPER1_MERRA2_Z_SOURCE:-${CORE_ROOT}/Z}"
OUT_ROOT="${PAPER1_DERIVED_ROOT}/MERRA2_Processed"
MANIFEST="${PAPER1_DERIVED_ROOT}/manifests/merra2_daily_sources_1980_2025.tsv"
CDO_BIN="${PAPER1_CDO_BIN:-/home/weiji/miniconda3/envs/cdo_tools/bin/cdo}"
MAX_JOBS="${PAPER1_MAX_JOBS:-4}"
CORE_VARS=(U V T O3)

[[ -d "${CORE_ROOT}" ]] || die "missing MERRA-2 core source: ${CORE_ROOT}"
[[ -d "${Z_ROOT}" ]] || die "missing MERRA-2 H source: ${Z_ROOT}"
[[ -x "${CDO_BIN}" ]] || die "CDO executable not found: ${CDO_BIN}"
prepare_output_dir "${OUT_ROOT}"
prepare_output_file "${MANIFEST}"

python "${SCRIPT_DIR}/paper1_source_audit.py" merra \
    --core-root "${CORE_ROOT}" \
    --z-root "${Z_ROOT}" \
    --start-year 1980 \
    --end-year 2025 \
    --output "${MANIFEST}"

if [[ "${PAPER1_AUDIT_ONLY}" == 1 ]]; then
    printf 'audit-only: validated MERRA-2 1980-2025 coverage and metadata; no yearly merge run\n'
    exit 0
fi

validate_existing() {
    local target="$1" variable="$2" expected_days="$3"
    [[ -s "${target}" ]] || return 1
    ncks -m -v "${variable}" "${target}" >/dev/null 2>&1 || return 1
    [[ $("${CDO_BIN}" -s ntime "${target}") -eq "${expected_days}" ]]
}

process_year() (
    local year="$1" expected_days="$2" core_mode="$3" z_mode="$4"
    local year_tmp="${OUT_ROOT}/_tmp/${year}.$$"
    prepare_output_dir "${year_tmp}"
    trap 'rm -f -- "${year_tmp}"/*.nc; [[ -z "${packed:-}" ]] || rm -f -- "${packed}"; [[ -z "${z_packed:-}" ]] || rm -f -- "${z_packed}"; rmdir -- "${year_tmp}" 2>/dev/null || true' EXIT
    shopt -s nullglob
    local core_files=("${CORE_ROOT}"/MERRA2_*.inst6_3d_ana_Np."${year}"????.SUB.nc)
    local z_files=("${Z_ROOT}"/MERRA2_*.inst6_3d_ana_Np."${year}"????.SUB.nc)
    shopt -u nullglob
    [[ ${#core_files[@]} -eq ${expected_days} ]] \
        || { echo "wrong core file count for ${year}" >&2; return 1; }
    [[ ${#z_files[@]} -eq ${expected_days} ]] \
        || { echo "wrong H file count for ${year}" >&2; return 1; }

    local need_core=0 variable target
    for variable in U V T O3; do
        target="${OUT_ROOT}/${variable}/MERRA2.${variable}.${year}.nc"
        prepare_output_file "${target}"
        if [[ -e "${target}" ]]; then
            validate_existing "${target}" "${variable}" "${expected_days}" \
                || { echo "existing output failed validation: ${target}" >&2; return 1; }
        else
            need_core=1
        fi
    done
    if ((need_core)); then
        local core_merged="${year_tmp}/core_merged.nc"
        local core_daily="${core_merged}"
        assert_under_staging "${core_merged}"
        "${CDO_BIN}" -L -s -O mergetime "${core_files[@]}" "${core_merged}"
        if [[ "${core_mode}" == four_inst6_records ]]; then
            core_daily="${year_tmp}/core_daily.nc"
            assert_under_staging "${core_daily}"
            "${CDO_BIN}" -L -s -O daymean "${core_merged}" "${core_daily}"
        elif [[ "${core_mode}" != gesdisc_daily_mean ]]; then
            echo "unsupported core source mode: ${core_mode}" >&2
            return 1
        fi
        [[ $("${CDO_BIN}" -s ntime "${core_daily}") -eq "${expected_days}" ]] \
            || { echo "wrong merged core time count for ${year}" >&2; return 1; }
        for variable in U V T O3; do
            target="${OUT_ROOT}/${variable}/MERRA2.${variable}.${year}.nc"
            [[ -e "${target}" ]] && continue
            prepare_output_file "${target}"
            local packed="${target}.tmp.$$"
            assert_under_staging "${packed}"
            ncks -4 -L 1 -O -v "${variable},lev,lat,lon,time,time_bnds" \
                "${core_daily}" "${packed}"
            validate_existing "${packed}" "${variable}" "${expected_days}" \
                || { rm -f -- "${packed}"; return 1; }
            mv -- "${packed}" "${target}"
        done
    fi

    target="${OUT_ROOT}/Z3/MERRA2.Z3.${year}.nc"
    prepare_output_file "${target}"
    if [[ -e "${target}" ]]; then
        validate_existing "${target}" Z3 "${expected_days}" \
            || { echo "existing output failed validation: ${target}" >&2; return 1; }
    else
        local z_merged="${year_tmp}/z_merged.nc"
        local z_daily="${z_merged}"
        assert_under_staging "${z_merged}"
        "${CDO_BIN}" -L -s -O mergetime "${z_files[@]}" "${z_merged}"
        if [[ "${z_mode}" == four_inst6_records ]]; then
            z_daily="${year_tmp}/z_daily.nc"
            assert_under_staging "${z_daily}"
            "${CDO_BIN}" -L -s -O daymean "${z_merged}" "${z_daily}"
        elif [[ "${z_mode}" != gesdisc_daily_mean ]]; then
            echo "unsupported H source mode: ${z_mode}" >&2
            return 1
        fi
        [[ $("${CDO_BIN}" -s ntime "${z_daily}") -eq "${expected_days}" ]] \
            || { echo "wrong merged H time count for ${year}" >&2; return 1; }
        prepare_output_file "${target}"
        local z_packed="${target}.tmp.$$"
        assert_under_staging "${z_packed}"
        ncks -4 -L 1 -O -v 'H,lev,lat,lon,time,time_bnds' "${z_daily}" "${z_packed}"
        ncrename -O -v H,Z3 "${z_packed}"
        validate_existing "${z_packed}" Z3 "${expected_days}" \
            || { rm -f -- "${z_packed}"; return 1; }
        mv -- "${z_packed}" "${target}"
    fi
    rm -f -- "${year_tmp}"/*.nc
    rmdir -- "${year_tmp}" 2>/dev/null || true
    printf 'completed MERRA-2 year %s (%s days)\n' "${year}" "${expected_days}"
)

export -f validate_existing process_year
export CORE_ROOT Z_ROOT OUT_ROOT CDO_BIN

task_file=$(mktemp "${PAPER1_DERIVED_ROOT}/manifests/.merra2_tasks.XXXXXX")
assert_under_staging "${task_file}"
trap 'rm -f -- "${task_file}"' EXIT
while IFS=$'\t' read -r year expected_days _ _ core_mode z_mode _; do
    [[ "${year}" == year ]] && continue
    printf 'process_year %q %q %q %q\n' \
        "${year}" "${expected_days}" "${core_mode}" "${z_mode}" >> "${task_file}"
done < "${MANIFEST}"
xargs -P "${MAX_JOBS}" -I TASK bash -c 'TASK' < "${task_file}"

# This batch audit also inspects outputs that were retained from an earlier
# run; a filename and ntime count alone never qualify an annual product.
YEARLY_MANIFEST="${PAPER1_DERIVED_ROOT}/manifests/merra2_yearly_products_1980_2025.tsv"
prepare_output_file "${YEARLY_MANIFEST}"
python "${SCRIPT_DIR}/paper1_source_audit.py" merra-yearly \
    --root "${OUT_ROOT}" \
    --start-year 1980 \
    --end-year 2025 \
    --output "${YEARLY_MANIFEST}"
