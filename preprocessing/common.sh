#!/usr/bin/env bash
set -euo pipefail

# Shared configuration and write-safety helpers for Paper 1 preprocessing.
# Inputs are read below PAPER1_ARCHIVE_ROOT or the dataset-specific paths.
# All new products are written below PAPER1_DERIVED_ROOT.

COMMON_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PAPER1_PROJECT_ROOT=$(cd -- "${COMMON_SCRIPT_DIR}/.." && pwd)

if [[ -f "${PAPER1_PROJECT_ROOT}/config.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PAPER1_PROJECT_ROOT}/config.sh"
fi

PAPER1_ARCHIVE_ROOT="${PAPER1_ARCHIVE_ROOT:-${PAPER1_PROJECT_ROOT}/data}"
PAPER1_DERIVED_ROOT="${PAPER1_DERIVED_ROOT:-${PAPER1_PROJECT_ROOT}/work}"
PAPER1_AUDIT_ONLY="${PAPER1_AUDIT_ONLY:-0}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_commands() {
    local command_name
    for command_name in "$@"; do
        command -v "${command_name}" >/dev/null 2>&1 \
            || die "required command is not available: ${command_name}"
    done
}

is_within() {
    local child="$1" parent="$2"
    [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

assert_safe_staging_root() {
    require_commands realpath
    [[ -n "${PAPER1_DERIVED_ROOT}" ]] || die "PAPER1_DERIVED_ROOT is empty"
    [[ "${PAPER1_DERIVED_ROOT}" = /* ]] || die "PAPER1_DERIVED_ROOT must be absolute"
    PAPER1_PROJECT_ROOT=$(realpath -m -- "${PAPER1_PROJECT_ROOT}")
    PAPER1_ARCHIVE_ROOT=$(realpath -m -- "${PAPER1_ARCHIVE_ROOT}")
    PAPER1_DERIVED_ROOT=$(realpath -m -- "${PAPER1_DERIVED_ROOT}")
    export PAPER1_PROJECT_ROOT PAPER1_ARCHIVE_ROOT PAPER1_DERIVED_ROOT
    [[ "${PAPER1_DERIVED_ROOT}" != "/" ]] || die "refusing filesystem root"
    if is_within "${PAPER1_DERIVED_ROOT}" "${PAPER1_ARCHIVE_ROOT}" \
        || is_within "${PAPER1_ARCHIVE_ROOT}" "${PAPER1_DERIVED_ROOT}"; then
        die "input and output roots must not overlap"
    fi
}

assert_under_staging() {
    local resolved
    resolved=$(realpath -m -- "$1")
    is_within "${resolved}" "${PAPER1_DERIVED_ROOT}" \
        || die "output escapes PAPER1_DERIVED_ROOT: $1 -> ${resolved}"
}

prepare_output_dir() {
    assert_under_staging "$1"
    mkdir -p "$1"
    assert_under_staging "$1"
}

prepare_output_file() {
    assert_under_staging "$1"
    prepare_output_dir "$(dirname -- "$1")"
}

refuse_bad_existing_output() {
    if [[ -e "$1" && ! -s "$1" ]]; then
        die "existing output is empty: $1"
    fi
}

atomic_text_from_file() {
    local source_file="$1" target="$2" temporary
    prepare_output_file "${target}"
    refuse_bad_existing_output "${target}"
    if [[ -s "${target}" ]]; then
        printf 'retained existing inventory: %s\n' "${target}"
        return 0
    fi
    temporary="${target}.tmp.$$"
    assert_under_staging "${temporary}"
    cp -- "${source_file}" "${temporary}"
    mv -- "${temporary}" "${target}"
}

assert_safe_staging_root
[[ "${PAPER1_AUDIT_ONLY}" == 0 || "${PAPER1_AUDIT_ONLY}" == 1 ]] \
    || die "PAPER1_AUDIT_ONLY must be 0 or 1"
export PAPER1_AUDIT_ONLY
export -f die is_within assert_under_staging prepare_output_dir
export -f prepare_output_file refuse_bad_existing_output
