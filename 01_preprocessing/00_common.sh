#!/usr/bin/env bash
set -euo pipefail

# INPUT
#   Environment variables shared by the Paper 1 preprocessing scripts.
# OUTPUT
#   No scientific data. This file only defines safety and utility functions.
# ACTION
#   Resolve the staging root, reject legacy public-product destinations, check
#   required commands, and provide atomic-write helpers.

PAPER1_PUBLIC_ROOT="/mnt/soclim0/public_data/weiji"
COMMON_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PAPER1_CODE_CLEANED_ROOT=$(cd -- "${COMMON_SCRIPT_DIR}/../.." && pwd)
PAPER1_PROJECT_ROOT=$(cd -- "${COMMON_SCRIPT_DIR}/.." && pwd)
PAPER1_RUNTIME_ROOT="${PAPER1_PROJECT_ROOT}/runtime"
PAPER1_DERIVED_ROOT="${PAPER1_DERIVED_ROOT:-${PAPER1_RUNTIME_ROOT}}"
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

assert_safe_staging_root() {
    command -v realpath >/dev/null 2>&1 || die "required command is not available: realpath"
    [[ -n "${PAPER1_DERIVED_ROOT}" ]] || die "PAPER1_DERIVED_ROOT is empty"
    [[ "${PAPER1_DERIVED_ROOT}" = /* ]] || die "PAPER1_DERIVED_ROOT must be absolute"
    PAPER1_PUBLIC_ROOT=$(realpath -m -- "${PAPER1_PUBLIC_ROOT}")
    PAPER1_CODE_CLEANED_ROOT=$(realpath -m -- "${PAPER1_CODE_CLEANED_ROOT}")
    PAPER1_PROJECT_ROOT=$(realpath -m -- "${PAPER1_PROJECT_ROOT}")
    PAPER1_RUNTIME_ROOT=$(realpath -m -- "${PAPER1_RUNTIME_ROOT}")
    PAPER1_DERIVED_ROOT=$(realpath -m -- "${PAPER1_DERIVED_ROOT}")
    export PAPER1_PUBLIC_ROOT PAPER1_CODE_CLEANED_ROOT PAPER1_PROJECT_ROOT
    export PAPER1_RUNTIME_ROOT PAPER1_DERIVED_ROOT
    [[ "${PAPER1_DERIVED_ROOT}" != "/" ]] || die "refusing filesystem root"
    case "${PAPER1_RUNTIME_ROOT}" in
        "${PAPER1_PROJECT_ROOT}"/*) ;;
        *) die "runtime resolves outside the checked-out code directory: ${PAPER1_RUNTIME_ROOT}" ;;
    esac
    case "${PAPER1_DERIVED_ROOT}" in
        "${PAPER1_RUNTIME_ROOT}"|"${PAPER1_RUNTIME_ROOT}/"*) ;;
        *) die "PAPER1_DERIVED_ROOT must be Paper1/runtime or its descendant: ${PAPER1_DERIVED_ROOT}" ;;
    esac

    # Raw archives and accepted public products are read-only. Runtime output
    # belongs under this checkout's code tree, never under either root.
    local protected
    for protected in \
        "/mnt/backup_ETH" \
        "${PAPER1_PUBLIC_ROOT}"; do
        case "${PAPER1_DERIVED_ROOT}/" in
            "${protected}/"|"${protected}/"*)
                die "refusing protected legacy destination: ${PAPER1_DERIVED_ROOT}"
                ;;
        esac
    done
    assert_not_protected_destination "${PAPER1_DERIVED_ROOT}"
}

assert_not_protected_destination() {
    local candidate="$1"
    local resolved protected
    resolved=$(realpath -m -- "${candidate}")
    for protected in \
        "/mnt/backup_ETH" \
        "${PAPER1_PUBLIC_ROOT}"; do
        protected=$(realpath -m -- "${protected}")
        case "${resolved}/" in
            "${protected}/"|"${protected}/"*)
                die "refusing protected legacy destination: ${candidate} -> ${resolved}"
                ;;
        esac
    done
}

assert_under_staging() {
    local candidate="$1"
    local resolved
    resolved=$(realpath -m -- "${candidate}")
    case "${resolved}" in
        "${PAPER1_DERIVED_ROOT}"|"${PAPER1_DERIVED_ROOT}/"*) ;;
        *) die "output escapes PAPER1_DERIVED_ROOT: ${candidate} -> ${resolved}" ;;
    esac
    case "${resolved}" in
        "${PAPER1_PROJECT_ROOT}"/*) ;;
        *) die "output escapes the checked-out code directory: ${candidate} -> ${resolved}" ;;
    esac
    case "${resolved}" in
        "${PAPER1_RUNTIME_ROOT}"|"${PAPER1_RUNTIME_ROOT}/"*) ;;
        *) die "output escapes Paper1/runtime: ${candidate} -> ${resolved}" ;;
    esac
    assert_not_protected_destination "${resolved}"
}

prepare_output_dir() {
    local directory="$1"
    assert_under_staging "${directory}"
    mkdir -p "${directory}"
    # Re-resolve after mkdir so a pre-existing symlink in the final component
    # cannot turn a lexically safe staging path into a legacy-product path.
    assert_under_staging "${directory}"
}

prepare_output_file() {
    local target="$1"
    assert_under_staging "${target}"
    prepare_output_dir "$(dirname -- "${target}")"
    # Check the final filename as well as its parent. This rejects an existing
    # target symlink before validation, replacement, or creation is attempted.
    assert_under_staging "${target}"
}

refuse_bad_existing_output() {
    local target="$1"
    if [[ -e "${target}" && ! -s "${target}" ]]; then
        die "existing output is empty; inspect it manually: ${target}"
    fi
}

atomic_text_from_file() {
    local source_file="$1" target="$2"
    prepare_output_dir "$(dirname "${target}")"
    refuse_bad_existing_output "${target}"
    if [[ -s "${target}" ]]; then
        printf 'retained existing inventory: %s\n' "${target}"
        return 0
    fi
    local temporary="${target}.tmp.$$"
    assert_under_staging "${temporary}"
    cp -- "${source_file}" "${temporary}"
    mv -- "${temporary}" "${target}"
}

assert_safe_staging_root
[[ "${PAPER1_AUDIT_ONLY}" == 0 || "${PAPER1_AUDIT_ONLY}" == 1 ]] \
    || die "PAPER1_AUDIT_ONLY must be 0 or 1"
export PAPER1_AUDIT_ONLY
export -f die assert_not_protected_destination assert_under_staging
export -f prepare_output_dir prepare_output_file refuse_bad_existing_output
