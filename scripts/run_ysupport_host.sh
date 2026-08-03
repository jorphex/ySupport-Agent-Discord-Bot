#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing project Python environment: ${PYTHON}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
: "${TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE:?TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE must point to service-owned auth}"

exec "${PYTHON}" ysupport.py
