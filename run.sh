#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash run.sh [all|S|D|M] [train.py options]

  bash run.sh                            Run S, D, and M sequentially.
  bash run.sh S                          Run only QuatEC-S.
  bash run.sh all --device cuda:0        Select a training device.
  bash run.sh M --seed 1234 --repeats 3   Override training options.

Activate your Python environment and prepare the data before running.
Relative data/output paths are resolved from the repository root.
Set QUATEC_PYTHON to use a different Python executable.
USAGE
}

quatec_selection=all
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  all|S|D|M) quatec_selection="$1"; shift ;;
  ""|--*) ;;
  *) usage >&2; exit 2 ;;
esac

for quatec_arg in "$@"; do
  case "$quatec_arg" in
    --variant|--variant=*)
      printf 'Select the variant with run.sh S, D, M, or all instead of --variant.\n' >&2
      exit 2
      ;;
  esac
done

quatec_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$quatec_root"
quatec_python="${QUATEC_PYTHON:-python}"
if ! command -v "$quatec_python" >/dev/null 2>&1; then
  printf 'Python executable not found: %s. Activate your environment first.\n' "$quatec_python" >&2
  exit 127
fi

quatec_variants=("$quatec_selection")
if [[ "$quatec_selection" == all ]]; then
  quatec_variants=(S D M)
fi

for quatec_variant in "${quatec_variants[@]}"; do
  printf 'Running QuatEC-%s\n' "$quatec_variant"
  "$quatec_python" train.py --variant "$quatec_variant" "$@"
done
