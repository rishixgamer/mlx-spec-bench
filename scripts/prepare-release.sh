#!/usr/bin/env bash
# Build and exercise the exact committed source that the reference sweep records.
set -euo pipefail

usage() {
  echo "Usage: $0 OUTPUT_DIR 'actual host load notes'" >&2
  echo "       $0 --smoke-only OUTPUT_DIR" >&2
  exit 2
}

[[ $# == 2 ]] || usage
if [[ $1 == --smoke-only ]]; then
  mode=smoke
  output=$2
else
  mode=reference
  output=$1
  host_notes=$2
  [[ $host_notes =~ [^[:space:]] ]] || usage
fi
[[ -n $output ]] || usage
python_bin=${PYTHON:-python3}
"$python_bin" - <<'PY'
import platform
import sys

if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    sys.exit("Use native arm64 Python on an Apple Silicon Mac.")
if int(platform.mac_ver()[0].split(".")[0]) < 15 or not (3, 11) <= sys.version_info[:2] <= (3, 13):
    sys.exit("Use macOS 15+ and Python 3.11, 3.12, or 3.13. Set PYTHON to select the interpreter.")
PY
if [[ $mode == reference && $(sysctl -n hw.memsize) != 51539607552 ]]; then
  echo "Reference qualification requires a 48 GB Apple Silicon Mac." >&2
  exit 1
fi

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse HEAD)
mkdir -- "$output"  # Use a new directory so earlier evidence survives retries.
output=$(cd "$output" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/source" "$output/dist"
git -C "$repo" archive "$commit" | tar -x -C "$work/source"
cd "$work/source"
echo "Preparing commit $commit in $output"
"$python_bin" -m venv "$work/env"
source "$work/env/bin/activate"
python -m pip install build
python -m build --outdir "$output/dist"
wheels=("$output"/dist/*.whl)
python -m pip install "${wheels[0]}[metal,test]"
export GITHUB_SHA="$commit"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/smoke_metal.py "$output/smoke"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python examples/fixture.py
if [[ $mode == smoke ]]; then
  exit 0
fi

echo "Running the 96 reference trials. Raw evidence will be under $output/runs."
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
run_dir=$(mlx-bench run examples/reference-sweep.yaml --source-commit "$commit" \
  --host-notes "$host_notes" --output-dir "$output/runs")
mlx-bench report "$run_dir" --output "$output/dist/report.md"
mlx-bench qualify "$run_dir" --output "$output/dist/reference-v1.tar.gz"
cp "$output/smoke/smoke-gate.json" "$output/dist/"
cd "$output/dist"
shasum -a 256 ./*.whl ./*.tar.gz report.md smoke-gate.json > SHA256SUMS
shasum -a 256 -c SHA256SUMS
echo "Qualified release files for $commit are in $output/dist"
