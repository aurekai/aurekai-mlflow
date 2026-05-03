#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rm -rf artifacts mlruns
mkdir -p artifacts
python3 scripts/run_pipeline.py
for f in doctor-report.json manifest-verify.json model-memory-pack.json sae-audit.json semantic-cache-bench.json proof-bundle.json release-gate.json; do
  test -f "artifacts/$f"
done
echo "[validate-scripts] PASS"
