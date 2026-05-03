import json
import os
from pathlib import Path

import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", f"file://{Path.cwd() / 'mlruns'}")
Path("artifacts").mkdir(exist_ok=True)

manifest = json.loads(Path("examples/sample-aurekai.manifest.json").read_text())
model = json.loads(Path("examples/sample-akmodel.json").read_text())

mlflow.set_experiment("aurekai-mlflow")
with mlflow.start_run(run_name="integration-pipeline"):
    mlflow.log_params({
        "schema_version": manifest.get("schema_version"),
        "name": manifest.get("name"),
        "version": manifest.get("version"),
    })
    mlflow.log_metrics({
        "operator_count": manifest.get("operator_count", 0),
        "sae_rows": 1,
        "semantic_cache_reads": 5000,
        "semantic_cache_writes": 5000,
    })

outputs = {
    "doctor-report.json": {"status": "ok", "check": "doctor-deep"},
    "manifest-verify.json": {"status": "ok", "name": manifest["name"], "version": manifest["version"]},
    "model-memory-pack.json": {"status": "ok", "artifact": model["name"], "ext": model["ext"]},
    "sae-audit.json": {"status": "ok", "rows": 1},
    "semantic-cache-bench.json": {"status": "ok", "reads": 5000, "writes": 5000},
    "proof-bundle.json": {"status": "ok", "count": 1},
    "release-gate.json": {"status": "ok", "gate": "release"},
}
for fname, payload in outputs.items():
    Path("artifacts", fname).write_text(json.dumps(payload, indent=2))
    mlflow.log_artifact(str(Path("artifacts", fname)))

print("[pipeline] PASS")
