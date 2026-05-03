<p align="center">
  <img src="https://raw.githubusercontent.com/aurekai/aurekai/main/assets/aurekai-logo.svg" alt="Aurekai" width="520" />
</p>

# `aurekai-mlflow` · v0.8.0-alpha.5

Official MLflow integration for Aurekai — `pyfunc` model flavor, proof completeness evaluation metric, pipeline tracing, gateway route config.

## Features

| Feature | Description |
|---|---|
| `AurekaiPyfunc` | MLflow pyfunc flavor — runs any Akai operator via `predict()` |
| `AurekaiProofCompletenessMetric` | Custom metric for `mlflow.evaluate()` — checks proof_uri completeness |
| `trace_akai_pipeline()` | Multi-stage pipeline tracing with nested MLflow runs |
| `log_aurekai_model()` | Log an AurekaiPyfunc model to the MLflow registry |
| `evaluate_aurekai_model()` | Evaluate a logged model against expected proof URIs |
| `GATEWAY_ROUTE_CONFIG` | MLflow Gateway route config for Aurekai MCP proxy |

## Quick Start

```python
from aurekai_mlflow import AurekaiPyfunc, log_aurekai_model, trace_akai_pipeline

# Log model
run_id = log_aurekai_model()

# Trace pipeline
trace_akai_pipeline("audio-to-brief", [
    ("transcribe", ["transcribe", "audio", "--input", "audio.wav"]),
    ("clean",      ["transcript", "clean", "--id", "latest"]),
    ("brief",      ["brief", "generate", "--artifact", "latest"]),
    ("proof",      ["proof", "bundle"]),
])
```

## MLflow Evaluate

```python
import pandas as pd
from aurekai_mlflow import evaluate_aurekai_model

eval_data = pd.DataFrame({"operator": ["proof bundle", "doctor --deep"]})
results = evaluate_aurekai_model(f"runs:/{run_id}/aurekai-model", eval_data)
print(results.metrics["proof_completeness/mean"])
```


Aurekai integration surface for MLflow with local-file experiment tracking for manifests, model memory, SAE audits, semantic cache benchmarks, proof bundles, and release gates.

Status: active
Type: data-ml

## Quick Start

python3 -m pip install -r requirements.txt
bash tests/validate-scripts.sh

## Canonical References

- Platform: https://github.com/aurekai/aurekai
- Native runtime: https://github.com/aurekai/native-runtime
- Integration registry: https://github.com/aurekai/aurekai/blob/main/registry/integrations.json
- Ecosystem map: https://github.com/aurekai/aurekai/blob/main/ECOSYSTEM_NAMES.md
