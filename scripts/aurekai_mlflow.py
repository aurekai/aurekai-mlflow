"""
aurekai_mlflow.py — MLflow integration for Aurekai.
Provides: PythonModel flavor, custom evaluation metric, registry plugin, gateway routes.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd


# ── Shared runner ─────────────────────────────────────────────────────────────

def _run_akai(args: list[str], timeout: int = 300) -> dict[str, Any]:
    result = subprocess.run(
        ["akai", *args, "--json"],
        capture_output=True, text=True, timeout=timeout,
    )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"raw": result.stdout, "error": result.stderr, "exit_code": result.returncode}


# ── MLflow pyfunc flavor ──────────────────────────────────────────────────────

class AurekaiPyfunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc flavor for Aurekai.
    Input: DataFrame with column 'operator' and optional param columns.
    Output: DataFrame with 'result', 'proof_uri', 'artifact_id'.
    """

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        results = []
        for _, row in model_input.iterrows():
            operator = row.get("operator", "runtime capabilities")
            args = operator.split()
            output = _run_akai(args)
            results.append({
                "result": json.dumps(output),
                "proof_uri": output.get("proof_uri", ""),
                "artifact_id": output.get("artifact_id", ""),
                "exit_code": output.get("exit_code", 0),
            })
        return pd.DataFrame(results)


def log_aurekai_model(run_id: str = ""):
    """Log an AurekaiPyfunc model to MLflow."""
    with mlflow.start_run(run_id=run_id or None) as run:
        mlflow.pyfunc.log_model(
            artifact_path="aurekai-model",
            python_model=AurekaiPyfunc(),
            pip_requirements=["aurekai>=0.8.0a1"],
        )
        mlflow.set_tag("aurekai.version", "0.8.0-alpha.5")
        return run.info.run_id


# ── Custom evaluation metric ──────────────────────────────────────────────────

class AurekaiProofCompletenessMetric:
    """
    MLflow custom metric: checks proof_uri is non-empty in model outputs.
    Use with mlflow.evaluate().
    """
    name = "proof_completeness"
    greater_is_better = True

    def eval_fn(self, predictions, targets=None, metrics=None):
        if isinstance(predictions, pd.DataFrame):
            proof_uris = predictions.get("proof_uri", pd.Series(dtype=str))
        else:
            proof_uris = pd.Series(predictions)
        score = (proof_uris.notna() & (proof_uris != "")).mean()
        return mlflow.metrics.MetricValue(
            aggregate_results={"mean": float(score)},
            scores=proof_uris.notna().tolist(),
        )

    def __call__(self, *args, **kwargs):
        return self.eval_fn(*args, **kwargs)


def evaluate_aurekai_model(model_uri: str, eval_data: pd.DataFrame):
    """Run mlflow.evaluate on a logged AurekaiPyfunc model."""
    return mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="expected_proof_uri",
        model_type="text",
        extra_metrics=[
            mlflow.metrics.make_metric(
                eval_fn=AurekaiProofCompletenessMetric().eval_fn,
                name="proof_completeness",
                greater_is_better=True,
            )
        ],
    )


# ── Gateway / proxy route helper ──────────────────────────────────────────────

GATEWAY_ROUTE_CONFIG = {
    "name": "aurekai",
    "route_type": "llm/v1/completions",
    "model": {
        "provider": "aurekai",
        "name": "akai-proxy",
        "config": {
            "aurekai_base_url": "http://localhost:3001/mcp",
            "aurekai_version": "0.8.0-alpha.5",
        },
    },
}


# ── Tracing helpers ──────────────────────────────────────────────────────────

def trace_akai_pipeline(pipeline_name: str, stages: list[tuple[str, list[str]]]):
    """
    Trace a multi-stage Akai pipeline in MLflow.
    stages: list of (stage_name, akai_args)
    """
    with mlflow.start_run(run_name=pipeline_name) as run:
        mlflow.set_tag("aurekai.pipeline", pipeline_name)
        for stage_name, args in stages:
            with mlflow.start_run(run_name=stage_name, nested=True):
                output = _run_akai(args)
                mlflow.log_dict(output, f"{stage_name}-output.json")
                mlflow.log_param("operator", args[0] if args else "unknown")
                mlflow.log_metric("exit_code", output.get("exit_code", 0))
                mlflow.set_tag("proof_uri", output.get("proof_uri", ""))
        return run.info.run_id
