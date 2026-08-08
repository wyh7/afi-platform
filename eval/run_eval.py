"""Run Eval — execute the eval grid and collect scored results.

Orchestrates: generate grid → run each spec → detect_all → score → aggregate.
Supports partial execution (skip already-completed runs) and parallel batching.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from eval.grid import RunSpec, generate_grid, grid_summary
from eval.labels import load_labels_from_yaml
from eval.findings import detect_all, Finding
from eval.scoring import ScoreCard, AggregatedScore, score_run, aggregate_scores
from eval.verifier import verify_labels
from eval.diff import detect_naive, compute_delta_recall


@dataclass
class RunResult:
    """Result of a single eval run (spec + findings + score)."""
    spec: RunSpec
    findings: List[Finding] = field(default_factory=list)
    naive_findings: List[Finding] = field(default_factory=list)
    score: Optional[ScoreCard] = None
    naive_score: Optional[ScoreCard] = None
    delta_recall: float = 0.0
    status: str = "pending"  # pending | running | completed | failed | skipped
    error: str = ""
    duration_sec: float = 0.0


@dataclass
class EvalReport:
    """Aggregated report across all runs in the grid."""
    results: List[RunResult] = field(default_factory=list)
    aggregated: List[AggregatedScore] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ── Run execution ─────────────────────────────────────────────────────────────


def _run_experiment(spec: RunSpec, base_dir: Path, timeout: int = 7200) -> str:
    """Launch a single AS2 experiment run via run_inprocess.py.

    Uses afi.world.scenario.write_config() to generate init_config.json +
    steps.yaml from the scenario YAML, then calls run_inprocess.py.

    Returns the run_dir path on success, raises on failure.
    """
    import sys
    run_dir = base_dir / spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate init_config.json + steps.yaml from scenario YAML
    #    write_config() uses afi.world EW seed data (constitution/profiles/etc.)
    sys.path.insert(0, str(base_dir))
    try:
        from afi.world.scenario import load_scenario, write_config
        scenario = load_scenario(spec.scenario_path)
        cfg_path, steps_path = write_config(scenario, run_dir / "_config")
    except Exception as e:
        raise RuntimeError(f"Failed to build config from scenario: {e}") from e

    # 2. Use the same python that's running this process (avoids PATH guessing)
    import sys as _sys
    python = _sys.executable

    # 3. Build environment — inject model + optional local API base
    env_vars = {
        **os.environ,
        "AGENTSOCIETY_LLM_MODEL": spec.model,
        "AGENTSOCIETY_LLM_REQUEST_TIMEOUT": "600",
    }
    # LLM config — default to local HF server if not set
    local_key = "local-key"
    local_base = "http://127.0.0.1:8007/v1"
    if not env_vars.get("AGENTSOCIETY_LLM_API_KEY"):
        env_vars["AGENTSOCIETY_LLM_API_KEY"] = local_key
    if not env_vars.get("AGENTSOCIETY_LLM_API_BASE"):
        env_vars["AGENTSOCIETY_LLM_API_BASE"] = local_base
    # Coder LLM (used by AS2 router_codegen._generate_observe_code)
    # must also point to local server, otherwise falls back to external API
    if not env_vars.get("AGENTSOCIETY_CODER_LLM_API_KEY"):
        env_vars["AGENTSOCIETY_CODER_LLM_API_KEY"] = env_vars["AGENTSOCIETY_LLM_API_KEY"]
    if not env_vars.get("AGENTSOCIETY_CODER_LLM_API_BASE"):
        env_vars["AGENTSOCIETY_CODER_LLM_API_BASE"] = env_vars["AGENTSOCIETY_LLM_API_BASE"]
    if not env_vars.get("AGENTSOCIETY_CODER_LLM_MODEL"):
        env_vars["AGENTSOCIETY_CODER_LLM_MODEL"] = env_vars.get("AGENTSOCIETY_LLM_MODEL", spec.model)
    # WORKSPACE_PATH: AS2 needs this to find custom envs in afi-platform
    if not env_vars.get("WORKSPACE_PATH"):
        env_vars["WORKSPACE_PATH"] = str(base_dir)

    # 4. Run
    cmd = [
        python, "run_inprocess.py",
        "--config", str(cfg_path),
        "--steps", str(steps_path),
        "--run-dir", str(run_dir),
        "--log-level", "WARNING",
    ]

    log_path = run_dir / "run.log"
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                cwd=str(base_dir),
                env=env_vars,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8")[-1000:]
            raise RuntimeError(f"Run failed (rc={proc.returncode}):\n{tail}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Run timed out after {timeout}s")

    return str(run_dir)


def _score_single(spec: RunSpec, run_dir: Path) -> RunResult:
    """Score a completed run: detect_all → verify → score → diff."""
    result = RunResult(spec=spec, status="completed")

    # Load labels from scenario YAML
    labels = load_labels_from_yaml(spec.scenario_path)

    # Run full detection suite
    result.findings = detect_all(run_dir, include_collude=True)

    # Run naive baseline
    result.naive_findings = detect_naive(run_dir)

    # Verify labels (skip for natural emergence)
    if labels and not spec.is_natural:
        valid_labels, dropped = verify_labels(run_dir, labels, spec.scenario_name)
        seed_dropped = len(dropped)
    else:
        valid_labels = labels
        seed_dropped = 0

    # Score
    result.score = score_run(result.findings, valid_labels, seed_dropped)
    result.naive_score = score_run(result.naive_findings, valid_labels, seed_dropped)

    # Delta recall
    if valid_labels:
        diff = compute_delta_recall(result.findings, result.naive_findings, valid_labels)
        result.delta_recall = diff["delta_recall"]

    return result


# ── Public API ────────────────────────────────────────────────────────────────


def check_api_keys(models: List[str]) -> dict:
    """Check which API keys are available for the requested models.

    Returns a dict:
      {
        "ok":      [list of models with valid key],
        "missing": [list of models missing their required key],
        "local":   [list of models that need no key],
        "warnings": [human-readable warning strings],
      }

    Does NOT make any network requests — only checks env vars are non-empty.
    """
    from eval.grid import MODEL_KEY_MAP

    ok, missing, local, warnings = [], [], [], []

    for model in models:
        key_var = MODEL_KEY_MAP.get(model)
        if key_var is None:
            local.append(model)
            continue
        val = os.environ.get(key_var, "").strip()
        if val:
            ok.append(model)
        else:
            missing.append(model)
            warnings.append(
                f"Model '{model}' requires {key_var} but it is not set in environment. "
                f"Set it in .env or export it before running the grid."
            )

    return {"ok": ok, "missing": missing, "local": local, "warnings": warnings}


def print_report(report: EvalReport) -> None:
    """Score an already-completed run without re-executing it.

    Useful for scoring existing runs (e.g., b8_qwen_cooperative) against
    eval scenario labels, or for testing the scoring pipeline offline.
    """
    run_dir = Path(run_dir)
    scenario_path = Path(scenario_path)

    spec = RunSpec(
        scenario_name=scenario_path.stem,
        scenario_path=scenario_path,
        model="unknown",
        seed=0,
    )

    return _score_single(spec, run_dir)


def run_grid_offline(base_dir: str | Path) -> EvalReport:
    """Score all existing eval runs in base_dir/runs/eval/ without launching new ones.

    Scans for completed run directories and scores them.
    """
    base_dir = Path(base_dir)
    eval_runs_dir = base_dir / "runs" / "eval"
    report = EvalReport()

    if not eval_runs_dir.is_dir():
        report.summary = {"error": "No eval runs directory found"}
        return report

    # Find all completed run dirs
    for run_dir in sorted(eval_runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Try to identify scenario from run_dir name
        scenario_name = None
        for template in ["single_agent_drift", "collusion_formation",
                         "governance_stagnation", "economic_collapse",
                         "population_collapse", "governance_capture",
                         "natural_emergence"]:
            if template in run_dir.name:
                scenario_name = template
                break

        if scenario_name is None:
            continue

        scenario_path = Path(__file__).parent / "scenarios" / f"{scenario_name}.yaml"
        if not scenario_path.exists():
            continue

        try:
            result = score_existing_run(run_dir, scenario_path)
            report.results.append(result)
        except Exception as e:
            report.results.append(RunResult(
                spec=RunSpec(scenario_name=scenario_name, scenario_path=scenario_path,
                             model="unknown", seed=0),
                status="failed",
                error=str(e),
            ))

    return report


def run_eval(
    base_dir: str | Path,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    scenarios: Optional[List[str]] = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> EvalReport:
    """Execute the full eval pipeline: generate grid → run → score → aggregate.

    Args:
        base_dir: afi-platform root directory.
        models: Model list (default: all configured).
        seeds: Seed list (default: [0,1,2]).
        scenarios: Scenario subset (default: all 6 injected + natural).
        skip_existing: Skip runs whose run_dir already exists.
        dry_run: If True, only generate grid + score existing, don't launch new runs.

    Returns:
        EvalReport with per-run results and aggregated scores.
    """
    base_dir = Path(base_dir)
    grid = generate_grid(models=models, seeds=seeds, scenarios=scenarios)
    report = EvalReport(summary=grid_summary(grid))

    for spec in grid:
        run_dir = base_dir / spec.run_dir

        # Check if already completed
        if skip_existing and run_dir.is_dir() and (run_dir / "trace").is_dir():
            try:
                result = _score_single(spec, run_dir)
                result.status = "completed"
                report.results.append(result)
                continue
            except Exception as e:
                report.results.append(RunResult(
                    spec=spec, status="failed", error=str(e)
                ))
                continue

        # Dry run: skip execution
        if dry_run:
            report.results.append(RunResult(spec=spec, status="skipped"))
            continue

        # Execute the run
        t0 = time.time()
        try:
            _run_experiment(spec, base_dir)
            result = _score_single(spec, run_dir)
            result.duration_sec = time.time() - t0
            report.results.append(result)
        except Exception as e:
            report.results.append(RunResult(
                spec=spec, status="failed", error=str(e),
                duration_sec=time.time() - t0,
            ))

    # Aggregate by (scenario, model)
    completed = [r for r in report.results if r.status == "completed" and r.score]
    scenario_model_groups: dict = {}
    for r in completed:
        key = (r.spec.scenario_name, r.spec.model)
        scenario_model_groups.setdefault(key, []).append(r.score)

    for (scenario, model), scores in scenario_model_groups.items():
        report.aggregated.append(aggregate_scores(scores, scenario, model))

    return report


def export_csv(report: EvalReport, output_path: str | Path):
    """Export per-run and aggregated results to a CSV file."""
    import csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "scenario", "model", "seed", "status",
            "precision", "recall", "f1",
            "latency_median", "severity_mae",
            "tp", "fp", "fn", "seed_dropped",
            "naive_recall", "delta_recall",
            "duration_sec",
        ])
        for r in report.results:
            sc = r.score
            ns = r.naive_score
            writer.writerow([
                r.spec.scenario_name, r.spec.model, r.spec.seed, r.status,
                sc.precision if sc else "", sc.recall if sc else "",
                sc.f1 if sc else "",
                sc.latency_median if sc else "", sc.severity_mae if sc else "",
                sc.tp if sc else "", sc.fp if sc else "",
                sc.fn if sc else "", sc.seed_dropped if sc else "",
                ns.recall if ns else "", r.delta_recall,
                round(r.duration_sec, 1),
            ])

    # Aggregated sheet
    agg_path = output_path.with_name(output_path.stem + "_agg.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "model", "n_runs",
            "precision_mean", "precision_ci95",
            "recall_mean", "recall_ci95",
            "f1_mean", "f1_ci95",
            "latency_mean", "severity_mae_mean", "seed_drop_rate",
        ])
        for a in report.aggregated:
            writer.writerow([
                a.scenario, a.model, a.n_runs,
                a.precision_mean, a.precision_ci95,
                a.recall_mean, a.recall_ci95,
                a.f1_mean, a.f1_ci95,
                a.latency_mean, a.severity_mae_mean, a.seed_drop_rate,
            ])

    return output_path, agg_path


def print_report(report: EvalReport):
    """Print a summary of the eval report to stdout."""
    print("=" * 70)
    print("EVAL SUITE REPORT")
    print("=" * 70)
    print(f"Grid: {report.summary}")
    print()

    completed = [r for r in report.results if r.status == "completed"]
    failed = [r for r in report.results if r.status == "failed"]
    skipped = [r for r in report.results if r.status == "skipped"]

    print(f"Completed: {len(completed)} | Failed: {len(failed)} | Skipped: {len(skipped)}")
    print()

    if completed:
        print(f"{'Scenario':<25} {'Model':<18} {'P':>6} {'R':>6} {'F1':>6} {'Lat':>5} {'ΔR':>6}")
        print("-" * 70)
        for r in completed:
            if r.score:
                print(f"{r.spec.scenario_name:<25} {r.spec.model:<18} "
                      f"{r.score.precision:>6.3f} {r.score.recall:>6.3f} "
                      f"{r.score.f1:>6.3f} {r.score.latency_median:>5.1f} "
                      f"{r.delta_recall:>+6.3f}")

    if report.aggregated:
        print()
        print("AGGREGATED (per scenario × model):")
        print(f"{'Scenario':<25} {'Model':<18} {'R̄':>6} {'±CI95':>7} {'F̄1':>6} {'n':>3}")
        print("-" * 70)
        for a in report.aggregated:
            print(f"{a.scenario:<25} {a.model:<18} "
                  f"{a.recall_mean:>6.3f} {a.recall_ci95:>7.3f} "
                  f"{a.f1_mean:>6.3f} {a.n_runs:>3}")
