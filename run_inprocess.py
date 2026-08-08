"""run_inprocess.py — Single-process AS2 experiment runner for the eval grid.

Called by eval/run_eval._run_experiment() with:
  python run_inprocess.py \
    --config  <path>/init_config.json \
    --steps   <path>/steps.yaml \
    --run-dir <path>/runs/eval/<run_id> \
    [--log-level WARNING]

Environment variables (set by run_eval._run_experiment):
  AGENTSOCIETY_LLM_MODEL       — model identifier
  AGENTSOCIETY_LLM_API_KEY     — API key (or 'local-key' for local HF server)
  AGENTSOCIETY_LLM_API_BASE    — API base URL (default: http://127.0.0.1:8007/v1)
  WORKSPACE_PATH               — afi-platform root (custom/envs/ hot-loading)

Exit codes:
  0 — success
  1 — AS import error (agentsociety2 not installed)
  2 — config/steps parse error
  3 — runtime error during simulation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single AS2 experiment from pre-built config files."
    )
    p.add_argument("--config",    required=True, help="Path to init_config.json")
    p.add_argument("--steps",     required=True, help="Path to steps.yaml")
    p.add_argument("--run-dir",   required=True, dest="run_dir", help="Output directory")
    p.add_argument("--log-level", default="WARNING", dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.WARNING),
        stream=sys.stderr,
    )


def _load_config(cfg_path: Path) -> dict:
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Cannot parse init_config.json: {e}", file=sys.stderr)
        sys.exit(2)


def _load_steps(steps_path: Path) -> dict:
    try:
        import yaml  # pyyaml required for steps.yaml
        return yaml.safe_load(steps_path.read_text(encoding="utf-8"))
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot parse steps.yaml: {e}", file=sys.stderr)
        sys.exit(2)


def _ensure_agentsociety2() -> None:
    try:
        import importlib
        if importlib.util.find_spec("agentsociety2") is None:
            raise ImportError("not found")
    except Exception:
        print(
            "ERROR: agentsociety2 not importable.\n"
            "  pip mode:      pip install agentsociety2\n"
            "  checkout mode: set AS_HOME and run from the AS venv python",
            file=sys.stderr,
        )
        sys.exit(1)


def _inject_workspace_path() -> None:
    """Ensure WORKSPACE_PATH points to afi-platform root so AS finds custom/envs/."""
    ws = os.environ.get("WORKSPACE_PATH")
    if not ws:
        # Default: directory containing this script = afi-platform root
        os.environ["WORKSPACE_PATH"] = str(Path(__file__).resolve().parent)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    _ensure_agentsociety2()
    _inject_workspace_path()

    cfg_path   = Path(args.config).resolve()
    steps_path = Path(args.steps).resolve()
    run_dir    = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("run_inprocess")
    log.info("config=%s steps=%s run_dir=%s", cfg_path, steps_path, run_dir)

    # Load files
    init_config = _load_config(cfg_path)
    steps_doc   = _load_steps(steps_path)

    # Run via AS2 CLI function (avoids spawning a subprocess from within a subprocess)
    try:
        from agentsociety2.society.cli import run_from_files  # type: ignore
        run_from_files(
            config_path  = str(cfg_path),
            steps_path   = str(steps_path),
            run_dir      = str(run_dir),
            log_level    = args.log_level,
        )
    except ImportError:
        # Older AS2 versions may not have run_from_files — fall back to module invocation
        log.warning("run_from_files not found, falling back to -m agentsociety2.society.cli")
        import subprocess
        cmd = [
            sys.executable, "-m", "agentsociety2.society.cli",
            "--config",    str(cfg_path),
            "--steps",     str(steps_path),
            "--run-dir",   str(run_dir),
            "--log-level", args.log_level,
        ]
        result = subprocess.run(cmd, env=os.environ)
        if result.returncode != 0:
            print(f"ERROR: AS CLI exited with code {result.returncode}", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}", file=sys.stderr)
        log.exception("Simulation error")
        sys.exit(3)

    log.info("Run complete: %s", run_dir)
    print(f"OK run_dir={run_dir}")


if __name__ == "__main__":
    main()
