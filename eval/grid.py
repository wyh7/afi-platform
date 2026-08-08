"""Grid — L2 parameterized expansion of L1 scenario templates.

Generates a run specification grid: 6 injected templates × N models × M seeds.
Each RunSpec defines everything needed to launch and score one experiment run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# L1 templates (injected)
INJECTED_TEMPLATES = [
    "single_agent_drift",
    "collusion_formation",
    "governance_stagnation",
    "economic_collapse",
    "population_collapse",
    "governance_capture",
]

# Natural emergence (no labels, discovery-only)
NATURAL_TEMPLATE = "natural_emergence"

# Model grid (extend as API keys become available)
DEFAULT_MODELS = [
    "qwen2.5-7b",             # local CPU via HF server (always available)
    "qwen-plus",              # Alibaba Bailian — BAILIAN_API_KEY
    "gemini-2.5-flash",       # Google via UniAPI — JINGZHE_API_KEY
]

# Full model catalog (use --models flag to select subset)
ALL_MODELS = [
    "qwen2.5-7b",             # local CPU
    "qwen-plus",              # Alibaba Qwen API
    "qwen-turbo",             # Alibaba Qwen (faster/cheaper)
    "gemini-2.5-flash",       # Google Gemini
    "gpt-4.1",                # OpenAI via Yunhe — YUNHE_API_KEY
    "claude-sonnet-4-6",      # Anthropic via JD Cloud — JD_API_KEY
    "deepseek-v3",            # DeepSeek via Bailian — BAILIAN_API_KEY
]

# Model → API key env var mapping (for documentation + key-check warnings)
MODEL_KEY_MAP = {
    "qwen2.5-7b":          None,               # local, no key needed
    "qwen-plus":           "BAILIAN_API_KEY",
    "qwen-turbo":          "BAILIAN_API_KEY",
    "deepseek-v3":         "BAILIAN_API_KEY",
    "gemini-2.5-flash":    "JINGZHE_API_KEY",
    "gpt-4.1":             "YUNHE_API_KEY",
    "claude-sonnet-4-6":   "JD_API_KEY",
}

# Seeds for statistical robustness
DEFAULT_SEEDS = [0, 1, 2]

# Agent count (fixed at 5 due to EW_PROFILES constraint)
AGENT_COUNT = 5


@dataclass
class RunSpec:
    """Specification for a single eval run.

    Attributes:
        scenario_name: Template name (e.g., "single_agent_drift").
        scenario_path: Path to the scenario YAML.
        model: LLM model identifier.
        seed: Random seed for reproducibility.
        run_id: Unique identifier for the run directory.
        is_natural: True if this is the natural emergence control.
    """
    scenario_name: str
    scenario_path: Path
    model: str
    seed: int
    run_id: str = ""
    is_natural: bool = False

    def __post_init__(self):
        if not self.run_id:
            # Deterministic run_id from (scenario, model, seed)
            key = f"{self.scenario_name}_{self.model}_{self.seed}"
            short_hash = hashlib.md5(key.encode()).hexdigest()[:6]
            self.run_id = f"eval_{self.scenario_name}_{self.model.replace('.', '')}_{self.seed}_{short_hash}"

    @property
    def run_dir(self) -> str:
        """Relative path for the run output directory."""
        return f"runs/eval/{self.run_id}"


def generate_grid(
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    scenarios: Optional[List[str]] = None,
    include_natural: bool = True,
) -> List[RunSpec]:
    """Generate the full parameterized run grid.

    Args:
        models: List of model identifiers. Defaults to DEFAULT_MODELS.
        seeds: List of random seeds. Defaults to DEFAULT_SEEDS.
        scenarios: Subset of INJECTED_TEMPLATES to include. None = all 6.
        include_natural: Whether to include natural_emergence runs.

    Returns:
        List of RunSpec objects defining all runs to execute.
    """
    models = models or DEFAULT_MODELS
    seeds = seeds or DEFAULT_SEEDS
    templates = scenarios or INJECTED_TEMPLATES

    grid: List[RunSpec] = []

    # Injected scenarios: templates × models × seeds
    for template in templates:
        yaml_path = EVAL_SCENARIOS_DIR / f"{template}.yaml"
        if not yaml_path.exists():
            continue
        for model in models:
            for seed in seeds:
                grid.append(RunSpec(
                    scenario_name=template,
                    scenario_path=yaml_path,
                    model=model,
                    seed=seed,
                    is_natural=False,
                ))

    # Natural emergence: models × seeds (no injection)
    if include_natural:
        natural_path = EVAL_SCENARIOS_DIR / f"{NATURAL_TEMPLATE}.yaml"
        if natural_path.exists():
            for model in models:
                for seed in seeds:
                    grid.append(RunSpec(
                        scenario_name=NATURAL_TEMPLATE,
                        scenario_path=natural_path,
                        model=model,
                        seed=seed,
                        is_natural=True,
                    ))

    return grid


def grid_summary(grid: List[RunSpec]) -> dict:
    """Summarize the grid for display."""
    models = sorted(set(s.model for s in grid))
    scenarios = sorted(set(s.scenario_name for s in grid))
    seeds = sorted(set(s.seed for s in grid))
    injected = [s for s in grid if not s.is_natural]
    natural = [s for s in grid if s.is_natural]

    return {
        "total_runs": len(grid),
        "injected_runs": len(injected),
        "natural_runs": len(natural),
        "models": models,
        "scenarios": scenarios,
        "seeds": seeds,
        "formula": f"{len([s for s in scenarios if s != NATURAL_TEMPLATE])} templates × {len(models)} models × {len(seeds)} seeds = {len(injected)} injected + {len(natural)} natural",
    }
