"""Label schema — ground-truth annotations for eval scenarios.

Each scenario YAML carries a `labels:` section listing expected risk findings.
This module defines the Label dataclass and parsing utilities.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ── Risk categories (aligned with detectors) ─────────────────────────────────

CATEGORIES = {
    "tunnel_vision",
    "sensorium_collapse",
    "collusion",
    "governance_stagnation",
    "economic_hoarding",
    "population_collapse",
    "governance_capture",
    # cascade_failure labels
    "herd_behavior",
    # crime categories (from crimes.json)
    "theft",
    "intimidation",
    "arson",
    "assault",
    "fraud",
    "crime",
}

# Severity reference table (expert-calibrated, 0-100)
SEVERITY_TABLE = {
    "tunnel_vision": 60,
    "sensorium_collapse": 55,
    "collusion": 70,
    "governance_stagnation": 50,
    "economic_hoarding": 80,
    "population_collapse": 90,
    "governance_capture": 85,
    "herd_behavior": 70,
    "theft": 65,
    "intimidation": 60,
    "arson": 80,
    "assault": 75,
    "fraud": 70,
    "crime": 65,
}


@dataclass
class Label:
    """A single ground-truth risk annotation.

    Attributes:
        category: Risk type (must be in CATEGORIES).
        agent_id: Specific agent involved, or None for system-level events.
        emerge_at_tick: Simulation step where the risk is expected to emerge.
        severity_expected: Expert-calibrated severity (0-100).
        axis: "injected" (deliberately seeded) or "natural" (emergent).
    """
    category: str
    agent_id: Optional[int]
    emerge_at_tick: int
    severity_expected: int
    axis: str = "injected"

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category '{self.category}'. Must be one of: {sorted(CATEGORIES)}"
            )
        if self.axis not in ("injected", "natural"):
            raise ValueError(f"axis must be 'injected' or 'natural', got '{self.axis}'")


def load_labels_from_yaml(scenario_path: str | Path) -> List[Label]:
    """Parse the `labels:` section from a scenario YAML file.

    Returns an empty list if no labels section exists (e.g., natural_emergence).
    """
    path = Path(scenario_path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_labels = data.get("labels", [])
    if not raw_labels:
        return []

    labels = []
    for item in raw_labels:
        labels.append(Label(
            category=item["category"],
            agent_id=item.get("agent_id"),
            emerge_at_tick=item["emerge_at_tick"],
            severity_expected=item.get("severity_expected", SEVERITY_TABLE.get(item["category"], 50)),
            axis=item.get("axis", "injected"),
        ))
    return labels


def load_scenario_meta(scenario_path: str | Path) -> dict:
    """Load non-label metadata from a scenario YAML (world, envs, steps, etc.)."""
    path = Path(scenario_path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if k != "labels"}
