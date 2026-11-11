"""Paths and constants shared across the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("THINEDGE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(os.environ.get("THINEDGE_REPORTS_DIR", ROOT / "reports"))

def reports_dir_for(sources: list[str] | None) -> Path:
    """Reports go to reports/all or reports/<a>+<b>, so runs on different league sets coexist."""
    return REPORTS_DIR / ("+".join(sorted(set(sources))) if sources else "all")


CRICSHEET_BASE = "https://cricsheet.org/downloads/"

# Dismissals that are not a bowling-side wicket for credit purposes.
BOWLER_WICKET_KINDS = frozenset(
    {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
)
# "retired hurt" and "retired not out" appear in Cricsheet's wickets list but the batter has not been
# dismissed and the team has not lost a wicket (the batter may return). "retired out" is a dismissal.
NOT_A_DISMISSAL = frozenset({"retired hurt", "retired not out"})

TABLES = ("matches", "innings", "balls", "states", "venues")
