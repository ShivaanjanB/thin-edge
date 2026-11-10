"""Simulated matches shared by the model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic(n_matches=400, seed=0, start="2020-01-01", year_drift=0.0):
    """Simulated matches with Poisson ball-by-ball scoring.

    Each side has a hidden scoring rate, the chase target is the first total + 1, and the result is
    whether the chase reaches it. So the state (runs needed, balls left) genuinely predicts the
    result, and a model can only do well by reading it.

    ``year_drift`` raises every team's scoring rate by that fraction per calendar year after the
    first. Results stay equally balanced (both sides speed up), but a par curve learned earlier goes
    stale. That is the mechanism the walk-forward experiment is meant to detect.
    """
    rng = np.random.default_rng(seed)
    first_year = pd.Timestamp(start).year
    rows = []
    for m in range(n_matches):
        date = pd.Timestamp(start) + pd.Timedelta(days=m)
        era = 1.0 + year_drift * (date.year - first_year)
        mu1, mu2 = rng.uniform(1.0, 1.9, size=2) * era
        paths = {
            1: np.concatenate([[0], np.cumsum(rng.poisson(mu1, 120))]),
            2: np.concatenate([[0], np.cumsum(rng.poisson(mu2, 120))]),
        }
        target = int(paths[1][-1]) + 1
        chase_won = int(paths[2][-1] >= target)
        for innings, path in paths.items():
            for ball in range(0, 120, 6):
                rows.append({
                    "match_id": m, "innings_no": innings, "source": "ipl", "date": date,
                    "balls_per_over": 6, "max_balls": 120,
                    "target_runs": float(target) if innings == 2 else np.nan,
                    "score_pre": int(path[ball]),
                    "wickets_pre": min(9, int(ball / 20 + rng.integers(0, 2))),
                    "legal_balls_pre": ball, "balls_remaining_pre": 120 - ball,
                    "y": chase_won if innings == 2 else 1 - chase_won,
                })
    return pd.DataFrame(rows)
