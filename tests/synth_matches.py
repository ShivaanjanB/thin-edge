"""Write a small synthetic league as Cricsheet-style JSON, with known player strengths.

Outcomes are drawn from the same ordered-logistic and wicket model the code fits, so a correct
pipeline should recover the strengths and a leak-free evaluation should reward modelling them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thinedge.bayes import CUT_INIT, RUN_VALUES, outcome_probs


def true_effects(n_players: int, seed: int = 0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    return {
        "bat_w": rng.normal(0, 0.4 * scale, n_players), "bowl_w": rng.normal(0, 0.3 * scale, n_players),
        "bat_r": rng.normal(0, 0.5 * scale, n_players), "bowl_r": rng.normal(0, 0.3 * scale, n_players),
    }


def write_matches(raw_dir: Path, n_matches: int = 60, n_players: int = 24, seed: int = 0, scale: float = 1.0):
    """One match every 20 days from 2022-01-01. Returns the true per-player effects."""
    rng = np.random.default_rng(seed + 1)
    eff = true_effects(n_players, seed, scale)
    names = [f"Player {i:02d}" for i in range(n_players)]
    people = {n: f"id{i:02d}" for i, n in enumerate(names)}
    sides = {"Team A": list(range(0, n_players // 2)), "Team B": list(range(n_players // 2, n_players))}
    raw_dir.mkdir(parents=True, exist_ok=True)
    cut = CUT_INIT

    def innings(bat_team: str, bowl_team: str):
        total, overs = 0, []
        for over in range(20):
            deliveries = []
            for _ in range(6):
                b = int(rng.choice(sides[bat_team])); w = int(rng.choice(sides[bowl_team]))
                ns = int(rng.choice([p for p in sides[bat_team] if p != b]))
                probs = outcome_probs(
                    np.array([-3.0 + eff["bat_w"][b] + eff["bowl_w"][w]]),
                    np.array([eff["bat_r"][b] + eff["bowl_r"][w]]), cut,
                )[0]
                k = int(rng.choice(len(probs), p=probs))
                runs = 0 if k == 0 else RUN_VALUES[k - 1]
                d = {"batter": names[b], "bowler": names[w], "non_striker": names[ns],
                     "runs": {"batter": runs, "extras": 0, "total": runs}}
                if k == 0:
                    d["wickets"] = [{"player_out": names[b], "kind": "caught"}]
                total += runs
                deliveries.append(d)
            overs.append({"over": over, "deliveries": deliveries})
        return total, overs

    start = pd.Timestamp("2022-01-01")
    for m in range(n_matches):
        date = (start + pd.Timedelta(days=20 * m)).strftime("%Y-%m-%d")
        t1, o1 = innings("Team A", "Team B")
        t2, o2 = innings("Team B", "Team A")
        winner = "Team B" if t2 >= t1 + 1 else "Team A"
        doc = {
            "meta": {"data_version": "1.3.0"},
            "info": {
                "balls_per_over": 6, "overs": 20, "gender": "male", "match_type": "T20", "team_type": "club",
                "event": {"name": "Synth League"}, "season": date[:4], "dates": [date],
                "venue": "Synth Ground", "city": "Synthville", "teams": ["Team A", "Team B"],
                "toss": {"winner": "Team A", "decision": "bat"},
                "outcome": {"winner": winner, "by": {"runs": 1}},
                "registry": {"people": people},
            },
            "innings": [
                {"team": "Team A", "overs": o1, "powerplays": [{"from": 0.1, "to": 5.6, "type": "mandatory"}]},
                {"team": "Team B", "overs": o2, "target": {"overs": 20, "runs": t1 + 1},
                 "powerplays": [{"from": 0.1, "to": 5.6, "type": "mandatory"}]},
            ],
        }
        (raw_dir / f"{1000 + m}.json").write_text(json.dumps(doc), encoding="utf-8")
    return eff, people
