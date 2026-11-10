"""Data-quality checks run after every build.

Severity
- ``error``: a violated invariant that would corrupt modelling (duplicates, runs that do not add
  up, a winner who did not play). The CLI exits non-zero if any error check is non-zero.
- ``warn``: known Cricsheet quirks or things to look at (miscounted overs, a target that does not
  equal first-innings runs + 1). Counted and sampled, never fatal.
"""

from __future__ import annotations

import pandas as pd

from .features import reduced_match_ids

SAMPLE = 5


def _check(count: int, severity: str, ids=None) -> dict:
    out = {"count": int(count), "severity": severity}
    if ids is not None and len(ids):
        out["sample_match_ids"] = [int(i) for i in list(ids)[:SAMPLE]]
    return out


def quality_report(matches: pd.DataFrame, innings: pd.DataFrame, balls: pd.DataFrame) -> dict:
    checks: dict[str, dict] = {}
    main = innings[~innings["is_super_over"]]

    dup = balls.duplicated(["match_id", "innings_no", "delivery_seq"])
    checks["duplicate_deliveries"] = _check(dup.sum(), "error", balls.loc[dup, "match_id"].unique())

    bad_runs = balls["runs_total"] != balls["runs_batter"] + balls["runs_extras"]
    checks["runs_total_mismatch"] = _check(bad_runs.sum(), "error", balls.loc[bad_runs, "match_id"].unique())

    wins = matches[matches["result"] == "win"]
    bad_winner = wins[(wins["winner"] != wins["team_1"]) & (wins["winner"] != wins["team_2"])]
    checks["winner_not_in_teams"] = _check(len(bad_winner), "error", bad_winner["match_id"])

    n_main = main.groupby("match_id").size().reindex(matches["match_id"], fill_value=0)
    decided = matches.set_index("match_id")["result"].isin(["win", "tie"])
    wrong_innings = n_main[decided.reindex(n_main.index) & (n_main != 2)]
    checks["decided_match_without_two_innings"] = _check(len(wrong_innings), "error", wrong_innings.index)

    over = main[
        (main["legal_balls"] > main["max_balls"]) & ~main["has_miscounted_overs"]
    ]
    checks["legal_balls_exceed_max_unflagged"] = _check(len(over), "warn", over["match_id"].unique())
    checks["innings_flagged_miscounted_overs"] = _check(main["has_miscounted_overs"].sum(), "warn")

    first = main[main["innings_no"] == 1].set_index("match_id")["total_runs"]
    second = main[main["innings_no"] == 2].set_index("match_id")["target_runs"]
    joined = pd.concat([first.rename("first"), second.rename("target")], axis=1).dropna()
    joined = joined[~joined.index.isin(reduced_match_ids(matches, innings))]
    mismatch = joined[joined["first"] + 1 != joined["target"]]
    checks["target_not_first_innings_plus_one"] = _check(len(mismatch), "warn", mismatch.index)

    decided_ids = matches.loc[matches["result"].isin(["win", "tie"]), "match_id"]
    second = main[(main["innings_no"] == 2) & main["match_id"].isin(decided_ids)]
    no_target = second[second["target_runs"].isna()]
    # Cricsheet omits the chase target in some matches. They cannot be modelled (and a missing
    # target may hide a rain revision), so modelling excludes them rather than guessing.
    checks["decided_chase_without_target"] = _check(len(no_target), "warn", no_target["match_id"])

    checks["matches_missing_venue"] = _check(matches["venue"].isna().sum(), "warn")
    checks["matches_missing_city"] = _check(matches["city"].isna().sum(), "warn")

    errors = {k: v["count"] for k, v in checks.items() if v["severity"] == "error" and v["count"]}
    return {
        "n_matches": int(len(matches)),
        "n_innings": int(len(innings)),
        "n_balls": int(len(balls)),
        "data_versions": {str(k): int(v) for k, v in matches["data_version"].value_counts(dropna=False).items()},
        "checks": checks,
        "ok": not errors,
        "errors": errors,
    }
