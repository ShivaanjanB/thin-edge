"""Ball-by-ball match state table for win-probability modelling.

One row per delivery, super overs included (flagged by ``is_super_over``), describing the match
state *after* that delivery, plus the *pre*-delivery score, wickets and balls (the inputs a
ball-outcome model needs). The label is ``batting_team_won`` and is null whenever the match
had no outright winner in regulation (tie, no result) or the innings is a super over.

Modelling notes
- ``runs_needed_post`` is clipped at 0 once the target is reached.
- ``balls_remaining_post`` is clipped at 0 because a handful of innings have miscounted overs.
- ``is_reduced`` marks rain-affected matches: a D/L method was applied, or any regulation
  innings was shorter than the nominal length (this catches no-result washouts too). Exclude
  them when training. First-innings length is assumed nominal, which is wrong for matches cut
  short mid-innings, and the flag is how you avoid learning from them.
- Phase is format-agnostic: powerplay as recorded, "death" is the final 20% of the innings
  (4 overs in T20, 20 balls in The Hundred).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEATH_FRACTION = 0.2


def assign_phase(
    in_powerplay: np.ndarray,
    balls_remaining_pre: np.ndarray,
    max_balls: np.ndarray,
    is_super_over: np.ndarray,
) -> np.ndarray:
    return np.select(
        [is_super_over, in_powerplay, balls_remaining_pre <= DEATH_FRACTION * max_balls],
        ["super_over", "powerplay", "death"],
        default="middle",
    )


def reduced_match_ids(matches: pd.DataFrame, innings: pd.DataFrame) -> set[int]:
    """Matches affected by rain: a D/L method was applied or a regulation innings was shortened."""
    nominal = matches.set_index("match_id").eval("overs * balls_per_over")
    main = innings[~innings["is_super_over"]]
    shortened = main.loc[main["max_balls"] < main["match_id"].map(nominal), "match_id"]
    method = matches.loc[matches["method"].notna(), "match_id"]
    return set(shortened) | set(method)


def build_state_table(
    matches: pd.DataFrame, innings: pd.DataFrame, balls: pd.DataFrame
) -> pd.DataFrame:
    b = balls.sort_values(["match_id", "innings_no", "delivery_seq"]).reset_index(drop=True)
    b = b.merge(
        innings[
            [
                "match_id", "innings_no", "bowling_team", "target_runs", "max_balls",
                "has_miscounted_overs", "penalty_runs_pre",
            ]
        ],
        on=["match_id", "innings_no"],
        how="left",
        validate="many_to_one",
    ).merge(
        matches[["match_id", "balls_per_over", "result", "winner", "method"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    g = b.groupby(["match_id", "innings_no"], sort=False)
    # Penalty runs awarded before an innings start it above zero (the chase target already counts them).
    b["score_post"] = b["penalty_runs_pre"] + g["runs_total"].cumsum()
    b["wickets_post"] = g["n_wickets"].cumsum()
    b["legal_balls_post"] = g["is_legal"].transform(lambda s: s.astype(int).cumsum())
    b["score_pre"] = b["score_post"] - b["runs_total"]
    b["wickets_pre"] = b["wickets_post"] - b["n_wickets"]
    b["legal_balls_pre"] = b["legal_balls_post"] - b["is_legal"].astype(int)

    b["balls_remaining_pre"] = (b["max_balls"] - b["legal_balls_pre"]).clip(lower=0)
    b["balls_remaining_post"] = (b["max_balls"] - b["legal_balls_post"]).clip(lower=0)
    b["wickets_in_hand_post"] = 10 - b["wickets_post"]

    bpo = b["balls_per_over"]
    b["crr_post"] = np.where(
        b["legal_balls_post"] > 0, b["score_post"] / b["legal_balls_post"].replace(0, np.nan) * bpo, np.nan
    )
    b["runs_needed_post"] = (b["target_runs"] - b["score_post"]).clip(lower=0)
    b["rrr_post"] = np.where(
        b["target_runs"].notna() & (b["balls_remaining_post"] > 0),
        b["runs_needed_post"] / b["balls_remaining_post"].replace(0, np.nan) * bpo,
        np.nan,
    )
    b["is_chase"] = b["target_runs"].notna()

    b["phase"] = assign_phase(
        b["in_powerplay"].to_numpy(),
        b["balls_remaining_pre"].to_numpy(),
        b["max_balls"].to_numpy(),
        b["is_super_over"].to_numpy(),
    )
    b["is_reduced"] = b["match_id"].isin(reduced_match_ids(matches, innings))

    labelled = b["result"].eq("win") & ~b["is_super_over"]
    label = pd.Series(pd.array([pd.NA] * len(b), dtype="Int8"), index=b.index)
    label[labelled] = (b.loc[labelled, "batting_team"] == b.loc[labelled, "winner"]).astype("Int8")
    b["batting_team_won"] = label

    keep = [
        "match_id", "innings_no", "delivery_seq", "over_idx", "is_super_over",
        "batting_team", "bowling_team", "batter_id", "bowler_id",
        "runs_total", "is_legal", "is_wicket",
        "score_pre", "wickets_pre", "legal_balls_pre",
        "score_post", "wickets_post", "legal_balls_post",
        "max_balls", "balls_remaining_pre", "balls_remaining_post", "wickets_in_hand_post",
        "crr_post", "is_chase", "target_runs", "runs_needed_post", "rrr_post",
        "phase", "in_powerplay", "has_miscounted_overs", "is_reduced", "batting_team_won",
    ]
    return b[keep]
