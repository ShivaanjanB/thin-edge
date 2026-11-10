"""Walk-forward test of the baseline's "stale par curve" explanation.

The fixed baseline was trained once on matches before 2024. Its first innings over-favours the team
batting first in recent IPL seasons, and the suspected cause is a par curve (expected score at each
ball) that no longer matches how fast teams now score. This module tests that directly.

Four arms are scored on exactly the same test rows:

- ``fixed``         the baseline as shipped, trained once on data before the first test year.
- ``par_oracle``    the same coefficients, with the par curve fit on the test year's own scores
                    (scores only, never results). DIAGNOSTIC, not deployable: it is the ceiling for
                    how much a wrong par level can explain.
- ``par_refresh``   the same coefficients, with the par curve refit each January on all earlier data.
                    The deployable version of the fix. It cannot change the chase, which has no par.
- ``full_retrain``  everything refit each January on all earlier data.

The first test year is identical for ``fixed``, ``par_refresh`` and ``full_retrain`` by construction
(same history), which is checked and reported.

Cause test (pre-registered before the real data was run)
    The explanation counts as supported only if, for IPL matches in test years after the first,
    ``par_oracle`` cuts the first-innings gap (mean prediction minus observed win rate) by at least
    ``MIN_GAP_REDUCTION`` AND its Brier gain over ``fixed`` has a 95% match-level interval above zero.

Design history, kept because it matters for trusting the result
    The first draft applied that rule to ``par_refresh``. Checked on simulated data in which scoring
    drifts upward every year (so stale par is a true cause by construction), ``par_refresh`` cut only
    18% of the gap and the rule would have rejected a real cause: an expanding window keeps averaging
    in old seasons, so under continued drift it still lags. The cause test was therefore moved to
    ``par_oracle`` and the practical arms are reported as recovery, with no pass/fail label. This
    change was made on simulated data, before any real walk-forward result had been seen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import Baseline, calibration_table, impact_start_dates, load_frame, paired_brier_gain, summarise
from .config import PROCESSED_DIR, REPORTS_DIR
from .db import connect

ARMS = ("fixed", "par_oracle", "par_refresh", "full_retrain")
CHALLENGERS = ARMS[1:]
MIN_GAP_REDUCTION = 0.5
INNINGS = ((1, "first_innings"), (2, "chase"))


def arm_predictions(
    df: pd.DataFrame, impact: dict[str, pd.Timestamp], first_test_year: int, use_regime: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Test rows (dated from ``first_test_year``) and each arm's predictions for them."""
    start = pd.Timestamp(year=first_test_year, month=1, day=1)
    fixed = Baseline(use_regime).fit(df[df["date"] < start], impact)
    test = df[df["date"] >= start]
    parts: dict[str, list[pd.Series]] = {arm: [] for arm in ARMS}
    for year, block in test.groupby(test["date"].dt.year):
        history = df[df["date"] < pd.Timestamp(year=int(year), month=1, day=1)]
        parts["fixed"].append(fixed.predict(block))
        parts["par_oracle"].append(fixed.with_fresh_par(block, impact).predict(block))
        parts["par_refresh"].append(fixed.with_fresh_par(history, impact).predict(block))
        parts["full_retrain"].append(Baseline(use_regime).fit(history, impact).predict(block))
    return test, {arm: pd.concat(p).reindex(test.index) for arm, p in parts.items()}


def _recovery(arms: dict, paired: dict, arm: str, label: str) -> dict | None:
    base = arms["fixed"].get("ipl_refreshable", {}).get(label)
    new = arms[arm].get("ipl_refreshable", {}).get(label)
    if base is None or new is None:
        return None
    gain = paired[f"{arm}_vs_fixed"]["ipl_refreshable"][label]
    return {
        "gap_fixed": base["gap"],
        "gap_new": new["gap"],
        "gap_reduction": float(1 - abs(new["gap"]) / abs(base["gap"])) if base["gap"] else 0.0,
        "brier_gain": gain["mean"],
        "brier_gain_ci": gain["ci"],
    }


def _verdict(arms: dict, paired: dict) -> dict:
    cause = _recovery(arms, paired, "par_oracle", "first_innings")
    if cause is not None:
        cause["supported"] = bool(cause["gap_reduction"] >= MIN_GAP_REDUCTION and cause["brier_gain_ci"][0] > 0)
    practical = {}
    for arm in ("par_refresh", "full_retrain"):
        for _, label in INNINGS:
            r = _recovery(arms, paired, arm, label)
            if r is not None:
                practical[f"{arm}/{label}"] = r
    return {"cause_test": {"arm": "par_oracle", "first_innings": cause}, "practical_recovery": practical}


def analyse(
    df: pd.DataFrame,
    impact: dict[str, pd.Timestamp],
    first_test_year: int = 2024,
    n_boot: int = 1000,
    use_regime: bool = True,
) -> dict:
    test, preds = arm_predictions(df, impact, first_test_year, use_regime)
    start = pd.Timestamp(year=first_test_year, month=1, day=1)
    ref = test["innings_no"].map(df[df["date"] < start].groupby("innings_no")["y"].mean())
    year = test["date"].dt.year
    masks = {
        "all": pd.Series(True, index=test.index),
        "ipl": test["source"] == "ipl",
        "hnd": test["source"] == "hnd",
        "ipl_refreshable": (test["source"] == "ipl") & (year > first_test_year),
    }
    masks = {name: m for name, m in masks.items() if m.any()}

    arms: dict = {}
    for arm, p in preds.items():
        arms[arm] = {}
        for name, mask in masks.items():
            arms[arm][name] = {}
            for innings, label in INNINGS:
                g = test[mask & (test["innings_no"] == innings)].assign(p=p, ref=ref)
                if g.empty:
                    continue
                arms[arm][name][label] = {
                    **summarise(g, n_boot, seed=0),
                    "mean_pred": float(g["p"].mean()),
                    "mean_obs": float(g["y"].mean()),
                    "gap": float(g["p"].mean() - g["y"].mean()),
                }

    paired: dict = {}
    for arm in CHALLENGERS:
        paired[f"{arm}_vs_fixed"] = {}
        for name, mask in masks.items():
            paired[f"{arm}_vs_fixed"][name] = {}
            for innings, label in INNINGS:
                sel = mask & (test["innings_no"] == innings)
                if sel.any():
                    paired[f"{arm}_vs_fixed"][name][label] = paired_brier_gain(
                        test[sel], preds[arm][sel], preds["fixed"][sel], n_boot, seed=1
                    )

    by_year = []
    for arm, p in preds.items():
        f = test.assign(p=p, year=year)
        for (source, innings, yr), g in f.groupby(["source", "innings_no", "year"]):
            by_year.append(
                {
                    "arm": arm, "source": source, "innings": int(innings), "year": int(yr),
                    "n_matches": int(g["match_id"].nunique()),
                    "mean_pred": float(g["p"].mean()), "mean_obs": float(g["y"].mean()),
                }
            )

    calibration = {}
    if "ipl_refreshable" in masks:
        for arm, p in preds.items():
            calibration[arm] = {}
            for innings, label in INNINGS:
                sel = masks["ipl_refreshable"] & (test["innings_no"] == innings)
                calibration[arm][label] = calibration_table(test.loc[sel, "y"], p[sel])[0]

    first_year_rows = year == year.min()
    spread = max(
        float(np.max(np.abs(preds[a][first_year_rows] - preds["fixed"][first_year_rows])))
        for a in ("par_refresh", "full_retrain")
    )
    return {
        "first_test_year": first_test_year,
        "test_years": sorted(int(y) for y in year.unique()),
        "n_test_matches": int(test["match_id"].nunique()),
        "first_year_max_abs_difference_between_arms": spread,
        "criterion": {
            "min_gap_reduction": MIN_GAP_REDUCTION,
            "rule": "cause test: IPL, test years after the first, par_oracle cuts the first-innings gap "
                    "by at least min_gap_reduction and its paired Brier gain over `fixed` has a 95% "
                    "match-level CI above zero",
        },
        "arms": arms,
        "paired_vs_fixed": paired,
        "by_year": by_year,
        "calibration_ipl_refreshable": calibration,
        "verdict": _verdict(arms, paired),
    }


def run_walkforward(
    first_test_year: int = 2024,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    n_boot: int = 1000,
    sources: list[str] | None = None,
) -> dict:
    con = connect(processed_dir)
    report = analyse(load_frame(con, sources), impact_start_dates(con), first_test_year, n_boot)
    report["sources"] = sorted(sources) if sources else "all"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "walkforward.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
