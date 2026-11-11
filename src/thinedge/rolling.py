"""Rolling-origin evaluation: choose how the model copes with a changing scoring environment.

Why this exists
    The baseline's Impact Player flag was estimated from a single season (71 matches) and turned a
    noisy swing into a large intercept shift. A binary flag for a rule change cannot be validated
    before it has been in force for several seasons, so the choice of how to handle drift has to be
    made on strategies that *can* be validated on the long pre-2024 history.

Method
    Rolling origin: for each year Y, train on matches before 1 January Y (optionally only the last
    ``window_years``) and predict year Y. Repeat across validation years, which all precede the test
    period. Every candidate scores exactly the same rows.

Candidates (``DEFAULT_CANDIDATES``; ``rank`` is simplicity, lower is simpler)
    0 no_flag/expanding   no regime flag, all history (the baseline without the flag)
    1 no_flag/window5     no flag, last 5 years only
    2 no_flag/window3     no flag, last 3 years only
    3 no_flag/window2     no flag, last 2 years only
    4 flag/expanding      the current baseline: Impact Player flag, all history
    The flag has the highest rank (least preferred) because it is the least validated mechanism.

Selection rule (fixed before the real data was run; it uses validation years only)
    1. Primary metric: pooled Brier score over all validation rows.
    2. ``best`` is the candidate with the lowest pooled Brier.
    3. A candidate is indistinguishable from ``best`` if the 95% match-level paired interval for
       (its Brier - best's Brier) includes zero.
    4. Choose the simplest (lowest rank) candidate among ``best`` and those indistinguishable from it.

Final exam
    The chosen candidate, and every other candidate for transparency, is then scored on the test
    years (2024 onward) with the same yearly retraining, next to the shipped fixed baseline.

Confirmation on competitions not previously examined (added with the extra leagues)
    Earlier analysis looked at the 2024 onward matches of ``ipl`` and ``hnd`` only (``examined``).
    Matches from 2024 onward in every other competition are ``fresh``: they were not scored by any
    model or inspected before this step. One hypothesis was fixed before those matches were scored:

        H1: on the fresh matches, no_flag/window2 has lower pooled Brier than no_flag/expanding,
            with the 95% paired match-level interval entirely above zero.

    H1 is the only confirmatory comparison. The post-hoc IPL finding that motivated it was that
    short memory beats all history; window2 was the best candidate there. Every other number in the
    fresh section (other windows, the flag, the shipped baseline) is descriptive, not a test. The
    selection rule above is unchanged and now sees every competition's validation years.
    Fresh matches share the calendar period with examined ones, so trends common to all T20 cricket
    carry over: this is a test on new competitions, not on new years.

Limits worth stating up front
    - The test period is NOT a clean holdout: earlier steps examined it closely, and dropping the
      flag was suggested by that examination. The selection itself uses only validation years, but
      the final exam cannot be read as independent confirmation. That needs data that does not exist
      yet (a later season) or a different competition.
    - Competitions with fewer than ``MIN_SOURCE_MATCHES`` earlier matches are not scored in a fold
      (a cold start is a different problem), identically for all candidates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import Baseline, brier, impact_start_dates, load_frame, paired_brier_gain, summarise
from .config import PROCESSED_DIR, REPORTS_DIR
from .db import connect

MIN_SOURCE_MATCHES = 50


@dataclass(frozen=True)
class Candidate:
    name: str
    rank: int
    use_regime: bool
    window_years: int | None = None


DEFAULT_CANDIDATES = (
    Candidate("no_flag/expanding", 0, False, None),
    Candidate("no_flag/window5", 1, False, 5),
    Candidate("no_flag/window3", 2, False, 3),
    Candidate("no_flag/window2", 3, False, 2),
    Candidate("flag/expanding", 4, True, None),
)


def history_for(df: pd.DataFrame, year: int, window_years: int | None) -> pd.DataFrame:
    """Matches strictly before 1 January ``year``, optionally only the last ``window_years``."""
    end = pd.Timestamp(year=year, month=1, day=1)
    history = df[df["date"] < end]
    if window_years is not None:
        history = history[history["date"] >= end - pd.DateOffset(years=window_years)]
    return history


def rolling_predictions(
    df: pd.DataFrame,
    impact: dict[str, pd.Timestamp],
    candidates,
    years,
    min_source_matches: int = MIN_SOURCE_MATCHES,
) -> pd.DataFrame:
    """Rows of each year in ``years`` with one prediction column per candidate."""
    parts = []
    for year in years:
        start = pd.Timestamp(year=int(year), month=1, day=1)
        block = df[(df["date"] >= start) & (df["date"] < start + pd.DateOffset(years=1))]
        seen = df[df["date"] < start].groupby("source")["match_id"].nunique()
        block = block[block["source"].isin(seen[seen >= min_source_matches].index)]
        if block.empty:
            continue
        part = block.assign(year=int(year))
        for c in candidates:
            history = history_for(df, int(year), c.window_years)
            if history.empty:
                raise ValueError(f"{c.name}: no history before {year}")
            part[c.name] = Baseline(c.use_regime).fit(history, impact).predict(block)
        parts.append(part)
    if not parts:
        raise ValueError("no rows to score; check the years and the data")
    return pd.concat(parts)


def select(summary: dict[str, dict], candidates) -> dict:
    """Apply the selection rule. ``summary[name]`` needs ``brier`` and, for all but the best,
    ``vs_best_ci`` (95% interval of Brier minus the best's Brier). Sees validation results only."""
    ranks = {c.name: c.rank for c in candidates}
    if len(set(ranks.values())) != len(ranks):
        raise ValueError("candidate ranks must be unique")
    best = min(summary, key=lambda n: summary[n]["brier"])
    tied = [n for n in summary if n != best and summary[n]["vs_best_ci"][0] <= 0]
    chosen = min([best, *tied], key=lambda n: ranks[n])
    return {"best": best, "indistinguishable_from_best": sorted(tied, key=ranks.get), "chosen": chosen}


def _per_candidate(frame: pd.DataFrame, names, n_boot: int) -> dict:
    out = {}
    for name in names:
        by_year = {int(y): brier(g["y"], g[name]) for y, g in frame.groupby("year")}
        entry = {"brier": brier(frame["y"], frame[name]), "brier_by_year": by_year}
        for innings, label in ((1, "first_innings"), (2, "chase")):
            g = frame[frame["innings_no"] == innings]
            entry[f"brier_{label}"] = brier(g["y"], g[name])
            entry[f"gap_{label}"] = float(g[name].mean() - g["y"].mean())
        out[name] = entry
    return out


EXAMINED_SOURCES = ("ipl", "hnd")
BASE_CANDIDATE = "no_flag/expanding"
H1_CANDIDATE = "no_flag/window2"


def _exam_block(sub: pd.DataFrame, ref: pd.Series, columns, n_boot: int, test_start_year: int) -> dict:
    """Pooled and per-innings scores for each prediction column on one slice of the test rows."""
    out = {}
    for name in columns:
        entry = {}
        for label, mask in (
            ("all", sub["innings_no"] > 0),
            ("first_innings", sub["innings_no"] == 1),
            ("chase", sub["innings_no"] == 2),
        ):
            entry[label] = summarise(sub[mask].assign(p=sub[name], ref=ref), n_boot, seed=0)
        late = (sub["source"] == "ipl") & (sub["year"] > test_start_year)
        for innings, label in ((1, "first_innings"), (2, "chase")):
            g = sub[late & (sub["innings_no"] == innings)]
            if not g.empty:
                entry[f"ipl_after_first_test_year_{label}"] = {
                    "brier": brier(g["y"], g[name]), "gap": float(g[name].mean() - g["y"].mean()),
                }
        out[name] = entry
    return out


def analyse(
    df: pd.DataFrame,
    impact: dict[str, pd.Timestamp],
    candidates=DEFAULT_CANDIDATES,
    test_start_year: int = 2024,
    first_val_year: int = 2015,
    n_boot: int = 1000,
    examined_sources=EXAMINED_SOURCES,
) -> dict:
    names = [c.name for c in candidates]
    val_years = list(range(first_val_year, test_start_year))
    val = rolling_predictions(df, impact, candidates, val_years)
    summary = _per_candidate(val, names, n_boot)
    best = min(names, key=lambda n: summary[n]["brier"])
    for name in names:
        if name == best:
            summary[name]["vs_best"] = None
            summary[name]["vs_best_ci"] = None
            continue
        gain = paired_brier_gain(val, val[best], val[name], n_boot, seed=1)
        summary[name]["vs_best"], summary[name]["vs_best_ci"] = gain["mean"], gain["ci"]
    selection = select(summary, candidates)

    flag_diff = None
    if {"flag/expanding", "no_flag/expanding"} <= set(names):
        flag_diff = float(np.max(np.abs(val["flag/expanding"] - val["no_flag/expanding"])))

    # ---- final exam: runs only after the selection above has been fixed -------------------------
    start = pd.Timestamp(year=test_start_year, month=1, day=1)
    test_years = sorted(int(y) for y in df.loc[df["date"] >= start, "date"].dt.year.unique())
    test = rolling_predictions(df, impact, candidates, test_years)
    test["fixed_baseline"] = Baseline(True).fit(df[df["date"] < start], impact).predict(test)
    ref = test["innings_no"].map(df[df["date"] < start].groupby("innings_no")["y"].mean())
    columns = [*names, "fixed_baseline"]
    subsets = {
        "all": test["source"].notna(),
        "examined": test["source"].isin(examined_sources),
        "fresh": ~test["source"].isin(examined_sources),
    }
    by_subset = {
        sname: {
            "n_matches": int(test.loc[smask, "match_id"].nunique()),
            "by_candidate": _exam_block(test[smask], ref[smask], columns, n_boot, test_start_year),
        }
        for sname, smask in subsets.items()
        if smask.any()
    }
    exam = by_subset["all"]["by_candidate"]

    paired_vs_base = {}
    for sname in ("examined", "fresh"):
        if sname in by_subset and BASE_CANDIDATE in columns:
            smask = subsets[sname]
            paired_vs_base[sname] = {
                c: paired_brier_gain(test[smask], test.loc[smask, c], test.loc[smask, BASE_CANDIDATE], n_boot, seed=4)
                for c in columns
                if c != BASE_CANDIDATE
            }
    confirmation = None
    if "fresh" in paired_vs_base and H1_CANDIDATE in paired_vs_base["fresh"]:
        g = paired_vs_base["fresh"][H1_CANDIDATE]
        fm = subsets["fresh"]
        # The interval comes from a bootstrap, so a lower bound that sits near zero can change sign
        # with the random seed. `supported` is the single comparison fixed before the real run; `robust` asks
        # that the lower bound stay above zero for every one of several seeds.
        lowers = [
            paired_brier_gain(test[fm], test.loc[fm, H1_CANDIDATE], test.loc[fm, BASE_CANDIDATE], n_boot, seed=s)["ci"][0]
            for s in range(5)
        ]
        confirmation = {
            "hypothesis": f"H1: on fresh matches {H1_CANDIDATE} has lower pooled Brier than {BASE_CANDIDATE}, "
                          "95% paired match-level interval entirely above zero",
            "n_fresh_matches": by_subset["fresh"]["n_matches"],
            "brier_gain": g["mean"],
            "ci": g["ci"],
            "supported": bool(g["ci"][0] > 0),
            "ci_lower_across_5_seeds": [float(min(lowers)), float(max(lowers))],
            "robust": bool(min(lowers) > 0),
        }

    chosen = selection["chosen"]
    paired = {
        f"{chosen}_vs_{other}": paired_brier_gain(test, test[chosen], test[other], n_boot, seed=2)
        for other in ("fixed_baseline", "flag/expanding")
        if other != chosen
    }
    return {
        "validation_years": sorted(int(y) for y in val["year"].unique()),
        "test_years": test_years,
        "n_validation_matches": int(val["match_id"].nunique()),
        "n_test_matches": int(test["match_id"].nunique()),
        "candidates": [asdict(c) for c in candidates],
        "validation": {
            "by_candidate": summary,
            "flag_vs_no_flag_max_abs_prediction_difference": flag_diff,
        },
        "selection": selection,
        "final_exam": {
            "note": "The test period is not a clean holdout (see module docstring); the selection above "
                    "used validation years only.",
            "by_candidate": exam,
            "by_subset": by_subset,
            "examined_sources": list(examined_sources),
            "paired_vs_no_flag_expanding": paired_vs_base,
            "confirmation": confirmation,
            "chosen_paired": paired,
        },
    }


def run_rolling(
    test_start_year: int = 2024,
    first_val_year: int = 2015,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    n_boot: int = 1000,
    sources: list[str] | None = None,
) -> dict:
    con = connect(processed_dir)
    report = analyse(
        load_frame(con, sources), impact_start_dates(con),
        test_start_year=test_start_year, first_val_year=first_val_year, n_boot=n_boot,
    )
    report["sources"] = sorted(sources) if sources else "all"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "rolling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
