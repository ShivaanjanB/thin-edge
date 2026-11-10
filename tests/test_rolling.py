import json

import numpy as np
import pandas as pd
import pytest

from synth import synthetic
from thinedge.rolling import (
    DEFAULT_CANDIDATES,
    Candidate,
    analyse,
    history_for,
    rolling_predictions,
    select,
)

NAMES = [c.name for c in DEFAULT_CANDIDATES]


def make(drift, seed=5):
    return synthetic(n_matches=3000, seed=seed, start="2016-01-01", year_drift=drift)


@pytest.fixture(scope="module")
def df():
    return make(0.12)


@pytest.fixture(scope="module")
def drifting(df):
    return analyse(df, {}, test_start_year=2024, first_val_year=2019, n_boot=200)


@pytest.fixture(scope="module")
def stable():
    return analyse(make(0.0), {}, test_start_year=2024, first_val_year=2019, n_boot=200)


def test_history_windows_are_half_open_on_new_year():
    d = pd.DataFrame({"date": pd.to_datetime(["2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01"])})
    assert history_for(d, 2022, None)["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2019-12-31", "2020-01-01", "2021-12-31"
    ]
    # window of 2 years before 2022-01-01 is [2020-01-01, 2022-01-01)
    assert history_for(d, 2022, 2)["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2021-12-31"]


def test_every_candidate_scores_the_same_complete_rows(df):
    out = rolling_predictions(df, {}, DEFAULT_CANDIDATES, [2021, 2022])
    assert set(out["year"]) == {2021, 2022}
    for name in NAMES:
        assert not out[name].isna().any() and out[name].between(0, 1).all()
    expected = df[(df["date"].dt.year >= 2021) & (df["date"].dt.year <= 2022)]
    assert out.index.equals(expected.index)


def test_no_candidate_sees_the_labels_of_the_year_it_predicts(df):
    flipped = df.copy()
    mask = flipped["date"].dt.year == 2022
    flipped.loc[mask, "y"] = 1 - flipped.loc[mask, "y"]
    a = rolling_predictions(df, {}, DEFAULT_CANDIDATES, [2020, 2021, 2022])
    b = rolling_predictions(flipped, {}, DEFAULT_CANDIDATES, [2020, 2021, 2022])
    for name in NAMES:  # years before 2022 never touch 2022 labels either
        assert a[name].to_numpy() == pytest.approx(b[name].to_numpy(), abs=1e-12), name


def test_a_competition_with_too_little_history_is_not_scored(df):
    d = df.copy()
    late = d["date"].dt.year == 2022
    d.loc[late & (d["match_id"] % 2 == 0), "source"] = "newcomer"
    out = rolling_predictions(d, {}, DEFAULT_CANDIDATES[:1], [2022])
    assert "newcomer" not in set(out["source"]) and len(out) > 0


def test_flag_cannot_differ_from_no_flag_before_it_has_been_seen_in_training(df):
    # The rule starts in 2023, so every fold up to and including 2023 trains without flagged rows.
    impact = {"ipl": pd.Timestamp("2023-01-01")}
    out = rolling_predictions(df, impact, DEFAULT_CANDIDATES, [2021, 2022, 2023])
    assert np.max(np.abs(out["flag/expanding"] - out["no_flag/expanding"])) < 1e-9
    # Once 2023 is in the history the flag is live and the two must differ.
    later = rolling_predictions(df, impact, DEFAULT_CANDIDATES, [2024])
    assert np.max(np.abs(later["flag/expanding"] - later["no_flag/expanding"])) > 1e-6


def make_split(drift, seed=5):
    """Half the matches are relabelled as a competition the earlier analysis never examined."""
    d = synthetic(n_matches=3150, seed=seed, start="2016-01-01", year_drift=drift)
    d.loc[d["match_id"] % 2 == 1, "source"] = "fresh"
    return d


@pytest.fixture(scope="module")
def split_drifting():
    return analyse(make_split(0.12), {}, test_start_year=2024, first_val_year=2019, n_boot=200,
                   examined_sources=("ipl",))


@pytest.fixture(scope="module")
def split_stable():
    return analyse(make_split(0.0), {}, test_start_year=2024, first_val_year=2019, n_boot=200,
                   examined_sources=("ipl",))


def test_final_exam_separates_examined_from_fresh_competitions(split_drifting):
    fe = split_drifting["final_exam"]
    sub = fe["by_subset"]
    assert set(sub) == {"all", "examined", "fresh"}
    assert sub["examined"]["n_matches"] + sub["fresh"]["n_matches"] == sub["all"]["n_matches"]
    assert fe["examined_sources"] == ["ipl"]
    # the original all-rows table is kept for the CLI and plots
    assert fe["by_candidate"] == sub["all"]["by_candidate"]
    assert set(fe["paired_vs_no_flag_expanding"]["fresh"]) == set(NAMES) - {"no_flag/expanding"} | {"fixed_baseline"}


def test_h1_is_supported_on_fresh_matches_when_scoring_drifts(split_drifting):
    h1 = split_drifting["final_exam"]["confirmation"]
    assert h1["supported"] and h1["ci"][0] > 0 and h1["brier_gain"] > 0
    assert h1["n_fresh_matches"] == split_drifting["final_exam"]["by_subset"]["fresh"]["n_matches"]


def test_h1_is_not_supported_without_drift(split_stable):
    h1 = split_stable["final_exam"]["confirmation"]
    assert not h1["supported"] and h1["ci"][0] <= 0
    assert not h1["robust"]


def test_h1_support_is_robust_to_the_bootstrap_seed_when_the_effect_is_large(split_drifting):
    h1 = split_drifting["final_exam"]["confirmation"]
    lo, hi = h1["ci_lower_across_5_seeds"]
    assert h1["robust"] and 0 < lo <= hi <= h1["brier_gain"]


def test_no_fresh_section_when_every_competition_was_examined():
    out = analyse(make_split(0.12), {}, test_start_year=2024, first_val_year=2022, n_boot=50,
                  examined_sources=("ipl", "fresh"))
    assert "fresh" not in out["final_exam"]["by_subset"]
    assert out["final_exam"]["confirmation"] is None
    json.dumps(out)


def _summary(briers, cis):
    return {n: {"brier": b, "vs_best_ci": cis.get(n)} for n, b in briers.items()}


def test_select_prefers_the_simplest_candidate_it_cannot_tell_from_the_best():
    cs = [Candidate("simple", 0, False), Candidate("mid", 1, False), Candidate("complex", 2, True)]
    s = _summary({"simple": 0.2010, "mid": 0.2000, "complex": 0.1990},
                 {"simple": [-0.001, 0.003], "mid": [-0.002, 0.004]})
    out = select(s, cs)
    assert out["best"] == "complex" and out["chosen"] == "simple"
    assert out["indistinguishable_from_best"] == ["simple", "mid"]


def test_select_keeps_a_clearly_better_complex_candidate():
    cs = [Candidate("simple", 0, False), Candidate("complex", 1, True)]
    out = select(_summary({"simple": 0.25, "complex": 0.20}, {"simple": [0.04, 0.06]}), cs)
    assert out["chosen"] == "complex" and out["indistinguishable_from_best"] == []


def test_select_rejects_duplicate_ranks():
    cs = [Candidate("a", 0, False), Candidate("b", 0, False)]
    with pytest.raises(ValueError):
        select(_summary({"a": 0.2, "b": 0.3}, {"b": [0.0, 0.2]}), cs)


def test_harness_picks_a_short_window_when_scoring_drifts(drifting):
    by = drifting["validation"]["by_candidate"]
    assert drifting["selection"]["chosen"].startswith("no_flag/window")
    assert by["no_flag/expanding"]["vs_best_ci"][0] > 0  # expanding is clearly worse
    # longer memory is monotonically worse under steady drift
    order = [by[n]["brier"] for n in ("no_flag/window2", "no_flag/window3", "no_flag/window5", "no_flag/expanding")]
    assert order == sorted(order)


def test_harness_falls_back_to_the_simplest_without_drift(stable):
    assert stable["selection"]["chosen"] == "no_flag/expanding"
    by = stable["validation"]["by_candidate"]
    spread = max(e["brier"] for e in by.values()) - min(e["brier"] for e in by.values())
    assert spread < 0.001


def test_flag_is_untestable_in_validation_and_the_report_says_so(drifting):
    assert drifting["validation"]["flag_vs_no_flag_max_abs_prediction_difference"] < 1e-9


def test_final_exam_scores_every_candidate_next_to_the_fixed_baseline(drifting):
    exam = drifting["final_exam"]["by_candidate"]
    assert set(exam) == set(NAMES) | {"fixed_baseline"}
    assert drifting["test_years"] == [2024]
    assert drifting["final_exam"]["chosen_paired"]
    assert "not a clean holdout" in drifting["final_exam"]["note"]
    json.dumps(drifting)
