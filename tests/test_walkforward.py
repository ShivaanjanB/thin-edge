import numpy as np
import pandas as pd
import pytest

from synth import synthetic
from thinedge.walkforward import ARMS, analyse, arm_predictions


def make(drift, seed=3):
    return synthetic(n_matches=1300, seed=seed, start="2021-01-01", year_drift=drift)


@pytest.fixture(scope="module")
def drifting_df():
    return make(0.15)


@pytest.fixture(scope="module")
def drifting(drifting_df):
    return analyse(drifting_df, {}, first_test_year=2023, n_boot=200)


@pytest.fixture(scope="module")
def stable():
    return analyse(make(0.0), {}, first_test_year=2023, n_boot=200)


def test_non_oracle_arms_agree_in_the_first_test_year(drifting):
    # Same history, so refreshing or retraining cannot change anything yet.
    assert drifting["first_year_max_abs_difference_between_arms"] < 1e-9
    assert drifting["test_years"] == [2023, 2024]


def test_arms_cover_the_same_rows_with_valid_probabilities(drifting_df):
    test, preds = arm_predictions(drifting_df, {}, 2023)
    assert set(preds) == set(ARMS)
    for p in preds.values():
        assert p.index.equals(test.index) and not p.isna().any() and p.between(0, 1).all()


def test_par_arms_cannot_change_the_chase(drifting_df):
    test, preds = arm_predictions(drifting_df, {}, 2023)
    chase = test["innings_no"] == 2
    for arm in ("par_oracle", "par_refresh"):
        assert preds[arm][chase].to_numpy() == pytest.approx(preds["fixed"][chase].to_numpy(), abs=1e-12)
    assert not np.allclose(preds["full_retrain"][chase & (test["date"].dt.year == 2024)],
                           preds["fixed"][chase & (test["date"].dt.year == 2024)])


def test_no_arm_sees_the_labels_of_the_year_it_predicts(drifting_df):
    # Flip every label in the last test year. Walk-forward may use earlier years' labels, never the
    # predicted year's, so no arm's predictions may move. This includes the oracle, which uses that
    # year's scores but not its results.
    last = drifting_df["date"].dt.year.max()
    flipped = drifting_df.copy()
    mask = flipped["date"].dt.year == last
    flipped.loc[mask, "y"] = 1 - flipped.loc[mask, "y"]
    _, a = arm_predictions(drifting_df, {}, 2023)
    _, b = arm_predictions(flipped, {}, 2023)
    for arm in ARMS:
        assert a[arm].to_numpy() == pytest.approx(b[arm].to_numpy(), abs=1e-12), arm


def test_cause_test_detects_stale_par_when_scoring_drifts(drifting):
    cause = drifting["verdict"]["cause_test"]["first_innings"]
    assert cause["supported"]
    assert cause["gap_reduction"] > 0.8  # measured about 0.95 on this seed
    assert cause["brier_gain_ci"][0] > 0


def test_cause_test_does_not_manufacture_an_effect_without_drift(stable):
    cause = stable["verdict"]["cause_test"]["first_innings"]
    assert not cause["supported"]
    assert cause["brier_gain_ci"][0] <= 0  # the interval does not exclude "no improvement"
    assert abs(stable["arms"]["fixed"]["ipl_refreshable"]["first_innings"]["gap"]) < 0.05


def test_practical_refresh_helps_but_lags_under_persistent_drift(drifting):
    # The reason the cause test uses an oracle: an expanding window keeps averaging in old seasons.
    refresh = drifting["verdict"]["practical_recovery"]["par_refresh/first_innings"]
    assert refresh["brier_gain_ci"][0] > 0
    assert refresh["gap_reduction"] < 0.5  # measured about 0.18 on this seed


def test_report_is_json_serialisable(drifting):
    import json

    json.dumps(drifting)
    assert {"arms", "paired_vs_fixed", "by_year", "calibration_ipl_refreshable", "verdict"} <= set(drifting)
    years = {row["year"] for row in drifting["by_year"]}
    assert years == {2023, 2024}
