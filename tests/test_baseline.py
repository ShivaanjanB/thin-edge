import numpy as np
import pandas as pd
import pytest

from synth import synthetic
from thinedge.baseline import (
    CHASE_FEATURES,
    Baseline,
    Logit,
    add_features,
    brier,
    calibration_table,
    checkpoint_table,
    evaluate,
    fit_par_curve,
    log_loss,
    match_bootstrap,
    paired_brier_gain,
    summarise,
)


@pytest.fixture(scope="module")
def data():
    return synthetic()


def _logistic_data(n=20000, seed=0):
    # Well conditioned on purpose: comparable scales and a balanced outcome, so the individual
    # coefficients (not just their combinations) are identifiable at this sample size.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [1.0, 2.0, 0.5] + [0.0, 1.0, 0.0]
    true_w, true_b = np.array([0.8, -0.3, 1.5]), -0.5
    p = 1 / (1 + np.exp(-(X @ true_w + true_b)))
    return X, (rng.random(n) < p).astype(float), true_w, true_b


def test_logit_recovers_known_coefficients_on_raw_scale():
    X, y, true_w, true_b = _logistic_data()
    m = Logit(c=1e6).fit(X, y)  # effectively unpenalised
    raw_w = m.coef_ / m.scale_
    raw_b = m.intercept_ - np.sum(m.coef_ * m.mean_ / m.scale_)
    assert raw_w == pytest.approx(true_w, abs=0.1)
    assert raw_b == pytest.approx(true_b, abs=0.1)


def test_logit_solution_satisfies_optimality_condition():
    X, y, _, _ = _logistic_data(n=3000)
    for c in (0.01, 1.0):
        m = Logit(c=c).fit(X, y)
        A = m._design(X)
        w = np.concatenate([[m.intercept_], m.coef_])
        reg = np.full(len(w), 1 / c)
        reg[0] = 0
        grad = A.T @ (m.proba(X) - y) + reg * w
        assert np.max(np.abs(grad)) < 1e-5


def test_logit_regularisation_shrinks_coefficients():
    X, y, _, _ = _logistic_data(n=3000)
    # Smaller C means a stronger penalty, so the coefficient norm must fall at every step, and a
    # penalty far larger than the data's curvature (~n * 0.2 per standardised unit) must crush it.
    norms = [np.linalg.norm(Logit(c=c).fit(X, y).coef_) for c in (100, 1, 0.01, 1e-4)]
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 0.15 * norms[0]


def test_logit_refuses_non_finite_input_instead_of_returning_nan():
    X, y, _, _ = _logistic_data(n=200)
    for bad in (np.nan, np.inf):
        Xb = X.copy()
        Xb[3, 1] = bad
        with pytest.raises(ValueError, match="finite"):
            Logit().fit(Xb, y)
    yb = y.copy()
    yb[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Logit().fit(X, yb)
    fitted = Logit().fit(X, y)
    Xp = X.copy()
    Xp[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fitted.proba(Xp)


def test_baseline_names_the_offending_column_when_a_chase_has_no_target(data):
    broken = data.copy()
    rows = (broken["innings_no"] == 2) & (broken["match_id"] == 0)
    broken.loc[rows, "target_runs"] = np.nan
    with pytest.raises(ValueError, match="log_runs_needed"):
        Baseline(use_regime=False).fit(broken, {})
    ok = Baseline(use_regime=False).fit(data, {})
    with pytest.raises(ValueError, match="chase features"):
        ok.predict(broken)


def test_logit_handles_constant_column_and_separable_data():
    X = np.column_stack([np.linspace(-1, 1, 50), np.ones(50)])
    y = (X[:, 0] > 0).astype(float)  # perfectly separable
    m = Logit(c=1.0).fit(X, y)
    p = m.proba(X)
    assert np.isfinite(p).all() and np.isfinite(m.coef_).all()
    assert p[-1] > 0.9 and p[0] < 0.1


def test_features_use_only_pre_ball_columns(data):
    # Post-ball and label-like columns are absent; feature building must still work.
    assert not any(c.endswith("_post") for c in data.columns)
    d = add_features(data, {})
    assert set(CHASE_FEATURES) <= set(d.columns)
    assert (d["impact_rule"] == 0).all()  # no rule dates given


def test_features_do_not_change_with_the_label(data):
    flipped = data.assign(y=1 - data["y"])
    a, b = add_features(data, {}), add_features(flipped, {})
    pd.testing.assert_frame_equal(a.drop(columns="y"), b.drop(columns="y"))


def test_impact_rule_flag_is_date_and_competition_specific():
    df = pd.DataFrame({
        "source": ["ipl", "ipl", "hnd"],
        "date": pd.to_datetime(["2022-05-01", "2023-04-01", "2024-08-01"]),
        "balls_per_over": [6, 6, 5], "max_balls": [120, 120, 100], "target_runs": [150.0] * 3,
        "score_pre": [0] * 3, "wickets_pre": [0] * 3, "legal_balls_pre": [0] * 3,
        "balls_remaining_pre": [120, 120, 100], "match_id": [1, 2, 3], "innings_no": [2] * 3, "y": [0] * 3,
    })
    d = add_features(df, {"ipl": pd.Timestamp("2023-03-31")})
    assert d["impact_rule"].tolist() == [0.0, 1.0, 0.0]
    assert d["is_hundred"].tolist() == [0.0, 0.0, 1.0]


def test_par_curve_recovers_a_linear_scoring_rate():
    rows = [{"source": "ipl", "legal_balls_pre": b, "score_pre": 1.5 * b} for _ in range(30) for b in range(120)]
    par = fit_par_curve(pd.DataFrame(rows), ["source"])
    probe = pd.DataFrame({"source": ["ipl"] * 3, "legal_balls_pre": [0, 60, 119]})
    # the 9-ball smoothing window is one-sided at the edges, so allow a small edge bias
    assert par.at(probe) == pytest.approx([0.0, 90.0, 178.5], abs=3.5)


def test_par_curve_unseen_regime_falls_back_to_the_same_competition_not_the_pool():
    rows = [{"source": s, "impact_rule": 0.0, "legal_balls_pre": b, "score_pre": rate * b}
            for s, rate in (("ipl", 1.0), ("hnd", 3.0)) for b in range(60) for _ in range(5)]
    par = fit_par_curve(pd.DataFrame(rows), ["source", "impact_rule"])
    # (hnd, regime 1) was never seen in training. It must use hnd's own curve (3 per ball), not the
    # ipl+hnd average (2 per ball) that the pooled curve would give.
    probe = pd.DataFrame({"source": ["hnd", "ipl"], "impact_rule": [1.0, 1.0], "legal_balls_pre": [30, 30]})
    assert par.at(probe) == pytest.approx([90.0, 30.0], abs=1.0)


def test_par_curve_falls_back_for_unseen_competition():
    rows = [{"source": "ipl", "legal_balls_pre": b, "score_pre": 2.0 * b} for b in range(50)]
    par = fit_par_curve(pd.DataFrame(rows), ["source"])
    probe = pd.DataFrame({"source": ["never_seen"], "legal_balls_pre": [10]})
    assert par.at(probe)[0] == pytest.approx(20.0, abs=2)


def test_higher_required_rate_lowers_chase_win_probability(data):
    model = Baseline(use_regime=False).fit(data, {})
    probe = data[(data["innings_no"] == 2) & (data["legal_balls_pre"] == 60)].head(1)
    easy = probe.assign(score_pre=probe["target_runs"] - 20)
    hard = probe.assign(score_pre=probe["target_runs"] - 110)
    assert model.predict(easy).iloc[0] > model.predict(hard).iloc[0]


def test_baseline_beats_constant_reference_on_signal(data):
    train, test = data[data["match_id"] < 300], data[data["match_id"] >= 300]
    model = Baseline(use_regime=False).fit(train, {})
    p = model.predict(test)
    ref = test["innings_no"].map(train.groupby("innings_no")["y"].mean())
    # Simulated results depend on the state, so both models must clearly beat knowing nothing.
    # Measured gaps are about 0.17 (chase) and 0.06 (first innings); the margins are conservative.
    for innings, margin in ((2, 0.10), (1, 0.03)):
        s = test["innings_no"] == innings
        assert brier(test.loc[s, "y"], p[s]) < brier(test.loc[s, "y"], ref[s]) - margin
    assert p.between(0, 1).all() and not p.isna().any()


def test_predict_keeps_index_alignment(data):
    model = Baseline(use_regime=False).fit(data, {})
    shuffled = data.sample(frac=1, random_state=3)
    p = model.predict(shuffled)
    assert p.index.equals(shuffled.index)
    again = model.predict(data)
    assert p.sort_index().to_numpy() == pytest.approx(again.sort_index().to_numpy())


def test_brier_and_log_loss():
    y, p = [1, 0, 1, 0], [0.9, 0.1, 0.9, 0.1]
    assert brier(y, p) == pytest.approx(0.01)
    assert log_loss(y, p) == pytest.approx(-np.log(0.9))
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_calibration_table_known_values():
    y = [0, 0, 1, 1, 1, 1, 0, 1]
    p = [0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05]
    rows, ece = calibration_table(y, p, bins=10)
    low, high = rows[0], rows[-1]
    assert (low["n"], low["mean_pred"], low["mean_obs"]) == (3, pytest.approx(0.05), pytest.approx(1 / 3))
    assert (high["n"], high["mean_obs"]) == (5, pytest.approx(0.8))
    assert ece == pytest.approx(3 / 8 * abs(0.05 - 1 / 3) + 5 / 8 * abs(0.95 - 0.8))


def test_calibration_of_a_perfectly_calibrated_model_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert calibration_table(y, p)[1] < 0.02


def test_match_bootstrap_resamples_whole_matches():
    f = pd.DataFrame({"match_id": [1, 1, 2, 2, 2], "v": [1.0, 1.0, 10.0, 10.0, 10.0]})
    boot = match_bootstrap(f, ["v"], n_boot=500, seed=0)
    # Two matches (2 rows worth 2.0, 3 rows worth 30.0) drawn with replacement: only whole-match
    # combinations are possible. Row-level resampling would produce other totals.
    assert set(np.unique(boot["n"])) == {4, 5, 6}
    assert set(np.unique(boot["v"])) == {4.0, 32.0, 60.0}


def test_summarise_ci_brackets_point_estimate_and_is_reproducible(data):
    f = data[data["innings_no"] == 2].assign(p=0.5, ref=0.5)
    a, b = summarise(f, 300, seed=2), summarise(f, 300, seed=2)
    assert a == b
    assert a["brier_ci"][0] <= a["brier"] <= a["brier_ci"][1]
    assert a["skill"] == pytest.approx(0.0, abs=1e-12)


def test_checkpoints_take_first_delivery_after_k_legal_balls():
    rows = []
    for m in (1, 2):
        for innings in (1, 2):
            for seq, balls in enumerate([0, 1, 1, 2, 30, 60, 90]):  # a wide repeats legal_balls_pre = 1
                rows.append({"match_id": m, "innings_no": innings, "max_balls": 120,
                             "legal_balls_pre": balls, "y": 1, "p": 0.5, "ref": 0.5})
    table = checkpoint_table(pd.DataFrame(rows))
    assert {(t["innings"], t["fraction_of_innings"]): t["n"] for t in table} == {
        (i, f): 2 for i in (1, 2) for f in (0.25, 0.5, 0.75)
    }


def test_paired_gain_sign_and_evaluate_structure(data):
    test = data[data["match_id"] >= 300]
    good = test["y"].astype(float).clip(0.1, 0.9)
    bad = pd.Series(0.5, index=test.index)
    gain = paired_brier_gain(test, good, bad, 200, seed=0)
    # the per-row difference is constant, so the CI collapses onto the mean up to float rounding
    assert gain["mean"] > 0.2
    assert gain["ci"][0] - 1e-9 <= gain["mean"] <= gain["ci"][1] + 1e-9
    out = evaluate(test, good, bad, n_boot=50)
    assert set(out["metrics"]) == {"all", "first_innings", "chase"}
    assert set(out["calibration"]) == {"first_innings", "chase"}
