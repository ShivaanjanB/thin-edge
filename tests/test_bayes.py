import json

import numpy as np
import pandas as pd
import pytest

from synth_matches import true_effects, write_matches
from thinedge.bayes import (
    CUT_INIT,
    BallData,
    encode,
    outcome_probs,
    paired_logscore_gain,
    player_index,
)

pm = pytest.importorskip("pymc")

from thinedge.bayes import build_model, diagnostics, fit, predictive_scores, run_bayes  # noqa: E402
from thinedge.build import build  # noqa: E402

# ------------------------------------------------------------------------------ data handling


def raw_frame(rows):
    cols = ["runs_batter", "is_wicket", "over_idx", "in_powerplay", "innings_no", "batter_id", "bowler_id"]
    df = pd.DataFrame(rows, columns=cols)
    df["match_id"], df["date"], df["source"] = 1, pd.Timestamp("2022-01-01"), "x"
    df["batter"], df["bowler"] = "B", "W"
    return df


def test_encode_codes_outcomes_and_reports_what_it_drops():
    df = raw_frame([
        (0, False, 0, True, 1, "a", "w"),    # dot in the powerplay, first innings
        (6, False, 18, False, 2, "a", "w"),  # six at the death, second innings
        (1, True, 10, False, 1, "a", "w"),   # run out after a run: the wicket takes precedence
        (5, False, 5, True, 1, "a", "w"),    # 5 off the bat is not represented: dropped
        (7, False, 5, True, 1, "a", "w"),    # so is 7
        (3, False, 12, False, 2, "a", "w"),
        (1, False, 3, True, 1, None, "w"),   # no player id: dropped
    ])
    out, dropped = encode(df)
    assert dropped == {"balls_in": 7, "dropped_no_player_id": 1, "dropped_odd_runs": 2}
    assert out["wicket"].tolist() == [0, 0, 1, 0]
    assert out["run_cat"].tolist() == [0, 5, 1, 3]  # 6 is the sixth category; the wicket ball keeps its runs
    # group = phase * 2 + (innings - 1), with phase 0 powerplay, 1 middle, 2 death
    assert out["group"].tolist() == [0, 5, 2, 3]


def test_player_index_is_learned_from_training_only_and_buckets_the_rest():
    train = pd.Series(["a"] * 70 + ["b"] * 59 + ["c"] * 60)
    test = pd.Series(["a", "b", "c", "d"])
    tr, te, n, levels = player_index(train, test, min_balls=60)
    assert levels == ["a", "c"] and n == 3
    assert set(tr[train == "b"]) == {0}  # too few balls: shared "other" level
    assert te.tolist() == [1, 0, 2, 0]  # b is rare, d was never seen: both are "other"


def test_outcome_probabilities_are_a_distribution():
    rng = np.random.default_rng(0)
    p = outcome_probs(rng.normal(-3, 1, 500), rng.normal(0, 1, 500), CUT_INIT)
    assert p.shape == (500, 7) and (p >= 0).all()
    assert p.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    pw = 1 / (1 + np.exp(3.0))
    assert outcome_probs(np.array([-3.0]), np.array([0.0]), CUT_INIT)[0, 0] == pytest.approx(pw)


def test_outcome_probabilities_match_the_pymc_likelihood():
    eta_r = np.array([-1.2, 0.0, 0.9, 2.0])
    eta_w = np.array([-3.0, 0.0, 0.5, -1.0])
    probs = outcome_probs(eta_w, eta_r, CUT_INIT)
    for k in range(6):
        logp = pm.logp(pm.OrderedLogistic.dist(eta=eta_r, cutpoints=CUT_INIT), np.full(4, k)).eval()
        expected = np.exp(logp) * (1 - 1 / (1 + np.exp(-eta_w)))
        assert probs[:, 1 + k] == pytest.approx(expected, rel=1e-9)
    wicket = pm.logp(pm.Bernoulli.dist(logit_p=eta_w), np.ones(4)).eval()
    assert probs[:, 0] == pytest.approx(np.exp(wicket), rel=1e-9)


def test_paired_logscore_gain_sign_ci_and_nan_handling():
    f = pd.DataFrame({"match_id": np.repeat(np.arange(40), 5), "a": -1.0, "b": -1.01})
    g = paired_logscore_gain(f, "a", "b", n_boot=200, seeds=range(3))
    assert g["mean"] == pytest.approx(0.01) and g["ci"][0] == pytest.approx(0.01) and g["robust"]
    same = paired_logscore_gain(f.assign(b=f["a"]), "a", "b", n_boot=200, seeds=range(3))
    assert same["mean"] == 0 and not same["robust"]
    f.loc[f.index[:10], "a"] = np.nan  # the runs part is NaN on wicket balls: ignored, not propagated
    assert np.isfinite(paired_logscore_gain(f, "a", "b", n_boot=100, seeds=[0])["mean"])


# -------------------------------------------------------------------- recovery on simulated play


def simulate(n, n_players=30, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    eff = true_effects(n_players, seed + 100, scale)
    full = lambda v: np.concatenate([[0.0], v])  # noqa: E731  index 0 is the shared "other" level
    bat, bowl = rng.integers(1, n_players + 1, n), rng.integers(1, n_players + 1, n)
    group = rng.integers(0, 6, n)
    eta_w = -3.0 + full(eff["bat_w"])[bat] + full(eff["bowl_w"])[bowl]
    eta_r = full(eff["bat_r"])[bat] + full(eff["bowl_r"])[bowl]
    k = (rng.random(n)[:, None] > outcome_probs(eta_w, eta_r, CUT_INIT).cumsum(axis=1)).sum(axis=1)
    return BallData(group, bat, bowl, (k == 0).astype(int), np.where(k == 0, -1, k - 1), n_players + 1, n_players + 1), eff


def take(d, sl):
    return BallData(d.group[sl], d.bat[sl], d.bowl[sl], d.wicket[sl], d.run_cat[sl], d.n_bat, d.n_bowl)


def sample_small(d, use_bat, use_bowl):
    return fit(build_model(d, use_bat, use_bowl), draws=250, tune=250, chains=2, cores=1, seed=3)


def score_gain(a, b, n):
    frame = pd.DataFrame({"match_id": np.arange(n) // 240, "a": a, "b": b})
    return paired_logscore_gain(frame, "a", "b", n_boot=300, seeds=range(3))


@pytest.fixture(scope="module")
def signal_world():
    d, eff = simulate(20000, seed=1, scale=1.0)
    train, test = take(d, slice(0, 15000)), take(d, slice(15000, None))
    out = {"eff": eff, "test": test}
    for name, flags in (("M0", (False, False)), ("M_bat", (True, False)), ("M_full", (True, True))):
        idata = sample_small(train, *flags)
        out[name] = (idata, predictive_scores(idata, test, *flags, max_draws=150))
    return out


@pytest.fixture(scope="module")
def null_world():
    d, _ = simulate(20000, seed=2, scale=0.0)  # nobody is better or worse than anyone else
    train, test = take(d, slice(0, 15000)), take(d, slice(15000, None))
    out = {"test": test}
    for name, flags in (("M0", (False, False)), ("M_full", (True, True))):
        idata = sample_small(train, *flags)
        out[name] = predictive_scores(idata, test, *flags, max_draws=150)
    return out


@pytest.mark.slow
def test_modelling_players_improves_held_out_scores_when_players_differ(signal_world):
    s = {k: signal_world[k][1]["joint"] for k in ("M0", "M_bat", "M_full")}
    n = len(s["M0"])
    assert s["M_full"].mean() > s["M_bat"].mean() > s["M0"].mean()
    assert score_gain(s["M_full"], s["M0"], n)["robust"]
    assert score_gain(s["M_bat"], s["M0"], n)["mean"] > 0.002  # a clear gain, far above sampling noise


@pytest.mark.slow
def test_the_model_recovers_the_true_player_strengths(signal_world):
    idata = signal_world["M_full"][0]
    for part, truth in (("bat_r", signal_world["eff"]["bat_r"]), ("bowl_w", signal_world["eff"]["bowl_w"])):
        z = idata.posterior[f"z_{part}"].values.reshape(-1, 31)
        sigma = idata.posterior[f"sigma_{part}"].values.reshape(-1)
        est = (sigma[:, None] * z).mean(axis=0)[1:]
        assert np.corrcoef(est, truth)[0, 1] > 0.8, part


@pytest.mark.slow
def test_no_gain_is_manufactured_when_players_do_not_differ(null_world):
    n = len(null_world["M0"]["joint"])
    g = score_gain(null_world["M_full"]["joint"], null_world["M0"]["joint"], n)
    assert abs(g["mean"]) < 0.002 and not g["robust"]


@pytest.mark.slow
def test_predictive_scores_are_valid_log_probabilities(signal_world):
    for name in ("M0", "M_bat", "M_full"):
        sc = signal_world[name][1]
        assert np.isfinite(sc["joint"]).all() and (sc["joint"] <= 0).all()
        assert np.isfinite(sc["wicket"]).all() and (sc["wicket"] <= 0).all()
        wk = signal_world["test"].wicket == 1
        assert np.isnan(sc["runs"][wk]).all() and np.isfinite(sc["runs"][~wk]).all()


@pytest.mark.slow
def test_diagnostics_report_convergence_fields(signal_world):
    diag = diagnostics(signal_world["M_full"][0])
    assert {"max_rhat_global", "min_ess_bulk_global", "max_rhat_all", "divergences", "gate_passed"} <= set(diag)
    assert np.isfinite(diag["max_rhat_all"]) and diag["n_draws"] == 500


# ----------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_run_bayes_end_to_end_on_a_synthetic_league(tmp_path):
    raw = tmp_path / "raw"
    write_matches(raw / "synth", n_matches=70, n_players=24, seed=5)
    build(raw_dir=raw, out_dir=tmp_path / "processed")
    report = run_bayes(
        sources=["synth"], train_end="2024-01-01", min_balls=30, draws=100, tune=100, chains=2, cores=1,
        n_boot=100, models={"M0": (False, False), "M_full": (True, True)},
        processed_dir=tmp_path / "processed", reports_dir=tmp_path / "reports",
    )
    d = report["data"]
    assert (d["train_matches"], d["test_matches"]) == (37, 33)
    assert d["test_balls"] == 33 * 240 and d["dropped_odd_runs"] == 0
    assert d["modelled_batters"] == 24 and d["test_balls_with_other_batter"] == 0
    assert set(report["models"]) == {"M0", "M_full"} and "M_full_vs_M0" in report["comparisons"]
    assert report["interpretable"] == all(m["diagnostics"]["gate_passed"] for m in report["models"].values())
    assert set(report["players"]) == {"batter_runs", "batter_wicket", "bowler_runs", "bowler_wicket"}
    saved = json.loads((tmp_path / "reports" / "bayes.json").read_text())
    assert saved["data"]["train_matches"] == 37
