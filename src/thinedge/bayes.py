"""Hierarchical Bayesian ball-outcome model: do batter and bowler strength predict individual balls?

The baseline win-probability model knows nothing about who is batting or bowling, which is why the
first innings has so little to go on before a ball is bowled. This module tests, at the level of single
deliveries, whether player strength carries predictive information, before any of it is built into a
win-probability simulator.

The model (legal deliveries of full-length regulation matches; wides and no-balls are not modelled)
    wicket   ~ Bernoulli(logit p = a_w[group] + bat_w[batter] + bowl_w[bowler])
    runs     ~ OrderedLogistic(eta = a_r[group] + bat_r[batter] + bowl_r[bowler], cutpoints)
               for balls with no wicket; categories 0, 1, 2, 3, 4, 6 off the bat (5 and 7+ are rare and
               dropped). Ordered by runs scored, with one shared set of cutpoints.
    group    = phase (powerplay / middle / death) x innings (1 or 2): six baseline levels.
    players  = non-centred, partially pooled effects, one set per role and per part, so a batter's
               and a bowler's strength are separate. Players with fewer than ``MIN_BALLS`` training balls
               (and every player first seen after training) share one "other" effect.

Not modelled here: venue, a scoring environment that changes over seasons, team effects, wides and
no-balls, byes and leg byes, or any link to match outcome. The ordered-logistic form assumes a single
"scoring" dimension, so a batter who both blocks and hits big is only partly captured.

Variants compared on the same data
    M0      no player effects (group baselines only)
    M_bat   batter effects only
    M_full  batter and bowler effects

Evaluation
    Train on matches before ``TRAIN_END``, score every ball from then on (a time split). The score is the
    held-out log predictive probability per ball, averaging the model's probability over posterior draws
    (a proper scoring rule that includes the uncertainty about each player). Balls within a match are
    dependent, so intervals resample whole matches.

Pre-registered before any real result was seen
    H1: M_full has a higher held-out mean log score per ball than M0, with the 95% match-level interval
        entirely above zero for every one of five bootstrap seeds.
    H2: M_full beats M_bat in the same way (does the bowler add information beyond the batter?).
    Validity gate: a fit is interpreted only if, for the global parameters (group baselines, cutpoints,
    player standard deviations), R-hat <= 1.01 and bulk effective sample size >= 400; for every parameter
    R-hat <= 1.02; and divergent transitions <= 0.1% of draws. A fit that fails the gate is reported as
    not interpretable, never as a result.
    Descriptive only, not tests: which part (wicket or runs) benefits, the share of test balls with an
    unseen or rare player, and a sanity check that the top and bottom players look plausible.
    Effects are expected to be small (hundredths of a nat per ball or less). Any gain is reported with its
    size, not just its sign.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import match_bootstrap, source_clause
from .config import PROCESSED_DIR, REPORTS_DIR
from .db import connect

RUN_VALUES = (0, 1, 2, 3, 4, 6)
N_RUN = len(RUN_VALUES)
N_GROUPS = 6
TRAIN_END = "2024-01-01"
MIN_BALLS = 60
CUT_INIT = np.array([-0.5, 1.0, 1.5, 1.6, 2.7])  # about the cumulative logits of an average IPL ball
MODELS = {"M0": (False, False), "M_bat": (True, False), "M_full": (True, True)}
GLOBALS = ("a_w", "a_r", "cut", "sigma_bat_w", "sigma_bat_r", "sigma_bowl_w", "sigma_bowl_r")


# ---------------------------------------------------------------------------------- data


def load_balls(con, sources: list[str] | None = None) -> pd.DataFrame:
    """Legal deliveries of regulation, full-length matches (reduced and super-over balls excluded)."""
    return con.sql(
        f"""
        select b.match_id, b.innings_no, b.over_idx, b.in_powerplay, b.batter, b.bowler,
               b.batter_id, b.bowler_id, b.runs_batter, b.is_wicket, m.date, m.source
        from balls b join matches m using (match_id)
        where b.is_legal and not b.is_super_over and b.innings_no in (1, 2)
              and b.match_id not in (select match_id from states where is_reduced)
              {source_clause(sources)}
        order by b.match_id, b.innings_no, b.delivery_seq
        """
    ).df()


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add outcome codes; drop balls the model cannot represent, and say how many."""
    n0 = len(df)
    d = df.dropna(subset=["batter_id", "bowler_id"]).copy()
    n_no_id = n0 - len(d)
    d["wicket"] = d["is_wicket"].astype(int)
    d["run_cat"] = d["runs_batter"].map({v: i for i, v in enumerate(RUN_VALUES)}).fillna(-1).astype(int)
    odd = (d["wicket"] == 0) & (d["run_cat"] < 0)
    d = d[~odd].copy()
    phase = np.where(d["in_powerplay"], 0, np.where(d["over_idx"] >= 16, 2, 1))
    d["group"] = phase * 2 + (d["innings_no"].to_numpy() - 1)
    return d.reset_index(drop=True), {"balls_in": n0, "dropped_no_player_id": int(n_no_id), "dropped_odd_runs": int(odd.sum())}


def player_index(train: pd.Series, other: pd.Series, min_balls: int = MIN_BALLS):
    """Integer index per player, learned from ``train`` only. 0 is the shared "other" level, used for
    players with fewer than ``min_balls`` training balls and for anyone first seen later."""
    counts = train.value_counts()
    levels = sorted(counts[counts >= min_balls].index)
    lut = {p: i + 1 for i, p in enumerate(levels)}
    return (
        train.map(lut).fillna(0).astype(int).to_numpy(),
        other.map(lut).fillna(0).astype(int).to_numpy(),
        len(levels) + 1,
        levels,
    )


@dataclass
class BallData:
    group: np.ndarray
    bat: np.ndarray
    bowl: np.ndarray
    wicket: np.ndarray
    run_cat: np.ndarray
    n_bat: int
    n_bowl: int

    def __len__(self) -> int:
        return len(self.group)


def make_data(frame: pd.DataFrame, bat: np.ndarray, bowl: np.ndarray, n_bat: int, n_bowl: int) -> BallData:
    return BallData(
        group=frame["group"].to_numpy(), bat=bat, bowl=bowl,
        wicket=frame["wicket"].to_numpy(), run_cat=frame["run_cat"].to_numpy(), n_bat=n_bat, n_bowl=n_bowl,
    )


# --------------------------------------------------------------------------------- model


def build_model(d: BallData, use_batter: bool, use_bowler: bool):
    import pymc as pm
    from pymc.distributions.transforms import ordered

    nw = np.flatnonzero(d.wicket == 0)
    with pm.Model() as model:
        a_w = pm.Normal("a_w", -3.0, 1.5, shape=N_GROUPS)
        a_r = pm.Normal("a_r", 0.0, 1.0, shape=N_GROUPS)
        cut = pm.Normal("cut", mu=CUT_INIT, sigma=1.5, shape=N_RUN - 1, transform=ordered, initval=CUT_INIT)
        eta_w, eta_r = a_w[d.group], a_r[d.group]
        for role, on, idx, n in (("bat", use_batter, d.bat, d.n_bat), ("bowl", use_bowler, d.bowl, d.n_bowl)):
            if not on:
                continue
            for part in ("w", "r"):
                sigma = pm.HalfNormal(f"sigma_{role}_{part}", 0.5)
                z = pm.Normal(f"z_{role}_{part}", 0.0, 1.0, shape=n)
                effect = (sigma * z)[idx]
                if part == "w":
                    eta_w = eta_w + effect
                else:
                    eta_r = eta_r + effect
        pm.Bernoulli("wicket", logit_p=eta_w, observed=d.wicket)
        pm.OrderedLogistic("runs", eta=eta_r[nw], cutpoints=cut, observed=d.run_cat[nw])
    return model


def fit(model, draws: int, tune: int, chains: int, cores: int, seed: int):
    import pymc as pm

    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, random_seed=seed,
            progressbar=False, target_accept=0.9,
        )


def diagnostics(idata) -> dict:
    import arviz as az

    names = list(idata.posterior.data_vars)
    summ = az.summary(idata, var_names=names)
    rc = [c for c in summ.columns if "r_hat" in str(c).lower() or "rhat" in str(c).lower()][0]
    ec = [c for c in summ.columns if "ess_bulk" in str(c).lower()][0]
    glob = summ.index.map(lambda s: str(s).split("[")[0]).isin(GLOBALS)
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    div = int(idata.sample_stats["diverging"].sum())
    out = {
        "n_parameters": int(len(summ)),
        "max_rhat_global": float(summ.loc[glob, rc].max()),
        "min_ess_bulk_global": float(summ.loc[glob, ec].min()),
        "max_rhat_all": float(summ[rc].max()),
        "min_ess_bulk_all": float(summ[ec].min()),
        "divergences": div,
        "n_draws": n_draws,
    }
    out["gate_passed"] = bool(
        out["max_rhat_global"] <= 1.01 and out["min_ess_bulk_global"] >= 400
        and out["max_rhat_all"] <= 1.02 and div <= 0.001 * n_draws
    )
    return out


# ----------------------------------------------------------------------------- prediction


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def outcome_probs(eta_w: np.ndarray, eta_r: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """(n, 1 + N_RUN) probabilities: column 0 is a wicket, columns 1.. are run categories given no wicket."""
    pw = _sigmoid(eta_w)
    cum = _sigmoid(cut[None, :] - eta_r[:, None])
    n = len(eta_r)
    run = np.diff(np.concatenate([np.zeros((n, 1)), cum, np.ones((n, 1))], axis=1), axis=1)
    return np.concatenate([pw[:, None], (1 - pw)[:, None] * run], axis=1)


def _linear_predictors(draw: dict, d: BallData, use_batter: bool, use_bowler: bool):
    eta_w, eta_r = draw["a_w"][d.group].copy(), draw["a_r"][d.group].copy()
    for role, on, idx in (("bat", use_batter, d.bat), ("bowl", use_bowler, d.bowl)):
        if on:
            eta_w += (draw[f"sigma_{role}_w"] * draw[f"z_{role}_w"])[idx]
            eta_r += (draw[f"sigma_{role}_r"] * draw[f"z_{role}_r"])[idx]
    return eta_w, eta_r


def _flat(idata, name: str) -> np.ndarray:
    a = idata.posterior[name].values
    return a.reshape(-1, *a.shape[2:])


def predictive_scores(idata, d: BallData, use_batter: bool, use_bowler: bool, max_draws: int = 300, seed: int = 0) -> dict:
    """Held-out log predictive probability of each ball, averaged over posterior draws.

    ``joint`` is the probability of the whole outcome (wicket, or no wicket and the runs). ``wicket`` and
    ``runs`` score the two parts as separate predictive distributions (the runs score is NaN on wicket
    balls). Each is the log of the posterior-mean probability, a proper scoring rule."""
    names = ["a_w", "a_r", "cut"] + [f"{p}_{r}_{q}" for r, on in (("bat", use_batter), ("bowl", use_bowler)) if on
                                      for p in ("sigma", "z") for q in ("w", "r")]
    arrays = {k: _flat(idata, k) for k in names}
    total = next(iter(arrays.values())).shape[0]
    picks = np.sort(np.random.default_rng(seed).choice(total, size=min(max_draws, total), replace=False))
    n = len(d)
    rows = np.arange(n)
    col = np.where(d.wicket == 1, 0, 1 + np.clip(d.run_cat, 0, None))
    wk = d.wicket == 1
    p_joint = np.zeros(n); p_wick = np.zeros(n); p_runs = np.zeros(n)
    for s in picks:
        draw = {k: v[s] for k, v in arrays.items()}
        eta_w, eta_r = _linear_predictors(draw, d, use_batter, use_bowler)
        probs = outcome_probs(eta_w, eta_r, draw["cut"])
        p_joint += probs[rows, col]
        pw = probs[:, 0]
        p_wick += np.where(wk, pw, 1 - pw)
        p_runs += probs[rows, col] / np.where(wk, 1.0, np.maximum(1 - pw, 1e-12))
    k = len(picks)
    runs = np.where(wk, np.nan, np.log(p_runs / k))
    return {"joint": np.log(p_joint / k), "wicket": np.log(p_wick / k), "runs": runs, "n_draws_used": int(k)}


def paired_logscore_gain(frame: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seeds=range(5)) -> dict:
    """Mean per-ball log-score gain of column ``a`` over ``b`` with match-level 95% intervals.

    Positive means ``a`` predicts better. ``robust`` asks for the interval's lower bound to stay above
    zero for every bootstrap seed, because a bound near zero can flip with the seed."""
    f = frame.assign(_d=frame[a] - frame[b]).dropna(subset=["_d"])
    lowers, first = [], None
    for seed in seeds:
        boot = match_bootstrap(f, ["_d"], n_boot, seed)
        mean_gain = boot["_d"] / boot["n"]
        lowers.append(float(np.percentile(mean_gain, 2.5)))
        if first is None:
            first = [float(np.percentile(mean_gain, 2.5)), float(np.percentile(mean_gain, 97.5))]
    return {
        "mean": float(f["_d"].mean()), "ci": first,
        "lower_across_seeds": [min(lowers), max(lowers)],
        "robust": bool(min(lowers) > 0),
    }


# -------------------------------------------------------------------------------- the run


def _names(frame: pd.DataFrame, id_col: str, name_col: str) -> dict:
    return frame.groupby(id_col)[name_col].agg(lambda s: s.value_counts().index[0]).to_dict()


def player_table(idata, levels, frame, id_col, name_col, effect: str, top: int = 8) -> dict:
    """Posterior-mean effect of each modelled player (descriptive sanity check, not a test)."""
    z = _flat(idata, f"z_{effect}"); sigma = _flat(idata, f"sigma_{effect}")
    eff = (sigma[:, None] * z).mean(axis=0)
    balls = frame[id_col].value_counts()
    names = _names(frame, id_col, name_col)
    rows = [(names.get(p, p), float(eff[i + 1]), int(balls.get(p, 0))) for i, p in enumerate(levels)]
    rows.sort(key=lambda r: r[1])
    return {"lowest": rows[:top], "highest": rows[::-1][:top]}


def run_bayes(
    sources: list[str] | None = ("ipl",),
    train_end: str = TRAIN_END,
    min_balls: int = MIN_BALLS,
    draws: int = 500, tune: int = 500, chains: int = 4, cores: int = 4, seed: int = 1,
    n_boot: int = 2000,
    models: dict | None = None,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    con = connect(processed_dir)
    balls, dropped = encode(load_balls(con, list(sources) if sources else None))
    cut = pd.Timestamp(train_end)
    train, test = balls[balls["date"] < cut].reset_index(drop=True), balls[balls["date"] >= cut].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"train_end {train_end} leaves an empty train or test set")

    bat_tr, bat_te, n_bat, bat_levels = player_index(train["batter_id"], test["batter_id"], min_balls)
    bowl_tr, bowl_te, n_bowl, bowl_levels = player_index(train["bowler_id"], test["bowler_id"], min_balls)
    d_train = make_data(train, bat_tr, bowl_tr, n_bat, n_bowl)
    d_test = make_data(test, bat_te, bowl_te, n_bat, n_bowl)

    report = {
        "sources": list(sources) if sources else "all",
        "train_end": train_end, "min_balls": min_balls,
        "sampler": {"draws": draws, "tune": tune, "chains": chains, "seed": seed},
        "data": {
            **dropped,
            "train_balls": len(train), "test_balls": len(test),
            "train_matches": int(train["match_id"].nunique()), "test_matches": int(test["match_id"].nunique()),
            "modelled_batters": n_bat - 1, "modelled_bowlers": n_bowl - 1,
            "test_balls_with_other_batter": float((bat_te == 0).mean()),
            "test_balls_with_other_bowler": float((bowl_te == 0).mean()),
        },
        "models": {},
    }
    scored = test[["match_id"]].copy()
    fits = {}
    for name, (use_bat, use_bowl) in (models or MODELS).items():
        t0 = time.time()
        idata = fit(build_model(d_train, use_bat, use_bowl), draws, tune, chains, cores, seed)
        seconds = time.time() - t0
        diag = diagnostics(idata)
        sc = predictive_scores(idata, d_test, use_bat, use_bowl)
        for part in ("joint", "wicket", "runs"):
            scored[f"{name}:{part}"] = sc[part]
        report["models"][name] = {
            "fit_seconds": round(seconds, 1), "diagnostics": diag,
            "mean_log_score": {p: float(np.nanmean(sc[p])) for p in ("joint", "wicket", "runs")},
        }
        fits[name] = idata
        print(f"{name}: {seconds:.0f}s, gate {'passed' if diag['gate_passed'] else 'FAILED'}, "
              f"mean log score {report['models'][name]['mean_log_score']['joint']:.5f}", flush=True)

    names = list(report["models"])
    gate_ok = all(report["models"][n]["diagnostics"]["gate_passed"] for n in names)
    comparisons = {}
    for a, b in (("M_full", "M0"), ("M_full", "M_bat"), ("M_bat", "M0")):
        if a in names and b in names:
            comparisons[f"{a}_vs_{b}"] = {
                part: paired_logscore_gain(scored, f"{a}:{part}", f"{b}:{part}", n_boot)
                for part in ("joint", "wicket", "runs")
            }
    report["comparisons"] = comparisons
    report["interpretable"] = gate_ok
    if "M_full" in fits:
        report["players"] = {
            "batter_runs": player_table(fits["M_full"], bat_levels, train, "batter_id", "batter", "bat_r"),
            "batter_wicket": player_table(fits["M_full"], bat_levels, train, "batter_id", "batter", "bat_w"),
            "bowler_runs": player_table(fits["M_full"], bowl_levels, train, "bowler_id", "bowler", "bowl_r"),
            "bowler_wicket": player_table(fits["M_full"], bowl_levels, train, "bowler_id", "bowler", "bowl_w"),
        }
        report["player_effect_sd"] = {
            k: float(_flat(fits["M_full"], k).mean()) for k in
            ("sigma_bat_w", "sigma_bat_r", "sigma_bowl_w", "sigma_bowl_r")
        }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "bayes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
