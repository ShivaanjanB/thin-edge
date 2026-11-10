"""Baseline ball-by-ball win probability: two logistic regressions and an honest evaluation.

Protocol
- **Predict before the ball.** Every row is the state *before* a delivery. Post-ball columns are
  never selected (see ``load_frame``), so a model cannot see the ball it is predicting. Rows after
  a result is already certain therefore cannot flatter the score.
- **Two models.** First innings (no target yet) and the chase (target known).
- **Split by time.** Train on matches before ``TRAIN_END``, test on the rest. The test period is the
  IPL Impact Player era, a deliberate stress test for a model trained mostly on older seasons.
- **Match-level uncertainty.** Balls within a match are strongly dependent, so confidence
  intervals resample whole matches, not rows.
- **Reference.** A constant per-innings base rate, so skill is measured against knowing nothing.

Not modelled yet: team and player strength, venue, gender, weather. Those are the job of the
Bayesian model; this baseline only sets the floor it has to beat.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, REPORTS_DIR
from .db import connect

TRAIN_END = "2024-01-01"
RRR_CAP = 36.0
CHECKPOINTS = (0.25, 0.5, 0.75)
CALIBRATION_BINS = 10

CHASE_FEATURES = ["log_runs_needed", "rrr", "wickets_in_hand", "frac_remaining", "wkts_x_frac", "is_hundred"]
FIRST_FEATURES = ["above_par", "wickets_in_hand", "frac_remaining", "par_x_frac", "is_hundred"]
REGIME_FEATURES = ["impact_rule"]


# --------------------------------------------------------------------------------------- data


def source_clause(sources: list[str] | None) -> str:
    """SQL filter for a list of competitions. Names are validated, not trusted, before inlining."""
    if not sources:
        return ""
    for s in sources:
        if not re.fullmatch(r"[a-z0-9_]+", s):
            raise ValueError(f"invalid source name: {s!r}")
    return " and m.source in (" + ", ".join(f"'{s}'" for s in sources) + ")"


def load_frame(con, sources: list[str] | None = None) -> pd.DataFrame:
    """Regulation, full-length, labelled deliveries, using pre-ball columns only.

    ``sources`` limits the competitions (``None`` means all loaded)."""
    df = con.sql(
        f"""
        select s.match_id, s.innings_no, m.source, m.date, m.balls_per_over,
               s.max_balls, s.target_runs,
               s.score_pre, s.wickets_pre, s.legal_balls_pre, s.balls_remaining_pre,
               s.batting_team_won as y
        from states s join matches m using (match_id)
        where s.batting_team_won is not null and not s.is_reduced and not s.is_super_over
              and not exists (  -- a decided chase with no recorded target cannot be modelled
                  select 1 from innings i2
                  where i2.match_id = s.match_id and i2.innings_no = 2
                        and not i2.is_super_over and i2.target_runs is null)
              {source_clause(sources)}
        """
    ).df()
    df["target_runs"] = df["target_runs"].astype(float)
    df["y"] = df["y"].astype(int)
    return df.sort_values(["match_id", "innings_no", "legal_balls_pre"], kind="stable").reset_index(drop=True)


def impact_start_dates(con) -> dict[str, pd.Timestamp]:
    """First date each competition recorded an Impact Player substitution (the rule's start)."""
    rows = con.sql("select source, min(date) from matches where impact_subs > 0 group by 1").fetchall()
    return {source: pd.Timestamp(d) for source, d in rows}


def add_features(df: pd.DataFrame, impact_start: dict[str, pd.Timestamp]) -> pd.DataFrame:
    d = df.copy()
    d["wickets_in_hand"] = 10 - d["wickets_pre"]
    d["frac_remaining"] = d["balls_remaining_pre"] / d["max_balls"]
    d["is_hundred"] = (d["balls_per_over"] == 5).astype(float)
    start = pd.to_datetime(d["source"].map(impact_start))
    d["impact_rule"] = (d["date"] >= start).astype(float)  # NaT (no rule) compares False
    runs_needed = d["target_runs"] - d["score_pre"]
    d["log_runs_needed"] = np.log(runs_needed.clip(lower=1))
    rrr = np.where(
        d["balls_remaining_pre"] > 0,
        runs_needed / d["balls_remaining_pre"].clip(lower=1) * d["balls_per_over"],
        RRR_CAP,
    )
    d["rrr"] = np.clip(rrr, 0, RRR_CAP)
    d["wkts_x_frac"] = d["wickets_in_hand"] * d["frac_remaining"]
    return d


# --------------------------------------------------------------------------------- par curve


@dataclass
class ParCurve:
    """Expected score after each legal ball, by competition (and regime), learned from training."""

    keys: list[str]
    curves: dict[tuple, np.ndarray]
    pooled: np.ndarray
    coarse: dict[tuple, np.ndarray] | None = None

    def at(self, df: pd.DataFrame) -> np.ndarray:
        """Par for each row. An unseen (competition, regime) pair falls back to the same
        competition's curve, and only then to the curve pooled over everything."""
        out = np.empty(len(df))
        balls = df["legal_balls_pre"].to_numpy()
        for key, idx in df.groupby(self.keys).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            curve = self.curves.get(key)
            if curve is None:
                curve = (self.coarse or {}).get(key[:1], self.pooled)
            out[idx] = curve[np.minimum(balls[idx], len(curve) - 1)]
        return out


def _curve(first: pd.DataFrame, window: int) -> np.ndarray:
    by_ball = first.groupby("legal_balls_pre")["score_pre"].mean()
    full = by_ball.reindex(range(int(by_ball.index.max()) + 1)).interpolate(limit_direction="both")
    return full.rolling(window, center=True, min_periods=1).mean().to_numpy()


def fit_par_curve(first: pd.DataFrame, keys: list[str], window: int = 9) -> ParCurve:
    def by(cols):
        return {
            (k if isinstance(k, tuple) else (k,)): _curve(grp, window) for k, grp in first.groupby(cols)
        }

    coarse = by(keys[:1]) if len(keys) > 1 else None
    return ParCurve(keys, by(keys), _curve(first, window), coarse)


# --------------------------------------------------------------------------------- the model


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class Logit:
    """L2-regularised logistic regression on standardised features, fitted by Newton's method.

    Minimises ``sum(log loss) + (1 / (2C)) * ||w||^2`` with the intercept unpenalised, the same
    objective as scikit-learn's ``LogisticRegression(C=C)``. Written directly in numpy because the
    project runs on a machine whose application-control policy blocks scikit-learn's compiled
    extensions. ``coef_`` is on standardised features, so sizes are comparable across features.
    """

    def __init__(self, c: float = 1.0, max_iter: int = 100, tol: float = 1e-9):
        self.c, self.max_iter, self.tol = c, max_iter, tol

    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Logit: X contains NaN or infinite values")
        z = (X - self.mean_) / self.scale_
        return np.hstack([np.ones((len(z), 1)), z])

    def _loss(self, A, y, w, reg) -> float:
        eta = A @ w
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.sum(reg * w * w))

    def fit(self, X, y) -> "Logit":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("Logit: X and y must be finite (found NaN or infinite values)")
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        A = self._design(X)
        reg = np.full(A.shape[1], 1.0 / self.c)
        reg[0] = 0.0
        w = np.zeros(A.shape[1])
        loss = self._loss(A, y, w, reg)
        for _ in range(self.max_iter):
            p = _sigmoid(A @ w)
            grad = A.T @ (p - y) + reg * w
            hess = A.T @ (A * (p * (1 - p))[:, None]) + np.diag(reg) + 1e-10 * np.eye(A.shape[1])
            step = np.linalg.solve(hess, grad)
            scale = 1.0
            for _ in range(30):  # halve the step if the objective would increase
                trial = w - scale * step
                trial_loss = self._loss(A, y, trial, reg)
                if trial_loss <= loss:
                    break
                scale /= 2
            w, loss = trial, trial_loss
            if np.max(np.abs(scale * step)) < self.tol:
                break
        self.intercept_, self.coef_ = float(w[0]), w[1:]
        return self

    def proba(self, X) -> np.ndarray:
        A = self._design(X)
        return _sigmoid(A @ np.concatenate([[self.intercept_], self.coef_]))


def _logit(c: float) -> Logit:
    return Logit(c)


def _finite(X: pd.DataFrame, where: str) -> pd.DataFrame:
    """Fail loudly, naming the columns, rather than let NaN flow silently through a model."""
    bad = [c for c in X.columns if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
    if bad:
        raise ValueError(f"{where}: non-finite values in {bad} (for example a chase with no target)")
    return X


class Baseline:
    def __init__(self, use_regime: bool = True, c: float = 1.0):
        self.use_regime = use_regime
        self.c = c

    @property
    def _extra(self) -> list[str]:
        return REGIME_FEATURES if self.use_regime else []

    def _first_x(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["above_par"] = d["score_pre"] - self.par.at(d)
        d["par_x_frac"] = d["above_par"] * d["frac_remaining"]
        return d[FIRST_FEATURES + self._extra]

    def fit(self, train: pd.DataFrame, impact_start: dict[str, pd.Timestamp]) -> "Baseline":
        self.impact_start = impact_start
        d = add_features(train, impact_start)
        first, chase = d[d["innings_no"] == 1], d[d["innings_no"] == 2]
        self.par = fit_par_curve(first, ["source"] + self._extra)
        self.first_model = _logit(self.c).fit(_finite(self._first_x(first), "first-innings features"), first["y"])
        self.chase_model = _logit(self.c).fit(
            _finite(chase[CHASE_FEATURES + self._extra], "chase features"), chase["y"]
        )
        return self

    def with_fresh_par(self, history: pd.DataFrame, impact_start: dict[str, pd.Timestamp]) -> "Baseline":
        """Copy of this model with only the par curve refit on ``history``; coefficients unchanged."""
        d = add_features(history, impact_start)
        clone = copy.copy(self)
        clone.par = fit_par_curve(d[d["innings_no"] == 1], ["source"] + self._extra)
        return clone

    def predict(self, df: pd.DataFrame) -> pd.Series:
        d = add_features(df, self.impact_start)
        p = pd.Series(np.nan, index=df.index)
        first, chase = d["innings_no"] == 1, d["innings_no"] == 2
        if first.any():
            p[first] = self.first_model.proba(_finite(self._first_x(d[first]), "first-innings features"))
        if chase.any():
            p[chase] = self.chase_model.proba(
                _finite(d.loc[chase, CHASE_FEATURES + self._extra], "chase features")
            )
        return p

    def coefficients(self) -> dict[str, dict[str, float]]:
        """Coefficients on standardised features, so their sizes are comparable."""
        def pairs(model, names):
            return {n: round(float(c), 4) for n, c in zip(names, model.coef_)}
        return {
            "first_innings": pairs(self.first_model, FIRST_FEATURES + self._extra),
            "chase": pairs(self.chase_model, CHASE_FEATURES + self._extra),
        }


# ------------------------------------------------------------------------------- evaluation


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y, p, bins: int = CALIBRATION_BINS) -> tuple[list[dict], float]:
    """Equal-width reliability bins and the expected calibration error (count-weighted gap)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = which == b
        n = int(m.sum())
        if n == 0:
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "mean_pred": mp, "mean_obs": my})
        ece += n / len(p) * abs(mp - my)
    return rows, float(ece)


def match_bootstrap(frame: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap totals of ``cols`` (plus row counts ``n``) by resampling whole matches."""
    per = frame.groupby("match_id")[cols].sum()
    per["n"] = frame.groupby("match_id").size()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    return {c: per[c].to_numpy()[idx].sum(axis=1) for c in per.columns}


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarise(f: pd.DataFrame, n_boot: int, seed: int) -> dict:
    f = f.assign(se_m=(f["p"] - f["y"]) ** 2, se_r=(f["ref"] - f["y"]) ** 2)
    boot = match_bootstrap(f, ["se_m", "se_r"], n_boot, seed)
    ece = calibration_table(f["y"], f["p"])[1]
    return {
        "n_rows": int(len(f)),
        "n_matches": int(f["match_id"].nunique()),
        "brier": brier(f["y"], f["p"]),
        "brier_ci": _ci(boot["se_m"] / boot["n"]),
        "ref_brier": brier(f["y"], f["ref"]),
        "skill": 1 - float(f["se_m"].sum() / f["se_r"].sum()),
        "skill_ci": _ci(1 - boot["se_m"] / boot["se_r"]),
        "log_loss": log_loss(f["y"], f["p"]),
        "ece": ece,
    }


def checkpoint_table(f: pd.DataFrame) -> list[dict]:
    """Brier at fixed points of the innings (the first delivery after k legal balls)."""
    rows = []
    for innings in (1, 2):
        sub = f[f["innings_no"] == innings]
        for frac in CHECKPOINTS:
            k = (sub["max_balls"] * frac).round().astype(int)
            at = sub[sub["legal_balls_pre"] == k].groupby(["match_id", "innings_no"]).head(1)
            if at.empty:
                continue
            rows.append(
                {
                    "innings": innings,
                    "fraction_of_innings": frac,
                    "n": int(len(at)),
                    "brier": brier(at["y"], at["p"]),
                    "ref_brier": brier(at["y"], at["ref"]),
                }
            )
    return rows


def evaluate(test: pd.DataFrame, p: pd.Series, ref: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    f = test.assign(p=p, ref=ref)
    return {
        "metrics": {
            "all": summarise(f, n_boot, seed),
            "first_innings": summarise(f[f["innings_no"] == 1], n_boot, seed),
            "chase": summarise(f[f["innings_no"] == 2], n_boot, seed),
        },
        "checkpoints": checkpoint_table(f),
        "calibration": {
            name: calibration_table(g["y"], g["p"])[0]
            for name, g in (("first_innings", f[f["innings_no"] == 1]), ("chase", f[f["innings_no"] == 2]))
        },
    }


def paired_brier_gain(test: pd.DataFrame, p_a: pd.Series, p_b: pd.Series, n_boot: int, seed: int) -> dict:
    """Brier(b) - Brier(a) with a match-level CI. Positive means model ``a`` is better."""
    f = test.assign(d=(p_b - test["y"]) ** 2 - (p_a - test["y"]) ** 2)
    boot = match_bootstrap(f, ["d"], n_boot, seed)
    mean_gain = boot["d"] / boot["n"]
    return {"mean": float(f["d"].mean()), "ci": _ci(mean_gain)}


def run_baseline(
    train_end: str = TRAIN_END,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    n_boot: int = 1000,
    sources: list[str] | None = None,
) -> dict:
    con = connect(processed_dir)
    df = load_frame(con, sources)
    impact = impact_start_dates(con)
    cut = pd.Timestamp(train_end)
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    if train.empty or test.empty:
        raise ValueError(f"train_end {train_end} leaves an empty train or test set")

    base_rate = train.groupby("innings_no")["y"].mean()
    ref = test["innings_no"].map(base_rate)

    report: dict = {
        "sources": sorted(sources) if sources else "all",
        "train_end": train_end,
        "split": {
            "train_matches": int(train["match_id"].nunique()),
            "test_matches": int(test["match_id"].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_matches_by_source": {k: int(v) for k, v in test.groupby("source")["match_id"].nunique().items()},
        },
        "impact_rule_start": {k: v.date().isoformat() for k, v in impact.items()},
        "base_rates": {int(k): float(v) for k, v in base_rate.items()},
        "variants": {},
    }
    preds = {}
    for name, use_regime in (("with_regime", True), ("no_regime", False)):
        model = Baseline(use_regime).fit(train, impact)
        preds[name] = model.predict(test)
        report["variants"][name] = {
            "coefficients": model.coefficients(),
            **evaluate(test, preds[name], ref, n_boot),
        }
    report["regime_gain"] = {
        "description": "Brier(no_regime) - Brier(with_regime) on the test set; positive favours the regime flag",
        **paired_brier_gain(test, preds["with_regime"], preds["no_regime"], n_boot, seed=1),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
