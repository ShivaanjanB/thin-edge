"""Property-based tests (Hypothesis).

Example tests check a handful of hand-picked matches. These check rules that must hold for every
input: a generated match must parse into tables that add up, the state table must not depend on row
order, the logistic fit must not care how features are scaled, and so on. Each property is something
the code could actually fail; none restates the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from thinedge.baseline import (
    Logit,
    _sigmoid,
    calibration_table,
    match_bootstrap,
    paired_brier_gain,
)
from thinedge.features import build_state_table
from thinedge.parse import overs_to_balls, parse_match
from thinedge.validate import quality_report

SLOW = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FAST = settings(max_examples=100, deadline=None)

# ---------------------------------------------------------------------------------- generators

KINDS = ["bowled", "caught", "lbw", "run out", "stumped", "retired hurt", "retired not out"]
NOT_OUT = {"retired hurt", "retired not out"}  # recorded as wickets by Cricsheet, but nobody is out
EXTRAS = ["wides", "noballs", "byes", "legbyes"]


@st.composite
def innings_overs(draw, bpo: int, max_legal: int, stop_at: int | None = None, min_balls: int = 0):
    """A list of overs in Cricsheet shape. Stops at ``max_legal`` legal balls, 10 wickets, or
    (for a chase) as soon as ``stop_at`` runs are reached."""
    overs, legal, wickets, score = [], 0, 0, 0
    n_target = draw(st.integers(min_balls, max_legal + 6))
    sent = 0
    current: list[dict] = []
    over_no = 0
    while sent < n_target and legal < max_legal and wickets < 10:
        if stop_at is not None and score >= stop_at:
            break
        runs = draw(st.sampled_from([0, 0, 0, 1, 1, 2, 3, 4, 6]))
        extra_kind = draw(st.one_of(st.none(), st.sampled_from(EXTRAS)))
        extra_runs = draw(st.integers(1, 2)) if extra_kind else 0
        if extra_kind in ("wides",):
            runs = 0
        d = {
            "batter": "Bat",
            "bowler": "Bowl",
            "non_striker": "NS",
            "runs": {"batter": runs, "extras": extra_runs, "total": runs + extra_runs},
        }
        if extra_kind:
            d["extras"] = {extra_kind: extra_runs}
        kind = draw(st.one_of(st.none(), st.none(), st.none(), st.sampled_from(KINDS)))
        if kind and not (kind not in NOT_OUT and extra_kind == "wides" and kind in ("bowled", "caught", "lbw")):
            d["wickets"] = [{"player_out": "Bat", "kind": kind}]
            # now and then a second wicket on the same delivery, in either order (seen in real data)
            if wickets + (kind not in NOT_OUT) < 10 and draw(st.integers(0, 9)) == 0:
                second = draw(st.sampled_from(KINDS + ["retired out", "retired not out"]))
                d["wickets"].append({"player_out": "NS", "kind": second})
                if draw(st.booleans()):
                    d["wickets"].reverse()
            wickets += sum(w["kind"] not in NOT_OUT for w in d["wickets"])
        is_legal = extra_kind not in ("wides", "noballs")
        legal += is_legal
        score += runs + extra_runs
        current.append(d)
        sent += 1
        if sum(1 for x in current if "wides" not in x.get("extras", {}) and "noballs" not in x.get("extras", {})) == bpo:
            overs.append({"over": over_no, "deliveries": current})
            current, over_no = [], over_no + 1
    if current:
        overs.append({"over": over_no, "deliveries": current})
    return overs


def innings_total(overs) -> int:
    """Runs off the deliveries only; penalty runs are added separately."""
    return sum(d["runs"]["total"] for o in overs for d in o["deliveries"])


def dismissals_in(overs) -> int:
    return sum(w["kind"] not in NOT_OUT for o in overs for d in o["deliveries"] for w in d.get("wickets", []))


def penalty(inn) -> tuple[int, int]:
    p = inn.get("penalty_runs", {})
    return p.get("pre", 0), p.get("post", 0)


@st.composite
def match_docs(draw):
    bpo = draw(st.sampled_from([5, 6]))
    nominal = draw(st.integers(1, 4))
    cap = nominal * bpo
    first = draw(innings_overs(bpo, cap, min_balls=1))
    pen1 = draw(st.one_of(st.just({}), st.fixed_dictionaries({}, optional={"pre": st.integers(1, 6), "post": st.integers(1, 6)})))
    pen2 = draw(st.one_of(st.just({}), st.fixed_dictionaries({"pre": st.integers(1, 6)})))
    t1 = innings_total(first) + sum(pen1.values())
    # the chase target already counts penalty runs awarded to the chasing side before it starts
    second = draw(innings_overs(bpo, cap, stop_at=t1 + 1 - pen2.get("pre", 0)))
    t2 = innings_total(second) + sum(pen2.values())
    outcome = draw(st.sampled_from(["natural", "no_result"]))
    if outcome == "no_result":
        result = {"result": "no result"}
    elif t2 > t1:
        result = {"winner": "Team B", "by": {"wickets": 10 - dismissals_in(second)}}
    elif t2 < t1:
        result = {"winner": "Team A", "by": {"runs": t1 - t2}}
    else:
        result = {"result": "tie"}
    date = draw(st.dates(min_value=pd.Timestamp("2015-01-01").date(), max_value=pd.Timestamp("2026-12-31").date()))
    return {
        "meta": {"data_version": "1.3.0"},
        "info": {
            "balls_per_over": bpo, "overs": nominal, "gender": "male", "match_type": "T20",
            "team_type": "club", "season": str(date.year), "dates": [date.isoformat()],
            "venue": "V", "city": "C", "teams": ["Team A", "Team B"],
            "outcome": result, "registry": {"people": {"Bat": "id-bat", "Bowl": "id-bowl", "NS": "id-ns"}},
        },
        "innings": [
            {"team": "Team A", "overs": first, "powerplays": [{"from": 0.1, "to": 0.6}],
             **({"penalty_runs": pen1} if pen1 else {})},
            {"team": "Team B", "overs": second, "target": {"overs": nominal, "runs": t1 + 1},
             "powerplays": [{"from": 0.1, "to": 0.6}], **({"penalty_runs": pen2} if pen2 else {})},
        ],
    }


def tables(docs):
    ms, ins, bs = [], [], []
    for i, doc in enumerate(docs, start=1):
        m, i_, b = parse_match(i, doc, "synth")
        ms.append(m); ins += i_; bs += b
    matches = pd.DataFrame(ms)
    matches["date"] = pd.to_datetime(matches["date"])
    innings = pd.DataFrame(ins)
    innings["target_runs"] = innings["target_runs"].astype("Int64")
    balls = pd.DataFrame(bs)
    return matches, innings, balls


# -------------------------------------------------------------------------------- over notation


@given(st.sampled_from([5, 6]), st.integers(0, 400))
@example(5, 0)
@example(6, 119)
@FAST
def test_over_notation_roundtrips(bpo, balls):
    """Writing a ball count as Cricsheet 'overs.balls' and converting back is the identity."""
    notation = balls // bpo + (balls % bpo) / 10
    assert overs_to_balls(notation, bpo) == balls


@given(st.sampled_from([5, 6]), st.integers(0, 40), st.integers(0, 9))
@FAST
def test_over_notation_rejects_impossible_ball_counts(bpo, whole, partial):
    notation = whole + partial / 10
    if partial < bpo:
        assert overs_to_balls(notation, bpo) == whole * bpo + partial
    else:
        with pytest.raises(ValueError):
            overs_to_balls(notation, bpo)


# ------------------------------------------------------------------------------- parse + states


@given(st.lists(match_docs(), min_size=1, max_size=3))
@SLOW
def test_parsed_tables_add_up(docs):
    """Innings totals, legal counts and wicket counts equal what the raw JSON says, ball by ball."""
    matches, innings, balls = tables(docs)
    assert len(balls) == sum(len(o["deliveries"]) for d in docs for i in d["innings"] for o in i["overs"])
    for (mid, inn), b in balls.groupby(["match_id", "innings_no"]):
        row = innings[(innings.match_id == mid) & (innings.innings_no == inn)].iloc[0]
        raw_innings = docs[mid - 1]["innings"][inn - 1]
        raw = raw_innings["overs"]
        assert row.total_runs == innings_total(raw) + sum(penalty(raw_innings)) == b.runs_total.sum() + sum(penalty(raw_innings))
        assert row.legal_balls == int(b.is_legal.sum())
        assert row.wickets_lost == int(b.n_wickets.sum()) == dismissals_in(raw)
        assert int(b.is_wicket.sum()) <= row.wickets_lost  # a delivery can carry two dismissals
        assert (b.is_wicket == (b.n_wickets > 0)).all()
        assert (b.runs_total == b.runs_batter + b.runs_extras).all()
        assert b.delivery_seq.tolist() == list(range(1, len(b) + 1))


@given(st.lists(match_docs(), min_size=1, max_size=3))
@SLOW
def test_state_table_is_internally_consistent(docs):
    matches, innings, balls = tables(docs)
    states = build_state_table(matches, innings, balls)
    assert len(states) == len(balls)
    assert not states.duplicated(["match_id", "innings_no", "delivery_seq"]).any()
    for (mid, inn), s in states.groupby(["match_id", "innings_no"]):
        s = s.sort_values("delivery_seq")
        # what the ball started with is what the previous ball ended with; an innings starts on its
        # pre-innings penalty runs, if any
        start = penalty(docs[mid - 1]["innings"][inn - 1])[0]
        assert s.score_pre.iloc[0] == start and s.wickets_pre.iloc[0] == 0 and s.legal_balls_pre.iloc[0] == 0
        assert (s.score_pre.iloc[1:].to_numpy() == s.score_post.iloc[:-1].to_numpy()).all()
        assert (s.wickets_pre.iloc[1:].to_numpy() == s.wickets_post.iloc[:-1].to_numpy()).all()
        assert (s.legal_balls_pre.iloc[1:].to_numpy() == s.legal_balls_post.iloc[:-1].to_numpy()).all()
        assert (s.score_post - s.score_pre == s.runs_total).all()
        # time only runs forward
        assert s.balls_remaining_pre.is_monotonic_decreasing
        assert (s.balls_remaining_pre >= 0).all() and (s.balls_remaining_pre <= s.max_balls).all()
        assert (s.runs_needed_post.dropna() >= 0).all()
        assert (s.wickets_in_hand_post <= 10).all()
    assert set(states.phase) <= {"powerplay", "middle", "death", "super_over"}
    # the rate is undefined, not infinite, once no balls remain
    assert states.loc[states.balls_remaining_post == 0, "rrr_post"].isna().all()
    assert np.isfinite(states["crr_post"].dropna()).all() and np.isfinite(states["rrr_post"].dropna()).all()


@given(st.lists(match_docs(), min_size=1, max_size=3))
@SLOW
def test_the_pipeline_agrees_with_the_margin_the_source_records(docs):
    """Cricsheet records 'won by N runs / N wickets'. For any internally consistent match the parsed
    scores must reproduce it, so the quality report's two margin checks must stay silent. (This is the
    check that exposed unhandled penalty runs and double wickets in the real data.)"""
    matches, innings, balls = tables(docs)
    checks = quality_report(matches, innings, balls)["checks"]
    assert checks["win_by_runs_mismatch"]["count"] == 0
    assert checks["win_by_wickets_mismatch"]["count"] == 0
    assert checks["more_than_ten_wickets"]["count"] == 0
    # and the chase never needs more than the target says
    states = build_state_table(matches, innings, balls)
    for mid, doc in enumerate(docs, start=1):
        s = states[(states.match_id == mid) & (states.innings_no == 2)]
        if s.empty:
            continue
        target = doc["innings"][1]["target"]["runs"]
        assert (s.runs_needed_post == (target - s.score_post).clip(lower=0)).all()


@given(st.lists(match_docs(), min_size=1, max_size=3), st.randoms(use_true_random=False))
@SLOW
def test_state_table_ignores_input_row_order(docs, rnd):
    matches, innings, balls = tables(docs)
    expected = build_state_table(matches, innings, balls)
    order = list(range(len(balls)))
    rnd.shuffle(order)
    got = build_state_table(matches, innings, balls.iloc[order].reset_index(drop=True))
    pd.testing.assert_frame_equal(expected, got)


@given(match_docs())
@SLOW
def test_label_is_the_same_fact_seen_from_both_sides(doc):
    """If team A won, A's rows say 1 and B's say 0. Matches without a winner have no label."""
    matches, innings, balls = tables([doc])
    states = build_state_table(matches, innings, balls)
    result = matches.iloc[0]
    if result.result != "win":
        assert states.batting_team_won.isna().all()
        return
    won = states.groupby("batting_team")["batting_team_won"].agg(lambda s: set(s.dropna().astype(int)))
    assert won.get(result.winner, {1}) == {1}
    loser = "Team A" if result.winner == "Team B" else "Team B"
    assert won.get(loser, {0}) == {0}
    assert states.batting_team_won.notna().all()


# ----------------------------------------------------------------------------------- numerics


finite = st.floats(-8, 8, allow_nan=False, allow_infinity=False)


@given(st.lists(st.floats(-1e4, 1e4, allow_nan=False), min_size=1, max_size=50))
@example([-745.0, 745.0, 0.0, -1e4, 1e4])
@FAST
def test_sigmoid_is_a_probability_and_odd_around_half(xs):
    x = np.array(xs, dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow to 0 is harmless
        p = _sigmoid(x)
        q = _sigmoid(-x)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert np.allclose(p + q, 1.0, atol=1e-12)
    order = np.argsort(x)
    assert (np.diff(p[order]) >= -1e-15).all()


@st.composite
def logit_problem(draw):
    n = draw(st.integers(40, 90))
    k = draw(st.integers(1, 3))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k)) * draw(st.floats(0.5, 3.0))
    w = rng.normal(size=k)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w)))).astype(float)
    y[0], y[1] = 0.0, 1.0  # both classes present
    return X, y


@given(logit_problem(), st.floats(0.05, 20.0), st.floats(-50, 50))
@SLOW
def test_logit_does_not_care_how_features_are_scaled_or_shifted(prob, a, b):
    """Features are standardised internally, so rescaling or shifting them (or flipping a sign)
    must leave the fitted probabilities unchanged."""
    X, y = prob
    base = Logit().fit(X, y).proba(X)
    for factor in (a, -a):
        Z = X * factor + b
        assert np.allclose(Logit().fit(Z, y).proba(Z), base, atol=1e-6)


@given(logit_problem())
@SLOW
def test_logit_swapping_the_labels_swaps_the_probabilities(prob):
    X, y = prob
    p = Logit().fit(X, y).proba(X)
    q = Logit().fit(X, 1 - y).proba(X)
    assert np.allclose(p + q, 1.0, atol=1e-6)


@given(logit_problem(), st.randoms(use_true_random=False))
@SLOW
def test_logit_ignores_row_order(prob, rnd):
    X, y = prob
    order = list(range(len(y)))
    rnd.shuffle(order)
    a = Logit().fit(X, y).proba(X)
    b = Logit().fit(X[order], y[order]).proba(X)
    assert np.allclose(a, b, atol=1e-6)


@given(logit_problem())
@SLOW
def test_logit_probabilities_are_strictly_inside_zero_one_and_calibrated_in_total(prob):
    """With an unpenalised intercept the mean prediction on the training data equals the base rate."""
    X, y = prob
    p = Logit().fit(X, y).proba(X)
    assert ((p > 0) & (p < 1)).all()
    assert p.mean() == pytest.approx(y.mean(), abs=1e-6)


# ------------------------------------------------------------------------- bootstrap and scoring


@st.composite
def scored_frames(draw):
    n_matches = draw(st.integers(2, 12))
    sizes = draw(st.lists(st.integers(1, 8), min_size=n_matches, max_size=n_matches))
    ids = draw(st.lists(st.integers(1, 10**6), min_size=n_matches, max_size=n_matches, unique=True))
    rows = []
    for mid, size in zip(ids, sizes):
        for _ in range(size):
            rows.append((mid, draw(st.integers(0, 1)), draw(st.floats(0.0, 1.0)), draw(st.floats(0.0, 1.0))))
    return pd.DataFrame(rows, columns=["match_id", "y", "pa", "pb"])


@given(scored_frames(), st.integers(0, 10**6))
@SLOW
def test_bootstrap_is_reproducible_and_a_function_of_matches_only(f, seed):
    f = f.assign(v=f["pa"])
    a = match_bootstrap(f, ["v"], 50, seed)
    b = match_bootstrap(f.sample(frac=1, random_state=1), ["v"], 50, seed)
    # same matches in any row order give the same draws only if matches are visited in id order
    assert np.allclose(a["v"], b["v"]) and np.array_equal(a["n"], b["n"])
    relabelled = f.assign(match_id=f["match_id"] * 7 + 3)  # a monotone relabelling keeps the order
    c = match_bootstrap(relabelled, ["v"], 50, seed)
    assert np.allclose(a["v"], c["v"])


@given(scored_frames(), st.integers(0, 10**6))
@SLOW
def test_bootstrap_resamples_whole_matches(f, seed):
    """Every bootstrap row count is a sum of whole-match sizes, and a bounded per-row quantity
    stays within its bounds once divided by the row count."""
    f = f.assign(v=f["pa"])
    boot = match_bootstrap(f, ["v"], 40, seed)
    sizes = set(f.groupby("match_id").size())
    assert (boot["n"] >= min(sizes) * f["match_id"].nunique()).all()
    mean = boot["v"] / boot["n"]
    assert ((mean >= -1e-12) & (mean <= 1 + 1e-12)).all()


@given(scored_frames(), st.integers(0, 10**6))
@SLOW
def test_paired_gain_is_antisymmetric(f, seed):
    ab = paired_brier_gain(f, f["pa"], f["pb"], 60, seed)
    ba = paired_brier_gain(f, f["pb"], f["pa"], 60, seed)
    assert ab["mean"] == pytest.approx(-ba["mean"], abs=1e-12)
    assert ab["ci"][0] == pytest.approx(-ba["ci"][1], abs=1e-9)
    assert ab["ci"][1] == pytest.approx(-ba["ci"][0], abs=1e-9)
    same = paired_brier_gain(f, f["pa"], f["pa"], 60, seed)
    assert same["mean"] == 0 and same["ci"] == [0.0, 0.0]


@given(st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=1, max_size=200), st.integers(2, 20))
@example([(1.0, 1), (0.0, 0)], 10)
@FAST
def test_calibration_bins_partition_the_data(pairs, bins):
    p = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    rows, ece = calibration_table(y, p, bins)
    assert sum(r["n"] for r in rows) == len(p)
    assert 0.0 <= ece <= 1.0 + 1e-12
    for r in rows:
        assert r["lo"] - 1e-12 <= r["mean_pred"] <= r["hi"] + 1e-12
