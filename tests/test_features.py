import numpy as np
import pandas as pd
import pytest

from thinedge.features import assign_phase, build_state_table
from thinedge.parse import parse_match
from thinedge.validate import quality_report


def frames(match_json, match_id=1):
    m, i, b = parse_match(match_id, match_json, "unit")
    return pd.DataFrame([m]), pd.DataFrame(i), pd.DataFrame(b)


@pytest.fixture
def states(match_json):
    matches, innings, balls = frames(match_json)
    return build_state_table(matches, innings, balls)


def test_one_row_per_delivery(states):
    assert len(states) == 12
    assert states.groupby("innings_no").size().to_dict() == {1: 8, 2: 4}


def test_running_totals_first_innings(states):
    inn1 = states[states.innings_no == 1].reset_index(drop=True)
    assert inn1.score_post.tolist() == [1, 2, 6, 6, 12, 12, 15, 16]
    assert inn1.wickets_post.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert inn1.legal_balls_post.tolist() == [1, 1, 2, 3, 4, 5, 5, 6]


def test_pre_state_is_post_minus_this_ball(states):
    inn1 = states[states.innings_no == 1].reset_index(drop=True)
    assert inn1.score_pre.tolist() == [0, 1, 2, 6, 6, 12, 12, 15]
    assert inn1.wickets_pre.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert inn1.legal_balls_pre.tolist() == [0, 1, 1, 2, 3, 4, 5, 5]
    # no leakage: a ball's pre-state never contains that ball's own runs
    assert (inn1.score_pre + inn1.runs_total == inn1.score_post).all()


def test_balls_remaining(states):
    inn1 = states[states.innings_no == 1].reset_index(drop=True)
    assert inn1.balls_remaining_post.iloc[0] == 11
    assert inn1.balls_remaining_post.iloc[-1] == 6


def test_run_rates_first_innings(states):
    inn1 = states[states.innings_no == 1].reset_index(drop=True)
    assert inn1.crr_post.iloc[0] == pytest.approx(6.0)  # 1 run off 1 ball, per 6-ball over
    assert inn1.crr_post.iloc[-1] == pytest.approx(16 / 6 * 6)
    assert inn1.rrr_post.isna().all() and not inn1.is_chase.any()


def test_chase_features(states):
    inn2 = states[states.innings_no == 2].reset_index(drop=True)
    assert inn2.target_runs.eq(17).all() and inn2.is_chase.all()
    assert inn2.runs_needed_post.tolist() == [11, 5, 1, 0]  # clipped at 0 once target reached
    assert inn2.balls_remaining_post.tolist() == [11, 10, 9, 8]
    assert inn2.rrr_post.iloc[0] == pytest.approx(11 / 11 * 6)
    assert inn2.crr_post.iloc[0] == pytest.approx(36.0)


def test_phase(states):
    inn1 = states[states.innings_no == 1].reset_index(drop=True)
    assert inn1.phase.iloc[:4].eq("powerplay").all()
    assert inn1.phase.iloc[4:].eq("middle").all()


def test_label_follows_winner(states):
    assert states[states.innings_no == 1].batting_team_won.eq(0).all()
    assert states[states.innings_no == 2].batting_team_won.eq(1).all()


@pytest.mark.parametrize("outcome", [{"result": "tie"}, {"result": "no result"}])
def test_label_null_without_regulation_winner(match_json, outcome):
    match_json["info"]["outcome"] = outcome
    states = build_state_table(*frames(match_json))
    assert states.batting_team_won.isna().all()


def test_super_over_rows_unlabelled(match_json):
    match_json["info"]["outcome"] = {"result": "tie", "eliminator": "Team A"}
    match_json["innings"].append(
        {"team": "Team A", "super_over": True,
         "overs": [{"over": 0, "deliveries": [{"batter": "A1", "bowler": "B1", "non_striker": "A2",
                                               "runs": {"batter": 1, "extras": 0, "total": 1}}]}]}
    )
    states = build_state_table(*frames(match_json))
    so = states[states.is_super_over]
    assert len(so) == 1 and so.batting_team_won.isna().all() and so.phase.iloc[0] == "super_over"


def test_is_reduced_false_for_full_length_match(states):
    assert not states.is_reduced.any()


def test_is_reduced_when_chase_is_shortened_even_without_dls_method(match_json):
    # A rain-affected no-result match carries no D/L method but has a shortened chase.
    match_json["info"]["outcome"] = {"result": "no result"}
    match_json["innings"][1]["target"] = {"overs": 1, "runs": 9}
    states = build_state_table(*frames(match_json))
    assert states.is_reduced.all()


def test_is_reduced_when_dls_method_present(match_json):
    match_json["info"]["outcome"] = {"winner": "Team B", "by": {"wickets": 3}, "method": "D/L"}
    states = build_state_table(*frames(match_json))
    assert states.is_reduced.all()


def test_reduced_matches_excluded_from_target_check(match_json):
    match_json["innings"][1]["target"] = {"overs": 1, "runs": 9}  # not first innings + 1, but rain
    report = quality_report(*frames(match_json))
    assert report["checks"]["target_not_first_innings_plus_one"]["count"] == 0


def test_assign_phase_boundaries():
    # 120-ball innings: death is the last 24 balls (<= 24 remaining before the ball)
    out = assign_phase(
        in_powerplay=np.array([True, False, False, False, False]),
        balls_remaining_pre=np.array([114, 100, 25, 24, 1]),
        max_balls=np.array([120] * 5),
        is_super_over=np.array([False] * 5),
    )
    assert out.tolist() == ["powerplay", "middle", "middle", "death", "death"]


def test_quality_report_clean_on_good_match(match_json):
    matches, innings, balls = frames(match_json)
    report = quality_report(matches, innings, balls)
    assert report["ok"], report["errors"]
    assert report["checks"]["target_not_first_innings_plus_one"]["count"] == 0


def test_quality_report_catches_corruption(match_json):
    matches, innings, balls = frames(match_json)
    balls.loc[0, "runs_total"] += 3
    balls = pd.concat([balls, balls.iloc[[1]]], ignore_index=True)
    matches.loc[0, "winner"] = "Team Z"
    report = quality_report(matches, innings, balls)
    assert not report["ok"]
    assert {"runs_total_mismatch", "duplicate_deliveries", "winner_not_in_teams"} <= set(report["errors"])
