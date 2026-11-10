import copy

import pandas as pd
import pytest

from thinedge.parse import overs_to_balls, parse_match


@pytest.mark.parametrize(
    "overs, bpo, expected",
    [(20, 6, 120), (9.2, 6, 56), (0.1, 6, 1), (19.5, 6, 119), (20, 5, 100), (3.4, 5, 19)],
)
def test_overs_to_balls(overs, bpo, expected):
    assert overs_to_balls(overs, bpo) == expected


def test_overs_to_balls_rejects_impossible_notation():
    with pytest.raises(ValueError):
        overs_to_balls(5.6, 5)  # a 5-ball over has no sixth ball


def test_match_row(match_json):
    match, innings, balls = parse_match(42, match_json, "unit")
    assert match["match_id"] == 42 and match["source"] == "unit"
    assert match["result"] == "win" and match["winner"] == "Team B"
    assert match["win_by_wickets"] == 10 and match["win_by_runs"] is None
    assert match["event"] == "Test League" and match["season"] == "2026"
    assert (match["team_1"], match["team_2"]) == ("Team A", "Team B")
    assert len(innings) == 2 and len(balls) == 12


def test_innings_rows(match_json):
    _, innings, _ = parse_match(1, match_json, "unit")
    first, second = innings
    assert (first["total_runs"], first["wickets_lost"], first["legal_balls"]) == (16, 1, 6)
    assert first["max_balls"] == 12 and first["target_runs"] is None
    assert first["bowling_team"] == "Team B"
    assert second["target_runs"] == 17 and second["max_balls"] == 12
    assert second["total_runs"] == 17


def test_legality_and_extras(match_json):
    _, _, balls = parse_match(1, match_json, "unit")
    df = pd.DataFrame(balls)
    inn1 = df[df.innings_no == 1].reset_index(drop=True)
    assert inn1.is_legal.tolist() == [True, False, True, True, True, True, False, True]
    assert inn1.wides.sum() == 1 and inn1.noballs.sum() == 1 and inn1.legbyes.sum() == 1
    assert inn1.delivery_seq.tolist() == list(range(1, 9))


def test_retired_hurt_is_not_a_wicket(match_json):
    _, _, balls = parse_match(1, match_json, "unit")
    df = pd.DataFrame(balls)
    inn1 = df[df.innings_no == 1]
    assert inn1.is_wicket.sum() == 1
    assert inn1.wicket_kind.dropna().tolist() == ["bowled", "retired hurt"]
    assert inn1.is_bowler_wicket.sum() == 1


def test_players_keyed_by_registry_id(match_json):
    _, _, balls = parse_match(1, match_json, "unit")
    first = balls[0]
    assert first["batter_id"] == "id-a1" and first["bowler_id"] == "id-b1"
    assert first["non_striker_id"] == "id-a2"


def test_powerplay_by_over(match_json):
    _, _, balls = parse_match(1, match_json, "unit")
    df = pd.DataFrame(balls)
    inn1 = df[df.innings_no == 1]
    assert inn1[inn1.over_idx == 0].in_powerplay.all()
    assert not inn1[inn1.over_idx == 1].in_powerplay.any()


def test_boundaries_respect_non_boundary(match_json):
    match_json["innings"][1]["overs"][0]["deliveries"][2]["runs"]["non_boundary"] = True
    _, _, balls = parse_match(1, match_json, "unit")
    inn2 = [b for b in balls if b["innings_no"] == 2]
    assert [b["is_six"] for b in inn2] == [True, True, False, False]
    assert [b["is_four"] for b in inn2] == [False, False, False, False]  # the 4 was run, not hit


@pytest.mark.parametrize(
    "outcome, result, winner, decided_by",
    [
        ({"result": "tie", "eliminator": "Team A"}, "tie", "Team A", "eliminator"),
        ({"result": "tie", "bowl_out": "Team B"}, "tie", "Team B", "bowl_out"),
        ({"result": "no result"}, "no_result", None, None),
        ({"winner": "Team A", "by": {"runs": 5}, "method": "D/L"}, "win", "Team A", None),
    ],
)
def test_outcomes(match_json, outcome, result, winner, decided_by):
    m = copy.deepcopy(match_json)
    m["info"]["outcome"] = outcome
    match, _, _ = parse_match(1, m, "unit")
    assert (match["result"], match["winner"], match["decided_by"]) == (result, winner, decided_by)


def test_dls_method_recorded(match_json):
    match_json["info"]["outcome"] = {"winner": "Team A", "by": {"runs": 5}, "method": "D/L"}
    match, _, _ = parse_match(1, match_json, "unit")
    assert match["method"] == "D/L"


def test_super_over_flagged_and_sized(match_json):
    match_json["innings"].append(
        {"team": "Team A", "super_over": True,
         "overs": [{"over": 0, "deliveries": [{"batter": "A1", "bowler": "B1", "non_striker": "A2",
                                               "runs": {"batter": 1, "extras": 0, "total": 1}}]}]}
    )
    _, innings, balls = parse_match(1, match_json, "unit")
    assert innings[2]["is_super_over"] and innings[2]["max_balls"] == 6
    assert sum(b["is_super_over"] for b in balls) == 1


def test_impact_player_substitution_counted(match_json):
    d = match_json["innings"][1]["overs"][0]["deliveries"][0]
    d["replacements"] = {"match": [{"in": "X", "out": "Y", "reason": "impact_player", "team": "Team B"}]}
    match, _, balls = parse_match(1, match_json, "unit")
    assert match["impact_subs"] == 1
    assert sum(b["impact_sub"] for b in balls) == 1


def test_non_limited_overs_match_rejected(match_json):
    del match_json["info"]["overs"]
    with pytest.raises(ValueError):
        parse_match(1, match_json, "unit")
