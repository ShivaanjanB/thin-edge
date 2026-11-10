"""A tiny hand-built Cricsheet-style match so every expected number can be checked by hand.

Team A bats first (2-over innings, 6-ball overs), Team B chases 17 and wins.

Innings 1 deliveries (runs_total, legal?, wicket)             score  wkts  legal
 over 0: 1 run                                                    1     0      1
         wide (1 extra)                                           2     0      1
         4 runs                                                   6     0      2
         bowled                                                   6     1      3
 over 1: 6 runs                                                  12     1      4
         retired hurt (not a dismissal)                          12     1      5
         no-ball, 2 off the bat + 1 extra                        15     1      5
         leg bye 1                                               16     1      6
Innings 2 (target 17): 6, 6, 4, 1                                17 after 4 balls
"""

from __future__ import annotations

import pytest

PLAYERS = {"A1": "id-a1", "A2": "id-a2", "B1": "id-b1", "B2": "id-b2"}


def delivery(batter_runs=0, extras=None, wicket=None, bowler="B1", batter="A1", **kw):
    extras = extras or {}
    total = batter_runs + sum(v for k, v in extras.items())
    d = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "A2",
        "runs": {"batter": batter_runs, "extras": total - batter_runs, "total": total},
    }
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"player_out": batter, "kind": wicket}]
    d.update(kw)
    return d


@pytest.fixture
def match_json():
    inn1 = [
        {"over": 0, "deliveries": [
            delivery(1),
            delivery(0, {"wides": 1}),
            delivery(4),
            delivery(0, wicket="bowled"),
        ]},
        {"over": 1, "deliveries": [
            delivery(6),
            delivery(0, wicket="retired hurt"),
            delivery(2, {"noballs": 1}),
            delivery(0, {"legbyes": 1}),
        ]},
    ]
    inn2 = [
        {"over": 0, "deliveries": [
            delivery(6, batter="B1", bowler="A1"),
            delivery(6, batter="B1", bowler="A1"),
            delivery(4, batter="B1", bowler="A1"),
            delivery(1, batter="B1", bowler="A1"),
        ]},
    ]
    return {
        "meta": {"data_version": "1.3.0"},
        "info": {
            "balls_per_over": 6,
            "overs": 2,
            "gender": "male",
            "match_type": "T20",
            "team_type": "club",
            "event": {"name": "Test League", "match_number": 1},
            "season": "2026",
            "dates": ["2026-04-01"],
            "venue": "Test Ground",
            "city": "St Albans",
            "teams": ["Team A", "Team B"],
            "toss": {"winner": "Team A", "decision": "bat"},
            "outcome": {"winner": "Team B", "by": {"wickets": 10}},
            "registry": {"people": PLAYERS},
        },
        "innings": [
            {"team": "Team A", "overs": inn1,
             "powerplays": [{"from": 0.1, "to": 0.6, "type": "mandatory"}]},
            {"team": "Team B", "overs": inn2, "target": {"overs": 2, "runs": 17}},
        ],
    }
