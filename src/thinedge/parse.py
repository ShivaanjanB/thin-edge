"""Turn one Cricsheet match JSON into flat rows: one match row, innings rows, ball rows.

Conventions
- ``over_idx`` is Cricsheet's 0-based over number.
- ``delivery_seq`` counts every delivery in the innings (wides and no-balls included), from 1.
- A delivery is *legal* unless it is a wide or a no-ball. Byes and leg byes are legal.
- ``is_wicket`` excludes "retired hurt", which is recorded as a wicket but is not a dismissal.
- Players are keyed by Cricsheet registry id, because names collide ("Mohammed Shami" etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import BOWLER_WICKET_KINDS, NOT_A_DISMISSAL


def overs_to_balls(overs: float | int, balls_per_over: int) -> int:
    """Convert cricket over notation to balls: 9.2 overs of 6 is 56 balls (not 9.2 * 6)."""
    whole = int(overs)
    partial = round((overs - whole) * 10)
    if not 0 <= partial < balls_per_over:
        raise ValueError(f"{overs!r} is not valid over notation for {balls_per_over}-ball overs")
    return whole * balls_per_over + partial


def _event_name(info: dict) -> str | None:
    event = info.get("event")
    if isinstance(event, dict):
        return event.get("name")
    return event


def _outcome(info: dict) -> dict:
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    if "winner" in outcome:
        result, winner, decided_by = "win", outcome["winner"], None
    elif outcome.get("result") == "tie":
        # A tied match may be settled by a super over ("eliminator") or bowl-out.
        if "eliminator" in outcome:
            winner, decided_by = outcome["eliminator"], "eliminator"
        elif "bowl_out" in outcome:
            winner, decided_by = outcome["bowl_out"], "bowl_out"
        else:
            winner, decided_by = None, None
        result = "tie"
    else:
        result = str(outcome.get("result", "unknown")).replace(" ", "_")
        winner, decided_by = None, None
    return {
        "result": result,
        "winner": winner,
        "decided_by": decided_by,
        "method": outcome.get("method"),
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
    }


def parse_match(match_id: int, data: dict, source: str) -> tuple[dict, list[dict], list[dict]]:
    """Parse a decoded Cricsheet JSON document.

    Returns ``(match_row, innings_rows, ball_rows)``. Raises ``ValueError`` for matches that are
    not limited-overs (no ``overs`` in ``info``), because the pipeline models fixed-length formats.
    """
    info = data["info"]
    nominal_overs = info.get("overs")
    if nominal_overs is None:
        raise ValueError(f"match {match_id} has no 'overs' (not a limited-overs match)")
    bpo = int(info["balls_per_over"])
    teams = list(info.get("teams", []))
    people = info.get("registry", {}).get("people", {})
    toss = info.get("toss", {})

    innings_rows: list[dict] = []
    ball_rows: list[dict] = []
    impact_subs = 0

    for inn_no, inn in enumerate(data.get("innings", []), start=1):
        team = inn["team"]
        opponents = [t for t in teams if t != team]
        is_super = bool(inn.get("super_over", False))
        target = inn.get("target")
        if is_super:
            max_balls = bpo
        elif target is not None:
            max_balls = overs_to_balls(target["overs"], bpo)
        else:
            max_balls = int(nominal_overs) * bpo
        powerplays = [(int(p["from"]), int(p["to"])) for p in inn.get("powerplays", [])]

        seq = runs = wickets = legal = 0
        for over in inn.get("overs", []):
            over_idx = over["over"]
            in_powerplay = any(lo <= over_idx <= hi for lo, hi in powerplays)
            for d in over["deliveries"]:
                seq += 1
                r = d["runs"]
                extras = d.get("extras", {})
                is_legal = "wides" not in extras and "noballs" not in extras
                wicket = (d.get("wickets") or [None])[0]
                kind = wicket["kind"] if wicket else None
                is_wicket = kind is not None and kind not in NOT_A_DISMISSAL
                non_boundary = bool(r.get("non_boundary", False))
                match_replacements = d.get("replacements", {}).get("match", [])
                impact = any(x.get("reason") == "impact_player" for x in match_replacements)
                impact_subs += sum(x.get("reason") == "impact_player" for x in match_replacements)

                runs += r["total"]
                wickets += is_wicket
                legal += is_legal
                ball_rows.append(
                    {
                        "match_id": match_id,
                        "innings_no": inn_no,
                        "is_super_over": is_super,
                        "batting_team": team,
                        "over_idx": over_idx,
                        "delivery_seq": seq,
                        "batter": d["batter"],
                        "batter_id": people.get(d["batter"]),
                        "bowler": d["bowler"],
                        "bowler_id": people.get(d["bowler"]),
                        "non_striker": d.get("non_striker"),
                        "non_striker_id": people.get(d.get("non_striker")),
                        "runs_batter": r["batter"],
                        "runs_extras": r["extras"],
                        "runs_total": r["total"],
                        "wides": extras.get("wides", 0),
                        "noballs": extras.get("noballs", 0),
                        "byes": extras.get("byes", 0),
                        "legbyes": extras.get("legbyes", 0),
                        "penalty": extras.get("penalty", 0),
                        "is_legal": is_legal,
                        "is_four": r["batter"] == 4 and not non_boundary,
                        "is_six": r["batter"] == 6 and not non_boundary,
                        "wicket_kind": kind,
                        "is_wicket": is_wicket,
                        "is_bowler_wicket": kind in BOWLER_WICKET_KINDS,
                        "player_out": wicket["player_out"] if wicket else None,
                        "player_out_id": people.get(wicket["player_out"]) if wicket else None,
                        "in_powerplay": in_powerplay,
                        "impact_sub": impact,
                    }
                )

        innings_rows.append(
            {
                "match_id": match_id,
                "innings_no": inn_no,
                "is_super_over": is_super,
                "batting_team": team,
                "bowling_team": opponents[0] if len(opponents) == 1 else None,
                "target_runs": target["runs"] if target else None,
                "max_balls": max_balls,
                "total_runs": runs,
                "wickets_lost": wickets,
                "legal_balls": legal,
                "has_miscounted_overs": "miscounted_overs" in inn,
            }
        )

    dates = info.get("dates", [])
    match_row = {
        "match_id": match_id,
        "source": source,
        "data_version": data.get("meta", {}).get("data_version"),
        "event": _event_name(info),
        "season": str(info["season"]) if "season" in info else None,
        "date": dates[0] if dates else None,
        "gender": info.get("gender"),
        "match_type": info.get("match_type"),
        "team_type": info.get("team_type"),
        "balls_per_over": bpo,
        "overs": int(nominal_overs),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_1": teams[0] if len(teams) > 0 else None,
        "team_2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "player_of_match": (info.get("player_of_match") or [None])[0],
        "impact_subs": impact_subs,
        **_outcome(info),
    }
    return match_row, innings_rows, ball_rows


def parse_file(path: Path, source: str) -> tuple[dict, list[dict], list[dict]]:
    """Parse ``<match_id>.json``; the match id is the file stem."""
    match_id = int(path.stem)
    with path.open(encoding="utf-8") as fh:
        return parse_match(match_id, json.load(fh), source)
