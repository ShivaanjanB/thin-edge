"""Regression tests for defects found by exploratory testing (see DECISIONS.md, 2026-11-10).

Each test failed on the code as it stood when the defect was found. The first four were found by
checking parsed tables against the margins Cricsheet itself records (won by N wickets / N runs);
the rest by running the command line and the build with bad input.
"""

import copy
import json

import pandas as pd
import pytest
from conftest import delivery

from thinedge.baseline import load_frame, source_clause
from thinedge.build import build
from thinedge.cli import main
from thinedge.config import reports_dir_for
from thinedge.db import connect
from thinedge.features import build_state_table
from thinedge.parse import parse_match
from thinedge.validate import quality_report


def tables(doc):
    m, i, b = parse_match(1, doc, "t")
    matches, innings, balls = pd.DataFrame([m]), pd.DataFrame(i), pd.DataFrame(b)
    matches["date"] = pd.to_datetime(matches["date"])
    innings["target_runs"] = innings["target_runs"].astype("Int64")
    return matches, innings, balls


# ------------------------------------------------------------------- wickets on a delivery


def test_two_dismissals_on_one_delivery_are_both_counted(match_json):
    """Match 1208093 (BPL 2019): a run out and a 'retired out' on the same ball."""
    ball = match_json["innings"][0]["overs"][0]["deliveries"][0]
    ball["wickets"] = [
        {"player_out": "A1", "kind": "run out"},
        {"player_out": "A2", "kind": "retired out"},
    ]
    matches, innings, balls = tables(match_json)
    first = innings[innings.innings_no == 1].iloc[0]
    assert first["wickets_lost"] == 1 + 2  # the existing bowled ball, plus both on this delivery
    states = build_state_table(matches, innings, balls)
    s = states[states.innings_no == 1].sort_values("delivery_seq")
    assert s.iloc[0]["wickets_post"] == 2 and s.iloc[-1]["wickets_post"] == 3
    assert s.iloc[1]["wickets_pre"] == 2  # the next ball starts with both wickets gone


def test_a_retired_hurt_listed_first_does_not_hide_a_real_dismissal(match_json):
    ball = match_json["innings"][0]["overs"][0]["deliveries"][0]
    ball["wickets"] = [
        {"player_out": "A2", "kind": "retired hurt"},
        {"player_out": "A1", "kind": "run out"},
    ]
    _, innings, balls = tables(match_json)
    row = balls[(balls.innings_no == 1) & (balls.delivery_seq == 1)].iloc[0]
    assert row["is_wicket"] and row["wicket_kind"] == "run out" and row["player_out"] == "A1"
    assert innings[innings.innings_no == 1].iloc[0]["wickets_lost"] == 2


def test_retired_not_out_is_not_a_dismissal(match_json):
    """Matches 1250237 and 1369575: the batter retired and may return, so no wicket has fallen."""
    ball = match_json["innings"][0]["overs"][0]["deliveries"][0]
    ball["wickets"] = [{"player_out": "A1", "kind": "retired not out"}]
    _, innings, balls = tables(match_json)
    assert not balls[(balls.innings_no == 1) & (balls.delivery_seq == 1)].iloc[0]["is_wicket"]
    assert innings[innings.innings_no == 1].iloc[0]["wickets_lost"] == 1  # only the bowled ball


# ---------------------------------------------------------------------------- penalty runs


def test_penalty_runs_awarded_after_an_innings_count_in_its_total(match_json):
    match_json["innings"][0]["penalty_runs"] = {"post": 5}
    _, innings, _ = tables(match_json)
    first = innings[innings.innings_no == 1].iloc[0]
    assert first["total_runs"] == 16 + 5 and first["penalty_runs_post"] == 5


def test_penalty_runs_awarded_before_an_innings_start_the_score_above_zero(match_json):
    """Match 947121 (T20 Blast): the chasing side began on 6. The target already includes them, so
    ignoring them overstated the runs needed throughout the chase."""
    match_json["innings"][1]["penalty_runs"] = {"pre": 5}
    match_json["innings"][1]["target"]["runs"] = 22  # first innings 16 + 1 + 5
    matches, innings, balls = tables(match_json)
    chase = build_state_table(matches, innings, balls).query("innings_no == 2").sort_values("delivery_seq")
    assert chase.iloc[0]["score_pre"] == 5 and chase.iloc[0]["score_post"] == 11
    assert chase.iloc[-1]["score_post"] == 5 + 17 and chase.iloc[-1]["runs_needed_post"] == 0
    assert innings[innings.innings_no == 2].iloc[0]["total_runs"] == 22


def test_the_recorded_win_margin_check_flags_a_disagreement(match_json):
    matches, innings, balls = tables(match_json)
    ok = quality_report(matches, innings, balls)["checks"]
    assert ok["win_by_runs_mismatch"]["count"] == 0 and ok["win_by_wickets_mismatch"]["count"] == 0
    matches.loc[0, "win_by_wickets"] = 3  # the pipeline says 0 wickets were lost in the chase
    bad = quality_report(matches, innings, balls)["checks"]
    assert bad["win_by_wickets_mismatch"]["count"] == 1
    assert bad["win_by_wickets_mismatch"]["severity"] == "warn"


# ----------------------------------------------------------------------- the build step


def raw_folder(tmp_path, match_json, extra: dict[str, str]):
    raw = tmp_path / "raw"
    (raw / "alpha").mkdir(parents=True)
    (raw / "alpha" / "1.json").write_text(json.dumps(match_json), encoding="utf-8")
    for name, text in extra.items():
        (raw / "alpha" / name).write_text(text, encoding="utf-8")
    return raw


@pytest.mark.parametrize("text", ['{"info": {"overs": 20', "", "\x00\x01"], ids=["truncated", "empty", "junk"])
def test_a_corrupt_file_is_reported_not_silently_dropped(tmp_path, match_json, text):
    raw = raw_folder(tmp_path, match_json, {"2.json": text})
    report = build(raw_dir=raw, out_dir=tmp_path / "out")
    assert report["n_matches"] == 1
    assert report["skipped_not_limited_overs"] == 0
    assert [f["file"] for f in report["malformed_files"]] == ["alpha/2.json"]
    assert report["ok"] is False and report["errors"]["malformed_files"] == 1


def test_a_missing_field_names_the_file_instead_of_aborting_the_build(tmp_path, match_json):
    broken = copy.deepcopy(match_json)
    del broken["innings"][0]["overs"][0]["deliveries"][0]["runs"]
    raw = raw_folder(tmp_path, match_json, {"2.json": json.dumps(broken)})
    report = build(raw_dir=raw, out_dir=tmp_path / "out")
    assert report["n_matches"] == 1
    assert "runs" in report["malformed_files"][0]["error"]


def test_invalid_over_notation_is_malformed_rather_than_a_non_limited_overs_match(tmp_path, match_json):
    broken = copy.deepcopy(match_json)
    broken["innings"][1]["target"]["overs"] = 1.9  # no such thing as the 9th ball of a 6-ball over
    raw = raw_folder(tmp_path, match_json, {"2.json": json.dumps(broken)})
    report = build(raw_dir=raw, out_dir=tmp_path / "out")
    assert report["skipped_not_limited_overs"] == 0 and len(report["malformed_files"]) == 1


def test_a_test_match_is_still_counted_as_not_limited_overs(tmp_path, match_json):
    test_match = copy.deepcopy(match_json)
    del test_match["info"]["overs"]
    raw = raw_folder(tmp_path, match_json, {"2.json": json.dumps(test_match)})
    report = build(raw_dir=raw, out_dir=tmp_path / "out")
    assert report["skipped_not_limited_overs"] == 1 and report["malformed_files"] == [] and report["ok"]


def test_a_stray_non_numeric_json_file_is_ignored_and_listed(tmp_path, match_json):
    raw = raw_folder(tmp_path, match_json, {"notes.json": "{}"})
    report = build(raw_dir=raw, out_dir=tmp_path / "out")
    assert report["n_matches"] == 1 and report["ignored_files"] == ["alpha/notes.json"]


# ------------------------------------------------------------------ sources and the command line


@pytest.fixture
def con(tmp_path, match_json):
    raw = tmp_path / "raw"
    for source, first in (("alpha", 100), ("beta", 200)):
        (raw / source).mkdir(parents=True)
        for i in range(2):
            (raw / source / f"{first + i}.json").write_text(json.dumps(match_json), encoding="utf-8")
    build(raw_dir=raw, out_dir=tmp_path / "out")
    return connect(tmp_path / "out")


def test_an_unknown_source_is_named_instead_of_blamed_on_the_split_date(con):
    with pytest.raises(ValueError, match=r"no data for.*'nope'.*alpha.*beta"):
        load_frame(con, ["alpha", "nope"])


def test_repeating_a_source_changes_nothing(con):
    assert source_clause(["alpha", "alpha"]) == source_clause(["alpha"])
    assert len(load_frame(con, ["alpha", "alpha"])) == len(load_frame(con, ["alpha"]))
    assert reports_dir_for(["ipl", "ipl"]) == reports_dir_for(["ipl"])


def test_the_command_line_explains_a_user_error_without_a_traceback(capsys, monkeypatch):
    monkeypatch.delenv("THINEDGE_DEBUG", raising=False)
    code = main(["download", "../evil"])
    err = capsys.readouterr().err
    assert code == 2 and err.startswith("thinedge: error:") and "invalid source name" in err
    assert "Traceback" not in err


def test_the_command_line_can_still_show_the_traceback_when_asked(monkeypatch):
    monkeypatch.setenv("THINEDGE_DEBUG", "1")
    with pytest.raises(ValueError):
        main(["download", "../evil"])


@pytest.mark.parametrize("flag", ["--draws", "--tune", "--chains", "--cores", "--min-balls"])
def test_bayes_options_must_be_positive_whole_numbers(flag, capsys):
    for bad in ("0", "-2", "1.5"):
        with pytest.raises(SystemExit) as exc:
            main(["bayes", flag, bad])
        assert exc.value.code == 2
    assert "positive whole number" in capsys.readouterr().err


def test_walkforward_with_no_earlier_data_says_so():
    from thinedge.walkforward import analyse

    df = pd.DataFrame({"date": pd.to_datetime(["2024-03-01", "2024-03-02"]), "match_id": [1, 1]})
    with pytest.raises(ValueError, match="no matches before 2000-01-01"):
        analyse(df, {}, first_test_year=2000)
    with pytest.raises(ValueError, match="no matches from 2030-01-01"):
        analyse(df, {}, first_test_year=2030)


# --------------------------------------------------------------------------------- downloads


def test_an_archive_whose_members_add_up_to_too_much_is_refused(tmp_path, monkeypatch):
    import zipfile

    from thinedge import download

    monkeypatch.setattr(download, "MAX_TOTAL_BYTES", 1000)
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(5):
            zf.writestr(f"{i}.json", "0" * 400)  # each member is small and compresses to almost nothing
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ValueError, match="extract to"):
        download.extract_json_members(z, dest)
    assert list(dest.iterdir()) == []  # refused before anything was written


def test_member_names_that_differ_only_in_case_are_refused_rather_than_overwritten(tmp_path):
    import zipfile

    from thinedge.download import extract_json_members

    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("A.json", "first")
        zf.writestr("a.json", "second")
    with pytest.raises(ValueError, match="letter case"):
        extract_json_members(z, tmp_path)
