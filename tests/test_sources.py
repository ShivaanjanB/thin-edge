"""End-to-end checks that several competitions flow through build, filtering and reports."""

import copy
import json

import pytest

from thinedge.baseline import load_frame, source_clause
from thinedge.build import build
from thinedge.config import reports_dir_for
from thinedge.db import connect


@pytest.fixture
def tables(tmp_path, match_json):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    for source, first_id in (("alpha", 100), ("beta", 200)):
        (raw / source).mkdir(parents=True)
        for i in range(3):
            doc = copy.deepcopy(match_json)
            (raw / source / f"{first_id + i}.json").write_text(json.dumps(doc), encoding="utf-8")
    build(raw_dir=raw, out_dir=out)
    return connect(out)


def test_build_keeps_every_competition_and_labels_its_source(tables):
    rows = tables.sql("select source, count(*) from matches group by 1 order by 1").fetchall()
    assert rows == [("alpha", 3), ("beta", 3)]


def test_load_frame_filters_by_competition(tables):
    everything = load_frame(tables)
    only_alpha = load_frame(tables, ["alpha"])
    assert set(everything["source"]) == {"alpha", "beta"}
    assert set(only_alpha["source"]) == {"alpha"}
    assert len(only_alpha) * 2 == len(everything)


def test_load_frame_rejects_a_name_that_could_alter_the_query(tables):
    for bad in ("x'; drop table matches; --", "A", "a b", ""):
        with pytest.raises(ValueError):
            load_frame(tables, [bad])
    assert source_clause(None) == "" and source_clause([]) == ""


def test_a_decided_chase_with_no_target_is_excluded_from_modelling_and_reported(tmp_path, match_json):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    (raw / "alpha").mkdir(parents=True)
    for i in range(3):
        doc = copy.deepcopy(match_json)
        if i == 1:
            del doc["innings"][1]["target"]  # Cricsheet omits the target in some matches
        (raw / "alpha" / f"{300 + i}.json").write_text(json.dumps(doc), encoding="utf-8")
    report = build(raw_dir=raw, out_dir=out)

    check = report["checks"]["decided_chase_without_target"]
    assert check["count"] == 1 and check["sample_match_ids"] == [301] and check["severity"] == "warn"
    assert report["ok"]  # a warning, not an error

    frame = load_frame(connect(out))
    assert set(frame["match_id"]) == {300, 302}  # excluded in both innings, not just the chase
    assert frame.groupby("match_id")["innings_no"].nunique().eq(2).all()
    assert frame["target_runs"].notna().sum() == frame[frame["innings_no"] == 2].shape[0]


def test_reports_land_in_a_folder_named_for_the_league_set():
    assert reports_dir_for(None).name == "all"
    assert reports_dir_for(["ipl", "hnd"]).name == "hnd+ipl"
    assert reports_dir_for(["hnd", "ipl"]) == reports_dir_for(["ipl", "hnd"])
