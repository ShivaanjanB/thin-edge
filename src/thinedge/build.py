"""Parse raw Cricsheet JSON into Parquet tables under data/processed/.

The build is deterministic and idempotent: the same raw files always produce byte-for-byte
comparable tables (sorted by match, innings, delivery), written atomically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR
from .features import build_state_table
from .parse import NotLimitedOvers, parse_file
from .validate import quality_report


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def discover(
    raw_dir: Path, sources: list[str] | None
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Match files (named ``<match id>.json``) per source, and any other ``.json`` files found, which
    are ignored rather than allowed to abort the build."""
    names = list(dict.fromkeys(sources or sorted(p.name for p in raw_dir.iterdir() if p.is_dir())))
    found: list[tuple[str, Path]] = []
    ignored: list[str] = []
    for name in names:
        folder = raw_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f"no raw data for source {name!r} in {raw_dir}")
        files = sorted(folder.glob("*.json"))
        found += [(name, f) for f in sorted((f for f in files if f.stem.isdecimal()), key=lambda p: int(p.stem))]
        ignored += [f"{name}/{f.name}" for f in files if not f.stem.isdecimal()]
    return found, ignored


def build(
    sources: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    """Parse, validate and write all tables. Returns the quality report plus build stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    matches, innings, balls = [], [], []
    seen: set[int] = set()
    skipped_not_limited_overs = skipped_duplicates = 0
    malformed: list[dict] = []
    files, ignored = discover(raw_dir, sources)

    for source, path in files:
        match_id = int(path.stem)
        if match_id in seen:  # the same match can appear in several Cricsheet archives
            skipped_duplicates += 1
            continue
        try:
            m, i, b = parse_file(path, source)
        except NotLimitedOvers:
            skipped_not_limited_overs += 1
            continue
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            # A truncated download or an unexpected field. Name the file; do not drop it silently
            # (JSONDecodeError is a ValueError) and do not let one bad file abort the whole build.
            malformed.append({"file": f"{source}/{path.name}", "error": f"{type(exc).__name__}: {exc}"})
            continue
        seen.add(match_id)
        matches.append(m)
        innings += i
        balls += b

    if not matches:
        raise ValueError(f"no matches parsed from {raw_dir}; run `thinedge download` first")

    matches_df = pd.DataFrame(matches).sort_values("match_id").reset_index(drop=True)
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    for col in ("win_by_runs", "win_by_wickets"):
        matches_df[col] = matches_df[col].astype("Int64")
    innings_df = pd.DataFrame(innings).sort_values(["match_id", "innings_no"]).reset_index(drop=True)
    innings_df["target_runs"] = innings_df["target_runs"].astype("Int64")
    balls_df = (
        pd.DataFrame(balls).sort_values(["match_id", "innings_no", "delivery_seq"]).reset_index(drop=True)
    )

    states_df = build_state_table(matches_df, innings_df, balls_df)
    venues_df = (
        matches_df.groupby(["venue", "city"], dropna=False)
        .agg(n_matches=("match_id", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
        .sort_values(["venue", "city"])
        .reset_index(drop=True)
    )
    venues_df["latitude"] = pd.Series(pd.NA, index=venues_df.index, dtype="Float64")
    venues_df["longitude"] = pd.Series(pd.NA, index=venues_df.index, dtype="Float64")

    report = quality_report(matches_df, innings_df, balls_df)
    report["skipped_not_limited_overs"] = skipped_not_limited_overs
    report["skipped_duplicate_match_ids"] = skipped_duplicates
    report["malformed_files"] = malformed
    report["ignored_files"] = ignored
    if malformed:
        report["errors"]["malformed_files"] = len(malformed)
        report["ok"] = False
    report["n_states"] = int(len(states_df))

    for name, frame in {
        "matches": matches_df,
        "innings": innings_df,
        "balls": balls_df,
        "states": states_df,
        "venues": venues_df,
    }.items():
        _write_parquet(frame, out_dir / f"{name}.parquet")
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
