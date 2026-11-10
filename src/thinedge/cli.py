"""Command line: ``thinedge download ipl hnd`` then ``thinedge build``."""

from __future__ import annotations

import argparse
import sys

from .build import build
from .config import PROCESSED_DIR, RAW_DIR
from .download import download_source


def _download(args: argparse.Namespace) -> int:
    for name in args.sources:
        entry = download_source(name, RAW_DIR, refresh=args.refresh)
        state = "cached" if entry["skipped"] else "downloaded"
        print(f"{name}: {state}, {entry['n_json']} matches, sha256 {entry['sha256'][:12]}")
    return 0


def _build(args: argparse.Namespace) -> int:
    report = build(args.sources or None)
    print(f"matches {report['n_matches']}, innings {report['n_innings']}, balls {report['n_balls']}, "
          f"states {report['n_states']}")
    print(f"skipped: {report['skipped_not_limited_overs']} not limited-overs, "
          f"{report['skipped_duplicate_match_ids']} duplicate ids")
    for name, check in report["checks"].items():
        if check["count"]:
            print(f"  [{check['severity']}] {name}: {check['count']}")
    print(f"wrote {PROCESSED_DIR}")
    if not report["ok"]:
        print(f"ERRORS: {report['errors']}", file=sys.stderr)
        return 1
    return 0


def _baseline(args: argparse.Namespace) -> int:
    try:
        from .baseline import run_baseline
        from .config import reports_dir_for
        from .plots import calibration_figure
    except ImportError as exc:
        print(f"{exc}\nInstall the model extras: pip install -e '.[model]'", file=sys.stderr)
        return 1

    out = reports_dir_for(args.sources)
    report = run_baseline(train_end=args.train_end, reports_dir=out, sources=args.sources)
    print(f"sources: {report['sources']}")
    split = report["split"]
    print(f"train: {split['train_matches']} matches before {report['train_end']}; "
          f"test: {split['test_matches']} matches {split['test_matches_by_source']}")
    print(f"{'variant':<12} {'innings':<14} {'brier':>7} {'ref':>7} {'skill':>7}  {'95% CI':<16} {'ece':>6}")
    for name, variant in report["variants"].items():
        for key in ("first_innings", "chase", "all"):
            m = variant["metrics"][key]
            lo, hi = m["skill_ci"]
            print(f"{name:<12} {key:<14} {m['brier']:7.4f} {m['ref_brier']:7.4f} {m['skill']:7.3f}  "
                  f"[{lo:6.3f},{hi:6.3f}]  {m['ece']:6.3f}")
    gain = report["regime_gain"]
    print(f"regime flag, Brier gain per ball: {gain['mean']:+.5f}  95% CI [{gain['ci'][0]:+.5f}, {gain['ci'][1]:+.5f}]")
    print(f"wrote {out / 'baseline.json'}")
    if not args.no_plot:
        print(f"wrote {calibration_figure(report, out / 'baseline_calibration.png')}")
    return 0


def _walkforward(args: argparse.Namespace) -> int:
    try:
        from .config import reports_dir_for
        from .plots import walkforward_figure
        from .walkforward import run_walkforward
    except ImportError as exc:
        print(f"{exc}\nInstall the model extras: pip install -e '.[model]'", file=sys.stderr)
        return 1

    out = reports_dir_for(args.sources)
    r = run_walkforward(first_test_year=args.first_test_year, reports_dir=out, sources=args.sources)
    print(f"sources: {r['sources']}")
    print(f"test years {r['test_years']}, {r['n_test_matches']} matches; "
          f"first-year difference between arms: {r['first_year_max_abs_difference_between_arms']:.1e}")
    print(f"{'IPL, years after the first':<28} {'first-inn gap':>13} {'brier':>7}   {'chase gap':>10} {'brier':>7}")
    for arm in ("fixed", "par_oracle", "par_refresh", "full_retrain"):
        s = r["arms"][arm].get("ipl_refreshable", {})
        if s:
            fi, ch = s["first_innings"], s["chase"]
            print(f"{arm:<28} {fi['gap']:+13.3f} {fi['brier']:7.4f}   {ch['gap']:+10.3f} {ch['brier']:7.4f}")
    cause = r["verdict"]["cause_test"]["first_innings"]
    if cause:
        lo, hi = cause["brier_gain_ci"]
        print(f"\nCAUSE TEST (par_oracle, first innings): cuts the gap by {cause['gap_reduction']:.0%}; "
              f"Brier gain {cause['brier_gain']:+.5f} [{lo:+.5f}, {hi:+.5f}] -> "
              f"{'SUPPORTED' if cause['supported'] else 'NOT supported'}")
    print("practical recovery (no pass/fail label):")
    for key, v in r["verdict"]["practical_recovery"].items():
        lo, hi = v["brier_gain_ci"]
        print(f"  {key:<28} gap cut {v['gap_reduction']:+.0%}; Brier gain {v['brier_gain']:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    print(f"wrote {out / 'walkforward.json'}")
    if not args.no_plot:
        print(f"wrote {walkforward_figure(r, out / 'walkforward_calibration.png')}")
    return 0


def _rolling(args: argparse.Namespace) -> int:
    try:
        from .config import reports_dir_for
        from .plots import rolling_figure
        from .rolling import run_rolling
    except ImportError as exc:
        print(f"{exc}\nInstall the model extras: pip install -e '.[model]'", file=sys.stderr)
        return 1

    out = reports_dir_for(args.sources)
    r = run_rolling(test_start_year=args.test_start_year, first_val_year=args.first_val_year,
                    reports_dir=out, sources=args.sources)
    print(f"sources: {r['sources']}")
    v = r["validation"]
    print(f"validation {r['validation_years'][0]}-{r['validation_years'][-1]} ({r['n_validation_matches']} matches); "
          f"flag vs no-flag max prediction difference: {v['flag_vs_no_flag_max_abs_prediction_difference']:.1e}")
    print(f"{'candidate':<20}{'brier':>9}  vs best (95% CI)")
    for name, e in v["by_candidate"].items():
        ci = "best" if e["vs_best_ci"] is None else f"{e['vs_best']:+.5f} [{e['vs_best_ci'][0]:+.5f}, {e['vs_best_ci'][1]:+.5f}]"
        print(f"{name:<20}{e['brier']:9.5f}  {ci}")
    s = r["selection"]
    print(f"best: {s['best']} | indistinguishable: {s['indistinguishable_from_best']} | CHOSEN: {s['chosen']}")
    fe = r["final_exam"]
    print(f"\nfinal exam {r['test_years'][0]}-{r['test_years'][-1]} ({r['n_test_matches']} matches)")
    for sname in ("examined", "fresh"):
        block = fe["by_subset"].get(sname)
        if not block:
            continue
        label = {"examined": f"examined: {', '.join(fe['examined_sources'])}", "fresh": "fresh: all other competitions"}[sname]
        print(f"\n{label} ({block['n_matches']} matches)")
        print(f"{'':<20}{'brier':>9}{'first inn':>11}{'chase':>9}   vs no_flag/expanding (gain, 95% CI; + is better)")
        for name, e in block["by_candidate"].items():
            g = fe["paired_vs_no_flag_expanding"].get(sname, {}).get(name)
            gain = f"{g['mean']:+.5f} [{g['ci'][0]:+.5f}, {g['ci'][1]:+.5f}]" if g else ""
            print(f"{name:<20}{e['all']['brier']:9.5f}{e['first_innings']['brier']:11.5f}{e['chase']['brier']:9.5f}   {gain}")
    c = fe["confirmation"]
    if c:
        if c["supported"] and c["robust"]:
            verdict = "SUPPORTED (holds across bootstrap seeds)"
        elif c["supported"]:
            verdict = "SUPPORTED on this seed only: BORDERLINE, the lower bound changes sign with the seed"
        else:
            verdict = "NOT supported"
        lo, hi = c["ci_lower_across_5_seeds"]
        print(f"\nPRE-REGISTERED H1 on {c['n_fresh_matches']} fresh matches: no_flag/window2 minus no_flag/expanding Brier gain "
              f"{c['brier_gain']:+.5f} [{c['ci'][0]:+.5f}, {c['ci'][1]:+.5f}] -> {verdict}")
        print(f"  lower bound across 5 bootstrap seeds: {lo:+.5f} to {hi:+.5f}")
    for key, g in fe["chosen_paired"].items():
        print(f"{key} (all rows): Brier gain {g['mean']:+.5f} [{g['ci'][0]:+.5f}, {g['ci'][1]:+.5f}]")
    print(f"wrote {out / 'rolling.json'}")
    if not args.no_plot:
        print(f"wrote {rolling_figure(r, out / 'rolling.png')}")
    return 0


def _bayes(args: argparse.Namespace) -> int:
    try:
        from .bayes import run_bayes
        from .config import reports_dir_for
    except ImportError as exc:
        print(f"{exc}\nInstall the Bayesian extras: pip install -e '.[bayes]'", file=sys.stderr)
        return 1

    sources = args.sources or ["ipl"]
    out = reports_dir_for(sources)
    r = run_bayes(sources=sources, train_end=args.train_end, min_balls=args.min_balls, draws=args.draws,
                  tune=args.tune, chains=args.chains, cores=args.cores, reports_dir=out)
    d = r["data"]
    print(f"\nsources {r['sources']}: train {d['train_balls']} balls / {d['train_matches']} matches, "
          f"test {d['test_balls']} balls / {d['test_matches']} matches")
    print(f"modelled players: {d['modelled_batters']} batters, {d['modelled_bowlers']} bowlers; test balls with a "
          f"rare or unseen batter {d['test_balls_with_other_batter']:.1%}, bowler {d['test_balls_with_other_bowler']:.1%}")
    print(f"\n{'model':<8}{'gate':>8}{'log score/ball':>16}{'wicket part':>13}{'runs part':>11}")
    for name, m in r["models"].items():
        s = m["mean_log_score"]
        print(f"{name:<8}{'ok' if m['diagnostics']['gate_passed'] else 'FAILED':>8}{s['joint']:16.5f}{s['wicket']:13.5f}{s['runs']:11.5f}")
    print("\nheld-out gain per ball (positive means the first model predicts better), match-level 95% CI")
    for key, parts in r["comparisons"].items():
        j = parts["joint"]
        print(f"  {key:<16} total {j['mean']:+.5f} [{j['ci'][0]:+.5f}, {j['ci'][1]:+.5f}] "
              f"lower bound across 5 seeds {j['lower_across_seeds'][0]:+.5f} to {j['lower_across_seeds'][1]:+.5f} "
              f"-> {'ROBUST' if j['robust'] else 'not robust'}")
    print("\ninterpretable (every fit passed the convergence gate):", r["interpretable"])
    print(f"wrote {out / 'bayes.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thinedge")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="fetch Cricsheet archives (e.g. ipl hnd t20s)")
    dl.add_argument("sources", nargs="+")
    dl.add_argument("--refresh", action="store_true", help="re-download even if cached")
    dl.set_defaults(func=_download)

    bd = sub.add_parser("build", help="parse raw JSON into Parquet tables and validate")
    bd.add_argument("sources", nargs="*", help="limit to these raw sources (default: all)")
    bd.set_defaults(func=_build)

    bl = sub.add_parser("baseline", help="fit and evaluate the logistic win-probability baseline")
    bl.add_argument("--train-end", default="2024-01-01", help="train on matches before this date")
    bl.add_argument("--sources", nargs="+", help="limit to these competitions (default: all loaded)")
    bl.add_argument("--no-plot", action="store_true")
    bl.set_defaults(func=_baseline)

    wf = sub.add_parser("walkforward", help="test whether yearly retraining fixes the baseline's drift")
    wf.add_argument("--first-test-year", type=int, default=2024)
    wf.add_argument("--sources", nargs="+", help="limit to these competitions (default: all loaded)")
    wf.add_argument("--no-plot", action="store_true")
    wf.set_defaults(func=_walkforward)

    ro = sub.add_parser("rolling", help="choose how to handle drift using rolling-origin validation")
    ro.add_argument("--test-start-year", type=int, default=2024)
    ro.add_argument("--first-val-year", type=int, default=2015)
    ro.add_argument("--sources", nargs="+", help="limit to these competitions (default: all loaded)")
    ro.add_argument("--no-plot", action="store_true")
    ro.set_defaults(func=_rolling)

    by = sub.add_parser("bayes", help="fit hierarchical ball-outcome models and score them on held-out balls")
    by.add_argument("--sources", nargs="+", help="competitions to use (default: ipl)")
    by.add_argument("--train-end", default="2024-01-01")
    by.add_argument("--min-balls", type=int, default=60, help="players with fewer training balls share one level")
    by.add_argument("--draws", type=int, default=500)
    by.add_argument("--tune", type=int, default=500)
    by.add_argument("--chains", type=int, default=4)
    by.add_argument("--cores", type=int, default=4)
    by.set_defaults(func=_bayes)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
