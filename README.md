# Thin Edge

For every ball of a T20 match, Thin Edge estimates the chance that the batting side wins, using only
what is known before that ball, and checks whether those chances can be trusted (when it says 70%,
does that happen about 70% of the time?).

*A thin edge* is both a small betting advantage and a faint nick off the bat.

## In short

- **What works.** In the chase, on 399 IPL and Hundred matches from 2024 to 2026 that the model never
  saw, its error is 47% lower than a forecast that only knows the long-run win rate (Brier 0.132 against
  0.249).
- **What does not.** The first innings is poorly calibrated: in the IPL test matches the side batting
  first was given a 59% chance on average and won 45% of the time. Most of that traces to a rule-change flag learned from
  just 71 matches ([DECISIONS.md](DECISIONS.md), walk-forward section).
- **The yardstick is modest.** The only comparison so far is that constant win rate. It has not been
  compared with a par-score or Duckworth-Lewis-style rule, or with betting markets; that is the next step.
- **Player model: inconclusive.** A Bayesian model of individual batters and bowlers has had one short
  run on real IPL data. It did not pass the convergence rule set in advance. Taken at face value, the
  player ratings predicted 2024+ matches slightly *worse* than a model with no players.
- **How it was checked.** Confidence intervals resample whole matches. Decision rules were written into
  the code before each real run; there is no outside proof of that, as the git history starts on
  10 Nov 2026. Parsed scores are cross-checked against the win margins Cricsheet records, which found
  and fixed three parsing bugs.
- **Built with Claude Code** (an AI coding agent). [DECISIONS.md](DECISIONS.md) records each decision
  and each place the AI was wrong.

*Terms.* **Brier score**: the average squared gap between a forecast probability and what happened
(0 is perfect; always saying 50% scores 0.25). **Skill**: the share of the reference forecast's error
that the model removes (0.47 here). **Chase**: the second innings, when the target is known.

## Findings in more detail

The baseline explains the chase well (Brier skill 0.47 over a constant reference) but is poorly
calibrated in the first innings and in the newest IPL seasons. A walk-forward test found that a stale par curve explains only
about a third of that; most of it came from a regime flag fitted on 71 matches. A rolling-origin
evaluation showed that seasons before 2024 cannot choose a drift strategy (they contain no drift to
adapt to), while in 2024-26 a short training memory or a yearly-retrained flag both help the IPL. Ten
more leagues (12 competitions, 5,707 matches) were then added to test that on competitions never
examined: the direction repeats but the effect is a fifth the size, uneven across leagues, and not robust.
Drift handling is not the main lever; team and player strength is the next one.

The hierarchical Bayesian ball-outcome model (PyMC, `src/thinedge/bayes.py`) passes its tests on
simulated matches. One short run on real IPL data (995 training matches, tested on 214 from 2024 on;
about 6.7 hours on a laptop) **failed the convergence rule**, so the result is inconclusive. Read
informally, adding batter and bowler effects made held-out predictions of runs worse, not better
(`reports/ipl/bayes.json`; DECISIONS.md, player-model section). The model is not yet linked to match
win probability.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,model]"   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m thinedge download ipl hnd bbl psl cpl sat ilt mlc ntb bpl lpl msl   # cached
.venv/Scripts/python -m thinedge build                  # parse, validate, write Parquet
.venv/Scripts/python -m thinedge baseline               # fit, evaluate, write reports/all/
.venv/Scripts/python -m thinedge walkforward            # test whether yearly retraining fixes the drift
.venv/Scripts/python -m thinedge rolling                # choose a drift strategy on pre-2024 seasons only
.venv/Scripts/python -m pytest                          # fast tests incl. property tests, about a minute
.venv/Scripts/python -m pytest -m slow                  # Bayesian sampling tests, about 30 minutes
```

The Bayesian model needs `pip install -e ".[bayes]"` and is run with `python -m thinedge bayes`
(one short IPL run so far; it did not converge, see above).

Other Cricsheet sources work the same way: `t20s` (T20 internationals), `wbb`, `wpl`...
(the name is the file stem on https://cricsheet.org/downloads/).

`baseline`, `walkforward` and `rolling` take `--sources` to restrict the competitions, and write to
`reports/<sources>/` (`reports/all/` by default). The IPL-and-Hundred results quoted in
[DECISIONS.md](DECISIONS.md) up to the "ten more leagues" section are `--sources ipl hnd`, in
`reports/hnd+ipl/`.

## Reproduced from a fresh clone

On 2026-11-10 the first commit (`bf44bae`) was cloned into an empty folder, given a new virtual
environment, installed with `pip install -e ".[dev,model]"`, and run from nothing: the IPL and The
Hundred archives were downloaded again, the tables rebuilt, and `baseline`, `walkforward` and `rolling`
re-run with `--sources ipl hnd`.

- The two Cricsheet archives were byte-identical to the originals (same SHA-256).
- Every value in the three reports matched the originals (maximum difference 0), apart from the
  ILT20 and LPL rule dates, which only exist when those leagues are loaded.
- The headline figures reproduced: chase Brier skill 0.470, first-innings skill 0.112.
- The fast tests passed (101, with the Bayesian module skipped because the `bayes` extra was not installed).

Not covered by that check: the all-league runs (they need ten more downloads), installing the `bayes`
extra, and the slow Bayesian sampling tests.

That check compared the code with itself. A later testing round (property tests and exploratory
sessions, see [DECISIONS.md](DECISIONS.md)) compared it with Cricsheet's own recorded win margins and
found three parsing bugs in 10 matches (penalty runs, a double wicket, "retired not out"), none in the
IPL or The Hundred. They are fixed; the IPL-and-Hundred numbers above are unchanged and the all-league
report moved by at most 0.00016. A fresh clone built before that round will differ in those matches.

## Tables (`data/processed/*.parquet`)

| Table | Grain | Notes |
|---|---|---|
| `matches` | one per match | result, winner, toss, venue, D/L method, Impact Player substitution count |
| `innings` | one per innings | target, `max_balls`, totals, `has_miscounted_overs`, super-over flag |
| `balls` | one per delivery | batter/bowler **registry ids**, extras, wicket kind, powerplay, legality |
| `states` | one per delivery | match state before and after the ball, run rates, phase, **label** `batting_team_won` |
| `venues` | one per venue and city | match counts; `latitude`/`longitude` empty, reserved for the weather join |
| `quality_report.json` | n/a | invariants checked on every build |

```python
from thinedge.db import connect
con = connect()   # in-memory DuckDB with a view per table
con.sql("""
    select season, round(avg(total_runs), 1) as avg_first_innings
    from innings join matches using (match_id)
    where source = 'ipl' and innings_no = 1 and not is_super_over and match_id not in
          (select match_id from states where is_reduced)
    group by 1 order by 1
""").show()
```

## Roadmap

1. ~~Data layer: Cricsheet to Parquet and DuckDB, validated~~
2. ~~Baseline: logistic win-probability model on `states`, with calibration plots and Brier score~~
   (`reports/hnd+ipl/baseline_calibration.png`; findings in DECISIONS.md)
3. Bayesian ball-outcome model (PyMC): batter, bowler, phase and venue effects, a scoring environment
   that varies by competition and season, retrained walk-forward, leagues sharing strength
4. Monte Carlo innings simulator, then a neural surrogate that reproduces it in milliseconds
5. Market comparison against Betfair historic prices (needs a Betfair account; check their data terms first)
6. Regime-shift radar (the 2023 IPL Impact Player rule is already visible in the data), a replay UI, an LLM match report

Design decisions and the data issues found so far are in [DECISIONS.md](DECISIONS.md).

## Data and attribution

Match data comes from [Cricsheet](https://cricsheet.org) under the
[Open Data Commons Attribution License](http://opendatacommons.org/licenses/by/1.0/). Any public
use, including a demo or write-up, must attribute Cricsheet and keep the licence notices.
Raw data is downloaded locally and is not committed (`data/` is git-ignored).
