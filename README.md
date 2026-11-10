# Thin Edge

Ball-by-ball T20 win probability with honest uncertainty, built to be fast enough to reprice every
delivery and checked for calibration rather than just accuracy.

*A thin edge* is both a small betting advantage and a faint nick off the bat.

**Status: data layer, logistic baseline and a walk-forward test are done.** The baseline explains
the chase well (Brier skill 0.47 over a constant reference) but is poorly calibrated in the first
innings and in the newest IPL seasons. A walk-forward test found that a stale par curve explains only
about a third of that; most of it came from a regime flag fitted on 71 matches. A rolling-origin
evaluation showed that seasons before 2024 cannot choose a drift strategy (they contain no drift to
adapt to), while in 2024-26 a short training memory or a yearly-retrained flag both help the IPL. Ten
more leagues (12 competitions, 5,707 matches) were then added to test that on competitions never
examined: the direction repeats but the effect is a fifth the size, uneven across leagues, and not robust.
Drift handling is not the main lever; team and player strength is the next one.

**Not yet done:** a hierarchical Bayesian ball-outcome model (PyMC, `src/thinedge/bayes.py`) is written
and passes its tests on *simulated* matches, but it has **not been run on the real data**. There is no
link from it to match win probability yet. See [DECISIONS.md](DECISIONS.md).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,model]"   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m thinedge download ipl hnd bbl psl cpl sat ilt mlc ntb bpl lpl msl   # cached
.venv/Scripts/python -m thinedge build                  # parse, validate, write Parquet
.venv/Scripts/python -m thinedge baseline               # fit, evaluate, write reports/all/
.venv/Scripts/python -m thinedge walkforward            # test whether yearly retraining fixes the drift
.venv/Scripts/python -m thinedge rolling                # choose a drift strategy on pre-2024 seasons only
.venv/Scripts/python -m pytest                          # fast tests, about a minute
.venv/Scripts/python -m pytest -m slow                  # Bayesian sampling tests, about 30 minutes
```

The Bayesian model needs `pip install -e ".[bayes]"` and is run with `python -m thinedge bayes`
(it has not been run on the real data; see above).

Other Cricsheet sources work the same way: `t20s` (T20 internationals), `wbb`, `wpl`...
(the name is the file stem on https://cricsheet.org/downloads/).

`baseline`, `walkforward` and `rolling` take `--sources` to restrict the competitions, and write to
`reports/<sources>/` (`reports/all/` by default). The IPL-and-Hundred results quoted in
[DECISIONS.md](DECISIONS.md) up to the "ten more leagues" section are `--sources ipl hnd`, in
`reports/hnd+ipl/`.

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
