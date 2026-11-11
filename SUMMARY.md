# Thin Edge: a one-page summary

**What it does.** For every ball of a T20 cricket match, it estimates the probability that the side
batting will win, using only what is known before that ball. The aim was a model whose probabilities
can be trusted (when it says 70%, that happens about 70% of the time), not just one that picks winners.

**Data.** About 5,700 matches and 1.3 million deliveries from 12 T20 leagues (IPL, The Hundred, Big Bash,
PSL and others), from Cricsheet, which is credited below. Everything is rebuilt from the raw files with
one command, and a fresh clone reproduced every reported number exactly.

## What worked

- **The chase is predictable.** Trained on matches before 2024 and tested on 399 IPL and Hundred matches
  from 2024 to 2026, the model's average squared error (Brier score) in the second innings was 0.132. A
  forecast that only knows the long-run win rate scores 0.249, so the model cuts that error by 47%.
- **Retraining every year helps.** It improved both innings in the IPL, and on 853 matches from leagues
  that were never examined, a model trained once was reliably worse than one retrained each year.
- **Checks that catch mistakes.** Intervals resample whole matches rather than single balls, because balls
  in one match are not independent. Each experiment was tried first on simulated matches where the right
  answer was known.

## What did not work, and what I got wrong

- **The first innings is poorly calibrated.** In the 2024-26 IPL matches the model gave the side batting
  first an average 59% chance, but they won 45% of the time.
- **My first explanation was wrong.** I blamed an out-of-date par score. A pre-set test showed that
  explains only about a third of the gap. Most of the rest came from a rule-change flag the model had
  learned from just 71 matches.
- **A result that looked confirmed was not.** One pre-set hypothesis printed "supported" by a hair. A
  different random seed flipped it, so it is reported as not reliably confirmed.
- **Bugs my own tests missed.** My first tests and a clean-clone rerun only showed that the code
  reproduces itself. Checking it against the win margins Cricsheet itself records, and then running
  property-based and exploratory tests, found three parsing bugs (penalty runs, a double wicket, "retired
  not out") in 10 of 5,707 matches, none in the IPL or The Hundred. All are fixed, with tests that failed
  first.
- **A bug that stayed silent.** The first run over all 12 leagues printed `nan`, because 124 matches had no
  recorded chase target and the model let the gaps pass through. It now stops with an error, and the
  invalid run was discarded.
- **Drift is a small effect.** Handling the 2023 Impact Player rule change matters mostly for the IPL.
  Across the new leagues the effect is about a fifth the size and not consistent from league to league.

## Not done

- A hierarchical Bayesian model of individual batters and bowlers passes its tests on simulated players.
  One short run on real IPL data (about 6.7 hours) **did not pass the convergence rule I set in advance**,
  so it is inconclusive. Taken at face value, player ratings predicted 2024+ matches slightly worse than
  a model with no players. A proper run needs roughly ten times as many samples. The model is not yet
  linked to match win probability.
- The only yardstick so far is the constant win rate. I have not compared against a par-score or
  Duckworth-Lewis-style rule, or against betting-market prices.
- Team and player strength is the obvious next step, since the first innings has almost nothing to go on
  before a ball is bowled.
- The evaluation matches have all been looked at now, so a fair new test needs 2027 data or other leagues.

## How AI was used

Built with Claude Code (Anthropic's AI coding agent), which wrote the code, tests and experiments.
[Add one or two sentences of your own here: why you chose cricket, and what you checked or decided yourself.]
`DECISIONS.md` records each decision and each place the AI was wrong, including the ones above.

## Data and licence

Match data is from [Cricsheet](https://cricsheet.org) under the
[Open Data Commons Attribution License](http://opendatacommons.org/licenses/by/1.0/). Full results,
protocols and limits are in [DECISIONS.md](DECISIONS.md); how to run it is in [README.md](README.md).
