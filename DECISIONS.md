# Decisions and findings

A running log of what was decided, why, and what the data turned out to contain. Written as the
work happens, so it can be reused as the "aims, decisions, learnings" note for an application.

## 2026-11-10: phase 1, data layer

**Scope of the first load.** IPL (1,243 matches) and The Hundred (389). Together they give two
ball formats (6 and 5 balls per over), a rule change (Impact Player, 2023) and rain-affected matches.
That is enough to exercise every code path. Add `t20s` and the other leagues once the model
needs more data.

### Decisions

| Decision | Why |
|---|---|
| Players keyed by Cricsheet **registry id**, not name | Names collide and change; the id is stable. |
| Legal ball = not a wide, not a no-ball | Byes and leg byes still use up a ball. Wides and no-balls do not. |
| "retired hurt" is **not** a wicket | Cricsheet lists it under `wickets`, but the team has not lost a wicket. Counting it would corrupt wickets-in-hand. |
| Over notation converted properly (9.2 overs of 6 = 56 balls) | Treating it as a decimal gives 55.2 balls. This affects every D/L target. |
| `states` keeps **pre and post** columns | Win probability uses the post-ball state. A ball-outcome model must use the pre-ball state or it sees the answer. |
| Label is null for ties, no-results and super overs | No clean winner in regulation, so those rows teach nothing about regulation win probability. They are kept and flagged. |
| `is_reduced` = D/L method present **or** any regulation innings shorter than nominal | The first version used only the D/L method and missed a rain-abandoned Hundred match (id 1471034). A shortened chase is the more reliable signal. |
| `balls_remaining` and `runs_needed` clipped at 0 | Miscounted overs and finished chases would otherwise go negative. |
| Phase is format-agnostic: "death" = last 20% of the innings | 24 balls in T20, 20 balls in The Hundred. Powerplay is taken as recorded. |
| Archives treated as untrusted: flat `.json` names only, size caps, per-source folder | Cheap to do, and it removes a whole class of zip-extraction problems. |
| Separate virtual environment and folder from the interview-prep project | PyMC will be heavy; it should not touch that environment. |

### What the data showed

- Row counts match Cricsheet's published counts: 1,632 matches, 3,288 innings, 371,368 deliveries.
- Opening IPL match of 2017: SRH 207/4, RCB 172 all out, 35-run win. The pipeline reproduces it.
- **Impact Player shows up in the data.** Mean IPL first-innings score (full-length matches):
  171.1 in 2022, 182.2 in 2023, 189.6 in 2024, 189.3 in 2025, 192.8 in 2026. A model trained across
  that boundary without a regime term will be miscalibrated for recent seasons.
- IPL chasing sides won 54.7% of 1,218 decided regulation matches.
- 46 matches are rain-reduced (34 IPL, 12 Hundred). 10 innings have miscounted overs (up to 121 legal balls in a
  120-ball innings). 184 super-over deliveries.

### Known data issues, not yet fixed

- **Venue names are not canonical.** 54 venue and city rows cover 49 distinct names; for example
  Chinnaswamy appears as "Bangalore" and "Bengaluru" under two spellings. Resolve this before
  geocoding venues for the weather join.
- 51 matches have no city.
- Team names change across seasons (Delhi Daredevils to Delhi Capitals, Kings XI Punjab to
  Punjab Kings, Bangalore to Bengaluru). Needed for any team-strength feature.
- First-innings length is assumed nominal. In a match cut short during the first innings
  that is wrong, which is why `is_reduced` matters.
- Cricsheet has **no timestamps per ball**, so aligning ball-by-ball events to in-play market
  prices will need a matching trick (large price jumps to wickets, then time warping).

### Not built yet

Weather join, Betfair comparison, any model. Betfair historic data needs the user's own account,
and its terms must be read before use.

## 2026-11-10: baseline win-probability model

Two logistic regressions (first innings, chase) fitted by a numpy Newton solver, evaluated on a time
split: train on 1,151 matches before 2024-01-01, test on 399 matches after it (208 IPL 2024-26, 191
The Hundred; 87,800 pre-ball states). Rain-reduced matches are excluded. Reproduce with
`python -m thinedge baseline --sources ipl hnd`; the numbers and bins are in
`reports/hnd+ipl/baseline.json` and the plot in `reports/hnd+ipl/baseline_calibration.png`. (Every
experiment up to the "ten more leagues" section was run on IPL and The Hundred only, so all of them
need `--sources ipl hnd`, which re-runs to identical numbers.)

### Protocol decisions

| Decision | Why |
|---|---|
| Predict from the state **before** each ball | Post-ball states after the result is certain would flatter every score. Only pre-ball columns are selected from `states`. |
| Split by **date**, not randomly | Random splits leak the same season, venue and era into both sides. The test period is the Impact Player era, deliberately a stress test. |
| Confidence intervals resample **whole matches** | Balls within a match are strongly dependent; row-level intervals would be far too narrow. |
| Reference is a constant per-innings base rate | "Brier skill" then means improvement over knowing nothing. |
| Nothing tuned on the test period | Regularisation is the default `C=1`, the split date and the regime ablation were fixed before looking at results. |
| Regime flag run as an **ablation** (with and without) | So its value is measured, not assumed. |

### Results (with regime flag, test period)

| | Brier | Constant reference | Skill vs reference (95% CI) | Calibration error |
|---|---|---|---|---|
| Chase | 0.132 | 0.249 | 0.470 (0.408 to 0.531) | 0.060 |
| First innings | 0.220 | 0.247 | 0.112 (0.050 to 0.173) | 0.106 |
| All | 0.178 | 0.248 | 0.284 (0.229 to 0.342) | 0.033 |

- Chase Brier at 25%, 50% and 75% of the innings: 0.154, 0.131, 0.106 (reference about 0.248).
- First-innings Brier at the same points: 0.232, 0.218, 0.197 (reference 0.247). Before a ball is
  bowled the first innings says little, which is expected without team or player strength.
- The regime flag helps overall: Brier gain of +0.00124 per ball (95% CI +0.00074 to +0.00177). The
  gain is in the chase (0.1319 vs 0.1362 without it). In the first innings it does not help (0.2195
  vs 0.2180).

### What is wrong with it (the useful part)

- **First innings over-favours the team batting first.** In the IPL test matches the model's mean
  prediction is 0.589 against an observed 0.446 (The Hundred: 0.490 vs 0.433). The whole
  reliability curve sits below the diagonal.
  - The par curve (expected score at each ball, learned in training) is about 12 runs too low in
    the test period: 171.9 against an actual 184.0 at ball 114 in IPL matches. "Runs above par" is
    therefore inflated for every side, and the model reads an era-wide rise in scoring as a
    first-innings advantage.
  - Scoring kept rising after the 71 Impact-era training matches (all 2023): 182.2 in 2023, 189.6
    in 2024, 192.8 in 2026.
  - **Correction (added after the walk-forward test below):** I first called stale par "the most
    likely cause". That was incomplete. The test did not support it as the main cause (par explains
    roughly a third of the gap), and the largest piece turned out to be the regime flag itself.
- **Chase under-rates the chasing side in the IPL**: mean prediction 0.448 vs observed 0.534 (The Hundred:
  0.515 vs 0.538). Required rates that were rare before 2023 are being chased down more often now.
  A single on/off flag cannot express that, and its coefficient is small (0.05).
- The Hundred is better calibrated than the IPL (first innings 0.082 vs 0.143, chase 0.035 vs 0.086),
  which fits the problem being a moving scoring environment rather than a bad model form.

### What this means for the plan

The baseline sets a clear target for the Bayesian model: a **scoring environment that varies by
competition and season and is partially pooled**, retrained walk-forward, instead of a fixed par
curve and a flag. Beating this is judged on calibration in the latest seasons, not average accuracy.
Cheap things the baseline still lacks: walk-forward retraining, team and player strength, venue.

### Environment note

scikit-learn could not be used here: Windows Application Control blocked its compiled extension
(`_dist_metrics`). I did not try to get around the policy. The logistic regression is a small numpy
Newton solver (`Logit`), checked against its own optimality condition (gradient below 1e-5) and by
recovering known coefficients. The `model` extra now needs only matplotlib.

**PyMC check (done in a throwaway environment, not the project one).** PyMC 6.4.0 with PyTensor 3.3.8
installs and runs on this machine. numba and llvmlite import and JIT-compile without being blocked.
There is no C compiler here, so PyTensor uses its non-C backend; the default mode performs the same as
choosing numba explicitly (100,000 rows, 2 chains of 300 draws: 70.9 s against 64.6 s, identical
estimates), so nothing special is needed.

A 7-feature logistic NUTS model on simulated data recovers its true coefficients:

| Rows | Chains x draws | Time | Max R-hat | Min effective sample size | Max coefficient error |
|---|---|---|---|---|---|
| 5,000 | 4 x 1,000 | 15.6 s | 1.002 | 2,935 | 0.038 |
| 200,000 | 4 x 500 | 154.5 s | 1.003 | 2,028 | 0.011 |

Cost grows roughly linearly with rows, so by extrapolation (not measured) the full 1.3M-row table would
take about 15 minutes for this simplest model, and hierarchical player and venue effects will cost more.
Practical notes:

- On Windows, scripts that sample several chains in parallel need an `if __name__ == "__main__":`
  guard. My first multi-process failure was this, not the policy.
- ArviZ 1.x returns `DataTree` objects, so older code such as `az.rhat(idata).to_array()` breaks; use
  `az.summary`.
- PyMC is **not** installed in the project environment and is not yet in `pyproject.toml`.

## 2026-11-10: walk-forward test of the "stale par" explanation

`python -m thinedge walkforward --sources ipl hnd`; numbers in `reports/hnd+ipl/walkforward.json`, plot in
`reports/hnd+ipl/walkforward_calibration.png`. Test years 2024-26 (399 matches). The comparison that matters
is IPL in 2025 and 2026 (138 matches), the years where a refresh has newer data to use. In 2024 all
non-oracle arms are identical to the fixed model by construction (checked: difference 0).

### Arms

| Arm | What changes | IPL 2025-26 first-innings gap | Brier | Chase gap | Brier |
|---|---|---|---|---|---|
| fixed | the shipped baseline | +0.180 | 0.2437 | -0.103 | 0.1293 |
| par_oracle | par curve fit on the test year's scores (diagnostic, uses no results) | +0.120 | 0.2242 | -0.103 | 0.1293 |
| par_refresh | par curve refit each January on earlier data | +0.162 | 0.2364 | -0.103 | 0.1293 |
| full_retrain | everything refit each January | +0.137 | 0.2284 | -0.074 | 0.1203 |

Gap is mean prediction minus observed win rate, so positive means over-favouring the side batting first
(or, in the chase, negative means under-rating the chasing side).

### Verdict on the pre-set criterion: not supported

The rule, fixed before the real data was run: the explanation is supported only if the oracle par
curve cuts the IPL first-innings gap by at least 50% and its Brier gain over the fixed model has a
95% interval above zero. Result: the oracle cut the gap by **33%**, with a Brier gain of +0.0195
(95% CI +0.0096 to +0.0283). The gain is real but the gap is mostly still there, so stale par is a
partial contributor, not the main cause.

Practical recovery, with no pass or fail label:

- par_refresh, first innings: gap cut 10%, Brier gain +0.0073 (+0.0041 to +0.0101).
- full_retrain, first innings: gap cut 24%, Brier gain +0.0153 (+0.0084 to +0.0216).
- full_retrain, chase: gap cut 28%, Brier gain +0.0089 (+0.0047 to +0.0128). Yearly retraining
  helps the chase, so part of that error is stale coefficients, but most of it remains.

**The criterion was changed once, and why.** The first draft applied the 50% rule to `par_refresh`.
I checked the design on simulated matches where scoring drifts upward every year, so stale par is a
true cause by construction. There `par_refresh` cut only 18% of the gap, so the rule would have
rejected a cause I know is real: an expanding window keeps averaging in old seasons and lags behind
continued drift. I moved the cause test to the oracle arm. On simulated data the oracle cuts 95% of
the gap with drift and does nothing without it. The change was made on simulated data, before any real
walk-forward result had been seen.

### Where the first-innings gap actually comes from (exploratory, after the test)

This part was done after seeing the verdict, so treat it as a lead to verify, not a tested result.
IPL 2025-26, mean prediction minus observed win rate at points of the innings:

| Point of innings | fixed | par_oracle |
|---|---|---|
| Before the first ball | +0.110 | +0.105 |
| 25% | +0.167 | +0.124 |
| 50% | +0.186 | +0.125 |
| 75% | +0.193 | +0.113 |

- A gap of +0.11 exists before a ball is bowled, and no par curve can touch it. Par accounts for the
  part that grows through the innings (about 0.06 by the halfway point, 0.08 by 75%).
- **The regime flag causes most of the constant part.** In the 71 Impact-era training matches the
  side batting first won 54.9% (standard error about 0.06), against 45.0% in the 906 earlier IPL
  matches. The model turned that small-sample swing into a +0.44 log-odds shift. For IPL 2024-26 the
  first-ball prediction is 0.530 with the flag, 0.431 without it, and 0.442 observed. In training
  the first-ball prediction was fine (0.440 vs 0.465 observed), so this is not a model-form problem.
- This matches the earlier ablation: the flag helped the chase but not the first innings (Brier
  0.2195 with it, 0.2180 without).
- Batting-first win rate in the IPL test matches was 0.485 in 2025 (68 matches) and 0.357 in 2026 (70
  matches), against 0.450 in earlier training. One season of 70 matches cannot separate a real rise in
  the chasing advantage from noise (standard error about 0.06). The fixed model's gap by year was
  +0.070 (2024), +0.109 (2025), +0.249 (2026), so 2026 drives much of the total.

### What this means

- **Do not fit a regime indicator from one season.** The flag's first-innings coefficient is mostly noise from
  71 matches. The next model should pool or shrink regime effects, or select them on a rolling origin
  rather than by looking at the test set. (Update: the rolling-origin section below tried this. Pre-2024
  validation could not choose, and the flag stops hurting once 2-3 Impact seasons are in the training data.)
- Par drift is real and worth modelling, but it is the smaller part of the first-innings error.
- Yearly retraining is worth keeping: it improves both innings with intervals above zero.
- I did not re-tune the baseline against this test set; doing so would make the reported numbers
  optimistic. The next model should be judged with rolling-origin evaluation (train on past, predict
  the next season, repeat), not a single split.

### Limits

One split, three test seasons, two competitions, 138 matches in the key comparison. Intervals resample
whole matches but cannot account for season-level shocks. The decomposition above is post-hoc.

## 2026-11-10: rolling-origin evaluation for handling drift

`python -m thinedge rolling --sources ipl hnd`; numbers in `reports/hnd+ipl/rolling.json`, plot in
`reports/hnd+ipl/rolling.png`.

**Idea.** The flag cannot be validated before it has been in force for several seasons, so choose among
strategies that *can* be validated on the long pre-2024 history: no flag with all history, no flag with
only the last 5, 3 or 2 years, and the current flag model. For each year from 2015 to 2023, train on
everything before it (or only its window) and predict it. The selection rule was written into the
module docstring before the real data was run: lowest pooled validation Brier, then the simplest
candidate whose 95% paired match-level interval cannot be told apart from the best. The flag ranks as
least preferred because it is the least validated.

### Validation 2015-2023 (653 matches): the data could not choose

| Candidate | Pooled Brier | Minus best (95% CI) |
|---|---|---|
| no_flag/window5 | 0.18972 | best |
| no_flag/expanding | 0.18985 | +0.00013 (-0.00170 to +0.00192) |
| flag/expanding | 0.18985 | identical to no_flag/expanding |
| no_flag/window2 | 0.19039 | +0.00066 (-0.00177 to +0.00302) |
| no_flag/window3 | 0.19106 | +0.00134 (-0.00050 to +0.00302) |

Every interval includes zero, so by the rule the choice is the simplest: **no_flag/expanding**. The flag
model's predictions match the no-flag model's to 2e-9 in every fold, because no fold before 2024 trains
on any flagged season. The harness shows by construction that the flag cannot be assessed on data before
2024. More fundamentally, scoring in 2015-2022 was stable enough that no strategy had anything to adapt to.

### Final exam 2024-2026 (399 matches): the selected choice did poorly

| Candidate | Pooled Brier | First innings | Chase | IPL 2025-26 gap, first / chase |
|---|---|---|---|---|
| no_flag/expanding (selected) | 0.17721 | 0.21607 | 0.13489 | +0.155 / -0.122 |
| no_flag/window5 | 0.17378 | 0.21444 | 0.12951 | +0.157 / -0.097 |
| no_flag/window3 | 0.17347 | 0.21495 | 0.12829 | +0.155 / -0.080 |
| no_flag/window2 | 0.17143 | 0.21323 | 0.12591 | +0.131 / -0.064 |
| flag/expanding | 0.17253 | 0.21311 | 0.12834 | +0.137 / -0.074 |
| fixed baseline (shipped) | 0.17759 | 0.21952 | 0.13194 | +0.180 / -0.103 |

- The selected candidate is indistinguishable from the shipped baseline (Brier gain +0.00038, 95% CI
  -0.00029 to +0.00109) and significantly worse than the flag model retrained yearly (-0.00468, 95% CI
  -0.00650 to -0.00279). I did **not** re-select on this table; that would be tuning on the test set. The
  shipped default is unchanged.
- Regression check: the flag/expanding row reproduces the earlier walk-forward `full_retrain` gaps
  (+0.137 and -0.074).

### Post-hoc reading of the final exam (exploratory)

- Memory length matters in the IPL, in order: all history 0.1813, last 5 years 0.1767, last 3 years
  0.1743, last 2 years 0.1712 (flag model 0.1732, shipped baseline 0.1813). The Hundred, which had no rule
  change, shows no pattern (all candidates 0.170 to 0.173).
- The 2-year window beats the shipped baseline by +0.0062 (95% CI +0.0027 to +0.0097) and no-flag/expanding by
  +0.0058 (+0.0025 to +0.0090). It cannot be told apart from the flag model (+0.0011, -0.0013 to +0.0036).
- The advantage is not uniform across years. Pooled Brier by year (2024 / 2025 / 2026): expanding
  0.1724 / 0.1675 / 0.1914; 2-year window 0.1703 / 0.1668 / 0.1772; flag model 0.1716 / 0.1644 / 0.1814.
  The window wins clearly only in 2026, the season where the side batting first won 35.7%, so part of the
  gain may be adapting to one unusual season.
- The 2-year window is the shortest candidate, at the edge of the grid, so the best memory length is unknown.

### What this means

- **The flag is not inherently harmful.** The earlier finding was about a flag fitted from one season. With
  yearly retraining and 2-3 seasons of Impact data it helps the first innings (0.2131 vs 0.2161 without it).
  A short memory does about as well without needing to know the rule date, and would also cope with
  changes nobody anticipated.
- **Validation on the past cannot choose a response to a change that has not yet happened in it.** To
  validate drift handling, the validation needs several drift episodes. The cheapest source is more
  competitions (BBL, PSL, CPL, SA20 and others) which have their own rule and era changes, and which would
  also give a holdout that has not been examined.
- The test period is **not a clean holdout** any more. Treat this section as a finding about where the
  information is, not as confirmation of a method.
- Design pointer for the Bayesian model: a scoring environment that evolves over time and is partially
  pooled, which sits between "all history" and "last two years".

### Fix made along the way

The par-curve lookup fell back to a curve pooled over all competitions when a (competition, regime) pair
was unseen in training. In a rolling fold where the flag first appears in the predicted year, that would
have given the flag model IPL-plus-Hundred scores and looked like an effect of the flag. It now falls back
to the same competition first. No earlier result changed (all 1,660 values in the baseline and
walk-forward reports re-ran identically).

## 2026-11-10: ten more leagues

Added BBL, PSL, CPL, SA20 (`sat`), ILT20, MLC, T20 Blast (`ntb`), BPL, LPL and Mzansi Super League
(`msl`): about 16.7 MB of Cricsheet archives. The match counts agree with Cricsheet's published figures.
The tables now hold 5,707 matches, 11,416 innings and 1,315,435 deliveries across 12 competitions.

### Why these, and what was left out

- The aim was independent drift episodes for validating a drift strategy, and a holdout that earlier
  steps never looked at. The new leagues' 2024+ matches have not been scored by any model or inspected
  before this step, so they serve as the "fresh" set.
- Men's 20-over leagues only (all are 20 overs of 6 balls). Women's leagues are left out because gender is
  not modelled (The Hundred already pools men and women, a known gap). T20 internationals are left out
  because their opposition quality is far more varied.
- Coverage of full-length, decided matches per year (examples): T20 Blast about 100-120 a year since 2014,
  BBL 30-60, CPL about 30, PSL about 30. SA20, ILT20 and MLC start in 2023, so under the cold-start rule
  (at least 50 earlier matches) they are scored from 2025, 2025 and 2026 respectively. About 1,000 matches
  dated 2024+ fall outside IPL and The Hundred.

### Findings in the new data

- **Substitution rules are not IPL-only.** Impact-style substitutions appear in ILT20 (from 2024-01-19:
  32 matches before, every one of 102 after) and LPL (from 2026-07-17). That gives the flag more than one
  league's evidence, and a clean second regime episode in a different competition.
- 23 innings now have miscounted overs (up from 10). 183 matches have no city. No error-level
  checks fail.
- **124 decided matches have a chase with no recorded target** (about 120 of them full-length: 119 T20
  Blast and 1 CPL, none in IPL or The Hundred). Their `target_runs` is empty, which made every chase
  feature NaN. Modelling now excludes these matches (both innings, so the two innings cover the same
  matches), and the build reports them as a warning. I did not fill the target in as first innings + 1,
  because a missing target could also hide a rain-shortened chase that looks full-length.
- **The first all-league run was invalid and was discarded.** The NaN flowed silently through the
  logistic fits, scores and bootstraps, giving `nan` Brier scores and intervals that did not contain
  their own estimates, and the plot crashed on it. The model now raises an error naming the offending
  column on any non-finite input, and the all-league runs are launched with numeric warnings turned
  into errors.
- Four matches (three T20 Blast, one BBL) had a chase target that is not first innings + 1. I guessed
  "rain revisions or penalty runs" and left them as warnings. The testing round below showed they are
  penalty runs, which the parser was ignoring; that is fixed and the warning now reads zero.

### Reproducibility

`--sources ipl hnd` re-runs the earlier experiments on the original two competitions. After the new
leagues were loaded and the target-missing exclusion was added, I re-ran all three on freshly written
reports and compared them to the originals: all 2,014 earlier values (364 baseline, 1,296 walk-forward,
354 rolling) match with a maximum difference of 0. New reports go to `reports/hnd+ipl/`; unrestricted
runs go to `reports/all/`.

### Results with all twelve competitions

`python -m thinedge rolling` (numbers in `reports/all/rolling.json`, plot `reports/all/rolling.png`).
These numbers were regenerated after the parser fixes in the testing section below. They moved by at
most 0.00016 (a validation-year Brier); every conclusion is unchanged. The IPL-and-Hundred reports did
not move at all.
Validation 2015-2023 is 2,706 matches (up from 653). The test years 2024-26 hold 1,252 matches: 399
examined (IPL and The Hundred) and 853 fresh. The models now pool all competitions (shared coefficients,
a par curve per competition), so absolute numbers are **not comparable** with the IPL-and-Hundred-only
sections above.

**Validation still could not choose.** Pooled Brier: no_flag/expanding 0.18161 (best); window5 +0.00018
(-0.00017 to +0.00050); window3 +0.00038 (-0.00027 to +0.00101); window2 +0.00008 (-0.00070 to +0.00085);
the flag model identical to no-flag (difference 2e-9, the solver's tolerance, as before). By the rule the choice is again
no_flag/expanding. Four times as many validation matches changed nothing: before 2024 there is no drift
for these strategies to exploit.

**Final exam, pooled Brier (gain over no_flag/expanding, 95% CI, positive is better):**

| Candidate | Examined (399 matches) | Fresh (853 matches) |
|---|---|---|
| no_flag/expanding | 0.18164 | 0.17701 |
| no_flag/window5 | 0.17771 (+0.00393, +0.00278 to +0.00520) | 0.17593 (+0.00107, +0.00057 to +0.00162) |
| no_flag/window3 | 0.17525 (+0.00639, +0.00346 to +0.00957) | 0.17537 (+0.00164, +0.00043 to +0.00295) |
| no_flag/window2 | 0.17270 (+0.00894, +0.00515 to +0.01328) | 0.17516 (+0.00185, +0.00010 to +0.00360) |
| flag/expanding | 0.17058 (+0.01106, +0.00701 to +0.01523) | 0.17714 (-0.00013, -0.00074 to +0.00046) |
| fixed baseline (trained once) | 0.17654 (+0.00509, +0.00330 to +0.00726) | 0.17785 (-0.00084, -0.00148 to -0.00021) |

**H1 (fixed in the code before the real run): not reliably confirmed.** On the fresh matches no_flag/window2 beats
no_flag/expanding by +0.00185 Brier (95% CI +0.00010 to +0.00360). The rule as coded printed "SUPPORTED",
but that is a knife-edge: across five bootstrap seeds the interval's lower bound ranges from -0.00004 to
+0.00014 at 1,000 draws, and sat at zero (-0.00000 to +0.00001) at 10,000 (that last check was made before
the parser fixes and not repeated). The verdict flips with the seed, so I do not count it as confirmation. This was a gap in my protocol: I had fixed the comparison but
not the number of bootstrap draws or seeds. The code now also reports whether the lower bound stays above
zero for all five seeds (`robust`), which says no here. The effect is about 1% of the Brier score.

**Post-hoc detail (fresh matches, window2 minus expanding; exploratory):**

- By competition (recomputed after the parser fixes; the point estimates are unchanged): PSL +0.0099 (105 matches, CI +0.0032 to +0.0164) is the only one clearly above zero.
  CPL +0.0046, BBL +0.0041 and ILT20 +0.0040 have intervals that include zero. BPL +0.0005. The T20 Blast,
  the largest fresh league (228 matches), shows nothing (-0.0002). SA20 -0.0034, MLC -0.0029 and LPL -0.0067
  are negative, and those three have the shortest histories.
- By year: 2024 -0.0001 (280 matches), 2025 +0.0024 (368), 2026 +0.0035 (205); none is individually
  distinguishable from zero. As in the IPL, any gain is recent.
- Without the T20 Blast the gain is +0.0026 (625 matches, CI +0.0002 to +0.0047). This is post hoc and
  should not be read as a finding.

**Reading it:**

- The direction replicates in competitions that were never examined: a shorter training memory is slightly
  better, in order of window length (5 years +0.0011, 3 years +0.0016, 2 years +0.0019). But it is about a
  fifth of the IPL effect (+0.0089 for the 2-year window), it varies in sign across leagues, and it is not
  robust. Drift handling looks like an IPL-specific, recent effect rather than a general property of T20 data.
- The Impact Player flag helps where the rule exists (IPL, best of all candidates at +0.0111) and does
  nothing on the fresh set, as expected, since only ILT20 and LPL there have such a rule.
- Retraining every year, rather than training once, is the one cheap change that helps in both subsets
  against the shipped baseline (the fixed model is significantly worse on fresh: -0.00084).
- The selection rule's pick, no_flag/expanding, again did worse than the alternatives on the examined set,
  and is significantly worse than the shipped baseline over all test rows (-0.00096, CI -0.00168 to -0.00023).

**What this means for the project.** More effort on drift handling for this baseline will not pay: it is a
small, uneven effect. I made no change to the shipped default. The remaining headroom is in the model
itself, above all team and player strength, since the first innings still has little to go on before a ball
is bowled. A time-varying competition effect remains a sensible part of the Bayesian model, but expect it
to matter mostly for the IPL.

**Limits.** The fresh set shares calendar years with the examined set, so trends common to all T20 cricket
carry over. 853 matches is modest and per-league intervals are wide. The fresh set is now **spent**: it has
been scored and examined, so it cannot serve as a clean holdout again. A new one needs later data (2027) or
other competitions (women's leagues, T20 internationals).

## 2026-11-10: property-based and exploratory testing

Two methods, applied to the finished pipeline: property-based tests (Hypothesis) and charter-based
exploratory sessions against the real data and the command line. The exploratory sessions were informal
and not timed to the usual 45 to 90 minutes; each was a few minutes of scripted probing, and the notes
below are the log.

### Property-based tests (`tests/test_properties.py`, 15 properties)

A generator builds random but internally consistent matches in Cricsheet's shape (5- and 6-ball overs,
extras, retired batters, double wickets, penalty runs, no-result and tied matches). The properties:

- parsed tables add up to the raw JSON (totals, legal balls, wickets) and ball sequence numbers run 1..n;
- the state table is consistent ball to ball (a ball starts where the last one ended), time only runs
  forward, and shuffling the input rows changes nothing;
- the label is the same fact seen from both sides, and absent when there is no winner;
- the parsed scores reproduce the "won by N runs / N wickets" margin the source records;
- over notation round-trips for any ball count;
- the logistic fit is unchanged by rescaling, shifting or sign-flipping a feature, row order, and swapping
  the labels swaps the probabilities (and mean prediction equals the base rate);
- the bootstrap is reproducible and resamples whole matches, the paired gain is antisymmetric, and
  calibration bins partition the data.

**Do they test anything?** I planted bugs and checked that a property fails. The first nine plants: seven
were caught. The other two (a no-op `np.minimum` in the calibration bins and a reordering that does not
change the bootstrap distribution) turned out to be equivalent code, not missed bugs. A second set of six
plants, each reintroducing a defect found below, was caught by six of six. One weakness: the
"whole matches" bootstrap property only checks counts and bounds, so a subtle row-level resample could
slip past it.

### Exploratory sessions and findings

| Charter | Resources | What it found |
|---|---|---|
| Explore the parser's assumptions with all 5,707 real raw files | wicket kinds, wickets per delivery, extras, innings keys | double-wicket deliveries, an unhandled wicket kind, an unhandled innings key |
| Explore the parsed tables with Cricsheet's own recorded margins as the oracle | win by N wickets (2,813 matches), win by N runs | 2 wicket mismatches and 5 run mismatches, explained below |
| Explore the command line with bad arguments and boundary dates | unknown and repeated sources, bad dates, empty splits, zero options | six issues (F6) |
| Explore the build with corrupt and odd raw files | truncated, empty and binary files, missing fields, stray names | three issues (F4, F5) |
| Explore archive extraction with hostile members | case collisions, device names, many small members | two hardening gaps (F7) |

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | Innings-level `penalty_runs` (7 innings) were ignored. Totals were short by the penalty; in the 2 chases awarded penalty runs *before* the innings, runs needed was overstated for the whole chase. This is also why 4 matches failed the "target = first innings + 1" check, which I had guessed was rain. | data, small | fixed |
| F2 | Only the first wicket on a delivery was read. One match (BPL 1208093) has a run out and a "retired out" on the same ball, so its chase showed 7 wickets lost instead of 8. | data, small | fixed |
| F3 | "retired not out" counted as a dismissal (it means the batter may return), in 2 innings (T20 Blast 1250237, CPL 1369575). | data, small | fixed |
| F4 | A truncated, empty or corrupt JSON file, or invalid over notation, was counted as "not limited-overs" and the build reported `ok`. A missing field aborted the whole build without naming the file. | robustness, medium | fixed: files named, `ok` false |
| F5 | One stray non-numeric `.json` name crashed the build. | robustness, low | fixed: ignored and listed |
| F6 | An unknown source gave "train_end leaves an empty split" (blaming the date); expected errors printed raw tracebacks; `walkforward --first-test-year 1900` died with "cannot convert float NaN to integer"; `--sources ipl ipl` made a folder `ipl+ipl`; `walkforward --sources hnd` exited 0 with an empty analysis; `bayes --chains 0` crashed inside PyMC and `--draws 0` started a fit. | usability, low | fixed |
| F7 | Member names differing only in case overwrote each other (and the count overstated), and the total uncompressed size of an archive was uncapped. | hardening, low | fixed |

Not reproduced: Windows device names such as `NUL.json` extracted as ordinary files here. Rejected
correctly all along: `--sources ../x`, `download ../evil`, a Test match, an empty innings list.

### Effect on the real data

The rebuild changed 10 innings in 10 matches (369 state rows in 5 matches), all in the T20 Blast, BBL, BPL
and CPL; none is in the IPL or The Hundred. Every cross-check now reads zero: wins by wickets (2,813
matches) and by runs agree with the recorded margin, no innings has more than 10 wickets, and the
"target = first innings + 1" warning is gone. The IPL-and-Hundred reports re-ran identically (maximum
difference 0 across 2,352 values). The all-league rolling report moved by at most 0.00016, no conclusion
changed, and it was regenerated and recommitted. The margin check is now permanent in the quality report
(`win_by_wickets_mismatch`, `win_by_runs_mismatch`, `more_than_ten_wickets`).

### Not covered

Plots, the Bayesian model on real data (run later; see the next section), network failures in the downloader, and
anything needing more than a few hundred generated examples per property (each ran 60 to 100 examples
under three fixed seeds). The oracle only checks matches decided by a margin and not rain-affected, so a
tie, no-result or D/L match with bad scoring would not show up there.

## 2026-11-11: first run of the player model on real IPL data

Command: `thinedge bayes --sources ipl --draws 200 --tune 200 --chains 2` (seed 1). Training is IPL
before 2024 (995 matches, 230,309 balls); the test is 2024 on (214 matches, 48,515 balls). Players with
at least 60 balls are modelled individually: 342 batters and 355 bowlers. Everyone else shares one
"other" rating, which covers 23% of test balls for the batter and 21% for the bowler. 72 balls with an
odd run value were dropped. Each fit took about two hours; the three took 6.7 hours together.

### Convergence: every fit failed the gate, so the run is not interpretable

The gate was set in `bayes.py` before the run: R-hat at most 1.01 and bulk ESS at least 400 for the
global parameters, R-hat at most 1.02 for all parameters, and divergences at most 0.1% of draws.

| Model | Max R-hat (global / all) | Min bulk ESS | Divergences |
|---|---|---|---|
| M0 (no players) | 1.017 / 1.017 | 64 | 0 |
| M_bat (batters) | 1.040 / 1.065 | 72 | 0 |
| M_full (batters and bowlers) | 1.034 / 1.090 | 71 | 0 |

There were no divergences, which points to a run that was too short rather than a broken model.
**The run could hardly have passed:** 2 chains of 200 kept draws give 400 draws, and an ESS of 400
would need almost uncorrelated samples. The settings were chosen to keep a laptop run under a day, and
that choice was not checked against the gate beforehand. A run that can pass needs roughly 4 chains of
1,000 draws, which is about ten times the work here and more than a day on this laptop. It has not been
done.

### What the numbers show, read informally

Mean held-out log score per ball (higher is better):

| Model | Total | Wicket part | Runs part |
|---|---|---|---|
| M0 | **-1.5425** | -0.2037 | **-1.4130** |
| M_bat | -1.5602 | -0.2038 | -1.4317 |
| M_full | -1.5558 | -0.2037 | -1.4271 |

The match-level bootstrap gives:
- M_full minus M0: -0.0133 (95% CI -0.0167 to -0.0101).
- M_bat minus M0: -0.0178 (CI -0.0213 to -0.0143).
- M_full minus M_bat: +0.0045 (CI +0.0033 to +0.0056), robust across all 5 seeds.

So, if these fits are taken at face value:
- **Player ratings made predictions worse on 2024+ matches.** All of the loss is in predicting runs; the
  wicket part does not change.
- Adding bowlers to batters recovers about a quarter of what the batter effects lost.

### Possible reasons (not tested)

- **Sampling.** The fits did not converge, and the player effects are where R-hat is worst.
- **Drift.** IPL scoring rose sharply from 2023 (the Impact Player rule). Player ratings learned from
  2008-2023 career data may describe players as they were, not as they bat in 2024. The baseline showed
  the same pattern: a model trained on the old regime is miscalibrated on the new one.
- **The pooled "other" rating.** It stands in for players with fewer than 60 balls, and those players
  may differ systematically from the modelled ones.

### What would test this

- A longer run that passes the gate.
- Splitting the test score by whether the batter is modelled or "other".
- Training only on 2023 onwards, or giving player effects a time trend.

None of this has been done. Until then the honest summary is **"inconclusive, and not encouraging"**.
Team and player strength remains the likely lever for the first innings, but this run gives no evidence
that this model delivers it.

## AI workflow log

Keep this honest. Record where an AI agent helped and where it was wrong.

- 2026-11-10: Claude Code scaffolded the package, parser, features, validation and tests.
  The tests were written against a hand-computed 12-delivery match; one test failed because of a
  test-code mistake (missing helper), not a pipeline bug. The `is_reduced` flaw above was found
  by reading the pipeline's own data-quality warning against the real data, not by the tests.
- 2026-11-10 (baseline): three tests failed on the first run, all because of my test design, not the
  solver: a badly conditioned fixture (2.3% positive outcomes, one feature almost collinear with the
  intercept), a synthetic generator whose results did not depend on the state (so "beats the
  reference" had nothing to find), and a float comparison on a deliberately constant difference. The
  second also showed that an earlier monotonicity test had passed by luck on that signal-free data.
  I fixed the fixtures instead of loosening the assertions, and replaced one arbitrary shrinkage
  threshold with the property that actually holds (coefficients fall monotonically as the penalty grows).
- The sklearn block was a real environment constraint, handled by writing and testing the solver
  rather than by bypassing the policy.
- 2026-11-10 (walk-forward): I got the explanation wrong twice. First, I called stale par the likely
  cause of the first-innings bias; the pre-set test did not support it as the main cause. Second, while
  decomposing the leftover gap I suggested the first-ball offset might be a model-form flaw; checking
  the training-period predictions showed it was not, and a further check traced it to the regime flag.
  Separately, my first version of the pass/fail rule would have rejected a true cause (found by
  running it on simulated drift) and was redesigned before touching real results. The tests include a
  leakage check (flipping the predicted year's labels must not move any arm's predictions) and a
  no-drift control.
- 2026-11-10 (rolling origin): I found a latent fallback flaw in the par curve before building on it and
  checked that fixing it changed nothing already reported. The harness was validated on simulated drift
  and no-drift data first, and the selection rule was written down before the real run. The real run
  then selected a candidate that did badly on the final exam; I reported that and did not re-select. I
  also had to revise my own earlier advice: "the flag is harmful" was true for a flag fitted from one
  season, not for the flag in general.
- 2026-11-10 (more leagues): two verification mistakes of mine, both caught. (1) My first check that the
  earlier results were unchanged compared the old reports with a copy of themselves: the script read
  `reports/ipl+hnd/` while the new runs wrote to `reports/hnd+ipl/`. I redid it against freshly written
  files (identical, max difference 0) and corrected the record. (2) The first all-league run printed
  `nan` and a crash, and the cause was a model that let NaN pass silently, not only a data gap. The
  data fix alone would have left that defect in place; both were fixed and tested.
- 2026-11-10 (more leagues, result): the hypothesis fixed before the run printed "SUPPORTED" with a lower bound of
  +0.00009 (+0.00010 after the parser fixes). Before accepting it I re-ran the bootstrap with other seeds and more draws, found the verdict
  flips with the seed, and downgraded it to "not reliably confirmed". The protocol gap (no fixed seeds or
  draws for the one confirmatory comparison) was mine and is now closed in the code. I also noted that the
  largest fresh league shows no effect and three small ones go the wrong way, so the average hides mixed
  results.
- 2026-11-10 (testing round): the pipeline I had called verified had three data bugs, none caught by my
  own example tests or by the fresh-clone reproduction, because all of those checked that the code
  reproduces itself. What found them was an independent oracle (the margins Cricsheet records). Two of my
  earlier statements were wrong: I wrote that the four "target is not first innings + 1" matches "look
  like rain revisions or penalty runs" without checking, and it was penalty runs. Process slips this
  round: my first mutation script crashed on a path error and left one source file modified (only line
  endings, caught by `git status` and restored with git); my first run of the new regression tests hung
  because the unfixed command line accepted `--draws 0` and began a real model fit, and I stopped it
  with a process-name match that also ended three shell wrapper processes (no data or code was lost).
  I added Hypothesis to the `dev` extra because the testing skill I was asked to use needs it; it is
  easy to drop if you would rather not carry it.
- 2026-11-11 (player model): I suggested `--draws 200 --tune 200 --chains 2` to keep the laptop run
  short, without checking it against the model's own gate (bulk ESS at least 400). With only 400 draws
  in total, the run could hardly have passed. The run still gave a usable informal reading, but 6.7 hours
  went into a result that cannot be counted under the rule we set in advance.
