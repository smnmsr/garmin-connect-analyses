# Balance analysis: plan and run order

Question: within a ride, does left/right balance shift with power, cadence, heart rate, time and work?

Rules were set before the first model run. That run was reviewed and the rules were patched once. The patched rules are the ones reported. See "Changes after the first run".

## Sample

- One rider (the author). Road rides only (`indoor` = 0): 87 rides. All on one power meter model (Garmin Rally 200) in 2024–2026. Every result describes this rider, not a population.
- Usable rows: known side (`left_pct` not empty) and power > 0. Then complete cases over all predictors: 743,594 rows.
- Indoor is not in this run. 24 indoor rides use the Rally meter, 29 use a Tacx trainer (2025–2026), so year and device are confounded indoors. Needs a device column first. The 2026 indoor ride is too few to analyse.

## Outcomes

| Name | Role | Definition |
|---|---|---|
| `left_pct` | primary | Left share of power, % |
| `left_pct_10s` | robustness | Trailing 10 s mean of `left_pct` |
| `left_sd_60s` | secondary, variability | Trailing 60 s population SD of `left_pct`, needs 30 usable samples |
| `abs_diff_W` | secondary | Left minus right in W: `power × |2 × left_pct − 100| / 100` |

## Predictors

`power` (1 s, W), `power_60s` (trailing 60 s mean, W), `cadence` (rpm), `heart_rate_30s` (bpm), `elapsed_s` (s), `cum_work_kJ` (kJ).

## Models

- Within-ride fixed effects: each ride's mean removed from outcome and predictors, then OLS. Standard errors clustered by ride, t distribution with (rides − 1) degrees of freedom.
- Effects per 1 SD of within-ride variation of the predictor, in the outcome's unit.
- Three blocks per variant:
  - `joint_time`: power, cadence, heart rate, elapsed time.
  - `joint_work`: power, cadence, heart rate, cumulative work. Time and work never appear together: they are collinear (VIF about 20).
  - `univariate`: each predictor alone.
- Holm correction within each block.
- All blocks in a variant use the same rows (complete cases over all predictors).

## Screen

- Within-ride Spearman rho per ride, for rides with at least 300 usable rows where feature and outcome are known.
- Reported: mean rho across rides, 95% CI from 2000 ride-level bootstrap draws (seed 0), p from a one-sample t-test over rides, Holm across features.
- Pooled Spearman not run: it mixes between-ride differences into the estimate.

## Variants

- `primary`: road rides, `left_pct`, 1 s power.
- `power_60s`: same as primary, with 60 s trailing power in place of 1 s power. Tests whether heart rate still matters once power is smoothed the same way.
- `rides_20min_plus`: rides with at least 1200 s of records.
- `left_pct_10s`: 10 s trailing outcome.
- `sd_60s`: variability outcome.
- `abs_diff_W`: absolute difference in W.
- `year_2024`, `year_2025`, `year_2026`: split by ride year (road device is the same in all years).

Each variant is its own Holm family. Variants are not corrected across each other. Read them as robustness checks, not as separate claims.

## Not in this run

- Indoor analysis (needs a device column).
- Outlier-drop variant: flagged rows (gradient, drift) are not model inputs.
- Wild cluster bootstrap. Cluster-robust SEs with 87 rides; check if a result sits near its threshold.
- Family-wide Holm across variants.
- Spline of the relation. Bins only (see "Shape" below).
- Standing time and stand count: not used. Session values exist for 8 rides only, and stand count does not add up (see `docs/balance_features.md`).

## Known limits

- The 30-sample rule is not random. Among analysis rows, 11% of 60 s windows are lost. Lost windows have a median power of 166 W, kept windows 179 W (`scripts/balance_windows.py`).
- `left_sd_60s` may move with power for a mechanical reason: the left share of a small power is a noisier ratio. The same holds for `abs_diff_W`, which multiplies the imbalance by power. Time and work effects on the SD are adjusted for 1 s power only, not for window power, so they may partly reflect the same ratio noise.
- `heart_rate_30s` and `power_60s` both track sustained effort. The models cannot separate them cleanly.
- Balance is device-estimated. Results describe Rally-meter road riding only.
- Within-ride association. Not causation.
- Mechanism fields (torque effectiveness, pedal smoothness, power phase, PCO) come from the same meter as balance. They restate balance; they cannot be independent causes.
- Torque effectiveness and pedal smoothness exist only for 32 road rides, all in 2026. Results for them rest on one year. PCO and power phase exist for all 87 road rides (2024–26).
- Device info in the FIT files shows software version changes in 2024–26 on some devices. Not yet linked to the balance meter. Could explain the between-ride drift (see "Other metrics").

## Changes after the first run

1. Clustered p-values use the t distribution with (rides − 1) df. statsmodels defaults to z, which understates p-values for few clusters.
2. Joint model split into `joint_time` and `joint_work`. Time and work were together before, with VIF about 20.
3. `power_60s` variant added.
4. `hr_power_drift` is empty before the 20 min baseline has ended. Earlier values used baseline data from later in the ride.
5. `ride_fraction` dropped from the screen: within a ride it ranks exactly like `elapsed_s`.
6. Complete cases over all predictors, so every model in a variant uses the same rows. Primary rows went from 744,844 to 743,594 (first 60 s of each ride dropped, where `power_60s` is empty).

## Checks on odd correlations

`scripts/balance_checks.py` writes `output/balance_checks.csv`. Road rides, same effects per 1 SD as the models.

- **A, drift vs power:** `hr_power_drift` alone is +0.26 pp per 23% (p 8e-10). Controlling for 5 min power (the drift's own denominator) flips it to −0.28 pp (p 0.008). Controlling for 60 s power makes it +0.03 (ns). The positive correlation is power, not drift.
- **B, SD vs window power:** two controls for window power, run side by side.
  - `power_60s` includes coasting zeros. The SD uses only usable samples, so this is the wrong denominator. With it, the time and work SD effects vanish (B3, B5).
  - `usable_60s` (mean power of the samples behind the SD) is the matching denominator. It was written into the script before the results were seen. With it, time (+0.18 pp per 106 min, p 1e-4) and work (+0.17 pp per 655 kJ, p 4e-4) survive, and HR stays strong (−0.81 pp per 15 bpm).
  - The power-level link to SD is large (−0.68 pp per 48 W) but shrinks to −0.16 once HR, cadence and time are added.
- **C, meter drift (rides of at least 3 h, 57 road rides):**
  - Zero offset: power above 0 while cadence is 0 in 0.20%, 0.13% and 0.16% of records (thirds of the ride). Median 0 W. No trend.
  - Mean left % against time, controlling for power, cadence and HR: +0.10 pp per 116 min (p 0.31). No mean drift.
  - 60 s SD against time, usable-power control: +0.21 pp per 116 min (p 1e-4). Same effect as in all rides.
  - First pass used a 60 min cut. It kept 84 of 87 road rides, so it did not test long rides, and was replaced by the 3 h cut. Its direction was the same (mean +0.08, p 0.35; SD +0.18, p 1e-4).
  - Verdict: no sign of meter drift in zero or mean. Meter noise that changes over time is not ruled out. Pedal technique under fatigue is not checked.

## Shape (added after the model run)

`scripts/balance_shape.py` writes `output/balance_shape.csv`. Within-ride effect of each bin of power (1 s and 60 s), HR (30 s) and cadence on left %, vs one mid-range reference bin, with ride-clustered CIs. Holm within each feature. Bins are round values chosen by eye, not pre-registered. Descriptive: it checks whether the linear slopes above hide a non-linear pattern.

## Other metrics (added after the model run)

Post-hoc. Read as leads, not pre-registered findings.

- `scripts/balance_screen.py`: within-ride Spearman, same method as above. Now also covers temperature, respiration rate and the mechanism fields. The Holm family grew from 12 to 22 features, so Holm p-values moved. Raw rho did not.
- `scripts/balance_partial.py` → `output/balance_partial.csv`: each metric in a within-ride FE model with 60 s power, 30 s heart rate and cadence held fixed. Effect per 1 SD, Holm across metrics. Mechanism rows use only rides where the field is reported.
- `scripts/balance_rides_level.py` → `output/balance_rides_level.csv`: one row per road ride. Ride mean left % against date and session totals (Spearman, Holm). Date is also tested with ride mean power held fixed (OLS, printed).

## Plots

`scripts/balance_plots.py` writes `output/plots/`:

1. `01_forest_partial_effects.png`: partial effects, effort held fixed.
2. `02_shape_power_hr.png`: shape by bin of 60 s power and HR.
3. `03_heatmap_power_hr.png`: balance across 60 s power × HR, cells with at least 300 rows.
4. `04_spread_by_power.png`: within-ride spread of 1 s balance by power.
5. `05_ride_averages_timeline.png`: ride mean balance by date.
6. `06_example_ride.png`: one road ride, balance, power and HR over time.

## Status of results

- **Balance vs metrics, per sample:** weak. Within-ride |ρ| ≤ 0.11. Strongest links: 60 s power and HR (negative), drift (positive, but see A). Reported as weak, not as a ranking.
- **Low load → left (the working hypothesis):** partly, and not stable.
  - 60 s power 50–150 W: +0.3 to +0.4 pp vs 150–200 W. Below 50 W: −0.2, ns. 1 s power shows the opposite at low values (−0.4 to −0.7 pp). Low-power samples are likely pedal-stroke dead spots and coasting, not a clean signal.
  - High load → right is the robust half: 60 s power 200–250 W −0.65 pp, 250–300 W −1.15, above 300 W −1.5 (vs 150–200 W). Same direction at 1 s, and for HR (above 170 bpm −1.2 pp vs 140–150).
  - Size is small: about 1.5–2 pp across the whole power range, against 6 pp SD per 1 s sample.
- **Low load → noisier, not clearly left:** spread of 1 s left % around the ride mean is much larger at low power (under 50 W: SD 11.7 pp; 250–300 W: 4 pp). Per-sample "left-heavy at low power" is mostly noise. Mechanical ratio noise is not ruled out (see Known limits).
- **Cadence:** not monotone. 80–90 rpm +0.75 pp, 100–110 rpm −1.7, above 110 −2.7 (vs 90–100). The linear cadence slope hides this.
- **Low load → left, by effort cell (heat map):** at low effort (60 s power 75–125 W, HR 100–110) 64% of rides lean left, median +0.6 pp, 72 rides. At moderate effort (125–175 W, HR 140–150) 72% of rides lean left, median +0.4 pp. At high effort (225–250 W, HR 140–145; 275–350 W, HR 160–170) 72–74% of rides lean right, median −0.9 pp. The right-lean at high effort is larger and more consistent.
- **Other metrics beyond effort (partial):**
  - Torque effectiveness (left +0.8, right −0.8 pp per SD) and pedal smoothness (left +0.7, right −0.4): large, but mechanism from the same meter, all 2026. Restates balance.
  - Right power phase span: −0.16 pp per 25°, p_holm 7e-12. Closest to a riding-style metric.
  - Speed: +0.16 pp per 7 km/h, p_holm 0.015. Weak.
  - Nothing after effort control: cumulative work, time into ride, time since break, temperature, crank torque, HR drift, respiration rate (ns after Holm).
- **Between rides, date:** ride mean left % drifts right. Plain OLS slope −0.41 pp per year (p 0.03). Spearman rho −0.24 (p 0.02, Holm 0.32 across 13 covariates). With ride mean power held fixed: −0.50 pp per year (p 0.01). Average ride means: 49.6% in 2024, 49.2% in 2025, 48.8% in 2026. Not explained by average power, normalised power, intensity factor, TSS, ascent, temperature, duration, work, cadence or RPE (all p ≥ 0.12).
- **Drift:** not a finding. Its positive correlation is power.
- **Mean balance vs time within a ride, work:** no link, in all rides and in rides of at least 3 h.
- **Variability (60 s SD), candidates, not findings:**
  - HR: −0.8 pp per 15 bpm, robust to the usable-power control.
  - Time and work: +0.18 pp per 106 min and +0.17 pp per 655 kJ with the usable-power control. The 60 s-power control removes them. Survive the meter check, but meter noise and fatigue-related technique change are not ruled out.

## Run order

```
uv run scripts/balance_features.py   # writes output/balance_features.parquet (reads cache)
uv run scripts/balance_windows.py    # prints the 30-sample check
uv run scripts/balance_screen.py     # writes output/balance_screen.csv
uv run scripts/balance_model.py      # writes output/balance_model.csv
uv run scripts/balance_checks.py     # writes output/balance_checks.csv
uv run scripts/balance_shape.py      # writes output/balance_shape.csv
uv run scripts/balance_partial.py    # writes output/balance_partial.csv
uv run scripts/balance_rides_level.py  # writes output/balance_rides_level.csv, prints date check
uv run scripts/balance_plots.py      # writes output/plots/*.png
```
