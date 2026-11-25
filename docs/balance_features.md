# Balance analysis: feature definitions

One row = one FIT record (about 1 s). Rows from all rides are pooled.

## Conventions

- Every feature is read directly from the FIT file or the Garmin activity index, or is simple arithmetic on those values (unit conversion, mean, time window, ratio). No inference, no modelling, no guessing.
- Missing stays missing. Never filled with 0, a neighbouring value or an estimate.
- Metric units, or the standard unit for the quantity. Every column states its unit.
- Values are as decoded by `garmin-fit-sdk` (scale and offset already applied).
- Time in seconds (s). Timestamps in UTC.
- Windows are trailing and time-based: they use only samples with a timestamp in the past window, not a sample count.
- The target analysis uses only rows with `power > 0` and a non-empty `left_pct`.

## Target

| Column | Definition | Unit | Source and notes |
|---|---|---|---|
| `left_pct` | Left leg share of power, decoded from `left_right_balance` | % | FIT field. Empty when the side is unclear: no right flag, raw `0x80` (`'right'`), or value over 100%. Also empty when left is 5% or less, or 95% or more (data-quality band, see note below). |
| `left_pct_10s` | Trailing 10 s mean of `left_pct` | % | Empty if fewer than 5 usable samples in the window. |

Note on the data-quality band: samples with left at 5% or less, or 95% or more, are excluded. This is a data-quality decision, not a FIT rule. The 0% spike (about 1,345 samples at exactly 0%) falls inside this band.

## Drivers

| Column | Definition | Unit | Source and notes |
|---|---|---|---|
| `power` | Record `power` | W | FIT field. Zero kept in the table, excluded from the target analysis. |
| `power_60s` | Trailing 60 s mean of `power` | W | Mean. |
| `cadence` | Record `cadence` | rpm | FIT field, integer. `fractional_cadence` not used. |
| `crank_torque` | `power ÷ (cadence × 2π / 60)` | N·m | Arithmetic. Empty when cadence < 30 rpm, to avoid dividing by near-zero angular speed. |
| `heart_rate` | Record `heart_rate` | bpm | FIT field. |
| `heart_rate_30s` | Trailing 30 s mean of `heart_rate` | bpm | Mean. Only heart-rate window used. |

## Time and fatigue

| Column | Definition | Unit | Source and notes |
|---|---|---|---|
| `elapsed_s` | Seconds since the first record of the ride | s | Timestamp difference. |
| `ride_fraction` | `elapsed_s ÷ session total_elapsed_time` | dimensionless, 0–1 | Ratio of FIT values. |
| `cum_work_kJ` | Cumulative work: `accumulated_power ÷ 1000` | kJ | FIT field. The profile labels `accumulated_power` as "watts", but its last value equals the session `total_work` in J, so it is cumulative energy. |
| `since_break_s` | Seconds since the end of the last pause. Counts from ride start if no pause has ended yet. | s | Pauses come directly from the FIT `timer` events: a `stop_all` followed by the next `start`. Only pauses of at least 5 minutes (300 s) count, auto or manual. Shorter pauses are ignored. A `stop_all` with no following `start` (the end of the ride) is not a pause. Only past events are used. |
| `hr_power_drift` | `(5 min trailing mean HR ÷ 5 min trailing mean power) ÷ (same ratio over the first 20 min) − 1` | % | Ratio of means. Empty for rides shorter than 20 min or with zero baseline power. Positive = heart rate rising relative to power. |

## Context

| Column | Definition | Unit | Source and notes |
|---|---|---|---|
| `indoor` | 1 if activity type is `indoor_cycling`, else 0 | boolean (0/1) | Activity type from the Garmin activity index. Always included as a control. |
| `gradient_15s` | Trailing 15 s `Δ altitude ÷ Δ distance` × 100 | % | Arithmetic on FIT `enhanced_altitude` and `distance`. Road rides only. |
| `speed` | Record `enhanced_speed` × 3.6 | km/h | FIT field in m/s, unit conversion only. |
| `year` | Calendar year of the ride start | year (integer) | From the activity index. |

## Mechanism only (not predictors)

These restate the same forces that produce balance. Use them to explain a pattern, not to predict it. Not extracted in the first pass.

| Field | Unit |
|---|---|
| left and right power phase | degrees |
| left and right torque effectiveness | % |
| left and right pedal smoothness | % |
| left and right pedal centre offset (PCO) | mm |

## Ride-level covariates (not per sample)

Deferred. Listed so they are not confused with row-level features.

| Field | Unit |
|---|---|
| normalised power | W |
| intensity factor | dimensionless |
| training stress score | dimensionless |
| time standing | s |
| stand count | count |

## Not used: standing

The FIT `rider_position_change` events (`seated`, `standing`, `transition_to_standing`, `transition_to_seated`) are present for all 87 kept road rides and for 24 of 53 kept indoor rides. They do not add up, so they are not used:

- **Sequence check:** no road ride follows a consistent seated/standing sequence. Transitions are often followed by `seated`, `standing` appears directly after `seated`, and events repeat.
- **Count check:** the number of `transition_to_standing` events equals the session `stand_count` in 4 of 87 road rides.

Rule: a standing source is used only if it adds up. If it does not, it is ignored. The check above was not designed to pass any ride; it reports the consistency of the event sequence as recorded.
