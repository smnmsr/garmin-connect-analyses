"""Two checks on odd correlations, road rides only.

A: does hr_power_drift still relate to left % once power is controlled for?
B: does the 60 s SD of left % still relate to time and work once window power is controlled for?
C: could power-meter drift over long rides produce the time and work effects? (C1 zero offset, printed;
   C2 mean left % and C3 60 s SD against time, long rides only.)
Writes output/balance_checks.csv.
"""
import csv
from pathlib import Path

import numpy as np

from garmin_connect_analyses.balance import trailing_mean
from garmin_connect_analyses.stats import analysis_mask, fe_ols, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
COLUMNS = [
    "activity_id", "indoor", "elapsed_s", "ride_fraction", "left_pct", "left_sd_60s", "power", "power_60s",
    "cadence", "heart_rate_30s", "cum_work_kJ", "hr_power_drift",
]
LONG_RIDE_S = 3 * 3600  # meter drift check: rides of at least 3 h (57 of 87 road rides; median road ride 4.2 h)
UNITS = {
    "hr_power_drift": "%", "power": "W", "power_60s": "W", "power_300s": "W", "usable_60s": "W",
    "cadence": "rpm", "heart_rate_30s": "bpm", "elapsed_s": "s", "cum_work_kJ": "kJ",
}


def per_ride_trailing(t, window, values):
    """Trailing mean within each ride, with balance.py's time-based rules. NaN where empty."""
    out = np.full(len(values), np.nan)
    _, inv = np.unique(t["activity_id"], return_inverse=True)
    for g in range(inv.max() + 1):
        idx = np.flatnonzero(inv == g)
        idx = idx[np.argsort(t["elapsed_s"][idx], kind="stable")]
        means = trailing_mean(t["elapsed_s"][idx].tolist(), [None if np.isnan(v) else v for v in values[idx]], window)
        out[idx] = [np.nan if m is None else m for m in means]
    return out


def complete(t, columns):
    return np.isfinite(np.column_stack([t[c] for c in columns])).all(axis=1)


def zero_offset(t, long_ride):
    """Power while cadence is 0 (not pedalling), by third of long rides. A drifting zero shows as power > 0 later."""
    stopped = long_ride & (t["cadence"] == 0) & np.isfinite(t["power"]) & np.isfinite(t["ride_fraction"])
    third = np.minimum(np.floor(t["ride_fraction"] * 3), 2)
    for k in range(3):
        p = t["power"][stopped & (third == k)]
        print(f"C1 zero offset, third {k + 1}: records {p.size}, power > 0 in {np.mean(p > 0):.2%}, "
              f"median {np.median(p):.0f} W")


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", COLUMNS)
    usable = np.where(np.isfinite(t["left_pct"]) & (t["power"] > 0), t["power"], np.nan)  # same rule as balance.py
    t["power_300s"] = per_ride_trailing(t, 300, t["power"])  # the denominator of hr_power_drift, zeros included
    t["usable_60s"] = per_ride_trailing(t, 60, usable)  # the mean power of the samples behind left_sd_60s

    _, inv = np.unique(t["activity_id"], return_inverse=True)
    span = np.zeros(inv.max() + 1)
    np.maximum.at(span, inv, t["elapsed_s"])
    long_ride = span[inv] >= LONG_RIDE_S
    road = analysis_mask(t)
    print(f"long road rides (at least {LONG_RIDE_S // 60} min): {np.unique(t['activity_id'][long_ride & (t['indoor'] == 0)]).size}")
    zero_offset(t, long_ride & (t["indoor"] == 0))

    a_rows = road & complete(t, ["left_pct", "hr_power_drift", "power_300s", "power_60s"])
    b_rows = road & complete(
        t, ["left_sd_60s", "power_60s", "usable_60s", "heart_rate_30s", "cadence", "elapsed_s", "cum_work_kJ"])
    c_mean = road & long_ride & complete(t, ["left_pct", "power", "cadence", "heart_rate_30s", "elapsed_s"])
    c_sd = road & long_ride & complete(t, ["left_sd_60s", "usable_60s", "cadence", "heart_rate_30s", "elapsed_s"])
    jobs = [  # (check, outcome column, rows, predictors; several predictors = one joint model)
        ("A1", "left_pct", a_rows, ["hr_power_drift"]),
        ("A2", "left_pct", a_rows, ["hr_power_drift", "power_300s"]),
        ("A3", "left_pct", a_rows, ["hr_power_drift", "power_60s"]),
        ("A4", "left_pct", a_rows, ["power_300s"]),
        ("A5", "left_pct", a_rows, ["power_60s"]),
        ("B1", "left_sd_60s", b_rows, ["power_60s"]),
        ("B2", "left_sd_60s", b_rows, ["usable_60s"]),
        ("B3", "left_sd_60s", b_rows, ["power_60s", "cadence", "heart_rate_30s", "elapsed_s"]),
        ("B4", "left_sd_60s", b_rows, ["usable_60s", "cadence", "heart_rate_30s", "elapsed_s"]),
        ("B5", "left_sd_60s", b_rows, ["power_60s", "cadence", "heart_rate_30s", "cum_work_kJ"]),
        ("B6", "left_sd_60s", b_rows, ["usable_60s", "cadence", "heart_rate_30s", "cum_work_kJ"]),
        ("C2", "left_pct", c_mean, ["power", "cadence", "heart_rate_30s", "elapsed_s"]),
        ("C3", "left_sd_60s", c_sd, ["usable_60s", "cadence", "heart_rate_30s", "elapsed_s"]),
    ]

    out = []
    for check, outcome, rows, names in jobs:
        for r in fe_ols(t[outcome][rows], np.column_stack([t[c][rows] for c in names]), t["activity_id"][rows], names):
            out.append({
                "check": check, "outcome": outcome, **r, "feature_unit": UNITS[r["feature"]],
                "rows": int(rows.sum()), "rides": np.unique(t["activity_id"][rows]).size,
            })

    with open(OUTPUT / "balance_checks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for r in out:
        print(f"{r['check']} {r['outcome']:12s} {r['feature']:15s} {r['coef_per_sd']:+.4f} pp per "
              f"{r['sd_within']:.2f} {r['feature_unit']} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p {r['p']:.3g}  rows {r['rows']} rides {r['rides']}")


if __name__ == "__main__":
    main()
