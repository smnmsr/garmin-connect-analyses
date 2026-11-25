"""Within-ride fixed-effects models: left % on power, cadence, heart rate, time and work. Writes output/balance_model.csv."""
import csv
from pathlib import Path

import numpy as np

from garmin_connect_analyses.stats import analysis_mask, fe_ols, holm, read_columns, vif, within_ride

OUTPUT = Path(__file__).resolve().parents[1] / "output"
UNITS = {"power": "W", "power_60s": "W", "cadence": "rpm", "heart_rate_30s": "bpm", "elapsed_s": "s", "cum_work_kJ": "kJ"}
COLUMNS = [
    "activity_id", "indoor", "year", "left_pct", "left_pct_10s", "left_sd_60s",
    "power", "power_60s", "cadence", "heart_rate_30s", "elapsed_s", "cum_work_kJ",
]
ALL_PREDICTORS = ["power", "power_60s", "cadence", "heart_rate_30s", "elapsed_s", "cum_work_kJ"]
TIME_SET = ["power", "cadence", "heart_rate_30s", "elapsed_s"]  # joint models keep one of time or work:
WORK_SET = ["power", "cadence", "heart_rate_30s", "cum_work_kJ"]  # they are collinear (VIF about 20)
UNIVARIATE_SET = ["power", "cadence", "heart_rate_30s", "elapsed_s", "cum_work_kJ"]
LONG_RIDE_S = 1200  # robustness: rides of at least 20 min


def swap_power(names, power):
    """The given power column in place of 'power'."""
    return [power if n == "power" else n for n in names]


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", COLUMNS)
    base = analysis_mask(t)
    complete = np.isfinite(np.column_stack([t[c] for c in ALL_PREDICTORS])).all(axis=1)  # same rows for all models
    _, inv = np.unique(t["activity_id"], return_inverse=True)
    span = np.zeros(inv.max() + 1)
    np.maximum.at(span, inv, t["elapsed_s"])
    long_ride = span[inv] >= LONG_RIDE_S
    watts_apart = t["power"] * np.abs(2 * t["left_pct"] - 100) / 100  # left minus right, in W

    variants = {  # name: (outcome, unit, rows, power column)
        "primary": (t["left_pct"], "%-points", base, "power"),
        "rides_20min_plus": (t["left_pct"], "%-points", base & long_ride, "power"),
        "left_pct_10s": (t["left_pct_10s"], "%-points", base, "power"),
        "sd_60s": (t["left_sd_60s"], "%-points", base, "power"),
        "abs_diff_W": (watts_apart, "W", base, "power"),
        **{f"year_{y}": (t["left_pct"], "%-points", base & (t["year"] == y), "power") for y in (2024, 2025, 2026)},
        "power_60s": (t["left_pct"], "%-points", base, "power_60s"),  # smoothed power in place of 1 s power
    }

    rows = []
    for variant, (target, unit, mask, power) in variants.items():
        m = mask & complete & np.isfinite(target)
        y, groups = target[m], t["activity_id"][m]
        X = {c: t[c][m] for c in ALL_PREDICTORS}
        time_set, work_set, uni_set = (swap_power(s, power) for s in (TIME_SET, WORK_SET, UNIVARIATE_SET))
        if variant == "primary":
            _, inv_m = np.unique(groups, return_inverse=True)
            for label, names in (("time", time_set), ("work", work_set)):
                Z = within_ride(np.column_stack([X[c] for c in names]), inv_m)
                print(f"VIF, primary joint_{label}:", dict(zip(names, np.round(vif(Z), 2).tolist())))
        blocks = {
            "joint_time": fe_ols(y, np.column_stack([X[c] for c in time_set]), groups, time_set),
            "joint_work": fe_ols(y, np.column_stack([X[c] for c in work_set]), groups, work_set),
            "univariate": [fe_ols(y, X[c][:, None], groups, [c])[0] for c in uni_set],
        }
        rides = np.unique(groups).size
        for model, block in blocks.items():
            for r, p_holm in zip(block, holm([r["p"] for r in block])):  # Holm within each block
                rows.append({
                    "variant": variant, "model": model, **r, "p_holm": p_holm,
                    "target_unit": unit, "feature_unit": UNITS[r["feature"]],
                    "rows": int(m.sum()), "rides": rides,
                })

    with open(OUTPUT / "balance_model.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['variant']:17s} {r['model']:10s} {r['feature']:14s} {r['coef_per_sd']:+.4f} {r['target_unit']} "
              f"per {r['sd_within']:.1f} {r['feature_unit']} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p_holm {r['p_holm']:.3g}  n {r['rows']} rides {r['rides']}")


if __name__ == "__main__":
    main()
