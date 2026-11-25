"""Partial within-ride associations: each metric against left %, with effort held fixed.

Effort controls: 60 s power, 30 s heart rate, cadence (within-ride FE, same as balance_model.py).
Road rides, known side, power > 0. A metric is run only on rows where it and all controls are known.
Writes output/balance_partial.csv.
"""
import csv
from pathlib import Path

import numpy as np

from garmin_connect_analyses.stats import analysis_mask, fe_ols, holm, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
CONTROLS = ["power_60s", "heart_rate_30s", "cadence"]
METRICS = [
    "cum_work_kJ", "elapsed_s", "since_break_s", "hr_power_drift", "gradient_15s", "speed", "crank_torque",
    "temperature", "respiration_rate", "left_torque_eff", "right_torque_eff", "left_pedal_smooth",
    "right_pedal_smooth", "left_pco", "right_pco", "left_phase_deg", "right_phase_deg",
]


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", ["activity_id", "indoor", "left_pct", "power", *CONTROLS, *METRICS])
    base = analysis_mask(t)
    rows = []
    for metric in METRICS:
        m = base & np.isfinite(t[metric]) & np.isfinite(np.column_stack([t[c] for c in CONTROLS])).all(axis=1)
        names = [metric, *CONTROLS]
        X = np.column_stack([t[c][m] for c in names])
        result = fe_ols(t["left_pct"][m], X, t["activity_id"][m], names)
        r = result[0]  # the metric; the controls are not reported
        rows.append({
            "metric": metric, "coef_per_sd": r["coef_per_sd"], "sd_within": r["sd_within"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "p": r["p"], "rows": int(m.sum()), "rides": np.unique(t["activity_id"][m]).size,
        })

    for r, p_holm in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p_holm
    with open(OUTPUT / "balance_partial.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['metric']:19s} {r['coef_per_sd']:+.3f} pp per {r['sd_within']:9.2f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  p_holm {r['p_holm']:.3g}  rides {r['rides']}")


if __name__ == "__main__":
    main()
