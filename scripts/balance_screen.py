"""Within-ride Spearman screening: does each feature track left % inside a ride? Writes output/balance_screen.csv."""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from garmin_connect_analyses.stats import analysis_mask, holm, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
FEATURES = [
    "power", "power_60s", "cadence", "crank_torque", "heart_rate", "heart_rate_30s", "cum_work_kJ",
    "since_break_s", "hr_power_drift", "gradient_15s", "speed", "elapsed_s", "temperature", "respiration_rate",
    # mechanism: restate the forces behind balance, so they explain a pattern rather than predict it
    "left_torque_eff", "right_torque_eff", "left_pedal_smooth", "right_pedal_smooth", "left_pco", "right_pco",
    "left_phase_deg", "right_phase_deg",
]  # ride_fraction left out: within a ride it ranks exactly like elapsed_s
OUTCOMES = ["left_pct", "left_sd_60s"]
MIN_ROWS = 300  # usable rows per ride (about 5 min): fewer and the rank correlation is too noisy
BOOTSTRAP = 2000
RNG = np.random.default_rng(0)  # fixed seed: the CI is reproducible


def ride_rhos(x, y, rides):
    """Spearman rho inside each ride, for rides with at least MIN_ROWS rows where x and y are known."""
    rhos = []
    for idx in rides:
        m = np.isfinite(x[idx]) & np.isfinite(y[idx])
        if m.sum() >= MIN_ROWS:
            rhos.append(stats.spearmanr(x[idx][m], y[idx][m]).statistic)
    return np.array(rhos)


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", ["activity_id", "indoor", "power", *OUTCOMES, *FEATURES])
    keep = analysis_mask(t)
    t = {c: v[keep] for c, v in t.items()}
    _, inv = np.unique(t["activity_id"], return_inverse=True)
    rides = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]

    rows = []
    for outcome in OUTCOMES:
        block = []
        for feature in FEATURES:
            rhos = ride_rhos(t[feature], t[outcome], rides)
            boot = RNG.choice(rhos, size=(BOOTSTRAP, rhos.size)).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            block.append({
                "outcome": outcome, "feature": feature, "rides": rhos.size, "mean_rho": rhos.mean(),
                "ci_low": lo, "ci_high": hi, "p": stats.ttest_1samp(rhos, 0).pvalue,
            })
        for row, p_holm in zip(block, holm([r["p"] for r in block])):
            row["p_holm"] = p_holm
        rows += block

    with open(OUTPUT / "balance_screen.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['outcome']:12s} {r['feature']:15s} rides {r['rides']:3d}  rho {r['mean_rho']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  p_holm {r['p_holm']:.3g}")


if __name__ == "__main__":
    main()
