"""Between-ride check: does a road ride's mean left % track its date or its session totals? One row per ride.

Ride mean left % is the mean over analysis rows (road, known side, power > 0). Session totals come from the FIT
session message. Spearman rho across rides, Holm across covariates. Date is also tested with ride mean power held
fixed (OLS). Writes output/balance_rides_level.csv.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

from garmin_connect_analyses.garmin import DATA, decode_zip
from garmin_connect_analyses.stats import analysis_mask, holm, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
COVARIATES = [
    "avg_power", "normalized_power", "intensity_factor", "training_stress_score", "avg_heart_rate", "avg_cadence",
    "total_ascent", "avg_temperature", "total_elapsed_time", "total_work", "workout_rpe", "workout_feel",
]
MIN_RIDES = 30  # a covariate with fewer rides is not reported
YEAR_ZERO = datetime(2024, 1, 1)


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", ["activity_id", "indoor", "left_pct", "power"])
    keep = analysis_mask(t)
    rides = np.unique(t["activity_id"][t["indoor"] == 0])
    ride_left = np.array([t["left_pct"][keep & (t["activity_id"] == a)].mean() for a in rides])
    ride_power = np.array([t["power"][keep & (t["activity_id"] == a)].mean() for a in rides])
    start = {int(r["activity_id"]): datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S")
             for r in csv.DictReader(open(OUTPUT / "balance_rides.csv"))}
    years = np.array([(start[int(a)] - YEAR_ZERO).days / 365.25 for a in rides])

    sessions = {a: decode_zip(DATA / "activities" / f"{a}.zip")["session_mesgs"][0] for a in rides}
    columns = {"date (years)": list(years), **{c: [sessions[a].get(c) for a in rides] for c in COVARIATES}}
    rows = []
    for name, col in columns.items():
        known = np.array([v is not None for v in col])
        if known.sum() < MIN_RIDES:
            print(f"{name}: only {known.sum()} rides, not reported")
            continue
        x = np.array(col, dtype=object)[known].astype(float)
        test = stats.spearmanr(x, ride_left[known])
        rows.append({"covariate": name, "rides": int(known.sum()), "rho": test.statistic, "p": test.pvalue})
    for r, p_holm in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p_holm

    with open(OUTPUT / "balance_rides_level.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit = sm.OLS(ride_left, sm.add_constant(np.column_stack([years, ride_power]))).fit()
    print(f"road rides: {rides.size}, ride mean left % SD across rides: {ride_left.std():.2f} pp")
    for r in rows:
        print(f"{r['covariate']:22s} rides {r['rides']:3d}  rho {r['rho']:+.3f}  p {r['p']:.3g}  p_holm {r['p_holm']:.3g}")
    print(f"date, ride mean power held fixed: {fit.params[1]:+.3f} pp per year, p {fit.pvalues[1]:.3g}")


if __name__ == "__main__":
    main()
