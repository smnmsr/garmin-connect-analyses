"""Shape of left % against power, heart rate and cadence: within-ride effect of each bin vs one reference bin.

Road rides, known side, power > 0. Writes output/balance_shape.csv.
"""
import csv
from pathlib import Path

import numpy as np

from garmin_connect_analyses.stats import analysis_mask, bin_effects, holm, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
BINS = {  # feature: (bin edges, index of the reference bin); mid-range bins, so other bins read as above or below typical
    "power_60s": ([0, 50, 100, 150, 200, 250, 300, 1000], 3),
    "power": ([0, 50, 100, 150, 200, 250, 300, 400, 2000], 3),
    "heart_rate_30s": ([0, 110, 120, 130, 140, 150, 160, 170, 250], 4),
    "cadence": ([0, 60, 70, 80, 90, 100, 110, 200], 4),
}


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", ["activity_id", "indoor", "left_pct", "power", *BINS])
    base = analysis_mask(t)
    rows = []
    for feature, (edges, ref) in BINS.items():
        m = base & np.isfinite(t[feature])
        effects = bin_effects(t["left_pct"][m], t[feature][m], t["activity_id"][m], edges, ref)
        reference = f"{edges[ref]:g}-{edges[ref + 1]:g}"
        for e, p_holm in zip(effects, holm([e["p"] for e in effects])):
            rows.append({"feature": feature, "reference": reference, **e, "p_holm": p_holm})

    with open(OUTPUT / "balance_shape.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['feature']:15s} {r['bin']:>10s} vs {r['reference']:>8s}  {r['coef']:+.3f} pp "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  p_holm {r['p_holm']:.3g}  n {r['rows']} rides {r['rides']}")


if __name__ == "__main__":
    main()
