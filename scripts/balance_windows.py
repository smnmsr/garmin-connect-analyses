"""Diagnostic for the 30-usable-sample rule: which 60 s windows are lost, and do they differ in power? Prints only."""
from pathlib import Path

import numpy as np

from garmin_connect_analyses.balance import SD_MIN_SAMPLES, SD_WINDOW_S
from garmin_connect_analyses.stats import analysis_mask, read_columns

OUTPUT = Path(__file__).resolve().parents[1] / "output"
COLUMNS = ["activity_id", "indoor", "elapsed_s", "left_pct", "left_sd_60s", "power"]


def report(t):
    lost = np.isnan(t["left_sd_60s"])
    print(f"windows: {len(lost)}, lost to fewer than {SD_MIN_SAMPLES} usable samples: {lost.mean():.1%}")

    _, inv = np.unique(t["activity_id"], return_inverse=True)
    share = np.bincount(inv, weights=lost) / np.bincount(inv)
    print(f"rides: {len(share)}, lost share per ride (10/50/90th pct): {np.percentile(share, [10, 50, 90]).round(3)}")
    print(f"rides with more than half of windows lost: {(share > 0.5).sum()}")

    for label, m in (("lost", lost), ("kept", ~lost)):
        power = t["power"][m]
        power = power[np.isfinite(power)]
        print(f"{label}: median power {np.median(power):.0f} W, zero power {np.mean(power == 0):.1%}, "
              f"known side {np.isfinite(t['left_pct'][m]).mean():.1%}")


def main():
    t = read_columns(OUTPUT / "balance_features.parquet", COLUMNS)
    t = {c: v[t["elapsed_s"] >= SD_WINDOW_S] for c, v in t.items()}  # windows exist once the ride has run 60 s

    print("All records, road and indoor (context):")
    print(f"  road {np.isnan(t['left_sd_60s'][t['indoor'] == 0]).mean():.1%}, "
          f"indoor {np.isnan(t['left_sd_60s'][t['indoor'] == 1]).mean():.1%}")
    print("Analysis rows only (road, known side, power > 0):")
    report({c: v[analysis_mask(t)] for c, v in t.items()})


if __name__ == "__main__":
    main()
