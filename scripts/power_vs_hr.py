import sys
from pathlib import Path

import matplotlib.pyplot as plt

from garmin_connect_analyses.garmin import fit_messages

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def main():
    activity_id, messages = fit_messages()
    print(f"activity {activity_id}", file=sys.stderr)

    records = messages.get("record_mesgs", [])
    assert records, "no record messages"
    points = [
        (r["heart_rate"], r["power"])
        for r in records
        if r.get("heart_rate") is not None and r.get("power") is not None
    ]
    heart_rate, power = zip(*points)
    plt.scatter(heart_rate, power, s=4, alpha=0.3)
    plt.xlabel("Heart rate (bpm)")
    plt.ylabel("Power (W)")
    plt.title(f"Activity {activity_id}: power vs heart rate")

    OUTPUT.mkdir(exist_ok=True)
    plt.savefig(OUTPUT / f"power_vs_hr_{activity_id}.png", dpi=150)


if __name__ == "__main__":
    main()
