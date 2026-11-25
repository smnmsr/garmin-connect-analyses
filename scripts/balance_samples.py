import collections
import csv
import sys
from pathlib import Path

from garmin_connect_analyses.balance import ride_samples, ride_status
from garmin_connect_analyses.garmin import GarminDownloader, decode_zip

OUTPUT = Path(__file__).resolve().parents[1] / "output"
RIDE_FIELDS = ["activity_id", "type", "start", "summary_left", "mean_left", "samples", "status"]
SAMPLE_FIELDS = ["activity_id", "elapsed_s", "left_pct", "power", "heart_rate", "cadence"]


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with GarminDownloader() as garmin:
        activities = garmin.list_activities()
        rides = [a for a in activities if a.get("avgPower") is not None]
        ride_out, sample_out = [], []
        for i, ride in enumerate(rides, 1):
            aid = ride["activityId"]
            print(f"{i}/{len(rides)} {aid}", file=sys.stderr)
            samples = ride_samples(decode_zip(garmin.download_activity(aid)))
            status, mean = ride_status(samples, ride.get("avgLeftBalance"))
            ride_out.append({
                "activity_id": aid,
                "type": ride["activityType"]["typeKey"],
                "start": ride["startTimeLocal"],
                "summary_left": ride.get("avgLeftBalance"),
                "mean_left": round(mean, 2) if mean is not None else "",
                "samples": len(samples),
                "status": status,
            })
            if status == "ok":
                sample_out += [{"activity_id": aid, **s} for s in samples]

    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "balance_rides.csv", RIDE_FIELDS, ride_out)
    write_csv(OUTPUT / "balance_samples.csv", SAMPLE_FIELDS, sample_out)

    print(f"power rides:    {len(rides)}")
    for status, n in collections.Counter(r["status"] for r in ride_out).items():
        print(f"{n:4d}  {status}")
    print(f"samples kept:   {len(sample_out)}")


if __name__ == "__main__":
    main()
