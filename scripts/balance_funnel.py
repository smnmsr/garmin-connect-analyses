import csv
import sys
from pathlib import Path

from garmin_connect_analyses.balance import check_rides, ride_rows
from garmin_connect_analyses.garmin import GarminDownloader, decode_zip

OUTPUT = Path(__file__).resolve().parents[1] / "output"
FIELDS = ["activity_id", "type", "start", "status", "records", "power_samples", "balance_samples", "error"]


def main():
    with GarminDownloader() as garmin:
        activities = garmin.list_activities()
        rides = [a for a in activities if a.get("avgPower") is not None]
        rows = ride_rows(rides)

        def fetch(aid):
            print(aid, file=sys.stderr)
            return decode_zip(garmin.download_activity(aid))

        try:
            check_rides(fetch, rows)
        finally:  # report is written even when the run stops early
            OUTPUT.mkdir(exist_ok=True)
            with open(OUTPUT / "balance_funnel.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "failed"]
    with_power = [r for r in ok if r["power_samples"] > 0]
    with_balance = [r for r in with_power if r["balance_samples"] > 0]
    print(f"index activities:       {len(activities)}")
    print(f"power in summary:       {len(rides)}")
    print(f"ok:                     {len(ok)}")
    print(f"failed:                 {len(failed)}")
    print(f"not attempted:          {sum(r['status'] == 'not attempted' for r in rows)}")
    print(f"power samples in FIT:   {len(with_power)}")
    print(f"balance samples in FIT: {len(with_balance)}")
    for r in failed:
        print(f"failed {r['activity_id']}: {r['error']}")


if __name__ == "__main__":
    main()
