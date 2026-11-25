import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from garmin_connect_analyses.balance import FEATURE_COLUMNS, feature_rows, ride_samples, ride_status
from garmin_connect_analyses.garmin import GarminDownloader, decode_zip

OUTPUT = Path(__file__).resolve().parents[1] / "output"
INT_COLUMNS = {"activity_id", "year", "indoor", "left_pct", "power", "cadence", "heart_rate"}
SCHEMA = pa.schema([(c, pa.int64() if c in INT_COLUMNS else pa.float64()) for c in FEATURE_COLUMNS])


def main():
    OUTPUT.mkdir(exist_ok=True)
    rides_kept = rows_written = 0
    with GarminDownloader() as garmin, pq.ParquetWriter(OUTPUT / "balance_features.parquet", SCHEMA) as writer:
        rides = [a for a in garmin.list_activities() if a.get("avgPower") is not None]
        for i, ride in enumerate(rides, 1):
            print(f"{i}/{len(rides)} {ride['activityId']}", file=sys.stderr)
            messages = decode_zip(garmin.download_activity(ride["activityId"]))
            status, _ = ride_status(ride_samples(messages), ride.get("avgLeftBalance"))
            if status != "ok":  # same validated rides as balance_samples.py
                continue
            rows = feature_rows(messages, ride)
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            rides_kept += 1
            rows_written += len(rows)

    print(f"rides: {rides_kept}")
    print(f"rows:  {rows_written}")


if __name__ == "__main__":
    main()
