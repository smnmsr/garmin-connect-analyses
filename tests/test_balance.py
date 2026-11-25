from datetime import datetime, timedelta

from garmin_connect_analyses.balance import (
    check_rides,
    left_pct,
    ride_rows,
    ride_samples,
    ride_status,
    summarise,
)

GOOD = {"record_mesgs": [{"power": 200, "left_right_balance": 181}, {"power": None}, {"left_right_balance": "right"}]}


def ride(aid):
    return {"activityId": aid, "activityType": {"typeKey": "road_biking"}, "startTimeLocal": "2026-01-01 10:00:00"}


def test_summarise_counts_power_and_balance_samples():
    assert summarise(GOOD) == {"records": 3, "power_samples": 1, "balance_samples": 2}


def test_summarise_handles_no_records():
    assert summarise({}) == {"records": 0, "power_samples": 0, "balance_samples": 0}


def test_bad_file_is_recorded_and_run_continues():
    def fetch(aid):
        if aid == 2:
            raise ValueError("not a zip")
        return GOOD

    rows = ride_rows([ride(1), ride(2), ride(4)])
    check_rides(fetch, rows)
    assert [r["status"] for r in rows] == ["ok", "failed", "ok"]
    assert rows[1]["error"] == "ValueError: not a zip"
    assert rows[0]["power_samples"] == 1


def test_other_error_is_recorded_and_stops_run():
    def fetch(aid):
        if aid == 3:
            raise ConnectionError("gone")
        return GOOD

    rows = ride_rows([ride(1), ride(3), ride(4)])
    check_rides(fetch, rows)
    assert [r["status"] for r in rows] == ["ok", "failed", "not attempted"]


def test_left_pct_reads_flagged_values_and_rejects_unclear_ones():
    assert left_pct(181) == 47  # right flag set, 53% right
    assert left_pct(129) is None  # 99% left: above the data-quality band
    assert left_pct(49) is None  # no right flag: side unknown
    assert left_pct(0x80 | 101) is None  # over 100%
    assert left_pct(0x80 | 100) is None  # 0% left: below the data-quality band
    assert left_pct(0x80 | 95) is None  # 5% left: band edge, excluded
    assert left_pct(0x80 | 94) == 6  # 6% left: just inside the band
    assert left_pct("right") is None  # raw 0x80


def test_ride_samples_keeps_only_usable_records():
    t0 = datetime(2026, 1, 1, 10)
    records = [
        {"timestamp": t0, "power": 200, "left_right_balance": 181, "heart_rate": 150},
        {"timestamp": t0 + timedelta(seconds=1), "power": 0, "left_right_balance": 181},
        {"timestamp": t0 + timedelta(seconds=2), "power": 210, "left_right_balance": 49},
        {"timestamp": t0 + timedelta(seconds=3), "power": 220, "left_right_balance": "right"},
        {"timestamp": t0 + timedelta(seconds=4), "power": 230},
    ]
    assert ride_samples({"record_mesgs": records}) == [
        {"elapsed_s": 0.0, "left_pct": 47, "power": 200, "heart_rate": 150, "cadence": None}
    ]


def test_ride_status_validates_against_garmin():
    samples = [{"left_pct": 47}, {"left_pct": 47}]
    assert ride_status(samples, 47.2)[0] == "ok"  # 0.2 pp off, within tolerance
    assert ride_status(samples, 48)[0].startswith("excluded")  # 1.0 pp off
    assert ride_status([], 47)[0].startswith("excluded")
    assert ride_status(samples, None)[0].startswith("excluded")
