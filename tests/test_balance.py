from garmin_connect_analyses.balance import check_rides, ride_rows, summarise

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
