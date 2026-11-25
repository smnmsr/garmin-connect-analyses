import math
from datetime import datetime, timedelta

import pytest

from garmin_connect_analyses.balance import FEATURE_COLUMNS, feature_rows, pause_ends, trailing_mean, trailing_sd

T0 = datetime(2026, 1, 1, 10)
ROAD = {"activityId": 1, "activityType": {"typeKey": "road_biking"}, "startTimeLocal": "2026-01-01 10:00:00"}
INDOOR = {**ROAD, "activityType": {"typeKey": "indoor_cycling"}}


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def timer(seconds, kind):
    return {"event": "timer", "event_type": kind, "timestamp": at(seconds)}


def make_ride(seconds, events=(), **fields):
    """One record per second in seconds. A field is a value, or a function of the second."""
    fields = {"left_right_balance": 181, **fields}  # 47% left
    records = [
        {"timestamp": at(t), **{k: v(t) if callable(v) else v for k, v in fields.items()}}
        for t in seconds
    ]
    return {"record_mesgs": records, "event_mesgs": list(events), "session_mesgs": [{"total_elapsed_time": seconds[-1]}]}


def test_trailing_mean_needs_full_window_and_enough_values():
    means = trailing_mean([float(t) for t in range(70)], list(range(70)), 60)
    assert means[59] is None  # ride shorter than the window
    assert means[60] == 30.5  # mean of 1..60 in (0, 60]

    values = [1 if 11 <= t <= 15 else None for t in range(20)]
    means = trailing_mean([float(t) for t in range(20)], values, 10, min_count=5)
    assert means[14] is None  # 4 values in (4, 14]
    assert means[15] == 1  # 5 values in (5, 15]


def test_trailing_sd_is_population_sd_and_needs_30_values():
    times = [float(t) for t in range(120)]
    sds = trailing_sd(times, [40 if t % 2 else 60 for t in range(120)], 60, min_count=30)
    assert sds[59] is None  # ride shorter than the window
    assert sds[60] == pytest.approx(10)  # 30 values at 40 and 30 at 60 in (0, 60]

    sparse = trailing_sd(times, [60 if t % 3 == 0 else None for t in range(120)], 60, min_count=30)
    assert sparse[60] is None  # only 20 values in (0, 60]


def test_pause_counts_only_if_five_minutes_and_has_a_restart():
    events = [timer(100, "stop_all"), timer(400, "start"),  # 300 s: counts
              timer(1000, "stop_all"), timer(1299, "start"),  # 299 s: ignored
              timer(2000, "stop_all")]  # no restart: end of ride
    assert pause_ends(events, T0) == [400]


def test_since_break_counts_from_end_of_long_pause_only():
    seconds = list(range(0, 100)) + list(range(400, 600)) + list(range(700, 800))
    events = [timer(100, "stop_all"), timer(400, "start"), timer(600, "stop_all"), timer(700, "start")]
    by_t = {r["elapsed_s"]: r for r in feature_rows(make_ride(seconds, events), ROAD)}
    assert by_t[50]["since_break_s"] == 50  # no pause has ended yet: counts from ride start
    assert by_t[400]["since_break_s"] == 0  # 300 s pause just ended
    assert by_t[750]["since_break_s"] == 350  # 100 s pause ignored


def test_crank_torque_needs_cadence_of_at_least_30_rpm():
    rows = feature_rows(make_ride([0, 1, 2], power=200, cadence=lambda t: [29, 30, 90][t]), ROAD)
    assert rows[0]["crank_torque"] is None
    assert rows[1]["crank_torque"] == pytest.approx(200 / (30 * 2 * math.pi / 60))
    assert rows[2]["crank_torque"] == pytest.approx(21.22, abs=0.01)


def test_drift_needs_a_20_minute_ride_and_measures_hr_rise():
    short = feature_rows(make_ride(range(1000), power=200, heart_rate=150), ROAD)
    assert all(r["hr_power_drift"] is None for r in short)

    long = {r["elapsed_s"]: r for r in feature_rows(
        make_ride(range(1600), power=200, heart_rate=lambda t: 150 if t < 1200 else 165), ROAD)}
    assert long[1100]["hr_power_drift"] is None  # baseline not yet complete: no later data used
    assert long[1500]["hr_power_drift"] == pytest.approx(10)  # 165 bpm vs 150 bpm at 200 W

    coast = {r["elapsed_s"]: r for r in feature_rows(
        make_ride(range(1600), power=lambda t: 200 if t < 1200 else 50, heart_rate=150), ROAD)}
    assert coast[1500]["hr_power_drift"] is None  # 50 W is below half the 200 W baseline


def test_gradient_is_altitude_gain_over_distance_in_15_s():
    ride = make_ride(range(40), enhanced_altitude=lambda t: 0.5 * t, distance=lambda t: 5.0 * t)
    by_t = {r["elapsed_s"]: r for r in feature_rows(ride, ROAD)}
    assert by_t[14]["gradient_15s"] is None  # no sample at or before t - 15
    assert by_t[30]["gradient_15s"] == pytest.approx(10)  # 7.5 m over 75 m
    indoor = feature_rows(ride, INDOOR)
    assert {r["gradient_15s"] for r in indoor} == {None}  # road rides only

    slow = {r["elapsed_s"]: r for r in feature_rows(
        make_ride(range(40), enhanced_altitude=lambda t: 0.5 * t, distance=lambda t: 1.0 * t), ROAD)}
    assert slow[30]["gradient_15s"] is None  # 15 m covered in the window, below 30 m


def test_ride_fraction_uses_record_span_not_session_total():
    ride = make_ride(range(100))
    ride["session_mesgs"][0]["total_elapsed_time"] = 50  # session total shorter than the records
    rows = feature_rows(ride, ROAD)
    assert rows[-1]["ride_fraction"] == 1
    assert rows[49]["ride_fraction"] == pytest.approx(49 / 99)


def test_mechanism_fields_treat_zero_torque_effectiveness_as_missing_and_span_phase_mod_360():
    ride = make_ride(range(3), left_torque_effectiveness=lambda t: [0, 65, 70][t],
                     left_power_phase=[300.0, 30.0], right_power_phase=None)
    rows = feature_rows(ride, ROAD)
    assert rows[0]["left_torque_eff"] is None  # 0 means no reading
    assert rows[1]["left_torque_eff"] == 65
    assert rows[1]["left_phase_deg"] == pytest.approx(90)  # 300 deg to 30 deg, wraps past 360
    assert rows[1]["right_phase_deg"] is None


def test_feature_rows_one_per_record_with_every_column():
    rows = feature_rows(make_ride(range(70), power=200, cadence=90, heart_rate=150), ROAD)
    assert len(rows) == 70
    assert set(rows[0]) == set(FEATURE_COLUMNS)
    assert rows[9]["left_pct_10s"] is None  # window not full
    assert rows[10]["left_pct_10s"] == 47
    assert rows[59]["power_60s"] is None
    assert rows[60]["power_60s"] == 200
