import bisect
import math
import statistics

TOLERANCE_PP = 0.5  # max gap, in percentage points, between our ride mean and Garmin's avgLeftBalance
PAUSE_MIN_S = 300  # shorter timer pauses are not breaks
DRIFT_BASELINE_S = 1200  # first 20 min: drift baseline, and the shortest ride with a drift value
DRIFT_MIN_POWER_SHARE = 0.5  # trailing 5 min power must reach this share of baseline power
GRADIENT_MIN_M = 30  # less distance than this in the 15 s window: grade is noise
MIN_TORQUE_RPM = 30  # below this, crank torque is not computed (freewheeling)
FEATURE_COLUMNS = [
    "activity_id", "year", "indoor", "elapsed_s", "ride_fraction", "left_pct", "left_pct_10s",
    "power", "power_60s", "cadence", "crank_torque", "heart_rate", "heart_rate_30s",
    "cum_work_kJ", "since_break_s", "hr_power_drift", "gradient_15s", "speed",
]


def summarise(messages):
    """Count records, power samples and left/right balance samples in one decoded activity."""
    records = messages.get("record_mesgs", [])
    return {
        "records": len(records),
        "power_samples": sum(r.get("power") is not None for r in records),
        "balance_samples": sum(r.get("left_right_balance") is not None for r in records),
    }


def ride_rows(rides):
    """One row per ride, all 'not attempted' until check_rides fills them in."""
    return [
        {
            "activity_id": r["activityId"],
            "type": r["activityType"]["typeKey"],
            "start": r["startTimeLocal"],
            "status": "not attempted",
            "error": "",
        }
        for r in rides
    ]


def check_rides(fetch, rows):
    """Fill in status and summary of each row, in place.

    fetch(activity_id) returns decoded messages. A bad file (ValueError) is recorded and skipped.
    Any other error is recorded and stops the run, so the remaining rides stay 'not attempted'.
    """
    for row in rows:
        try:
            row.update(summarise(fetch(row["activity_id"])), status="ok")
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
            if not isinstance(e, ValueError):
                break


def left_pct(value):
    """Left share in percent of one raw record balance value. None if the side or value is unclear."""
    if not isinstance(value, int) or value <= 0x80:  # 'right' string is raw 0x80; below it the side is unknown
        return None
    left = 100 - (value & 0x7F)
    return left if 5 < left < 95 else None  # data-quality band: 5% or less, or 95% or more, is excluded


def ride_samples(messages):
    """Usable samples of one ride: known side and power > 0. Times are seconds from the first record."""
    records = messages.get("record_mesgs", [])
    if not records:
        return []
    start = records[0]["timestamp"]
    samples = []
    for r in records:
        left = left_pct(r.get("left_right_balance"))
        if left is None or not r.get("power"):
            continue
        samples.append({
            "elapsed_s": (r["timestamp"] - start).total_seconds(),
            "left_pct": left,
            "power": r["power"],
            "heart_rate": r.get("heart_rate"),
            "cadence": r.get("cadence"),
        })
    return samples


def ride_status(samples, summary_left):
    """(status, mean left %) of one ride. Status is 'ok' or the reason the ride is excluded."""
    if not samples:
        return "excluded: no usable balance samples", None
    mean = statistics.mean(s["left_pct"] for s in samples)
    if summary_left is None:
        return "excluded: no Garmin balance to validate against", mean
    if abs(mean - summary_left) > TOLERANCE_PP:
        return f"excluded: {mean - summary_left:+.2f} pp off Garmin", mean
    return "ok", mean


def trailing_mean(times, values, window, min_count=1):
    """Mean of values over the trailing window (t - window, t] at each sample time t.

    None while the ride is shorter than the window, or the window holds fewer than min_count values.
    None values are skipped. Two pointers keep this linear in the number of samples.
    """
    means, lo, total, count = [], 0, 0, 0
    for hi, t in enumerate(times):
        if values[hi] is not None:
            total += values[hi]
            count += 1
        while times[lo] <= t - window:
            if values[lo] is not None:
                total -= values[lo]
                count -= 1
            lo += 1
        means.append(total / count if t >= window and count >= min_count else None)
    return means


def pause_ends(events, start, min_s=PAUSE_MIN_S):
    """Elapsed seconds at which each pause of at least min_s ended.

    A pause is a timer stop_all followed by the next timer start. A stop with no restart (end of ride) is not a pause.
    """
    ends, stopped = [], None
    timer = sorted((e for e in events if str(e.get("event")) == "timer"), key=lambda e: e["timestamp"])
    for e in timer:
        kind = str(e.get("event_type"))
        if kind == "stop_all":
            stopped = e["timestamp"]
        elif kind == "start" and stopped is not None:
            if (e["timestamp"] - stopped).total_seconds() >= min_s:
                ends.append((e["timestamp"] - start).total_seconds())
            stopped = None
    return ends


def gradient_pct(times, altitude, distance, window=15):
    """Altitude change over distance covered in the trailing window, in %.

    Reference is the latest sample at or before t - window. None without a reference or when less than
    GRADIENT_MIN_M of distance was covered.
    """
    grades = []
    for i, t in enumerate(times):
        j = bisect.bisect_right(times, t - window) - 1
        if j < 0 or None in (altitude[i], distance[i], altitude[j], distance[j]) or distance[i] - distance[j] < GRADIENT_MIN_M:
            grades.append(None)
        else:
            grades.append((altitude[i] - altitude[j]) / (distance[i] - distance[j]) * 100)
    return grades


def hr_power_drift(times, hr, power, total_s):
    """Percent change in heart rate per watt: trailing 5 min against the first 20 min. None for rides under 20 min.

    Empty where trailing 5 min power is below DRIFT_MIN_POWER_SHARE of the baseline power (coasting, stops).
    """
    if not total_s or total_s < DRIFT_BASELINE_S:
        return [None] * len(times)
    base_hr = [h for t, h in zip(times, hr) if t < DRIFT_BASELINE_S and h is not None]
    base_w = [p for t, p in zip(times, power) if t < DRIFT_BASELINE_S and p is not None]
    if not base_hr or not base_w or sum(base_w) == 0:
        return [None] * len(times)
    base_w_mean = statistics.mean(base_w)
    base = statistics.mean(base_hr) / base_w_mean
    floor = DRIFT_MIN_POWER_SHARE * base_w_mean
    hr_5, w_5 = trailing_mean(times, hr, 300), trailing_mean(times, power, 300)
    return [
        (h / p / base - 1) * 100 if h is not None and p is not None and p >= floor else None
        for h, p in zip(hr_5, w_5)
    ]


def feature_rows(messages, ride):
    """One row per FIT record of a ride. Column definitions: docs/balance_features.md. None means missing."""
    records = messages["record_mesgs"]
    total_s = messages["session_mesgs"][0].get("total_elapsed_time")
    start = records[0]["timestamp"]
    indoor = ride["activityType"]["typeKey"] == "indoor_cycling"
    times = [(r["timestamp"] - start).total_seconds() for r in records]
    span_s = times[-1]  # ride_fraction uses the records, not the session total, so it stays within 0-1

    def field(name):
        return [r.get(name) for r in records]

    power, cadence, hr = field("power"), field("cadence"), field("heart_rate")
    speed, altitude, distance = field("enhanced_speed"), field("enhanced_altitude"), field("distance")
    acc = field("accumulated_power")
    left = [left_pct(r.get("left_right_balance")) for r in records]
    usable = [pct if pct is not None and p else None for pct, p in zip(left, power)]  # known side and power > 0
    left_10s = trailing_mean(times, usable, 10, min_count=5)
    power_60s = trailing_mean(times, power, 60)
    hr_30s = trailing_mean(times, hr, 30)
    drift = hr_power_drift(times, hr, power, total_s)
    grade = [None] * len(times) if indoor else gradient_pct(times, altitude, distance)
    breaks = pause_ends(messages.get("event_mesgs", []), start)
    year = int(ride["startTimeLocal"][:4])

    rows = []
    for i, t in enumerate(times):
        p, c = power[i], cadence[i]
        k = bisect.bisect_right(breaks, t) - 1
        rows.append({
            "activity_id": ride["activityId"],
            "year": year,
            "indoor": int(indoor),
            "elapsed_s": t,
            "ride_fraction": t / span_s if span_s else None,
            "left_pct": left[i],
            "left_pct_10s": left_10s[i],
            "power": p,
            "power_60s": power_60s[i],
            "cadence": c,
            "crank_torque": p / (c * 2 * math.pi / 60) if p is not None and c is not None and c >= MIN_TORQUE_RPM else None,
            "heart_rate": hr[i],
            "heart_rate_30s": hr_30s[i],
            "cum_work_kJ": acc[i] / 1000 if acc[i] is not None else None,
            "since_break_s": t - breaks[k] if k >= 0 else t,
            "hr_power_drift": drift[i],
            "gradient_15s": grade[i],
            "speed": speed[i] * 3.6 if speed[i] is not None else None,
        })
    return rows
