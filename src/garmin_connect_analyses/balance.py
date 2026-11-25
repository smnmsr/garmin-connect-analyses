import statistics

TOLERANCE_PP = 0.5  # max gap, in percentage points, between our ride mean and Garmin's avgLeftBalance


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
