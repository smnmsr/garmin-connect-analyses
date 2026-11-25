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
