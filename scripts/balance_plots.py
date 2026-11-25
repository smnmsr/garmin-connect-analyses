"""Plots for the balance analysis. Reads output/*.csv and the features table; writes output/plots/*.png."""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from garmin_connect_analyses.stats import analysis_mask, read_columns  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
PLOTS = OUTPUT / "plots"
LABELS = {  # metric: (name, unit of its SD)
    "cum_work_kJ": ("cumulative work", "kJ"), "elapsed_s": ("time into ride", "s"),
    "since_break_s": ("time since last break", "s"), "hr_power_drift": ("HR-power drift", "%"),
    "gradient_15s": ("gradient", "%"), "speed": ("speed", "km/h"), "crank_torque": ("crank torque", "N·m"),
    "temperature": ("temperature", "°C"), "respiration_rate": ("respiration rate", "/min"),
    "left_torque_eff": ("left torque effectiveness", "%"), "right_torque_eff": ("right torque effectiveness", "%"),
    "left_pedal_smooth": ("left pedal smoothness", "%"), "right_pedal_smooth": ("right pedal smoothness", "%"),
    "left_pco": ("left pedal centre offset", "mm"), "right_pco": ("right pedal centre offset", "mm"),
    "left_phase_deg": ("left power phase span", "°"), "right_phase_deg": ("right power phase span", "°"),
}
EFFORT_REF = {"power_60s": "60 s power (effort ref.)", "heart_rate_30s": "heart rate (effort ref.)"}
MIN_CELL_ROWS = 300  # heat map cells with fewer rows are blank
MIN_RIDE_S = 7200  # example ride: at least 2 h


def read_csv(name):
    """Rows of output/<name> as dicts, numbers converted where possible."""
    def num(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(OUTPUT / name, newline="") as f:
        return [{k: num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def ride_deviation(activity_id, value, keep):
    """Value minus its own ride's mean, over rows where keep is True (NaN elsewhere)."""
    dev = np.full(value.size, np.nan)
    _, inv = np.unique(activity_id[keep], return_inverse=True)
    x = value[keep]
    dev[keep] = x - (np.bincount(inv, weights=x) / np.bincount(inv))[inv]
    return dev


def edges_of(label):
    lo, hi = label.split("-")
    return float(lo), float(hi)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


def sd_text(sd, unit):
    return f"1 SD = {sd:,.0f} {unit}" if sd >= 100 else f"1 SD = {sd:.3g} {unit}"


def plot_forest(partial, model):
    """Effect of each metric on left % per 1 SD, with effort held fixed. Effort references for scale."""
    rows = [
        {"name": f"{LABELS[r['metric']][0]} ({sd_text(r['sd_within'], LABELS[r['metric']][1])}, {int(r['rides'])} rides)",
         "coef": r["coef_per_sd"], "lo": r["ci_low"], "hi": r["ci_high"], "sig": r["p_holm"] < 0.05, "ref": False}
        for r in partial
    ]
    rows += [
        {"name": f"{EFFORT_REF[r['feature']]} ({sd_text(r['sd_within'], 'W' if r['feature'] == 'power_60s' else 'bpm')}, "
                 f"{int(r['rides'])} rides)",
         "coef": r["coef_per_sd"], "lo": r["ci_low"], "hi": r["ci_high"], "sig": r["p_holm"] < 0.05, "ref": True}
        for r in model if r["variant"] == "power_60s" and r["model"] == "joint_time" and r["feature"] in EFFORT_REF
    ]
    rows.sort(key=lambda r: r["coef"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.5))
    for y, r in enumerate(rows):
        color = "grey" if r["ref"] else ("C3" if r["coef"] > 0 else "C0")
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=2 if r["sig"] else 1, alpha=1 if r["sig"] else 0.5)
        ax.plot(r["coef"], y, "o", color=color, mfc=color if r["sig"] else "white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=9)
    ax.set_xlabel("left % per 1 SD of the metric (pp), 95% CI. Filled: Holm p < 0.05")
    ax.set_title("Other metrics vs left share, with effort held fixed (within-ride)")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "01_forest_partial_effects.png")


def plot_shape(shape):
    """Left % by bin of power (60 s) and heart rate (30 s), within-ride, vs a reference bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, feature, title, unit in zip(axes, ["power_60s", "heart_rate_30s"],
                                        ["60 s power", "heart rate, 30 s mean"], ["W", "bpm"]):
        rows = [r for r in shape if r["feature"] == feature]
        ref = rows[0]["reference"]
        points = [(edges_of(r["bin"])[0], r["bin"], r["coef"], r["ci_low"], r["ci_high"], r["p_holm"] < 0.05)
                  for r in rows]
        points.append((edges_of(ref)[0], ref, 0.0, 0.0, 0.0, None))  # reference bin, zero by definition
        points.sort()
        for i, (_, label, c, lo, hi, sig) in enumerate(points):
            if sig is None:
                ax.plot(i, 0, "D", color="black", ms=7)
                continue
            color = "C3" if c > 0 else "C0"
            ax.errorbar(i, c, yerr=[[c - lo], [hi - c]], fmt="o", color=color, mfc=color if sig else "white", capsize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(points)), [p[1] for p in points], fontsize=9)
        ax.set_xlabel(f"{title} ({unit}), bin")
        ax.set_ylabel(f"left % vs reference bin {ref} {unit} (pp)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Shape: left share rises slightly at 50–150 W, then falls above 200 W "
                 "(diamond = reference, filled = Holm p < 0.05)")
    save(fig, "02_shape_power_hr.png")


def plot_heatmap(t, base):
    """Ride-centred left % in cells of 60 s power and 30 s heart rate. Shows which one moves balance."""
    keep = base & np.isfinite(t["power_60s"]) & np.isfinite(t["heart_rate_30s"])
    dev = ride_deviation(t["activity_id"], t["left_pct"], keep)[keep]
    xe, ye = np.arange(0, 401, 25), np.arange(100, 181, 5)
    xi, yi = np.digitize(t["power_60s"][keep], xe) - 1, np.digitize(t["heart_rate_30s"][keep], ye) - 1
    ok = (xi >= 0) & (xi < xe.size - 1) & (yi >= 0) & (yi < ye.size - 1)
    cell = yi[ok] * (xe.size - 1) + xi[ok]
    n = np.bincount(cell, minlength=(xe.size - 1) * (ye.size - 1))
    total = np.bincount(cell, weights=dev[ok], minlength=n.size)
    mean = np.where(n >= MIN_CELL_ROWS, total / np.maximum(n, 1), np.nan).reshape(ye.size - 1, xe.size - 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    lim = np.nanmax(np.abs(mean))
    mesh = ax.pcolormesh(xe, ye, mean, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))  # red = more left
    fig.colorbar(mesh, ax=ax, label="left % vs own ride average (pp): red = more left, blue = more right")
    ax.set_xlabel("60 s power (W)")
    ax.set_ylabel("heart rate, 30 s mean (bpm)")
    ax.set_title("Balance across effort: power and heart rate together (blank = fewer than 300 rows)")
    save(fig, "03_heatmap_power_hr.png")


def plot_spread(t, base):
    """Within-ride spread of left % by 1 s power: the per-sample noise at low power."""
    keep = base & np.isfinite(t["power"])
    dev = ride_deviation(t["activity_id"], t["left_pct"], keep)[keep]
    edges = np.array([0, 50, 100, 150, 200, 250, 300, 400, 2000])
    idx = np.digitize(t["power"][keep], edges) - 1
    labels = [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]
    sd = [dev[idx == i].std() for i in range(len(labels))]
    n = [int((idx == i).sum()) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(labels)), sd, color="grey")
    for i, (s, k) in enumerate(zip(sd, n)):
        ax.text(i, s + 0.2, f"{s:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)), [f"{lab}\nn={k // 1000}k" for lab, k in zip(labels, n)], fontsize=9)
    ax.set_xlabel("1 s power (W)")
    ax.set_ylabel("SD of left % around the ride mean (pp)")
    ax.set_title("Per-sample balance is much noisier at low power")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "04_spread_by_power.png")


def plot_timeline(t, base, rides):
    """Ride mean left % over time, coloured by ride mean power."""
    ids = np.unique(t["activity_id"][base])
    mean_left = np.array([t["left_pct"][base & (t["activity_id"] == a)].mean() for a in ids])
    mean_w = np.array([t["power"][base & (t["activity_id"] == a)].mean() for a in ids])
    dates = [datetime.strptime(rides[int(a)], "%Y-%m-%d %H:%M:%S") for a in ids]

    years = np.array([(d - datetime(2024, 1, 1)).days / 365.25 for d in dates])
    slope = np.polyfit(years, mean_left, 1)[0]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sc = ax.scatter(dates, mean_left, c=mean_w, cmap="viridis", s=40, edgecolor="black", lw=0.4)
    fig.colorbar(sc, ax=ax, label="ride mean power (W)")
    ax.axhline(mean_left.mean(), color="black", lw=0.8, ls="--", label=f"mean of rides {mean_left.mean():.1f}%")
    ax.axhline(50, color="grey", lw=0.8, ls=":", label="50 %")
    ax.set_ylabel("ride mean left %")
    ax.set_title(f"Ride averages drift right: {slope:+.2f} pp per year, ride-to-ride SD {mean_left.std():.1f} pp")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "05_ride_averages_timeline.png")


def plot_example_ride(t, base):
    """One long road ride: balance, 60 s power and heart rate over time. Most recent ride of at least 2 h."""
    span = {a: t["elapsed_s"][t["activity_id"] == a].max() for a in np.unique(t["activity_id"][base])}
    ride = max(a for a, s in span.items() if s >= MIN_RIDE_S)
    m = (t["activity_id"] == ride) & base
    minutes = t["elapsed_s"][m] / 60

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(minutes, t["left_pct_10s"][m], color="C2", lw=0.8)
    axes[0].axhline(t["left_pct"][m].mean(), color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("left % (10 s)")
    axes[1].plot(minutes, t["power_60s"][m], color="C1", lw=0.8)
    axes[1].set_ylabel("power 60 s (W)")
    axes[2].plot(minutes, t["heart_rate_30s"][m], color="C3", lw=0.8)
    axes[2].set_ylabel("HR 30 s (bpm)")
    axes[2].set_xlabel("minutes into ride")
    fig.suptitle(f"One road ride (activity {ride}): balance, power and heart rate over time")
    for ax in axes:
        ax.grid(alpha=0.3)
    save(fig, "06_example_ride.png")


def main():
    PLOTS.mkdir(exist_ok=True)
    t = read_columns(OUTPUT / "balance_features.parquet", [
        "activity_id", "indoor", "elapsed_s", "left_pct", "left_pct_10s", "power", "power_60s", "heart_rate_30s"])
    base = analysis_mask(t)
    rides = {int(r["activity_id"]): r["start"] for r in read_csv("balance_rides.csv")}

    plot_forest(read_csv("balance_partial.csv"), read_csv("balance_model.csv"))
    plot_shape(read_csv("balance_shape.csv"))
    plot_heatmap(t, base)
    plot_spread(t, base)
    plot_timeline(t, base, rides)
    plot_example_ride(t, base)


if __name__ == "__main__":
    main()
