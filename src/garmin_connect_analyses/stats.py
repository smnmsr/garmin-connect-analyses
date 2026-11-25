import numpy as np
import pyarrow.parquet as pq
import statsmodels.api as sm


def read_columns(path, columns):
    """Named columns of a parquet file as numpy arrays. Nulls become NaN. Repeated names are read once."""
    columns = list(dict.fromkeys(columns))
    table = pq.read_table(path, columns=columns)
    return {c: table[c].to_numpy(zero_copy_only=False) for c in columns}


def analysis_mask(t):
    """Rows of the balance analysis: road rides, known side, power > 0."""
    return (t["indoor"] == 0) & np.isfinite(t["left_pct"]) & (t["power"] > 0)


def within_ride(x, inv):
    """Subtract each ride's mean from every column of x (rows x columns). inv: ride index per row."""
    counts = np.bincount(inv)
    means = np.stack([np.bincount(inv, weights=col) for col in x.T], axis=1) / counts[:, None]
    return x - means[inv]


def fe_ols(y, X, groups, names):
    """Within-ride OLS: ride means removed from y and X, standard errors clustered by ride.

    Inference uses the t distribution with (rides - 1) degrees of freedom.
    Effects are per 1 SD of the within-ride variation of each column, in the unit of y.
    """
    _, inv = np.unique(groups, return_inverse=True)
    yd = within_ride(y[:, None], inv)[:, 0]
    Xd = within_ride(X, inv)
    fit = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": inv}, use_t=True)
    sd = Xd.std(axis=0)
    lo, hi = fit.conf_int().T
    return [
        {"feature": name, "sd_within": s, "coef_per_sd": b * s, "ci_low": l * s, "ci_high": h * s, "p": p}
        for name, s, b, l, h, p in zip(names, sd, fit.params, lo, hi, fit.pvalues)
    ]


def bin_effects(y, x, groups, edges, ref):
    """Within-ride effect of each bin of x on y, in the unit of y, against bin ref (index into the bins).

    Bins are [edges[i], edges[i+1]). Rows outside the edges, and NaN in x, are left out. Inference as fe_ols.
    """
    idx = np.digitize(x, edges) - 1
    keep = np.isfinite(x) & (idx >= 0) & (idx < len(edges) - 1)
    bins = [b for b in range(len(edges) - 1) if b != ref]
    labels = [f"{edges[b]:g}-{edges[b + 1]:g}" for b in bins]
    X = np.column_stack([idx[keep] == b for b in bins]).astype(float)
    rows = fe_ols(y[keep], X, groups[keep], labels)
    return [
        {
            "bin": label, "coef": r["coef_per_sd"] / r["sd_within"], "ci_low": r["ci_low"] / r["sd_within"],
            "ci_high": r["ci_high"] / r["sd_within"], "p": r["p"],
            "rows": int((idx[keep] == b).sum()), "rides": np.unique(groups[keep][idx[keep] == b]).size,
        }
        for b, label, r in zip(bins, labels, rows)
    ]


def holm(p):
    """Holm-adjusted p-values, in input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1)
    return out


def vif(X):
    """Variance inflation factor of each column of a demeaned matrix (rows x columns)."""
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        beta = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        out.append(X[:, j].var() / (X[:, j] - others @ beta).var())
    return out
