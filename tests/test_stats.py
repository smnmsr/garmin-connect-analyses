import numpy as np

from garmin_connect_analyses.stats import bin_effects, fe_ols, holm, vif


def test_fe_ols_matches_regression_with_ride_dummies():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 200)
    X = rng.normal(size=(1000, 2)) + groups[:, None]  # rides differ in mean X
    y = 3 * X[:, 0] - 2 * X[:, 1] + 10 * groups + rng.normal(size=1000)
    dummies = (groups[:, None] == np.arange(5)).astype(float)
    expected = np.linalg.lstsq(np.hstack([X, dummies]), y, rcond=None)[0][:2]

    rows = fe_ols(y, X, groups, ["a", "b"])
    slopes = [r["coef_per_sd"] / r["sd_within"] for r in rows]
    np.testing.assert_allclose(slopes, expected, rtol=1e-8)


def test_bin_effects_are_offsets_from_reference_bin_within_rides():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(4), 500)
    x = rng.uniform(0, 300, size=2000)
    y = np.array([2.0, 0.0, -1.0])[np.digitize(x, [100, 200])] + 5 * groups + rng.normal(scale=0.1, size=2000)

    rows = bin_effects(y, x, groups, [0, 100, 200, 300], ref=1)
    assert [r["bin"] for r in rows] == ["0-100", "200-300"]  # reference bin 100-200 is not listed
    np.testing.assert_allclose([r["coef"] for r in rows], [2, -1], atol=0.05)


def test_holm_adjusts_in_order_and_keeps_input_order():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_vif_flags_collinear_columns():
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    X = np.column_stack([a, a + 0.01 * rng.normal(size=500), rng.normal(size=500)])
    assert min(vif(X)[:2]) > 1000
    assert vif(X)[2] < 1.1
