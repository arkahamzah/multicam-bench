from __future__ import annotations

import pytest

from multicam_bench.model.fit import FitPoint, fit_capacity_model, predict_cost

TRUE_A = 2.0
TRUE_B = 3.0


def _exact_points() -> list[FitPoint]:
    # cost = 2*pixel_rate + 3*N, no noise, pixel_rate and N vary independently.
    combos = [(100.0, 1), (100.0, 2), (200.0, 1), (200.0, 2)]
    return [
        FitPoint(pixel_rate=pr, n_streams=n, cost=TRUE_A * pr + TRUE_B * n) for pr, n in combos
    ]


def test_fit_recovers_exact_coefficients_from_noiseless_data() -> None:
    fit = fit_capacity_model(_exact_points())
    assert fit.a.value == pytest.approx(TRUE_A, abs=1e-6)
    assert fit.b.value == pytest.approx(TRUE_B, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)


def test_fit_reports_correct_degrees_of_freedom() -> None:
    fit = fit_capacity_model(_exact_points())
    assert fit.n_points == 4
    assert fit.dof == 2  # n - 2 params


def test_fit_confidence_interval_contains_true_value_with_noise() -> None:
    combos = [(100.0, 1), (100.0, 2), (100.0, 4), (200.0, 1), (200.0, 2), (200.0, 4)]
    noise = [0.5, -0.5, 0.3, -0.3, 0.4, -0.4]
    points = [
        FitPoint(pixel_rate=pr, n_streams=n, cost=TRUE_A * pr + TRUE_B * n + e)
        for (pr, n), e in zip(combos, noise, strict=True)
    ]
    fit = fit_capacity_model(points, confidence=0.95)
    assert fit.a.ci_low <= TRUE_A <= fit.a.ci_high
    assert fit.b.ci_low <= TRUE_B <= fit.b.ci_high


def test_fit_wider_confidence_level_gives_wider_interval() -> None:
    points = _exact_points()
    # Add a touch of asymmetric noise so the interval has nonzero width to compare.
    noisy = [
        FitPoint(pixel_rate=p.pixel_rate, n_streams=p.n_streams, cost=p.cost + i * 0.01)
        for i, p in enumerate(points)
    ]
    fit_90 = fit_capacity_model(noisy, confidence=0.90)
    fit_99 = fit_capacity_model(noisy, confidence=0.99)
    width_90 = fit_90.a.ci_high - fit_90.a.ci_low
    width_99 = fit_99.a.ci_high - fit_99.a.ci_low
    assert width_99 >= width_90


def test_fit_rejects_too_few_points() -> None:
    points = _exact_points()[:2]  # only 2, need > 2 for nonzero dof
    with pytest.raises(ValueError, match="degrees of freedom"):
        fit_capacity_model(points)


def test_fit_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        fit_capacity_model(_exact_points(), confidence=1.0)


def test_fit_rejects_perfectly_collinear_design() -> None:
    # pixel_rate = 50 * n_streams for every point: a and b are not separable.
    points = [
        FitPoint(pixel_rate=50.0 * n, n_streams=n, cost=1000.0) for n in [1, 2, 3, 4]
    ]
    with pytest.raises(ValueError, match="singular"):
        fit_capacity_model(points)


def test_predict_cost_matches_fitted_model() -> None:
    fit = fit_capacity_model(_exact_points())
    predicted = predict_cost(fit, pixel_rate=150.0, n_streams=3)
    assert predicted == pytest.approx(TRUE_A * 150.0 + TRUE_B * 3, abs=1e-6)
