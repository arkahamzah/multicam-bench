"""Two-term capacity cost model: `cost = a*pixel_rate + b*N`
(THREATS-TO-VALIDITY.md §5, PRIOR-ART.md §4.3).

`a` is the marginal cost per unit of per-stream pixel rate; `b` is the marginal
cost per additional stream, independent of its pixel rate — the per-stream
overhead this project never assumes is zero. No intercept term: zero streams and
zero pixel rate must cost zero by construction.

Ordinary least squares, closed-form, with confidence intervals from the
t-distribution and explicit degrees of freedom — "fitting two parameters to five
noisy points is stated as such rather than presented as an established law"
(THREATS-TO-VALIDITY.md §5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

N_PARAMS = 2  # a (pixel_rate coefficient), b (per-stream coefficient) — no intercept


@dataclass(frozen=True)
class FitPoint:
    pixel_rate: float  # width * height * fps for this data point's content
    n_streams: int
    cost: float  # e.g. total reader-process CPU% at this operating point


@dataclass(frozen=True)
class CoefficientEstimate:
    value: float
    stderr: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class CapacityFit:
    a: CoefficientEstimate
    b: CoefficientEstimate
    residuals: list[float]
    dof: int
    r_squared: float
    n_points: int
    confidence: float


def fit_capacity_model(points: Sequence[FitPoint], confidence: float = 0.95) -> CapacityFit:
    """OLS fit of `cost = a*pixel_rate + b*N`, no intercept.

    Needs at least 3 points (dof = n - 2 > 0) and genuine variation in both
    `pixel_rate` and `n_streams` independently — a sweep that only ever varies one
    axis cannot separate `a` from `b` and this raises rather than returning a
    fit that looks precise but isn't identified.
    """
    n = len(points)
    if n <= N_PARAMS:
        raise ValueError(
            f"need more than {N_PARAMS} data points for nonzero degrees of freedom, got {n}"
        )
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    x = np.array([[pt.pixel_rate, float(pt.n_streams)] for pt in points], dtype=float)
    y = np.array([pt.cost for pt in points], dtype=float)

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "design matrix is singular — pixel_rate and n_streams must vary "
            "independently across the fitted points (e.g. more than one resolution/"
            "fps combination AND more than one N) to separate a from b"
        ) from exc

    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    dof = n - N_PARAMS
    sigma2 = float(np.sum(residuals**2) / dof)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))

    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def coef(idx: int) -> CoefficientEstimate:
        val = float(beta[idx])
        err = float(se[idx])
        return CoefficientEstimate(
            value=val, stderr=err, ci_low=val - t_crit * err, ci_high=val + t_crit * err
        )

    return CapacityFit(
        a=coef(0),
        b=coef(1),
        residuals=residuals.tolist(),
        dof=dof,
        r_squared=r_squared,
        n_points=n,
        confidence=confidence,
    )


def predict_cost(fit: CapacityFit, pixel_rate: float, n_streams: int) -> float:
    """cost = a*pixel_rate + b*N using the fitted point estimates."""
    return fit.a.value * pixel_rate + fit.b.value * n_streams
