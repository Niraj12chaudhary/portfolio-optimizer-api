"""
The five required optimization strategies, plus the bonus factor-exposure
strategy. Each returns a plain numpy weight vector (fractions summing to 1)
for the given tickers, in the same order as `returns.columns`.

All strategies except Equal Weights are solved with scipy's SLSQP, which
handles bounded variables plus arbitrary equality/inequality constraints --
exactly what we need for security-level bounds + portfolio-level
constraints (min CAGR, volatility range, max drawdown, min dividend yield).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app import stats, factors
from app.constraints import BuiltConstraints, validate_solution
from app.data import DataStore
from app.errors import OptimizationError
from app.schemas import Direction


def _initial_guess(built: BuiltConstraints, n: int) -> np.ndarray:
    """Start at the midpoint of each security's bounds, then rescale to sum to 1."""
    mids = np.array([(lo + hi) / 2 for lo, hi in built.bounds])
    if mids.sum() <= 0:
        mids = np.full(n, 1.0 / n)
    return mids / mids.sum()


def _solve(objective, built: BuiltConstraints, returns: pd.DataFrame, n: int) -> np.ndarray:
    x0 = _initial_guess(built, n)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=built.bounds,
        constraints=built.scipy_constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if not result.success:
        # Retry once from equal weights in case the midpoint start was a bad seed.
        result2 = minimize(
            objective,
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=built.bounds,
            constraints=built.scipy_constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result2.success:
            result = result2

    if not result.success:
        raise OptimizationError(
            f"Could not find a feasible portfolio satisfying all constraints for this strategy "
            f"(optimizer message: {result.message})"
        )

    w = np.clip(result.x, 0, None)
    w = w / w.sum()
    validate_solution(w, built, returns)
    return w


def equal_weights(built: BuiltConstraints, returns: pd.DataFrame, n: int) -> np.ndarray:
    w = np.full(n, 1.0 / n)
    validate_solution(w, built, returns)
    return w


def risk_parity(built: BuiltConstraints, returns: pd.DataFrame, n: int) -> np.ndarray:
    # Work in risk *shares* (each RC as a fraction of total variance) rather
    # than raw variance units: raw RCs are ~1e-3, so their squared deviations
    # (~1e-8) fall under SLSQP's ftol and it stops before contributions equalize.
    def objective(w):
        rc = stats.risk_contributions(returns, w)
        shares = rc / rc.sum()
        return float(np.sum((shares - 1.0 / n) ** 2))

    return _solve(objective, built, returns, n)


def minimize_volatility(built: BuiltConstraints, returns: pd.DataFrame, n: int) -> np.ndarray:
    def objective(w):
        return stats.annualized_volatility(returns, w)

    return _solve(objective, built, returns, n)


def maximize_sharpe(built: BuiltConstraints, returns: pd.DataFrame, n: int, risk_free_rate: float = 0.0) -> np.ndarray:
    def objective(w):
        return -stats.sharpe_ratio(returns, w, risk_free_rate)

    return _solve(objective, built, returns, n)


def minimize_drawdown(built: BuiltConstraints, returns: pd.DataFrame, n: int) -> np.ndarray:
    def objective(w):
        return stats.max_drawdown(returns, w)

    return _solve(objective, built, returns, n)


def optimize_factor_exposure(
    built: BuiltConstraints,
    returns: pd.DataFrame,
    n: int,
    store: DataStore,
    factor: str,
    direction: Direction,
) -> np.ndarray:
    sign = -1.0 if direction == Direction.MAXIMIZE else 1.0

    try:
        fund_betas = factors.fund_factor_betas(store, returns)
    except ValueError as exc:
        raise OptimizationError(str(exc))
    # Portfolio beta is linear in the weights (see fund_factor_betas), so the
    # objective is just a dot product with each fund's beta to this factor.
    betas = fund_betas[:, factors.FACTOR_ORDER.index(factor)]

    def objective(w):
        return float(sign * (w @ betas))

    return _solve(objective, built, returns, n)
