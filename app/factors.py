"""
Factor exposure (bonus): regress a portfolio's daily return series against
the Momentum / Value / Size factor return series over their common date
range, and return the resulting betas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import DataStore
from app import stats

FACTOR_ORDER = ["momentum", "value", "size"]


def portfolio_return_series(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return pd.Series(returns.to_numpy() @ w, index=returns.index)


def compute_betas(store: DataStore, returns: pd.DataFrame, w: np.ndarray) -> dict[str, float]:
    """
    OLS regression: portfolio_return = alpha + b1*Momentum + b2*Value + b3*Size + error,
    fit over the dates common to both the portfolio's return series and the
    factor return series.
    """
    port = portfolio_return_series(returns, w)
    joined = store.aligned_with_factors(port)
    if len(joined) < 30:
        raise ValueError("Not enough overlapping history between the portfolio and factor data to run a regression")

    y = joined["portfolio"].to_numpy()
    X = joined[FACTOR_ORDER].to_numpy()
    X_design = np.column_stack([np.ones(len(X)), X])  # add intercept (alpha)

    coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    # coeffs[0] = alpha, coeffs[1:] = betas in FACTOR_ORDER
    return {factor: float(coeffs[i + 1]) for i, factor in enumerate(FACTOR_ORDER)}


def fund_factor_betas(store: DataStore, returns: pd.DataFrame) -> np.ndarray:
    """
    Per-security 3-factor betas, shape (n_securities, 3) in FACTOR_ORDER.

    OLS coefficients are linear in y, and a portfolio's return series is
    linear in its weights, so over a fixed date window:
        beta_portfolio(w) = w @ fund_factor_betas(...)
    exactly. The optimizer therefore only needs these n regressions once,
    and every candidate portfolio's beta is a dot product -- the same
    3-factor model that compute_betas reports, not an approximation of it.
    """
    n = returns.shape[1]
    rows = []
    for i in range(n):
        betas = compute_betas(store, returns, np.eye(n)[i])
        rows.append([betas[f] for f in FACTOR_ORDER])
    return np.array(rows)
