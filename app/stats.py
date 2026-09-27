"""
Pure portfolio-math helpers. Everything here operates on a daily-return
matrix (pandas DataFrame, columns = tickers, in a fixed order matching the
weight vector) and a numpy weight vector `w`. No I/O, no FastAPI, no
optimizer-specific code -- just the math, so it's easy to unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def portfolio_daily_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Weighted sum of daily returns -> single portfolio return series."""
    return returns.to_numpy() @ w


def annualized_volatility(returns: pd.DataFrame, w: np.ndarray) -> float:
    port = portfolio_daily_returns(returns, w)
    return float(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    """Annualized covariance matrix of the aligned daily returns."""
    return returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR


def annualized_return(returns: pd.DataFrame, w: np.ndarray) -> float:
    """Mean daily portfolio return, annualized by simple scaling (arithmetic)."""
    port = portfolio_daily_returns(returns, w)
    return float(np.mean(port) * TRADING_DAYS_PER_YEAR)


def cagr(returns: pd.DataFrame, w: np.ndarray) -> float:
    """
    Compound annual growth rate of the weighted portfolio's cumulative
    return series over the aligned window.
    """
    port = portfolio_daily_returns(returns, w)
    n = len(port)
    if n == 0:
        return 0.0
    cumulative_growth = np.prod(1.0 + port)
    years = n / TRADING_DAYS_PER_YEAR
    if years <= 0 or cumulative_growth <= 0:
        return -1.0
    return float(cumulative_growth ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: pd.DataFrame, w: np.ndarray, risk_free_rate: float = 0.0) -> float:
    vol = annualized_volatility(returns, w)
    if vol == 0:
        return 0.0
    return (annualized_return(returns, w) - risk_free_rate) / vol


def drawdown_series(returns: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    port = portfolio_daily_returns(returns, w)
    wealth = np.cumprod(1.0 + port)
    running_max = np.maximum.accumulate(wealth)
    return wealth / running_max - 1.0


def max_drawdown(returns: pd.DataFrame, w: np.ndarray) -> float:
    """Returned as a positive magnitude (e.g. 0.15 == a 15% max drawdown)."""
    dd = drawdown_series(returns, w)
    if len(dd) == 0:
        return 0.0
    return float(-dd.min())


def risk_contributions(returns: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """
    Each security's contribution to total portfolio variance:
    RC_i = w_i * (Sigma @ w)_i. Sums to total portfolio variance.
    Used by the risk-parity objective.
    """
    sigma = covariance_matrix(returns)
    marginal = sigma @ w
    return w * marginal


def portfolio_dividend_yield(w: np.ndarray, yields: list[float | None]) -> float:
    """
    Weighted-average dividend yield. A security with an unknown/undisclosed
    yield (None) contributes 0 -- see README for why this is treated as a
    true zero rather than an unknown quantity for GLD specifically.
    """
    filled = np.array([y if y is not None else 0.0 for y in yields])
    return float(np.dot(w, filled))
