"""
Turns the request's `constraints` block into:
  1. scipy `bounds` (per-security min/max weight)
  2. a list of scipy inequality/equality constraint dicts
     (portfolio-level: min dividend yield, min CAGR, volatility range,
     max drawdown, plus the always-on "weights sum to 1")
  3. an upfront feasibility pre-check that catches obviously-impossible
     bound combinations *before* we ever call the optimizer
  4. a post-solve re-check, so a non-converged or barely-violating
     result never gets returned as if it were valid

All weight-ish constraint inputs arrive from the API as percentages
(0-100); everything here works in fractions (0-1) to match the return
data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.errors import OptimizationError
from app.schemas import Constraints
from app import stats

TOL = 1e-4


@dataclass
class BuiltConstraints:
    bounds: list[tuple[float, float]]
    scipy_constraints: list[dict]
    # kept around so we can re-validate the final result after solving
    min_dividend_yield: float | None = None
    min_cagr: float | None = None
    vol_min: float | None = None
    vol_max: float | None = None
    max_drawdown: float | None = None
    dividend_yields: list[float] = field(default_factory=list)


def build(
    tickers: list[str],
    dividend_yields: list[float],
    returns: pd.DataFrame,
    constraints: Constraints | None,
) -> BuiltConstraints:
    n = len(tickers)

    global_min = (constraints.min_weight / 100.0) if (constraints and constraints.min_weight is not None) else 0.0
    global_max = (constraints.max_weight / 100.0) if (constraints and constraints.max_weight is not None) else 1.0

    bounds: list[tuple[float, float]] = []
    for t in tickers:
        lo, hi = global_min, global_max
        if constraints and constraints.security_constraints and t in constraints.security_constraints:
            override = constraints.security_constraints[t]
            if override.min_weight is not None:
                lo = override.min_weight / 100.0
            if override.max_weight is not None:
                hi = override.max_weight / 100.0
        if lo > hi:
            raise OptimizationError(f"Infeasible constraints for {t}: min_weight ({lo*100:.2f}%) exceeds max_weight ({hi*100:.2f}%)")
        bounds.append((lo, hi))

    # --- Upfront feasibility: can the per-security bounds even sum to 100%? ---
    sum_lo = sum(b[0] for b in bounds)
    sum_hi = sum(b[1] for b in bounds)
    if sum_lo > 1.0 + TOL:
        raise OptimizationError(
            f"Infeasible constraints: minimum weights sum to {sum_lo*100:.2f}%, which already exceeds 100%"
        )
    if sum_hi < 1.0 - TOL:
        raise OptimizationError(
            f"Infeasible constraints: maximum weights sum to {sum_hi*100:.2f}%, which cannot reach 100%"
        )

    scipy_constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    min_dividend_yield = None
    min_cagr = None
    vol_min = None
    vol_max = None
    max_dd = None

    if constraints:
        if constraints.min_dividend_yield is not None:
            min_dividend_yield = constraints.min_dividend_yield / 100.0
            scipy_constraints.append(
                {"type": "ineq", "fun": lambda w: stats.portfolio_dividend_yield(w, dividend_yields) - min_dividend_yield}
            )
            # Feasibility pre-check: even at the best achievable weighting within
            # bounds, is the max possible portfolio yield >= the requirement?
            # (Best case: pile weight onto the highest-yield securities within bounds.)
            best_possible = _max_achievable_weighted_value(bounds, [y if y is not None else 0.0 for y in dividend_yields])
            if best_possible < min_dividend_yield - TOL:
                raise OptimizationError(
                    f"Infeasible constraints: even the best achievable allocation within the given weight "
                    f"bounds yields a portfolio dividend yield of {best_possible*100:.2f}%, "
                    f"below the required minimum of {constraints.min_dividend_yield:.2f}%"
                )

        if constraints.min_cagr is not None:
            min_cagr = constraints.min_cagr / 100.0
            scipy_constraints.append(
                {"type": "ineq", "fun": lambda w: stats.cagr(returns, w) - min_cagr}
            )

        if constraints.volatility_range is not None:
            if constraints.volatility_range.min is not None:
                vol_min = constraints.volatility_range.min / 100.0
                scipy_constraints.append(
                    {"type": "ineq", "fun": lambda w: stats.annualized_volatility(returns, w) - vol_min}
                )
            if constraints.volatility_range.max is not None:
                vol_max = constraints.volatility_range.max / 100.0
                scipy_constraints.append(
                    {"type": "ineq", "fun": lambda w: vol_max - stats.annualized_volatility(returns, w)}
                )

        if constraints.max_drawdown is not None:
            max_dd = constraints.max_drawdown / 100.0
            scipy_constraints.append(
                {"type": "ineq", "fun": lambda w: max_dd - stats.max_drawdown(returns, w)}
            )

    return BuiltConstraints(
        bounds=bounds,
        scipy_constraints=scipy_constraints,
        min_dividend_yield=min_dividend_yield,
        min_cagr=min_cagr,
        vol_min=vol_min,
        vol_max=vol_max,
        max_drawdown=max_dd,
        dividend_yields=dividend_yields,
    )


def _max_achievable_weighted_value(bounds: list[tuple[float, float]], values: list[float]) -> float:
    """
    Given per-security [lo, hi] weight bounds that must sum to 1, and a
    per-security value (e.g. dividend yield), returns the maximum
    achievable weighted average: start every security at its minimum,
    then greedily pour the remaining slack into the highest-value
    securities first.
    """
    n = len(bounds)
    w = np.array([b[0] for b in bounds], dtype=float)
    slack = 1.0 - w.sum()
    capacity = np.array([b[1] - b[0] for b in bounds], dtype=float)
    order = np.argsort(values)[::-1]  # highest value first
    for i in order:
        if slack <= 0:
            break
        take = min(capacity[i], slack)
        w[i] += take
        slack -= take
    return float(np.dot(w, values))


def validate_solution(
    w: np.ndarray,
    built: BuiltConstraints,
    returns: pd.DataFrame,
) -> None:
    """Post-solve sanity check -- raises if the optimizer handed back something invalid."""
    if abs(w.sum() - 1.0) > 1e-3:
        raise OptimizationError("Optimizer failed to converge on a feasible portfolio (weights do not sum to 100%)")
    if np.any(w < -1e-6):
        raise OptimizationError("Optimizer failed to converge on a feasible portfolio (negative weight produced)")
    for wi, (lo, hi) in zip(w, built.bounds):
        if wi < lo - 1e-3 or wi > hi + 1e-3:
            raise OptimizationError("Optimizer result violates security-level min/max weight constraints")
    if built.min_dividend_yield is not None:
        y = stats.portfolio_dividend_yield(w, built.dividend_yields)
        if y < built.min_dividend_yield - 1e-3:
            raise OptimizationError("Optimizer could not satisfy the minimum dividend yield constraint")
    if built.min_cagr is not None:
        if stats.cagr(returns, w) < built.min_cagr - 1e-3:
            raise OptimizationError("Optimizer could not satisfy the minimum CAGR constraint")
    if built.vol_min is not None or built.vol_max is not None:
        vol = stats.annualized_volatility(returns, w)
        if built.vol_min is not None and vol < built.vol_min - 1e-3:
            raise OptimizationError("Optimizer could not satisfy the minimum volatility constraint")
        if built.vol_max is not None and vol > built.vol_max + 1e-3:
            raise OptimizationError("Optimizer could not satisfy the maximum volatility constraint")
    if built.max_drawdown is not None:
        if stats.max_drawdown(returns, w) > built.max_drawdown + 1e-3:
            raise OptimizationError("Optimizer could not satisfy the maximum drawdown constraint")
