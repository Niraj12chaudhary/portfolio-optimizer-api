from __future__ import annotations

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import constraints as constraints_module
from app import factors, optimizer
from app.data import get_store
from app.errors import OptimizationError
from app.schemas import OptimizeRequest, OptimizeResponse, Strategy

app = FastAPI(
    title="Portfolio Optimizer API",
    description="Replicates the core allocation strategies of Finominal's Portfolio Optimizer.",
    version="1.0.0",
)

MIN_HISTORY_POINTS = 30

STRATEGY_FN = {
    Strategy.EQUAL_WEIGHTS: lambda built, returns, n, store, req: optimizer.equal_weights(built, returns, n),
    Strategy.RISK_PARITY: lambda built, returns, n, store, req: optimizer.risk_parity(built, returns, n),
    Strategy.MINIMIZE_VOLATILITY: lambda built, returns, n, store, req: optimizer.minimize_volatility(built, returns, n),
    Strategy.MAXIMIZE_SHARPE: lambda built, returns, n, store, req: optimizer.maximize_sharpe(built, returns, n),
    Strategy.MINIMIZE_DRAWDOWN: lambda built, returns, n, store, req: optimizer.minimize_drawdown(built, returns, n),
    Strategy.OPTIMIZE_FACTOR_EXPOSURE: lambda built, returns, n, store, req: optimizer.optimize_factor_exposure(
        built, returns, n, store, req.factor_objective.factor.value, req.factor_objective.direction
    ),
}


@app.exception_handler(OptimizationError)
async def optimization_error_handler(request: Request, exc: OptimizationError):
    return JSONResponse(status_code=422, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Surface pydantic's messages directly instead of FastAPI's default verbose
    # shape, prefixed with the offending field's path when there is one.
    messages = []
    for e in exc.errors():
        loc = ".".join(str(part) for part in e.get("loc", ()) if part != "body")
        msg = e.get("msg", str(e))
        messages.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


def round_to_100(w: np.ndarray) -> np.ndarray:
    """
    Converts fractional weights (summing to 1) into percentages rounded to
    2 decimals that sum to exactly 100.00, via the largest-remainder method:
    floor every weight to the nearest 0.01%, then hand the leftover 0.01%
    units to the weights that lost the most in flooring. Naive rounding can
    give 99.99 or 100.01 (e.g. three equal weights -> 3 x 33.33).
    """
    units = w * 10_000.0  # weights in hundredths of a percent
    floored = np.floor(units)
    leftover = int(round(10_000 - floored.sum()))
    for i in np.argsort(-(units - floored))[:leftover]:
        floored[i] += 1
    return floored / 100.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    store = get_store()

    tickers = [s.ticker for s in req.securities]
    unknown = [t for t in tickers if t not in store.known_tickers()]
    if unknown:
        raise OptimizationError(f"Unknown ticker(s): {', '.join(unknown)}. Known tickers: {sorted(store.known_tickers())}")

    if req.strategy in (Strategy.RISK_PARITY, Strategy.MINIMIZE_VOLATILITY, Strategy.MAXIMIZE_SHARPE,
                         Strategy.MINIMIZE_DRAWDOWN, Strategy.OPTIMIZE_FACTOR_EXPOSURE) and len(tickers) < 2:
        raise OptimizationError(f"Strategy '{req.strategy.value}' requires at least 2 securities to be meaningful")

    returns = store.aligned_returns(tickers)
    if len(returns) < MIN_HISTORY_POINTS:
        raise OptimizationError(
            f"Insufficient overlapping historical data for {tickers}: only {len(returns)} common trading days "
            f"available (need at least {MIN_HISTORY_POINTS})"
        )
    # Ensure column order matches `tickers` exactly, since weight vectors are positional.
    returns = returns[tickers]

    metas = [store.get_meta(t) for t in tickers]
    dividend_yields = [m.dividend_yield for m in metas]

    built = constraints_module.build(tickers, dividend_yields, returns, req.constraints)

    n = len(tickers)
    w = STRATEGY_FN[req.strategy](built, returns, n, store, req)

    current_weights = np.array([s.current_weight for s in req.securities])
    optimized_pct = round_to_100(w)

    allocation_changes = []
    for i, t in enumerate(tickers):
        cw = float(current_weights[i])
        ow = float(optimized_pct[i])
        allocation_changes.append(
            {
                "ticker": t,
                "security_name": metas[i].security_name,
                "current_weight": round(cw, 2),
                "optimized_weight": ow,
                "change": round(ow - cw, 2),
            }
        )

    factor_betas = None
    if req.strategy == Strategy.OPTIMIZE_FACTOR_EXPOSURE:
        current_w = current_weights / 100.0
        try:
            current_betas = factors.compute_betas(store, returns, current_w)
            optimized_betas = factors.compute_betas(store, returns, w)
        except ValueError as exc:
            raise OptimizationError(str(exc))
        factor_betas = {
            "current_portfolio": {k: round(v, 4) for k, v in current_betas.items()},
            "optimized_portfolio": {k: round(v, 4) for k, v in optimized_betas.items()},
        }

    return OptimizeResponse(
        optimization_strategy=req.strategy,
        allocation_changes=allocation_changes,
        factor_betas=factor_betas,
    )
