import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def optimize(payload):
    return client.post("/optimize", json=payload)


def test_equal_weights_sums_to_100_and_non_negative():
    resp = optimize({
        "securities": [{"ticker": "IEFA", "current_weight": 25}, {"ticker": "SPY", "current_weight": 75}],
        "strategy": "equal_weights",
    })
    assert resp.status_code == 200
    data = resp.json()
    weights = [a["optimized_weight"] for a in data["allocation_changes"]]
    assert all(w >= 0 for w in weights)
    assert abs(sum(weights) - 100.0) < 0.01
    assert all(abs(w - 50.0) < 0.01 for w in weights)


def test_unknown_ticker_returns_422():
    resp = optimize({
        "securities": [{"ticker": "NOTAREALTICKER", "current_weight": 100}],
        "strategy": "equal_weights",
    })
    assert resp.status_code == 422


def test_invalid_strategy_returns_422():
    resp = optimize({
        "securities": [{"ticker": "SPY", "current_weight": 100}],
        "strategy": "not_a_real_strategy",
    })
    assert resp.status_code == 422


def test_negative_current_weight_returns_422():
    resp = optimize({
        "securities": [{"ticker": "SPY", "current_weight": -10}, {"ticker": "AGG", "current_weight": 110}],
        "strategy": "equal_weights",
    })
    assert resp.status_code == 422


def test_weights_not_summing_to_100_returns_422():
    resp = optimize({
        "securities": [{"ticker": "SPY", "current_weight": 50}, {"ticker": "AGG", "current_weight": 40}],
        "strategy": "equal_weights",
    })
    assert resp.status_code == 422


def test_min_max_weight_constraints_respected():
    resp = optimize({
        "securities": [
            {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
            {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
            {"ticker": "SPY", "current_weight": 20},
        ],
        "strategy": "maximize_sharpe",
        "constraints": {"min_weight": 5, "max_weight": 40},
    })
    assert resp.status_code == 200
    weights = [a["optimized_weight"] for a in resp.json()["allocation_changes"]]
    for w in weights:
        assert w >= 5.0 - 0.5
        assert w <= 40.0 + 0.5
    assert abs(sum(weights) - 100.0) < 0.1


def test_infeasible_bound_constraints_return_422():
    resp = optimize({
        "securities": [{"ticker": "SPY", "current_weight": 50}, {"ticker": "AGG", "current_weight": 50}],
        "strategy": "minimize_volatility",
        "constraints": {"min_weight": 60, "max_weight": 100},  # 2 * 60% > 100%
    })
    assert resp.status_code == 422


def test_min_dividend_yield_constraint_respected_with_gld_zero():
    resp = optimize({
        "securities": [
            {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
            {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
            {"ticker": "SPY", "current_weight": 20},
        ],
        "strategy": "maximize_sharpe",
        "constraints": {"min_dividend_yield": 2.5, "min_weight": 5, "max_weight": 40},
    })
    assert resp.status_code == 200
    data = resp.json()["allocation_changes"]
    yields = {"IEFA": 3.278, "GLD": 0.0, "AGG": 3.974, "VEA": 2.034, "SPY": 0.987}
    portfolio_yield = sum((a["optimized_weight"] / 100.0) * yields[a["ticker"]] for a in data)
    assert portfolio_yield >= 2.5 - 0.05


def test_impossible_dividend_yield_constraint_returns_422():
    resp = optimize({
        "securities": [{"ticker": "GLD", "current_weight": 50}, {"ticker": "AGG", "current_weight": 50}],
        "strategy": "maximize_sharpe",
        "constraints": {"min_dividend_yield": 10},  # way above what AGG+GLD can produce
    })
    assert resp.status_code == 422


def test_minimize_volatility_beats_equal_weight_baseline():
    securities = [
        {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
        {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
        {"ticker": "SPY", "current_weight": 20},
    ]
    eq = optimize({"securities": securities, "strategy": "equal_weights"}).json()
    mv = optimize({"securities": securities, "strategy": "minimize_volatility"}).json()

    from app.data import get_store
    from app import stats
    import numpy as np

    store = get_store()
    tickers = [s["ticker"] for s in eq["allocation_changes"]]
    returns = store.aligned_returns(tickers)[tickers]

    w_eq = np.array([a["optimized_weight"] / 100.0 for a in eq["allocation_changes"]])
    w_mv = np.array([a["optimized_weight"] / 100.0 for a in mv["allocation_changes"]])

    assert stats.annualized_volatility(returns, w_mv) <= stats.annualized_volatility(returns, w_eq) + 1e-6


def test_maximize_sharpe_runs_and_sums_to_100():
    resp = optimize({
        "securities": [
            {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
            {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
            {"ticker": "SPY", "current_weight": 20},
        ],
        "strategy": "maximize_sharpe",
    })
    assert resp.status_code == 200
    weights = [a["optimized_weight"] for a in resp.json()["allocation_changes"]]
    assert abs(sum(weights) - 100.0) < 0.01
    assert all(w >= -0.01 for w in weights)


def test_factor_exposure_bonus_increases_momentum():
    securities = [
        {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
        {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
        {"ticker": "SPY", "current_weight": 20},
    ]
    resp = optimize({
        "securities": securities,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "momentum", "direction": "maximize"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["factor_betas"] is not None
    assert data["factor_betas"]["optimized_portfolio"]["momentum"] >= data["factor_betas"]["current_portfolio"]["momentum"]
    weights = [a["optimized_weight"] for a in data["allocation_changes"]]
    assert abs(sum(weights) - 100.0) < 0.01
    assert all(w >= -0.01 for w in weights)


# --- Helpers for the numerical checks below -------------------------------

import numpy as np
from scipy.optimize import linprog

from app import factors, stats
from app.data import get_store

FIVE_FUNDS = [
    {"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20},
    {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20},
    {"ticker": "SPY", "current_weight": 20},
]
DIVIDEND_YIELDS = {"IEFA": 0.03278, "GLD": 0.0, "AGG": 0.03974, "VEA": 0.02034, "SPY": 0.00987}


def weights_and_returns(data):
    """Fractional weight vector from a response, plus the matching aligned returns."""
    tickers = [a["ticker"] for a in data["allocation_changes"]]
    w = np.array([a["optimized_weight"] / 100.0 for a in data["allocation_changes"]])
    return tickers, w, get_store().aligned_returns(tickers)[tickers]


# --- Rounding --------------------------------------------------------------

THREE_FUNDS = [
    {"ticker": "SPY", "current_weight": 60}, {"ticker": "AGG", "current_weight": 30},
    {"ticker": "GLD", "current_weight": 10},
]


@pytest.mark.parametrize("securities,strategy", [
    (THREE_FUNDS, "equal_weights"),        # 3 x 33.33 used to sum to 99.99
    (THREE_FUNDS, "minimize_volatility"),  # used to sum to 99.99
    (FIVE_FUNDS, "maximize_sharpe"),       # used to sum to 100.01
    (FIVE_FUNDS, "risk_parity"),
    (FIVE_FUNDS, "minimize_drawdown"),
])
def test_optimized_weights_sum_to_exactly_100(securities, strategy):
    resp = optimize({"securities": securities, "strategy": strategy})
    assert resp.status_code == 200
    weights = [a["optimized_weight"] for a in resp.json()["allocation_changes"]]
    assert round(sum(weights), 2) == 100.00
    assert all(round(w, 2) == w for w in weights)


# --- Risk parity -----------------------------------------------------------

def test_risk_parity_two_assets_matches_inverse_volatility():
    # With two assets, equal risk contribution reduces to w_i proportional to 1/sigma_i.
    resp = optimize({
        "securities": [{"ticker": "VEA", "current_weight": 25}, {"ticker": "AGG", "current_weight": 75}],
        "strategy": "risk_parity",
    })
    assert resp.status_code == 200
    _, w, returns = weights_and_returns(resp.json())
    inv_vol = 1.0 / returns.std().to_numpy()
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(w, expected, atol=0.0005)


def test_risk_parity_five_assets_equalizes_risk_contributions():
    resp = optimize({"securities": FIVE_FUNDS, "strategy": "risk_parity"})
    assert resp.status_code == 200
    _, w, returns = weights_and_returns(resp.json())
    rc = stats.risk_contributions(returns, w)
    shares = rc / rc.sum()
    # Each security should carry ~20% of total risk (was 17.6%-24.5% before the fix).
    assert np.allclose(shares, 0.2, atol=0.002)


# --- Minimize drawdown -----------------------------------------------------

def test_minimize_drawdown_beats_baselines():
    resp = optimize({"securities": FIVE_FUNDS, "strategy": "minimize_drawdown"})
    assert resp.status_code == 200
    _, w, returns = weights_and_returns(resp.json())
    optimized_dd = stats.max_drawdown(returns, w)

    assert optimized_dd <= stats.max_drawdown(returns, np.full(5, 0.2)) + 1e-6
    for i in range(5):
        assert optimized_dd <= stats.max_drawdown(returns, np.eye(5)[i]) + 1e-6
    rng = np.random.default_rng(0)
    for candidate in rng.dirichlet(np.ones(5), 500):
        assert optimized_dd <= stats.max_drawdown(returns, candidate) + 1e-4


def test_minimize_drawdown_respects_bounds():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "minimize_drawdown",
        "constraints": {"min_weight": 10, "max_weight": 30},
    })
    assert resp.status_code == 200
    weights = [a["optimized_weight"] for a in resp.json()["allocation_changes"]]
    assert all(10 - 0.01 <= w <= 30 + 0.01 for w in weights)


# --- Factor exposure (bonus) -----------------------------------------------

def _reference_factor_optimum(tickers, factor, maximize, bounds=None, min_yield=None):
    """Independent LP solution: portfolio beta = w @ per-fund betas."""
    store = get_store()
    returns = store.aligned_returns(tickers)[tickers]
    n = len(tickers)
    betas = np.array([factors.compute_betas(store, returns, np.eye(n)[i])[factor] for i in range(n)])
    kwargs = {}
    if min_yield is not None:
        kwargs = {"A_ub": [[-DIVIDEND_YIELDS[t] for t in tickers]], "b_ub": [-min_yield]}
    res = linprog(-betas if maximize else betas, A_eq=[np.ones(n)], b_eq=[1],
                  bounds=bounds or [(0, 1)] * n, **kwargs)
    return float(res.x @ betas)


def test_factor_exposure_maximize_momentum_hits_true_optimum():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "momentum", "direction": "maximize"},
    })
    assert resp.status_code == 200
    data = resp.json()
    tickers = [a["ticker"] for a in data["allocation_changes"]]
    best_beta = _reference_factor_optimum(tickers, "momentum", maximize=True)

    # The *reported* 3-factor momentum beta must be the best achievable one
    # (100% VEA, ~0.187), not just "higher than before".
    assert data["factor_betas"]["optimized_portfolio"]["momentum"] == pytest.approx(best_beta, abs=1e-3)
    weights = {a["ticker"]: a["optimized_weight"] for a in data["allocation_changes"]}
    assert weights["VEA"] == pytest.approx(100.0, abs=0.05)


def test_factor_exposure_with_case5_constraints_hits_true_optimum():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "momentum", "direction": "maximize"},
        "constraints": {"min_dividend_yield": 2.5, "min_weight": 5, "max_weight": 40},
    })
    assert resp.status_code == 200
    data = resp.json()
    tickers, w, _ = weights_and_returns(data)
    best_beta = _reference_factor_optimum(
        tickers, "momentum", maximize=True, bounds=[(0.05, 0.40)] * 5, min_yield=0.025
    )
    optimized = data["factor_betas"]["optimized_portfolio"]["momentum"]
    assert optimized == pytest.approx(best_beta, abs=1e-3)  # ~0.162
    # Previously "maximize" *lowered* momentum under these constraints.
    assert optimized > data["factor_betas"]["current_portfolio"]["momentum"]
    assert all(0.05 - 1e-4 <= wi <= 0.40 + 1e-4 for wi in w)
    assert sum(wi * DIVIDEND_YIELDS[t] for t, wi in zip(tickers, w)) >= 0.025 - 1e-4


def test_factor_exposure_minimize_value_hits_true_optimum():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "value", "direction": "minimize"},
    })
    assert resp.status_code == 200
    data = resp.json()
    tickers = [a["ticker"] for a in data["allocation_changes"]]
    best_beta = _reference_factor_optimum(tickers, "value", maximize=False)
    assert data["factor_betas"]["optimized_portfolio"]["value"] == pytest.approx(best_beta, abs=1e-3)


# --- security_constraints validation ----------------------------------------

def test_security_constraint_keys_are_case_insensitive():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "maximize_sharpe",
        "constraints": {"security_constraints": {"gld": {"max_weight": 5}}},
    })
    assert resp.status_code == 200
    weights = {a["ticker"]: a["optimized_weight"] for a in resp.json()["allocation_changes"]}
    assert weights["GLD"] <= 5.0 + 0.01  # unconstrained optimum holds ~20% GLD


def test_security_constraint_for_ticker_not_in_request_returns_422():
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "maximize_sharpe",
        "constraints": {"security_constraints": {"TSLA": {"max_weight": 5}}},
    })
    assert resp.status_code == 422
    assert "TSLA" in resp.json()["detail"]


@pytest.mark.parametrize("bound", [{"min_weight": -50}, {"max_weight": 500}])
def test_security_constraint_out_of_range_returns_422(bound):
    resp = optimize({
        "securities": FIVE_FUNDS,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "momentum"},
        "constraints": {"security_constraints": {"AGG": bound}},
    })
    assert resp.status_code == 422


# --- Non-finite input --------------------------------------------------------

def _post_raw(body: str):
    # Raw body: Python's json module will happily emit/parse NaN and Infinity.
    return client.post("/optimize", content=body, headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_weight_returns_422_not_500(bad):
    resp = _post_raw(
        '{"securities":[{"ticker":"SPY","current_weight":' + bad + '},'
        '{"ticker":"AGG","current_weight":50}],"strategy":"equal_weights"}'
    )
    assert resp.status_code == 422


def test_non_finite_constraint_returns_422_not_500():
    resp = _post_raw(
        '{"securities":[{"ticker":"SPY","current_weight":50},{"ticker":"AGG","current_weight":50}],'
        '"strategy":"maximize_sharpe","constraints":{"min_dividend_yield":NaN}}'
    )
    assert resp.status_code == 422
