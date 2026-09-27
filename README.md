# Portfolio Optimizer API

A REST API that replicates the core allocation logic of Finominal's
Portfolio Optimizer tool: given a set of securities, their current weights,
and a strategy, it returns optimized portfolio weights.

## What it does

`POST /optimize` accepts a list of securities (ticker + current weight),
an optimization strategy, and optional constraints, and returns the
optimized weights, the allocation change per security, and (for the bonus
factor-exposure strategy) factor betas for both the current and optimized
portfolio.

## Tech stack

- **FastAPI** + **Pydantic** — API layer and request/response validation
- **pandas** / **NumPy** — data loading, alignment, return statistics
- **SciPy** (`scipy.optimize.minimize`, SLSQP) — all constrained optimization
- No database, no auth, no background workers, no Docker/Kubernetes — none
  of that is needed for a stateless, single-endpoint optimizer over a small
  static dataset.

## Project structure

```
portfolio-optimizer/
├── app/
│   ├── main.py          # FastAPI app, the /optimize endpoint, error handling
│   ├── schemas.py        # Pydantic request/response models + input validation
│   ├── data.py           # Loads Data.xlsx once at startup, date-alignment helpers
│   ├── stats.py           # Pure portfolio math: vol, CAGR, Sharpe, drawdown, risk contributions
│   ├── constraints.py    # Builds scipy bounds/constraints, feasibility checks
│   ├── optimizer.py       # The 5 required strategies + the bonus factor-exposure strategy
│   ├── factors.py         # OLS regression for factor betas (bonus)
│   └── errors.py          # One exception type -> clean HTTP 422 responses
├── data/Data.xlsx          # Provided fund/factor return data
├── tests/test_optimizer.py
├── requirements.txt
└── README.md
```

## Setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

The API is then available at `http://127.0.0.1:8000`, with interactive docs
at `http://127.0.0.1:8000/docs`.

## API

### `POST /optimize`

**Request**

```json
{
  "securities": [
    { "ticker": "SPY", "current_weight": 60 },
    { "ticker": "AGG", "current_weight": 30 },
    { "ticker": "GLD", "current_weight": 10 }
  ],
  "strategy": "minimize_volatility",
  "constraints": {
    "min_weight": 5,
    "max_weight": 40,
    "min_dividend_yield": 2.5,
    "min_cagr": 3,
    "volatility_range": { "min": 2, "max": 15 },
    "max_drawdown": 20,
    "security_constraints": {
      "GLD": { "min_weight": 0, "max_weight": 20 }
    }
  }
}
```

All weight/percentage fields are in **percent** (e.g. `60` for 60%), not
fractions. `constraints` is entirely optional; omit any field you don't
need. `security_constraints` lets you override the global `min_weight` /
`max_weight` for individual tickers.

For `strategy: "optimize_factor_exposure"` (bonus), also pass:

```json
"factor_objective": { "factor": "momentum", "direction": "maximize" }
```

**Response**

```json
{
  "optimization_strategy": "minimize_volatility",
  "allocation_changes": [
    {
      "ticker": "AGG",
      "security_name": "iShares Core US Aggregate Bond ETF",
      "current_weight": 30.0,
      "optimized_weight": 91.21,
      "change": 61.21
    }
  ],
  "factor_betas": null
}
```

`factor_betas` is populated only for `optimize_factor_exposure`:

```json
"factor_betas": {
  "current_portfolio": { "momentum": 0.13, "value": 0.17, "size": -0.06 },
  "optimized_portfolio": { "momentum": 0.19, "value": 0.35, "size": -0.06 }
}
```

**Errors**: every business-rule failure (unknown ticker, unsupported
strategy, weights not summing to 100%, negative weight, infeasible
constraints, insufficient overlapping historical data) returns **HTTP 422**
with a `{"detail": "..."}` message — never a silently-wrong portfolio or a
raw stack trace. Validation errors name the offending field (e.g.
`constraints.security_constraints.AGG.min_weight: Input should be greater
than or equal to 0`), and non-finite numbers (`NaN`, `Infinity`) are
rejected.

## Supported strategies

| Strategy | Key |
|---|---|
| Equal Weights | `equal_weights` |
| Risk Parity | `risk_parity` |
| Minimize Drawdown | `minimize_drawdown` |
| Minimize Volatility | `minimize_volatility` |
| Maximize Sharpe Ratio | `maximize_sharpe` |
| Optimize Factor Exposure (bonus) | `optimize_factor_exposure` |

All (except Equal Weights, which is closed-form `1/N`) are solved with
`scipy.optimize.minimize(method="SLSQP")` over daily historical returns,
using an annualization factor of 252 trading days:

- **Risk Parity** minimizes the sum of squared deviations between each
  security's *share* of total risk (`w_i * (Σw)_i / w^T Σ w`) and `1/N`, so
  every security contributes equally to total portfolio variance. Shares
  are used rather than raw variance contributions because the latter are
  so small (~1e-3) that their squared deviations fall below the solver's
  tolerance and it stops before contributions actually equalize.
- **Minimize Volatility** minimizes annualized portfolio standard deviation
  (`sqrt(w^T Σ w)`).
- **Maximize Sharpe Ratio** maximizes `(annualized return − risk-free rate) /
  annualized volatility`, with **risk-free rate = 0%** (explicitly allowed
  by the assignment).
- **Minimize Drawdown** reconstructs the weighted daily portfolio return
  series, computes the cumulative wealth curve and its running peak, and
  minimizes the resulting maximum peak-to-trough drawdown.
- **Optimize Factor Exposure (bonus)** uses the 3-factor OLS model
  `portfolio_return = α + β₁·Momentum + β₂·Value + β₃·Size + ε` over the
  dates common to the portfolio and factor series, and maximizes/minimizes
  the chosen factor's beta. OLS coefficients are linear in the dependent
  variable, and a portfolio's return series is linear in its weights, so
  the portfolio beta is exactly `w · β_funds`. Each fund is regressed once
  and the optimizer works with that dot product: it is exact, fast, and
  the same model as the betas reported in the response. Because the
  objective is linear, the unconstrained optimum is a corner: 100% in
  the fund with the highest (or lowest) beta. Weight bounds and
  portfolio-level constraints produce a diversified answer.

## Supported constraints

- **Security-level `min_weight` / `max_weight`** (global, percent) — fed
  into SciPy as box bounds. Per-ticker overrides via `security_constraints`
  (keys are case-insensitive and must be tickers present in `securities`;
  all min/max values must be within 0–100).
- **`min_dividend_yield`** (portfolio-level, percent) — enforced as
  `sum(w_i * yield_i) >= min`.
- **`min_cagr`** (portfolio-level, percent) — enforced against the CAGR of
  the weighted historical return series.
- **`volatility_range`** (`{min, max}`, percent) — enforced against
  annualized portfolio volatility.
- **`max_drawdown`** (percent) — enforced against the portfolio's maximum
  historical drawdown.

**Feasibility handling.**

- **Upfront pre-checks** run before the optimizer, for constraints whose
  feasibility can be decided exactly and cheaply:
  - Can the per-security min/max bounds sum to 100%?
  - Can the best achievable allocation within those bounds reach
    `min_dividend_yield`? This is linear, so it's a greedy fill.

  These return a specific 422 message, e.g. "even the best achievable
  allocation ... yields 3.15%, below the required minimum of 3.90%".
- **`min_cagr`, `volatility_range` and `max_drawdown` are *not*
  pre-checked.** Deciding whether they are reachable is itself an
  optimization problem, since they are non-linear in the weights. An
  impossible value is caught when SLSQP fails to find a feasible point,
  and the API returns a 422, "Could not find a feasible portfolio
  satisfying all constraints", with the solver's message. That message
  does not say *which* constraint was impossible.
- **Post-solve check:** every returned portfolio is re-validated against
  every constraint within a small numerical tolerance. A non-converged or
  slightly-violating result is returned as a 422, never as a portfolio.

## Important implementation decisions

- **GLD's missing dividend yield.** The source data has `NaN` for GLD's
  dividend yield. This isn't a data-quality gap — GLD is a physical gold
  ETF and structurally pays no distributions. It is therefore treated as a
  **true zero** (`0.0`) everywhere a portfolio-level dividend yield is
  computed, never silently invented or estimated. This is called out here
  rather than hidden.
- **Historical date alignment is per-request, not fixed.** The five funds
  have different inception dates (SPY from 1993, AGG 2003, GLD 2004, VEA
  2007, IEFA 2012). Rather than forcing every request onto one fixed
  calendar window, each request aligns on the **intersection of dates**
  available for exactly the securities it asked for (inner join). A
  2-security request (e.g. VEA + AGG) therefore uses more history than a
  request that includes IEFA. This maximizes the genuinely overlapping,
  comparable data used for each calculation. Data was verified to have zero
  nulls and zero duplicate `(date, ticker)` rows in both `Fund Returns` and
  `Factor Returns`.
- **Risk-free rate = 0%** for Sharpe Ratio, as explicitly permitted by the
  assignment.
- **Annualization**: daily returns are annualized using 252 trading days
  (standard convention) for volatility, return, and CAGR.
- **Equal Weights and constraints.** Equal Weights is a fixed `1/N`
  baseline, not something the optimizer searches over. If the supplied
  constraints are incompatible with equal weighting (e.g. a `min_weight`
  that a `1/N` allocation can't satisfy, or a dividend-yield floor the
  equal-weight portfolio doesn't clear), the API returns a 422 rather than
  silently returning a constraint-violating "equal weight" portfolio.
- **Rounding**: output weights are rounded to 2 decimal places using the
  **largest-remainder method**, so they always sum to exactly 100.00.
  Naive rounding can give 99.99 or 100.01, e.g. three equal weights round
  to 3 × 33.33. `change = optimized_weight - current_weight` is computed
  from the rounded values for internal consistency with what's displayed.

## Known limitations

- The bonus Factor Exposure strategy uses only the three factors provided
  (Momentum, Value, Size). The assignment explicitly notes the live
  Finominal tool uses a broader internal factor model, so exact beta
  matches aren't expected — the goal is a defensible, correct three-factor
  regression, not a re-creation of their proprietary model.
- `min_cagr`, `volatility_range`, and `max_drawdown` portfolio-level
  constraints aren't part of any of the 6 required test scenarios. They are
  implemented because the assignment lists them as constraint types the API
  should support, and were checked manually. They are not yet covered by
  automated tests, and they are not pre-checked for feasibility (see
  above).
- Strategies other than Equal Weights require at least 2 securities (a
  single-security "optimization" isn't a meaningful problem).
- No persistence — every request is computed fresh from the in-memory data
  loaded once at startup; there's nothing to keep in a database.

## Testing

```bash
pytest tests/ -v
```

32 tests cover:

- **Input validation:** unknown ticker, invalid strategy, negative or
  non-finite (`NaN`/`Infinity`) weights, weights not summing to 100%, and
  `security_constraints` handling (case-insensitive keys, unknown tickers
  rejected, out-of-range bounds rejected).
- **Constraints:** per-security min/max enforcement, infeasible bounds, and
  minimum dividend yield, including the GLD = 0% case and an impossible
  floor.
- **Rounding:** optimized weights sum to exactly 100.00 across strategies.
- **Strategy correctness, checked against independent references rather
  than just "it ran":**
  - Minimize Volatility beats equal weight.
  - 2-asset Risk Parity matches the closed-form inverse-volatility
    weights.
  - 5-asset Risk Parity gives every security 20% of total risk.
  - Minimize Drawdown beats equal weight, every single fund, and 500
    random portfolios, and respects bounds.
  - Factor exposure hits the true optimum from an independent linear
    program, both unconstrained (100% VEA, momentum β ≈ 0.187) and under
    Case 5's constraints (β ≈ 0.162).

## Validating against the live tool

The 6 required scenarios from the assignment were run against this API
(see below) and all satisfy the acceptance checks (weights sum to 100%,
non-negative, security/portfolio constraints respected, infeasible
constraints rejected with a clear error, factor exposure increases in the
bonus case). Cross-checking the exact numbers against Finominal's live
tool (https://finominal.com/portfolio-optimizer/US) and capturing
screenshots is a manual step to do against the running app.

Example (Case 5 — Maximize Sharpe with Min Dividend Yield 2.5% and 5%/40%
per-security bounds on 5 equal-weighted funds):

```bash
curl -X POST http://127.0.0.1:8000/optimize -H "Content-Type: application/json" -d '{
  "securities": [
    {"ticker":"IEFA","current_weight":20},{"ticker":"GLD","current_weight":20},
    {"ticker":"AGG","current_weight":20},{"ticker":"VEA","current_weight":20},
    {"ticker":"SPY","current_weight":20}
  ],
  "strategy": "maximize_sharpe",
  "constraints": {"min_dividend_yield": 2.5, "min_weight": 5, "max_weight": 40}
}'
```
