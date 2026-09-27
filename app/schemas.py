from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Reject NaN / +-Infinity in every numeric input: Python's JSON parser accepts
# them, and they would otherwise slip past range checks and crash the math.
_FINITE = ConfigDict(allow_inf_nan=False)


class Strategy(str, Enum):
    EQUAL_WEIGHTS = "equal_weights"
    RISK_PARITY = "risk_parity"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    OPTIMIZE_FACTOR_EXPOSURE = "optimize_factor_exposure"  # bonus


class Factor(str, Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SecurityInput(BaseModel):
    model_config = _FINITE

    ticker: str
    current_weight: float = Field(..., description="Percentage, e.g. 60 for 60%")

    @field_validator("current_weight")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("current_weight cannot be negative (no short selling)")
        return v

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.strip().upper()


class SecurityWeightBound(BaseModel):
    """Per-ticker override of the global min/max weight, both in percent."""
    model_config = _FINITE

    min_weight: Optional[float] = Field(None, ge=0, le=100)
    max_weight: Optional[float] = Field(None, ge=0, le=100)


class VolatilityRange(BaseModel):
    model_config = _FINITE

    min: Optional[float] = None  # annualized volatility, percent
    max: Optional[float] = None


class Constraints(BaseModel):
    model_config = _FINITE

    min_weight: Optional[float] = Field(None, ge=0, le=100, description="Global per-security minimum, percent")
    max_weight: Optional[float] = Field(None, ge=0, le=100, description="Global per-security maximum, percent")
    security_constraints: Optional[dict[str, SecurityWeightBound]] = Field(
        None, description="Per-ticker min/max weight overrides, percent"
    )
    min_dividend_yield: Optional[float] = Field(None, description="Percent")
    min_cagr: Optional[float] = Field(None, description="Percent")
    volatility_range: Optional[VolatilityRange] = None
    max_drawdown: Optional[float] = Field(None, description="Percent, e.g. 15 means <=15% drawdown")

    @field_validator("security_constraints")
    @classmethod
    def normalize_tickers(cls, v):
        # Match SecurityInput's normalization so {"gld": ...} applies to GLD
        # instead of being silently ignored.
        if v is None:
            return v
        normalized = {}
        for ticker, bound in v.items():
            key = ticker.strip().upper()
            if key in normalized:
                raise ValueError(f"security_constraints lists {key} more than once")
            normalized[key] = bound
        return normalized

    @model_validator(mode="after")
    def check_bounds(self):
        if self.min_weight is not None and self.max_weight is not None:
            if self.min_weight > self.max_weight:
                raise ValueError("constraints.min_weight cannot exceed constraints.max_weight")
        if self.security_constraints:
            for ticker, b in self.security_constraints.items():
                if b.min_weight is not None and b.max_weight is not None and b.min_weight > b.max_weight:
                    raise ValueError(f"security_constraints[{ticker}]: min_weight cannot exceed max_weight")
        if self.volatility_range and self.volatility_range.min is not None and self.volatility_range.max is not None:
            if self.volatility_range.min > self.volatility_range.max:
                raise ValueError("volatility_range.min cannot exceed volatility_range.max")
        return self


class FactorObjective(BaseModel):
    factor: Factor
    direction: Direction = Direction.MAXIMIZE


class OptimizeRequest(BaseModel):
    securities: list[SecurityInput]
    strategy: Strategy
    constraints: Optional[Constraints] = None
    factor_objective: Optional[FactorObjective] = None

    @model_validator(mode="after")
    def validate_securities(self):
        if len(self.securities) < 1:
            raise ValueError("At least one security is required")

        tickers = [s.ticker for s in self.securities]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers in securities list")

        total = sum(s.current_weight for s in self.securities)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"current_weight values must sum to 100 (got {total})")

        if self.constraints and self.constraints.security_constraints:
            unknown = sorted(set(self.constraints.security_constraints) - set(tickers))
            if unknown:
                raise ValueError(f"security_constraints references ticker(s) not in securities: {', '.join(unknown)}")

        if self.strategy == Strategy.OPTIMIZE_FACTOR_EXPOSURE and self.factor_objective is None:
            raise ValueError("factor_objective is required when strategy is optimize_factor_exposure")

        return self


class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class FactorBetas(BaseModel):
    momentum: float
    value: float
    size: float


class FactorBetasResponse(BaseModel):
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas


class OptimizeResponse(BaseModel):
    optimization_strategy: Strategy
    allocation_changes: list[AllocationChange]
    factor_betas: Optional[FactorBetasResponse] = None
