"""
Loads Data.xlsx once at process startup and exposes clean, in-memory
representations of:
  - fund metadata (name, dividend yield)
  - a wide fund-returns matrix (date index x ticker columns)
  - a wide factor-returns matrix (date index x factor columns)

All downstream code should go through `get_store()` rather than
re-reading the Excel file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Data.xlsx")

# Friendly, lowercase keys for the three factors as they'll appear in the API.
FACTOR_NAME_MAP = {
    "Momentum Factor": "momentum",
    "Value Factor": "value",
    "Size Factor": "size",
}


@dataclass(frozen=True)
class FundMeta:
    ticker: str
    security_name: str
    dividend_yield: float | None  # None when not disclosed (e.g. GLD)


class DataStore:
    def __init__(self, path: str = DATA_PATH):
        fund_info = pd.read_excel(path, sheet_name="Fund Info")
        fund_returns = pd.read_excel(path, sheet_name="Fund Returns")
        factor_returns = pd.read_excel(path, sheet_name="Factor Returns")

        # --- Fund metadata ---
        self.funds: dict[str, FundMeta] = {}
        for _, row in fund_info.iterrows():
            dy = row["dividend_yield"]
            self.funds[row["ticker"]] = FundMeta(
                ticker=row["ticker"],
                security_name=row["fund_name"],
                dividend_yield=None if pd.isna(dy) else float(dy),
            )

        # --- Wide fund returns matrix: index=date, columns=ticker ---
        self.fund_returns = (
            fund_returns.pivot(index="date", columns="ticker", values="total_return")
            .sort_index()
        )

        # --- Wide factor returns matrix: index=date, columns=momentum/value/size ---
        factor_returns = factor_returns.copy()
        factor_returns["factor"] = factor_returns["index_ticker"].map(FACTOR_NAME_MAP)
        self.factor_returns = (
            factor_returns.pivot(index="date", columns="factor", values="total_return")
            .sort_index()
        )

    def known_tickers(self) -> set[str]:
        return set(self.funds.keys())

    def get_meta(self, ticker: str) -> FundMeta:
        return self.funds[ticker]

    def aligned_returns(self, tickers: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame of daily returns for exactly `tickers`, restricted
        to the intersection of dates on which ALL of them have a value
        (inner join). Different security subsets can therefore span
        different historical windows -- this uses the maximum amount of
        genuinely overlapping data for whichever securities were requested,
        rather than forcing every request onto one fixed calendar window.
        """
        sub = self.fund_returns[tickers].dropna(how="any")
        return sub

    def aligned_with_factors(self, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Inner-joins a portfolio return series (indexed by date) with the
        factor returns matrix, on the common date range, for regression.
        """
        joined = self.factor_returns.join(portfolio_returns.rename("portfolio"), how="inner")
        return joined.dropna(how="any")


@lru_cache(maxsize=1)
def get_store() -> DataStore:
    return DataStore()
