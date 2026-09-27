"""
A single exception type for every business-rule failure (unknown ticker,
bad weights, infeasible constraints, insufficient history, ...). main.py
catches this once and turns it into a clean HTTP 422 response instead of
letting the optimizer blow up with a cryptic scipy/numpy traceback.
"""


class OptimizationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
