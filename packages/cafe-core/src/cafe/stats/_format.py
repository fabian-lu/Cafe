"""Shared formatting for the statistics tables: R-style significance codes."""

from __future__ import annotations

# The familiar R legend, printed under a coefficient/effects table.
SIG_LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"


def sig_code(p: float | None) -> str:
    """R's significance stars for a p-value ('***' / '**' / '*' / '.' / '')."""
    if p is None or p != p:  # None or NaN
        return " "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "
