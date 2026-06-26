"""Design of experiments: turn factors into the configurations to run."""

from cafe.design.factorial import Config, full_factorial, generate, single, size
from cafe.design.fractional import (
    FractionalDesign,
    fractional_factorial,
    fractional_factorial_design,
)

__all__ = [
    "Config",
    "generate",
    "size",
    "full_factorial",
    "single",
    "fractional_factorial",
    "fractional_factorial_design",
    "FractionalDesign",
]
