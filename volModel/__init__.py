"""
Volatility Model module for CleanIV_Correlation project.

This module contains volatility model implementations including SABR, SVI,
polynomial fits, and term-structure utilities.
"""

from .termFit import fit_term_structure, term_structure_iv

__all__ = [
    "fit_term_structure",
    "term_structure_iv",
]
