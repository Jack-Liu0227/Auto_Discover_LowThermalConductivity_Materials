"""
Kappa Library - Thermal Conductivity Calculation Module
Modularized version of ai4kappa for MCP integration
"""

__version__ = "1.0.0"
__all__ = ["ThermalConductivityCalculator", "calculate_kappa_p", "calculate_kappa_mtp", "is_kappa_available"]

from .calculator import ThermalConductivityCalculator, calculate_kappa_p, calculate_kappa_mtp, is_kappa_available

