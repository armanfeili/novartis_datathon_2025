"""
Pytest configuration for Novartis Datathon 2025 tests.

This file configures pytest behavior including warning filters.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with warning filters."""
    # Filter known warnings from external libraries that we cannot fix
    # Torch/NumPy initialization warning (external library compiled with different numpy version)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Failed to initialize NumPy.*:UserWarning"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*_ARRAY_API not found.*:UserWarning"
    )
    # Module compiled with NumPy 1.x warning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*module that was compiled using NumPy 1.x.*:UserWarning"
    )
    # A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
    config.addinivalue_line(
        "filterwarnings",
        "ignore::UserWarning"
    )
