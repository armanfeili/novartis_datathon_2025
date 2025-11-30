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
    # Statsmodels FutureWarning about index support
    config.addinivalue_line(
        "filterwarnings",
        "ignore:No supported index is available.*:FutureWarning"
    )
    # Pandas FutureWarning for Series.__getitem__
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Series.__getitem__ treating keys as positions is deprecated.*:FutureWarning"
    )
    # Ignore all FutureWarnings from statsmodels
    config.addinivalue_line(
        "filterwarnings",
        "ignore::FutureWarning:statsmodels.*"
    )
    # Ignore DeprecationWarnings from external packages
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning"
    )
