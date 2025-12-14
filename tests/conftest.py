"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def model_path():
    """Path to model file"""
    return "models/weather_best_models.pkl"


@pytest.fixture(scope="session")
def data_path():
    """Path to data file"""
    return "data/weatherHistory.csv"


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "model: mark test as a model validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )