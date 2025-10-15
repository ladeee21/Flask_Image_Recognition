
# conftest.py

"""Configuration file for pytest fixtures."""
import pytest
from app import app  # This imports the Flask app for testing

@pytest.fixture
def client():
    """Fixture for the Flask test client."""
    with app.test_client() as client:
        yield client
