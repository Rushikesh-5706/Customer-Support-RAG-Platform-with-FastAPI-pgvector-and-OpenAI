"""
pytest configuration for IntelliSupport.

Patches the OpenAI client and psycopg2 at module import time so that no
live API or database connections are required when running the test suite.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=False)
def mock_openai():
    """Patch OpenAI at the top-level so no real API calls are made."""
    with patch("openai.OpenAI") as mock:
        yield mock


@pytest.fixture(autouse=False)
def mock_psycopg2():
    """Patch psycopg2.connect so no real DB connections are made."""
    with patch("psycopg2.connect") as mock:
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = None
        conn.cursor.return_value.__enter__.return_value = cursor
        mock.return_value = conn
        yield mock
