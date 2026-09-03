"""Smoke tests for package scaffold."""

import evt


def test_version() -> None:
    assert evt.__version__ == "0.1.0"
