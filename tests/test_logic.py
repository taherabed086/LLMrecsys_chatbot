"""Unit tests for the pure logic in recsys_chat (no browser, no API)."""

import pandas as pd

from recsys_chat.search import confidence_level


def test_high_confidence():
    assert confidence_level(0.6) == "high"


def test_medium_confidence():
    assert confidence_level(0.4) == "medium"


def test_low_confidence():
    assert confidence_level(0.1) == "low"


def test_no_confidence_for_zero_or_negative():
    assert confidence_level(0.0) == "none"
    assert confidence_level(-0.2) == "none"


def test_boundaries_belong_to_the_higher_level():
    # A score exactly on a threshold counts as that level
    assert confidence_level(0.55) == "high"
    assert confidence_level(0.35) == "medium"