"""
tests/test_features.py
----------------------
Unit tests for feature engineering functions in src/features/engineering.py.

Run with:
    pytest tests/test_features.py -v
"""

import pytest
from src.features.engineering import age_bin, bp_risk_level, hr_stress_index


# ---------------------------------------------------------------------------
# age_bin tests
# ---------------------------------------------------------------------------

def test_age_bin_under_40():
    assert age_bin(35) == "<40"
    assert age_bin(39) == "<40"


def test_age_bin_40s():
    assert age_bin(40) == "40-49"
    assert age_bin(49) == "40-49"


def test_age_bin_50s():
    assert age_bin(50) == "50-59"
    assert age_bin(59) == "50-59"


def test_age_bin_60s():
    assert age_bin(60) == "60-69"
    assert age_bin(69) == "60-69"


def test_age_bin_70_plus():
    assert age_bin(70) == "70+"
    assert age_bin(85) == "70+"


# ---------------------------------------------------------------------------
# bp_risk_level tests (AHA categories)
# ---------------------------------------------------------------------------

def test_bp_risk_normal():
    assert bp_risk_level(110) == "Normal"
    assert bp_risk_level(119) == "Normal"


def test_bp_risk_elevated():
    assert bp_risk_level(120) == "Elevated"
    assert bp_risk_level(129) == "Elevated"


def test_bp_risk_stage1():
    assert bp_risk_level(130) == "Stage1"
    assert bp_risk_level(139) == "Stage1"


def test_bp_risk_stage2():
    assert bp_risk_level(140) == "Stage2"
    assert bp_risk_level(170) == "Stage2"


def test_bp_risk_crisis():
    assert bp_risk_level(180) == "Crisis"
    assert bp_risk_level(200) == "Crisis"


# ---------------------------------------------------------------------------
# hr_stress_index tests (MaxHR / (220 - Age))
# ---------------------------------------------------------------------------

def test_hr_stress_canonical():
    # 50-year-old, max HR 170 -> 170 / (220-50) = 170/170 = 1.0
    assert hr_stress_index(170, 50) == 1.0


def test_hr_stress_below_max():
    # 60-year-old, max HR 100 -> 100 / 160 = 0.625
    result = hr_stress_index(100, 60)
    assert result == pytest.approx(0.625, abs=0.001)


def test_hr_stress_young_patient():
    # 30-year-old, max HR 180 -> 180 / 190 = 0.947
    result = hr_stress_index(180, 30)
    assert result == pytest.approx(0.947, abs=0.01)


def test_hr_stress_returns_float():
    result = hr_stress_index(150, 55)
    assert isinstance(result, float)
