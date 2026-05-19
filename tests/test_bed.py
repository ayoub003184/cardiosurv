"""
tests/test_bed.py
-----------------
Unit tests for src/clinical/bed.py.

All expected BED values are derived from the closed-form formula:
    BED = D_total × (1 + d / (α/β))     where d = D_total / n_fractions

Reference values are cross-checked against:
  • Fowler JF (1989). The linear-quadratic formula.  Br J Radiol 62:679-694.
  • RTOG 0236 trial dosimetry (54 Gy / 3 fx → BED = 151.2 Gy, α/β = 10)
  • Standard cardiac SBRT literature (25 Gy / 1 fx → BED = 87.5 Gy)
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clinical.bed import (
    compute_bed,
    validate_sbrt_bed,
    BEDResult,
    DEFAULT_ALPHA_BETA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expected_bed(total_dose: float, n_fractions: int, ab: float = 10.0) -> float:
    """Ground-truth BED from first principles."""
    d = total_dose / n_fractions
    return total_dose * (1.0 + d / ab)


def expected_eqd2(total_dose: float, n_fractions: int, ab: float = 10.0) -> float:
    bed = expected_bed(total_dose, n_fractions, ab)
    return bed / (1.0 + 2.0 / ab)


# ---------------------------------------------------------------------------
# 1. Formula correctness — known clinical reference values
# ---------------------------------------------------------------------------

class TestBEDFormula:
    """BED = D × (1 + d/α/β) verified against published literature."""

    def test_cardiac_sbrt_single_fraction_25gy(self):
        """25 Gy / 1 fx → BED = 87.5 Gy  (standard cardiac SBRT, α/β=10)."""
        r = compute_bed(25.0, 1)
        assert pytest.approx(r.bed_gy, abs=0.01) == 87.5

    def test_rtog0236_54gy_3fx(self):
        """54 Gy / 3 fx → BED = 151.2 Gy  (RTOG 0236 lung SBRT, α/β=10)."""
        r = compute_bed(54.0, 3)
        assert pytest.approx(r.bed_gy, abs=0.01) == 151.2

    def test_conventional_fractionation_60gy_30fx(self):
        """60 Gy / 30 fx → BED = 72.0 Gy  (2 Gy/fx conventional RT)."""
        r = compute_bed(60.0, 30)
        assert pytest.approx(r.bed_gy, abs=0.01) == 72.0

    def test_hypofractionation_45gy_5fx(self):
        """45 Gy / 5 fx (9 Gy/fx) → BED = 45*(1+9/10) = 85.5 Gy."""
        r = compute_bed(45.0, 5)
        assert pytest.approx(r.bed_gy, abs=0.01) == 85.5

    def test_custom_alpha_beta_3(self):
        """Late-responding tissue: α/β = 3 Gy.  60 Gy / 30 fx → BED = 100.0 Gy."""
        r = compute_bed(60.0, 30, alpha_beta_gy=3.0)
        # d = 2 Gy;  BED = 60*(1 + 2/3) = 100.0
        assert pytest.approx(r.bed_gy, abs=0.01) == 100.0

    def test_bed_equals_formula_for_arbitrary_inputs(self):
        """BED matches closed-form formula for a range of dose/fraction combos."""
        cases = [
            (20.0, 4,  10.0),
            (30.0, 5,  10.0),
            (48.0, 8,  10.0),
            (70.0, 35,  3.0),
        ]
        for D, n, ab in cases:
            r = compute_bed(D, n, ab)
            assert pytest.approx(r.bed_gy, rel=1e-5) == expected_bed(D, n, ab), \
                f"Failed for D={D}, n={n}, ab={ab}"

    def test_dose_per_fraction_stored_correctly(self):
        """dose_per_fraction_gy = total_dose / n_fractions."""
        r = compute_bed(30.0, 5)
        assert pytest.approx(r.dose_per_fraction_gy, abs=1e-4) == 6.0

    def test_eqd2_formula(self):
        """EQD2 = BED / (1 + 2/α/β)."""
        r = compute_bed(54.0, 3)
        assert pytest.approx(r.eqd2_gy, abs=0.01) == expected_eqd2(54.0, 3)

    def test_single_fraction_eqd2_equals_bed_over_1p2ab(self):
        """For 1 fx: EQD2 = BED / 1.2  when α/β=10."""
        r = compute_bed(25.0, 1)
        assert pytest.approx(r.eqd2_gy, abs=0.01) == r.bed_gy / 1.2

    def test_default_alpha_beta_is_10(self):
        """DEFAULT_ALPHA_BETA must be 10.0 Gy as specified in the flowchart."""
        assert DEFAULT_ALPHA_BETA == 10.0

    def test_result_stores_all_inputs(self):
        """BEDResult echoes back every input unchanged."""
        r = compute_bed(36.0, 6, alpha_beta_gy=10.0)
        assert r.total_dose_gy == 36.0
        assert r.n_fractions == 6
        assert r.alpha_beta_gy == 10.0


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------

class TestBEDValidation:

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="total_dose_gy"):
            compute_bed(0.0, 5)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            compute_bed(-10.0, 3)

    def test_zero_fractions_raises(self):
        with pytest.raises(ValueError, match="n_fractions"):
            compute_bed(30.0, 0)

    def test_negative_fractions_raises(self):
        with pytest.raises(ValueError):
            compute_bed(30.0, -1)

    def test_zero_alpha_beta_raises(self):
        with pytest.raises(ValueError, match="alpha_beta_gy"):
            compute_bed(30.0, 5, alpha_beta_gy=0.0)

    def test_negative_alpha_beta_raises(self):
        with pytest.raises(ValueError):
            compute_bed(30.0, 5, alpha_beta_gy=-5.0)


# ---------------------------------------------------------------------------
# 3. validate_sbrt_bed
# ---------------------------------------------------------------------------

class TestValidateSBRTBed:

    def test_25gy_1fx_fails_ablative_threshold(self):
        """25 Gy / 1 fx → BED 87.5 Gy < 100 Gy ablative threshold → FAIL."""
        _, valid, msg = validate_sbrt_bed(25.0, 1)
        assert not valid
        assert "FAIL" in msg

    def test_34gy_1fx_passes_ablative_threshold(self):
        """34 Gy / 1 fx → BED = 34*(1+34/10) = 149.6 Gy ≥ 100 Gy → PASS."""
        _, valid, msg = validate_sbrt_bed(34.0, 1)
        assert valid
        assert "PASS" in msg

    def test_54gy_3fx_passes(self):
        """54 Gy / 3 fx → BED = 151.2 Gy → PASS."""
        _, valid, _ = validate_sbrt_bed(54.0, 3)
        assert valid

    def test_custom_threshold(self):
        """Custom min_bed_gy threshold is respected."""
        _, valid, _ = validate_sbrt_bed(25.0, 1, min_bed_gy=80.0)
        assert valid   # 87.5 ≥ 80

    def test_returns_bed_result_object(self):
        """validate_sbrt_bed must return a BEDResult as first element."""
        result, _, _ = validate_sbrt_bed(54.0, 3)
        assert isinstance(result, BEDResult)

    def test_message_contains_gy_value(self):
        """Message must report the BED value in Gy."""
        _, _, msg = validate_sbrt_bed(25.0, 1)
        assert "87.5" in msg or "87" in msg


# ---------------------------------------------------------------------------
# 4. Return-type contract
# ---------------------------------------------------------------------------

class TestBEDReturnType:

    def test_returns_bed_result(self):
        assert isinstance(compute_bed(25.0, 1), BEDResult)

    def test_bed_gy_is_positive(self):
        r = compute_bed(10.0, 2)
        assert r.bed_gy > 0

    def test_eqd2_less_than_bed_for_high_dose_per_fraction(self):
        """When d > 2 Gy, EQD2 < BED."""
        r = compute_bed(30.0, 3)  # d = 10 Gy > 2 Gy
        assert r.eqd2_gy < r.bed_gy

    def test_eqd2_equals_bed_when_d_is_2gy(self):
        """When d == 2 Gy (conventional), EQD2 == BED / 1.2 and BED == D * 1.2."""
        r = compute_bed(60.0, 30)  # d = 2 Gy exactly
        assert pytest.approx(r.eqd2_gy, abs=0.01) == 60.0  # EQD2 = total dose
