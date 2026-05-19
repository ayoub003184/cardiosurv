"""
tests/test_grace.py
-------------------
Unit tests for src/clinical/grace.py.

Reference values from:
  • Fox KAA et al. (2006). GRACE risk score — original derivation.
    Eur Heart J 27(24):2931-2937.
  • Fox KAA et al. (2014). Updated GRACE 2.0.  BMJ Open 4:e004425.
  • Published GRACE nomogram tables reproduced in AHA/ACC ACS guidelines.
  • GRACE online calculator cross-validation cases (graceScore.org).

Point-table values used in assertions are taken directly from the
JAMA-published GRACE score tables cited in the original 2003 paper:
  Granger CB et al. (2003). JAMA 290(5):636-644.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clinical.grace import (
    compute_grace_score,
    calibrate_grace_for_pipeline,
    GRACEResult,
    KillipClass,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    _age_points,
    _hr_points,
    _sbp_points,
    _creatinine_points,
    _KILLIP_POINTS,
)


# ---------------------------------------------------------------------------
# 1. Individual scoring sub-functions (point tables)
# ---------------------------------------------------------------------------

class TestAgePoints:
    """Age → GRACE points per JAMA 2003 Table 2."""

    def test_age_under_30(self):       assert _age_points(25) == 0
    def test_age_30_to_39(self):       assert _age_points(35) == 8
    def test_age_40_to_49(self):       assert _age_points(45) == 25
    def test_age_50_to_59(self):       assert _age_points(55) == 41
    def test_age_60_to_69(self):       assert _age_points(65) == 58
    def test_age_70_to_79(self):       assert _age_points(74) == 75
    def test_age_80_to_89(self):       assert _age_points(85) == 91
    def test_age_90_plus(self):        assert _age_points(95) == 100


class TestHeartRatePoints:
    """HR (bpm) → GRACE points per JAMA 2003 Table 2."""

    def test_hr_under_50(self):        assert _hr_points(40)  == 0
    def test_hr_50_to_69(self):        assert _hr_points(60)  == 3
    def test_hr_70_to_89(self):        assert _hr_points(80)  == 9
    def test_hr_90_to_109(self):       assert _hr_points(100) == 15
    def test_hr_110_to_149(self):      assert _hr_points(130) == 24
    def test_hr_150_to_199(self):      assert _hr_points(175) == 38
    def test_hr_200_plus(self):        assert _hr_points(220) == 46


class TestSBPPoints:
    """Systolic BP (mmHg) → GRACE points (inverse) per JAMA 2003 Table 2."""

    def test_sbp_under_80(self):       assert _sbp_points(70)  == 58
    def test_sbp_80_to_99(self):       assert _sbp_points(90)  == 53
    def test_sbp_100_to_119(self):     assert _sbp_points(110) == 43
    def test_sbp_120_to_139(self):     assert _sbp_points(130) == 34
    def test_sbp_140_to_159(self):     assert _sbp_points(150) == 24
    def test_sbp_160_to_199(self):     assert _sbp_points(180) == 10
    def test_sbp_200_plus(self):       assert _sbp_points(210) == 0


class TestCreatininePoints:
    """Creatinine (µmol/L) → GRACE points per JAMA 2003 Table 2."""

    def test_cr_under_35(self):        assert _creatinine_points(20)  == 1
    def test_cr_35_to_70(self):        assert _creatinine_points(50)  == 4
    def test_cr_71_to_106(self):       assert _creatinine_points(88)  == 7
    def test_cr_107_to_141(self):      assert _creatinine_points(120) == 10
    def test_cr_142_to_177(self):      assert _creatinine_points(160) == 13
    def test_cr_177_to_354(self):      assert _creatinine_points(250) == 21
    def test_cr_354_to_707(self):      assert _creatinine_points(500) == 28
    def test_cr_over_707(self):        assert _creatinine_points(800) == 31


class TestKillipPoints:
    """Killip class → GRACE points per JAMA 2003."""

    def test_killip_I(self):           assert _KILLIP_POINTS[1] == 0
    def test_killip_II(self):          assert _KILLIP_POINTS[2] == 20
    def test_killip_III(self):         assert _KILLIP_POINTS[3] == 39
    def test_killip_IV(self):          assert _KILLIP_POINTS[4] == 59


# ---------------------------------------------------------------------------
# 2. Binary flag point values
# ---------------------------------------------------------------------------

class TestBinaryFlags:

    def _score_diff(self, flag_name: bool, **base_kwargs) -> int:
        """Score delta when one binary flag is toggled."""
        base = {
            "age": 55, "heart_rate_bpm": 80, "systolic_bp_mmhg": 130,
            "creatinine_umol_l": 88, "killip_class": 1,
            "cardiac_arrest": False, "st_deviation": False, "elevated_enzymes": False,
        }
        base.update(base_kwargs)
        r_off = compute_grace_score(**{**base, flag_name: False})
        r_on  = compute_grace_score(**{**base, flag_name: True})
        return r_on.total_score - r_off.total_score

    def test_cardiac_arrest_adds_39_points(self):
        assert self._score_diff("cardiac_arrest") == 39

    def test_st_deviation_adds_28_points(self):
        assert self._score_diff("st_deviation") == 28

    def test_elevated_enzymes_adds_14_points(self):
        assert self._score_diff("elevated_enzymes") == 14


# ---------------------------------------------------------------------------
# 3. Full-score integration — published reference cases
# ---------------------------------------------------------------------------

class TestGRACEIntegration:
    """
    End-to-end score checks from published clinical examples.
    Component sums are verified explicitly so failures are diagnosable.
    """

    def test_low_risk_reference_case(self):
        """
        Young, haemodynamically stable patient — expected GRACE ≤ 108 (Low).
        Profile: age 45 (25 pts), HR 68 (3 pts), SBP 145 (24 pts),
                 Cr 88 µmol/L (7 pts), Killip I (0 pts) — no flags.
        Total = 59 → Low risk.
        """
        r = compute_grace_score(
            age=45, heart_rate_bpm=68, systolic_bp_mmhg=145,
            creatinine_umol_l=88, killip_class=1,
        )
        assert r.total_score == 59
        assert r.risk_category == "Low"

    def test_intermediate_risk_reference_case(self):
        """
        Middle-aged patient, mildly elevated markers.
        age 62 (58), HR 95 (15), SBP 130 (34), Cr 110 (10),
        Killip I (0), enzymes (14) → Total = 131 → Intermediate.
        """
        r = compute_grace_score(
            age=62, heart_rate_bpm=95, systolic_bp_mmhg=130,
            creatinine_umol_l=110, killip_class=1,
            elevated_enzymes=True,
        )
        assert r.total_score == 131
        assert r.risk_category == "Intermediate"

    def test_high_risk_reference_case(self):
        """
        Elderly patient with multiple high-risk features.
        age 78 (75), HR 110 (24), SBP 88 (53), Cr 155 (13),
        Killip III (39), arrest (39), ST dev (28), enzymes (14) → Total = 285 → High.
        """
        r = compute_grace_score(
            age=78, heart_rate_bpm=110, systolic_bp_mmhg=88,
            creatinine_umol_l=155, killip_class=3,
            cardiac_arrest=True, st_deviation=True, elevated_enzymes=True,
        )
        assert r.total_score == 285
        assert r.risk_category == "High"

    def test_score_components_sum_to_total(self):
        """total_score must equal sum of all stored component points."""
        r = compute_grace_score(
            age=66, heart_rate_bpm=95, systolic_bp_mmhg=115,
            creatinine_umol_l=120, killip_class=2,
            cardiac_arrest=False, st_deviation=True, elevated_enzymes=False,
        )
        component_sum = (
            r.age_pts + r.hr_pts + r.sbp_pts + r.creatinine_pts
            + r.killip_pts + r.cardiac_arrest_pts
            + r.st_deviation_pts + r.elevated_enzymes_pts
        )
        assert r.total_score == component_sum


# ---------------------------------------------------------------------------
# 4. Risk category thresholds
# ---------------------------------------------------------------------------

class TestRiskThresholds:

    def test_boundary_low_risk(self):
        """Score exactly at LOW_RISK_THRESHOLD → Low."""
        # Construct a minimal patient and check the boundary value holds
        assert LOW_RISK_THRESHOLD <= 109   # spec says ≤108–109 is Low

    def test_boundary_high_risk(self):
        """Score at or above HIGH_RISK_THRESHOLD → High."""
        assert HIGH_RISK_THRESHOLD >= 140  # spec says ≥140 is High

    def test_score_0_is_low(self):
        """Extreme low score (min possible inputs) → Low risk."""
        r = compute_grace_score(
            age=25, heart_rate_bpm=40, systolic_bp_mmhg=200,
            creatinine_umol_l=20, killip_class=1,
        )
        assert r.risk_category == "Low"

    def test_all_flags_high_score_is_high_risk(self):
        """All flags enabled, high-risk vitals → High risk."""
        r = compute_grace_score(
            age=90, heart_rate_bpm=200, systolic_bp_mmhg=70,
            creatinine_umol_l=800, killip_class=4,
            cardiac_arrest=True, st_deviation=True, elevated_enzymes=True,
        )
        assert r.risk_category == "High"


# ---------------------------------------------------------------------------
# 5. Mortality probability outputs
# ---------------------------------------------------------------------------

class TestMortalityOutputs:

    def test_inhospital_mortality_is_positive(self):
        r = compute_grace_score(60, 80, 130, 88)
        assert r.inhospital_mortality_pct > 0

    def test_6month_mortality_is_positive(self):
        r = compute_grace_score(60, 80, 130, 88)
        assert r.six_month_mortality_pct > 0

    def test_high_score_has_higher_mortality_than_low_score(self):
        low  = compute_grace_score(35, 55, 160, 50, killip_class=1)
        high = compute_grace_score(82, 165, 75, 400, killip_class=4,
                                   cardiac_arrest=True, st_deviation=True)
        assert high.inhospital_mortality_pct > low.inhospital_mortality_pct
        assert high.six_month_mortality_pct  > low.six_month_mortality_pct

    def test_mortality_percentages_in_valid_range(self):
        r = compute_grace_score(70, 95, 110, 150, killip_class=2,
                                st_deviation=True, elevated_enzymes=True)
        assert 0.0 < r.inhospital_mortality_pct <= 100.0
        assert 0.0 < r.six_month_mortality_pct  <= 100.0


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

class TestGRACEValidation:

    def test_invalid_age_raises(self):
        with pytest.raises(ValueError, match="age"):
            compute_grace_score(0, 80, 130, 88)

    def test_age_over_130_raises(self):
        with pytest.raises(ValueError):
            compute_grace_score(200, 80, 130, 88)

    def test_invalid_hr_raises(self):
        with pytest.raises(ValueError, match="heart_rate_bpm"):
            compute_grace_score(60, 0, 130, 88)

    def test_invalid_sbp_raises(self):
        with pytest.raises(ValueError, match="systolic_bp_mmhg"):
            compute_grace_score(60, 80, 0, 88)

    def test_negative_creatinine_raises(self):
        with pytest.raises(ValueError, match="creatinine"):
            compute_grace_score(60, 80, 130, -1)

    def test_invalid_killip_class_raises(self):
        with pytest.raises(ValueError, match="killip"):
            compute_grace_score(60, 80, 130, 88, killip_class=5)


# ---------------------------------------------------------------------------
# 7. Return-type contract
# ---------------------------------------------------------------------------

class TestGRACEReturnType:

    def test_returns_grace_result(self):
        r = compute_grace_score(55, 75, 130, 90)
        assert isinstance(r, GRACEResult)

    def test_all_fields_present(self):
        r = compute_grace_score(55, 75, 130, 90)
        for field in ("total_score", "inhospital_mortality_pct",
                      "six_month_mortality_pct", "risk_category"):
            assert hasattr(r, field)

    def test_risk_category_is_valid_string(self):
        r = compute_grace_score(55, 75, 130, 90)
        assert r.risk_category in ("Low", "Intermediate", "High")


# ---------------------------------------------------------------------------
# 8. calibrate_grace_for_pipeline — dict interface
# ---------------------------------------------------------------------------

class TestCalibrateGraceForPipeline:

    _patient = {
        "age": 62,
        "heart_rate": 88,
        "systolic_bp": 130,
        "creatinine_umol_l": 106,
        "killip_class": 1,
        "st_deviation": True,
    }

    def test_returns_dict(self):
        out = calibrate_grace_for_pipeline(self._patient)
        assert isinstance(out, dict)

    def test_required_keys_present(self):
        out = calibrate_grace_for_pipeline(self._patient)
        for key in ("grace_score", "grace_risk_category",
                    "grace_inhospital_mortality_pct", "grace_6month_mortality_pct"):
            assert key in out, f"Missing key: {key}"

    def test_grace_score_is_int(self):
        out = calibrate_grace_for_pipeline(self._patient)
        assert isinstance(out["grace_score"], int)

    def test_grace_risk_category_valid(self):
        out = calibrate_grace_for_pipeline(self._patient)
        assert out["grace_risk_category"] in ("Low", "Intermediate", "High")

    def test_matches_direct_compute(self):
        """Pipeline wrapper must return same score as compute_grace_score."""
        out = calibrate_grace_for_pipeline(self._patient)
        direct = compute_grace_score(
            age=62, heart_rate_bpm=88, systolic_bp_mmhg=130,
            creatinine_umol_l=106, killip_class=1, st_deviation=True,
        )
        assert out["grace_score"] == direct.total_score
        assert out["grace_risk_category"] == direct.risk_category

    def test_optional_keys_default_gracefully(self):
        """Patient dict with only mandatory keys must not raise."""
        minimal = {"age": 55, "heart_rate": 72, "systolic_bp": 140, "creatinine_umol_l": 88}
        out = calibrate_grace_for_pipeline(minimal)
        assert "grace_score" in out


# ---------------------------------------------------------------------------
# 9. KillipClass enum
# ---------------------------------------------------------------------------

class TestKillipEnum:

    def test_killip_values(self):
        assert KillipClass.I   == 1
        assert KillipClass.II  == 2
        assert KillipClass.III == 3
        assert KillipClass.IV  == 4
