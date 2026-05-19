"""
GRACE (Global Registry of Acute Coronary Events) 2.0 Risk Score.

Predicts all-cause mortality at hospital discharge and at 6 months
for patients presenting with Acute Coronary Syndrome (ACS).

Variables used (original GRACE 2.0 model):
  - Age (years)
  - Heart rate (bpm)
  - Systolic blood pressure (mmHg)
  - Creatinine (µmol/L)
  - Killip class (I–IV)
  - Cardiac arrest at admission (bool)
  - ST-segment deviation on ECG (bool)
  - Elevated cardiac enzymes / markers (bool)

Reference:
  Fox KAA et al. (2014). Should patients with acute coronary disease be
  stratified for management according to their risk? Derivation, external
  validation and outcomes using the updated GRACE risk score. BMJ Open.
  doi:10.1136/bmjopen-2013-004425
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# Killip classification
# ---------------------------------------------------------------------------

class KillipClass(IntEnum):
    """Killip-Kimball classification of heart failure severity in ACS."""
    I   = 1   # No heart failure signs
    II  = 2   # Mild heart failure (rales, S3 gallop)
    III = 3   # Acute pulmonary oedema
    IV  = 4   # Cardiogenic shock


# ---------------------------------------------------------------------------
# GRACE point tables (GRACE 1.0 / original validated integer score)
# ---------------------------------------------------------------------------

def _age_points(age: int) -> int:
    if age < 30:   return 0
    if age < 40:   return 8
    if age < 50:   return 25
    if age < 60:   return 41
    if age < 70:   return 58
    if age < 80:   return 75
    if age < 90:   return 91
    return 100


def _hr_points(hr: float) -> int:
    """Heart rate (bpm) → GRACE points."""
    if hr < 50:    return 0
    if hr < 70:    return 3
    if hr < 90:    return 9
    if hr < 110:   return 15
    if hr < 150:   return 24
    if hr < 200:   return 38
    return 46


def _sbp_points(sbp: float) -> int:
    """Systolic BP (mmHg) → GRACE points (inverse relationship)."""
    if sbp < 80:   return 58
    if sbp < 100:  return 53
    if sbp < 120:  return 43
    if sbp < 140:  return 34
    if sbp < 160:  return 24
    if sbp < 200:  return 10
    return 0


def _creatinine_points(creatinine_umol_l: float) -> int:
    """Creatinine (µmol/L) → GRACE points."""
    if creatinine_umol_l < 35.4:   return 1
    if creatinine_umol_l < 70.7:   return 4
    if creatinine_umol_l < 106.1:  return 7
    if creatinine_umol_l < 141.4:  return 10
    if creatinine_umol_l < 176.8:  return 13
    if creatinine_umol_l < 353.6:  return 21
    if creatinine_umol_l < 707.2:  return 28
    return 31


_KILLIP_POINTS: dict[int, int] = {1: 0, 2: 20, 3: 39, 4: 59}


# ---------------------------------------------------------------------------
# Mortality risk lookup (GRACE integer score → in-hospital mortality %)
# Derived from the published GRACE nomogram tables.
# ---------------------------------------------------------------------------

def _score_to_inhospital_mortality(score: int) -> float:
    """
    Map integer GRACE score to approximate in-hospital mortality (%).
    Piecewise-linear interpolation of the published GRACE nomogram.
    """
    # Anchor points (score, mortality %)
    anchors = [
        (0,   0.2),
        (60,  0.5),
        (80,  1.0),
        (100, 1.9),
        (120, 3.0),
        (140, 5.0),
        (160, 9.0),
        (180, 14.0),
        (200, 20.0),
        (220, 29.0),
        (240, 40.0),
        (263, 52.0),
    ]
    if score <= anchors[0][0]:
        return anchors[0][1]
    if score >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        s0, m0 = anchors[i]
        s1, m1 = anchors[i + 1]
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return round(m0 + t * (m1 - m0), 2)
    return 0.0


def _score_to_6month_mortality(score: int) -> float:
    """
    Map integer GRACE score to approximate 6-month post-discharge mortality (%).
    Piecewise-linear interpolation of the published GRACE nomogram.
    """
    anchors = [
        (0,   0.4),
        (60,  1.0),
        (88,  3.0),
        (118, 8.0),
        (140, 15.0),
        (155, 20.0),
        (176, 30.0),
        (210, 52.0),
        (263, 80.0),
    ]
    if score <= anchors[0][0]:
        return anchors[0][1]
    if score >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        s0, m0 = anchors[i]
        s1, m1 = anchors[i + 1]
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return round(m0 + t * (m1 - m0), 2)
    return 0.0


# ---------------------------------------------------------------------------
# Risk category thresholds (in-hospital mortality)
# ---------------------------------------------------------------------------

LOW_RISK_THRESHOLD    = 109   # ≤ 108 → low risk
HIGH_RISK_THRESHOLD   = 140   # ≥ 140 → high risk


def _risk_category(score: int) -> str:
    if score <= LOW_RISK_THRESHOLD:
        return "Low"
    if score < HIGH_RISK_THRESHOLD:
        return "Intermediate"
    return "High"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GRACEResult:
    """Output of GRACE score calculation."""

    # Inputs
    age: int
    heart_rate_bpm: float
    systolic_bp_mmhg: float
    creatinine_umol_l: float
    killip_class: int
    cardiac_arrest: bool
    st_deviation: bool
    elevated_enzymes: bool

    # Score components
    age_pts: int
    hr_pts: int
    sbp_pts: int
    creatinine_pts: int
    killip_pts: int
    cardiac_arrest_pts: int
    st_deviation_pts: int
    elevated_enzymes_pts: int

    # Totals
    total_score: int
    inhospital_mortality_pct: float
    six_month_mortality_pct: float
    risk_category: str          # "Low" | "Intermediate" | "High"

    def summary(self) -> str:
        return (
            f"GRACE Score: {self.total_score}  |  "
            f"Risk: {self.risk_category}  |  "
            f"In-hospital mortality: {self.inhospital_mortality_pct:.1f}%  |  "
            f"6-month mortality: {self.six_month_mortality_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# Main calculation function
# ---------------------------------------------------------------------------

def compute_grace_score(
    age: int,
    heart_rate_bpm: float,
    systolic_bp_mmhg: float,
    creatinine_umol_l: float,
    killip_class: int = 1,
    cardiac_arrest: bool = False,
    st_deviation: bool = False,
    elevated_enzymes: bool = False,
) -> GRACEResult:
    """
    Compute the GRACE ACS Risk Score.

    Parameters
    ----------
    age : int
        Patient age in years.
    heart_rate_bpm : float
        Admission heart rate in beats per minute.
    systolic_bp_mmhg : float
        Admission systolic blood pressure in mmHg.
    creatinine_umol_l : float
        Serum creatinine in µmol/L.
    killip_class : int
        Killip class I–IV (default I).
    cardiac_arrest : bool
        Cardiac arrest at admission (adds 39 pts).
    st_deviation : bool
        ST-segment deviation on ECG (adds 28 pts).
    elevated_enzymes : bool
        Elevated cardiac enzymes/markers (adds 14 pts).

    Returns
    -------
    GRACEResult
        Scored result including risk category and mortality estimates.

    Raises
    ------
    ValueError
        If inputs are out of physiological range.

    Examples
    --------
    >>> r = compute_grace_score(age=65, heart_rate_bpm=88, systolic_bp_mmhg=130,
    ...                         creatinine_umol_l=106, killip_class=1)
    >>> r.risk_category
    'Low'
    """
    # --- Validation ---
    if not (0 < age < 130):
        raise ValueError(f"age must be in (0, 130), got {age}")
    if not (0 < heart_rate_bpm < 300):
        raise ValueError(f"heart_rate_bpm out of range: {heart_rate_bpm}")
    if not (0 < systolic_bp_mmhg < 350):
        raise ValueError(f"systolic_bp_mmhg out of range: {systolic_bp_mmhg}")
    if creatinine_umol_l < 0:
        raise ValueError(f"creatinine_umol_l must be >= 0, got {creatinine_umol_l}")
    if killip_class not in (1, 2, 3, 4):
        raise ValueError(f"killip_class must be 1–4, got {killip_class}")

    # --- Score components ---
    age_pts             = _age_points(age)
    hr_pts              = _hr_points(heart_rate_bpm)
    sbp_pts             = _sbp_points(systolic_bp_mmhg)
    creatinine_pts      = _creatinine_points(creatinine_umol_l)
    killip_pts          = _KILLIP_POINTS[killip_class]
    cardiac_arrest_pts  = 39 if cardiac_arrest else 0
    st_deviation_pts    = 28 if st_deviation else 0
    elevated_enzymes_pts = 14 if elevated_enzymes else 0

    total = (
        age_pts + hr_pts + sbp_pts + creatinine_pts + killip_pts
        + cardiac_arrest_pts + st_deviation_pts + elevated_enzymes_pts
    )

    return GRACEResult(
        age=age,
        heart_rate_bpm=heart_rate_bpm,
        systolic_bp_mmhg=systolic_bp_mmhg,
        creatinine_umol_l=creatinine_umol_l,
        killip_class=killip_class,
        cardiac_arrest=cardiac_arrest,
        st_deviation=st_deviation,
        elevated_enzymes=elevated_enzymes,
        age_pts=age_pts,
        hr_pts=hr_pts,
        sbp_pts=sbp_pts,
        creatinine_pts=creatinine_pts,
        killip_pts=killip_pts,
        cardiac_arrest_pts=cardiac_arrest_pts,
        st_deviation_pts=st_deviation_pts,
        elevated_enzymes_pts=elevated_enzymes_pts,
        total_score=total,
        inhospital_mortality_pct=_score_to_inhospital_mortality(total),
        six_month_mortality_pct=_score_to_6month_mortality(total),
        risk_category=_risk_category(total),
    )


def calibrate_grace_for_pipeline(patient: dict) -> dict:
    """
    Thin wrapper that accepts a patient metadata dict (as produced by MIMIC-IV
    merge step) and returns a calibration dict suitable for the XGBoost
    intervention recommender.

    Expected keys in `patient`:
        age, heart_rate, systolic_bp, creatinine_umol_l,
        killip_class (optional, default 1),
        cardiac_arrest (optional, default False),
        st_deviation (optional, default False),
        elevated_enzymes (optional, default False)

    Returns dict with keys:
        grace_score, grace_risk_category,
        grace_inhospital_mortality_pct, grace_6month_mortality_pct
    """
    result = compute_grace_score(
        age=int(patient["age"]),
        heart_rate_bpm=float(patient["heart_rate"]),
        systolic_bp_mmhg=float(patient["systolic_bp"]),
        creatinine_umol_l=float(patient.get("creatinine_umol_l", 88.4)),
        killip_class=int(patient.get("killip_class", 1)),
        cardiac_arrest=bool(patient.get("cardiac_arrest", False)),
        st_deviation=bool(patient.get("st_deviation", False)),
        elevated_enzymes=bool(patient.get("elevated_enzymes", False)),
    )
    return {
        "grace_score": result.total_score,
        "grace_risk_category": result.risk_category,
        "grace_inhospital_mortality_pct": result.inhospital_mortality_pct,
        "grace_6month_mortality_pct": result.six_month_mortality_pct,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 72-year-old, HR 95, SBP 110, Cr 140 µmol/L, Killip II, ST↑
    r = compute_grace_score(
        age=72,
        heart_rate_bpm=95,
        systolic_bp_mmhg=110,
        creatinine_umol_l=140,
        killip_class=2,
        st_deviation=True,
        elevated_enzymes=True,
    )
    print(r.summary())
    print(f"Score breakdown: age={r.age_pts}, HR={r.hr_pts}, SBP={r.sbp_pts}, "
          f"Cr={r.creatinine_pts}, Killip={r.killip_pts}, "
          f"arrest={r.cardiac_arrest_pts}, ST={r.st_deviation_pts}, "
          f"enzymes={r.elevated_enzymes_pts}")
