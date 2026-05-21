
"""
src/clinical/routing.py
-----------------------
Routes a patient to the correct treatment branch based on:
  - Part 1 risk category (Low / Medium / High)
  - Arrhythmia flag (has_arrhythmia)
  - Vital signs (age, heart_rate, systolic_bp)

Returns a RouteResult dataclass consumed by main.py and part2_recommender.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# BED result (cardiac radioablation path)
# ---------------------------------------------------------------------------

@dataclass
class BEDResult:
    total_dose_gy: float
    n_fractions:   int
    dose_per_fx:   float
    alpha_beta_gy: float
    bed_gy:        float
    eqd2_gy:       float
    within_window: bool


def compute_bed(
    total_dose_gy: float,
    n_fractions: int,
    alpha_beta_gy: float = 10.0,
) -> BEDResult:
    """
    BED = D * (1 + d / (α/β))
    EQD2 = BED / (1 + 2 / (α/β))
    Clinical window for cardiac SBRT: BED 50 – 120 Gy
    """
    if total_dose_gy < 0:
        raise ValueError("total_dose_gy must be >= 0")
    if n_fractions <= 0:
        raise ValueError("n_fractions must be > 0")

    d      = total_dose_gy / n_fractions
    bed    = total_dose_gy * (1 + d / alpha_beta_gy)
    eqd2   = bed / (1 + 2 / alpha_beta_gy)
    window = 50.0 <= bed <= 120.0

    return BEDResult(
        total_dose_gy=total_dose_gy,
        n_fractions=n_fractions,
        dose_per_fx=round(d, 3),
        alpha_beta_gy=alpha_beta_gy,
        bed_gy=round(bed, 2),
        eqd2_gy=round(eqd2, 2),
        within_window=window,
    )


# ---------------------------------------------------------------------------
# GRACE result (medication path)
# ---------------------------------------------------------------------------

@dataclass
class GRACEResult:
    total_score:   int
    risk_category: str          # Low / Intermediate / High
    mortality_pct: float        # estimated in-hospital mortality %


def compute_grace(
    age: int,
    heart_rate: int,
    systolic_bp: float,
    has_st_deviation: bool = False,
    has_cardiac_arrest: bool = False,
    has_elevated_enzymes: bool = False,
    killip_class: int = 1,
    creatinine_umol: float = 100.0,
) -> GRACEResult:
    """
    Simplified GRACE score (validated for ACS risk stratification).
    Full GRACE uses lookup tables; this is a linear approximation suitable
    for the project's clinical decision support context.
    """
    score = 0

    # Age component (0–100)
    if age < 30:    score += 0
    elif age < 40:  score += 8
    elif age < 50:  score += 25
    elif age < 60:  score += 41
    elif age < 70:  score += 58
    elif age < 80:  score += 75
    else:           score += 91

    # Heart rate (0–46)
    if heart_rate < 50:    score += 0
    elif heart_rate < 70:  score += 3
    elif heart_rate < 90:  score += 9
    elif heart_rate < 110: score += 15
    elif heart_rate < 150: score += 24
    elif heart_rate < 200: score += 38
    else:                  score += 46

    # Systolic BP (0–58)
    if systolic_bp < 80:    score += 58
    elif systolic_bp < 100: score += 53
    elif systolic_bp < 120: score += 43
    elif systolic_bp < 140: score += 34
    elif systolic_bp < 160: score += 24
    elif systolic_bp < 200: score += 10
    else:                   score += 0

    # Creatinine
    if creatinine_umol >= 350:  score += 28
    elif creatinine_umol >= 175: score += 20
    elif creatinine_umol >= 106: score += 14

    # Binary flags
    if has_cardiac_arrest:    score += 39
    if has_st_deviation:      score += 28
    if has_elevated_enzymes:  score += 14

    # Killip class (I–IV)
    killip_map = {1: 0, 2: 20, 3: 39, 4: 59}
    score += killip_map.get(max(1, min(4, killip_class)), 0)

    # Risk category
    if score < 109:
        category = "Low"
        mortality = 1.0 + score * 0.01
    elif score < 140:
        category = "Intermediate"
        mortality = 3.0 + (score - 109) * 0.06
    else:
        category = "High"
        mortality = 5.0 + (score - 140) * 0.08

    return GRACEResult(
        total_score=score,
        risk_category=category,
        mortality_pct=round(min(mortality, 99.0), 1),
    )


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------

class RiskLevel:
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


# ---------------------------------------------------------------------------
# RouteResult
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    path:                 str                       # human-readable path label
    branch:               str                       # "SBRT" | "Medication"
    medication_intensity: Optional[str]             # "Low" | "Moderate" | "High" | None
    bed_result:           Optional[BEDResult]       # filled for SBRT path
    bed_valid:            Optional[bool]            # filled for SBRT path
    grace_result:         Optional[GRACEResult]     # filled for Medication path
    routing_path:         str                       # API-facing label


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def route_patient(
    patient_dict: dict,
    predicted_risk: str,
    confidence_score: float = 1.0,
) -> RouteResult:
    """
    Route a patient to the SBRT or Medication branch.

    Parameters
    ----------
    patient_dict    : must contain keys:
                        has_arrhythmia (bool)
                        age (int)
                        heart_rate (int)      ← MaxHR or resting HR
                        systolic_bp (float)   ← RestingBP
    predicted_risk  : "Low" | "Medium" | "High"
    confidence_score: float 0–1

    Returns
    -------
    RouteResult
    """
    has_arrhythmia = bool(patient_dict.get("has_arrhythmia", False))
    age            = int(patient_dict.get("age", 60))
    heart_rate     = int(patient_dict.get("heart_rate", patient_dict.get("MaxHR", 80)))
    systolic_bp    = float(patient_dict.get("systolic_bp", patient_dict.get("RestingBP", 120.0)))

    # ── High risk + arrhythmia → SBRT ──────────────────────────────────────
    if predicted_risk == RiskLevel.HIGH and has_arrhythmia:
        bed = compute_bed(total_dose_gy=25.0, n_fractions=1)
        return RouteResult(
            path="High-Risk Path: Cardiac Radioablation (SBRT)",
            branch="SBRT",
            medication_intensity=None,
            bed_result=bed,
            bed_valid=bed.within_window,
            grace_result=None,
            routing_path="High-Risk Path: Cardiac Radioablation (SBRT)",
        )

    # ── All other patients → Medication ────────────────────────────────────
    grace = compute_grace(
        age=age,
        heart_rate=heart_rate,
        systolic_bp=systolic_bp,
    )

    if predicted_risk == RiskLevel.HIGH:
        intensity = "High"
        path_label = "High-Risk Path: Aggressive Medication Regimen"
    elif predicted_risk == RiskLevel.MEDIUM or grace.risk_category == "Intermediate":
        intensity = "Moderate"
        path_label = "Medium-Risk Path: Medication Calibration (GRACE Intermediate)"
    else:
        intensity = "Low"
        path_label = "Low-Risk Path: Lifestyle + Low-Dose Medication"

    return RouteResult(
        path=path_label,
        branch="Medication",
        medication_intensity=intensity,
        bed_result=None,
        bed_valid=None,
        grace_result=grace,
        routing_path=path_label,
    )
