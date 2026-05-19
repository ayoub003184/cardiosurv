"""
Clinical routing module — implements the Risk Level branching logic from
the cardiovascular ML pipeline (see System Flowchart, Section 4).

Decision boundary
-----------------
  Predicted Risk   │  Path
  ─────────────────┼──────────────────────────────────────────
  High             │  High-Risk Path → Cardiac Radioablation (SBRT)
                   │    └─ BED formula validation (α/β = 10 Gy)
  Low / Medium     │  Low/Medium-Risk Path → Medication Intensity Calibration
                   │    └─ GRACE Risk Score Calibration

Both paths converge at the MIMIC-IV Patient Metadata Merge step, after which
a second XGBoost model produces an Intervention Type & Intensity Recommendation,
followed by Survival Probability Estimation and Counterfactual Output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bed import BEDResult, compute_bed, validate_sbrt_bed, DEFAULT_ALPHA_BETA
from .grace import GRACEResult, compute_grace_score, calibrate_grace_for_pipeline


# ---------------------------------------------------------------------------
# Risk level enum (mirrors classifier output)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


# ---------------------------------------------------------------------------
# Default SBRT dosing parameters
# ---------------------------------------------------------------------------

SBRT_DEFAULT_TOTAL_DOSE_GY  = 25.0   # standard single-fraction cardiac SBRT
SBRT_DEFAULT_N_FRACTIONS    = 1
SBRT_ABLATIVE_THRESHOLD_GY  = 100.0  # minimum BED (Gy) for ablative intent


# ---------------------------------------------------------------------------
# Route output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """
    Full output of the routing step for a single patient.

    Attributes
    ----------
    patient_id : Any
        Identifier passed through from the caller.
    risk_level : RiskLevel
        Predicted risk category from the upstream classifier.
    confidence_score : float
        Model confidence (0–1) for the predicted risk level.
    path : str
        Human-readable label for the selected clinical path.

    For High-Risk patients
    ----------------------
    bed_result : BEDResult | None
        BED calculation result.
    bed_valid : bool
        Whether the BED meets the ablative threshold.
    bed_validation_message : str
        Plain-English BED validation summary.

    For Low/Medium-Risk patients
    ----------------------------
    grace_result : GRACEResult | None
        GRACE score calculation result.
    medication_intensity : str | None
        Suggested medication intensity tier.

    Shared
    ------
    calibration_features : dict
        Feature dict ready to be merged with MIMIC-IV patient metadata
        and fed into the second XGBoost model.
    warnings : list[str]
        Any clinical flags raised during routing.
    """

    patient_id: Any
    risk_level: RiskLevel
    confidence_score: float
    path: str

    # High-risk SBRT fields
    bed_result: BEDResult | None = None
    bed_valid: bool = False
    bed_validation_message: str = ""

    # Low/Med risk fields
    grace_result: GRACEResult | None = None
    medication_intensity: str | None = None

    # Downstream merge features
    calibration_features: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Patient {self.patient_id} — Risk: {self.risk_level.value} "
            f"(confidence {self.confidence_score:.2%})",
            f"  Path: {self.path}",
        ]
        if self.bed_result is not None:
            lines.append(f"  BED: {self.bed_result}")
            lines.append(f"  BED validation: {self.bed_validation_message}")
        if self.grace_result is not None:
            lines.append(f"  GRACE: {self.grace_result.summary()}")
            lines.append(f"  Medication intensity: {self.medication_intensity}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Medication intensity calibration (Low/Med path)
# ---------------------------------------------------------------------------

def _calibrate_medication_intensity(grace_result: GRACEResult) -> str:
    """
    Map GRACE risk category + score to a medication intensity tier.

    Returns one of: "Standard", "Intensified", "Maximal"
    """
    if grace_result.risk_category == "Low":
        return "Standard"
    if grace_result.risk_category == "Intermediate":
        return "Intensified"
    # High GRACE score even on the Low/Med overall risk path → maximal medical therapy
    return "Maximal"


# ---------------------------------------------------------------------------
# High-risk SBRT path
# ---------------------------------------------------------------------------

def _route_high_risk(
    patient: dict,
    warnings: list[str],
    total_dose_gy: float,
    n_fractions: int,
    alpha_beta_gy: float,
) -> tuple[BEDResult, bool, str, dict]:
    """
    Execute the High-Risk path: BED validation for cardiac SBRT.

    Returns (bed_result, bed_valid, message, calibration_features).
    """
    bed_result, bed_valid, bed_msg = validate_sbrt_bed(
        total_dose_gy=total_dose_gy,
        n_fractions=n_fractions,
        alpha_beta_gy=alpha_beta_gy,
        min_bed_gy=SBRT_ABLATIVE_THRESHOLD_GY,
    )
    if not bed_valid:
        warnings.append(
            f"Prescribed SBRT regimen ({total_dose_gy} Gy / {n_fractions} fx) "
            f"does not meet ablative BED threshold ({SBRT_ABLATIVE_THRESHOLD_GY} Gy). "
            "Consider dose escalation."
        )

    calibration_features = {
        "intervention_type": "SBRT",
        "total_dose_gy": total_dose_gy,
        "n_fractions": n_fractions,
        "dose_per_fraction_gy": bed_result.dose_per_fraction_gy,
        "bed_gy": bed_result.bed_gy,
        "eqd2_gy": bed_result.eqd2_gy,
        "alpha_beta_gy": alpha_beta_gy,
        "bed_valid": int(bed_valid),
    }
    return bed_result, bed_valid, bed_msg, calibration_features


# ---------------------------------------------------------------------------
# Low/Medium-risk medication path
# ---------------------------------------------------------------------------

def _route_low_med_risk(
    patient: dict,
    warnings: list[str],
) -> tuple[GRACEResult, str, dict]:
    """
    Execute the Low/Medium-Risk path: GRACE calibration + medication intensity.

    Returns (grace_result, medication_intensity, calibration_features).
    """
    grace_result = compute_grace_score(
        age=int(patient["age"]),
        heart_rate_bpm=float(patient["heart_rate"]),
        systolic_bp_mmhg=float(patient["systolic_bp"]),
        creatinine_umol_l=float(patient.get("creatinine_umol_l", 88.4)),
        killip_class=int(patient.get("killip_class", 1)),
        cardiac_arrest=bool(patient.get("cardiac_arrest", False)),
        st_deviation=bool(patient.get("st_deviation", False)),
        elevated_enzymes=bool(patient.get("elevated_enzymes", False)),
    )

    # Flag cases where GRACE disagrees strongly with the ML risk label
    if grace_result.risk_category == "High":
        warnings.append(
            f"GRACE score {grace_result.total_score} (High) conflicts with "
            "ML-predicted Low/Medium risk. Manual clinical review recommended."
        )

    med_intensity = _calibrate_medication_intensity(grace_result)

    calibration_features = {
        "intervention_type": "Medication",
        "medication_intensity": med_intensity,
        "grace_score": grace_result.total_score,
        "grace_risk_category": grace_result.risk_category,
        "grace_inhospital_mortality_pct": grace_result.inhospital_mortality_pct,
        "grace_6month_mortality_pct": grace_result.six_month_mortality_pct,
    }
    return grace_result, med_intensity, calibration_features


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def route_patient(
    patient: dict,
    predicted_risk: str | RiskLevel,
    confidence_score: float,
    *,
    sbrt_total_dose_gy: float = SBRT_DEFAULT_TOTAL_DOSE_GY,
    sbrt_n_fractions: int = SBRT_DEFAULT_N_FRACTIONS,
    sbrt_alpha_beta_gy: float = DEFAULT_ALPHA_BETA,
) -> RouteResult:
    """
    Route a patient to the appropriate clinical pathway based on predicted risk.

    Parameters
    ----------
    patient : dict
        Patient feature dict. Required keys vary by path — see `_route_high_risk`
        and `_route_low_med_risk` for details.  The following keys are always
        expected for the Low/Med path:
            age, heart_rate, systolic_bp
        The following are optional (sensible defaults applied if absent):
            creatinine_umol_l, killip_class, cardiac_arrest,
            st_deviation, elevated_enzymes
    predicted_risk : str | RiskLevel
        Predicted risk level from the upstream Random Forest / XGBoost model.
        Accepted string values (case-insensitive): "Low", "Medium", "High".
    confidence_score : float
        Model confidence score ∈ [0, 1].
    sbrt_total_dose_gy : float
        Total SBRT dose to validate (High-Risk path only).
    sbrt_n_fractions : int
        Number of SBRT fractions (High-Risk path only).
    sbrt_alpha_beta_gy : float
        α/β ratio for BED calculation (default 10 Gy).

    Returns
    -------
    RouteResult
        Populated routing result ready for MIMIC-IV metadata merge.

    Raises
    ------
    ValueError
        If `predicted_risk` is not a recognised value.

    Examples
    --------
    >>> patient = {"age": 68, "heart_rate": 92, "systolic_bp": 105,
    ...            "creatinine_umol_l": 120, "killip_class": 2,
    ...            "st_deviation": True}
    >>> result = route_patient(patient, predicted_risk="High", confidence_score=0.91)
    >>> result.path
    'High-Risk Path: Cardiac Radioablation (SBRT)'
    """
    # Normalise risk level
    if isinstance(predicted_risk, str):
        try:
            risk_level = RiskLevel(predicted_risk.capitalize())
        except ValueError:
            raise ValueError(
                f"Unknown predicted_risk value '{predicted_risk}'. "
                "Expected one of: Low, Medium, High."
            )
    else:
        risk_level = predicted_risk

    patient_id = patient.get("patient_id", "UNKNOWN")
    warnings: list[str] = []

    # Low confidence warning
    if confidence_score < 0.6:
        warnings.append(
            f"Low model confidence ({confidence_score:.2%}). "
            "Clinical judgement should supersede automated routing."
        )

    # --- Branch on risk level ---
    if risk_level == RiskLevel.HIGH:
        path = "High-Risk Path: Cardiac Radioablation (SBRT)"
        bed_result, bed_valid, bed_msg, cal_features = _route_high_risk(
            patient, warnings,
            sbrt_total_dose_gy, sbrt_n_fractions, sbrt_alpha_beta_gy,
        )
        return RouteResult(
            patient_id=patient_id,
            risk_level=risk_level,
            confidence_score=confidence_score,
            path=path,
            bed_result=bed_result,
            bed_valid=bed_valid,
            bed_validation_message=bed_msg,
            calibration_features={
                "patient_id": patient_id,
                "risk_level": risk_level.value,
                "confidence_score": confidence_score,
                **cal_features,
            },
            warnings=warnings,
        )

    else:  # Low or Medium
        path = "Low/Medium-Risk Path: Medication Intensity Calibration"
        grace_result, med_intensity, cal_features = _route_low_med_risk(
            patient, warnings
        )
        return RouteResult(
            patient_id=patient_id,
            risk_level=risk_level,
            confidence_score=confidence_score,
            path=path,
            grace_result=grace_result,
            medication_intensity=med_intensity,
            calibration_features={
                "patient_id": patient_id,
                "risk_level": risk_level.value,
                "confidence_score": confidence_score,
                **cal_features,
            },
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Batch routing helper
# ---------------------------------------------------------------------------

def route_batch(
    patients: list[dict],
    predicted_risks: list[str | RiskLevel],
    confidence_scores: list[float],
    **kwargs: Any,
) -> list[RouteResult]:
    """
    Route a batch of patients in a loop.

    Parameters mirror `route_patient`; lists must be the same length.
    Exceptions for individual patients are caught and stored as warnings.
    """
    if not (len(patients) == len(predicted_risks) == len(confidence_scores)):
        raise ValueError("patients, predicted_risks, and confidence_scores must have equal length.")

    results: list[RouteResult] = []
    for patient, risk, conf in zip(patients, predicted_risks, confidence_scores):
        try:
            results.append(route_patient(patient, risk, conf, **kwargs))
        except Exception as exc:  # noqa: BLE001
            pid = patient.get("patient_id", "UNKNOWN")
            results.append(RouteResult(
                patient_id=pid,
                risk_level=RiskLevel.LOW,
                confidence_score=conf,
                path="ERROR",
                warnings=[f"Routing failed: {exc}"],
            ))
    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    high_risk_patient = {
        "patient_id": "P001",
        "age": 74,
        "heart_rate": 102,
        "systolic_bp": 95,
        "creatinine_umol_l": 180,
        "killip_class": 3,
        "cardiac_arrest": True,
        "st_deviation": True,
        "elevated_enzymes": True,
    }
    r_high = route_patient(high_risk_patient, "High", 0.93, sbrt_total_dose_gy=25.0, sbrt_n_fractions=1)
    print(r_high.summary())
    print()

    low_risk_patient = {
        "patient_id": "P002",
        "age": 55,
        "heart_rate": 72,
        "systolic_bp": 138,
        "creatinine_umol_l": 85,
        "killip_class": 1,
    }
    r_low = route_patient(low_risk_patient, "Low", 0.87)
    print(r_low.summary())
