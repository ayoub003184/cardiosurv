"""
Clinical routing module — implements the Risk Level branching logic from
the cardiovascular ML pipeline (see System Flowchart, Section 4).

Decision boundary
-----------------
  Condition                                    │  Path
  ─────────────────────────────────────────────┼──────────────────────────────
  risk_category == "High"                      │  High-Risk Path →
    OR (risk_category == "Medium"              │    Cardiac Radioablation (SBRT)
        AND has_arrhythmia == True)            │    └─ BED validation (α/β=10 Gy)
  ─────────────────────────────────────────────┼──────────────────────────────
  All other cases (Low, or Medium w/o          │  Low/Medium-Risk Path →
    arrhythmia)                                │    Medication Intensity Calibration
                                               │    └─ GRACE Risk Score Calibration

Spec reference (T3.2):
  "if risk_category == 'High' and has_arrhythmia → SBRT branch,
   else → Medication branch"

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
# Routing decision helper
# ---------------------------------------------------------------------------

def _should_route_to_sbrt(risk_level: RiskLevel, has_arrhythmia: bool) -> bool:
    """
    Return True if the patient should be routed to the SBRT (high-risk) path.

    Rules (T3.2 spec):
      • risk_level == HIGH                         → always SBRT
      • risk_level == MEDIUM AND has_arrhythmia    → SBRT (arrhythmia escalation)
      • everything else                            → Medication path
    """
    if risk_level == RiskLevel.HIGH:
        return True
    if risk_level == RiskLevel.MEDIUM and has_arrhythmia:
        return True
    return False


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
    has_arrhythmia : bool
        Whether the patient has a documented arrhythmia.
        Acts as a secondary escalation trigger for Medium-risk patients.
    path : str
        Human-readable label for the selected clinical path.

    For High-Risk / Arrhythmia-escalated patients
    ----------------------------------------------
    bed_result : BEDResult | None
        BED calculation result.
    bed_valid : bool
        Whether the BED meets the ablative threshold.
    bed_validation_message : str
        Plain-English BED validation summary.

    For Low/Medium-Risk patients (no arrhythmia escalation)
    --------------------------------------------------------
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
    has_arrhythmia: bool
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
        arrhythmia_tag = " [arrhythmia]" if self.has_arrhythmia else ""
        lines = [
            f"Patient {self.patient_id} — Risk: {self.risk_level.value}{arrhythmia_tag} "
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
    Map GRACE risk category to a medication intensity tier.

    Returns one of: "Standard", "Intensified", "Maximal"
    """
    if grace_result.risk_category == "Low":
        return "Standard"
    if grace_result.risk_category == "Intermediate":
        return "Intensified"
    # High GRACE score on the Low/Med overall risk path → maximal medical therapy
    return "Maximal"


# ---------------------------------------------------------------------------
# High-risk / SBRT path
# ---------------------------------------------------------------------------

def _route_high_risk(
    patient: dict,
    warnings: list[str],
    total_dose_gy: float,
    n_fractions: int,
    alpha_beta_gy: float,
    escalation_reason: str,
) -> tuple[BEDResult, bool, str, dict]:
    """
    Execute the High-Risk path: BED validation for cardiac SBRT.

    Parameters
    ----------
    escalation_reason : str
        Human-readable reason this patient reached the SBRT path
        (e.g. "High risk" or "Medium risk + arrhythmia").

    Returns
    -------
    (bed_result, bed_valid, message, calibration_features)
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
        "escalation_reason": escalation_reason,
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

    Returns
    -------
    (grace_result, medication_intensity, calibration_features)
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
    Route a patient to the appropriate clinical pathway based on predicted risk
    and arrhythmia status.

    Routing logic (T3.2 spec)
    -------------------------
    SBRT path  : risk == High
                 OR (risk == Medium AND has_arrhythmia)
    Medication : all other cases

    Parameters
    ----------
    patient : dict
        Patient feature dict.

        Keys consumed by this function:
            patient_id      (optional, default "UNKNOWN")
            has_arrhythmia  (optional bool, default False)

        Keys required for the Low/Med path (passed to GRACE):
            age, heart_rate, systolic_bp

        Optional keys for GRACE (defaults applied if absent):
            creatinine_umol_l, killip_class, cardiac_arrest,
            st_deviation, elevated_enzymes

    predicted_risk : str | RiskLevel
        Predicted risk level: "Low", "Medium", or "High" (case-insensitive).
    confidence_score : float
        Model confidence score ∈ [0, 1].
    sbrt_total_dose_gy : float
        Total SBRT dose to validate (SBRT path only, default 25 Gy).
    sbrt_n_fractions : int
        Number of SBRT fractions (SBRT path only, default 1).
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
    High risk → SBRT regardless of arrhythmia:

    >>> p = {"age": 68, "heart_rate": 92, "systolic_bp": 105}
    >>> route_patient(p, "High", 0.91).path
    'High-Risk Path: Cardiac Radioablation (SBRT)'

    Medium risk + arrhythmia → escalated to SBRT:

    >>> p = {"age": 68, "heart_rate": 92, "systolic_bp": 120, "has_arrhythmia": True}
    >>> route_patient(p, "Medium", 0.78).path
    'High-Risk Path: Cardiac Radioablation (SBRT) [arrhythmia escalation]'

    Medium risk without arrhythmia → Medication:

    >>> p = {"age": 55, "heart_rate": 72, "systolic_bp": 138}
    >>> route_patient(p, "Medium", 0.80).path
    'Low/Medium-Risk Path: Medication Intensity Calibration'
    """
    # --- Normalise risk level ---
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

    patient_id     = patient.get("patient_id", "UNKNOWN")
    has_arrhythmia = bool(patient.get("has_arrhythmia", False))
    warnings: list[str] = []

    # Low-confidence warning
    if confidence_score < 0.6:
        warnings.append(
            f"Low model confidence ({confidence_score:.2%}). "
            "Clinical judgement should supersede automated routing."
        )

    # --- Core routing decision ---
    go_sbrt = _should_route_to_sbrt(risk_level, has_arrhythmia)

    if go_sbrt:
        if risk_level == RiskLevel.HIGH:
            escalation_reason = "High risk"
            path = "High-Risk Path: Cardiac Radioablation (SBRT)"
        else:
            # Medium + arrhythmia escalation
            escalation_reason = "Medium risk + arrhythmia"
            path = "High-Risk Path: Cardiac Radioablation (SBRT) [arrhythmia escalation]"
            warnings.append(
                "Patient escalated from Medium to SBRT path due to documented arrhythmia."
            )

        bed_result, bed_valid, bed_msg, cal_features = _route_high_risk(
            patient, warnings,
            sbrt_total_dose_gy, sbrt_n_fractions, sbrt_alpha_beta_gy,
            escalation_reason,
        )
        return RouteResult(
            patient_id=patient_id,
            risk_level=risk_level,
            confidence_score=confidence_score,
            has_arrhythmia=has_arrhythmia,
            path=path,
            bed_result=bed_result,
            bed_valid=bed_valid,
            bed_validation_message=bed_msg,
            calibration_features={
                "patient_id": patient_id,
                "risk_level": risk_level.value,
                "has_arrhythmia": has_arrhythmia,
                "confidence_score": confidence_score,
                **cal_features,
            },
            warnings=warnings,
        )

    else:
        path = "Low/Medium-Risk Path: Medication Intensity Calibration"
        grace_result, med_intensity, cal_features = _route_low_med_risk(
            patient, warnings
        )
        return RouteResult(
            patient_id=patient_id,
            risk_level=risk_level,
            confidence_score=confidence_score,
            has_arrhythmia=has_arrhythmia,
            path=path,
            grace_result=grace_result,
            medication_intensity=med_intensity,
            calibration_features={
                "patient_id": patient_id,
                "risk_level": risk_level.value,
                "has_arrhythmia": has_arrhythmia,
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

    All three lists must be the same length. Exceptions for individual patients
    are caught and stored as warnings so one bad record does not abort the batch.
    """
    if not (len(patients) == len(predicted_risks) == len(confidence_scores)):
        raise ValueError(
            "patients, predicted_risks, and confidence_scores must have equal length."
        )

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
                has_arrhythmia=bool(patient.get("has_arrhythmia", False)),
                path="ERROR",
                warnings=[f"Routing failed: {exc}"],
            ))
    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = dict(age=68, heart_rate=92, systolic_bp=105,
                creatinine_umol_l=120, killip_class=2)

    # Case 1: High risk → SBRT
    r1 = route_patient({**base, "patient_id": "P001"}, "High", 0.93)
    print(r1.summary()); print()

    # Case 2: Medium + arrhythmia → escalated to SBRT
    r2 = route_patient({**base, "patient_id": "P002", "has_arrhythmia": True},
                       "Medium", 0.78)
    print(r2.summary()); print()

    # Case 3: Medium, no arrhythmia → Medication
    r3 = route_patient({**base, "patient_id": "P003"}, "Medium", 0.80)
    print(r3.summary()); print()

    # Case 4: Low → Medication
    r4 = route_patient(dict(patient_id="P004", age=52, heart_rate=68,
                            systolic_bp=138, creatinine_umol_l=85), "Low", 0.91)
    print(r4.summary())
