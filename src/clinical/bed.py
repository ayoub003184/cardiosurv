"""
Biologically Effective Dose (BED) calculator for Cardiac Radioablation (SBRT).

Formula: BED = D * (1 + d / (alpha_beta))
  where:
    D           = total dose (Gy)
    d           = dose per fraction (Gy)
    alpha_beta  = tissue-specific α/β ratio (default 10 Gy for cardiac/tumour tissue)

Reference: Fowler JF (1989). The linear-quadratic formula and progress in
fractionated radiotherapy. Br J Radiol 62(740):679-694.
"""

from __future__ import annotations

from dataclasses import dataclass


# Default α/β ratio for cardiac / tumour tissue as specified in the flowchart
DEFAULT_ALPHA_BETA: float = 10.0  # Gy


@dataclass(frozen=True)
class BEDResult:
    """Result of a BED calculation."""

    total_dose_gy: float          # D  – total physical dose (Gy)
    dose_per_fraction_gy: float   # d  – dose per fraction (Gy)
    n_fractions: int              # number of fractions
    alpha_beta_gy: float          # α/β ratio (Gy)
    bed_gy: float                 # Biologically Effective Dose (Gy)
    eqd2_gy: float                # Equivalent Dose in 2 Gy fractions (EQD2)

    def __str__(self) -> str:
        return (
            f"BED  = {self.bed_gy:.2f} Gy  "
            f"[D={self.total_dose_gy} Gy, d={self.dose_per_fraction_gy} Gy, "
            f"n={self.n_fractions}, α/β={self.alpha_beta_gy} Gy]  |  "
            f"EQD2 = {self.eqd2_gy:.2f} Gy"
        )


def compute_bed(
    total_dose_gy: float,
    n_fractions: int,
    alpha_beta_gy: float = DEFAULT_ALPHA_BETA,
) -> BEDResult:
    """
    Compute the Biologically Effective Dose (BED).

    Parameters
    ----------
    total_dose_gy : float
        Total prescribed dose D in Gray (Gy).
    n_fractions : int
        Number of treatment fractions.
    alpha_beta_gy : float
        α/β ratio in Gy (default 10 Gy for cardiac SBRT / tumour tissue).

    Returns
    -------
    BEDResult
        Dataclass containing BED, EQD2, and all input parameters.

    Raises
    ------
    ValueError
        If any input parameter is physically invalid.

    Examples
    --------
    >>> result = compute_bed(total_dose_gy=25.0, n_fractions=5)
    >>> round(result.bed_gy, 2)
    37.5
    """
    if total_dose_gy <= 0:
        raise ValueError(f"total_dose_gy must be > 0, got {total_dose_gy}")
    if n_fractions < 1:
        raise ValueError(f"n_fractions must be >= 1, got {n_fractions}")
    if alpha_beta_gy <= 0:
        raise ValueError(f"alpha_beta_gy must be > 0, got {alpha_beta_gy}")

    d = total_dose_gy / n_fractions          # dose per fraction
    bed = total_dose_gy * (1.0 + d / alpha_beta_gy)

    # EQD2: equivalent dose delivered in 2 Gy fractions
    # EQD2 = BED / (1 + 2 / (α/β))
    eqd2 = bed / (1.0 + 2.0 / alpha_beta_gy)

    return BEDResult(
        total_dose_gy=total_dose_gy,
        dose_per_fraction_gy=round(d, 4),
        n_fractions=n_fractions,
        alpha_beta_gy=alpha_beta_gy,
        bed_gy=round(bed, 4),
        eqd2_gy=round(eqd2, 4),
    )


def validate_sbrt_bed(
    total_dose_gy: float,
    n_fractions: int,
    alpha_beta_gy: float = DEFAULT_ALPHA_BETA,
    min_bed_gy: float = 100.0,
) -> tuple[BEDResult, bool, str]:
    """
    Compute BED and validate that it meets the ablative threshold for cardiac SBRT.

    Parameters
    ----------
    total_dose_gy : float
        Total prescribed dose in Gy.
    n_fractions : int
        Number of fractions.
    alpha_beta_gy : float
        α/β ratio (default 10 Gy).
    min_bed_gy : float
        Minimum BED (Gy) required to be considered ablative (default 100 Gy).

    Returns
    -------
    (BEDResult, is_valid: bool, message: str)
    """
    result = compute_bed(total_dose_gy, n_fractions, alpha_beta_gy)
    is_valid = result.bed_gy >= min_bed_gy
    message = (
        f"BED {result.bed_gy:.1f} Gy {'≥' if is_valid else '<'} "
        f"ablative threshold {min_bed_gy} Gy — "
        f"{'PASS' if is_valid else 'FAIL'}"
    )
    return result, is_valid, message


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Typical cardiac SBRT regimen: 25 Gy in 1 fraction
    r = compute_bed(total_dose_gy=25.0, n_fractions=1)
    print(r)

    # 5-fraction SBRT regimen
    r2 = compute_bed(total_dose_gy=50.0, n_fractions=5)
    print(r2)

    # Validation
    _, ok, msg = validate_sbrt_bed(25.0, 1)
    print(msg)
