"""
CardioSurv — Evaluation Metrics
================================
T-B: TAN GUAN HAN

Computes classification + survival evaluation metrics as plain dicts.
Used by:
  - notebooks/02_part1_modeling.ipynb (Part 1 risk classifier)
  - notebooks/03_part2_modeling.ipynb (Part 2 intervention recommender + Cox PH)
  - report figures

Spec source: docs/tasks_2.pdf §B
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


# ---------------------------------------------------------------------------
# Shared colour palette — imported by plots.py so figures stay consistent
# ---------------------------------------------------------------------------

PALETTE = {
    "Low":       "#2ecc71",   # green
    "Medium":    "#f39c12",   # orange
    "High":      "#e74c3c",   # red
    "primary":   "#3498db",   # blue
    "secondary": "#9b59b6",   # purple
}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_proba, labels=("Low", "Medium", "High")) -> dict:
    """
    Compute multi-class classification metrics.

    Parameters
    ----------
    y_true   : array-like of shape (n_samples,) — ground-truth class labels.
    y_pred   : array-like of shape (n_samples,) — predicted class labels.
    y_proba  : array-like of shape (n_samples, n_classes) — predicted
               class probabilities (rows sum to 1).
    labels   : ordered class names, default ('Low','Medium','High').

    Returns
    -------
    dict with keys:
        accuracy          : float
        f1_macro          : float
        auc_ovr           : float   (one-vs-rest macro AUC)
        per_class_f1      : dict[label -> float]
        confusion_matrix  : nested list, rows=true, cols=pred
    """
    labels = list(labels)

    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # AUC one-vs-rest. sklearn requires the `labels` kwarg to be sorted, but
    # our `labels` reflects clinical ordering (Low < Medium < High). The columns
    # of `y_proba` are assumed to align with `labels`, so we reorder them to
    # match sklearn's sorted-class expectation before calling roc_auc_score.
    try:
        y_proba_arr = np.asarray(y_proba)
        sorted_labels = sorted(labels)
        reorder = [labels.index(lbl) for lbl in sorted_labels]
        y_proba_sorted = y_proba_arr[:, reorder]
        auc_ovr = roc_auc_score(
            y_true, y_proba_sorted,
            multi_class="ovr", labels=sorted_labels,
        )
    except ValueError:
        # e.g. a class is missing from y_true in a tiny test split
        auc_ovr = float("nan")

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    per_class_f1 = {lbl: round(float(score), 4) for lbl, score in zip(labels, f1_per_class)}

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy":         round(float(acc), 4),
        "f1_macro":         round(float(f1_macro), 4),
        "auc_ovr":          round(float(auc_ovr), 4) if not np.isnan(auc_ovr) else None,
        "per_class_f1":     per_class_f1,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Survival metrics
# ---------------------------------------------------------------------------

def survival_metrics(durations, events, predicted_scores, group_labels=None) -> dict:
    """
    Compute survival-analysis metrics.

    Parameters
    ----------
    durations         : observed times (days or months).
    events            : 1 if event observed, 0 if censored.
    predicted_scores  : risk scores (higher = worse prognosis), e.g. Cox
                        partial hazard.
    group_labels      : optional array of group assignments (e.g. 'SBRT' /
                        'Medication'). If provided, a log-rank test is run
                        between the first two distinct groups.

    Returns
    -------
    dict with keys:
        concordance_index : float — Harrell's C-index. Higher is better,
                            0.5 = random, 1.0 = perfect.
        log_rank_p        : float | None — log-rank p-value between groups,
                            or None if group_labels not supplied or only one
                            group is present.
    """
    durations        = np.asarray(durations, dtype=float)
    events           = np.asarray(events, dtype=int)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    # lifelines' concordance_index expects higher score = longer survival,
    # so we negate the risk scores.
    c_index = concordance_index(durations, -predicted_scores, events)

    log_rank_p = None
    if group_labels is not None:
        group_labels = np.asarray(group_labels)
        unique_groups = np.unique(group_labels)
        if len(unique_groups) >= 2:
            g1, g2 = unique_groups[0], unique_groups[1]
            mask1 = group_labels == g1
            mask2 = group_labels == g2
            result = logrank_test(
                durations[mask1], durations[mask2],
                event_observed_A=events[mask1],
                event_observed_B=events[mask2],
            )
            log_rank_p = round(float(result.p_value), 4)

    return {
        "concordance_index": round(float(c_index), 4),
        "log_rank_p":        log_rank_p,
    }
