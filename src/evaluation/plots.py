"""
CardioSurv — Evaluation Plots
=============================

T-B: TAN GUAN HAN

Saves publication-quality PNGs to reports/figures/ for use in the report
and slides. All four plot functions accept a `save_path` and return the
matplotlib Figure so notebooks can display inline as well.

Spec source: docs/tasks_2.pdf §B
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from src.evaluation.metrics import PALETTE

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality defaults
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm,
    labels: Sequence[str] = ("Low", "Medium", "High"),
    save_path: str | Path = FIG_DIR / "confusion_matrix.png",
):
    """
    Annotated heatmap of a confusion matrix.

    Parameters
    ----------
    cm : 2D array-like — rows=true, cols=pred.
    labels : class names in row/column order.
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm)
    labels = list(labels)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title("Risk Classifier — Confusion Matrix", pad=12)
    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Actual Risk")

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 2. ROC curve — one-vs-rest, micro + macro average
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true,
    y_proba,
    labels: Sequence[str] = ("Low", "Medium", "High"),
    save_path: str | Path = FIG_DIR / "roc_curve.png",
):
    """
    Multi-class ROC curve (one-vs-rest), with micro + macro averages.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) — true class labels.
    y_proba : array-like of shape (n_samples, n_classes)
        — predicted probabilities aligned with `labels` ordering.
    labels : ordered class names.
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(labels)
    y_proba = np.asarray(y_proba)
    y_bin = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-class
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        fpr_dict[lbl], tpr_dict[lbl], auc_dict[lbl] = fpr, tpr, roc_auc

        ax.plot(
            fpr,
            tpr,
            label=f"{lbl} (AUC = {roc_auc:.3f})",
            color=PALETTE.get(lbl, None),
            linewidth=2,
        )

    # Micro-average (flatten all classes)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    ax.plot(
        fpr_micro,
        tpr_micro,
        label=f"micro-average (AUC = {auc_micro:.3f})",
        color=PALETTE["primary"],
        linestyle=":",
        linewidth=2,
    )

    # Macro-average (mean of per-class curves on a common FPR grid)
    all_fpr = np.unique(np.concatenate([fpr_dict[l] for l in labels]))

    mean_tpr = np.zeros_like(all_fpr)
    for lbl in labels:
        mean_tpr += np.interp(all_fpr, fpr_dict[lbl], tpr_dict[lbl])

    mean_tpr /= len(labels)
    auc_macro = auc(all_fpr, mean_tpr)

    ax.plot(
        all_fpr,
        mean_tpr,
        label=f"macro-average (AUC = {auc_macro:.3f})",
        color=PALETTE["secondary"],
        linestyle="--",
        linewidth=2,
    )

    # Reference diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color="grey",
        linestyle="-",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Risk Classifier — ROC Curves (One-vs-Rest)", pad=12)

    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. Kaplan-Meier curves stratified by treatment branch
# ---------------------------------------------------------------------------


def plot_km_curves(
    kmf_dict: dict,
    save_path: str | Path = FIG_DIR / "km_curves.png",
):
    """
    Plot fitted Kaplan-Meier curves with 95 % confidence bands.

    Parameters
    ----------
    kmf_dict : mapping of label -> fitted lifelines.KaplanMeierFitter.
        e.g. {'SBRT': kmf_sbrt, 'Medication': kmf_med}
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Stable colour assignment — branch names map to palette entries
    # when possible.
    branch_colours = {
        "SBRT": PALETTE["High"],
        "Medication": PALETTE["primary"],
    }

    for name, kmf in kmf_dict.items():
        colour = branch_colours.get(name, None)

        kmf.plot_survival_function(
            label=name,
            ci_show=True,
            ax=ax,
            color=colour,
            linewidth=2,
        )

    ax.set_title("Kaplan-Meier Survival — by Treatment Branch", pad=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")

    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Feature importance — top 15 horizontal bars
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str | Path = FIG_DIR / "feature_importance.png",
):
    """
    Horizontal bar chart of the top-15 most important features.

    Parameters
    ----------
    importance_df : DataFrame with columns ['feature','importance'].
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not {"feature", "importance"}.issubset(importance_df.columns):
        raise ValueError(
            "importance_df must have columns ['feature','importance']"
        )

    # Get top 15, already sorted in descending order
    top = (
        importance_df.sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use ax.barh directly instead of sns.barplot to avoid automatic sorting
    # Plot in ascending order so highest importance is at the top
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["importance"].values, color=PALETTE["primary"])
    
    # Set labels in correct order (highest at top)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].values)
    
    # Reverse y-axis so highest importance is at top
    ax.invert_yaxis()

    ax.set_title("XGBoost — Top 15 Feature Importances", pad=12)
    ax.set_xlabel("Importance (gain)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")

    return fig
