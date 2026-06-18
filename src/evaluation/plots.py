"""
CardioSurv — Evaluation Plots
==============================
T-B: TAN GUAN HAN

Saves publication-quality PNGs to reports/figures/ for use in the report
and slides. All four plot functions accept a `save_path` and return the
matplotlib Figure so notebooks can display inline as well.

plot_confusion_matrix() and plot_feature_importance() are model-aware:
pass model_key="rf" or model_key="xgb" to get a correctly titled and
correctly named figure for each classifier. Calling both functions once
per model produces the four comparison figures used in the report:

    reports/figures/confusion_matrix_rf.png
    reports/figures/confusion_matrix_xgb.png
    reports/figures/feature_importance_rf.png
    reports/figures/feature_importance_xgb.png

model_key is keyword-only so existing positional calls
(cm, labels, save_path) and (importance_df, save_path) still work
unchanged and default to the Random Forest figure names/titles.

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
plt.rcParams.update({
    "figure.dpi":   110,
    "savefig.dpi":  200,
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Display name + importance-axis label per model. RF importances are
# mean decrease in impurity (Gini); XGBoost's default feature_importances_
# is gain-based — these are not the same quantity, so the axis label
# is kept model-specific to avoid mislabeling either figure.
MODEL_INFO = {
    "rf":  {"display_name": "Random Forest", "importance_label": "Importance (mean decrease in impurity)"},
    "xgb": {"display_name": "XGBoost",        "importance_label": "Importance (gain)"},
}


def _model_info(model_key: str) -> dict:
    return MODEL_INFO.get(model_key, {"display_name": model_key, "importance_label": "Importance"})


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm,
    labels: Sequence[str] = ("Low", "Medium", "High"),
    save_path: str | Path | None = None,
    *,
    model_key: str = "rf",
):
    """
    Annotated heatmap of a confusion matrix for a single model.

    Call once per model (model_key="rf" and model_key="xgb") to produce
    the two comparison figures.

    Parameters
    ----------
    cm        : 2D array-like — rows=true, cols=pred.
    labels    : class names in row/column order.
    save_path : where to write the PNG. Defaults to
                reports/figures/confusion_matrix_{model_key}.png.
    model_key : "rf" or "xgb" — controls the title and default filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm)
    labels = list(labels)
    info = _model_info(model_key)

    if save_path is None:
        save_path = FIG_DIR / f"confusion_matrix_{model_key}.png"

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
    ax.set_title(f"{info['display_name']} — Confusion Matrix", pad=12)
    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Actual Risk")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plots] Saved {save_path}")
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
    y_true    : array-like of shape (n_samples,) — true class labels.
    y_proba   : array-like of shape (n_samples, n_classes) — predicted
                probabilities aligned with `labels` ordering.
    labels    : ordered class names.
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels  = list(labels)
    y_proba = np.asarray(y_proba)
    y_bin   = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-class
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[lbl], tpr_dict[lbl], auc_dict[lbl] = fpr, tpr, roc_auc
        ax.plot(
            fpr, tpr,
            label=f"{lbl}  (AUC = {roc_auc:.3f})",
            color=PALETTE.get(lbl, None),
            linewidth=2,
        )

    # Micro-average (flatten all classes)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro, tpr_micro,
        label=f"micro-average  (AUC = {auc_micro:.3f})",
        color=PALETTE["primary"],
        linestyle=":", linewidth=2,
    )

    # Macro-average (mean of per-class curves on a common FPR grid)
    all_fpr = np.unique(np.concatenate([fpr_dict[l] for l in labels]))
    mean_tpr = np.zeros_like(all_fpr)
    for lbl in labels:
        mean_tpr += np.interp(all_fpr, fpr_dict[lbl], tpr_dict[lbl])
    mean_tpr /= len(labels)
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr, mean_tpr,
        label=f"macro-average  (AUC = {auc_macro:.3f})",
        color=PALETTE["secondary"],
        linestyle="--", linewidth=2,
    )

    # Reference diagonal
    ax.plot([0, 1], [0, 1], color="grey", linestyle="-", linewidth=1, alpha=0.6)

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
    kmf_dict  : mapping of label -> fitted lifelines.KaplanMeierFitter.
                e.g. {'SBRT': kmf_sbrt, 'Medication': kmf_med}
    save_path : where to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Stable colour assignment — branch names map to palette entries when possible
    branch_colours = {
        "SBRT":       PALETTE["High"],
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
    save_path: str | Path | None = None,
    *,
    model_key: str = "rf",
):
    """
    Horizontal bar chart of the top-15 most important features for a
    single model.

    Call once per model (model_key="rf" and model_key="xgb") to produce
    the two comparison figures. The x-axis label is model-specific since
    Random Forest importances (mean decrease in impurity) and XGBoost's
    default importances (gain) are different quantities and shouldn't be
    presented under a shared label.

    Parameters
    ----------
    importance_df : DataFrame with columns ['feature','importance'].
    save_path     : where to write the PNG. Defaults to
                     reports/figures/feature_importance_{model_key}.png.
    model_key     : "rf" or "xgb" — controls the title, x-axis label, and
                     default filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not {"feature", "importance"}.issubset(importance_df.columns):
        raise ValueError("importance_df must have columns ['feature','importance']")

    info = _model_info(model_key)
    if save_path is None:
        save_path = FIG_DIR / f"feature_importance_{model_key}.png"

    top = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(15)
        .iloc[::-1]                       # reverse so largest is on top of the bar chart
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(
        data=top,
        x="importance",
        y="feature",
        color=PALETTE["primary"],
        ax=ax,
    )
    ax.set_title(f"{info['display_name']} — Top 15 Feature Importances", pad=12)
    ax.set_xlabel(info["importance_label"])
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[plots] Saved {save_path}")
    return fig
