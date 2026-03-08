# fraud_detection.py
# Vagif Novruzov - March/April 2026
#
# I started this after reading about how most fraud tutorials just
# train a classifier and call it done. The interesting part to me
# is that fraud detection is fundamentally a *threshold* problem,
# not an accuracy problem. Getting that framing right drove most
# of the decisions below.
#
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the same directory before running.

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -- paths & reproducibility ----------------------------------------------------
DATA_PATH   = "creditcard.csv"
FIGURES_DIR = "figures"
SEED        = 42            # fix this everywhere so results are reproducible

os.makedirs(FIGURES_DIR, exist_ok=True)


# -- 1. load & sanity-check -----------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    n_total = len(df)
    n_fraud = df["Class"].sum()
    fraud_pct = n_fraud / n_total * 100

    print(f"Loaded {n_total:,} transactions - {n_fraud} fraudulent ({fraud_pct:.3f}%)")
    print(f"Missing values: {df.isnull().sum().sum()}")  # should be 0

    # Quick sanity check: every V-feature should already be scaled (PCA output)
    v_cols = [c for c in df.columns if c.startswith("V")]
    means  = df[v_cols].mean().abs()
    if means.max() > 1:
        print("Warning: some V-features have large means - double-check preprocessing")

    return df


# -- 2. exploratory data analysis ----------------------------------------------
#
# Three things I wanted to understand before touching a model:
#   (a) how bad is the imbalance visually?
#   (b) do fraud transactions look different by amount or time?
#   (c) which PCA features correlate most with the fraud label?
#
# (c) won't tell me causation but it narrows down where signal lives.

def run_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Credit Card Fraud - Exploratory Overview", fontsize=14, fontweight="bold")

    # (a) class balance - the visual really hammers home how skewed this is
    counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=["#2196F3", "#F44336"])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Transactions")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

    # (b) amount distribution - fraud transactions tend to be smaller, which
    # surprised me initially; turns out small frauds are harder to detect
    df[df["Class"] == 0]["Amount"].hist(
        ax=axes[1], bins=80, alpha=0.6, color="#2196F3", label="Legitimate", density=True
    )
    df[df["Class"] == 1]["Amount"].hist(
        ax=axes[1], bins=80, alpha=0.6, color="#F44336", label="Fraud", density=True
    )
    axes[1].set_xlim(0, 2500)
    axes[1].set_title("Amount Distribution")
    axes[1].set_xlabel("Amount (USD)")
    axes[1].legend()

    # (c) scatter over time - no obvious temporal cluster for fraud,
    # which rules out a simple time-based rule
    axes[2].scatter(
        df[df["Class"] == 0]["Time"], df[df["Class"] == 0]["Amount"],
        alpha=0.01, s=1, color="#2196F3", label="Legitimate"
    )
    axes[2].scatter(
        df[df["Class"] == 1]["Time"], df[df["Class"] == 1]["Amount"],
        alpha=0.4, s=10, color="#F44336", label="Fraud"
    )
    axes[2].set_title("Transactions Over Time")
    axes[2].set_xlabel("Time (seconds from first transaction)")
    axes[2].set_ylabel("Amount (USD)")
    axes[2].legend()

    plt.tight_layout()
    path = f"{FIGURES_DIR}/01_eda_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    # Simple Pearson correlation - not a substitute for model-based importance
    # but useful for spotting obvious linear relationships quickly
    corr = df.corr()["Class"].drop("Class").sort_values()

    plt.figure(figsize=(10, 6))
    colors = ["#F44336" if v < 0 else "#2196F3" for v in corr.values]
    corr.plot(kind="barh", color=colors)
    plt.title("Feature Correlation with Fraud Label", fontweight="bold")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    path = f"{FIGURES_DIR}/02_feature_correlations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# -- 3. preprocessing ----------------------------------------------------------
#
# V1–V28 are already PCA-transformed by the dataset authors, so I only need
# to scale Amount and Time. I use separate scaler instances to avoid
# accidentally fitting on test data later.

def preprocess(df: pd.DataFrame):
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Amount", "Time"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratify=y is important here - without it, a random split could
    # put almost all fraud cases in train or test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud)")
    print(f"Test : {len(X_test):,} rows ({y_test.sum()} fraud)")
    return X_train, X_test, y_train, y_test


# -- 4. model pipelines --------------------------------------------------------
#
# Key decision: SMOTE goes *inside* the pipeline, not before the split.
# If I apply SMOTE first and then split, synthetic fraud samples could
# end up in both train and test - data leakage that inflates recall scores.
# Using imblearn.Pipeline keeps it contained to training folds only.

def build_pipelines() -> dict:
    smote = SMOTE(random_state=SEED)

    return {
        # Logistic Regression: baseline. class_weight="balanced" is an
        # alternative to SMOTE - I use both together here to see if stacking
        # them helps or hurts (turns out marginal difference in practice)
        "Logistic Regression": ImbPipeline([
            ("smote", smote),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED
            ))
        ]),

        # Random Forest: good at capturing non-linear patterns in V-features,
        # and gives us feature importances for free
        "Random Forest": ImbPipeline([
            ("smote", smote),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=SEED,
                n_jobs=-1
            ))
        ]),

        # XGBoost: scale_pos_weight sets internal class weighting;
        # 577 ≈ ratio of negatives to positives in training set
        "XGBoost": ImbPipeline([
            ("smote", smote),
            ("clf", XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                scale_pos_weight=577,
                random_state=SEED,
                eval_metric="aucpr",
                verbosity=0
            ))
        ]),
    }


def train_and_evaluate(pipelines: dict, X_train, X_test, y_train, y_test) -> dict:
    results = {}

    for name, pipe in pipelines.items():
        print(f"\n-- {name} --")
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        roc_auc  = roc_auc_score(y_test, y_proba)
        pr_auc   = average_precision_score(y_test, y_proba)

        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
        print(f"  ROC-AUC : {roc_auc:.4f}")
        print(f"  PR-AUC  : {pr_auc:.4f}")

        results[name] = {
            "pipe": pipe, "y_pred": y_pred, "y_proba": y_proba,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
        }

    return results


# -- 5. visualisations ---------------------------------------------------------

def plot_confusion_matrices(results: dict, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"]
        )
        ax.set_title(f"{name}\nROC-AUC: {res['roc_auc']:.3f}", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/03_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_roc_pr_curves(results: dict, y_test):
    # PR-AUC is more informative than ROC-AUC for heavily imbalanced data.
    # A model can have high ROC-AUC just by being good at the easy majority class.
    # PR-AUC forces the model to actually be precise on the minority class.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ["#2196F3", "#4CAF50", "#FF9800"]

    for (name, res), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0].plot(fpr, tpr, label=f"{name} ({res['roc_auc']:.3f})", color=color, lw=2)

        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        axes[1].plot(rec, prec, label=f"{name} ({res['pr_auc']:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend()

    baseline = y_test.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", lw=1, label=f"No-skill ({baseline:.3f})")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.suptitle("ROC & Precision-Recall Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/04_roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_feature_importance(results: dict, feature_names):
    if "Random Forest" not in results:
        return

    rf = results["Random Forest"]["pipe"].named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20).sort_values()

    plt.figure(figsize=(10, 7))
    top20.plot(kind="barh", color="#2196F3", edgecolor="white")
    plt.title("Top 20 Feature Importances (Random Forest)", fontweight="bold")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()

    path = f"{FIGURES_DIR}/05_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_model_comparison(results: dict):
    names     = list(results.keys())
    roc_vals  = [r["roc_auc"] for r in results.values()]
    pr_vals   = [r["pr_auc"]  for r in results.values()]
    x         = np.arange(len(names))
    w         = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, roc_vals, w, label="ROC-AUC", color="#2196F3")
    b2 = ax.bar(x + w / 2, pr_vals,  w, label="PR-AUC",  color="#4CAF50")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()

    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    path = f"{FIGURES_DIR}/06_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# -- 6. business impact --------------------------------------------------------
#
# This is the part most tutorials skip. Raw F1 scores don't mean much
# to a fraud team - they care about dollars recovered and analyst workload.
# I use the average fraud amount from the dataset as a rough proxy.

def business_impact(results: dict, y_test, avg_fraud_amount: float = 122.21):
    best = max(results, key=lambda k: results[k]["pr_auc"])
    cm   = confusion_matrix(y_test, results[best]["y_pred"])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n-- Business impact simulation ({best}) --")
    print(f"  Fraud caught   (TP): {tp:>5}   → ~${tp * avg_fraud_amount:,.0f} recovered")
    print(f"  Fraud missed   (FN): {fn:>5}   → ~${fn * avg_fraud_amount:,.0f} lost")
    print(f"  False alerts   (FP): {fp:>5}   (legitimate transactions flagged)")
    print(f"  Correct clears (TN): {tn:>5}")
    print(f"\n  Recovery rate: {tp / (tp + fn) * 100:.1f}%")
    print(f"  Analyst workload: {fp + tp} flagged transactions per {len(y_test):,} processed")


# -- main ----------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data(DATA_PATH)

    run_eda(df)
    plot_correlation_heatmap(df)

    X_train, X_test, y_train, y_test = preprocess(df)

    pipelines = build_pipelines()
    results   = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)

    plot_confusion_matrices(results, y_test)
    plot_roc_pr_curves(results, y_test)
    plot_feature_importance(results, X_train.columns)
    plot_model_comparison(results)

    business_impact(results, y_test)

    print("\nDone - figures saved to /figures")