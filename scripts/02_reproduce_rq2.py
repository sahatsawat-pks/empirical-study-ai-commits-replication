#!/usr/bin/env python3
"""
02_reproduce_rq2.py

Replication Script for Research Question 2 (RQ2):
"Can behavioral features alone distinguish AI from human commits, independent
 of the attribution rules used to build the labels?"

This script reproduces:
1. The 26 pure behavioral features definition and schema (Table 4).
2. Within-repository 80/20 train/test split (17,450 train / 4,360 held-out test).
3. 5-fold Stratified Cross-Validation on the training set.
4. Model performance comparison for Logistic Regression, Random Forest, and XGBoost (Table 8).
5. Detailed Confusion Matrix for the XGBoost classifier (Table 9).
6. Correlation analysis between behavioral features and ground-truth labels (Table 12).
7. High-resolution SHAP feature importance beeswarm plot (Figure 5).
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

def main():
    print("=" * 80)
    print(" REPRODUCING RQ2: BEHAVIORAL CLASSIFICATION (NO IDENTITY METADATA)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    fig_dir = base_dir / "figures"
    results_dir = base_dir / "results"
    fig_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    data_file = data_dir / "featured_dataset_with_comments.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing master dataset: {data_file}")

    print(f"\n[Step 1] Loading master dataset from {data_file.name}...")
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df):,} commits across {df['repo_full_name'].nunique()} repositories.")
    print(f"  Class balance: {df['validated_is_ai'].value_counts().to_dict()} (0=Human, 1=AI)")

    # -------------------------------------------------------------------------
    # 26 Behavioral Features Definition
    # -------------------------------------------------------------------------
    features_26 = [
        "message_length", "has_multiple_sentences", "has_issue_reference",
        "is_bugfix", "is_feature", "is_refactor", "is_perf", "is_doc", "is_test",
        "hour_of_day", "day_of_week", "is_weekend", "is_after_hours", "temporal_consistency",
        "additions", "deletions", "total_changed", "files_changed", "avg_lines_per_file",
        "addition_ratio", "size_category",
        "has_added_files", "has_deleted_files", "has_renamed_files", "is_pure_deletion", "is_bulk_change"
    ]
    target = "validated_is_ai"

    missing = [f for f in features_26 if f not in df.columns]
    assert not missing, f"Missing features: {missing}"

    # -------------------------------------------------------------------------
    # Feature Correlations with AI Ground Truth (Table 12)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Bivariate Feature Correlations with AI Ground-Truth Label (Table 12):")
    corr_records = []
    y_full = df[target].values

    for col in features_26:
        # For correlation, use numeric encoding if categorical
        if col in ["day_of_week", "size_category"]:
            s = pd.factorize(df[col])[0]
        else:
            s = df[col].fillna(0).values

        tau, _ = kendalltau(s, y_full)
        r, _ = pointbiserialr(s, y_full)
        direction = "Positive (Higher in AI)" if tau > 0 else "Negative (Higher in Human)"
        corr_records.append({
            "Feature": col,
            "Kendall_tau": round(float(tau), 4),
            "Point_Biserial_r": round(float(r), 4),
            "Direction": direction
        })

    df_corr = pd.DataFrame(corr_records).sort_values(by="Kendall_tau", key=abs, ascending=False)
    print(df_corr.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # Within-Repository 80/20 Train / Held-Out Test Split
    # -------------------------------------------------------------------------
    print("\n[Step 3] Partitioning Data via Within-Repository 80/20 Stratified Split...")
    train_indices = []
    test_indices = []

    for (repo, label), group in df.groupby(["repo_full_name", target]):
        n = len(group)
        n_test = int(np.round(n * 0.2))
        shuffled = group.sample(frac=1.0, random_state=42).index
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    print(f"  Training Set: {len(train_df):,} commits (AI: {(train_df[target]==1).sum():,}, Human: {(train_df[target]==0).sum():,})")
    print(f"  Held-out Test: {len(test_df):,} commits (AI: {(test_df[target]==1).sum():,}, Human: {(test_df[target]==0).sum():,})")

    X_train = train_df[features_26].copy()
    y_train = train_df[target].values
    X_test = test_df[features_26].copy()
    y_test = test_df[target].values

    # Preprocessing: log1p on skewed numeric columns
    skewed_cols = ["message_length", "additions", "deletions", "total_changed", "files_changed", "avg_lines_per_file"]
    for col in skewed_cols:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

    uniform_numeric = ["hour_of_day", "temporal_consistency", "addition_ratio"] + skewed_cols
    categorical = ["day_of_week", "size_category"]
    binary = [f for f in features_26 if f not in uniform_numeric and f not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), uniform_numeric),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical),
            ("bin", "passthrough", binary),
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical).tolist()
    all_feature_names = uniform_numeric + cat_names + binary

    # -------------------------------------------------------------------------
    # Model Training & Cross-Validation (Table 8)
    # -------------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=30, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1)
    }

    print("\n[Step 4] Running 5-Fold Stratified Cross-Validation & Held-Out Test Evaluation (Table 8):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_eval_records = []
    fitted_models = {}

    for name, model in models.items():
        # 5-fold CV on train set
        cv_scores = cross_validate(model, X_train_trans, y_train, cv=cv, scoring=["f1_macro", "roc_auc", "accuracy"])
        mean_cv_f1 = np.mean(cv_scores["test_f1_macro"])
        std_cv_f1 = np.std(cv_scores["test_f1_macro"])

        # Fit full train set and evaluate held-out test set
        model.fit(X_train_trans, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else y_pred

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_prob)
        prec_ai = precision_score(y_test, y_pred, pos_label=1)
        rec_ai = recall_score(y_test, y_pred, pos_label=1)
        acc = accuracy_score(y_test, y_pred)

        model_eval_records.append({
            "Model": name,
            "5-fold CV F1": f"{mean_cv_f1:.4f} +/- {std_cv_f1:.4f}",
            "Macro F1": round(macro_f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Precision (AI)": round(prec_ai, 4),
            "Recall (AI)": round(rec_ai, 4),
            "Accuracy": f"{acc * 100:.2f}%"
        })

    df_models = pd.DataFrame(model_eval_records)
    print(df_models.to_string(index=False))

    # -------------------------------------------------------------------------
    # Confusion Matrix for XGBoost (Table 9)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Confusion Matrix for Best-Performing Model (XGBoost, Table 9):")
    xgb = fitted_models["XGBoost"]
    y_pred_xgb = xgb.predict(X_test_trans)
    cm = confusion_matrix(y_test, y_pred_xgb)
    tn, fp, fn, tp = cm.ravel()

    cm_df = pd.DataFrame([
        {"Actual Class": "Human", "Predicted Human": f"{tn:,} (TN)", "Predicted AI": f"{fp:,} (FP)", "Class Total": f"{tn+fp:,}"},
        {"Actual Class": "AI", "Predicted Human": f"{fn:,} (FN)", "Predicted AI": f"{tp:,} (TP)", "Class Total": f"{fn+tp:,}"},
        {"Actual Class": "Total", "Predicted Human": f"{tn+fn:,}", "Predicted AI": f"{fp+tp:,}", "Class Total": f"{len(y_test):,}"}
    ])
    print(cm_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # SHAP Summary Beeswarm Plot (Figure 5)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating 300 DPI SHAP Summary Plot (Figure 5)...")
    explainer = shap.TreeExplainer(xgb)
    # Representative sample of 2,000 test instances for crisp visual density
    sample_indices = np.random.RandomState(42).choice(len(X_test_trans), size=min(2000, len(X_test_trans)), replace=False)
    shap_vals = explainer.shap_values(X_test_trans[sample_indices])

    plt.figure(figsize=(9, 8))
    shap.summary_plot(
        shap_vals,
        features=pd.DataFrame(X_test_trans[sample_indices], columns=all_feature_names),
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance (XGBoost Classifier)", fontsize=13, pad=15)
    plt.tight_layout()
    shap_out = fig_dir / "shap_summary.png"
    plt.savefig(shap_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved high-resolution figure to: {shap_out}")

    # -------------------------------------------------------------------------
    # Save Structured Results
    # -------------------------------------------------------------------------
    out_json = results_dir / "rq2_reproduction_results.json"
    results_payload = {
        "dataset_split": {
            "total_commits": len(df),
            "train_commits": len(train_df),
            "test_commits": len(test_df),
            "train_ai": int((train_df[target] == 1).sum()),
            "train_human": int((train_df[target] == 0).sum()),
            "test_ai": int((test_df[target] == 1).sum()),
            "test_human": int((test_df[target] == 0).sum())
        },
        "model_performance": model_eval_records,
        "xgboost_confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "top_feature_correlations": corr_records[:10]
    }

    with open(out_json, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[OK] RQ2 reproduction completed successfully. Summary saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
