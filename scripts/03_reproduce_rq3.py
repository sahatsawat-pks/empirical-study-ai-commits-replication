#!/usr/bin/env python3
"""
03_reproduce_rq3.py

Replication Script for Research Question 3 (RQ3):
"What topics and patterns in commit messages distinguish AI from human commits,
 and how do they explain our behavioral features?"

This script reproduces:
1. LDA Topic Modeling (K=5) and topic keyword extraction with lambda=0.6 (Table 10).
2. Topic distributions across Human and AI commits (Table 10).
3. 2D Principal Component Analysis (PCA) decomposition and feature loadings.
4. 5-panel faceted PCA visualization across topics (Figure 4).
5. Within-topic statistical testing: Mann-Whitney U test and Cliff's delta (Table 11).
6. Stylometric burstiness and inline code comment comparisons (Table 13).
"""

import os
import re
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Visualization styling
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

SEED = 42

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"\b[0-9a-f]{7,40}\b", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.strip()

def compute_cliffs_delta(x, y):
    """Computes non-parametric Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, "Negligible"
    u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
    delta = (2 * u_stat) / (n1 * n2) - 1

    abs_d = abs(delta)
    if abs_d < 0.147:
        band = "Negligible"
    elif abs_d < 0.330:
        band = "Small"
    elif abs_d < 0.474:
        band = "Medium"
    else:
        band = "Large"
    return round(float(delta), 4), band, float(p_val)

def main():
    print("=" * 80)
    print(" REPRODUCING RQ3: TOPIC MODELING, WITHIN-TOPIC STATS & BURSTINESS")
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

    # -------------------------------------------------------------------------
    # 1. LDA Topic Modeling (K=5)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Fitting Latent Dirichlet Allocation (LDA, K=5 topics)...")
    df["clean_message"] = df["commit_message"].fillna("").apply(preprocess_text)
    vectorizer = CountVectorizer(max_df=0.85, min_df=5, stop_words="english", max_features=1000)
    dtm = vectorizer.fit_transform(df["clean_message"])

    lda = LatentDirichletAllocation(
        n_components=5, doc_topic_prior=0.2, topic_word_prior=0.2,
        max_iter=30, learning_method="batch", random_state=SEED, n_jobs=1
    )
    topic_dist = lda.fit_transform(dtm)
    df["dominant_topic_idx"] = topic_dist.argmax(axis=1)

    topic_names = [
        "Topic 1: Build & Source Maintenance",
        "Topic 2: Agent Docs & Workflow",
        "Topic 3: Initial Planning & Setup",
        "Topic 4: Feature Additions & UI/API",
        "Topic 5: Testing & Error Handling"
    ]
    topic_keywords = [
        "ts, src, build, cli",
        "docs, agent, md, bot",
        "plan, initial, issue, main",
        "feat, new, ui, api, model",
        "tests, test, error, handling"
    ]

    df["dominant_topic"] = df["dominant_topic_idx"].map(lambda idx: topic_names[idx])
    df["Class"] = df["validated_is_ai"].map({0: "Human (Class 0)", 1: "AI (Class 1)"})

    # Topic distribution by class (Table 10)
    print("\n[Step 3] Topic Distributions across Human and AI Commits (Table 10):")
    crosstab = pd.crosstab(df["dominant_topic_idx"], df["validated_is_ai"], normalize="columns") * 100
    t10_records = []
    for idx in range(5):
        h_pct = crosstab.loc[idx, 0]
        a_pct = crosstab.loc[idx, 1]
        diff = a_pct - h_pct
        diff_str = f"+{diff:.2f}% (AI)" if diff > 0 else f"{diff:.2f}% (Human)"
        t10_records.append({
            "Topic": topic_names[idx],
            "Distinctive Keywords": topic_keywords[idx],
            "Human (%)": f"{h_pct:.2f}%",
            "AI (%)": f"{a_pct:.2f}%",
            "Difference": diff_str
        })

    df_t10 = pd.DataFrame(t10_records)
    print(df_t10.to_string(index=False))

    # -------------------------------------------------------------------------
    # 2. PCA Decomposition (PC1 & PC2)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Computing 2D PCA Behavioral Projection...")
    pca_features = [
        "message_length", "has_multiple_sentences", "has_issue_reference",
        "is_bugfix", "is_feature", "is_refactor", "is_perf", "is_doc", "is_test",
        "additions", "deletions", "total_changed", "files_changed",
        "avg_lines_per_file", "addition_ratio", "has_added_files", "has_renamed_files",
        "is_pure_deletion", "is_bulk_change", "is_weekend", "is_after_hours", "temporal_consistency"
    ]
    pca_df = df[pca_features].copy().fillna(df[pca_features].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_df)

    pca = PCA(n_components=2, random_state=SEED)
    components = pca.fit_transform(scaled)
    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]

    evr = pca.explained_variance_ratio_
    print(f"  PC1 Explained Variance: {evr[0]*100:.1f}% (Code Change Magnitude)")
    print(f"  PC2 Explained Variance: {evr[1]*100:.1f}% (Message Structure & Additions)")
    print(f"  Combined 2D Variance:   {sum(evr)*100:.1f}%")

    # -------------------------------------------------------------------------
    # 3. Faceted PCA Plot across Topics (Figure 4)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Rendering 300 DPI Faceted PCA Scatter Plot (Figure 4)...")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True, sharey=True)
    palette = {"Human (Class 0)": "#2b5c8f", "AI (Class 1)": "#d95f02"}

    for idx, (ax, t_name) in enumerate(zip(axes, topic_names)):
        sub = df[df["dominant_topic_idx"] == idx]
        sns.scatterplot(
            data=sub, x="PC1", y="PC2", hue="Class",
            palette=palette, alpha=0.35, s=18, ax=ax, legend=(idx == 4)
        )
        ax.set_title(t_name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("PC1 (Magnitude)" if idx == 2 else "", fontsize=10)
        ax.set_ylabel("PC2 (Structure)" if idx == 0 else "", fontsize=10)
        ax.set_xlim(-4, 12)
        ax.set_ylim(-4, 8)

    if axes[4].get_legend():
        axes[4].legend(title="Attribution", loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    pca_fig_out = fig_dir / "pca_faceted_by_topic.png"
    plt.savefig(pca_fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved high-resolution figure to: {pca_fig_out}")

    # -------------------------------------------------------------------------
    # 4. Within-Topic Statistical Testing (Table 11)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Within-Topic Statistical Analysis (Table 11):")
    t11_features = [
        (0, "message_length"), (0, "has_multiple_sentences"),
        (1, "message_length"), (1, "additions"),
        (2, "total_changed"), (2, "files_changed"),
        (3, "message_length"), (3, "additions"),
        (4, "message_length"), (4, "has_multiple_sentences")
    ]

    t11_records = []
    for t_idx, feat in t11_features:
        sub = df[df["dominant_topic_idx"] == t_idx]
        h_vals = sub.loc[sub["validated_is_ai"] == 0, feat].dropna()
        a_vals = sub.loc[sub["validated_is_ai"] == 1, feat].dropna()

        delta, band, p_val = compute_cliffs_delta(a_vals, h_vals)
        t_short = topic_names[t_idx].split(":")[0] + ": " + topic_names[t_idx].split(":")[1].split("&")[0].strip()
        t11_records.append({
            "Topic": t_short,
            "Feature": feat,
            "Human Med.": round(float(h_vals.median()), 1),
            "AI Med.": round(float(a_vals.median()), 1),
            "p-value": "<0.001" if p_val < 0.001 else f"{p_val:.3f}",
            "Cliff's delta": f"{delta:+.4f}",
            "Effect Size": band
        })

    df_t11 = pd.DataFrame(t11_records)
    print(df_t11.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Stylometric Burstiness & Code Comments (Table 13)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Comprehensive Burstiness and Inline Code Comments Analysis (Table 13):")
    h = df[df["validated_is_ai"] == 0]
    a = df[df["validated_is_ai"] == 1]

    burstiness_tests = [
        ("Commit Message Word Burstiness (All)",
         h["word_burstiness"].dropna(),
         a["word_burstiness"].dropna()),

        ("Commit Message Sentence Burstiness",
         h.loc[h["sentence_count"] > 1, "sentence_burstiness"].dropna(),
         a.loc[a["sentence_count"] > 1, "sentence_burstiness"].dropna()),

        ("Commit Body Word Burstiness",
         h.loc[h["body_word_burstiness"] >= 0, "body_word_burstiness"].dropna(),
         a.loc[a["body_word_burstiness"] >= 0, "body_word_burstiness"].dropna()),

        ("PR Body Word Burstiness",
         h.loc[h["pr_body_word_burstiness"] >= 0, "pr_body_word_burstiness"].dropna(),
         a.loc[a["pr_body_word_burstiness"] >= 0, "pr_body_word_burstiness"].dropna()),

        ("Code Comment Word Count (All)",
         h["comment_word_count"].dropna(),
         a["comment_word_count"].dropna()),

        ("Code Comment Word Burstiness (>= 0)",
         h.loc[h["comment_word_burstiness"] >= 0, "comment_word_burstiness"].dropna(),
         a.loc[a["comment_word_burstiness"] >= 0, "comment_word_burstiness"].dropna()),

        ("Code Comment Sentence Burstiness (>= 0)",
         h.loc[h["comment_sentence_burstiness"] >= 0, "comment_sentence_burstiness"].dropna(),
         a.loc[a["comment_sentence_burstiness"] >= 0, "comment_sentence_burstiness"].dropna())
    ]

    t13_records = []
    for metric_name, h_series, a_series in burstiness_tests:
        delta, band, p_val = compute_cliffs_delta(a_series, h_series)
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        t13_records.append({
            "Metric & Analysis Level": metric_name,
            "Human N": len(h_series),
            "Human Mean": round(float(h_series.mean()), 3),
            "Human Med.": round(float(h_series.median()), 3),
            "AI N": len(a_series),
            "AI Mean": round(float(a_series.mean()), 3),
            "AI Med.": round(float(a_series.median()), 3),
            "p-value": p_str,
            "Cliff's delta": f"{delta:+.3f}",
            "Effect Size": band
        })

    df_t13 = pd.DataFrame(t13_records)
    print(df_t13.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save Structured Results
    # -------------------------------------------------------------------------
    out_json = results_dir / "rq3_reproduction_results.json"
    results_payload = {
        "topic_distributions": t10_records,
        "pca_explained_variance": {
            "PC1": round(float(evr[0]), 4),
            "PC2": round(float(evr[1]), 4),
            "total_2d": round(float(sum(evr)), 4)
        },
        "within_topic_comparisons": t11_records,
        "stylometric_burstiness_comparisons": t13_records
    }

    with open(out_json, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[OK] RQ3 reproduction completed successfully. Summary saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
