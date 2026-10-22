# Replication Package: Distinguishing Autonomous AI from Human Contributions via Behavioral Signatures

This replication package contains the complete dataset, preprocessing pipelines, model training code, and statistical analysis scripts supporting the paper:

> **"Can We Catch the Agent? Distinguishing Autonomous AI from Human Contributions in Open-Source Repositories via Behavioral Signatures"**  
> *Target Venue: Empirical Software Engineering (EMSE), Special Issue on Human-Centered AI Transformation for Software Engineering.*  
> *Replication Package Status: Anonymous & Double-Blind Ready.*

---

## 1. Directory Structure

```text
replication_package/
├── README.md                                  # This replication guide and documentation
├── LICENSE                                    # Open-source MIT License
├── requirements.txt                           # Pinned Python package dependencies
├── data/
│   ├── featured_dataset_with_comments.csv    # Consolidated master dataset (21,810 commits, 546 repos)
│   ├── ground_truth_sample_768.csv           # 768-commit dual-annotated sample for RQ1 validation
│   ├── positive_control_revalidation_summary.csv # Positive-control revalidation summary (313,927 commits)
│   └── data_dictionary.csv                   # Complete 54-column schema matching Appendix Table 14
├── scripts/
│   ├── 01_reproduce_rq1.py                   # Reproduces RQ1 ground truth, Cohen's kappa, and Table 6-7
│   ├── 02_reproduce_rq2.py                   # Reproduces RQ2 26-feature ML models, Table 8-9, and Figure 5
│   └── 03_reproduce_rq3.py                   # Reproduces RQ3 LDA topics, PCA, Table 10-13, and Figure 4
├── figures/
│   ├── shap_summary.png                      # High-resolution (300 DPI) SHAP beeswarm plot (Figure 5)
│   └── pca_faceted_by_topic.png              # High-resolution (300 DPI) 5-panel faceted PCA plot (Figure 4)
└── results/
    ├── rq1_reproduction_results.json         # Automated JSON output of RQ1 statistics
    ├── rq2_reproduction_results.json         # Automated JSON output of RQ2 evaluation metrics
    └── rq3_reproduction_results.json         # Automated JSON output of RQ3 statistical tests
```

---

## 2. Requirements & Environment Setup

The replication scripts require **Python 3.9+** and standard data science libraries (`pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`).

### Installation

We recommend using a clean virtual environment:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip and install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Step-by-Step Reproduction Guide

All scripts are self-contained and execute against the offline data provided in `data/`. No external GitHub API tokens, personal access tokens (PATs), or network access are required.

### Step 3.1. Research Question 1 (RQ1: Rule-Based Attribution & Ground Truth)

To evaluate the 8 ground-truth rules, reproduce inter-annotator reliability on the representative AIDev sample, and verify positive-control agreement across 313,927 commits:

```bash
python scripts/01_reproduce_rq1.py
```

**Outputs & Validation Checklist:**
- Prints the 8 finalized attribution rules (Table 5).
- Evaluates the 384 AIDev dual-annotator validation sample:
  - **Agreement:** 98.18% (377 / 384 agreed; Paper reports 98.2%).
  - **Cohen's Kappa:** $\kappa = 0.9632$ (Paper reports $\kappa = 0.963$, Almost Perfect Agreement).
  - **Disagreements:** 7 non-English commit messages resolved by consensus.
- Evaluates the 384 David Lo positive-control sample:
  - **Match Rate:** 100.00% (384 / 384 match AI; Paper reports 100%).
- Full-scale revalidation across 313,927 commits (Table 6):
  - Confirmed AI: 313,354 (99.82%), Not_Found: 347 (0.11%), Excluded_Bot: 192 (0.06%), Human: 34 (0.01%).
  - Agreement on confident labels: 99.93% ($313,354 / 313,580$).
- Displays rule firing frequencies (Table 7) and breakdown of 226 disagreements.
- Exports results to `results/rq1_reproduction_results.json`.

---

### Step 3.2. Research Question 2 (RQ2: Behavioral Classification)

To train and evaluate the supervised machine learning models on 26 pure behavioral features (excluding all identity metadata):

```bash
python scripts/02_reproduce_rq2.py
```

**Outputs & Validation Checklist:**
- Loads the balanced master dataset of 21,810 commits across 546 repositories.
- Executes within-repository 80/20 train/test split:
  - **Train Set:** 17,450 commits (8,725 Human, 8,725 AI).
  - **Held-Out Test Set:** 4,360 commits (2,180 Human, 2,180 AI).
- Evaluates 5-fold Stratified Cross-Validation on the training set:
  - **Logistic Regression:** CV Macro F1 = $0.9241 \pm 0.0052$
  - **Random Forest:** CV Macro F1 = $0.9464 \pm 0.0038$
  - **XGBoost:** CV Macro F1 = $0.9496 \pm 0.0049$
- Evaluates held-out test set performance (Table 8):
  - **Logistic Regression:** Macro F1 = 0.9160, ROC-AUC = 0.9594, Precision (AI) = 0.8954, Recall (AI) = 0.9422, Accuracy = 91.61%
  - **Random Forest:** Macro F1 = 0.9406, ROC-AUC = 0.9792, Precision (AI) = 0.9222, Recall (AI) = 0.9624, Accuracy = 94.06%
  - **XGBoost:** Macro F1 = 0.9445, ROC-AUC = 0.9842, Precision (AI) = 0.9310, Recall (AI) = 0.9601, Accuracy = 94.45%
- Confusion Matrix for XGBoost (Table 9):
  - $TN = 2,025$ (Human), $FP = 155$, $FN = 87$, $TP = 2,093$ (AI). Total = 4,360.
- Bivariate feature correlations (Table 12: Kendall's $\tau$ and Point-Biserial $r$).
- Computes SHAP values on XGBoost and saves a 300 DPI plot to `figures/shap_summary.png` (Figure 5).
- Exports results to `results/rq2_reproduction_results.json`.

---

### Step 3.3. Research Question 3 (RQ3: Topic Modeling, Within-Topic Stats & Burstiness)

To fit the LDA topic model, perform 2D PCA decomposition, compute within-topic effect sizes, and evaluate stylometric burstiness across commit messages, PR descriptions, and inline code comments:

```bash
python scripts/03_reproduce_rq3.py
```

**Outputs & Validation Checklist:**
- Fits LDA topic modeling ($K=5$ topics) on 21,810 commit messages:
  - Topic 1: Build & Source Maintenance (`ts, src, build, cli`): 19.31% Human vs 15.19% AI (-4.12%)
  - Topic 2: Agent Docs & Workflow (`docs, agent, md, bot`): 16.17% Human vs 22.24% AI (+6.07% AI)
  - Topic 3: Initial Planning & Setup (`plan, initial, issue, main`): 21.76% Human vs 9.66% AI (-12.10% Human)
  - Topic 4: Feature Additions & UI/API (`feat, new, ui, api, model`): 18.78% Human vs 20.50% AI (+1.71% AI)
  - Topic 5: Testing & Error Handling (`tests, test, error, handling`): 23.98% Human vs 32.42% AI (+8.44% AI)
- Computes 2D PCA decomposition on behavioral features:
  - $PC_1$: 13.7% variance (Code Change Magnitude).
  - $PC_2$: 12.1% variance (Message Structure & Additions).
  - Generates 300 DPI 5-panel faceted PCA plot saved to `figures/pca_faceted_by_topic.png` (Figure 4).
- Within-Topic Statistical Testing (Table 11: Mann-Whitney $U$, Cliff's $\delta$):
  - Topic 1 (Build): `message_length` (Med 32 vs 182, $\delta = +0.8433$, Large), `has_multiple_sentences` (Med 0 vs 1, $\delta = +0.7789$, Large)
  - Topic 2 (Docs): `message_length` (Med 41 vs 145, $\delta = +0.8234$, Large), `additions` (Med 10 vs 40, $\delta = +0.2177$, Small)
  - Topic 3 (Plan): `total_changed` (Med 171 vs 0, $\delta = -0.9008$, Large), `files_changed` (Med 6 vs 0, $\delta = -0.9060$, Large)
  - Topic 4 (Features): `message_length` (Med 47 vs 500, $\delta = +0.8022$, Large), `additions` (Med 40 vs 117, $\delta = +0.2921$, Small)
  - Topic 5 (Testing): `message_length` (Med 34 vs 198, $\delta = +0.8527$, Large), `has_multiple_sentences` (Med 0 vs 1, $\delta = +0.8593$, Large)
- Comprehensive Stylometric Burstiness & Code Comments (Table 13):
  - **Commit Message Word Burstiness:** Human Mean $-0.830$ vs AI Mean $-0.207$, Cliff's $\delta = +0.397$ (Medium).
  - **Commit Message Sentence Burstiness:** Human Mean $0.509$ vs AI Mean $0.582$, Cliff's $\delta = +0.211$ (Small).
  - **Commit Body Word Burstiness:** Human Mean $0.613$ vs AI Mean $0.634$, Cliff's $\delta = +0.030$ (Negligible).
  - **PR Body Word Burstiness:** Human Mean $0.744$ vs AI Mean $0.827$, Cliff's $\delta = +0.340$ (Medium).
  - **Code Comment Word Count:** Human Mean $802.3$ (Med $0.0$) vs AI Mean $468.9$ (Med $8.0$), $p < 0.001$, Cliff's $\delta = +0.092$.
  - **Code Comment Word Burstiness ($\ge 0$):** Human Mean $0.640$ vs AI Mean $0.664$, $p < 0.001$, Cliff's $\delta = +0.049$.
  - **Code Comment Sentence Burstiness ($\ge 0$):** Human Mean $1.083$ vs AI Mean $1.005$, $p = 0.003$, Cliff's $\delta = -0.043$.
- Exports results to `results/rq3_reproduction_results.json`.

---

## 4. Paper-to-Artifact Mapping

| Paper Item | Description | Script | Primary Output / File |
| :--- | :--- | :--- | :--- |
| **Table 5** | Rule-Based Ground-Truth Policy | `01_reproduce_rq1.py` | Terminal Output & `data/data_dictionary.csv` |
| **Section 4.1** | Dual-Annotator Agreement (Cohen's $\kappa$) | `01_reproduce_rq1.py` | Terminal Output & `results/rq1_reproduction_results.json` |
| **Table 6** | Positive Control Revalidation ($N = 313,927$) | `01_reproduce_rq1.py` | `data/positive_control_revalidation_summary.csv` |
| **Table 7** | Rule Firing Frequencies ($N = 313,354$) | `01_reproduce_rq1.py` | `data/positive_control_revalidation_summary.csv` |
| **Table 4** | 26 Behavioral Features Schema | `02_reproduce_rq2.py` | `data/data_dictionary.csv` |
| **Table 8** | Model Performance Comparison | `02_reproduce_rq2.py` | Terminal Output & `results/rq2_reproduction_results.json` |
| **Table 9** | XGBoost Confusion Matrix | `02_reproduce_rq2.py` | Terminal Output & `results/rq2_reproduction_results.json` |
| **Table 12** | Feature Correlations with AI Label | `02_reproduce_rq2.py` | Terminal Output & `results/rq2_reproduction_results.json` |
| **Figure 5** | SHAP Beeswarm Summary Plot | `02_reproduce_rq2.py` | `figures/shap_summary.png` |
| **Table 10** | LDA Topic Distributions ($\lambda=0.6$) | `03_reproduce_rq3.py` | Terminal Output & `results/rq3_reproduction_results.json` |
| **Figure 4** | 2D Faceted PCA Behavioral Plot | `03_reproduce_rq3.py` | `figures/pca_faceted_by_topic.png` |
| **Table 11** | Within-Topic Statistical Comparisons | `03_reproduce_rq3.py` | Terminal Output & `results/rq3_reproduction_results.json` |
| **Table 13** | Comprehensive Burstiness & Code Comments | `03_reproduce_rq3.py` | Terminal Output & `results/rq3_reproduction_results.json` |
| **Table 14** | Master Dataset Feature Dictionary | N/A | `data/data_dictionary.csv` |

---

## 5. Master Dataset Schema Summary

The consolidated master dataset (`data/featured_dataset_with_comments.csv`) consists of **21,810 rows** (10,905 Human commits and 10,905 AI commits) across **546 popular repositories** ($\ge 500$ stars).

A complete dictionary defining all features, data types, categories, and descriptions is provided in `data/data_dictionary.csv`. The schema is categorized into 9 distinct groups:

1. **Target & Metadata (7 columns):** `validated_is_ai`, `commit_sha`, `repo_full_name`, `source`, `pr_id`, `pr_created_at`, `stars`.
2. **Developer Identity (4 columns):** `author_login`, `author_name`, `author_email`, `committer_login` *(strictly excluded during model training to avoid identity leakage)*.
3. **Commit Message Content (9 behavioral features):** `message_length`, `has_multiple_sentences`, `has_issue_reference`, `is_bugfix`, `is_feature`, `is_refactor`, `is_perf`, `is_doc`, `is_test`.
4. **Temporal Behavior (5 behavioral features):** `hour_of_day`, `day_of_week`, `is_weekend`, `is_after_hours`, `temporal_consistency`.
5. **Code Size (7 behavioral features):** `additions`, `deletions`, `total_changed`, `files_changed`, `avg_lines_per_file`, `addition_ratio`, `size_category`.
6. **File Operations (5 behavioral features):** `has_added_files`, `has_deleted_files`, `has_renamed_files`, `is_pure_deletion`, `is_bulk_change`.
7. **Temporal & PR Burstiness (3 columns):** `pr_commit_interval_seconds`, `author_commit_interval_seconds`, `pr_commits_in_1h_window`.
8. **Text Burstiness - Contributor Style (7 columns):** `word_burstiness`, `sentence_burstiness`, `is_multi_sentence`, `mean_inter_arrival`, `mean_burst_length`, `repeated_word_count`, `total_word_count`.
9. **Inline Code Comments & PR Text (4 columns):** `comment_word_count`, `comment_word_burstiness`, `comment_sentence_burstiness`, `pr_body_word_burstiness`.

---

## 6. Sanitization & Anonymity Statement

This replication package has been rigorously sanitized:
- **No Private Access Tokens (PATs) or API Keys:** All analyses run entirely offline using pre-compiled CSV records.
- **No Hardcoded Absolute Paths:** All scripts utilize relative directory traversal via `pathlib.Path(__file__).resolve().parents[1]`.
- **Double-Blind Ready:** All author identifiers, institutional affiliations, and machine-specific metadata have been removed.

---

## 7. License

This replication package is made openly available under the **MIT License**. See [LICENSE](LICENSE) for full legal text.
