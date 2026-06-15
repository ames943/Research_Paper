"""
LOOCV with fold-wise feature selection (leak-free).

For each of the 39 LOOCV folds:
  1. Prevalence filter  on the 38 TRAINING samples (raw > 0 in >= 10 % of 38)
  2. Variance filter    on the 38 TRAINING samples (keep top-50 % CLR variance)
  3. Univariate filter  on the 38 TRAINING samples (top-100 by |point-biserial r|)
  4. Train RandomForest on those features / training patients
  5. Predict held-out patient using the same features

No label information from the held-out patient is used at any stage of
feature selection, so the reported metrics are honest LOOCV estimates.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats

DATASET_PATH = "results/ml/immunotherapy_dataset.tsv"
RAW_PATH     = "results/ml/X_genus_raw.tsv"
RESULTS_PATH = "results/ml/loocv_fs_results.tsv"
SUMMARY_PATH = "results/ml/loocv_fs_summary.tsv"

PREVALENCE_FRAC        = 0.10   # genus must be present in >=10 % of training samples
VARIANCE_KEEP_FRACTION = 0.50   # keep top-50 % by CLR variance
TOP_N                  = 100    # final features per fold

# ── Load ────────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATASET_PATH, sep="\t")
raw = pd.read_csv(RAW_PATH,     sep="\t").set_index("run_accession")

feature_cols = [c for c in df.columns if c not in ("run_accession", "response")]

# Align raw to the same sample order as df (makes boolean indexing safe)
raw = raw.reindex(df["run_accession"].values)[feature_cols]
raw.index = df.index   # integer index to match df

print(f"Samples  : {len(df)}")
print(f"Features : {len(feature_cols)}  (before per-fold selection)")
print(f"Classes  : {df['response'].value_counts().to_dict()}")
print()

records = []

for idx in df.index:
    patient_id = df.at[idx, "run_accession"]
    actual     = df.at[idx, "response"]

    train_df  = df.drop(idx)
    train_raw = raw.drop(idx)
    test_row  = df.loc[[idx]]

    n_train   = len(train_df)
    min_prev  = max(2, int(np.ceil(PREVALENCE_FRAC * n_train)))

    # ── Step 1: Prevalence filter (on raw training counts) ────────────────
    presence      = (train_raw > 0).sum(axis=0)
    prevalent_cols = presence[presence >= min_prev].index.tolist()

    # ── Step 2: Variance filter (on training CLR values) ─────────────────
    train_X_prev = train_df[prevalent_cols]
    variances    = train_X_prev.var(axis=0)
    var_cutoff   = variances.quantile(1.0 - VARIANCE_KEEP_FRACTION)
    high_var_cols = variances[variances >= var_cutoff].index.tolist()

    # ── Step 3: Univariate filter (on training labels only) ───────────────
    train_labels = (train_df["response"] == "R").astype(int).values
    pb_corrs = {}
    for col in high_var_cols:
        r, _ = stats.pointbiserialr(train_labels, train_df[col].values)
        pb_corrs[col] = abs(r)

    pb_series     = pd.Series(pb_corrs).sort_values(ascending=False)
    n_select      = min(TOP_N, len(pb_series))
    selected_cols = pb_series.head(n_select).index.tolist()

    # ── Train and predict ─────────────────────────────────────────────────
    X_train = train_df[selected_cols].values
    y_train = train_df["response"].values
    X_test  = test_row[selected_cols].values

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs         = model.predict_proba(X_test)[0]
    class_order   = list(model.classes_)
    response_prob = probs[class_order.index("R")]
    predicted     = "R" if response_prob >= 0.5 else "NR"

    records.append({
        "run_accession":   patient_id,
        "actual":          actual,
        "predicted_prob_R": round(response_prob, 4),
        "predicted_class": predicted,
        "correct":         "YES" if predicted == actual else "NO",
        "n_features_used": n_select,
    })

    status = "CORRECT" if predicted == actual else "WRONG "
    print(f"  {patient_id}  actual={actual}  pred={predicted}  "
          f"prob_R={response_prob:.3f}  nfeat={n_select}  [{status}]")

# ── Aggregate metrics ────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)

y_true = results_df["actual"].values
y_pred = results_df["predicted_class"].values
y_prob = results_df["predicted_prob_R"].values

accuracy    = (y_true == y_pred).mean()
auc         = roc_auc_score((y_true == "R").astype(int), y_prob)
cm          = confusion_matrix(y_true, y_pred, labels=["NR", "R"])
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

print()
print("=" * 60)
print("LOOCV + FOLD-WISE FEATURE SELECTION  –  SUMMARY")
print("=" * 60)
print(f"Accuracy     : {accuracy:.4f}  ({int(accuracy * len(results_df))}/{len(results_df)})")
print(f"ROC-AUC      : {auc:.4f}")
print(f"Sensitivity  : {sensitivity:.4f}  (TP={tp}, FN={fn})")
print(f"Specificity  : {specificity:.4f}  (TN={tn}, FP={fp})")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"             pred_NR  pred_R")
print(f"  actual_NR    {tn:3d}      {fp:3d}")
print(f"  actual_R     {fn:3d}      {tp:3d}")
print()
print(f"Avg features per fold : {results_df['n_features_used'].mean():.1f}")
print(f"Min / max             : {results_df['n_features_used'].min()} / {results_df['n_features_used'].max()}")
print()
print("Comparison:")
print(f"  CLR, no feat. sel.  :  Acc=0.5128  AUC=0.4167  Sens=0.6250  Spec=0.3333")
print(f"  CLR + fold feat. sel.:  Acc={accuracy:.4f}  AUC={auc:.4f}  "
      f"Sens={sensitivity:.4f}  Spec={specificity:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("results/ml", exist_ok=True)
results_df.to_csv(RESULTS_PATH, sep="\t", index=False)
print(f"\nPer-patient results -> {RESULTS_PATH}")

summary = pd.DataFrame([{
    "n_samples":        len(results_df),
    "n_R":              int((y_true == "R").sum()),
    "n_NR":             int((y_true == "NR").sum()),
    "accuracy":         round(accuracy, 4),
    "roc_auc":          round(auc, 4),
    "sensitivity":      round(sensitivity, 4),
    "specificity":      round(specificity, 4),
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "feature_selection": f"prevalence({PREVALENCE_FRAC*100:.0f}%)"
                         f"+variance(top{VARIANCE_KEEP_FRACTION*100:.0f}%)"
                         f"+univariate_top{TOP_N}_per_fold",
}])
summary.to_csv(SUMMARY_PATH, sep="\t", index=False)
print(f"Summary metrics      -> {SUMMARY_PATH}")
