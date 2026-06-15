"""
LOOCV comparing L1 logistic regression, elastic-net logistic regression,
and Random Forest — all with the same fold-wise feature selection pipeline
(prevalence >= 10 % of training samples  →  variance top-50 %  →  top-100
by univariate |point-biserial r|, computed on the 38 training patients only).

Outputs:
  results/ml/loocv_linear_results_<model>.tsv  per-patient tables
  results/ml/loocv_linear_summary.tsv          side-by-side comparison
  results/ml/loocv_linear_top_features.tsv     top-10 stable features for best model
"""
import os
import warnings
import collections
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--dataset", default="results/ml/immunotherapy_dataset.tsv")
_p.add_argument("--raw",     default="results/ml/X_genus_raw.tsv")
_p.add_argument("--suffix",  default="",
                help="Appended to output filenames, e.g. '_combat'")
_args = _p.parse_args()

DATASET_PATH = _args.dataset
RAW_PATH     = _args.raw
SUFFIX       = _args.suffix

PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100

MODELS = {
    "L1_LogReg": LogisticRegression(
        penalty="l1", solver="liblinear",
        class_weight="balanced", C=1.0, max_iter=1000,
    ),
    "ElasticNet_LogReg": LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        class_weight="balanced", C=1.0, max_iter=5000, tol=1e-3,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
}

# ── Load ─────────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATASET_PATH, sep="\t")
raw = pd.read_csv(RAW_PATH, sep="\t").set_index("run_accession")

feature_cols = [c for c in df.columns if c not in ("run_accession", "response")]
raw = raw.reindex(df["run_accession"].values)[feature_cols]
raw.index = df.index

print(f"Samples  : {len(df)}")
print(f"Features : {len(feature_cols)}  (before per-fold selection)")
print(f"Classes  : {df['response'].value_counts().to_dict()}")
print()

# Storage: per-model records and per-fold coefficient maps (for logistic models)
records    = {name: [] for name in MODELS}
# coef_store[model][genus] = list of coefficients across folds where that genus
# was selected AND the model has linear coefficients
coef_store = {name: collections.defaultdict(list) for name in MODELS
              if name != "RandomForest"}

# ── LOOCV ────────────────────────────────────────────────────────────────────
for idx in df.index:
    patient_id = df.at[idx, "run_accession"]
    actual     = df.at[idx, "response"]

    train_df  = df.drop(idx)
    train_raw = raw.drop(idx)
    test_row  = df.loc[[idx]]

    n_train  = len(train_df)
    min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n_train)))

    # ── Feature selection on training set only ────────────────────────────
    presence       = (train_raw > 0).sum(axis=0)
    prevalent_cols = presence[presence >= min_prev].index.tolist()

    variances      = train_df[prevalent_cols].var(axis=0)
    var_cutoff     = variances.quantile(1.0 - VARIANCE_KEEP_FRACTION)
    high_var_cols  = variances[variances >= var_cutoff].index.tolist()

    train_labels = (train_df["response"] == "R").astype(int).values
    pb_corrs = {}
    for col in high_var_cols:
        r, _ = stats.pointbiserialr(train_labels, train_df[col].values)
        pb_corrs[col] = abs(r)

    pb_series     = pd.Series(pb_corrs).sort_values(ascending=False)
    selected_cols = pb_series.head(min(TOP_N, len(pb_series))).index.tolist()

    X_train = train_df[selected_cols].values
    y_train = train_df["response"].values
    X_test  = test_row[selected_cols].values

    # ── Train and predict with each model ────────────────────────────────
    for model_name, model_proto in MODELS.items():
        import sklearn.base
        model = sklearn.base.clone(model_proto)
        model.fit(X_train, y_train)

        probs       = model.predict_proba(X_test)[0]
        class_order = list(model.classes_)
        r_prob      = probs[class_order.index("R")]
        predicted   = "R" if r_prob >= 0.5 else "NR"

        records[model_name].append({
            "run_accession":    patient_id,
            "actual":           actual,
            "predicted_prob_R": round(r_prob, 4),
            "predicted_class":  predicted,
            "correct":          "YES" if predicted == actual else "NO",
            "n_features":       len(selected_cols),
        })

        # Collect coefficients for linear models
        if model_name != "RandomForest":
            # coef_ is (1, n_features) for binary; positive = R-associated
            # when classes_ = ['NR', 'R']
            coefs = model.coef_[0]
            r_sign = 1 if class_order[1] == "R" else -1
            for col, c in zip(selected_cols, coefs):
                # Always store with R-positive orientation
                coef_store[model_name][col].append(r_sign * c)

    print(f"  {patient_id}  "
          + "  ".join(
              f"{n}={'R' if records[n][-1]['predicted_prob_R']>=0.5 else 'NR'}"
              f"({records[n][-1]['predicted_prob_R']:.2f})"
              for n in MODELS
          )
          + f"  actual={actual}")

# ── Aggregate metrics ────────────────────────────────────────────────────────
def compute_metrics(recs):
    df_r = pd.DataFrame(recs)
    yt   = df_r["actual"].values
    yp   = df_r["predicted_class"].values
    ypr  = df_r["predicted_prob_R"].values
    acc  = (yt == yp).mean()
    auc  = roc_auc_score((yt == "R").astype(int), ypr)
    cm   = confusion_matrix(yt, yp, labels=["NR", "R"])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return dict(accuracy=round(acc, 4), roc_auc=round(auc, 4),
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

metrics = {name: compute_metrics(records[name]) for name in MODELS}

# ── Print comparison table ───────────────────────────────────────────────────
print()
print("=" * 72)
print("LOOCV COMPARISON  (fold-wise feature selection, ~59 features per fold)")
print("=" * 72)
header = f"{'Model':<22} {'Accuracy':>9} {'AUC':>7} {'Sens':>7} {'Spec':>7}  {'CM (TN/FP/FN/TP)'}"
print(header)
print("-" * 72)

# Previous RF baseline for reference
ref = {"accuracy": 0.5128, "roc_auc": 0.3931, "sensitivity": 0.7500,
       "specificity": 0.1333, "TN": 2, "FP": 13, "FN": 6, "TP": 18}
print(f"  {'RF+FS (prev run)':<20} {ref['accuracy']:>9.4f} {ref['roc_auc']:>7.4f} "
      f"{ref['sensitivity']:>7.4f} {ref['specificity']:>7.4f}  "
      f"TN={ref['TN']} FP={ref['FP']} FN={ref['FN']} TP={ref['TP']}  [reference]")

for name in MODELS:
    m = metrics[name]
    marker = "  <- best AUC" if m["roc_auc"] == max(v["roc_auc"] for v in metrics.values()) else ""
    print(f"  {name:<20} {m['accuracy']:>9.4f} {m['roc_auc']:>7.4f} "
          f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f}  "
          f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}{marker}")

print()

# ── Top features for best linear model by AUC ───────────────────────────────
linear_aucs = {n: metrics[n]["roc_auc"] for n in coef_store}
best_linear  = max(linear_aucs, key=linear_aucs.get)
print(f"Best linear model by AUC: {best_linear}  (AUC={linear_aucs[best_linear]:.4f})")
print()

# Summarise coefficient stability across folds
feature_stats = []
for genus, coef_list in coef_store[best_linear].items():
    n_nonzero  = sum(1 for c in coef_list if c != 0.0)
    n_selected = len(coef_list)   # folds where this genus was in selected_cols
    if n_nonzero == 0:
        continue
    avg_coef   = np.mean([c for c in coef_list if c != 0.0])
    feature_stats.append({
        "genus":        genus,
        "folds_selected": n_selected,
        "folds_nonzero":  n_nonzero,
        "pct_nonzero":    round(100 * n_nonzero / n_selected, 1),
        "avg_coef_nonzero": round(avg_coef, 4),
        "direction":    "R+" if avg_coef > 0 else "NR+",
    })

feat_df = (pd.DataFrame(feature_stats)
             .sort_values(["folds_nonzero", "pct_nonzero"], ascending=False)
             .reset_index(drop=True))

print(f"Top-10 most consistently nonzero features ({best_linear}):")
print(f"  {'Rank':<5} {'Genus':<32} {'Folds nonzero':>14} {'% nonzero':>10} "
      f"{'Avg coef':>10} {'Dir':>5}")
print("  " + "-" * 75)
for i, row in feat_df.head(10).iterrows():
    print(f"  {i+1:<5} {row['genus']:<32} "
          f"{row['folds_nonzero']:>5}/{row['folds_selected']:<8} "
          f"{row['pct_nonzero']:>9.1f}% "
          f"{row['avg_coef_nonzero']:>10.4f} "
          f"{row['direction']:>5}")

# ── Save outputs ─────────────────────────────────────────────────────────────
os.makedirs("results/ml", exist_ok=True)

for name in MODELS:
    path = f"results/ml/loocv_linear_results_{name}{SUFFIX}.tsv"
    pd.DataFrame(records[name]).to_csv(path, sep="\t", index=False)

summary_rows = []
for name in MODELS:
    m = metrics[name]
    summary_rows.append({"model": name, **m})
pd.DataFrame(summary_rows).to_csv(
    f"results/ml/loocv_linear_summary{SUFFIX}.tsv", sep="\t", index=False)

feat_df.to_csv(
    f"results/ml/loocv_linear_top_features{SUFFIX}.tsv", sep="\t", index=False)

print()
print(f"Saved: results/ml/loocv_linear_results_<model>{SUFFIX}.tsv")
print(f"Saved: results/ml/loocv_linear_summary{SUFFIX}.tsv")
print(f"Saved: results/ml/loocv_linear_top_features{SUFFIX}.tsv")
