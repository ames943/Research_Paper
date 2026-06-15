#!/usr/bin/env python3.13
"""
Leak-free LOOCV with per-fold batch correction.

Batch correction method (per fold):
  Additive batch mean centering in CLR space, fit on 78 training samples only.
  For each feature j and batch b:
    batch_effect[b][j] = mean(train_clr[batch==b, j]) - mean(train_clr[:, j])
  Applied to training and test samples:
    x_corrected[j] = x[j] - batch_effect[batch_of_x][j]

  This is the mean-only variant of ComBat (no EB shrinkage of variance).
  It is appropriate here because: (a) we have only 2 batches, (b) n per batch
  is small (39/40), making variance EB estimation noisy, and (c) it is fully
  leak-free — all correction parameters are estimated from training data only.

CLR is a per-sample transformation (no cross-sample information) so it is
computed globally before the LOOCV loop without introducing leakage.

Prevalence filter uses original raw proportions (presence/absence only),
which is also a per-sample property and introduces no leakage.

Usage:
    python3.13 scripts/run_loocv_combat.py
    python3.13 scripts/run_loocv_combat.py --n-perms 100
"""

import os
import sys
import time
import warnings
import collections
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import sklearn.base

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--n-perms", type=int, default=0,
                help="Permutations for permutation test (0 = skip)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
CLR_PATH    = "results/ml/X_genus_clr.tsv"
RAW_PATH    = "results/ml/X_genus_raw.tsv"
LABELS_PATH = "metadata/response_labels.tsv"
OUT_DIR     = "results/ml"
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Load ───────────────────────────────────────────────────────────────────────
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
batch    = pd.Series(
    ["SRR5930" if a.startswith("SRR5930") else "SRR11413" for a in clr.index],
    index=clr.index,
)

print(f"Samples  : {len(clr)}")
print(f"Features : {clr.shape[1]}  (CLR, before per-fold selection)")
print(f"Classes  : {response.value_counts().to_dict()}")
print(f"Batches  : {batch.value_counts().to_dict()}")
print()


# ── Per-fold batch correction ──────────────────────────────────────────────────
def batch_correct_fold(train_clr: pd.DataFrame,
                       test_clr:  pd.DataFrame,
                       train_batch: pd.Series,
                       test_batch:  pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit additive batch mean correction on training data, apply to both sets.

    Estimates per-batch offset = batch_mean - global_mean from training only.
    Corrected value = original - offset  (shifts each batch toward global mean).
    """
    global_mean = train_clr.mean(axis=0)
    offsets = {}
    for b in train_batch.unique():
        mask = train_batch == b
        offsets[b] = train_clr.loc[mask].mean(axis=0) - global_mean

    corrected_train = train_clr.copy()
    for b, off in offsets.items():
        mask = train_batch == b
        corrected_train.loc[mask] = train_clr.loc[mask] - off

    # Apply training-estimated offset to test sample
    tb = test_batch.iloc[0]
    if tb in offsets:
        corrected_test = test_clr - offsets[tb]
    else:
        corrected_test = test_clr.copy()

    return corrected_train, corrected_test


# ── LOOCV core ─────────────────────────────────────────────────────────────────
def run_loocv(response_vec, verbose: bool = False, model_subset: list | None = None):
    """
    Full 79-fold LOOCV with per-fold batch correction and feature selection.
    Returns (records_dict, coef_store_dict).

    model_subset: if given, only run these model names (for fast permutation testing).
    """
    active_models = model_subset if model_subset is not None else list(MODELS.keys())
    records    = {n: [] for n in active_models}
    coef_store = {n: collections.defaultdict(list)
                  for n in active_models if n != "RandomForest"}

    for idx in clr.index:
        actual    = response_vec[idx]
        train_idx = clr.index.difference([idx])

        train_clr   = clr.loc[train_idx]
        test_clr    = clr.loc[[idx]]
        train_batch = batch.loc[train_idx]
        test_batch  = batch.loc[[idx]]
        train_raw   = raw.loc[train_idx]

        # ── 1. Per-fold batch correction ──────────────────────────────────
        corr_train, corr_test = batch_correct_fold(
            train_clr, test_clr, train_batch, test_batch
        )

        # ── 2. Feature selection on corrected training data ───────────────
        n_train  = len(train_idx)
        min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n_train)))

        presence       = (train_raw > 0).sum(axis=0)
        prevalent_cols = presence[presence >= min_prev].index.tolist()

        corr_train_feat = corr_train[prevalent_cols]
        variances       = corr_train_feat.var(axis=0)
        var_cutoff      = variances.quantile(1.0 - VARIANCE_KEEP_FRACTION)
        high_var_cols   = variances[variances >= var_cutoff].index.tolist()

        train_labels_bin = (response_vec.loc[train_idx] == "R").astype(int).values
        pb_corrs = {
            col: abs(stats.pointbiserialr(train_labels_bin,
                                          corr_train[col].values)[0])
            for col in high_var_cols
        }
        pb_series     = pd.Series(pb_corrs).sort_values(ascending=False)
        selected_cols = pb_series.head(min(TOP_N, len(pb_series))).index.tolist()

        X_train = corr_train[selected_cols].values
        y_train = response_vec.loc[train_idx].values
        X_test  = corr_test[selected_cols].values

        # ── 3. Train / predict ────────────────────────────────────────────
        for mname in active_models:
            mproto      = MODELS[mname]
            model       = sklearn.base.clone(mproto)
            model.fit(X_train, y_train)
            probs       = model.predict_proba(X_test)[0]
            class_order = list(model.classes_)
            r_prob      = probs[class_order.index("R")]
            predicted   = "R" if r_prob >= 0.5 else "NR"

            records[mname].append({
                "run_accession":    idx,
                "actual":           actual,
                "predicted_prob_R": round(r_prob, 4),
                "predicted_class":  predicted,
                "correct":          "YES" if predicted == actual else "NO",
                "n_features":       len(selected_cols),
            })

            if mname != "RandomForest":
                r_sign = 1 if class_order[1] == "R" else -1
                for col, c in zip(selected_cols, model.coef_[0]):
                    coef_store[mname][col].append(r_sign * c)

        if verbose:
            print(f"  {idx}  "
                  + "  ".join(
                      f"{n}={'R' if records[n][-1]['predicted_prob_R']>=0.5 else 'NR'}"
                      f"({records[n][-1]['predicted_prob_R']:.2f})"
                      for n in MODELS
                  )
                  + f"  actual={actual}")

    return records, coef_store


def compute_metrics(recs: list) -> dict:
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


# ── Run LOOCV ─────────────────────────────────────────────────────────────────
records, coef_store = run_loocv(response, verbose=True)
metrics = {n: compute_metrics(records[n]) for n in MODELS}

# ── Print comparison table ────────────────────────────────────────────────────
print()
print("=" * 72)
print("LOOCV — per-fold ComBat (batch mean centering) + feature selection")
print("=" * 72)
print(f"{'Model':<22} {'Accuracy':>9} {'AUC':>7} {'Sens':>7} {'Spec':>7}  CM (TN/FP/FN/TP)")
print("-" * 72)
for name in MODELS:
    m      = metrics[name]
    marker = "  <- best AUC" if m["roc_auc"] == max(v["roc_auc"] for v in metrics.values()) else ""
    print(f"  {name:<20} {m['accuracy']:>9.4f} {m['roc_auc']:>7.4f} "
          f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f}  "
          f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}{marker}")
print()

# ── Top features for best linear model ───────────────────────────────────────
linear_aucs = {n: metrics[n]["roc_auc"] for n in coef_store}
best_linear  = max(linear_aucs, key=linear_aucs.get)
print(f"Best linear model by AUC: {best_linear}  (AUC={linear_aucs[best_linear]:.4f})")
print()

feat_stats = []
for genus, coef_list in coef_store[best_linear].items():
    n_nonzero = sum(1 for c in coef_list if c != 0.0)
    if n_nonzero == 0:
        continue
    avg_coef = np.mean([c for c in coef_list if c != 0.0])
    feat_stats.append({
        "genus":            genus,
        "folds_selected":   len(coef_list),
        "folds_nonzero":    n_nonzero,
        "pct_nonzero":      round(100 * n_nonzero / len(coef_list), 1),
        "avg_coef_nonzero": round(avg_coef, 4),
        "direction":        "R+" if avg_coef > 0 else "NR+",
    })

feat_df = (pd.DataFrame(feat_stats)
             .sort_values(["folds_nonzero", "pct_nonzero"], ascending=False)
             .reset_index(drop=True))

print(f"Top-10 stable features ({best_linear}):")
print(f"  {'Rank':<5} {'Genus':<32} {'Folds nonzero':>14} {'% nonzero':>10} "
      f"{'Avg coef':>10} {'Dir':>5}")
print("  " + "-" * 75)
for i, row in feat_df.head(10).iterrows():
    print(f"  {i+1:<5} {row['genus']:<32} "
          f"{row['folds_nonzero']:>5}/{row['folds_selected']:<8} "
          f"{row['pct_nonzero']:>9.1f}% "
          f"{row['avg_coef_nonzero']:>10.4f} "
          f"{row['direction']:>5}")

# ── Save LOOCV outputs ────────────────────────────────────────────────────────
for name in MODELS:
    pd.DataFrame(records[name]).to_csv(
        f"{OUT_DIR}/loocv_combat_results_{name}.tsv", sep="\t", index=False)

summary_rows = [{"model": n, **metrics[n]} for n in MODELS]
pd.DataFrame(summary_rows).to_csv(
    f"{OUT_DIR}/loocv_combat_summary.tsv", sep="\t", index=False)

feat_df.to_csv(f"{OUT_DIR}/loocv_combat_top_features.tsv", sep="\t", index=False)

print()
print(f"Saved: {OUT_DIR}/loocv_combat_results_<model>.tsv")
print(f"Saved: {OUT_DIR}/loocv_combat_summary.tsv")
print(f"Saved: {OUT_DIR}/loocv_combat_top_features.tsv")

# ── Permutation test ──────────────────────────────────────────────────────────
if args.n_perms > 0:
    best_model  = max(metrics, key=lambda n: metrics[n]["roc_auc"])
    observed_auc = metrics[best_model]["roc_auc"]
    print()
    print(f"Permutation test: N={args.n_perms}, best model={best_model}, "
          f"observed AUC={observed_auc:.4f}")
    print("(Each permutation reruns per-fold batch correction — fully leak-free)")
    print()

    rng       = np.random.default_rng(args.seed)
    true_arr  = response.values.copy()
    perm_aucs = []
    t0        = time.time()

    for perm_i in range(1, args.n_perms + 1):
        shuffled   = pd.Series(rng.permutation(true_arr), index=response.index)
        perm_recs, _ = run_loocv(shuffled, verbose=False, model_subset=[best_model])
        auc_i = compute_metrics(perm_recs[best_model])["roc_auc"]
        perm_aucs.append(auc_i)

        if perm_i % 10 == 0 or perm_i == 1:
            elapsed = time.time() - t0
            eta     = (args.n_perms - perm_i) / (perm_i / elapsed)
            print(f"  Perm {perm_i:3d}/{args.n_perms}  AUC={auc_i:.4f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    perm_aucs = np.array(perm_aucs)
    p_value   = (perm_aucs >= observed_auc).mean()
    print()
    print("=" * 55)
    print("PERMUTATION TEST RESULTS (per-fold ComBat, leak-free)")
    print("=" * 55)
    print(f"Best model      : {best_model}")
    print(f"Observed AUC    : {observed_auc:.4f}")
    print(f"Permuted AUC    : mean={perm_aucs.mean():.4f}  std={perm_aucs.std():.4f}")
    print(f"  min={perm_aucs.min():.4f}  max={perm_aucs.max():.4f}")
    print(f"Empirical p-val : {p_value:.4f}  "
          f"({int((perm_aucs >= observed_auc).sum())} / {args.n_perms} >= {observed_auc:.4f})")
    if p_value < 0.05:
        print("Result: SIGNIFICANT at alpha=0.05")
    elif p_value < 0.10:
        print("Result: MARGINAL (0.05 < p < 0.10)")
    else:
        print("Result: NOT SIGNIFICANT at alpha=0.05")

    pd.DataFrame({"perm_index": range(1, args.n_perms + 1),
                  "auc": perm_aucs}).to_csv(
        f"{OUT_DIR}/perm_combat_aucs.tsv", sep="\t", index=False)
    pd.DataFrame([{
        "best_model":     best_model,
        "observed_auc":   observed_auc,
        "n_perms":        args.n_perms,
        "perm_auc_mean":  round(float(perm_aucs.mean()), 4),
        "perm_auc_std":   round(float(perm_aucs.std()),  4),
        "perm_auc_min":   round(float(perm_aucs.min()),  4),
        "perm_auc_max":   round(float(perm_aucs.max()),  4),
        "n_perm_gte_obs": int((perm_aucs >= observed_auc).sum()),
        "empirical_pval": round(float(p_value), 4),
    }]).to_csv(f"{OUT_DIR}/perm_combat_summary.tsv", sep="\t", index=False)
    print(f"Saved: {OUT_DIR}/perm_combat_aucs.tsv")
    print(f"Saved: {OUT_DIR}/perm_combat_summary.tsv")
