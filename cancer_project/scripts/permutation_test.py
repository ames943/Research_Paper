"""
Permutation test for the Elastic Net + fold-wise feature selection pipeline.

For each permutation: shuffle the 39 response labels, run the full
LOOCV pipeline (prevalence -> variance -> top-100 -> ElasticNet), record AUC.
Empirical p-value = fraction of permuted AUCs >= observed AUC (0.4486).
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import sklearn.base

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Config ────────────────────────────────────────────────────────────────────
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--dataset",      default="results/ml/immunotherapy_dataset.tsv")
_p.add_argument("--raw",          default="results/ml/X_genus_raw.tsv")
_p.add_argument("--loocv-summary",default=None,
                help="Path to loocv_linear_summary*.tsv; reads ElasticNet AUC from it")
_p.add_argument("--observed-auc", type=float, default=None,
                help="Observed AUC override (skips --loocv-summary)")
_p.add_argument("--suffix",       default="",
                help="Appended to output filenames, e.g. '_combat'")
_args = _p.parse_args()

N_PERMS       = 100
SEED          = 42

PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100

DATASET_PATH  = _args.dataset
RAW_PATH      = _args.raw
SUFFIX        = _args.suffix
AUCS_PATH     = f"results/ml/permutation_aucs{SUFFIX}.tsv"
SUMMARY_PATH  = f"results/ml/permutation_summary{SUFFIX}.tsv"
PLOT_PATH     = f"results/ml/permutation_test_auc{SUFFIX}.png"

# Resolve observed AUC
if _args.observed_auc is not None:
    OBSERVED_AUC = _args.observed_auc
elif _args.loocv_summary is not None:
    _s = pd.read_csv(_args.loocv_summary, sep="\t")
    OBSERVED_AUC = float(_s.loc[_s["model"] == "ElasticNet_LogReg", "roc_auc"].iloc[0])
else:
    OBSERVED_AUC = 0.4003  # fallback: n=79 uncorrected
LOG_INTERVAL  = 10

MODEL_PROTO = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5,
    class_weight="balanced", C=1.0, max_iter=5000, tol=1e-3,
)

# ── Load ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATASET_PATH, sep="\t")
raw = pd.read_csv(RAW_PATH, sep="\t").set_index("run_accession")

feature_cols = [c for c in df.columns if c not in ("run_accession", "response")]
raw = raw.reindex(df["run_accession"].values)[feature_cols]
raw.index = df.index

true_labels = df["response"].values.copy()
rng = np.random.default_rng(SEED)

print(f"Permutation test: N={N_PERMS}, observed AUC={OBSERVED_AUC}")
print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
print(f"Pipeline: ElasticNet (l1_ratio=0.5) + fold-wise feature selection")
print()

os.makedirs("results/ml", exist_ok=True)

def run_loocv(response_labels):
    """Run one full LOOCV pass with the given response label array."""
    probs_list = []
    true_list  = []
    for idx in df.index:
        loc   = df.index.get_loc(idx)
        train_idx = df.index.difference([idx])
        train_df  = df.loc[train_idx].copy()
        train_df["response"] = np.delete(response_labels, loc)
        train_raw = raw.loc[train_idx]
        test_row  = df.loc[[idx]]

        n_train  = len(train_df)
        min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n_train)))

        presence      = (train_raw > 0).sum(axis=0)
        prevalent_cols = presence[presence >= min_prev].index.tolist()

        variances     = train_df[prevalent_cols].var(axis=0)
        var_cutoff    = variances.quantile(1.0 - VARIANCE_KEEP_FRACTION)
        high_var_cols = variances[variances >= var_cutoff].index.tolist()

        train_labels_binary = (train_df["response"] == "R").astype(int).values
        pb_corrs = {
            col: abs(stats.pointbiserialr(train_labels_binary,
                                          train_df[col].values)[0])
            for col in high_var_cols
        }
        pb_series     = pd.Series(pb_corrs).sort_values(ascending=False)
        selected_cols = pb_series.head(min(TOP_N, len(pb_series))).index.tolist()

        model = sklearn.base.clone(MODEL_PROTO)
        model.fit(train_df[selected_cols].values, train_df["response"].values)
        class_order   = list(model.classes_)
        r_prob        = model.predict_proba(test_row[selected_cols].values)[0][
                            class_order.index("R")]
        probs_list.append(r_prob)
        true_list.append(response_labels[loc])

    auc = roc_auc_score(
        (np.array(true_list) == "R").astype(int), probs_list
    )
    return auc

# ── Run permutations ─────────────────────────────────────────────────────────
perm_aucs  = []
wall_start = time.time()

for perm_i in range(1, N_PERMS + 1):
    shuffled = rng.permutation(true_labels)
    auc_i    = run_loocv(shuffled)
    perm_aucs.append(auc_i)

    if perm_i % LOG_INTERVAL == 0 or perm_i == 1:
        elapsed  = time.time() - wall_start
        rate     = perm_i / elapsed
        eta_sec  = (N_PERMS - perm_i) / rate
        print(f"  Perm {perm_i:3d}/{N_PERMS}  AUC={auc_i:.4f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta_sec:.0f}s")

total_time = time.time() - wall_start

# ── Compute empirical p-value ─────────────────────────────────────────────────
perm_aucs  = np.array(perm_aucs)
p_value    = (perm_aucs >= OBSERVED_AUC).mean()
perm_mean  = perm_aucs.mean()
perm_std   = perm_aucs.std()

print()
print("=" * 55)
print("PERMUTATION TEST RESULTS")
print("=" * 55)
print(f"Observed AUC    : {OBSERVED_AUC:.4f}")
print(f"Permuted AUC    : mean={perm_mean:.4f}  std={perm_std:.4f}")
print(f"  min={perm_aucs.min():.4f}  max={perm_aucs.max():.4f}")
print(f"Empirical p-val : {p_value:.4f}  ({int((perm_aucs >= OBSERVED_AUC).sum())} / {N_PERMS} >= {OBSERVED_AUC})")
print(f"N permutations  : {N_PERMS}")
print(f"Total runtime   : {total_time:.1f}s  ({total_time/60:.2f} min)")
print()
if p_value < 0.05:
    print("Result: SIGNIFICANT at alpha=0.05")
elif p_value < 0.10:
    print("Result: MARGINAL (0.05 < p < 0.10)")
else:
    print("Result: NOT SIGNIFICANT at alpha=0.05")

# ── Save AUC list ─────────────────────────────────────────────────────────────
pd.DataFrame({"perm_index": range(1, N_PERMS + 1), "auc": perm_aucs}).to_csv(
    AUCS_PATH, sep="\t", index=False)
print(f"Permuted AUCs saved to {AUCS_PATH}")

# ── Save summary ──────────────────────────────────────────────────────────────
pd.DataFrame([{
    "observed_auc":   OBSERVED_AUC,
    "n_perms":        N_PERMS,
    "perm_auc_mean":  round(float(perm_mean), 4),
    "perm_auc_std":   round(float(perm_std),  4),
    "perm_auc_min":   round(float(perm_aucs.min()), 4),
    "perm_auc_max":   round(float(perm_aucs.max()), 4),
    "n_perm_gte_obs": int((perm_aucs >= OBSERVED_AUC).sum()),
    "empirical_pval": round(float(p_value), 4),
    "runtime_sec":    round(total_time, 1),
}]).to_csv(SUMMARY_PATH, sep="\t", index=False)
print(f"Summary saved to {SUMMARY_PATH}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(perm_aucs, bins=20, color="steelblue", edgecolor="white",
        alpha=0.85, label=f"Null distribution\n(N={N_PERMS} permutations)")
ax.axvline(OBSERVED_AUC, color="crimson", linewidth=2,
           label=f"Observed AUC = {OBSERVED_AUC}\np = {p_value:.3f}")
ax.axvline(0.5, color="gray", linewidth=1, linestyle="--",
           label="Random chance (AUC=0.5)")
ax.set_xlabel("LOOCV AUC (ElasticNet + fold-wise FS)")
ax.set_ylabel("Count")
ax.set_title("Permutation test: immunotherapy response prediction\n"
             f"n=79 patients, fold-wise feature selection{' [ComBat]' if SUFFIX else ''}")
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150)
print(f"Histogram saved to {PLOT_PATH}")
