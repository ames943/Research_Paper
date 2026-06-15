#!/usr/bin/env python3
"""
Statistical foundations for batch-effect analysis.

1. PERMDISP (betadisper analog, Aitchison distance)
2. Bootstrap 95% CIs for PERMANOVA R² and LOOCV AUC
3. Effect sizes (Cohen's f² = R²/(1-R²)) appended to permanova_comparison.tsv
4. Simulation-based power analysis (n=39 empirical basis → n=39..300)

Usage:
    cd cancer_project/
    python3 scripts/statistical_foundations.py
"""

import collections
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as sp_norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
N118_CLR    = "results/ml/n118_3cohort/X_genus_clr.tsv"
N118_RAW    = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_3C   = "metadata/response_labels_3cohort.tsv"
PERM_CMP    = "results/ml/batch_diagnostics/permanova_comparison.tsv"

LOOCV_N79U  = "results/ml/loocv_linear_results_ElasticNet_LogReg.tsv"   # n=79 uncorrected ENet
LOOCV_N79C  = "results/ml/loocv_combat_results_ElasticNet_LogReg.tsv"   # n=79 per-fold ComBat ENet
LOOCV_N118  = "results/ml/n118_3cohort/loocv_3cohort_results_RandomForest.tsv"  # n=118 RF

OUT_DIAG    = Path("results/ml/batch_diagnostics")
OUT_POWER   = Path("results/ml/power_analysis")
OUT_DIAG.mkdir(parents=True, exist_ok=True)
OUT_POWER.mkdir(parents=True, exist_ok=True)

SEED               = 42
N_BOOT             = 1000
N_PERMS_PERMDISP   = 999
N_PERMS_POWER      = 199   # internal PERMANOVA perms per simulation replicate
N_REPS_POWER       = 200   # simulation replicates per target n
TARGET_NS          = [39, 60, 80, 100, 150, 200, 300]

# Feature-selection hyperparameters (must match run_loocv_linear.py)
PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100

ENET = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5,
    class_weight="balanced", C=1.0, max_iter=5000, tol=1e-3,
)

rng       = np.random.default_rng(SEED)
rng_power = np.random.default_rng(SEED + 999)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
clr_n118 = pd.read_csv(N118_CLR, sep="\t", index_col="run_accession")
raw_n118 = pd.read_csv(N118_RAW, sep="\t", index_col="run_accession")
labels   = pd.read_csv(LABELS_3C, sep="\t").set_index("run_accession")

response = labels["response"].reindex(clr_n118.index)
cohort_id = pd.Series(
    [1 if a.startswith("SRR5930") else 2 if a.startswith("SRR11413") else 3
     for a in clr_n118.index],
    index=clr_n118.index,
)

c1_mask  = cohort_id == 1
c12_mask = cohort_id.isin([1, 2])

clr_39 = clr_n118.loc[c1_mask];  raw_39 = raw_n118.loc[c1_mask];  resp_39 = response.loc[c1_mask]
clr_79 = clr_n118.loc[c12_mask]; raw_79 = raw_n118.loc[c12_mask]; resp_79 = response.loc[c12_mask]
coh_79 = cohort_id.loc[c12_mask]

clr_118 = clr_n118
resp_118 = response.loc[clr_n118.index]
coh_118  = cohort_id.loc[clr_n118.index]

print(f"  n=39  : {len(clr_39)} samples, {clr_39.shape[1]} genera")
print(f"  n=79  : {len(clr_79)} samples")
print(f"  n=118 : {len(clr_118)} samples")
print()


# ── Shared distance / PERMANOVA helpers ───────────────────────────────────────
def aitchison_dist(clr_df: pd.DataFrame) -> np.ndarray:
    return squareform(pdist(clr_df.values.astype(float), metric="euclidean"))


def _ss_total(d2: np.ndarray) -> float:
    return float(np.sum(np.triu(d2, k=1))) / d2.shape[0]


def _ss_within(d2: np.ndarray, grp: np.ndarray) -> float:
    sw = 0.0
    for g in np.unique(grp):
        m = grp == g; n_g = m.sum()
        if n_g < 2: continue
        sub = d2[np.ix_(m, m)]
        sw += float(np.sum(np.triu(sub, k=1))) / n_g
    return sw


def permanova_r2_only(D: np.ndarray, grp: np.ndarray) -> float:
    d2 = D ** 2
    ST = _ss_total(d2)
    return (ST - _ss_within(d2, grp)) / ST if ST > 0 else 0.0


# ── Wilson binomial CI ─────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, alpha: float = 0.05):
    z = sp_norm.ppf(1 - alpha / 2)
    p = k / n
    c = (p + z**2 / (2*n)) / (1 + z**2 / n)
    m = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2 / n)
    return max(0.0, c - m), min(1.0, c + m)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: PERMDISP
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 1: PERMDISP (betadisper — dispersion homogeneity)")
print("=" * 65)
print("For Euclidean/Aitchison distance, the group centroid in PCoA")
print("space equals the arithmetic mean CLR vector.  d_i = Euclidean")
print("distance from each sample to its group centroid.")
print()


def permdisp(X: np.ndarray, grp: np.ndarray,
             n_perms: int = 999, seed: int = 42) -> dict:
    """
    Anderson (2006) PERMDISP for Euclidean/Aitchison distance.
    Dispersion d_i = ||x_i - centroid_{group(i)}||₂.
    Tests H0: within-group dispersions are homogeneous across groups.
    """
    grp = np.asarray(grp)
    groups = np.unique(grp)
    q = len(groups)
    if q < 2:
        return {"F_stat": np.nan, "p_value": np.nan, "n_groups": q,
                "note": "< 2 groups — not applicable"}

    def _d2centroid(X, grp):
        d = np.zeros(len(grp))
        for g in np.unique(grp):
            m = grp == g
            centroid = X[m].mean(axis=0)
            d[m] = np.linalg.norm(X[m] - centroid, axis=1)
        return d

    def _anova_f(d, grp):
        groups = np.unique(grp); N = len(d); k = len(groups)
        gm = d.mean()
        ssb = sum(((d[grp == g]).mean() - gm)**2 * (grp == g).sum() for g in groups)
        ssw = sum(((d[grp == g]) - (d[grp == g]).mean() ** 0).var() * (grp == g).sum()
                  for g in groups)
        if ssw == 0:
            return np.nan
        return (ssb / (k - 1)) / (ssw / (N - k))

    d_obs = _d2centroid(X, grp)
    F_obs = _anova_f(d_obs, grp)

    rng2 = np.random.default_rng(seed)
    perm_Fs = []
    for _ in range(n_perms):
        grp_p = rng2.permutation(grp)
        perm_Fs.append(_anova_f(_d2centroid(X, grp_p), grp_p))
    perm_F = np.array(perm_Fs)
    valid = perm_F[~np.isnan(perm_F)]
    p_val = float((valid >= F_obs).sum()) / len(valid) if len(valid) > 0 else np.nan

    group_disp = {str(g): round(float(d_obs[grp == g].mean()), 4) for g in groups}
    return {
        "F_stat": round(float(F_obs), 4) if not np.isnan(F_obs) else np.nan,
        "p_value": round(p_val, 4),
        "n_groups": q,
        "n_perms": n_perms,
        "group_dispersions": str(group_disp),
        "perm_F_mean": round(float(valid.mean()), 4) if len(valid) > 0 else np.nan,
    }


datasets = [
    ("n39",  clr_39.values,  resp_39.values,  None,          "cohort1 only (n=39)"),
    ("n79",  clr_79.values,  resp_79.values,  coh_79.values, "cohort1+2 pooled (n=79)"),
    ("n118", clr_118.values, resp_118.values, coh_118.values,"all 3 cohorts (n=118)"),
]

permdisp_rows = []
for tag, X, resp, coh, desc in datasets:
    n = len(X)
    print(f"\n── {desc} ──")

    # Response grouping
    print(f"  PERMDISP(response, {N_PERMS_PERMDISP} perms) …", flush=True)
    r = permdisp(X, resp, n_perms=N_PERMS_PERMDISP, seed=SEED)
    sig = " ***" if r["p_value"] < 0.001 else " **" if r["p_value"] < 0.01 else \
          " *" if r["p_value"] < 0.05 else " ns"
    print(f"    F={r['F_stat']}  p={r['p_value']}{sig}")
    print(f"    group dispersions: {r['group_dispersions']}")
    permdisp_rows.append({
        "dataset": tag, "n": n, "factor": "response",
        "n_groups": r["n_groups"], "F_stat": r["F_stat"],
        "p_value": r["p_value"], "n_perms": N_PERMS_PERMDISP,
        "group_dispersions": r["group_dispersions"],
        "dispersion_homogeneous": r["p_value"] >= 0.05 if not np.isnan(r["p_value"]) else None,
    })

    # Batch grouping (n=79, n=118 only)
    if coh is not None:
        print(f"  PERMDISP(batch, {N_PERMS_PERMDISP} perms) …", flush=True)
        r = permdisp(X, coh, n_perms=N_PERMS_PERMDISP, seed=SEED + 1)
        sig = " ***" if r["p_value"] < 0.001 else " **" if r["p_value"] < 0.01 else \
              " *" if r["p_value"] < 0.05 else " ns"
        print(f"    F={r['F_stat']}  p={r['p_value']}{sig}")
        print(f"    group dispersions: {r['group_dispersions']}")
        permdisp_rows.append({
            "dataset": tag, "n": n, "factor": "batch",
            "n_groups": r["n_groups"], "F_stat": r["F_stat"],
            "p_value": r["p_value"], "n_perms": N_PERMS_PERMDISP,
            "group_dispersions": r["group_dispersions"],
            "dispersion_homogeneous": r["p_value"] >= 0.05 if not np.isnan(r["p_value"]) else None,
        })

pd.DataFrame(permdisp_rows).to_csv(OUT_DIAG / "permdisp_results.tsv", sep="\t", index=False)
print(f"\nSaved: {OUT_DIAG}/permdisp_results.tsv")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 2: Bootstrap 95% CIs (N=1000 resamples)")
print("=" * 65)

boot_rows = []

# ── 2a. PERMANOVA R² bootstrap (stratified by cohort) ─────────────────────────
print("\n2a. PERMANOVA R² bootstrap …")

D_39  = aitchison_dist(clr_39)
D_79  = aitchison_dist(clr_79)
D_118 = aitchison_dist(clr_118)

perm_boot_configs = [
    ("PERMANOVA_response_R2", "n=39",  D_39,  resp_39.values,  cohort_id.loc[c1_mask].values),
    ("PERMANOVA_response_R2", "n=79",  D_79,  resp_79.values,  coh_79.values),
    ("PERMANOVA_response_R2", "n=118", D_118, resp_118.values, coh_118.values),
    ("PERMANOVA_batch_R2",    "n=79",  D_79,  coh_79.values,   coh_79.values),
    ("PERMANOVA_batch_R2",    "n=118", D_118, coh_118.values,  coh_118.values),
]

for metric, n_label, D, factor_arr, cohort_arr in perm_boot_configs:
    factor_arr  = np.array(factor_arr)
    cohort_arr  = np.array(cohort_arr)
    obs_r2      = permanova_r2_only(D, factor_arr)
    boot_r2s    = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([
            rng.choice(np.where(cohort_arr == c)[0],
                       size=(cohort_arr == c).sum(), replace=True)
            for c in np.unique(cohort_arr)
        ])
        boot_r2s[b] = permanova_r2_only(D[np.ix_(idx, idx)], factor_arr[idx])
    ci_lo, ci_hi = np.percentile(boot_r2s, [2.5, 97.5])
    print(f"  {metric} {n_label}: R²={obs_r2:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    boot_rows.append({
        "metric": metric, "n": n_label,
        "point_estimate": round(obs_r2, 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "n_boot": N_BOOT,
    })


# ── 2b. Regenerate n=39 ElasticNet LOOCV predictions ─────────────────────────
print("\n2b. Regenerating n=39 ElasticNet LOOCV predictions …")

n39_preds = []
for leave_out in clr_39.index:
    actual   = resp_39[leave_out]
    tr_idx   = clr_39.index.difference([leave_out])
    X_tr_clr = clr_39.loc[tr_idx]; X_te_clr = clr_39.loc[[leave_out]]
    X_tr_raw = raw_39.loc[tr_idx]; y_tr = resp_39.loc[tr_idx]

    min_prev   = max(2, int(np.ceil(PREVALENCE_FRAC * len(tr_idx))))
    prev_cols  = (X_tr_raw > 0).sum()[lambda s: s >= min_prev].index.tolist()
    var_cut    = X_tr_clr[prev_cols].var().quantile(1.0 - VARIANCE_KEEP_FRACTION)
    hv_cols    = X_tr_clr[prev_cols].var()[lambda s: s >= var_cut].index.tolist()
    y_bin      = (y_tr == "R").astype(int).values
    pb         = {c: abs(sp_stats.pointbiserialr(y_bin, X_tr_clr[c].values)[0])
                  for c in hv_cols}
    sel        = sorted(pb, key=pb.get, reverse=True)[:min(TOP_N, len(pb))]

    m = sklearn.base.clone(ENET)
    m.fit(X_tr_clr[sel].values, y_tr.values)
    probs  = m.predict_proba(X_te_clr[sel].values)[0]
    r_prob = probs[list(m.classes_).index("R")]
    n39_preds.append({"run_accession": leave_out, "actual": actual,
                      "predicted_prob_R": round(r_prob, 6)})

n39_pred_df = pd.DataFrame(n39_preds)
n39_auc = roc_auc_score((n39_pred_df["actual"] == "R").astype(int),
                         n39_pred_df["predicted_prob_R"])
print(f"  n=39 ENet LOOCV AUC = {n39_auc:.4f}  (reference: 0.4486 from original run)")
n39_pred_df.to_csv("results/ml/loocv_n39_enet_results.tsv", sep="\t", index=False)


# ── 2c. AUC bootstrap ────────────────────────────────────────────────────────
print("\n2c. AUC bootstrap …")

pred_n79u = pd.read_csv(LOOCV_N79U, sep="\t")
pred_n79c = pd.read_csv(LOOCV_N79C, sep="\t")
pred_n118 = pd.read_csv(LOOCV_N118, sep="\t")

auc_configs = [
    ("LOOCV_AUC_ENet",  "n=39",              n39_pred_df),
    ("LOOCV_AUC_ENet",  "n=79_uncorrected",  pred_n79u),
    ("LOOCV_AUC_ENet",  "n=79_combat",       pred_n79c),
    ("LOOCV_AUC_RF",    "n=118_combat",      pred_n118),
]

for metric, n_label, pred_df in auc_configs:
    y_true  = (pred_df["actual"] == "R").astype(int).values
    y_score = pred_df["predicted_prob_R"].values
    obs_auc = roc_auc_score(y_true, y_score)
    boot_aucs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt = y_true[idx]; ys = y_score[idx]
        boot_aucs[b] = roc_auc_score(yt, ys) if 0 < yt.sum() < len(yt) else 0.5
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  {metric} {n_label}: AUC={obs_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    boot_rows.append({
        "metric": metric, "n": n_label,
        "point_estimate": round(obs_auc, 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "n_boot": N_BOOT,
    })

boot_df = pd.DataFrame(boot_rows)
boot_out = OUT_DIAG / "bootstrap_cis.tsv"
boot_df[["metric", "n", "point_estimate", "ci_lower", "ci_upper", "n_boot"]].to_csv(
    boot_out, sep="\t", index=False)
print(f"\nSaved: {boot_out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Effect sizes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3: Effect sizes — Cohen's f² added to permanova_comparison.tsv")
print("=" * 65)

cmp_df = pd.read_csv(PERM_CMP, sep="\t")
# For single-factor PERMANOVA, partial R² = R²
cmp_df["partial_R2"] = cmp_df["R2"].round(6)
# Cohen's f² = R² / (1 - R²);  f²<0.02 = small, 0.02-0.15 = medium, >0.35 = large
cmp_df["cohens_f2"]  = (cmp_df["R2"] / (1.0 - cmp_df["R2"])).round(6)
# Conventional labels
def _f2_label(f2):
    if np.isnan(f2): return "—"
    if f2 < 0.02:    return "negligible"
    if f2 < 0.15:    return "small"
    if f2 < 0.35:    return "medium"
    return "large"
cmp_df["effect_size"] = cmp_df["cohens_f2"].apply(_f2_label)

cmp_df.to_csv(PERM_CMP, sep="\t", index=False)
print(cmp_df[["analysis", "n", "factor", "R2", "cohens_f2", "effect_size", "p_value"]]
      .to_string(index=False))
print(f"\nUpdated: {PERM_CMP}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Simulation-based power analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 4: Simulation power analysis (empirical basis: cohort1 n=39)")
print(f"        {N_REPS_POWER} replicates × {N_PERMS_POWER} PERMANOVA perms each")
print("=" * 65)

y_base  = (resp_39 == "R").values.astype(np.int8)
n_base  = len(y_base)
D_base  = D_39   # precomputed 39×39; bootstrap indexes into this

print(f"\n{'n':>6}  {'power':>7}  {'95% CI':>18}  (Wilson binomial CI on {N_REPS_POWER} reps)")
print("-" * 45)

power_rows = []
power_vals = []

for n_target in TARGET_NS:
    n_sig = 0
    for _ in range(N_REPS_POWER):
        idx = rng_power.integers(0, n_base, size=n_target)
        D_s = D_base[np.ix_(idx, idx)]
        y_s = y_base[idx]

        # Skip degenerate resamples (single class; extremely rare at n≥39)
        if y_s.sum() == 0 or y_s.sum() == n_target:
            continue

        d2   = D_s ** 2
        ST   = float(np.sum(np.triu(d2, k=1))) / n_target

        def _sw_binary(grp):
            sw = 0.0
            for g in (0, 1):
                m = grp == g; ng = m.sum()
                if ng < 2: continue
                sub = d2[np.ix_(m, m)]
                sw += float(np.sum(np.triu(sub, k=1))) / ng
            return sw

        SW_obs = _sw_binary(y_s)
        SA_obs = ST - SW_obs
        F_obs  = (SA_obs) / (SW_obs / (n_target - 2)) if SW_obs > 0 else 0.0

        # Permutation test
        count_ge = 0
        for _ in range(N_PERMS_POWER):
            y_p = rng_power.permutation(y_s)
            sw_p = _sw_binary(y_p)
            sa_p = ST - sw_p
            F_p  = sa_p / (sw_p / (n_target - 2)) if sw_p > 0 else 0.0
            if F_p >= F_obs:
                count_ge += 1
        p_val = count_ge / N_PERMS_POWER
        if p_val < 0.05:
            n_sig += 1

    power = n_sig / N_REPS_POWER
    ci_lo, ci_hi = wilson_ci(n_sig, N_REPS_POWER)
    print(f"  {n_target:>4}  {power:>7.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")
    power_rows.append({"n": n_target, "power": round(power, 4),
                       "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                       "n_reps": N_REPS_POWER, "n_perms_internal": N_PERMS_POWER})
    power_vals.append(power)

power_df = pd.DataFrame(power_rows)
power_df.to_csv(OUT_POWER / "power_curve.tsv", sep="\t", index=False)
print(f"\nSaved: {OUT_POWER}/power_curve.tsv")

# ── Power curve plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(TARGET_NS, power_vals, "o-", color="#2196F3", linewidth=2, markersize=7)
ax.fill_between(TARGET_NS,
                [r["ci_lower"] for r in power_rows],
                [r["ci_upper"] for r in power_rows],
                alpha=0.20, color="#2196F3", label="95% Wilson CI")
ax.axhline(0.80, color="#F44336", linestyle="--", linewidth=1.2, label="80% power target")
ax.axhline(0.05, color="gray",    linestyle=":",  linewidth=1.0, label="α=0.05")
ax.set_xlabel("Sample size (n)", fontsize=12)
ax.set_ylabel("Empirical power (P(p<0.05))", fontsize=12)
ax.set_title(
    f"PERMANOVA(response) power — empirical bootstrap from cohort1 n=39\n"
    f"Observed R²=0.027 at n=39  ({N_REPS_POWER} reps × {N_PERMS_POWER} perms)",
    fontsize=11)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks(TARGET_NS)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_POWER / "power_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_POWER}/power_curve.png")

# 80% power crossing
n_80pct = None
for i, (nt, pwr) in enumerate(zip(TARGET_NS, power_vals)):
    if pwr >= 0.80:
        n_80pct = nt
        break
if n_80pct is None:
    # Interpolate
    for i in range(len(power_vals) - 1):
        if power_vals[i] < 0.80 <= power_vals[i + 1]:
            frac = (0.80 - power_vals[i]) / (power_vals[i + 1] - power_vals[i])
            n_80pct = int(TARGET_NS[i] + frac * (TARGET_NS[i + 1] - TARGET_NS[i]))
            break


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

# 1. Do bootstrap CIs for response R² overlap?
r2_cis = {r["n"]: (r["ci_lower"], r["ci_upper"])
           for r in boot_rows if r["metric"] == "PERMANOVA_response_R2"}
print("\n1. Bootstrap CIs for PERMANOVA response R²:")
for n_label, (lo, hi) in r2_cis.items():
    pt = next(r["point_estimate"] for r in boot_rows
              if r["metric"] == "PERMANOVA_response_R2" and r["n"] == n_label)
    print(f"   {n_label}: R²={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

# Check pairwise overlap
ns = list(r2_cis.keys())
overlaps = []
for i in range(len(ns)):
    for j in range(i+1, len(ns)):
        lo_i, hi_i = r2_cis[ns[i]]
        lo_j, hi_j = r2_cis[ns[j]]
        overlap = not (hi_i < lo_j or hi_j < lo_i)
        overlaps.append((ns[i], ns[j], overlap))
        print(f"   {ns[i]} vs {ns[j]}: CIs {'OVERLAP' if overlap else 'DO NOT overlap'}")

# 2. PERMDISP batch significance
print("\n2. PERMDISP(batch) — are cohort dispersions homogeneous?")
for row in permdisp_rows:
    if row["factor"] == "batch":
        hom = "homogeneous (p≥0.05)" if row["dispersion_homogeneous"] else "HETEROGENEOUS (p<0.05)"
        print(f"   {row['dataset']} n={row['n']}: F={row['F_stat']}, p={row['p_value']} — {hom}")
        if not row["dispersion_homogeneous"]:
            print(f"   ⚠  PERMANOVA batch R² at {row['dataset']} reflects BOTH centroid shift")
            print(f"      AND dispersion differences — results should be interpreted cautiously.")
        else:
            print(f"   ✓  PERMANOVA batch R² reflects centroid shift only (dispersion homogeneous).")

# 3. Power curve
print(f"\n3. Power analysis (PERMANOVA response, empirical R²≈0.027 at n=39):")
for r in power_rows:
    print(f"   n={r['n']:>4}: power={r['power']:.3f}  CI=[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]")
if n_80pct:
    print(f"\n   → 80% power first reached at approximately n={n_80pct}")
else:
    print(f"\n   → 80% power NOT reached within tested range (n≤{TARGET_NS[-1]})")
    print(f"      The observed effect size (R²≈0.027) is very small; n>300 likely needed.")

print()
print("All outputs saved to results/ml/batch_diagnostics/ and results/ml/power_analysis/")
