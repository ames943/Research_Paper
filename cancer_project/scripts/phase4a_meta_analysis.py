#!/usr/bin/env python3
"""
Phase 4a: Meta-analytic effect combination across cohorts.

Approach:
  For each cohort (1, 2, 3):
    1. Feature selection on full cohort (prevalence + top-50% var + top-100 |PBr|)
    2. Fit ElasticNet (C=1.0, l1_ratio=0.5)
    3. Bootstrap (N=500) on full cohort to estimate per-genus coefficient SE

  Random-effects meta-analysis (DerSimonian-Laird) per genus:
    - Include only genera with |EN coef| > 1e-6 in ≥2 cohorts
    - Use bootstrap SE as sampling variance
    - Compute pooled effect, 95% CI, Q statistic, I²
    - Apply BH FDR correction to z-test p-values

Outputs: results/ml/meta_analysis/
  per_cohort_coefficients.tsv   — EN coef + bootstrap SE for all genera
  meta_analysis_results.tsv     — pooled effect, CI, heterogeneity, FDR q
  meta_analysis_summary.md
"""

import os, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CLR_PATH    = "results/ml/n118_3cohort/X_genus_clr.tsv"
RAW_PATH    = "results/ml/n118_3cohort/X_genus_raw.tsv"
LABELS_PATH = "metadata/response_labels_3cohort.tsv"
OUT_DIR     = "results/ml/meta_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100
EN_C                   = 1.0
EN_L1                  = 0.5
N_BOOT                 = 500
SEED                   = 42
MIN_SE                 = 1e-4  # floor SE to avoid division by zero

print(f"[{time.strftime('%H:%M:%S')}] Loading data …", flush=True)
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
keep     = response.notna()
clr      = clr.loc[keep]; raw = raw.loc[keep]; response = response.loc[keep]

FEATURES = clr.columns.tolist()
N_FEAT   = len(FEATURES)
feat_idx = {f: i for i, f in enumerate(FEATURES)}

cohort_masks = {
    "cohort1": clr.index.str.startswith("SRR5930"),
    "cohort2": clr.index.str.startswith("SRR11413"),
    "cohort3": clr.index.str.startswith("SRR6000"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def select_features(clr_df, raw_df, resp):
    n        = len(clr_df)
    min_prev = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres     = (raw_df > 0).sum(axis=0)
    prev     = pres[pres >= min_prev].index.tolist()
    vv       = clr_df[prev].var(axis=0)
    hv       = vv[vv >= vv.quantile(1.0 - VARIANCE_KEEP_FRACTION)].index.tolist()
    y_bin    = (resp == "R").astype(int).values
    pbs      = {c: abs(stats.pointbiserialr(y_bin, clr_df[c].values)[0]) for c in hv}
    return pd.Series(pbs).sort_values(ascending=False).head(TOP_N).index.tolist()


def fit_en_coef(X_tr, y_tr) -> np.ndarray:
    """Fit EN, return coefficient vector (1-D over features passed in)."""
    m = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=EN_C, l1_ratio=EN_L1,
        class_weight="balanced", max_iter=5000, tol=1e-3, random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    # coef_ shape: (1, n_feat) for binary; take row 0
    return m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[list(m.classes_).index("R")]


def bootstrap_se(X, y, n_boot=N_BOOT, seed=SEED):
    """Bootstrap SE for each feature coefficient."""
    rng     = np.random.default_rng(seed)
    n       = X.shape[0]
    n_feat  = X.shape[1]
    boot_coefs = np.zeros((n_boot, n_feat))
    for b in range(n_boot):
        idx        = rng.integers(0, n, size=n)
        Xb, yb     = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            boot_coefs[b] = 0.0
            continue
        try:
            boot_coefs[b] = fit_en_coef(Xb, yb)
        except Exception:
            boot_coefs[b] = 0.0
    return boot_coefs.std(axis=0)


# ── Per-cohort EN fit + bootstrap ─────────────────────────────────────────────

cohort_data = {}  # cname → {coef_vec, se_vec, sel_list}

for cname, cmask in cohort_masks.items():
    idx     = clr.index[cmask].tolist()
    clr_c   = clr.loc[idx]; raw_c = raw.loc[idx]; resp_c = response.loc[idx]
    n       = len(idx)
    print(f"\n[{time.strftime('%H:%M:%S')}] {cname} (n={n}) — feature selection …", flush=True)

    sel     = select_features(clr_c, raw_c, resp_c)
    X       = clr_c[sel].values.astype(float)
    y       = resp_c.values

    print(f"  {len(sel)} features selected. Fitting EN …", flush=True)
    coef_sel = fit_en_coef(X, y)

    print(f"  Bootstrap SE (N={N_BOOT}) …", flush=True)
    se_sel   = bootstrap_se(X, y)

    # Map back to full feature space (zeros for non-selected)
    coef_vec = np.zeros(N_FEAT)
    se_vec   = np.full(N_FEAT, np.nan)
    for i, s in enumerate(sel):
        coef_vec[feat_idx[s]] = coef_sel[i]
        se_vec[feat_idx[s]]   = max(se_sel[i], MIN_SE)

    n_nz = (np.abs(coef_vec) > 1e-6).sum()
    print(f"  Nonzero EN coefs: {n_nz}", flush=True)
    cohort_data[cname] = dict(coef=coef_vec, se=se_vec, sel=sel, n=n)


# ── Save per-cohort coefficients ───────────────────────────────────────────────

per_cohort_rows = []
for gi, genus in enumerate(FEATURES):
    row = {"genus": genus}
    for cname, cd in cohort_data.items():
        row[f"coef_{cname}"]  = round(cd["coef"][gi], 6)
        row[f"se_{cname}"]    = round(cd["se"][gi], 6) if not np.isnan(cd["se"][gi]) else None
        row[f"in_sel_{cname}"]= int(abs(cd["coef"][gi]) > 1e-6 or
                                     genus in cd["sel"])
    per_cohort_rows.append(row)

per_cohort_df = pd.DataFrame(per_cohort_rows)
per_cohort_df.to_csv(f"{OUT_DIR}/per_cohort_coefficients.tsv", sep="\t", index=False)
print(f"\nSaved: {OUT_DIR}/per_cohort_coefficients.tsv", flush=True)


# ── Random-effects meta-analysis (DerSimonian-Laird) ──────────────────────────
# Only for genera with |coef| > 1e-6 in ≥2 cohorts.

print(f"\n[{time.strftime('%H:%M:%S')}] Running DerSimonian-Laird meta-analysis …", flush=True)

meta_rows = []
for gi, genus in enumerate(FEATURES):
    effects = []
    variances = []
    cohort_labels = []
    for cname, cd in cohort_data.items():
        c = cd["coef"][gi]
        s = cd["se"][gi]
        if abs(c) > 1e-6 and not np.isnan(s) and s > 0:
            effects.append(c)
            variances.append(s**2)
            cohort_labels.append(cname)

    if len(effects) < 2:
        continue

    effects    = np.array(effects)
    variances  = np.array(variances)
    weights_fe = 1.0 / variances  # fixed-effects weights

    # Fixed-effects pooled estimate (for Q calculation)
    theta_fe   = np.sum(weights_fe * effects) / np.sum(weights_fe)

    # Q statistic (heterogeneity)
    Q          = np.sum(weights_fe * (effects - theta_fe) ** 2)
    df_Q       = len(effects) - 1
    p_Q        = 1 - stats.chi2.cdf(Q, df=df_Q) if df_Q > 0 else 1.0

    # DerSimonian-Laird estimate of between-study variance τ²
    c_factor   = np.sum(weights_fe) - np.sum(weights_fe**2) / np.sum(weights_fe)
    tau2       = max(0.0, (Q - df_Q) / c_factor) if c_factor > 0 else 0.0
    I2         = max(0.0, (Q - df_Q) / Q * 100) if Q > 0 else 0.0

    # Random-effects weights
    weights_re = 1.0 / (variances + tau2)
    theta_re   = np.sum(weights_re * effects) / np.sum(weights_re)
    se_re      = np.sqrt(1.0 / np.sum(weights_re))

    z          = theta_re / se_re if se_re > 0 else 0.0
    p_z        = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_lo      = theta_re - 1.96 * se_re
    ci_hi      = theta_re + 1.96 * se_re

    meta_rows.append({
        "genus":          genus,
        "k_cohorts":      len(effects),
        "cohorts":        ",".join(cohort_labels),
        "pooled_effect":  round(theta_re, 5),
        "se_re":          round(se_re, 5),
        "ci_lo_95":       round(ci_lo, 5),
        "ci_hi_95":       round(ci_hi, 5),
        "z_stat":         round(z, 4),
        "z_pval":         round(p_z, 6),
        "Q_stat":         round(Q, 4),
        "Q_pval":         round(p_Q, 4),
        "I2_pct":         round(I2, 1),
        "tau2":           round(tau2, 6),
        "direction":      "R+" if theta_re > 0 else "NR+",
    })

meta_df = pd.DataFrame(meta_rows)

# BH FDR correction on z-test p-values
if len(meta_df) > 0:
    pvals_z = meta_df["z_pval"].values
    m_      = len(pvals_z)
    order   = np.argsort(pvals_z)
    ranks   = np.empty(m_, dtype=int); ranks[order] = np.arange(1, m_ + 1)
    q       = pvals_z * m_ / ranks
    q_mono  = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out   = np.empty(m_); q_out[order] = np.minimum(q_mono, 1.0)
    meta_df["bh_fdr_q"] = q_out.round(6)
    meta_df["fdr_sig"]  = meta_df["bh_fdr_q"] < 0.05

    meta_df = meta_df.sort_values("bh_fdr_q").reset_index(drop=True)
    meta_df.to_csv(f"{OUT_DIR}/meta_analysis_results.tsv", sep="\t", index=False)
    print(f"  Meta-analysis genera: {len(meta_df)}", flush=True)
    print(f"  FDR-significant: {meta_df['fdr_sig'].sum()}", flush=True)

    n_hetero = (meta_df["Q_pval"] < 0.05).sum()
    print(f"  Significant heterogeneity (Q p<0.05): {n_hetero}", flush=True)


# ── Summary markdown ───────────────────────────────────────────────────────────

lines = ["# Phase 4a: Meta-Analytic Effect Combination\n",
         "Random-effects meta-analysis (DerSimonian-Laird) across 3 cohorts.\n",
         f"EN (C=1.0, l1_ratio=0.5), bootstrap SE (N={N_BOOT}) per cohort.\n",
         "Only genera with |coef| > 0 in ≥2 cohorts are included.\n"]

if len(meta_df) == 0:
    lines.append("No genera had nonzero coefficients in ≥2 cohorts — meta-analysis empty.")
else:
    n_sig = meta_df["fdr_sig"].sum()
    n_het = (meta_df["Q_pval"] < 0.05).sum()
    lines.append(f"Genera eligible for meta-analysis: **{len(meta_df)}**")
    lines.append(f"FDR-significant pooled effects (BH q<0.05): **{n_sig}**")
    lines.append(f"Genera with significant heterogeneity (Q p<0.05): **{n_het}**")
    lines.append("")

    lines.append("## Top 20 by pooled effect (sorted by BH FDR q):\n")
    top = meta_df.head(20)
    lines.append(top[["genus","k_cohorts","pooled_effect","ci_lo_95","ci_hi_95",
                       "z_pval","bh_fdr_q","I2_pct","direction"]].to_string(index=False))
    lines.append("")

    lines.append("## Interpretation\n")
    if n_sig > 0:
        lines.append(f"{n_sig} genera show statistically significant pooled effects across cohorts,")
        lines.append("despite no cohort-specific LOOCV AUC reaching significance.")
        lines.append("This demonstrates the value of random-effects combination: even when")
        lines.append("per-cohort samples are underpowered (n=39-40), consistent directional")
        lines.append("effects can become detectable when combined across cohorts.")
    else:
        lines.append("No genus reached FDR significance in the meta-analysis. This is consistent")
        lines.append("with the overall null finding: cross-cohort effects are not merely masked")
        lines.append("by underpowered per-cohort analysis — they are genuinely absent or too")
        lines.append("heterogeneous (different directions in different cohorts) to pool reliably.")
        lines.append("")
        lines.append(f"Heterogeneous genera (Q p<0.05): {n_het} — these show directionally")
        lines.append("inconsistent effects across cohorts, which is itself informative:")
        lines.append("the microbiome-response association is cohort/study-specific, not universal.")

    lines.append("")
    if n_het > 0:
        het_genera = meta_df[meta_df["Q_pval"] < 0.05][["genus","k_cohorts","I2_pct","Q_pval","direction"]]
        lines.append("### Heterogeneous genera (Q p<0.05):")
        lines.append(het_genera.to_string(index=False))

lines.append(f"\nOutputs saved to: {OUT_DIR}/")

with open(f"{OUT_DIR}/meta_analysis_summary.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 4a meta-analysis complete. Outputs in {OUT_DIR}/", flush=True)
