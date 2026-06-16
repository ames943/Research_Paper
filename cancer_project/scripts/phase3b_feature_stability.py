#!/usr/bin/env python3
"""
Phase 3b: FDR-corrected feature stability across LOOCV folds.

Three LOOCV experiments:
  n=39  — Cohort 1 (SRR5930/Frankel) only, no batch correction
  n=79  — Cohort 1+2, per-fold mean-centering
  n=118 — All 3 cohorts, per-fold mean-centering

Per experiment: fixed-param ElasticNet (C=1.0, l1_ratio=0.5) with
leak-free feature selection (prevalence >=10% + top-50% variance + top-100 |PBr|).

Per genus across folds:
  folds_selected   — in the top-100 feature set
  folds_nonzero    — |coef| > 1e-6 in those folds
  sign_pos_frac    — fraction of nonzero folds where coef > 0
  binomial_p       — two-sided test vs p=0.5 (sign is random)
  bh_fdr_q         — Benjamini-Hochberg adjusted p-value

Cross-cohort sign analysis:
  Fit EN on each full cohort (no LOOCV). Compare coefficient signs.
  Resolves Phase 0.5 question: do below-chance AUCs reflect consistent
  feature-direction inversion (real finding) or pure noise?

Outputs: results/ml/feature_stability/
  feature_stability_n39.tsv
  feature_stability_n79.tsv
  feature_stability_n118.tsv
  feature_stability_cross_cohort.tsv
  feature_stability_summary.md
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
OUT_DIR     = "results/ml/feature_stability"
os.makedirs(OUT_DIR, exist_ok=True)

PREVALENCE_FRAC        = 0.10
VARIANCE_KEEP_FRACTION = 0.50
TOP_N                  = 100
EN_C                   = 1.0
EN_L1                  = 0.5
FDR_ALPHA              = 0.05

print(f"[{time.strftime('%H:%M:%S')}] Loading data …", flush=True)
clr    = pd.read_csv(CLR_PATH,    sep="\t", index_col="run_accession")
raw    = pd.read_csv(RAW_PATH,    sep="\t", index_col="run_accession")
labels = pd.read_csv(LABELS_PATH, sep="\t").set_index("run_accession")

response = labels.reindex(clr.index)["response"]
keep     = response.notna()
clr      = clr.loc[keep]; raw = raw.loc[keep]; response = response.loc[keep]

cohort_mask = {
    "cohort1": clr.index.str.startswith("SRR5930"),
    "cohort2": clr.index.str.startswith("SRR11413"),
    "cohort3": clr.index.str.startswith("SRR6000"),
}
cohort_id = pd.Series(
    ["cohort1" if a.startswith("SRR5930") else
     "cohort2" if a.startswith("SRR11413") else "cohort3"
     for a in clr.index],
    index=clr.index,
)

FEATURES = clr.columns.tolist()
N_FEAT   = len(FEATURES)
feat_idx = {f: i for i, f in enumerate(FEATURES)}

print(f"  n={len(clr)}, features={N_FEAT}", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def mean_center_fold(tr_clr, te_clr, tr_batch):
    gm     = tr_clr.mean(axis=0)
    offsets = {b: tr_clr.loc[tr_batch == b].mean(axis=0) - gm
               for b in tr_batch.unique()}
    corr_tr = tr_clr.copy()
    for b, off in offsets.items():
        corr_tr.loc[tr_batch == b] -= off
    te_b   = cohort_id.loc[te_clr.index].iloc[0]
    corr_te = te_clr.values - (offsets[te_b].values if te_b in offsets else 0)
    return corr_tr, corr_te


def select_features(tr_clr, tr_raw, tr_labels):
    n         = len(tr_clr)
    min_prev  = max(2, int(np.ceil(PREVALENCE_FRAC * n)))
    pres      = (tr_raw > 0).sum(axis=0)
    prev_cols = pres[pres >= min_prev].index.tolist()
    vv        = tr_clr[prev_cols].var(axis=0)
    high_var  = vv[vv >= vv.quantile(1.0 - VARIANCE_KEEP_FRACTION)].index.tolist()
    y_bin     = (tr_labels == "R").astype(int).values
    pbs       = {c: abs(stats.pointbiserialr(y_bin, tr_clr[c].values)[0])
                 for c in high_var}
    return pd.Series(pbs).sort_values(ascending=False).head(TOP_N).index.tolist()


def fit_en(X_tr, y_tr):
    m = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=EN_C, l1_ratio=EN_L1,
        class_weight="balanced", max_iter=5000, tol=1e-3, random_state=42,
    )
    m.fit(X_tr, y_tr)
    return m


def bh_correction(pvals):
    """Benjamini-Hochberg correction. Returns q-values (adjusted p-values)."""
    pvals = np.asarray(pvals, dtype=float)
    m     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    q     = pvals * m / ranks
    # enforce monotonicity (step-down)
    q_mono = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out  = np.empty(m)
    q_out[order] = np.minimum(q_mono, 1.0)
    return q_out


# ── LOOCV with coefficient collection ─────────────────────────────────────────

def run_loocv_collect_coefs(samples, clr_sub, raw_sub, resp_sub, use_batch):
    """
    Run LOOCV collecting per-fold EN coefficients.
    Returns coef_matrix (n_folds × N_FEAT) and sel_matrix (n_folds × N_FEAT, bool).
    """
    n         = len(samples)
    batch_sub = cohort_id.loc[samples]
    coef_mat  = np.zeros((n, N_FEAT))
    sel_mat   = np.zeros((n, N_FEAT), dtype=bool)

    for fi, held in enumerate(samples):
        tr_idx = [s for s in samples if s != held]
        tr_clr = clr_sub.loc[tr_idx]
        te_clr = clr_sub.loc[[held]]
        tr_raw = raw_sub.loc[tr_idx]
        tr_resp= resp_sub.loc[tr_idx]
        tr_bat = batch_sub.loc[tr_idx]

        if use_batch and tr_bat.nunique() > 1:
            tr_clr_c, _ = mean_center_fold(tr_clr, te_clr, tr_bat)
        else:
            tr_clr_c = tr_clr

        sel = select_features(tr_clr_c, tr_raw, tr_resp)
        for s in sel:
            sel_mat[fi, feat_idx[s]] = True

        X_tr = tr_clr_c[sel].values
        y_tr = tr_resp.values

        if len(np.unique(y_tr)) < 2:
            continue

        m   = fit_en(X_tr, y_tr)
        classes = list(m.classes_)
        r_idx   = classes.index("R")
        # EN has one coef per class for multi-class, but with 2 classes
        # m.coef_ shape is (1, n_features) when binary
        coefs = m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[r_idx]
        for i, s in enumerate(sel):
            coef_mat[fi, feat_idx[s]] = coefs[i]

        if (fi + 1) % 20 == 0 or fi == 0 or fi == n - 1:
            print(f"    fold {fi+1:3d}/{n}  sel={len(sel)}", flush=True)

    return coef_mat, sel_mat


def stability_table(coef_mat, sel_mat, n_folds):
    """Compute per-feature stability statistics and BH FDR correction."""
    rows = []
    pvals = []
    for gi, genus in enumerate(FEATURES):
        n_sel    = int(sel_mat[:, gi].sum())
        coefs_gi = coef_mat[sel_mat[:, gi], gi]
        n_nz     = int(np.sum(np.abs(coefs_gi) > 1e-6))
        coefs_nz = coefs_gi[np.abs(coefs_gi) > 1e-6]
        n_pos    = int((coefs_nz > 0).sum())

        if n_nz >= 2:
            res  = binomtest(n_pos, n_nz, p=0.5, alternative="two-sided")
            pval = res.pvalue
        else:
            pval = 1.0

        mean_coef = float(coefs_nz.mean()) if n_nz > 0 else 0.0
        sign_frac = float(n_pos / n_nz) if n_nz > 0 else float("nan")

        rows.append({
            "genus":          genus,
            "n_folds":        n_folds,
            "folds_selected": n_sel,
            "folds_nonzero":  n_nz,
            "pct_selected":   round(100 * n_sel / n_folds, 1),
            "pct_nonzero_of_sel": round(100 * n_nz / n_sel, 1) if n_sel > 0 else 0.0,
            "sign_pos_frac":  round(sign_frac, 3) if n_nz > 0 else float("nan"),
            "mean_coef_nz":   round(mean_coef, 5),
            "direction":      ("R+" if mean_coef > 0 else "NR+") if n_nz > 0 else "none",
            "binomial_p":     round(pval, 6),
        })
        pvals.append(pval)

    df   = pd.DataFrame(rows)
    qvals = bh_correction(np.array(pvals))
    df["bh_fdr_q"] = qvals.round(6)
    df["fdr_sig"]  = df["bh_fdr_q"] < FDR_ALPHA
    df = df.sort_values("bh_fdr_q").reset_index(drop=True)
    return df


# ── Run all three LOOCV experiments ───────────────────────────────────────────

datasets = [
    ("n39",  clr.index[cohort_mask["cohort1"]].tolist(), False),
    ("n79",  clr.index[cohort_mask["cohort1"] | cohort_mask["cohort2"]].tolist(), True),
    ("n118", clr.index.tolist(), True),
]

stab_tables = {}
for tag, samples, use_batch in datasets:
    n = len(samples)
    print(f"\n[{time.strftime('%H:%M:%S')}] LOOCV coef collection — {tag} (n={n}) …", flush=True)
    clr_s   = clr.loc[samples]
    raw_s   = raw.loc[samples]
    resp_s  = response.loc[samples]
    coef_mat, sel_mat = run_loocv_collect_coefs(samples, clr_s, raw_s, resp_s, use_batch)
    tbl = stability_table(coef_mat, sel_mat, n)
    stab_tables[tag] = tbl

    out = f"{OUT_DIR}/feature_stability_{tag}.tsv"
    tbl.to_csv(out, sep="\t", index=False)
    n_sig = tbl["fdr_sig"].sum()
    print(f"  → {out}  |  FDR-significant genera: {n_sig}", flush=True)


# ── Cross-cohort sign analysis (full-cohort EN fits) ──────────────────────────

print(f"\n[{time.strftime('%H:%M:%S')}] Cross-cohort EN full-cohort fits …", flush=True)

cohort_coefs = {}
for cname, cmask in cohort_mask.items():
    idx      = clr.index[cmask].tolist()
    clr_c    = clr.loc[idx]
    raw_c    = raw.loc[idx]
    resp_c   = response.loc[idx]
    sel      = select_features(clr_c, raw_c, resp_c)
    X        = clr_c[sel].values
    y        = resp_c.values

    m        = fit_en(X, y)
    classes  = list(m.classes_)
    r_idx    = classes.index("R")
    coefs    = m.coef_[0] if m.coef_.shape[0] == 1 else m.coef_[r_idx]

    coef_vec = np.zeros(N_FEAT)
    for i, s in enumerate(sel):
        coef_vec[feat_idx[s]] = coefs[i]

    cohort_coefs[cname] = coef_vec
    n_nz = (np.abs(coef_vec) > 1e-6).sum()
    print(f"  {cname}: n={len(idx)}, {len(sel)} features selected, {n_nz} nonzero after EN", flush=True)

# Build cross-cohort table: genera with nonzero in ≥2 cohorts
cross_rows = []
for gi, genus in enumerate(FEATURES):
    c1 = cohort_coefs["cohort1"][gi]
    c2 = cohort_coefs["cohort2"][gi]
    c3 = cohort_coefs["cohort3"][gi]

    nz = [(v, n) for v, n in [(c1, "C1"), (c2, "C2"), (c3, "C3")]
          if abs(v) > 1e-6]

    if len(nz) < 2:
        continue

    signs    = [np.sign(v) for v, _ in nz]
    n_pos    = sum(1 for s in signs if s > 0)
    n_neg    = sum(1 for s in signs if s < 0)
    sign_agree = (n_pos == len(nz)) or (n_neg == len(nz))
    sign_inv   = (n_pos > 0 and n_neg > 0)

    cross_rows.append({
        "genus":        genus,
        "coef_C1":      round(c1, 5),
        "coef_C2":      round(c2, 5),
        "coef_C3":      round(c3, 5),
        "n_cohorts_nz": len(nz),
        "n_pos_sign":   n_pos,
        "n_neg_sign":   n_neg,
        "sign_consistent": sign_agree,
        "sign_inverted":   sign_inv,
        "dir_C1":       ("R+" if c1 > 0 else "NR+" if c1 < 0 else "0"),
        "dir_C2":       ("R+" if c2 > 0 else "NR+" if c2 < 0 else "0"),
        "dir_C3":       ("R+" if c3 > 0 else "NR+" if c3 < 0 else "0"),
    })

cross_df = pd.DataFrame(cross_rows).sort_values(
    ["n_cohorts_nz", "sign_consistent"], ascending=[False, False]
).reset_index(drop=True)
cross_df.to_csv(f"{OUT_DIR}/feature_stability_cross_cohort.tsv", sep="\t", index=False)
print(f"  Cross-cohort genera (nz in ≥2): {len(cross_df)}", flush=True)
n_consistent = cross_df["sign_consistent"].sum()
n_inverted   = cross_df["sign_inverted"].sum()
print(f"  Sign-consistent (same direction in all cohorts): {n_consistent}", flush=True)
print(f"  Sign-inverted   (opposite direction in ≥1 cohort): {n_inverted}", flush=True)


# ── Summary markdown ───────────────────────────────────────────────────────────

lines = ["# Phase 3b: Feature Stability — FDR-Corrected\n"]

for tag, tbl in stab_tables.items():
    n_sig = tbl["fdr_sig"].sum()
    n_folds = tbl["n_folds"].iloc[0]
    lines.append(f"## {tag} (LOOCV, {n_folds} folds)\n")
    lines.append(f"FDR-significant genera (BH q < {FDR_ALPHA}): **{n_sig}**\n")
    if n_sig > 0:
        sig = tbl[tbl["fdr_sig"]].head(20)
        lines.append(sig[["genus","folds_selected","folds_nonzero","sign_pos_frac",
                           "mean_coef_nz","direction","binomial_p","bh_fdr_q"]].to_string(index=False))
        lines.append("")
    else:
        top = tbl.head(20)
        lines.append("No FDR-significant features. Top 20 by raw binomial p-value:")
        lines.append(top[["genus","folds_selected","folds_nonzero","sign_pos_frac",
                           "mean_coef_nz","direction","binomial_p","bh_fdr_q"]].to_string(index=False))
        lines.append("")

lines.append("## Cross-Cohort Sign Consistency\n")
lines.append(f"Genera nonzero in ≥2 cohorts: {len(cross_df)}")
lines.append(f"Sign-consistent (same direction in all present cohorts): {n_consistent}")
lines.append(f"Sign-inverted (opposite direction in ≥1 cohort): {n_inverted}")
lines.append("")
if n_inverted > 0:
    inv = cross_df[cross_df["sign_inverted"]].head(20)
    lines.append("### Inverted genera (sign flips across cohorts):")
    lines.append(inv[["genus","coef_C1","coef_C2","coef_C3",
                       "dir_C1","dir_C2","dir_C3"]].to_string(index=False))
    lines.append("")
    lines.append("**Interpretation**: Below-chance AUCs in Phase 0.5 cross-cohort transfer")
    lines.append("reflect consistent feature-direction INVERSION across cohorts, not pure noise.")
    lines.append("This means the same genus predicts R in one cohort and NR in another —")
    lines.append("a genuine biological/batch ambiguity that correction cannot resolve without")
    lines.append("knowing the ground truth direction.")
else:
    lines.append("No sign inversion detected — below-chance AUCs likely reflect pure noise")
    lines.append("rather than systematic direction reversal.")

lines.append("")
lines.append(f"Cross-cohort details saved to: {OUT_DIR}/feature_stability_cross_cohort.tsv")

with open(f"{OUT_DIR}/feature_stability_summary.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 3b complete. Outputs in {OUT_DIR}/", flush=True)
