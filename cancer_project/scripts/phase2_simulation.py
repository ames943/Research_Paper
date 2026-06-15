#!/usr/bin/env python3
"""
Phase 2: Simulation study — when do batch correction methods help vs. hurt?

CLR-space Gaussian simulation calibrated to real n=118 dataset parameters.
Sweeps signal R² × batch R² × n_per_cohort × n_cohorts × correction method.

Usage
-----
    cd cancer_project/

    # Step 1 — estimate runtime (always run first):
    python3 scripts/phase2_simulation.py --mode timing

    # Step 2 — after confirming timing:
    python3 scripts/phase2_simulation.py --mode full [--n-reps 5] [--max-cohorts 3]

    # Re-run calibration only:
    python3 scripts/phase2_simulation.py --mode calibrate [--force-calibrate]
"""

import argparse
import itertools
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
CLR_PATH = "results/ml/n118_3cohort/X_genus_clr.tsv"
RAW_PATH = "results/ml/n118_3cohort/X_genus_raw.tsv"
OUT_DIR  = Path("results/ml/simulation")

SIGNAL_F2_TARGETS = [0.007, 0.027, 0.05, 0.10]
BATCH_F2_TARGETS  = [0.000, 0.040, 0.080, 0.150]
N_PER_COHORT_GRID = [20, 40, 80]
N_COHORTS_GRID    = [1, 2, 3, 4]
METHODS           = ["none", "mean_centering", "location_scale",
                     "quantile_mapping", "percentile_norm", "cohort_covariate"]

TOP_P              = 75       # genera to retain (by prevalence)
N_SIG_FEAT         = 10       # feature indices 0–9 carry response signal
N_BATCH_FEAT       = 10       # feature indices 10–19 carry batch signal
N_REPS_DEFAULT     = 10       # replicates per grid cell
N_CV_FOLDS         = 5
N_CALIB_REPS       = 30       # datasets averaged per binary-search iteration
CALIB_BISECT_ITER  = 30
VARIANCE_KEEP_FRAC = 0.50
TOP_N_UNIVARIATE   = 100
SEED               = 42
REF_BATCH          = 1

# Per-cohort variance scale factors motivated by Phase 0 PERMDISP F-stats
BATCH_SCALE = {1: 1.0, 2: 1.3, 3: 0.8, 4: 1.1}

# Real data reference points for plot annotations
REAL_SIG_F2_N39  = 0.027   # PERMANOVA response R² on cohort-1-only (n=39)
REAL_SIG_F2_N118 = 0.007   # PERMANOVA response R² pooled n=118
REAL_BATCH_F2    = 0.077   # batch R² pooled n=118
REAL_N_PER       = 40      # per-cohort n (≈39/40/39)
REAL_N_COH       = 3


# ════════════════════════════════════════════════════════════════════════════════
# LOAD REAL DATA → FIT LEDOITWOLF  (executed at import)
# ════════════════════════════════════════════════════════════════════════════════
print("Loading real data …", flush=True)
clr_full = pd.read_csv(CLR_PATH, sep="\t", index_col="run_accession")
raw_full = pd.read_csv(RAW_PATH, sep="\t", index_col="run_accession")

prevalence = (raw_full > 0).mean(axis=0)
top_genera = prevalence.nlargest(TOP_P).index.tolist()
clr_top    = clr_full[top_genera].values.astype(np.float64)   # (118, 75)
n_real, p_real = clr_top.shape

print(f"  n={n_real}, p={p_real}  top-{TOP_P} genera "
      f"(prevalence {prevalence[top_genera].min():.2f}–{prevalence[top_genera].max():.2f})")

lw       = LedoitWolf().fit(clr_top)
mu_real  = clr_top.mean(axis=0)    # (75,)
cov_real = lw.covariance_          # (75, 75) regularized
print(f"  LedoitWolf shrinkage: {lw.shrinkage_:.4f}", flush=True)

# Fixed feature sets (reproducible)
SIG_FEAT_IDXS   = np.arange(N_SIG_FEAT)                            # 0–9
BATCH_FEAT_IDXS = np.arange(N_SIG_FEAT, N_SIG_FEAT + N_BATCH_FEAT) # 10–19

# Fixed per-cohort batch direction vectors (unit vectors in batch-feature subspace)
_dir_rng = np.random.default_rng(SEED + 999)
BATCH_DIRECTIONS: dict = {1: np.zeros(N_BATCH_FEAT)}
for _c in range(2, 5):
    _v = _dir_rng.standard_normal(N_BATCH_FEAT)
    BATCH_DIRECTIONS[_c] = _v / np.linalg.norm(_v)


# ════════════════════════════════════════════════════════════════════════════════
# FAST PERMANOVA R²  (no permutation — sufficient for calibration + diagnostics)
# ════════════════════════════════════════════════════════════════════════════════
def permanova_r2(X: np.ndarray, group: np.ndarray) -> float:
    """Aitchison (= Euclidean on CLR) PERMANOVA R². No permutation test."""
    d2 = squareform(pdist(X, metric="euclidean")) ** 2
    n  = d2.shape[0]
    ST = float(np.sum(np.triu(d2, k=1))) / n
    if ST < 1e-20:
        return 0.0
    SW = 0.0
    for g in np.unique(group):
        m = group == g; ng = int(m.sum())
        if ng < 2:
            continue
        SW += float(np.sum(np.triu(d2[np.ix_(m, m)], k=1))) / ng
    return max(0.0, float((ST - SW) / ST))


# ════════════════════════════════════════════════════════════════════════════════
# SIMULATION GENERATOR
# ════════════════════════════════════════════════════════════════════════════════
def simulate_dataset(delta_sig: float, delta_batch: float,
                     n_per_cohort: int, n_cohorts: int,
                     local_rng: np.random.Generator) -> tuple:
    """
    Generate a simulated CLR-space dataset from the LedoitWolf-fitted Gaussian.

    Labels are balanced per-cohort (first n_r samples = R=1, rest = NR=0).
    Returns X (n_total × p), y (n_total,), batch (n_total,).
    """
    n_total = n_per_cohort * n_cohorts
    X       = local_rng.multivariate_normal(mu_real, cov_real, size=n_total)
    y       = np.zeros(n_total, dtype=int)
    batch   = np.zeros(n_total, dtype=int)
    n_r     = n_per_cohort // 2

    for c_idx in range(n_cohorts):
        c     = c_idx + 1
        start = c_idx * n_per_cohort
        end   = start + n_per_cohort
        batch[start:end] = c
        y[start : start + n_r] = 1   # first half = R

        # Response signal on SIG_FEAT_IDXS for R-labeled samples
        if delta_sig > 0:
            X[start : start + n_r, SIG_FEAT_IDXS] += delta_sig

        # Batch effect: mean shift + variance scaling on BATCH_FEAT_IDXS
        if c > 1 and delta_batch > 0:
            shift = delta_batch * BATCH_DIRECTIONS[c]
            X[start:end, BATCH_FEAT_IDXS] += shift

            scale = BATCH_SCALE.get(c, 1.0)
            if scale != 1.0:
                feat_mean = X[start:end, BATCH_FEAT_IDXS].mean(axis=0)
                X[start:end, BATCH_FEAT_IDXS] = (
                    feat_mean + (X[start:end, BATCH_FEAT_IDXS] - feat_mean) * scale
                )

    return X, y, batch


# ════════════════════════════════════════════════════════════════════════════════
# BATCH CORRECTION  (numpy, per-fold, handles multi-sample test sets)
# ════════════════════════════════════════════════════════════════════════════════
def _correct_none(X_tr, X_te, bat_tr, bat_te):
    return X_tr, X_te


def _correct_mean_centering(X_tr, X_te, bat_tr, bat_te):
    gm = X_tr.mean(axis=0)
    offsets = {}
    X_tr_c = X_tr.copy()
    for b in np.unique(bat_tr):
        m = bat_tr == b
        offsets[b] = X_tr[m].mean(axis=0) - gm
        X_tr_c[m] -= offsets[b]
    X_te_c = X_te.copy()
    for b in np.unique(bat_te):
        X_te_c[bat_te == b] -= offsets.get(b, np.zeros(X_tr.shape[1]))
    return X_tr_c, X_te_c


def _correct_location_scale(X_tr, X_te, bat_tr, bat_te):
    gm = X_tr.mean(axis=0)
    gs = X_tr.std(axis=0); gs = np.where(gs > 1e-12, gs, 1.0)
    params = {}
    X_tr_c = X_tr.copy()
    for b in np.unique(bat_tr):
        m = bat_tr == b
        bm = X_tr[m].mean(axis=0)
        bs = X_tr[m].std(axis=0); bs = np.where(bs > 1e-12, bs, 1.0)
        params[b] = (bm, bs)
        X_tr_c[m] = (X_tr[m] - bm) / bs * gs + gm
    X_te_c = X_te.copy()
    for b in np.unique(bat_te):
        bm, bs = params.get(b, (gm, np.ones_like(gm)))
        X_te_c[bat_te == b] = (X_te[bat_te == b] - bm) / bs * gs + gm
    return X_tr_c, X_te_c


def _qmap_np(src_vals: np.ndarray, src_sorted: np.ndarray,
             ref_sorted: np.ndarray) -> np.ndarray:
    """Non-parametric quantile mapping from src to ref distribution."""
    n_q, p   = src_vals.shape
    n_s      = src_sorted.shape[0]
    n_ref    = ref_sorted.shape[0]
    pos      = (src_sorted[:, None, :] <= src_vals[None, :, :]).sum(axis=0)
    quantiles = pos.astype(float) / n_s
    ref_idxs  = np.minimum((quantiles * n_ref).astype(int), n_ref - 1)
    col_idx   = np.arange(p)[np.newaxis, :]
    return ref_sorted[ref_idxs, col_idx]


def _correct_quantile_mapping(X_tr, X_te, bat_tr, bat_te):
    ref_mask = bat_tr == REF_BATCH
    if not ref_mask.any():
        return X_tr, X_te
    ref_sorted = np.sort(X_tr[ref_mask], axis=0)
    b_sorteds  = {}
    X_tr_c     = X_tr.copy()
    for b in np.unique(bat_tr):
        m = bat_tr == b
        bs = np.sort(X_tr[m], axis=0)
        b_sorteds[b] = bs
        if b == REF_BATCH:
            continue
        X_tr_c[m] = _qmap_np(X_tr[m], bs, ref_sorted)
    X_te_c = X_te.copy()
    for b in np.unique(bat_te):
        if b == REF_BATCH:
            continue
        m = bat_te == b
        bs = b_sorteds.get(b, np.sort(X_te[m], axis=0))
        X_te_c[m] = _qmap_np(X_te[m], bs, ref_sorted)
    return X_tr_c, X_te_c


def _pct_rank_np(vals: np.ndarray, ref_sorted: np.ndarray, n_b: int) -> np.ndarray:
    pos = (ref_sorted[:, None, :] <= vals[None, :, :]).sum(axis=0)
    return pos.astype(float) / n_b


def _correct_percentile_norm(X_tr, X_te, bat_tr, bat_te):
    b_sorteds = {}
    X_tr_c = np.empty_like(X_tr)
    for b in np.unique(bat_tr):
        m = bat_tr == b
        bs = np.sort(X_tr[m], axis=0)
        nb = m.sum()
        b_sorteds[b] = (bs, nb)
        X_tr_c[m] = _pct_rank_np(X_tr[m], bs, nb)
    X_te_c = np.empty_like(X_te)
    for b in np.unique(bat_te):
        m = bat_te == b
        if m.sum() == 0:
            continue
        bs, nb = b_sorteds.get(b, (np.sort(X_te[m], axis=0), m.sum()))
        X_te_c[m] = _pct_rank_np(X_te[m], bs, nb)
    return X_tr_c, X_te_c


CORRECT_FNS = {
    "none":             _correct_none,
    "mean_centering":   _correct_mean_centering,
    "location_scale":   _correct_location_scale,
    "quantile_mapping": _correct_quantile_mapping,
    "percentile_norm":  _correct_percentile_norm,
    "cohort_covariate": _correct_none,   # no transform; dummies appended in CV
}


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION  (simulation mode: variance top-50% then univariate top-100)
# Prevalence step is skipped — all features are always dense in CLR-Gaussian data.
# ════════════════════════════════════════════════════════════════════════════════
def select_features_sim(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Returns column indices of selected features."""
    vars_ = X_tr.var(axis=0)
    cut   = np.quantile(vars_, 1.0 - VARIANCE_KEEP_FRAC)
    hv    = np.where(vars_ >= cut)[0]
    if len(hv) == 0:
        return np.arange(min(TOP_N_UNIVARIATE, X_tr.shape[1]))

    X_hv = X_tr[:, hv]
    y_f  = y_tr.astype(float)
    ym   = y_f - y_f.mean()
    ys2  = float((ym ** 2).sum())
    if ys2 < 1e-24 or len(np.unique(y_tr)) < 2:
        return hv[:min(TOP_N_UNIVARIATE, len(hv))]

    Xm    = X_hv - X_hv.mean(axis=0)
    xs2   = (Xm ** 2).sum(axis=0)
    xs2   = np.where(xs2 < 1e-24, 1.0, xs2)
    abs_r = np.abs((ym @ Xm) / np.sqrt(ys2 * xs2))
    top_k = min(TOP_N_UNIVARIATE, len(hv))
    return hv[np.argsort(abs_r)[::-1][:top_k]]


# ════════════════════════════════════════════════════════════════════════════════
# 5-FOLD STRATIFIED CV ENGINE
# ════════════════════════════════════════════════════════════════════════════════
_ENET_PROTO = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5,
    class_weight="balanced", C=1.0, max_iter=3000, tol=1e-3,
)


def _make_dummies(bat: np.ndarray, n_cohorts: int) -> np.ndarray:
    """(n,) batch IDs → (n, n_cohorts-1) one-hot dummies (cohort 1 = reference)."""
    out = np.zeros((len(bat), n_cohorts - 1), dtype=float)
    for i in range(n_cohorts - 1):
        out[:, i] = (bat == i + 2).astype(float)
    return out


def run_cv(X: np.ndarray, y: np.ndarray, batch: np.ndarray,
           method: str, n_cohorts: int, cv_seed: int) -> float:
    """5-fold stratified CV. Returns mean AUC over folds."""
    is_single = (n_cohorts == 1)
    is_cov    = (method == "cohort_covariate") and not is_single
    correct   = CORRECT_FNS[method]

    skf  = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=cv_seed)
    aucs = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te   = X[tr_idx],     X[te_idx]
        y_tr, y_te   = y[tr_idx],     y[te_idx]
        bat_tr, bat_te = batch[tr_idx], batch[te_idx]

        # Batch correction (skipped for single-cohort or covariate method)
        if not is_single and not is_cov:
            X_tr_c, X_te_c = correct(X_tr, X_te, bat_tr, bat_te)
        else:
            X_tr_c, X_te_c = X_tr, X_te

        # Feature selection (label-independent step already done above)
        sel    = select_features_sim(X_tr_c, y_tr)
        X_tr_s = X_tr_c[:, sel]
        X_te_s = X_te_c[:, sel]

        # Cohort-as-covariate: append dummies after feature selection
        if is_cov:
            X_tr_s = np.hstack([X_tr_s, _make_dummies(bat_tr, n_cohorts)])
            X_te_s = np.hstack([X_te_s, _make_dummies(bat_te, n_cohorts)])

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            aucs.append(0.5)
            continue

        model = clone(_ENET_PROTO)
        model.fit(X_tr_s, y_tr)
        try:
            r_idx = list(model.classes_).index(1)
            probs = model.predict_proba(X_te_s)[:, r_idx]
            aucs.append(float(roc_auc_score(y_te, probs)))
        except (ValueError, IndexError):
            aucs.append(0.5)

    return float(np.mean(aucs)) if aucs else 0.5


# ════════════════════════════════════════════════════════════════════════════════
# CALIBRATION — binary search on δ to hit target PERMANOVA R²
# Signal calibration: n=118, n_cohorts=1 (as specified).
# Batch calibration:  n_per_cohort=40, n_cohorts=3 (matches real data).
# ════════════════════════════════════════════════════════════════════════════════
def _mean_r2_calib(delta: float, mode: str,
                   cal_n_per: int, cal_n_coh: int,
                   cal_rng: np.random.Generator) -> float:
    r2s = []
    for _ in range(N_CALIB_REPS):
        if mode == "signal":
            X, y, _     = simulate_dataset(delta, 0.0, cal_n_per, cal_n_coh, cal_rng)
            r2s.append(permanova_r2(X, y))
        else:
            X, _, batch = simulate_dataset(0.0, delta, cal_n_per, cal_n_coh, cal_rng)
            r2s.append(permanova_r2(X, batch))
    return float(np.mean(r2s))


def calibrate_delta(target_r2: float, mode: str,
                    cal_n_per: int, cal_n_coh: int) -> float:
    if target_r2 <= 0.0:
        return 0.0
    seed_offset = int(target_r2 * 10000) + (0 if mode == "signal" else 1_000_000)
    cal_rng = np.random.default_rng(SEED + seed_offset)

    lo, hi = 0.0, 10.0
    while _mean_r2_calib(hi, mode, cal_n_per, cal_n_coh, cal_rng) < target_r2:
        hi *= 2.0
        if hi > 500:
            break

    for _ in range(CALIB_BISECT_ITER):
        mid    = (lo + hi) / 2.0
        r2_mid = _mean_r2_calib(mid, mode, cal_n_per, cal_n_coh, cal_rng)
        if r2_mid < target_r2:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def calibrate_all(force: bool = False) -> tuple:
    """Load cached calibration or run fresh. Returns (signal_deltas, batch_deltas)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calib_path = OUT_DIR / "calibration_deltas.tsv"

    if calib_path.exists() and not force:
        print(f"\nLoading cached calibration from {calib_path}")
        df = pd.read_csv(calib_path, sep="\t")
        sig = dict(zip(df.loc[df["mode"] == "signal", "target_r2"],
                       df.loc[df["mode"] == "signal", "delta"]))
        bat = dict(zip(df.loc[df["mode"] == "batch",  "target_r2"],
                       df.loc[df["mode"] == "batch",  "delta"]))
        for row in df.itertuples(index=False):
            print(f"  [{row.mode:<6}] target={row.target_r2:.3f}  "
                  f"δ={row.delta:.4f}  verified_R²={row.verified_r2:.4f}")
        return sig, bat

    print("\nCalibrating signal effect sizes  (n=118, 1 cohort) …", flush=True)
    rows, sig, bat = [], {}, {}

    for target in SIGNAL_F2_TARGETS:
        t0    = time.time()
        delta = calibrate_delta(target, "signal", 118, 1)
        vr2   = _mean_r2_calib(delta, "signal", 118, 1,
                                np.random.default_rng(SEED + 77777))
        print(f"  target={target:.3f}  δ={delta:.4f}  "
              f"verified_R²={vr2:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        sig[target] = delta
        rows.append({"mode": "signal", "target_r2": target,
                     "delta": delta, "verified_r2": vr2})

    print("\nCalibrating batch effect sizes  (n_per=40, 3 cohorts) …", flush=True)
    for target in BATCH_F2_TARGETS:
        if target == 0.0:
            bat[target] = 0.0
            rows.append({"mode": "batch", "target_r2": 0.0,
                         "delta": 0.0, "verified_r2": 0.0})
            print("  target=0.000  δ=0.0000  (trivial)")
            continue
        t0    = time.time()
        delta = calibrate_delta(target, "batch", 40, 3)
        vr2   = _mean_r2_calib(delta, "batch", 40, 3,
                                np.random.default_rng(SEED + 88888))
        print(f"  target={target:.3f}  δ={delta:.4f}  "
              f"verified_R²={vr2:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        bat[target] = delta
        rows.append({"mode": "batch", "target_r2": target,
                     "delta": delta, "verified_r2": vr2})

    pd.DataFrame(rows).to_csv(calib_path, sep="\t", index=False)
    print(f"\nCalibration saved → {calib_path}")
    return sig, bat


# ════════════════════════════════════════════════════════════════════════════════
# GRID CELL RUNNER
# ════════════════════════════════════════════════════════════════════════════════
def run_grid_cell(sig_f2: float, bat_f2: float,
                  n_per_cohort: int, n_cohorts: int,
                  method: str, n_reps: int,
                  delta_sig: float, delta_batch: float,
                  cell_seed: int) -> dict:
    """
    Run n_reps × 5-fold CV replicates for one grid cell.
    PERMANOVA R² diagnostics (no permutation) computed from rep 0.
    """
    cell_rng = np.random.default_rng(cell_seed)
    rep_aucs = []
    diag_r2_resp = diag_r2_batch = None

    for rep in range(n_reps):
        X, y, batch = simulate_dataset(delta_sig, delta_batch,
                                        n_per_cohort, n_cohorts, cell_rng)
        if rep == 0:
            diag_r2_resp  = permanova_r2(X, y)
            diag_r2_batch = permanova_r2(X, batch) if n_cohorts > 1 else 0.0

        cv_seed = int(cell_rng.integers(2 ** 31))
        rep_aucs.append(run_cv(X, y, batch, method, n_cohorts, cv_seed))

    return {
        "sig_f2":        sig_f2,
        "batch_f2":      bat_f2,
        "n_per_cohort":  n_per_cohort,
        "n_cohorts":     n_cohorts,
        "n_total":       n_per_cohort * n_cohorts,
        "method":        method,
        "n_reps":        n_reps,
        "mean_auc":      float(np.mean(rep_aucs)),
        "std_auc":       float(np.std(rep_aucs)),
        "r2_resp_diag":  round(float(diag_r2_resp), 5)  if diag_r2_resp  is not None else None,
        "r2_batch_diag": round(float(diag_r2_batch), 5) if diag_r2_batch is not None else None,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════════
def make_delta_auc_heatmaps(df: pd.DataFrame) -> None:
    """
    Per-method heatmap of Δ AUC = mean_auc − ceiling_auc over
    (signal_f2 × batch_f2), faceted by (n_per_cohort × n_cohorts).

    Real-data operating points marked per (npc=40) subplot:
      nc=1: sig≈0.027, bat=0.000  (Frankel n=39, 1 cohort)
      nc=2: sig≈0.027, bat≈0.080  (n=79, 2 cohorts, intrinsic signal)
      nc=3: sig≈0.027, bat≈0.080  (n=118 intrinsic est.)  — star
            sig≈0.007, bat≈0.080  (n=118 measured, ≈null) — circle

    The sig_f2=0.007 row is the NULL MODEL (δ=0 from calibration) — annotated.
    """
    sig_vals  = sorted(df["sig_f2"].unique())
    bat_vals  = sorted(df["batch_f2"].unique())
    npc_vals  = sorted(df["n_per_cohort"].unique())
    ncoh_vals = sorted(df["n_cohorts"].unique())
    sig_desc  = sorted(sig_vals, reverse=True)   # high→low (top→bottom in plot)
    bat_asc   = sorted(bat_vals)                 # low→high (left→right)
    vmin, vmax = -0.30, 0.10

    # Row index of the null signal level (sig=0.007) in the heatmap (0 = top)
    null_row = int(np.argmin([abs(s - 0.007) for s in sig_desc]))

    # Real-data markers per (n_per_cohort, n_cohorts):
    # list of (target_sig_f2, target_bat_f2, marker, color, zorder, label)
    REAL_MARKS = {
        (40, 1): [
            (0.027, 0.000, "*", "black",      12, "Real: 1 coh, n=39\n(sig≈0.027, bat=0)"),
        ],
        (40, 2): [
            (0.027, 0.080, "^", "dodgerblue", 12, "Real: 2 coh, n=79\n(sig≈0.027, bat≈0.08)"),
        ],
        (40, 3): [
            (0.027, 0.080, "*", "dodgerblue", 12, "3 coh, intrinsic sig≈0.027"),
            (0.007, 0.080, "o", "crimson",    11, "3 coh, meas. sig≈0.007 (≈null)"),
        ],
    }

    for method in METHODS:
        dfm  = df[df["method"] == method]
        nrow = len(npc_vals)
        ncol = len(ncoh_vals)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(3.8 * ncol, 3.2 * nrow),
                                 squeeze=False)
        ims = []

        for ri, npc in enumerate(npc_vals):
            for ci, nc in enumerate(ncoh_vals):
                ax  = axes[ri][ci]
                sub = dfm[(dfm["n_per_cohort"] == npc) & (dfm["n_cohorts"] == nc)]

                if sub.empty or sub["delta_auc"].isna().all():
                    ax.axis("off")
                    continue

                piv = (sub.pivot(index="sig_f2", columns="batch_f2", values="delta_auc")
                          .reindex(index=sig_desc, columns=bat_asc))

                im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn",
                               vmin=vmin, vmax=vmax, interpolation="nearest")
                ims.append(im)

                # Axis labels
                ax.set_xticks(range(len(bat_asc)))
                ax.set_xticklabels([f"{v:.3f}" for v in bat_asc],
                                   fontsize=7, rotation=45)
                # Y-tick labels: annotate null row with "(δ=0, null)"
                ylabels = []
                for s in sig_desc:
                    lbl = f"{s:.3f}"
                    if abs(s - 0.007) < 1e-6:
                        lbl += "\n(δ=0,null)"
                    ylabels.append(lbl)
                ax.set_yticks(range(len(sig_desc)))
                ax.set_yticklabels(ylabels, fontsize=6)

                ax.set_title(f"n/cohort={npc}, cohorts={nc}", fontsize=8, pad=3)
                if ri == nrow - 1:
                    ax.set_xlabel("Batch R²", fontsize=8)
                if ci == 0:
                    ax.set_ylabel("Signal R²", fontsize=8)

                # Subtle grey box around the null row
                from matplotlib.patches import FancyBboxPatch
                n_bat = len(bat_asc)
                rect = plt.Rectangle(
                    (-0.5, null_row - 0.5), n_bat, 1.0,
                    linewidth=0.8, edgecolor="dimgray", facecolor="none",
                    linestyle=":", zorder=5,
                )
                ax.add_patch(rect)

                # Real-data markers
                marks = REAL_MARKS.get((npc, nc), [])
                handles = []
                for sig_t, bat_t, mkr, col, zo, lbl in marks:
                    r_i = int(np.argmin([abs(s - sig_t) for s in sig_desc]))
                    c_i = int(np.argmin([abs(b - bat_t) for b in bat_asc]))
                    h = ax.plot(c_i, r_i, marker=mkr, color=col,
                                markersize=9, markeredgecolor="white",
                                markeredgewidth=0.6, zorder=zo, label=lbl)[0]
                    handles.append(h)

                if handles:
                    ax.legend(handles=handles, fontsize=4.5,
                              loc="lower right", framealpha=0.85,
                              borderpad=0.3, handlelength=1.0)

        fig.suptitle(
            f"Method: {method}  |  Δ AUC = mean AUC − ceiling AUC\n"
            f"(ceiling = no-batch, n_cohorts=1, same n/cohort, method=none)\n"
            f"sig_f2=0.007 row = NULL MODEL (δ=0); dotted box highlights it.",
            fontsize=8, y=1.02,
        )
        if ims:
            plt.colorbar(ims[-1], ax=axes.ravel().tolist(),
                         shrink=0.45, label="Δ AUC", pad=0.04)
        plt.tight_layout()
        out = OUT_DIR / f"heatmap_delta_auc_{method}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def make_auc_line_plots(df: pd.DataFrame) -> None:
    """
    AUC vs signal_f2 line plots, colored by batch_f2, faceted by n_cohorts.
    One figure per n_per_cohort.
    """
    npc_vals  = sorted(df["n_per_cohort"].unique())
    ncoh_vals = sorted(df["n_cohorts"].unique())
    bat_vals  = sorted(df["batch_f2"].unique())
    sig_vals  = sorted(df["sig_f2"].unique())
    colors    = plt.cm.Reds(np.linspace(0.3, 0.9, len(bat_vals)))

    for npc in npc_vals:
        dfn  = df[df["n_per_cohort"] == npc]
        nrow = len(ncoh_vals)
        ncol = len(METHODS)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(2.4 * ncol, 2.4 * nrow),
                                 squeeze=False, sharey=True)

        for ri, nc in enumerate(ncoh_vals):
            dfnc = dfn[dfn["n_cohorts"] == nc]
            for ci, method in enumerate(METHODS):
                ax  = axes[ri][ci]
                dfm = dfnc[dfnc["method"] == method].sort_values("sig_f2")
                for bi, bat in enumerate(bat_vals):
                    sub = dfm[dfm["batch_f2"] == bat]
                    if sub.empty:
                        continue
                    ax.plot(sub["sig_f2"], sub["mean_auc"],
                            color=colors[bi], marker="o", markersize=3,
                            linewidth=1.0, label=f"{bat:.3f}")
                ax.axhline(0.5, ls="--", c="gray", lw=0.7)
                ax.set_ylim(0.35, 0.85)
                ax.set_xticks(sig_vals)
                ax.set_xticklabels([f"{v:.3f}" for v in sig_vals],
                                   fontsize=5, rotation=45)
                ax.tick_params(axis="y", labelsize=6)
                if ri == 0:
                    ax.set_title(method.replace("_", "\n"), fontsize=7)
                if ri == nrow - 1:
                    ax.set_xlabel("Signal R²", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"coh={nc}\nMean AUC", fontsize=7)

        handles = [
            plt.Line2D([0], [0], color=colors[bi], marker="o",
                       markersize=4, label=f"batch R²={b:.3f}")
            for bi, b in enumerate(bat_vals)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(bat_vals),
                   fontsize=7, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"n_per_cohort={npc}: AUC by (signal R², batch R², method, n_cohorts)",
                     fontsize=9)
        plt.tight_layout()
        out = OUT_DIR / f"auc_lines_nper{npc}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def make_method_comparison_at_real_data(df: pd.DataFrame) -> None:
    """
    Bar chart: mean AUC for each method at the real-data operating point
    (sig_f2≈0.027, batch_f2≈0.08, n_per=40, n_cohorts=3).
    """
    sig_closest  = min(SIGNAL_F2_TARGETS, key=lambda s: abs(s - REAL_SIG_F2_N39))
    bat_closest  = min(BATCH_F2_TARGETS,  key=lambda b: abs(b - REAL_BATCH_F2))
    ceiling_auc  = (df[(df["sig_f2"] == sig_closest) & (df["batch_f2"] == 0.0) &
                       (df["n_cohorts"] == 1) & (df["n_per_cohort"] == REAL_N_PER) &
                       (df["method"] == "none")]["mean_auc"])
    ceil_val     = float(ceiling_auc.mean()) if not ceiling_auc.empty else np.nan

    sub = df[(df["sig_f2"] == sig_closest) & (df["batch_f2"] == bat_closest) &
             (df["n_per_cohort"] == REAL_N_PER) & (df["n_cohorts"] == REAL_N_COH)]
    if sub.empty:
        return

    sub = sub.set_index("method").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors  = ["#4dac26" if v >= 0.5 else "#d01c8b" for v in sub["mean_auc"]]
    bars    = ax.bar(METHODS, sub["mean_auc"], color=colors, alpha=0.8,
                     yerr=sub["std_auc"], capsize=4)
    ax.axhline(0.5,      ls="--", c="gray", lw=1.0, label="Chance (0.5)")
    if not np.isnan(ceil_val):
        ax.axhline(ceil_val, ls=":",  c="steelblue", lw=1.5,
                   label=f"Ceiling (no-batch, n_per={REAL_N_PER}): {ceil_val:.3f}")
    ax.set_ylabel("Mean AUC (5-fold CV, 10 reps)")
    ax.set_xlabel("Batch correction method")
    ax.set_title(
        f"Real-data operating point: signal R²≈{sig_closest:.3f}, "
        f"batch R²≈{bat_closest:.3f},\nn_per_cohort={REAL_N_PER}, n_cohorts={REAL_N_COH}"
    )
    ax.set_ylim(0.3, min(0.9, max(sub["mean_auc"].max() + 0.1, 0.7)))
    ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "method_comparison_real_data_point.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 simulation study for batch correction methods."
    )
    parser.add_argument("--mode", choices=["timing", "calibrate", "full"],
                        default="timing",
                        help="timing: calibrate + time 1 cell + report (DEFAULT); "
                             "calibrate: run calibration only; "
                             "full: run complete grid sweep.")
    parser.add_argument("--n-reps", type=int, default=N_REPS_DEFAULT,
                        help=f"Replicates per grid cell (default {N_REPS_DEFAULT})")
    parser.add_argument("--max-cohorts", type=int, default=4,
                        help="Maximum n_cohorts value in grid (default 4)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for joblib (default -1 = all cores)")
    parser.add_argument("--force-calibrate", action="store_true",
                        help="Re-run calibration even if cache exists")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Calibration ────────────────────────────────────────────────────────────
    signal_deltas, batch_deltas = calibrate_all(force=args.force_calibrate)

    if args.mode == "calibrate":
        return

    # ── Timing estimate ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TIMING — single grid cell closest to real data")
    print("=" * 65)

    _test_sig  = 0.027
    _test_bat  = 0.080
    _test_npc  = 40
    _test_nc   = 3
    _test_meth = "location_scale"    # medium-complexity method

    t0 = time.time()
    run_grid_cell(
        sig_f2=_test_sig, bat_f2=_test_bat,
        n_per_cohort=_test_npc, n_cohorts=_test_nc,
        method=_test_meth, n_reps=args.n_reps,
        delta_sig=signal_deltas[_test_sig],
        delta_batch=batch_deltas[_test_bat],
        cell_seed=SEED + 1,
    )
    t_cell = time.time() - t0

    n_coh_vals   = [c for c in N_COHORTS_GRID if c <= args.max_cohorts]
    n_cells_full = (len(SIGNAL_F2_TARGETS) * len(BATCH_F2_TARGETS) *
                    len(N_PER_COHORT_GRID) * len(n_coh_vals) * len(METHODS))
    n_cores      = min(os.cpu_count() or 1, 8) if args.n_jobs == -1 else abs(args.n_jobs)
    n_cores      = max(n_cores, 1)

    t_ser_min = t_cell * n_cells_full / 60
    t_par_min = t_ser_min / n_cores

    n_coh_red   = [c for c in N_COHORTS_GRID if c <= 3]
    n_cells_red = (len(SIGNAL_F2_TARGETS) * len(BATCH_F2_TARGETS) *
                   len(N_PER_COHORT_GRID) * len(n_coh_red) * len(METHODS))
    t_red_par   = (t_cell * 5 / args.n_reps) * n_cells_red / n_cores / 60

    print(f"\n  Timed cell: method={_test_meth}, "
          f"n={_test_npc * _test_nc}, {args.n_reps} reps × {N_CV_FOLDS} folds")
    print(f"  Single cell time: {t_cell:.1f}s\n")
    print(f"  Full grid:")
    print(f"    {len(SIGNAL_F2_TARGETS)} sig × {len(BATCH_F2_TARGETS)} bat × "
          f"{len(N_PER_COHORT_GRID)} n_per × {len(n_coh_vals)} n_coh × "
          f"{len(METHODS)} methods = {n_cells_full:,} cells")
    print(f"    Estimated serial:              {t_ser_min:.1f} min")
    print(f"    Estimated parallel ({n_cores} cores): {t_par_min:.1f} min")
    print(f"\n  PROPOSED REDUCED GRID (--n-reps 5 --max-cohorts 3):")
    print(f"    {n_cells_red:,} cells × 5 reps ≈ {t_red_par:.1f} min "
          f"parallel ({n_cores} cores)")
    print(f"\n  Run '--mode full' to launch.  "
          f"Add '--n-reps 5 --max-cohorts 3' for reduced grid.")

    if args.mode == "timing":
        return

    # ── Full grid sweep ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"GRID SWEEP  (n_reps={args.n_reps}, max_cohorts={args.max_cohorts})")
    print("=" * 65)

    grid = list(itertools.product(
        SIGNAL_F2_TARGETS, BATCH_F2_TARGETS,
        N_PER_COHORT_GRID, n_coh_vals, METHODS,
    ))
    cell_seeds = [SEED + 100 + i for i in range(len(grid))]

    print(f"  {len(grid):,} cells, {args.n_reps} reps, {N_CV_FOLDS} folds", flush=True)
    t_sweep = time.time()

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_grid_cell)(
                sig_f2=sf, bat_f2=bf, n_per_cohort=npc, n_cohorts=nc,
                method=m, n_reps=args.n_reps,
                delta_sig=signal_deltas[sf], delta_batch=batch_deltas[bf],
                cell_seed=seed,
            )
            for (sf, bf, npc, nc, m), seed in zip(grid, cell_seeds)
        )
    except ImportError:
        print("  joblib not available — running serially …")
        results = []
        for i, ((sf, bf, npc, nc, m), seed) in enumerate(zip(grid, cell_seeds)):
            if i % 100 == 0:
                elapsed = time.time() - t_sweep
                eta     = elapsed / max(i, 1) * (len(grid) - i) / 60
                print(f"  {i+1}/{len(grid)}  "
                      f"elapsed={elapsed/60:.1f}m  ETA={eta:.1f}m", flush=True)
            results.append(run_grid_cell(
                sig_f2=sf, bat_f2=bf, n_per_cohort=npc, n_cohorts=nc,
                method=m, n_reps=args.n_reps,
                delta_sig=signal_deltas[sf], delta_batch=batch_deltas[bf],
                cell_seed=seed,
            ))

    print(f"\n  Sweep complete in {(time.time()-t_sweep)/60:.1f} min")

    df = pd.DataFrame(results)

    # Ceiling AUC: method="none", n_cohorts=1, bat_f2=0, same (sig_f2, n_per_cohort)
    ceiling_map = (
        df[(df["batch_f2"] == 0.0) & (df["n_cohorts"] == 1) & (df["method"] == "none")]
        .groupby(["sig_f2", "n_per_cohort"])["mean_auc"].mean()
        .to_dict()
    )
    df["ceiling_auc"] = df.apply(
        lambda r: ceiling_map.get((r["sig_f2"], r["n_per_cohort"]), np.nan), axis=1
    )
    df["delta_auc"] = df["mean_auc"] - df["ceiling_auc"]

    # Save full results
    out_tsv = OUT_DIR / "grid_results.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved: {out_tsv}  ({len(df):,} rows × {len(df.columns)} cols)")

    # Summary per method
    summary_rows = []
    for method in METHODS:
        dfm = df[df["method"] == method]
        summary_rows.append({
            "method":         method,
            "mean_delta_auc": round(dfm["delta_auc"].mean(), 4),
            "std_delta_auc":  round(dfm["delta_auc"].std(),  4),
            "min_delta_auc":  round(dfm["delta_auc"].min(),  4),
            "max_delta_auc":  round(dfm["delta_auc"].max(),  4),
            "frac_helps":     round((dfm["delta_auc"] >  0.02).mean(), 3),
            "frac_hurts":     round((dfm["delta_auc"] < -0.02).mean(), 3),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_tsv = OUT_DIR / "grid_method_summary.tsv"
    sum_df.to_csv(sum_tsv, sep="\t", index=False)
    print(f"  Saved: {sum_tsv}")
    print("\n" + sum_df.to_string(index=False))

    # Plots
    print("\nGenerating figures …")
    make_delta_auc_heatmaps(df)
    make_auc_line_plots(df)
    make_method_comparison_at_real_data(df)

    print("\nAll outputs:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
