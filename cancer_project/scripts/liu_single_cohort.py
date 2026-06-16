import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings, time
warnings.filterwarnings("ignore")
import os
os.makedirs("results/ml/tumor/single_cohort", exist_ok=True)

def make_feature_cols(df):
    return ["TMB", "n_mutations"] + [c for c in df.columns if c.startswith("mut_")]

def make_models(y):
    sw = (1 - y).sum() / y.sum()
    return {
        "ElasticNet": LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, C=0.1,
            class_weight="balanced", max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=sw, random_state=42, eval_metric="auc", verbosity=0),
    }

def run_loocv(X, y, models):
    results = {}
    for name, model in models.items():
        t0 = time.time()
        preds = np.zeros(len(y))
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            sc = StandardScaler()
            model.fit(sc.fit_transform(X[mask]), y[mask])
            preds[i] = model.predict_proba(sc.transform(X[[i]]))[0][1]
        auc = roc_auc_score(y, preds)
        results[name] = (auc, preds)
        print(f"  {name}: AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return results


# ── Liu 2019 ───────────────────────────────────────────────────────────────
print("=== LIU 2019 (n=144) ===", flush=True)
liu = pd.read_csv("results/ml/tumor/liu2019_features.tsv", sep="\t")
liu = liu.dropna(subset=["response", "TMB"])
feat_liu = make_feature_cols(liu)
X_liu = liu[feat_liu].fillna(0).values.astype(float)
y_liu = (liu["response"] == "R").astype(int).values
print(f"  {len(liu)} patients  R={y_liu.sum()}  NR={(1-y_liu).sum()}", flush=True)

print("\n--- LOOCV Liu 2019 ---", flush=True)
models_liu = make_models(y_liu)
loocv_liu = run_loocv(X_liu, y_liu, models_liu)

best_name = max(loocv_liu, key=lambda k: loocv_liu[k][0])
best_auc  = loocv_liu[best_name][0]

# ── Permutation test on best model (Liu 2019) ─────────────────────────────
print(f"\n--- Permutation test ({best_name}, N=500) ---", flush=True)
best_model = models_liu[best_name]
null_aucs = []
rng = np.random.RandomState(42)
t_perm = time.time()
for perm in range(500):
    y_perm = rng.permutation(y_liu)
    pp = np.zeros(len(y_liu))
    for i in range(len(y_liu)):
        mask = np.arange(len(y_liu)) != i
        sc = StandardScaler()
        best_model.fit(sc.fit_transform(X_liu[mask]), y_perm[mask])
        pp[i] = best_model.predict_proba(sc.transform(X_liu[[i]]))[0][1]
    null_aucs.append(roc_auc_score(y_perm, pp))
    if (perm + 1) % 50 == 0:
        eta = (time.time() - t_perm) / (perm + 1) * (500 - perm - 1)
        print(f"  {perm+1}/500  null_mean={np.mean(null_aucs):.4f}  "
              f"p_so_far={np.mean(np.array(null_aucs) >= best_auc):.3f}  "
              f"ETA={eta/60:.1f}min", flush=True)

p_val = np.mean(np.array(null_aucs) >= best_auc)
print(f"\nObserved AUC ({best_name}): {best_auc:.4f}", flush=True)
print(f"Null mean ± std: {np.mean(null_aucs):.4f} ± {np.std(null_aucs):.4f}", flush=True)
print(f"Permutation p-value: {p_val:.4f}  (N=500)", flush=True)

# ── XGBoost feature importance (full-data fit) ────────────────────────────
print("\n--- Top features (XGBoost, full Liu 2019) ---", flush=True)
xgb_full = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    scale_pos_weight=(1 - y_liu).sum() / y_liu.sum(),
    random_state=42, eval_metric="auc", verbosity=0)
sc_full = StandardScaler()
xgb_full.fit(sc_full.fit_transform(X_liu), y_liu)
imp = pd.Series(xgb_full.feature_importances_, index=feat_liu).nlargest(10)
for gene, val in imp.items():
    print(f"  {gene}: {val:.4f}", flush=True)

# ── Riaz 2017 and Hugo 2016 single-cohort LOOCV ───────────────────────────
for label, fname in [("Riaz 2017", "riaz2017_features.tsv"),
                     ("Hugo 2016", "hugo2016_features.tsv")]:
    print(f"\n=== {label} ===", flush=True)
    dfc = pd.read_csv(f"results/ml/tumor/{fname}", sep="\t")
    dfc = dfc.dropna(subset=["response", "TMB"])
    feat_c = make_feature_cols(dfc)
    Xc = dfc[feat_c].fillna(0).values.astype(float)
    yc = (dfc["response"] == "R").astype(int).values
    print(f"  {len(dfc)} patients  R={yc.sum()}  NR={(1-yc).sum()}", flush=True)
    print("--- LOOCV ---", flush=True)
    run_loocv(Xc, yc, make_models(yc))

# ── Save ──────────────────────────────────────────────────────────────────
pd.DataFrame({"null_auc": null_aucs}).to_csv(
    "results/ml/tumor/single_cohort/liu2019_null_aucs.tsv", sep="\t", index=False)
pd.DataFrame({
    "model": [best_name], "observed_auc": [best_auc],
    "null_mean": [np.mean(null_aucs)], "null_std": [np.std(null_aucs)],
    "p_value": [p_val],
}).to_csv("results/ml/tumor/single_cohort/liu2019_permutation.tsv",
          sep="\t", index=False)

print("\nDONE", flush=True)
