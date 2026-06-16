import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import os
os.makedirs("results/ml/tumor/batch", exist_ok=True)

df = pd.read_csv("results/ml/tumor/combined_tumor_features.tsv", sep="\t")
df = df.dropna(subset=["response", "TMB"])

feature_cols = ["TMB", "n_mutations"] + [c for c in df.columns
                                          if c.startswith("mut_")]
X = df[feature_cols].fillna(0).values.astype(float)
y = (df["response"] == "R").astype(int).values
studies = df["study"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def permanova(X, groups, n_perm=999, seed=42):
    """PERMANOVA on Euclidean distance matrix."""
    rng = np.random.RandomState(seed)
    dist = pairwise_distances(X, metric="euclidean")
    n = len(groups)

    def f_stat(dist, groups):
        n = len(groups)
        SS_total = (dist ** 2).sum() / (2 * n)
        SS_within = 0
        for g in np.unique(groups):
            mask = groups == g
            ng = mask.sum()
            if ng < 2:
                continue
            SS_within += (dist[np.ix_(mask, mask)] ** 2).sum() / (2 * ng)
        SS_between = SS_total - SS_within
        k = len(np.unique(groups))
        F = (SS_between / (k - 1)) / (SS_within / (n - k))
        R2 = SS_between / SS_total
        return F, R2

    obs_F, obs_R2 = f_stat(dist, groups)
    null_F = [f_stat(dist, rng.permutation(groups))[0] for _ in range(n_perm)]
    p = np.mean(np.array(null_F) >= obs_F)
    return obs_R2, p, obs_F

print("Running PERMANOVA — batch (study) effect …", flush=True)
r2_batch, p_batch, f_batch = permanova(X_scaled, studies)
print(f"  Batch (study): R2={r2_batch:.4f}, p={p_batch:.4f}, F={f_batch:.3f}", flush=True)

print("Running PERMANOVA — response effect …", flush=True)
r2_resp, p_resp, f_resp = permanova(X_scaled, y.astype(str))
print(f"  Response:      R2={r2_resp:.4f}, p={p_resp:.4f}, F={f_resp:.3f}", flush=True)

ratio = r2_batch / r2_resp if r2_resp > 0 else float("inf")
print(f"  Batch:Signal ratio: {ratio:.1f}x", flush=True)

print("\nTMB distribution by study:", flush=True)
for s in np.unique(studies):
    tmb = df[df.study == s]["TMB"]
    print(f"  {s}: median={tmb.median():.2f}, mean={tmb.mean():.2f}, "
          f"range=[{tmb.min():.1f}, {tmb.max():.1f}], n={len(tmb)}", flush=True)

print("\nTop gene frequency by study (batch effect check):", flush=True)
for gene in ["mut_BRAF", "mut_NRAS", "mut_NF1", "mut_TP53", "mut_PTEN"]:
    if gene not in df.columns:
        continue
    freqs = df.groupby("study")[gene].mean()
    print(f"  {gene}: " + ", ".join(f"{s}={v:.2f}" for s, v in freqs.items()),
          flush=True)

summary = pd.DataFrame({
    "modality":          ["tumor_genomics"],
    "r2_batch":          [r2_batch],
    "p_batch":           [p_batch],
    "r2_response":       [r2_resp],
    "p_response":        [p_resp],
    "batch_signal_ratio":[ratio],
})
summary.to_csv("results/ml/tumor/batch/permanova_summary.tsv", sep="\t", index=False)
print("\nDONE", flush=True)
