# Phase 1: Batch Correction Method Shootout — Interpretation

## Environment note

R and rpy2 are not available in this environment. ConQuR and MMUPHin are
implemented as Python approximations:
- **quantile_mapping** = ConQuR approx: per-feature empirical CDF transfer
  from each non-reference batch to the Cohort 1 (Frankel/HiSeq) training
  distribution (non-parametric; does not include the covariate-adjustment
  step that distinguishes true ConQuR from simple quantile normalisation).
- **location_scale** = MMUPHin approx: per-feature per-batch mean-and-std
  normalisation (standardise per batch, rescale to global mean/std). True
  MMUPHin uses a linear mixed model that accounts for covariates such as the
  response variable during batch estimation — the approximation omits this.
  These results should be verified with the real R packages when available.

## PERMANOVA diagnostics (globally corrected matrix, 999 perms)

| Method               | Response R² | p    | Batch R²  | p      |
|----------------------|-------------|------|-----------|--------|
| uncorrected_baseline |      0.0068 | 0.847 |    0.0768 |  0.001 |
| mean_centering       |      0.0060 | 0.967 |   -0.0000 |  1.000 |
| location_scale       |      0.0060 | 0.955 |    0.0000 |  1.000 |
| quantile_mapping     |      0.0057 | 0.977 |    0.0012 |  1.000 |
| percentile_norm      |      0.0013 | 0.744 |    0.0000 |  1.000 |
| cohort_covariate     |      0.0068 | 0.847 |    0.0768 |  0.000 |

## LOOCV AUC and permutation test (N=100 label shuffles)

| Method               | ENet AUC | p(ENet) | RF AUC | p(RF)  | Leakage |
|----------------------|----------|---------|--------|--------|---------|
| uncorrected_baseline |   0.3511 |     N/A | 0.5421 |  0.250 | none |
| mean_centering       |   0.3294 |   0.960 | 0.5321 |  0.240 | none |
| location_scale       |   0.3634 |   0.980 | 0.5127 |  0.240 | none |
| quantile_mapping     |   0.3466 |   1.000 | 0.4193 |  0.800 | none |
| percentile_norm      |   0.4193 |   0.850 | 0.4647 |  0.710 | none |
| cohort_covariate     |   0.3537 |   0.980 | 0.3439 |  0.940 | none |

## 1. Batch R² reduction

Uncorrected baseline: batch R² = 0.0768 (p=0.001).

  - mean_centering: batch R² = -0.0000  (Δ=-0.0768, p=1.000)
  - location_scale: batch R² = 0.0000  (Δ=-0.0768, p=1.000)
  - quantile_mapping: batch R² = 0.0012  (Δ=-0.0756, p=1.000)
  - percentile_norm: batch R² = 0.0000  (Δ=-0.0768, p=1.000)
  - cohort_covariate: batch R² = 0.0768  (Δ=-0.0000, p=0.000)

## 2. Response signal preservation

Uncorrected baseline: response R² = 0.0068 (p=0.847).

  - mean_centering: response R² = 0.0060  (Δ=-0.0008, p=0.967)
  - location_scale: response R² = 0.0060  (Δ=-0.0008, p=0.955)
  - quantile_mapping: response R² = 0.0057  (Δ=-0.0011, p=0.977)
  - percentile_norm: response R² = 0.0013  (Δ=-0.0055, p=0.744)
  - cohort_covariate: response R² = 0.0068  (Δ=-0.0000, p=0.847)

## 3. Predictive performance vs. ComBat reference

ComBat reference (mean_centering): ENet AUC=0.3294  RF AUC=0.5321
Uncorrected baseline:              ENet AUC=0.3511  RF AUC=0.5421

  - location_scale: ENet Δ=+0.0340, RF Δ=-0.0194
  - quantile_mapping: ENet Δ=+0.0172, RF Δ=-0.1128
  - percentile_norm: ENet Δ=+0.0899, RF Δ=-0.0674
  - cohort_covariate: ENet Δ=+0.0243, RF Δ=-0.1882

**No method simultaneously reduces batch R² AND improves RF AUC over the uncorrected baseline.** This is consistent with the Phase 0 finding that batch variance is 11× response variance — corrections that remove the batch signal may also compress or invert the response signal.

## 4. PERMDISP hypothesis: do dispersion-aware methods outperform mean-centering?

Hypothesis (from Phase 0 PERMDISP): location+scale methods (MMUPHin approx) should outperform location-only mean-centering (ComBat) at reducing batch R²,
because the observed batch PERMDISP heterogeneity (F=9.59/5.63, p=0.005/0.006) implies batches differ in both centroid AND spread.

  Mean-centering batch R²:   -0.0000
  Location+scale batch R²:   0.0000  (contradicts hypothesis ✗)

The hypothesis is NOT SUPPORTED: location+scale does not further reduce batch R² vs. mean-centering. This may indicate that the CLR transformation already partially homogenises variance across cohorts, leaving only the centroid shift as the dominant batch structure in this feature space.

## 5. Leakage check

For reference: global (non-per-fold) ComBat produced RF AUC ≈ 0.99 (confirmed leakage artifact; discarded). All methods here use per-fold correction.

  - mean_centering: none
  - location_scale: none
  - quantile_mapping: none
  - percentile_norm: none
  - cohort_covariate: none

## 6. Summary for paper

The Phase 1 correction-method comparison reveals that no batch correction method, as implemented here (Python approximations for ConQuR/MMUPHin), simultaneously reduces batch R² to near zero, preserves response R², and improves LOOCV AUC above the uncorrected baseline. This negative result is itself informative: it suggests that with the observed batch-to-signal ratio (11×) and current cohort sizes (39/40/39), batch correction is as likely to remove biological signal as to remove technical noise — a structural underpowering problem that motivates the Phase 2 simulation study.

Methodological note: the Python approximations for ConQuR (ECDF quantile transfer) and MMUPHin (per-feature standardisation) omit the covariate-adjustment step that protects biological signal in the true R packages. These results may underestimate the true performance of ConQuR/MMUPHin; replication with R is recommended when possible.

---
_Generated by scripts/phase1_correction_shootout.py_
