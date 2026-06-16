# Phase 4a: Meta-Analytic Effect Combination

Random-effects meta-analysis (DerSimonian-Laird) across 3 cohorts.

EN (C=1.0, l1_ratio=0.5), bootstrap SE (N=500) per cohort.

Only genera with |coef| > 0 in ≥2 cohorts are included.

Genera eligible for meta-analysis: **48**
FDR-significant pooled effects (BH q<0.05): **0**
Genera with significant heterogeneity (Q p<0.05): **20**

## Top 20 by pooled effect (sorted by BH FDR q):

          genus  k_cohorts  pooled_effect  ci_lo_95  ci_hi_95   z_pval  bh_fdr_q  I2_pct direction
    Eggerthella          2        0.12688   0.03861   0.21516 0.004844  0.232512     0.0        R+
    Actinomyces          2       -0.09962  -0.18758  -0.01167 0.026417  0.500192     0.0       NR+
Catenibacterium          2       -0.14564  -0.27818  -0.01310 0.031262  0.500192    40.3       NR+
     Carjivirus          2       -0.10311  -0.20423  -0.00199 0.045645  0.547740    24.3       NR+
     Sellimonas          3        0.05538  -0.01389   0.12466 0.117139  0.664608     0.0        R+
  Porphyromonas          2       -0.16380  -0.34218   0.01457 0.071877  0.664608    65.0       NR+
   Lacrimispora          2        0.08244  -0.02201   0.18689 0.121871  0.664608    35.4        R+
          Wujia          3        0.11621  -0.03211   0.26453 0.124614  0.664608    74.7        R+
  Butyricimonas          3        0.04505  -0.01241   0.10251 0.124352  0.664608     0.0        R+
 Paraprevotella          3       -0.10382  -0.24581   0.03817 0.151810  0.728688    73.9       NR+
    Citrobacter          2       -0.06383  -0.15447   0.02682 0.167546  0.731110     0.0       NR+
     Prevotella          2        0.09925  -0.05376   0.25225 0.203612  0.779369    65.6        R+
   Enterobacter          2        0.10686  -0.06061   0.27433 0.211079  0.779369    71.1        R+
  Paenibacillus          2        0.03568  -0.02282   0.09418 0.231930  0.795189     0.0        R+
       Schaalia          2       -0.09060  -0.25517   0.07398 0.280593  0.833633    69.4       NR+
     Romboutsia          2       -0.03958  -0.11369   0.03454 0.295245  0.833633     0.0       NR+
    Barnesiella          2       -0.12825  -0.35176   0.09527 0.260766  0.833633    75.9       NR+
    Veillonella          3       -0.05062  -0.15239   0.05116 0.329661  0.855507    51.5       NR+
   Streptomyces          2       -0.03965  -0.12086   0.04157 0.338638  0.855507    44.7       NR+
  Lactobacillus          2        0.09281  -0.10592   0.29153 0.360024  0.864058    82.0        R+

## Interpretation

No genus reached FDR significance in the meta-analysis. This is consistent
with the overall null finding: cross-cohort effects are not merely masked
by underpowered per-cohort analysis — they are genuinely absent or too
heterogeneous (different directions in different cohorts) to pool reliably.

Heterogeneous genera (Q p<0.05): 20 — these show directionally
inconsistent effects across cohorts, which is itself informative:
the microbiome-response association is cohort/study-specific, not universal.

### Heterogeneous genera (Q p<0.05):
                genus  k_cohorts  I2_pct  Q_pval direction
                Wujia          3    74.7  0.0193        R+
       Paraprevotella          3    73.9  0.0216       NR+
          Barnesiella          2    75.9  0.0416       NR+
        Lactobacillus          2    82.0  0.0185        R+
          Lachnospira          2    81.5  0.0201       NR+
      Christensenella          2    89.5  0.0020        R+
   Lacticaseibacillus          2    74.6  0.0471       NR+
        Campylobacter          3    82.8  0.0030       NR+
       Marvinbryantia          2    88.7  0.0030        R+
           Sutterella          2    80.8  0.0225       NR+
          Akkermansia          3    81.0  0.0051        R+
               Qiania          3    80.3  0.0063       NR+
Phascolarctobacterium          2    78.9  0.0295        R+
           Blohavirus          3    82.1  0.0038        R+
           Emergencia          2    93.9  0.0000       NR+
   Methanobrevibacter          2    89.3  0.0022        R+
           Klebsiella          2    90.2  0.0014        R+
      Faecalibacillus          3    94.2  0.0000       NR+
          Turicimonas          2    83.1  0.0151        R+
          Haemophilus          2    76.3  0.0399       NR+

Outputs saved to: results/ml/meta_analysis/
