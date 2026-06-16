# Phase 3c: Taxonomic Level Sensitivity

PERMANOVA (Aitchison distance, 999 perms) — batch vs. response R² at each taxonomic level

Genus-level values from prior batch_diagnostics runs.

Level      Factor      n_taxa       R²        p batch:resp ratio
-----------------------------------------------------------------
phylum     batch           75   0.0588   0.0000 ***          8.4×
phylum     response        75   0.0070   0.6006 ns             —
genus      batch         2813   0.0768   0.0010 **         11.3×
genus      response      2813   0.0068   0.8470 ns             —
species    batch         7995   0.0741   0.0000 ***          9.3×
species    response      7995   0.0080   0.5485 ns             —

## Interpretation

If batch R² >> response R² at all three levels, the fundamental problem
(batch dominates signal regardless of taxonomic resolution) is not an artifact
of genus-level aggregation — it reflects a genuine platform/protocol confound.
Species level may show LARGER batch effects (more granular = more platform noise)
while phylum level may show SMALLER batch effects (more aggregated = less noise).
Either way, if response R² stays near genus level (0.007), the null finding is robust.
