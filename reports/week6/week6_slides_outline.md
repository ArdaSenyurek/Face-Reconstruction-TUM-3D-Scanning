# Week 6 Presentation Outline (10–12 slides)

## 1. Title
- Depth-Based Parametric Face Reconstruction from RGB-D (Biwi + BFM2019)
- Course / team / date

## 2. Problem
- Goal: fit a parametric 3D face model (identity + expression) to RGB-D sequences.
- Challenges: expression instability, “spiky” meshes, need for reproducible evaluation.

## 3. Method
- PCA morphable face model (BFM2019): identity α, expression δ, pose (R, t, scale).
- Energy: landmark reprojection + depth alignment + regularization.
- Optimizer: Gauss-Newton / Levenberg–Marquardt style (damping, line search).

## 4. Pipeline Overview
- Data: Biwi download → convert → landmarks (dlib) → pose init (Procrustes) → optimize / track.
- Week 6: 3-stage protocol (identity only → expression only → frame-by-frame tracking).

## 5. Rigid Alignment Results
- Procrustes alignment: landmark–model correspondence, similarity transform.
- Visuals: rigid overlay PLY (scan cyan, mesh red); sanity check on sequences 01, 14, 17, 19.

## 6. Optimization (GN/LM) Results
- Stage 1: identity only on neutral frame; strong λ_alpha.
- Stage 2: expression only with fixed identity.
- Convergence: energy vs iteration, step norms, damping; convergence.json.

## 7. Tracking + Smoothing Results
- Stage 3: fixed identity, δ per frame; optional EMA/SLERP for pose and expression.
- Outputs: meshes, metrics.json, temporal smoothness (when computed).

## 8. Quantitative Metrics Summary
- 2D landmark reprojection (mean/median/RMSE px).
- Depth error (MAE/RMSE mm) on valid overlap.
- 3D surface error: mean/median/RMSE and % under 5 mm, 10 mm, 20 mm.
- Summary table: outputs/week6/summary.csv.

## 9. Qualitative Results
- Rigid vs optimized 3D overlays; depth residual heatmaps; convergence plots; temporal smoothness (if available).

## 10. Limitations
- CPU-only; hard sequences (17, 19, 21); possible drift in long clips; smoothing vs responsiveness trade-off.

## 11. Future Work
- GPU acceleration; better neutral-frame selection; learned priors; real-time tracking.

## 12. Conclusion / Q&A
- 3-stage protocol and regularization improve stability and reproducibility; report and scripts in repo.
