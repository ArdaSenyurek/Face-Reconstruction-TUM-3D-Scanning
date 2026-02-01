# Week 6 Results: Depth-Based Parametric Face Reconstruction (Biwi + BFM2019)

## Evaluation Protocol (3-Stage)

1. **Stage 1 — Identity/Shape only (neutral frame)**  
   One neutral frame per sequence (e.g. `frame_00000`). Optimize identity α only (δ = 0, pose fixed from Procrustes). Strong regularization on α. Outputs: `outputs/week6/<seq>/<frame>/meshes/identity.ply`, `outputs/week6/<seq>/identity_state.json`, convergence logs.

2. **Stage 2 — Expression only (single frame)**  
   Identity fixed from Stage 1; optimize expression δ only. Pose fixed from Stage 1. Outputs: `outputs/week6/<seq>/<frame>/meshes/expression.ply`, convergence.

3. **Stage 3 — Frame-by-frame expression (tracking)**  
   Identity fixed from Stage 1; estimate δ per frame for N frames. Optional temporal smoothing (EMA/SLERP). Outputs: `outputs/week6/<seq>/<frame>/meshes/tracked.ply`, `outputs/week6/<seq>/metrics.json`, `outputs/week6/<seq>/convergence.json`.

## Selected Sequences

- **Good:** 01  
- **Medium:** 14  
- **Hard / failure modes:** 17, 19, 21  

Representative set for quantitative and qualitative evaluation.

## Quantitative Tables

- **Summary:** `outputs/week6/summary.csv` (aggregated from `outputs/week6/<seq>/metrics.json`).  
  Run: `python scripts/aggregate_week6_summary.py --week6-dir outputs/week6`.

- **Per-sequence metrics:** `outputs/week6/<seq>/metrics.json` (per-frame mesh paths and, when available, convergence and error metrics).

- **Convergence:** `outputs/week6/<seq>/convergence.json` and `outputs/week6/<seq>/convergence_stage1.json`, `convergence_stage2.json` (energy_history, step_norms, iterations, damping).

- **Run config:** `outputs/week6/run_config.json` and per-run settings for reproducibility.

## Qualitative Figures (paths)

- **3D overlay (rigid):** Scan (cyan) + rigid mesh (red). Open in MeshLab, point size 3–4.
  - **Available:** `outputs/overlays_3d/01/frame_00000_overlay_rigid.ply`, `outputs/overlays_3d/17/frame_00000_overlay_rigid.ply`; report copies: `reports/week6/figures/01_frame_00000_overlay_rigid.ply`, `reports/week6/figures/17_frame_00000_overlay_rigid.ply`.
- **3D overlay (opt):** `outputs/week6/<seq>/<frame>/overlays_3d/*_overlay_opt.ply` — scan + optimized mesh (after Week 6 or optimized run).
- **2D overlay:** `outputs/week6/<seq>/<frame>/overlay_rgb.png` — RGB with projected mesh.
- **Depth comparison:** `outputs/week6/<seq>/<frame>/depth_obs.png`, `depth_rend.png`, `depth_residual.png`.

Best figures for report/presentation can be copied to `reports/week6/figures/` using:

- `python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage rigid --report`  
- `python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage opt --report`

## Discussion

- **What improved:** 3-stage protocol decouples identity and expression, reducing expression instability and “spiky mesh” issues. Strong λ_alpha in Stage 1 and coefficient clamps (max_delta_norm) improve stability. Early stopping (depth not improving, Z out of range) avoids runaway fits. Convergence and run config JSON improve reproducibility.
- **Remaining failure modes:** Hard sequences (17, 19, 21) may still show drift or poor pose/expression. Temporal smoothing (EMA/SLERP) in Stage 3 can reduce jitter but may blur fast motion. GPU-free, CPU-friendly pipeline may limit real-time use on long clips.
