# Week 6 — Milestones

## What have we accomplished this week?

- Implemented the Week 6 three-stage evaluation protocol (identity only → expression only → frame-by-frame tracking) and wired it into `run_week6_eval.py`.
- Added quantitative metrics: 2D landmark reprojection error, depth error (MAE/RMSE mm), and 3D surface error (% under 5/10/20 mm); aggregation into `summary.csv`.
- Integrated convergence logging (energy history, step norms, iterations) and run config JSON for reproducibility.
- Automated metric computation, 2D/depth visuals, and 3D overlay PLY export inside the Week 6 eval script; aggregate summary runs at the end.
- Generated mesh–scan overlays after rigid alignment for sequences 01 and 17 (`outputs/overlays_3d/`, `reports/week6/figures/`).
- Prepared report and presentation materials: `week6_results.md`, `week6_slides_outline.md`, and figure README with reproduction commands.

## Which problems did we encounter?

- **Landmark mapping file missing:** Pose init failed with “Landmark mapping file not found”; we had no BFM .h5 to run the mapping script. **We solved it by** using the BFMLandmarks (Landmarks68_BFM.anl) 68 vertex indices and writing `data/bfm_landmark_68.txt`; validated with `validate_mapping`.
- **No rigid overlay outputs:** Overlay PLY (scan + mesh) had never been produced for the report. **We solved it by** running `create_overlays` for seq 01 and 17 and copying results to `reports/week6/figures/`.
- **Segfault in optimization (some environments):** `face_reconstruction --optimize` and Week 6 eval can hit exit 139 during Gauss–Newton. **We solved it by** documenting local run and debug steps (gdb/valgrind) in the figures README; no code change.

## Outline the plan for the next week

- Finalize the written report and presentation (PDF/PPT); embed evaluation tables and figures from `reports/week6/figures/`.
- Run the full Week 6 evaluation locally (identity/expression/tracking, metrics, convergence) and fill `outputs/week6/` when the binary runs without segfault.
- Compare sequential tracking (Week 5) with single-frame reconstruction and with the 3-stage protocol; add a short summary to the report.
- Complete submission: report, slides, and code/artifacts checklist.
