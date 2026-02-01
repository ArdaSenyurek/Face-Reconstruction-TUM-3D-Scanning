# Week 6 Figures

## Mesh–scan overlay (rigid alignment)

- **01_frame_00000_overlay_rigid.ply** — Sequence 01, frame_00000: scan (cyan) + rigid-aligned mesh (red). Combined overlay.
- **01_frame_00000_overlay_metrics.json** — Overlay metrics (NN-RMSE, bbox overlap, etc.).
- **17_frame_00000_overlay_rigid.ply** — Sequence 17, frame_00000: same format.
- **17_frame_00000_overlay_metrics.json** — Overlay metrics for 17.

**How to view:** Open the `.ply` files in MeshLab or CloudCompare; set point size to 3–4. Units: meters.

**Source:** Generated with `build/bin/create_overlays` from `outputs/pose_init/<seq>/frame_00000_aligned.ply` and `outputs/converted/<seq>/depth/frame_00000.png`. Full outputs: `outputs/overlays_3d/01/`, `outputs/overlays_3d/17/`.

## Reproducing / generating more figures

- **More rigid overlays:**  
  `build/bin/create_overlays --mesh-rigid outputs/pose_init/<seq>/frame_<N>_aligned.ply --depth outputs/converted/<seq>/depth/frame_<N>.png --intrinsics outputs/converted/<seq>/intrinsics.txt --out-dir outputs/overlays_3d/<seq> --frame-name frame_<N>`

- **Week 6 evaluation (identity/expression/tracking + metrics + visuals):**  
  `python scripts/run_week6_eval.py --sequences 01 14 17 19 --frames 5`

- **Optimized single-frame mesh:**  
  `python pipeline/main.py --frames 1 --optimize`  
  (or run `build/bin/face_reconstruction ... --optimize --output-convergence-json <path>` for one frame.)

- **Copy overlay to this folder:**  
  `python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage rigid --report`
