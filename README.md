# Depth-Based Parametric Face Reconstruction (Biwi + BFM2019)

3D face reconstruction from RGB-D using a PCA morphable model (BFM2019), Gauss-Newton optimization, and Biwi Kinect Head Pose dataset.

## Prerequisites: Basel Face Model (BFM)

Pipeline, yüz modeli için **Basel Face Model 2009** kullanır. Model olmadan reconstruction adımı çalışmaz.

1. **Kayıt:** https://faces.dmi.unibas.ch/bfm/ adresinde kayıt olun.
2. **İndirme:** BFM 2009’u indirin (örn. `01_MorphableModel.mat`).
3. **Yerleştirme:** Dosyayı `data/bfm/` klasörüne koyun (örn. `data/bfm/01_MorphableModel.mat`).
4. Pipeline’ı tekrar çalıştırın; Model Setup adımı bunu proje formatına dönüştürür.

**Model zaten `data/bfm/model_bfm/` içindeyse** pipeline varsayılan olarak bu klasörü kullanır; ek ayar gerekmez. Farklı bir yerdeyse `--model-dir <yol>` ile belirtin.

Ayrıntı için: `data/bfm/README.md`.

## Pipeline

- **Download / convert:** Biwi sequences → RGB + depth + intrinsics.
- **Landmarks:** dlib 68-point detection.
- **Pose init:** Procrustes alignment (landmark–model correspondence).
- **Optimize / track:** Gauss-Newton (landmark + depth + regularization); optional temporal smoothing.

## Week 6 Evaluation (3-Stage Protocol)

1. **Run full Week 6 evaluation** (Stage 1 → 2 → 3) on selected sequences:

   ```bash
   python scripts/run_week6_eval.py --sequences 01 14 17 19 --frames 10
   ```

   Outputs: `outputs/week6/<seq>/<frame>/meshes/` (identity.ply, expression.ply, tracked.ply), `outputs/week6/<seq>/identity_state.json`, `metrics.json`, `convergence.json`, `run_config.json`. By default the script also **computes per-frame metrics** (landmark reprojection, depth error, surface error), **generates visuals** (overlay_rgb.png, depth_obs/rend/residual.png), **exports 3D overlay PLY** (rigid + opt), and **runs aggregate** to produce `summary.csv`. Use `--no-metrics`, `--no-visuals`, `--no-overlays`, or `--no-aggregate` to skip any of these.

2. **Aggregate summary table** (run automatically at end; or run manually):

   ```bash
   python scripts/aggregate_week6_summary.py --week6-dir outputs/week6
   ```

   Produces `outputs/week6/summary.csv` (includes metric columns when per-frame metrics were computed).

### 3D Overlay PLY (MeshLab)

- Export rigid or optimized overlay for a sequence/frame:

  ```bash
  python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage rigid
  python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage opt
  ```

- **Open in MeshLab:** point size 3–4, verify overlap (scan cyan, mesh red). Units: meters; rigid and optimized meshes in the same frame.

### Reports and Slides

- **Results:** `reports/week6/week6_results.md` (protocol, sequences, metrics, figures list, discussion).
- **Slides outline (10–12):** `reports/week6/week6_slides_outline.md`.
- **Figures:** copy best overlays/plots to `reports/week6/figures/` (e.g. with `--report` in `export_overlay_ply.py`).

## Build and Quick Sanity Check

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
python pipeline/main.py --frames 1 --skip-reconstruct   # or minimal run
python pipeline/main.py --sequence 01 --frames 1 --optimize  # single-frame optimize (if supported by CLI)
```

## License

See LICENSE.
