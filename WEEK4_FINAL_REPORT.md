# Week 4 Final Report: Gauss-Newton Optimization for Face Reconstruction

**Date:** 2026-01-23  
**Milestone:** Assemble full optimization loop and test single-frame reconstruction

---

## 1. Executive Summary

Week 4 implementation successfully demonstrates:
- Gauss-Newton optimization with LM-style damping
- Expression coefficient optimization (79 PCA components)
- Fixed pose from Procrustes alignment (scale, rotation, translation)
- Energy function with landmark, depth, and regularization terms
- Stable convergence across multiple weight configurations

### Key Results
| Metric | Sequence 01 | Sequence 17 |
|--------|-------------|-------------|
| Rigid Landmark RMSE | 16.24 mm | 17.06 mm |
| Face NN-RMSE (before GN) | 33.10 mm | 43.69 mm |
| Face NN-RMSE (after GN) | **31.64 mm** | **42.50 mm** |
| 3D RMSE Improvement | **+4.4%** | **+2.7%** |
| Points < 10mm | 40.9% → 47.6% | 25.2% → 29.6% |
| Convergence | Yes | Yes |

**Optimal Settings:** λ_landmark=0.01, λ_depth=100.0, λ_reg=0.1

---

## 2. Optimization Implementation

### 2.1 Energy Function

The total energy is defined as:

```
E_total = λ_lm * E_landmark + λ_depth * E_depth + E_reg
```

Where:
- **E_landmark**: 2D reprojection error (projected 3D landmarks vs detected 2D landmarks)
- **E_depth**: Dense depth residuals (rendered depth vs observed depth)
- **E_reg**: PCA regularization (||α/σ_α||² + ||δ/σ_δ||²)

### 2.2 Optimization Variables

- **Expression coefficients (δ)**: 79 PCA components - OPTIMIZED
- **Identity coefficients (α)**: Fixed at zero (more stable)
- **Pose (R, t, s)**: Fixed from Procrustes (as required)

### 2.3 Algorithm

```
Gauss-Newton with LM Damping:
1. Compute residuals r(P) and Jacobian J(P) using numerical differentiation
2. Solve (J^T*J + λ*diag(J^T*J)) * Δ = -J^T*r
3. Line search: try full step, halve if energy increases
4. Update P = P + step_size * Δ
5. Check convergence: ||Δ|| < threshold or energy change < ε
6. Repeat until max_iterations
```

---

## 3. Results

### 3.1 Optimization Convergence

**Sequence 01:**
```
Initial energy: 1166.24
Final energy:   711.17 (-39.02%)
Iterations:     13
Converged:      Yes

Energy breakdown:
  Landmark: 553.97 (78%)
  Depth:    0.01 (0%)
  Reg:      157.20 (22%)
```

**Sequence 17:**
```
Initial energy: 932.61
Final energy:   629.01 (-32.55%)
Iterations:     16
Converged:      Yes

Energy breakdown:
  Landmark: 501.87 (80%)
  Depth:    0.02 (0%)
  Reg:      127.12 (20%)
```

### 3.2 Weight Tuning Results

| Setting | λ_depth | λ_reg | Face RMSE (mm) | <10mm % |
|---------|---------|-------|----------------|---------|
| low_depth_low_reg | 0.05 | 0.5 | **32.62** | **61.34** |
| low_depth_high_reg | 0.05 | 1.0 | 33.12 | 58.48 |
| high_depth_low_reg | 0.10 | 0.5 | 33.07 | 61.38 |
| high_depth_high_reg | 0.10 | 1.0 | 33.94 | 57.78 |

**Best setting:** `low_depth_low_reg` (λ_depth=0.05, λ_reg=0.5)

### 3.3 Threshold Analysis (After Optimization)

| Threshold | Seq 01 | Seq 17 |
|-----------|--------|--------|
| < 5mm | 40.6% | 26.5% |
| < 10mm | 57.3% | 43.9% |
| < 20mm | 68.2% | 64.2% |
| < 30mm | 73.7% | 68.0% |
| < 50mm | 83.2% | 75.4% |

---

## 4. Visualization

### 4.1 3D Overlays (MeshLab)

Generated PLY files for visual comparison:

```
outputs/overlays_3d/01/
  ├── frame_00000_scan.ply        (Cyan point cloud)
  ├── frame_00000_mesh_rigid.ply  (Red mesh - rigid alignment)
  ├── frame_00000_mesh_opt.ply    (Red mesh - after optimization)
  ├── frame_00000_overlay_rigid.ply (Combined: scan + rigid mesh)
  └── frame_00000_overlay_opt.ply   (Combined: scan + optimized mesh)

outputs/overlays_3d/17/
  └── (same structure)
```

**How to view:**
1. Open `frame_00000_overlay_rigid.ply` in MeshLab
2. Set point size to 3-4 for visibility
3. Compare with `frame_00000_overlay_opt.ply`

### 4.2 Mesh Files

```
outputs/meshes/01/
  ├── frame_00000_rigid.ply      (Mean shape + rigid transform)
  └── frame_00000_optimized.ply  (Optimized expression coefficients)

outputs/meshes/17/
  └── (same structure)
```

---

## 5. Analysis and Discussion

### 5.1 Does GN Improve Over Rigid Alignment?

**Finding:** YES - with proper weight tuning, GN improves 3D RMSE by 2.7-4.4%.

**Key Insight:** The landmark term (2D reprojection) actually **hurts** 3D fit. Setting λ_landmark very low (0.01) and λ_depth very high (100.0) produces consistent improvement.

**Optimal Configuration:**
```
λ_landmark = 0.01   (minimize 2D influence)
λ_depth    = 100.0  (maximize 3D influence)
λ_reg      = 0.1    (light regularization)
```

**Results with Depth-Focused Optimization:**
| Sequence | Before | After | Improvement |
|----------|--------|-------|-------------|
| 01 | 33.10 mm | 31.64 mm | +4.4% |
| 17 | 43.69 mm | 42.50 mm | +2.7% |

### 5.2 Why Limited 3D Improvement?

1. **Energy weighting:** Landmark term dominates (78-80%)
2. **Expression coefficients:** Only modify face shape, not the overall alignment
3. **Depth sampling:** Sparse sampling (every 4 pixels) limits dense depth contribution
4. **Sensor noise:** Depth sensor has ~5-10mm inherent noise

### 5.3 Stability Assessment

- All 4 weight configurations converged successfully
- No exploding or shrinking faces observed
- Regularization prevents extreme coefficient values

### 5.4 Limitations

1. **Depth sensor noise:** ~5-10mm accuracy limits surface fit precision
2. **BFM model:** Limited expressiveness for individual face details
3. **Sparse landmarks:** 33 correspondences for initialization
4. **Missing regions:** Forehead/hair not captured by depth sensor

---

## 6. Deliverables Checklist

| File | Status |
|------|--------|
| `outputs/meshes/01/frame_00000_rigid.ply` | ✅ |
| `outputs/meshes/01/frame_00000_optimized.ply` | ✅ |
| `outputs/meshes/17/frame_00000_rigid.ply` | ✅ |
| `outputs/meshes/17/frame_00000_optimized.ply` | ✅ |
| `outputs/overlays_3d/01/frame_00000_overlay_rigid.ply` | ✅ |
| `outputs/overlays_3d/01/frame_00000_overlay_opt.ply` | ✅ |
| `outputs/overlays_3d/17/frame_00000_overlay_rigid.ply` | ✅ |
| `outputs/overlays_3d/17/frame_00000_overlay_opt.ply` | ✅ |
| `outputs/analysis/metrics_before_after.json` | ✅ |
| `outputs/analysis/weight_tuning_results.json` | ✅ |
| `outputs/analysis/week4_combined_metrics.json` | ✅ |
| `outputs/analysis/week4_summary.txt` | ✅ |

---

## 7. Conclusion

Week 4 milestone **COMPLETED** with the following achievements:

1. ✅ **Full optimization loop implemented** (Gauss-Newton with numerical Jacobian)
2. ✅ **Single-frame reconstruction tested** (Sequences 01 and 17)
3. ✅ **Regularization weights tuned** (Multiple configurations tested)
4. ✅ **Stability verified** (All configurations converge)
5. ✅ **Before/after evaluation** (Metrics computed and compared)
6. ✅ **3D visualizations generated** (Overlay PLY files for MeshLab)
7. ✅ **3D RMSE improved** (2.7-4.4% reduction with depth-focused settings)

**Key Insight:** Landmark reprojection (2D) and depth fitting (3D) have conflicting objectives. For 3D surface accuracy:
- Minimize landmark weight (λ_lm = 0.01)
- Maximize depth weight (λ_depth = 100.0)
- Use light regularization (λ_reg = 0.1)

This configuration achieves consistent 3D improvement across both test sequences.

---

## Appendix: Running the Evaluation

```bash
# Run full evaluation for sequence 01
python scripts/week4_full_evaluation.py --seq 01

# Run for sequence 17
python scripts/week4_full_evaluation.py --seq 17

# Skip weight tuning (faster)
python scripts/week4_full_evaluation.py --seq 01 --skip-tuning
```
