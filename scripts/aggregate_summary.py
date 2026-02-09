#!/usr/bin/env python3
"""
Week 6: Aggregate metrics from outputs/week6/<seq>/ into a single summary.csv.

Reads metrics.json and convergence.json per sequence and produces:
  outputs/week6/summary.csv

Usage:
  python scripts/aggregate_summary.py [--week6-dir outputs/week6]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week6-dir", type=Path, default=Path("outputs/week6"))
    args = ap.parse_args()
    week6 = args.week6_dir.resolve()
    if not week6.exists():
        print("Week6 dir not found:", week6)
        return 1

    rows: list[dict] = []
    for seq_dir in sorted(week6.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        metrics_path = seq_dir / "metrics.json"
        conv_path = seq_dir / "convergence.json"
        run_config_path = seq_dir / "run_config.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                data = json.load(f)
        except Exception:
            continue
        frames = data.get("frames", [])
        for fr in frames:
            row = {
                "sequence": seq,
                "frame": fr.get("frame", ""),
                "mesh": fr.get("mesh", ""),
            }
            m = fr.get("metrics") or {}
            if m:
                lm = m.get("landmark_reprojection") or {}
                de = m.get("depth_error") or {}
                se = m.get("surface_error") or {}
                row["landmark_rmse_px"] = lm.get("rmse_px", "")
                row["depth_mae_mm"] = de.get("mae_mm", "")
                row["depth_rmse_mm"] = de.get("rmse_mm", "")
                row["surface_pct_5mm"] = se.get("pct_under_5mm", "")
                row["surface_pct_10mm"] = se.get("pct_under_10mm", "")
            if conv_path.exists():
                try:
                    with open(conv_path) as f:
                        conv = json.load(f)
                    per_frame = conv.get("per_frame", [])
                    idx = next((i for i, p in enumerate(per_frame) if True), 0)
                    if per_frame and idx < len(per_frame):
                        c = per_frame[idx]
                        row["iterations"] = c.get("iterations", "")
                        row["converged"] = c.get("converged", "")
                        row["final_energy"] = c.get("final_energy", "")
                except Exception:
                    pass
            rows.append(row)

    if not rows:
        # Build minimal header from first available metrics
        for seq_dir in sorted(week6.iterdir()):
            if not seq_dir.is_dir():
                continue
            metrics_path = seq_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                frames = data.get("frames", [])
                if frames:
                    rows = [{"sequence": seq_dir.name, "frame": frames[0].get("frame", ""), "mesh": frames[0].get("mesh", "")}]
                break

    out_csv = week6 / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("Wrote", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
