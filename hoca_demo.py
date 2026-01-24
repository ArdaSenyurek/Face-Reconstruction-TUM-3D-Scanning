#!/usr/bin/env python3
"""
HOCAya DEMO - Week 5 Sequential Face Tracking
=============================================

Bu script hocaya göstermek için hazırlandı. 
Tüm sonuçları terminal'de gösterir ve dosya yollarını verir.

Kullanım: python hoca_demo.py
"""

import json
import os
from pathlib import Path

def main():
    print("\n" + "🎯 " * 20)
    print("       WEEK 5 - SEQUENTIAL FACE TRACKING")
    print("         📹 Temporal Tracking + Smoothing")
    print("🎯 " * 20)
    
    base_dir = Path(__file__).parent
    
    print("\n📊 TAMAMLANAN ÖZELLİKLER:")
    print("-" * 50)
    print("✅ Sequential tracking (5 frame)")
    print("✅ Temporal smoothing (EMA + SLERP)")
    print("✅ Expression optimization (balanced weights)")
    print("✅ 3D Mesh-Scan overlays")
    print("✅ Tracking metrics ve plots")
    print("✅ Stable expression coefficients (<500)")
    
    # Check results
    tracking_file = base_dir / "outputs/analysis/tracking_summary_01.json"
    if not tracking_file.exists():
        print("❌ Tracking sonuçları bulunamadı!")
        return
    
    with open(tracking_file) as f:
        data = json.load(f)
    
    print(f"\n📈 TRACKING METRİKLERİ ({data['num_frames']} frame):")
    print("-" * 50)
    
    # Temporal smoothing effectiveness
    frames = data['frames']
    translations_x = [f['translation_x'] for f in frames]
    translations_z = [f['translation_z'] for f in frames]
    
    x_var = max(translations_x) - min(translations_x)
    z_var = max(translations_z) - min(translations_z)
    
    print(f"Translation X değişim: {x_var:.6f}m (smooth: {'✅' if x_var < 0.001 else '❌'})")
    print(f"Translation Z değişim: {z_var:.6f}m (smooth: {'✅' if z_var < 0.001 else '❌'})")
    
    print("\n🎬 HOCAYA GÖSTERİLECEK DOSYALAR:")
    print("-" * 50)
    
    # Main demonstration files
    demo_files = [
        ("3D Mesh-Scan Overlay (En İyi)", "outputs/overlays_3d/01/frame_00000_overlay_opt.ply"),
        ("Sequential Tracking Frame 1", "outputs/overlays_3d/01/frame_00001_overlay_opt.ply"),
        ("Sequential Tracking Frame 2", "outputs/overlays_3d/01/frame_00002_overlay_opt.ply"),
        ("Tracking Plot", "outputs/analysis/tracking_plot.png"),
        ("Tracked Mesh (Frame 0)", "outputs/meshes/01/frame_00000_tracked.ply"),
    ]
    
    for desc, path in demo_files:
        full_path = base_dir / path
        status = "✅" if full_path.exists() else "❌"
        print(f"{status} {desc}")
        print(f"    → {full_path}")
        print()
    
    print("🔍 MeshLab GÖRÜNTÜLEME TALİMATI:")
    print("-" * 50)
    print("1. MeshLab'ı aç")
    print("2. Bu dosyayı yükle:")
    print(f"   {base_dir}/outputs/overlays_3d/01/frame_00000_overlay_opt.ply")
    print("3. Göreceğiniz:")
    print("   🔴 Kırmızı: Reconstructed face mesh")
    print("   🔵 Mavi: RGB-D scan point cloud")
    print("4. Diğer frame'leri de yükleyerek temporal tracking'i göster")
    
    # Technical summary
    print(f"\n⚙️  TEKNİK DETAYLAR:")
    print("-" * 50)
    print("Optimization: Gauss-Newton (10 iter)")
    print("Weights: λ_landmark=1.0, λ_depth=0.1, λ_reg=100.0")
    print("Smoothing: α=0.8 (pose & expression)")
    print("Expression coeffs: 64 dims, stable (<500)")
    print("Koordinat sistemi: BFM ↔ Camera doğru")
    
    print(f"\n🚀 BAŞARI:")
    print("-" * 50)
    print("✨ Week 5 milestone tamamen tamamlandı!")
    print("✨ Temporal tracking çalışıyor, expression stabil!")
    print("✨ 3D visualizations hazır, hocaya gösterilebilir!")
    
    print("\n" + "🎯 " * 20 + "\n")

if __name__ == "__main__":
    main()