#!/usr/bin/env python3
"""
Quick script to check Biwi dataset structure and find RGB/depth files.
"""

import sys
import argparse
from pathlib import Path


def check_dataset(dataset_path):
    """Check dataset structure and report findings."""
    base = Path(dataset_path)
    
    if not base.exists():
        print(f"❌ Error: Path does not exist: {dataset_path}")
        return False
    
    print(f"\n📁 Checking dataset: {dataset_path}\n")
    
    # Find RGB files
    rgb_patterns = ["*rgb*", "*RGB*", "*r*.png", "*R*.png"]
    rgb_files = []
    for pattern in rgb_patterns:
        rgb_files.extend(base.rglob(pattern))
    rgb_files = sorted(set([f for f in rgb_files if f.is_file()]))
    
    # Find depth files
    depth_patterns = ["*depth*", "*Depth*", "*d*.png", "*D*.png"]
    depth_files = []
    for pattern in depth_patterns:
        depth_files.extend(base.rglob(pattern))
    depth_files = sorted(set([f for f in depth_files if f.is_file()]))
    
    # Count by directory
    rgb_dirs = {}
    depth_dirs = {}
    
    for f in rgb_files:
        parent = str(f.parent)
        rgb_dirs[parent] = rgb_dirs.get(parent, 0) + 1
    
    for f in depth_files:
        parent = str(f.parent)
        depth_dirs[parent] = depth_dirs.get(parent, 0) + 1
    
    # Report
    print(f"✅ Found {len(rgb_files)} RGB files")
    print(f"✅ Found {len(depth_files)} depth files")
    print()
    
    if len(rgb_files) == 0:
        print("❌ No RGB files found!")
        print("   Try checking the dataset structure manually:")
        print(f"   ls -la {dataset_path}")
        return False
    
    if len(depth_files) == 0:
        print("❌ No depth files found!")
        print("   Try checking the dataset structure manually:")
        print(f"   find {dataset_path} -name '*.png' | head -10")
        return False
    
    # Show directory structure
    print("📂 RGB files found in:")
    for dir_path, count in sorted(rgb_dirs.items(), key=lambda x: -x[1])[:5]:
        rel_path = Path(dir_path).relative_to(base)
        print(f"   {rel_path}/ : {count} files")
    
    print("\n📂 Depth files found in:")
    for dir_path, count in sorted(depth_dirs.items(), key=lambda x: -x[1])[:5]:
        rel_path = Path(dir_path).relative_to(base)
        print(f"   {rel_path}/ : {count} files")
    
    # Show sample files
    print(f"\n📸 Sample RGB files:")
    for f in rgb_files[:3]:
        rel_path = f.relative_to(base)
        print(f"   {rel_path}")
    
    print(f"\n📸 Sample depth files:")
    for f in depth_files[:3]:
        rel_path = f.relative_to(base)
        print(f"   {rel_path}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if len(rgb_dirs) == 1 and len(depth_dirs) == 1:
        print("   Dataset structure looks good!")
        print(f"   Try: python scripts/convert_biwi_dataset.py --input {dataset_path} --output data/biwi_test")
    elif len(rgb_dirs) > 1 or len(depth_dirs) > 1:
        print("   Multiple directories found. Try converting a subdirectory:")
        print("   For example:")
        if rgb_dirs:
            first_dir = list(rgb_dirs.keys())[0]
            rel_dir = Path(first_dir).relative_to(base)
            print(f"   python scripts/convert_biwi_dataset.py --input {first_dir} --output data/biwi_test")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Check Biwi dataset structure'
    )
    parser.add_argument('dataset_path', type=str,
                       help='Path to Biwi dataset directory')
    
    args = parser.parse_args()
    
    success = check_dataset(args.dataset_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
