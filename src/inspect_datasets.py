"""
Dataset Inspection Script for Driver Drowsiness Detection System.

Inspects all physically uploaded Kaggle datasets:
1. Driver Drowsiness Dataset (DDD)
2. MRL Eye Dataset
3. Yawn Dataset

Reports:
- Dataset name & paths
- Total files, images, videos, and annotation files
- Available classes, splits, and folder distribution
- File extensions breakdown
- Image dimensions (resolutions and color channels)
- Potential corrupted or unreadable files verification
"""

import sys
import os
import argparse
from pathlib import Path
from collections import Counter
from PIL import Image

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
    DDD_PATH = config.DDD_DATASET_DIR
    MRL_PATH = config.MRL_DATASET_DIR
    YAWN_PATH = config.YAWN_DATASET_DIR
except ImportError:
    DATASET_DIR = PROJECT_ROOT / "dataset"
    DDD_PATH = DATASET_DIR / "DDD Dataset"
    MRL_PATH = DATASET_DIR / "MRL Dataset"
    YAWN_PATH = DATASET_DIR / "Yawn Dataset"

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
ANNOTATION_EXTENSIONS = {'.csv', '.txt', '.xml', '.json'}


def inspect_dataset(dataset_name: str, dataset_path: Path, max_samples_for_dims: int = 500, verify_images: bool = True):
    """
    Inspects a single dataset folder and gathers comprehensive statistics.
    """
    stats = {
        "name": dataset_name,
        "path": dataset_path,
        "exists": dataset_path.exists(),
        "total_files": 0,
        "images": 0,
        "videos": 0,
        "annotations": 0,
        "other_files": 0,
        "extensions": Counter(),
        "classes": Counter(),
        "annotation_files": [],
        "dimensions": Counter(),
        "color_modes": Counter(),
        "corrupted_files": [],
    }

    if not stats["exists"]:
        return stats

    # Traverse dataset directory
    for root, dirs, files in os.walk(dataset_path):
        rel_root = Path(root).relative_to(dataset_path)
        for f in files:
            stats["total_files"] += 1
            file_path = Path(root) / f
            ext = file_path.suffix.lower()
            stats["extensions"][ext] += 1

            # Class determination based on subfolder structure
            if len(rel_root.parts) > 0:
                class_key = "/".join(rel_root.parts)
                stats["classes"][class_key] += 1
            else:
                stats["classes"]["[root]"] += 1

            # Categorize file types
            if ext in IMAGE_EXTENSIONS:
                stats["images"] += 1

                # Sample dimensions and color modes
                if stats["images"] <= max_samples_for_dims:
                    try:
                        with Image.open(file_path) as img:
                            stats["dimensions"][img.size] += 1
                            stats["color_modes"][img.mode] += 1
                            if verify_images:
                                img.verify()
                    except Exception as e:
                        stats["corrupted_files"].append((str(file_path), str(e)))
                elif verify_images and file_path.stat().st_size == 0:
                    stats["corrupted_files"].append((str(file_path), "Zero-byte file"))

            elif ext in VIDEO_EXTENSIONS:
                stats["videos"] += 1
            elif ext in ANNOTATION_EXTENSIONS:
                stats["annotations"] += 1
                stats["annotation_files"].append(str(rel_root / f))
            else:
                stats["other_files"] += 1

    return stats


def print_inspection_summary(results: list):
    """
    Prints a clear, formatted summary according to project specifications.
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "DATASET INSPECTION SUMMARY")
    print("=" * 60 + "\n")

    for stat in results:
        name = stat["name"]
        print(f"==============================")
        print(f"{name}")
        print(f"==============================")

        if not stat["exists"]:
            print(f"STATUS: Directory NOT found at: {stat['path']}\n")
            continue

        print(f"Location: {stat['path']}")
        print(f"Total Files: {stat['total_files']:,}")
        print(f"Images: {stat['images']:,}")
        print(f"Videos: {stat['videos']:,}")
        print(f"Annotations: {stat['annotations']:,}")
        if stat["annotation_files"]:
            print(f"  Annotation files found: {', '.join(stat['annotation_files'])}")

        print("\nClasses / Folders Distribution:")
        for cls_name, count in sorted(stat["classes"].items()):
            print(f"  - {cls_name}: {count:,} files")

        print("\nFile Extensions:")
        for ext, count in sorted(stat["extensions"].items(), key=lambda x: -x[1]):
            ext_label = ext if ext else "[no extension]"
            print(f"  - {ext_label}: {count:,}")

        print("\nImage Properties (Sampled):")
        print(f"  Color Modes: {dict(stat['color_modes'])}")
        top_dims = stat["dimensions"].most_common(5)
        dim_str = ", ".join([f"{w}x{h} ({cnt})" for (w, h), cnt in top_dims])
        print(f"  Frequent Dimensions (Width x Height): {dim_str if dim_str else 'N/A'}")

        print("\nData Integrity:")
        if stat["corrupted_files"]:
            print(f"  WARNING: {len(stat['corrupted_files'])} potential corrupted/unreadable files found!")
            for path, err in stat["corrupted_files"][:5]:
                print(f"    - {path}: {err}")
        else:
            print("  Potential Corrupted Files: 0 (All verified images are valid and intact)")

        print("-" * 60 + "\n")


def run_all_inspections(verify_all: bool = False):
    """
    Runs inspection across all 3 registered datasets.
    """
    datasets = [
        ("DDD Dataset (Driver Drowsiness Dataset)", DDD_PATH),
        ("MRL Eye Dataset", MRL_PATH),
        ("Yawn Dataset", YAWN_PATH),
    ]

    results = []
    for name, path in datasets:
        print(f"Scanning {name}...")
        sample_limit = 1000000 if verify_all else 1000
        stat = inspect_dataset(name, path, max_samples_for_dims=sample_limit, verify_images=True)
        results.append(stat)

    print_inspection_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Driver Drowsiness Detection Datasets")
    parser.add_argument("--verify-all", action="store_true", help="Perform verification on every single image in the dataset")
    args = parser.parse_args()

    run_all_inspections(verify_all=args.verify_all)
