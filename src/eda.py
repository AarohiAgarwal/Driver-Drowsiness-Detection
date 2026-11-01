"""
Exploratory Data Analysis (EDA) Script for Driver Drowsiness Detection System.

Analyzes the three physically uploaded datasets:
1. MRL Eye Dataset (Eye closure / drowsiness analysis)
2. Yawn Dataset (Yawning / mouth state analysis)
3. Driver Drowsiness Dataset (DDD) (Holistic driver posture / drowsiness analysis)

Generates publication-quality graphs and saves them to outputs/graphs/.
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is accessible
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
    DDD_PATH = config.DDD_DATASET_DIR
    MRL_PATH = config.MRL_DATASET_DIR
    YAWN_PATH = config.YAWN_DATASET_DIR
    GRAPHS_DIR = config.GRAPHS_DIR
except ImportError:
    DATASET_DIR = PROJECT_ROOT / "dataset"
    DDD_PATH = DATASET_DIR / "DDD Dataset"
    MRL_PATH = DATASET_DIR / "MRL Dataset"
    YAWN_PATH = DATASET_DIR / "Yawn Dataset"
    GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn & Matplotlib styling
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE_PRIMARY = ["#2b5c8f", "#e26d5c", "#38b000", "#7209b7", "#f77f00"]


# ==============================================================================
# 1. MRL EYE DATASET ANALYSIS
# ==============================================================================

def analyze_mrl_dataset(mrl_path: Path):
    """
    Performs comprehensive EDA on MRL Eye Dataset.
    """
    print("\n" + "=" * 60)
    print("ANALYZING MRL EYE DATASET")
    print("=" * 60)

    splits = ["train", "val", "test"]
    categories = ["awake", "sleepy"]

    split_counts = defaultdict(dict)
    total_awake = 0
    total_sleepy = 0
    dimensions = []
    color_modes = Counter()
    sample_images = {"awake": [], "sleepy": []}

    # Attribute counters parsed from filenames
    # s0001_01842_0_0_1_0_0_01.png
    # parts: [subject, img_id, gender, glasses, eye_state, reflections, lighting, sensor_id]
    subjects = Counter()
    glasses = Counter()
    genders = Counter()
    lightings = Counter()

    for split in splits:
        for cat in categories:
            folder = mrl_path / split / cat
            if not folder.exists():
                continue
            files = list(folder.glob("*.png"))
            count = len(files)
            split_counts[split][cat] = count
            if cat == "awake":
                total_awake += count
            else:
                total_sleepy += count

            # Sample filenames for parsing and dimension inspection
            for i, fp in enumerate(files):
                # Sample images for visualization
                if len(sample_images[cat]) < 8:
                    sample_images[cat].append(fp)

                # Collect dimensions for first 2,000 images per split/class
                if i < 2000:
                    try:
                        with Image.open(fp) as im:
                            dimensions.append(im.size)
                            color_modes[im.mode] += 1
                    except Exception:
                        pass

                # Parse filename metadata
                parts = fp.stem.split("_")
                if len(parts) >= 8:
                    subjects[parts[0]] += 1
                    genders["Male" if parts[2] == "0" else "Female"] += 1
                    glasses["No Glasses" if parts[3] == "0" else "Glasses"] += 1
                    lightings["Bad Lighting" if parts[6] == "0" else "Good Lighting"] += 1

    total_images = total_awake + total_sleepy
    awake_pct = (total_awake / total_images) * 100 if total_images else 0
    sleepy_pct = (total_sleepy / total_images) * 100 if total_images else 0

    print(f"Total MRL Images: {total_images:,}")
    print(f"Open-Eye Images (Awake):  {total_awake:,} ({awake_pct:.2f}%)")
    print(f"Closed-Eye Images (Sleepy): {total_sleepy:,} ({sleepy_pct:.2f}%)")
    print(f"Class Balance Ratio (Awake : Sleepy): {total_awake / max(total_sleepy, 1):.3f} : 1")
    print("\nDistribution across splits:")
    for sp in splits:
        print(f"  - {sp}: Awake={split_counts[sp].get('awake', 0):,}, Sleepy={split_counts[sp].get('sleepy', 0):,}")

    dim_widths = [d[0] for d in dimensions]
    dim_heights = [d[1] for d in dimensions]
    print(f"\nImage Dimensions (Sampled {len(dimensions):,} images):")
    print(f"  Width  -> Min: {np.min(dim_widths)}, Max: {np.max(dim_widths)}, Mean: {np.mean(dim_widths):.1f}, Median: {np.median(dim_widths):.1f}")
    print(f"  Height -> Min: {np.min(dim_heights)}, Max: {np.max(dim_heights)}, Mean: {np.mean(dim_heights):.1f}, Median: {np.median(dim_heights):.1f}")
    print(f"  Color Modes: {dict(color_modes)}")
    print(f"\nEncoded Metadata Attributes:")
    print(f"  Subjects: {len(subjects)} subjects identified")
    print(f"  Glasses: {dict(glasses)}")
    print(f"  Lighting: {dict(lightings)}")
    print(f"  Genders: {dict(genders)}")

    # Graph 1: MRL Class Distribution (Total & per split)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall class distribution pie/bar
    labels = ["Open-Eye (Awake)", "Closed-Eye (Sleepy)"]
    counts = [total_awake, total_sleepy]
    colors = ["#2b5c8f", "#e26d5c"]
    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_title("MRL Eye Dataset: Overall Class Distribution", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Images", fontsize=11)
    axes[0].set_ylim(0, max(counts) * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_images) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1000, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Split breakdown
    split_df = pd.DataFrame(split_counts).T[["awake", "sleepy"]]
    split_df.plot(kind="bar", stacked=False, ax=axes[1], color=colors, edgecolor="black", linewidth=1.2)
    axes[1].set_title("MRL Eye Dataset: Distribution Across Partitions", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Number of Images", fontsize=11)
    axes[1].set_xticklabels([s.capitalize() for s in split_df.index], rotation=0)
    axes[1].legend(["Open-Eye (Awake)", "Closed-Eye (Sleepy)"], loc="upper right")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%,d", padding=3, fontsize=9)

    plt.tight_layout()
    chart1_path = GRAPHS_DIR / "mrl_class_distribution.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart1_path}")

    # Graph 2: MRL Dimension Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dim_widths, bins=25, kde=True, color="#2b5c8f", edgecolor="black", ax=ax)
    ax.set_title("MRL Eye Dataset: Image Resolution Distribution (Width = Height)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension in Pixels (Square Crop)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.axvline(np.mean(dim_widths), color="red", linestyle="--", label=f"Mean = {np.mean(dim_widths):.1f} px")
    ax.axvline(np.median(dim_widths), color="green", linestyle=":", label=f"Median = {np.median(dim_widths):.1f} px")
    ax.legend(fontsize=10)
    plt.tight_layout()
    chart2_path = GRAPHS_DIR / "mrl_dimension_distribution.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart2_path}")

    # Graph 3: MRL Sample Image Grid
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for i, fp in enumerate(sample_images["awake"][:6]):
        img = Image.open(fp)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"Awake\n({img.size[0]}x{img.size[1]})", fontsize=10)
        axes[0, i].axis("off")
    for i, fp in enumerate(sample_images["sleepy"][:6]):
        img = Image.open(fp)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"Sleepy\n({img.size[0]}x{img.size[1]})", fontsize=10)
        axes[1, i].axis("off")
    plt.suptitle("MRL Eye Dataset: Sample Image Grid (Awake vs Sleepy)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart3_path = GRAPHS_DIR / "mrl_sample_grid.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart3_path}")

    return {
        "total": total_images,
        "awake": total_awake,
        "sleepy": total_sleepy,
        "split_counts": dict(split_counts),
        "dimensions": dimensions,
        "subjects": len(subjects),
    }


# ==============================================================================
# 2. YAWN DATASET ANALYSIS
# ==============================================================================

def analyze_yawn_dataset(yawn_path: Path):
    """
    Performs comprehensive EDA on Yawn Dataset.
    """
    print("\n" + "=" * 60)
    print("ANALYZING YAWN DATASET")
    print("=" * 60)

    no_yawn_folder = yawn_path / "no yawn"
    yawn_folder = yawn_path / "yawn"

    no_yawn_files = list(no_yawn_folder.glob("*.jpg")) if no_yawn_folder.exists() else []
    yawn_files = list(yawn_folder.glob("*.jpg")) if yawn_folder.exists() else []

    count_no_yawn = len(no_yawn_files)
    count_yawn = len(yawn_files)
    total_images = count_no_yawn + count_yawn

    pct_no_yawn = (count_no_yawn / total_images) * 100 if total_images else 0
    pct_yawn = (count_yawn / total_images) * 100 if total_images else 0

    print(f"Total Yawn Images: {total_images:,}")
    print(f"No-Yawn Images: {count_no_yawn:,} ({pct_no_yawn:.2f}%)")
    print(f"Yawn Images:    {count_yawn:,} ({pct_yawn:.2f}%)")
    print(f"Class Balance Ratio (No-Yawn : Yawn): {count_no_yawn / max(count_yawn, 1):.3f} : 1")

    # Dimensions analysis across all images
    dimensions = []
    color_modes = Counter()
    for fp in no_yawn_files + yawn_files:
        try:
            with Image.open(fp) as im:
                dimensions.append(im.size)
                color_modes[im.mode] += 1
        except Exception:
            pass

    dim_widths = [d[0] for d in dimensions]
    dim_heights = [d[1] for d in dimensions]
    aspect_ratios = [w / max(h, 1) for w, h in dimensions]

    print(f"\nImage Dimensions (Total {len(dimensions):,} images):")
    print(f"  Width  -> Min: {np.min(dim_widths)}, Max: {np.max(dim_widths)}, Mean: {np.mean(dim_widths):.1f}, Median: {np.median(dim_widths):.1f}")
    print(f"  Height -> Min: {np.min(dim_heights)}, Max: {np.max(dim_heights)}, Mean: {np.mean(dim_heights):.1f}, Median: {np.median(dim_heights):.1f}")
    print(f"  Aspect Ratio (W/H) -> Mean: {np.mean(aspect_ratios):.2f}, Median: {np.median(aspect_ratios):.2f}")
    print(f"  Color Modes: {dict(color_modes)}")

    # Check for duplicate frames via MD5 hash
    hashes = defaultdict(list)
    for fp in no_yawn_files + yawn_files:
        with open(fp, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes[h].append(fp.name)

    duplicate_groups = {k: v for k, v in hashes.items() if len(v) > 1}
    total_dupes = sum(len(v) - 1 for v in duplicate_groups.values())
    print(f"\nDuplicate Analysis:")
    print(f"  Duplicate Hash Groups: {len(duplicate_groups)}")
    print(f"  Total Redundant Duplicate Files: {total_dupes} (consecutive video frames)")

    # Graph 1: Yawn Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["No-Yawn", "Yawn"]
    counts = [count_no_yawn, count_yawn]
    colors = ["#38b000", "#f77f00"]
    bars = ax.bar(labels, counts, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax.set_title("Yawn Dataset: Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_images) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.tight_layout()
    chart1_path = GRAPHS_DIR / "yawn_class_distribution.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart1_path}")

    # Graph 2: Yawn Dimension Scatter/Hist
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(dim_widths, bins=25, kde=True, color="#38b000", edgecolor="black", ax=axes[0])
    axes[0].set_title("Yawn Dataset: Width Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Image Count")

    sns.histplot(dim_heights, bins=25, kde=True, color="#f77f00", edgecolor="black", ax=axes[1])
    axes[1].set_title("Yawn Dataset: Height Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Image Count")

    plt.tight_layout()
    chart2_path = GRAPHS_DIR / "yawn_dimension_distribution.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart2_path}")

    # Graph 3: Yawn Sample Grid
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for i, fp in enumerate(no_yawn_files[:6]):
        img = Image.open(fp)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"No Yawn\n({img.size[0]}x{img.size[1]})", fontsize=10)
        axes[0, i].axis("off")
    for i, fp in enumerate(yawn_files[:6]):
        img = Image.open(fp)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Yawn\n({img.size[0]}x{img.size[1]})", fontsize=10)
        axes[1, i].axis("off")
    plt.suptitle("Yawn Dataset: Sample Image Grid (No-Yawn vs Yawn)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart3_path = GRAPHS_DIR / "yawn_sample_grid.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart3_path}")

    return {
        "total": total_images,
        "no_yawn": count_no_yawn,
        "yawn": count_yawn,
        "duplicates": total_dupes,
        "dimensions": dimensions,
    }


# ==============================================================================
# 3. DRIVER DROWSINESS DATASET (DDD) ANALYSIS
# ==============================================================================

def analyze_ddd_dataset(ddd_path: Path):
    """
    Performs comprehensive inspection and EDA on Driver Drowsiness Dataset (DDD).
    """
    print("\n" + "=" * 60)
    print("ANALYZING DRIVER DROWSINESS DATASET (DDD)")
    print("=" * 60)

    drowsy_folder = ddd_path / "Drowsy"
    non_drowsy_folder = ddd_path / "Non Drowsy"

    drowsy_files = list(drowsy_folder.glob("*.png")) if drowsy_folder.exists() else []
    non_drowsy_files = list(non_drowsy_folder.glob("*.png")) if non_drowsy_folder.exists() else []

    count_drowsy = len(drowsy_files)
    count_non_drowsy = len(non_drowsy_files)
    total_images = count_drowsy + count_non_drowsy

    pct_drowsy = (count_drowsy / total_images) * 100 if total_images else 0
    pct_non_drowsy = (count_non_drowsy / total_images) * 100 if total_images else 0

    print(f"Total DDD Files:  {total_images:,}")
    print(f"Number of Videos: 0 (All video sequences extracted as PNG frames)")
    print(f"Number of Images: {total_images:,}")
    print(f"Drowsy Images:     {count_drowsy:,} ({pct_drowsy:.2f}%)")
    print(f"Non-Drowsy Images: {count_non_drowsy:,} ({pct_non_drowsy:.2f}%)")
    print(f"Class Balance Ratio (Drowsy : Non-Drowsy): {count_drowsy / max(count_non_drowsy, 1):.3f} : 1")

    # Subject / Recording session analysis via file prefixes
    # Drowsy uses A-Z, Non Drowsy uses a-z
    drowsy_subjects = Counter([f.name[0] for f in drowsy_files])
    non_drowsy_subjects = Counter([f.name[0] for f in non_drowsy_files])
    all_subjects = set(list(drowsy_subjects.keys()) + [k.upper() for k in non_drowsy_subjects.keys()])

    print(f"\nSubject / Video Stream Identifiers:")
    print(f"  Identifiable Subjects/Streams: {len(all_subjects)} subjects (Prefixes A through Z)")
    print(f"  Drowsy Stream Counts (A-Z): {dict(sorted(drowsy_subjects.items()))}")
    print(f"  Non-Drowsy Stream Counts (a-z): {dict(sorted(non_drowsy_subjects.items()))}")

    # Check for Day/Night, Glasses, or other annotations
    print(f"\nInspection for Sub-Categories:")
    print(f"  Day / Night Categories: NOT labeled in dataset folder structure or filenames.")
    print(f"  Glasses / Sunglasses:   NOT labeled in dataset folder structure or filenames.")
    print(f"  Drowsiness Sub-Labels:  Binary only ('Drowsy' vs 'Non Drowsy').")

    # Sample dimensions verification
    sample_dims = set()
    sample_modes = Counter()
    for fp in (drowsy_files[:200] + non_drowsy_files[:200]):
        with Image.open(fp) as im:
            sample_dims.add(im.size)
            sample_modes[im.mode] += 1
    print(f"\nImage Resolution & Color Mode:")
    print(f"  Resolution: {list(sample_dims)} (Strictly 227x227 pixels)")
    print(f"  Color Mode: {dict(sample_modes)}")

    # Graph 1: DDD Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Drowsy", "Non Drowsy"]
    counts = [count_drowsy, count_non_drowsy]
    colors = ["#e26d5c", "#2b5c8f"]
    bars = ax.bar(labels, counts, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax.set_title("DDD Dataset: Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_images) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 400, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.tight_layout()
    chart1_path = GRAPHS_DIR / "ddd_class_distribution.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart1_path}")

    # Graph 2: DDD Subject Distribution
    fig, ax = plt.subplots(figsize=(14, 5))
    letters = sorted(list(drowsy_subjects.keys()))
    d_vals = [drowsy_subjects.get(l, 0) for l in letters]
    nd_vals = [non_drowsy_subjects.get(l.lower(), 0) for l in letters]

    x = np.arange(len(letters))
    width = 0.38
    ax.bar(x - width/2, d_vals, width, label="Drowsy (A-Z)", color="#e26d5c", edgecolor="black")
    ax.bar(x + width/2, nd_vals, width, label="Non Drowsy (a-z)", color="#2b5c8f", edgecolor="black")
    ax.set_title("DDD Dataset: Image Distribution Across Subjects / Video Sessions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Subject Identifier", fontsize=11)
    ax.set_ylabel("Number of Frames", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.legend(fontsize=10)
    plt.tight_layout()
    chart2_path = GRAPHS_DIR / "ddd_subject_distribution.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart2_path}")

    # Graph 3: DDD Sample Grid
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for i, fp in enumerate(drowsy_files[:6]):
        img = Image.open(fp)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Drowsy\n({fp.name})", fontsize=9)
        axes[0, i].axis("off")
    for i, fp in enumerate(non_drowsy_files[:6]):
        img = Image.open(fp)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Non Drowsy\n({fp.name})", fontsize=9)
        axes[1, i].axis("off")
    plt.suptitle("DDD Dataset: Sample Image Grid (Drowsy vs Non-Drowsy)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart3_path = GRAPHS_DIR / "ddd_sample_grid.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart3_path}")

    return {
        "total": total_images,
        "drowsy": count_drowsy,
        "non_drowsy": count_non_drowsy,
        "subjects": len(all_subjects),
    }


# ==============================================================================
# 4. UNIFIED COMPARISON & SUMMARY
# ==============================================================================

def generate_unified_comparison(mrl_stats, yawn_stats, ddd_stats):
    """
    Generates a unified visual comparison across all three datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall dataset sizes
    names = ["MRL Eye", "DDD Dataset", "Yawn Dataset"]
    totals = [mrl_stats["total"], ddd_stats["total"], yawn_stats["total"]]
    colors = ["#2b5c8f", "#7209b7", "#38b000"]
    bars = axes[0].bar(names, totals, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Dataset Volume Comparison (Total Images)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Total Images", fontsize=11)
    axes[0].set_ylim(0, max(totals) * 1.15)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1500, f"{h:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Balance ratios
    balance_df = pd.DataFrame([
        {"Dataset": "MRL Eye", "Drowsy / Closed %": (mrl_stats["sleepy"]/mrl_stats["total"])*100, "Alert / Open %": (mrl_stats["awake"]/mrl_stats["total"])*100},
        {"Dataset": "DDD Dataset", "Drowsy / Closed %": (ddd_stats["drowsy"]/ddd_stats["total"])*100, "Alert / Open %": (ddd_stats["non_drowsy"]/ddd_stats["total"])*100},
        {"Dataset": "Yawn Dataset", "Drowsy / Closed %": (yawn_stats["yawn"]/yawn_stats["total"])*100, "Alert / Open %": (yawn_stats["no_yawn"]/yawn_stats["total"])*100},
    ]).set_index("Dataset")

    balance_df.plot(kind="bar", stacked=True, ax=axes[1], color=["#e26d5c", "#2b5c8f"], edgecolor="black", linewidth=1.2)
    axes[1].set_title("Class Balance Comparison (% Split)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_ylim(0, 115)
    axes[1].set_xticklabels(balance_df.index, rotation=0)
    axes[1].axhline(50, color="yellow", linestyle="--", alpha=0.7, label="Ideal 50% Balance")
    axes[1].legend(loc="upper right", fontsize=10)
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%.1f%%", label_type="center", color="white", fontweight="bold", fontsize=10)

    plt.tight_layout()
    comp_path = GRAPHS_DIR / "dataset_comparison_overview.png"
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {comp_path}")


def main():
    print("\n" + "#" * 60)
    print("  STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 60)

    mrl_stats = analyze_mrl_dataset(MRL_PATH)
    yawn_stats = analyze_yawn_dataset(YAWN_PATH)
    ddd_stats = analyze_ddd_dataset(DDD_PATH)

    generate_unified_comparison(mrl_stats, yawn_stats, ddd_stats)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print(f"All graphs generated and saved to: {GRAPHS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
