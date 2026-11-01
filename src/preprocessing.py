"""
Preprocessing Pipeline for Driver Drowsiness Detection System.

Handles:
- Task 1: Eye State Classification (MRL Eye Dataset: OPEN vs CLOSED)
- Task 2: Yawn Detection (Yawn Dataset: NO_YAWN vs YAWN)
- Task 3: Holistic Drowsiness (DDD Dataset: NON_DROWSY vs DROWSY)

Features:
- Strict Data Leakage Prevention (Subject-level & temporal sequence partitioning)
- Image Verification & Corruption Handling
- Consistent Resizing & Normalization [0, 1]
- Anatomically Realistic Augmentations (Rotation, zoom, horizontal flip, brightness)
- High-Performance On-The-Fly Generators (Zero disk duplication of 130k+ images)
- Lightweight Split Manifests (CSV) & Metadata (JSON) in data/processed/
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator

import cv2
import numpy as np
import pandas as pd
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
    PROCESSED_DIR = config.PROCESSED_DATA_DIR
    DEFAULT_IMG_SIZE = config.IMAGE_SIZE
except ImportError:
    DATASET_DIR = PROJECT_ROOT / "dataset"
    DDD_PATH = DATASET_DIR / "DDD Dataset"
    MRL_PATH = DATASET_DIR / "MRL Dataset"
    YAWN_PATH = DATASET_DIR / "Yawn Dataset"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    DEFAULT_IMG_SIZE = (224, 224)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Label Mappings (0 = Alert/Normal, 1 = Drowsy/Trigger state)
LABEL_MAPPINGS = {
    "eye": {
        "OPEN": 0,
        "CLOSED": 1,
    },
    "yawn": {
        "NO_YAWN": 0,
        "YAWN": 1,
    },
    "ddd": {
        "NON_DROWSY": 0,
        "DROWSY": 1,
    },
}

# Pre-computed Leak-Free Subject Partitions
MRL_SUBJECT_SPLITS = {
    "train": [
        "s0001", "s0002", "s0003", "s0005", "s0007", "s0008", "s0010", "s0011",
        "s0012", "s0013", "s0015", "s0018", "s0020", "s0021", "s0022", "s0025",
        "s0026", "s0028", "s0031", "s0032", "s0033", "s0034", "s0035", "s0036", "s0037"
    ],
    "val": ["s0016", "s0017", "s0019", "s0023", "s0029", "s0030"],
    "test": ["s0004", "s0006", "s0009", "s0014", "s0024", "s0027"],
}

DDD_SUBJECT_SPLITS = {
    "train": ["A", "B", "C", "D", "E", "F", "G", "I", "J", "L", "M", "O", "Q", "R", "S", "T", "X", "Z"],
    "val": ["H", "K", "P", "Y"],
    "test": ["N", "U", "V", "W"],
}


# ==============================================================================
# 1. IMAGE PROCESSING & REALISTIC AUGMENTATION
# ==============================================================================

def load_and_verify_image(filepath: Path) -> Optional[np.ndarray]:
    """
    Loads an image from disk using OpenCV, verifying integrity and skipping corrupted files.
    Returns RGB image array or None if corrupt.
    """
    try:
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        # Read with OpenCV
        img_bgr = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception:
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Resizes image to target (Height, Width) consistently.
    """
    h, w = target_size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities to floating point [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


def apply_realistic_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Applies realistic augmentations appropriate for driver monitoring:
    - Small rotation: [-10, +10] degrees (natural slight head tilt)
    - Zoom: [0.90, 1.10]
    - Horizontal Flip: 50% probability (eyes/mouths/face are horizontally symmetric)
    - Brightness variation: [-25, +25] pixel shift
    - Contrast variation: [0.85, 1.15]
    
    NOTE: Vertical flips and extreme distortions are strictly prohibited
    as they produce unrealistic eye/mouth anatomies.
    """
    h, w = image.shape[:2]

    # 1. Horizontal Flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 2. Small Rotation & Zoom
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.92, 1.08)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3. Brightness & Contrast Adjustment
    brightness = random.uniform(-25, 25)
    contrast = random.uniform(0.85, 1.15)
    img_float = image.astype(np.float32) * contrast + brightness
    image = np.clip(img_float, 0, 255).astype(np.uint8)

    return image


def preprocess_single_image(filepath: Path,
                            target_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                            augment: bool = False) -> Optional[np.ndarray]:
    """
    Complete single-image preprocessing pipeline:
    Load -> Verify -> (Augment) -> Resize -> Normalize.
    """
    img = load_and_verify_image(filepath)
    if img is None:
        return None

    if augment:
        img = apply_realistic_augmentation(img)

    img = resize_image(img, target_size=target_size)
    img_norm = normalize_image(img)
    return img_norm


# ==============================================================================
# 2. DATASET MANIFEST GENERATION (DATA LEAKAGE PREVENTION)
# ==============================================================================

def build_mrl_manifest(mrl_path: Path) -> pd.DataFrame:
    """
    Builds MRL Eye dataset manifest with subject-stratified leak-free partitioning.
    Zero subject overlap between train, val, and test.
    """
    print("Building MRL Eye dataset split manifest...")
    records = []
    
    subj_to_split = {}
    for sp, subjs in MRL_SUBJECT_SPLITS.items():
        for s in subjs:
            subj_to_split[s] = sp

    # Traverse all images
    for p in mrl_path.rglob("*.png"):
        parts = p.stem.split("_")
        if len(parts) >= 8:
            subj_id = parts[0]
            # State: 1 = Awake (OPEN), 0 = Sleepy (CLOSED)
            raw_state = parts[4]
            if raw_state == "1":
                class_name = "OPEN"
                label = LABEL_MAPPINGS["eye"]["OPEN"]
            else:
                class_name = "CLOSED"
                label = LABEL_MAPPINGS["eye"]["CLOSED"]

            split = subj_to_split.get(subj_id, "train")

            records.append({
                "filepath": str(p),
                "relative_path": str(p.relative_to(mrl_path)),
                "subject_id": subj_id,
                "class_name": class_name,
                "label": label,
                "split": split,
            })

    df = pd.DataFrame(records)
    print(f"MRL Eye Manifest: {len(df):,} total samples.")
    return df


def build_ddd_manifest(ddd_path: Path) -> pd.DataFrame:
    """
    Builds Driver Drowsiness Dataset (DDD) manifest with subject/stream leak-free partitioning.
    Zero driver/stream overlap between train, val, and test.
    """
    print("Building DDD dataset split manifest...")
    records = []

    stream_to_split = {}
    for sp, streams in DDD_SUBJECT_SPLITS.items():
        for st in streams:
            stream_to_split[st] = sp

    # Drowsy folder
    drowsy_folder = ddd_path / "Drowsy"
    if drowsy_folder.exists():
        for p in drowsy_folder.glob("*.png"):
            stream_id = p.name[0].upper()
            split = stream_to_split.get(stream_id, "train")
            records.append({
                "filepath": str(p),
                "relative_path": str(p.relative_to(ddd_path)),
                "subject_id": stream_id,
                "class_name": "DROWSY",
                "label": LABEL_MAPPINGS["ddd"]["DROWSY"],
                "split": split,
            })

    # Non Drowsy folder
    non_drowsy_folder = ddd_path / "Non Drowsy"
    if non_drowsy_folder.exists():
        for p in non_drowsy_folder.glob("*.png"):
            stream_id = p.name[0].upper()
            split = stream_to_split.get(stream_id, "train")
            records.append({
                "filepath": str(p),
                "relative_path": str(p.relative_to(ddd_path)),
                "subject_id": stream_id,
                "class_name": "NON_DROWSY",
                "label": LABEL_MAPPINGS["ddd"]["NON_DROWSY"],
                "split": split,
            })

    df = pd.DataFrame(records)
    print(f"DDD Manifest: {len(df):,} total samples.")
    return df


def build_yawn_manifest(yawn_path: Path) -> pd.DataFrame:
    """
    Builds Yawn Dataset manifest using contiguous temporal block partitioning.
    Since individual frame filenames lack subject IDs, contiguous chunking prevents
    adjacent video frames from leaking between train and test.
    """
    print("Building Yawn dataset split manifest...")
    records = []

    for category, class_name, label in [
        ("no yawn", "NO_YAWN", LABEL_MAPPINGS["yawn"]["NO_YAWN"]),
        ("yawn", "YAWN", LABEL_MAPPINGS["yawn"]["YAWN"]),
    ]:
        folder = yawn_path / category
        if not folder.exists():
            continue
        
        # Sort files numerically to respect temporal recording sequence
        files = sorted(list(folder.glob("*.jpg")), key=lambda f: int(f.stem) if f.stem.isdigit() else str(f.stem))
        n = len(files)
        
        # 70% Train, 15% Val, 15% Test contiguous chunks
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        for i, fp in enumerate(files):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            records.append({
                "filepath": str(fp),
                "relative_path": str(fp.relative_to(yawn_path)),
                "subject_id": f"sequence_chunk_{split}",
                "class_name": class_name,
                "label": label,
                "split": split,
            })

    df = pd.DataFrame(records)
    print(f"Yawn Manifest: {len(df):,} total samples.")
    return df


# ==============================================================================
# 3. ON-THE-FLY GENERATOR (ZERO DISK DUPLICATION)
# ==============================================================================

class DrowsinessDatasetGenerator:
    """
    Memory-efficient on-the-fly batch generator.
    Loads raw images directly, applies augmentation and normalization dynamically,
    preventing disk bloat while providing high throughput.
    """
    def __init__(self,
                 df_manifest: pd.DataFrame,
                 target_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 batch_size: int = 32,
                 augment: bool = False,
                 shuffle: bool = True):
        self.df = df_manifest.reset_index(drop=True)
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]

        batch_x = []
        batch_y = []

        for idx in batch_indices:
            row = self.df.iloc[idx]
            fp = Path(row["filepath"])
            img = preprocess_single_image(fp, target_size=self.target_size, augment=self.augment)
            if img is not None:
                batch_x.append(img)
                batch_y.append(row["label"])

        if len(batch_x) == 0:
            h, w = self.target_size
            return np.zeros((0, h, w, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.int32)

    def flow(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Infinite generator for model training/validation."""
        while True:
            for b in range(len(self)):
                yield self.get_batch(b)
            self.on_epoch_end()


# ==============================================================================
# 4. PIPELINE ORCHESTRATION & METADATA EXPORT
# ==============================================================================

def execute_preprocessing_pipeline():
    """
    Executes the complete preprocessing pipeline, verifies all datasets,
    generates leak-free split manifests, and saves structured metadata.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: PREPROCESSING & DATA LEAKAGE PREVENTION")
    print("=" * 60 + "\n")

    # 1. Generate Manifests
    mrl_df = build_mrl_manifest(MRL_PATH)
    yawn_df = build_yawn_manifest(YAWN_PATH)
    ddd_df = build_ddd_manifest(DDD_PATH)

    # 2. Save CSV Manifests
    mrl_csv = PROCESSED_DIR / "mrl_splits.csv"
    yawn_csv = PROCESSED_DIR / "yawn_splits.csv"
    ddd_csv = PROCESSED_DIR / "ddd_splits.csv"

    mrl_df.to_csv(mrl_csv, index=False)
    yawn_df.to_csv(yawn_csv, index=False)
    ddd_df.to_csv(ddd_csv, index=False)

    print(f"\nSaved Split Manifests to: {PROCESSED_DIR}")
    print(f"  - {mrl_csv.name} ({len(mrl_df):,} rows)")
    print(f"  - {yawn_csv.name} ({len(yawn_df):,} rows)")
    print(f"  - {ddd_csv.name} ({len(ddd_df):,} rows)")

    # 3. Compile Preprocessing Summary Metadata
    def get_summary(df: pd.DataFrame, label_map: Dict[str, int]) -> Dict:
        summary = {
            "total_samples": len(df),
            "splits": {},
            "class_distribution": {},
        }
        for sp in ["train", "val", "test"]:
            sub_df = df[df["split"] == sp]
            class_counts = sub_df["class_name"].value_counts().to_dict()
            summary["splits"][sp] = {
                "total": len(sub_df),
                "classes": class_counts,
            }
        summary["class_distribution"] = df["class_name"].value_counts().to_dict()
        summary["label_encoding"] = label_map
        return summary

    metadata = {
        "pipeline_version": "1.0",
        "image_size": {
            "height": DEFAULT_IMG_SIZE[0],
            "width": DEFAULT_IMG_SIZE[1],
            "channels": 3,
        },
        "normalization": {
            "range": "[0.0, 1.0]",
            "method": "pixel / 255.0",
        },
        "augmentation_policy": {
            "rotation_range": "[-10, +10] degrees",
            "zoom_range": "[0.92, 1.08]",
            "horizontal_flip": "50% probability",
            "brightness_shift": "[-25, +25] pixel delta",
            "contrast_range": "[0.85, 1.15]",
            "vertical_flip": False,
        },
        "tasks": {
            "task_1_eye_classification": {
                "dataset": "MRL Eye Dataset",
                "leakage_prevention": "Subject-level partitioning (37 subjects split into 25 train, 6 val, 6 test with 0% overlap)",
                "summary": get_summary(mrl_df, LABEL_MAPPINGS["eye"]),
            },
            "task_2_yawn_classification": {
                "dataset": "Yawn Dataset",
                "leakage_prevention": "Contiguous temporal block sequence partitioning (prevents consecutive video frame leakage)",
                "summary": get_summary(yawn_df, LABEL_MAPPINGS["yawn"]),
            },
            "task_3_ddd_classification": {
                "dataset": "Driver Drowsiness Dataset (DDD)",
                "leakage_prevention": "Subject/Stream letter prefix partitioning (26 subjects split into 18 train, 4 val, 4 test with 0% overlap)",
                "summary": get_summary(ddd_df, LABEL_MAPPINGS["ddd"]),
            },
        },
    }

    metadata_path = PROCESSED_DIR / "preprocessing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Preprocessing Metadata: {metadata_path}\n")

    # 4. Print Summary Tables
    print("=" * 65)
    print(f"{'TASK / DATASET':<28} | {'TRAIN':<9} | {'VAL':<8} | {'TEST':<8} | {'TOTAL':<8}")
    print("=" * 65)
    for task_key, task_data in metadata["tasks"].items():
        name = task_data["dataset"]
        s = task_data["summary"]
        tr = f"{s['splits']['train']['total']:,}"
        va = f"{s['splits']['val']['total']:,}"
        te = f"{s['splits']['test']['total']:,}"
        tot = f"{s['total_samples']:,}"
        print(f"{name:<28} | {tr:<9} | {va:<8} | {te:<8} | {tot:<8}")
    print("=" * 65)

    print("\nDetailed Class Distribution per Split:")
    for task_key, task_data in metadata["tasks"].items():
        name = task_data["dataset"]
        print(f"\n--- {name} ---")
        s = task_data["summary"]
        for sp in ["train", "val", "test"]:
            cls_str = ", ".join([f"{k}: {v:,}" for k, v in s['splits'][sp]['classes'].items()])
            print(f"  {sp.upper():<5} ({s['splits'][sp]['total']:,} samples): {cls_str}")

    # 5. Quick Test of Generator Pipeline
    print("\nValidating Generator Pipeline on a sample batch...")
    train_mrl_df = mrl_df[mrl_df["split"] == "train"]
    gen = DrowsinessDatasetGenerator(train_mrl_df, target_size=(224, 224), batch_size=8, augment=True)
    batch_x, batch_y = gen.get_batch(0)
    print(f"  Batch X shape: {batch_x.shape} (dtype={batch_x.dtype}, range=[{batch_x.min():.2f}, {batch_x.max():.2f}])")
    print(f"  Batch y shape: {batch_y.shape} (values={batch_y})")
    print("Generator successfully verified!\n")

    return metadata


if __name__ == "__main__":
    execute_preprocessing_pipeline()
