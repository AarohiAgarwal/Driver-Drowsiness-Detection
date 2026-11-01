import os
import shutil
from pathlib import Path

BASE_DIR = Path("D:/Projects_Aarohi/Driver Drowsiness Detection")
SRC_DATASET = BASE_DIR / "dataset"
DEST_DATASET = BASE_DIR / "data" / "sample_dataset"

def copy_samples(src_dir: Path, dest_dir: Path, n: int = 60):
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            shutil.copy2(f, dest_dir / f.name)
            count += 1
            if count >= n:
                break
    print(f"Copied {count} files to {dest_dir.relative_to(BASE_DIR)}")

# 1. DDD Dataset
copy_samples(SRC_DATASET / "DDD Dataset" / "Drowsy", DEST_DATASET / "DDD Dataset" / "Drowsy", 60)
copy_samples(SRC_DATASET / "DDD Dataset" / "Non Drowsy", DEST_DATASET / "DDD Dataset" / "Non Drowsy", 60)

# 2. MRL Dataset
for split in ["train", "val", "test"]:
    n_samples = 60 if split == "train" else 20
    copy_samples(SRC_DATASET / "MRL Dataset" / split / "awake", DEST_DATASET / "MRL Dataset" / split / "awake", n_samples)
    copy_samples(SRC_DATASET / "MRL Dataset" / split / "sleepy", DEST_DATASET / "MRL Dataset" / split / "sleepy", n_samples)

if (SRC_DATASET / "MRL Dataset" / "labels.txt").exists():
    shutil.copy2(SRC_DATASET / "MRL Dataset" / "labels.txt", DEST_DATASET / "MRL Dataset" / "labels.txt")
if (SRC_DATASET / "MRL Dataset" / "readme.md").exists():
    shutil.copy2(SRC_DATASET / "MRL Dataset" / "readme.md", DEST_DATASET / "MRL Dataset" / "readme.md")

# 3. Yawn Dataset
copy_samples(SRC_DATASET / "Yawn Dataset" / "yawn", DEST_DATASET / "Yawn Dataset" / "yawn", 60)
copy_samples(SRC_DATASET / "Yawn Dataset" / "no yawn", DEST_DATASET / "Yawn Dataset" / "no yawn", 60)

print("\nSample dataset creation complete!")
