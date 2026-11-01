# Dataset Documentation

This directory manages raw and preprocessed data splits for the **Driver Drowsiness Detection System**.

## Physical Dataset Locations

The primary Kaggle datasets are physically stored in the root `dataset/` directory to avoid unnecessary disk duplication:

- **Driver Drowsiness Dataset (DDD)**: `dataset/DDD Dataset/`
  - Total Images: 41,793 (.png)
  - Classes: `Drowsy` (22,348), `Non Drowsy` (19,445)
  - Image Size: 227x227 RGB

- **MRL Eye Dataset**: `dataset/MRL Dataset/`
  - Total Images: 84,898 (.png)
  - Splits: `train/` (awake: 25,770, sleepy: 25,167), `val/` (awake: 8,591, sleepy: 8,389), `test/` (awake: 8,591, sleepy: 8,390)
  - Annotations: `labels.txt` (awake, sleepy) + metadata encoded in image filenames
  - Image Size: Variable grayscale crops (mode 'L')

- **Yawn Dataset**: `dataset/Yawn Dataset/`
  - Total Images: 5,119 (.jpg)
  - Classes: `no yawn` (2,591), `yawn` (2,528)
  - Image Size: Variable RGB crops

## Directory Layout

- `data/raw/`: Placeholder for intermediate raw dumps or symbolic links.
- `data/processed/`: Standardized, cropped, resized, and partitioned datasets prepared for model training and evaluation.
