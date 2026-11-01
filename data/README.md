# Dataset Documentation

This directory manages raw and preprocessed data splits for the **Driver Drowsiness Detection System**.

## Curated Sample Dataset (Included in Git)

To allow instant testing and verification without downloading 2.89 GB of images, a curated sample dataset is tracked in Git at:
- **`data/sample_dataset/`** (440 sample images across all classes, ~9.2 MB)
  - `DDD Dataset/` (Drowsy, Non Drowsy)
  - `MRL Dataset/` (train, val, test splits for awake and sleepy)
  - `Yawn Dataset/` (yawn, no yawn)

## Full Kaggle Datasets (2.89 GB, 131,810 Images)

The complete datasets are physically placed in `dataset/` (excluded from git via `.gitignore` to conform with GitHub repository size quotas):

1. **Driver Drowsiness Dataset (DDD)**: `dataset/DDD Dataset/`
   - **Kaggle Link:** [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasrummin/driver-drowsiness-dataset-ddd)
   - Total Images: 41,793 (.png) | Classes: `Drowsy` (22,348), `Non Drowsy` (19,445)
   - Kaggle CLI:
     ```bash
     kaggle datasets download -d ismailnasrummin/driver-drowsiness-dataset-ddd -p dataset/"DDD Dataset" --unzip
     ```

2. **MRL Eye Dataset**: `dataset/MRL Dataset/`
   - **Kaggle Link:** [MRL Eye Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
   - Total Images: 84,898 (.png) | Classes: `awake` (42,952), `sleepy` (41,946) across 37 subjects
   - Kaggle CLI:
     ```bash
     kaggle datasets download -d dheerajperumandla/drowsiness-dataset -p dataset/"MRL Dataset" --unzip
     ```

3. **Yawn Dataset**: `dataset/Yawn Dataset/`
   - **Kaggle Link:** [Yawn Dataset](https://www.kaggle.com/datasets/vipoooool/yawn-dataset)
   - Total Images: 5,119 (.jpg) | Classes: `no yawn` (2,591), `yawn` (2,528)
   - Kaggle CLI:
     ```bash
     kaggle datasets download -d vipoooool/yawn-dataset -p dataset/"Yawn Dataset" --unzip
     ```

## Directory Layout

- `data/sample_dataset/`: Ready-to-use sample images tracked in git.
- `data/raw/`: Placeholder for intermediate raw dumps.
- `data/processed/`: Standardized CSV splits and metadata (`mrl_splits.csv`, `ddd_splits.csv`, `yawn_splits.csv`).
- `data/alerts/`: Pre-generated native multilingual voice alert audio files (English, Hindi, Punjabi).
