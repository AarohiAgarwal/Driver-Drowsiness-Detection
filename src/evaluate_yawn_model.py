"""
Evaluation Script for Trained Yawn State Classification CNN.

Evaluates the primary yawn model (best_yawn_model.keras) on the held-out test partition
comprising contiguous temporal block partitioning (zero data leakage).

Computes:
- Overall Accuracy
- Precision (Per-class & Macro/Weighted)
- Recall (Per-class & Macro/Weighted)
- F1-Score (Per-class & Macro/Weighted)
- Confusion Matrix (Counts & Percentages)
- Comprehensive Classification Report

Outputs:
- Enhanced Confusion Matrix plot -> outputs/confusion_matrix/yawn_confusion_matrix.png
- Qualitative Sample Prediction Grid (Correct + Incorrect cases) -> outputs/predictions/yawn_sample_predictions.png
- Structured JSON evaluation report -> outputs/predictions/yawn_evaluation_summary.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import keras

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing import preprocess_single_image, load_and_verify_image

# Directories
PROCESSED_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
YAWN_MODEL_DIR = getattr(config, "YAWN_MODEL_DIR", MODELS_DIR / "yawn_model")
CONFUSION_MATRIX_DIR = config.CONFUSION_MATRIX_DIR
PREDICTIONS_DIR = config.PREDICTIONS_DIR

CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (64, 64)
CLASS_NAMES = ["NO_YAWN", "YAWN"]  # 0 = NO_YAWN, 1 = YAWN


# ==============================================================================
# 1. EVALUATION CORE
# ==============================================================================

def run_evaluation(model_path: Path) -> Dict:
    """
    Loads trained yawn model and evaluates on held-out test images.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading trained yawn model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model successfully loaded.\n")

    manifest_csv = PROCESSED_DIR / "yawn_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Test split manifest not found at: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"Held-out Test Partition:")
    print(f"  • Total Test Samples: {len(test_df):,}")
    print(f"  • Class Distribution: {dict(test_df['class_name'].value_counts())}\n")

    # Predict in batches
    batch_size = 64
    all_preds_prob = []
    all_targets = []
    processed_records = []

    print("Running inference across test samples...")
    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        batch_slice = test_df.iloc[start:end]

        batch_imgs = []
        batch_labels = []
        valid_indices = []

        for idx, row in batch_slice.iterrows():
            fp = Path(row["filepath"])
            img = preprocess_single_image(fp, target_size=TARGET_SIZE, augment=False)
            if img is not None:
                batch_imgs.append(img)
                batch_labels.append(row["label"])
                valid_indices.append(idx)

        if not batch_imgs:
            continue

        batch_tensor = np.array(batch_imgs, dtype=np.float32)
        preds = model(batch_tensor, training=False)
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        elif hasattr(preds, "numpy"):
            preds = preds.numpy()
        probs = preds.flatten()

        all_preds_prob.extend(probs.tolist())
        all_targets.extend(batch_labels)

        for i, idx in enumerate(valid_indices):
            processed_records.append({
                "filepath": test_df.loc[idx, "filepath"],
                "target": batch_labels[i],
                "pred_prob": float(probs[i]),
                "pred_class": int(probs[i] >= 0.5),
            })

    all_preds_prob = np.array(all_preds_prob)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds_prob >= 0.5).astype(int)

    # Core Metrics
    accuracy = float(accuracy_score(all_targets, binary_preds))
    precision_macro = float(precision_score(all_targets, binary_preds, average="macro", zero_division=0))
    recall_macro = float(recall_score(all_targets, binary_preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(all_targets, binary_preds, average="macro", zero_division=0))

    precision_per_class = precision_score(all_targets, binary_preds, average=None, zero_division=0).tolist()
    recall_per_class = recall_score(all_targets, binary_preds, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(all_targets, binary_preds, average=None, zero_division=0).tolist()

    cm = confusion_matrix(all_targets, binary_preds)
    report_dict = classification_report(all_targets, binary_preds, target_names=CLASS_NAMES, output_dict=True)

    print("=" * 65)
    print("           YAWN CLASSIFIER - TEST SET PERFORMANCE")
    print("=" * 65)
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {precision_macro * 100:.2f}%")
    print(f"Macro Recall:      {recall_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%\n")

    print("Per-Class Breakdown:")
    for idx, cname in enumerate(CLASS_NAMES):
        print(f"  • {cname:<8} -> Precision: {precision_per_class[idx]*100:6.2f}% | "
              f"Recall: {recall_per_class[idx]*100:6.2f}% | "
              f"F1-Score: {f1_per_class[idx]*100:6.2f}%")

    print("\nConfusion Matrix:")
    print(f"                Predicted NO_YAWN   Predicted YAWN")
    print(f"Actual NO_YAWN       {cm[0, 0]:<19} {cm[0, 1]}")
    print(f"Actual YAWN          {cm[1, 0]:<19} {cm[1, 1]}")
    print("=" * 65 + "\n")

    # Plot Confusion Matrix
    plot_enhanced_confusion_matrix(cm, accuracy, f1_macro)

    # Plot Sample Predictions Grid
    plot_sample_predictions(processed_records)

    # Package Results
    results = {
        "model_path": str(model_path),
        "total_test_samples": len(all_targets),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1": f1_per_class[i],
            } for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    results_path = PREDICTIONS_DIR / "yawn_evaluation_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation report saved to: {results_path}")

    return results


# ==============================================================================
# 2. VISUALIZATIONS
# ==============================================================================

def plot_enhanced_confusion_matrix(cm: np.ndarray, accuracy: float, f1: float):
    """
    Plots normalized and count-annotated confusion matrix.
    """
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap="Purples", cbar=True, ax=ax,
                linewidths=1.5, linecolor="#ffffff")

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if count > cm.max() * 0.55 else "#222222"
            ax.text(j + 0.5, i + 0.5, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color,
                    fontsize=12, fontweight="bold")

    ax.set_title(f"Yawn CNN Confusion Matrix (Held-out Test Set)\nAccuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10, rotation=0)

    plt.tight_layout()
    out_file = CONFUSION_MATRIX_DIR / "yawn_confusion_matrix.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved: {out_file}")


def plot_sample_predictions(records: List[Dict], num_samples: int = 16):
    """
    Creates a grid of sample test predictions showing:
    - Actual vs Predicted Label
    - Model Confidence %
    - Green border for Correct, Red border for Incorrect
    """
    records_df = pd.DataFrame(records)

    correct_df = records_df[records_df["target"] == records_df["pred_class"]]
    incorrect_df = records_df[records_df["target"] != records_df["pred_class"]]

    n_incorrect = min(len(incorrect_df), num_samples // 2)
    n_correct = num_samples - n_incorrect

    sample_sub = pd.concat([
        correct_df.sample(n=min(len(correct_df), n_correct), random_state=42),
        incorrect_df.sample(n=n_incorrect, random_state=42) if n_incorrect > 0 else pd.DataFrame()
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes = axes.flatten()

    for idx, row in sample_sub.iterrows():
        if idx >= len(axes):
            break
        ax = axes[idx]
        img = load_and_verify_image(Path(row["filepath"]))
        if img is None:
            ax.axis("off")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target_name = CLASS_NAMES[row["target"]]
        pred_name = CLASS_NAMES[row["pred_class"]]
        prob = row["pred_prob"]
        conf = prob * 100 if row["pred_class"] == 1 else (1.0 - prob) * 100
        is_correct = (row["target"] == row["pred_class"])

        border_color = "#2a9d8f" if is_correct else "#e76f51"

        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3.5)

        title_text = f"Act: {target_name}\nPred: {pred_name}\nConf: {conf:.1f}%"
        status_label = "CORRECT" if is_correct else "INCORRECT"
        ax.set_title(f"[{status_label}]\n{title_text}", fontsize=9,
                     fontweight="bold", color=border_color)

    for i in range(len(sample_sub), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Qualitative Test Set Sample Predictions (Yawn CNN)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_file = PREDICTIONS_DIR / "yawn_sample_predictions.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Sample predictions plot saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Yawn State CNN on Test Split")
    parser.add_argument("--model", type=str,
                        default=str(YAWN_MODEL_DIR / "best_yawn_model.keras"),
                        help="Path to trained yawn model file")
    args = parser.parse_args()

    run_evaluation(Path(args.model))


if __name__ == "__main__":
    main()
