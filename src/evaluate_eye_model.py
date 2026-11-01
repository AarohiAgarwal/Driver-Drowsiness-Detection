"""
Evaluation Script for Trained Eye State Classification CNN.

Evaluates the primary model (best_eye_model.keras) on the held-out test partition
comprising completely disjoint human subjects (zero data leakage).

Computes:
- Overall Accuracy
- Precision (Per-class & Macro/Weighted)
- Recall (Per-class & Macro/Weighted)
- F1-Score (Per-class & Macro/Weighted)
- Confusion Matrix (Counts & Percentages)
- Comprehensive Classification Report

Outputs:
- Enhanced Confusion Matrix plot -> outputs/confusion_matrix/
- Qualitative Sample Prediction Grid (Correct + Incorrect cases) -> outputs/predictions/
- Structured JSON evaluation report -> outputs/predictions/evaluation_summary.json
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
EYE_MODEL_DIR = getattr(config, "EYE_MODEL_DIR", MODELS_DIR / "eye_model")
CONFUSION_MATRIX_DIR = config.CONFUSION_MATRIX_DIR
PREDICTIONS_DIR = config.PREDICTIONS_DIR

CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (64, 64)
CLASS_NAMES = ["OPEN", "CLOSED"]  # 0 = OPEN (Alert), 1 = CLOSED (Sleepy)


# ==============================================================================
# 1. EVALUATION CORE
# ==============================================================================

def run_evaluation(model_path: Path, max_samples: int = 3200) -> Dict:
    """
    Loads trained model and evaluates on held-out test images.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading trained eye model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model successfully loaded.\n")

    manifest_csv = PROCESSED_DIR / "mrl_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Test split manifest not found at: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"].copy()

    print(f"Held-out Test Partition:")
    print(f"  • Total Test Samples Available: {len(test_df):,}")
    print(f"  • Disjoint Test Subjects: {sorted(test_df['subject_id'].unique().tolist())}")

    # Subsample if requested for fast/focused evaluation, ensuring class balance
    if max_samples and max_samples < len(test_df):
        dfs = []
        for _, grp in test_df.groupby("label"):
            dfs.append(grp.sample(n=min(len(grp), max_samples // 2), random_state=42))
        eval_df = pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        eval_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  • Samples Selected for Evaluation: {len(eval_df):,} (Balanced 50/50)\n")

    # Predict in batches
    batch_size = 64
    all_preds_prob = []
    all_targets = []
    processed_records = []

    print("Running inference across test samples...")
    for start in range(0, len(eval_df), batch_size):
        end = min(start + batch_size, len(eval_df))
        batch_slice = eval_df.iloc[start:end]

        batch_x = []
        valid_indices = []
        for idx, row in batch_slice.iterrows():
            fp = Path(row["filepath"])
            img = preprocess_single_image(fp, target_size=TARGET_SIZE, augment=False)
            if img is not None:
                batch_x.append(img)
                valid_indices.append(idx)

        if len(batch_x) == 0:
            continue

        preds = model.predict_on_batch(np.array(batch_x, dtype=np.float32)).flatten()

        for pred_prob, idx in zip(preds, valid_indices):
            row = eval_df.iloc[idx]
            true_label = int(row["label"])
            pred_label = int(pred_prob >= 0.5)

            all_preds_prob.append(float(pred_prob))
            all_targets.append(true_label)
            processed_records.append({
                "filepath": row["filepath"],
                "subject_id": row["subject_id"],
                "true_label": true_label,
                "true_class": CLASS_NAMES[true_label],
                "pred_label": pred_label,
                "pred_class": CLASS_NAMES[pred_label],
                "prob_closed": float(pred_prob),
                "confidence": float(pred_prob if pred_label == 1 else (1.0 - pred_prob)),
                "correct": bool(true_label == pred_label),
            })

    all_preds_prob = np.array(all_preds_prob)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds_prob >= 0.5).astype(int)

    # Compute Metrics
    acc = accuracy_score(all_targets, binary_preds)
    prec_macro = precision_score(all_targets, binary_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, binary_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_targets, binary_preds, average="macro", zero_division=0)

    prec_closed = precision_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    rec_closed = recall_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    f1_closed = f1_score(all_targets, binary_preds, pos_label=1, zero_division=0)

    prec_open = precision_score(all_targets, binary_preds, pos_label=0, zero_division=0)
    rec_open = recall_score(all_targets, binary_preds, pos_label=0, zero_division=0)
    f1_open = f1_score(all_targets, binary_preds, pos_label=0, zero_division=0)

    cm = confusion_matrix(all_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()

    # Classification Report
    cr_text = classification_report(all_targets, binary_preds, target_names=CLASS_NAMES, digits=4)

    # Display Metrics in Console
    print("=" * 65)
    print("        HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Evaluated Samples: {len(all_targets):,}")
    print(f"Overall Accuracy:        {acc * 100:.2f}%\n")
    print(f"{'Metric':<18} | {'OPEN (Alert)':<14} | {'CLOSED (Sleepy)':<16} | {'Macro Avg':<10}")
    print("-" * 65)
    print(f"{'Precision':<18} | {prec_open*100:>12.2f}% | {prec_closed*100:>14.2f}% | {prec_macro*100:>8.2f}%")
    print(f"{'Recall':<18} | {rec_open*100:>12.2f}% | {rec_closed*100:>14.2f}% | {rec_macro*100:>8.2f}%")
    print(f"{'F1-Score':<18} | {f1_open*100:>12.2f}% | {f1_closed*100:>14.2f}% | {f1_macro*100:>8.2f}%")
    print("-" * 65)
    print("\nConfusion Matrix Breakdown:")
    print(f"  • True Negatives  (TN - Correctly OPEN):   {tn:,}")
    print(f"  • False Positives (FP - False Alarm):      {fp:,}")
    print(f"  • False Negatives (FN - Missed Drowsy):    {fn:,}")
    print(f"  • True Positives  (TP - Correctly CLOSED): {tp:,}\n")
    print("Full Classification Report:\n" + cr_text)

    # Visualizations
    cm_plot_path = plot_detailed_confusion_matrix(cm, acc)
    samples_plot_path = plot_prediction_samples(processed_records)

    # Package Results
    results = {
        "model_path": str(model_path),
        "total_test_samples": len(all_targets),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_score_macro": float(f1_macro),
        "per_class_metrics": {
            "OPEN": {
                "precision": float(prec_open),
                "recall": float(rec_open),
                "f1_score": float(f1_open),
                "support": int(tn + fp),
            },
            "CLOSED": {
                "precision": float(prec_closed),
                "recall": float(rec_closed),
                "f1_score": float(f1_closed),
                "support": int(fn + tp),
            },
        },
        "confusion_matrix": {
            "TN_true_open": int(tn),
            "FP_false_closed": int(fp),
            "FN_false_open": int(fn),
            "TP_true_closed": int(tp),
        },
        "artifacts_generated": {
            "confusion_matrix_plot": str(cm_plot_path),
            "sample_predictions_plot": str(samples_plot_path),
        },
    }

    summary_json_path = PREDICTIONS_DIR / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured evaluation report: {summary_json_path}")

    return results


# ==============================================================================
# 2. VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_detailed_confusion_matrix(cm: np.ndarray, accuracy: float) -> Path:
    """
    Plots an enhanced confusion matrix showing both raw counts and percentage of row/class.
    """
    total = np.sum(cm)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_pct = (cm / row_sums) * 100.0

    labels = []
    for r in range(2):
        row_labels = []
        for c in range(2):
            cnt = cm[r, c]
            pct = cm_pct[r, c]
            cell_text = f"{cnt:,}\n({pct:.1f}%)"
            row_labels.append(cell_text)
        labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(f"Eye State CNN: Confusion Matrix (Test Set)\nAccuracy: {accuracy*100:.2f}%",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Eye State", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Ground Truth State", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout()
    output_path = CONFUSION_MATRIX_DIR / "eye_model_confusion_matrix_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Detailed Confusion Matrix: {output_path}")
    return output_path


def plot_prediction_samples(records: List[Dict]) -> Path:
    """
    Creates a publication-quality visual grid showcasing:
    - Top rows: Correctly classified samples (True OPEN, True CLOSED)
    - Bottom rows: Misclassified samples (False Positives, False Negatives)
    with exact labels and confidence scores.
    """
    # Separate into categories
    correct_open = [r for r in records if r["true_label"] == 0 and r["correct"]]
    correct_closed = [r for r in records if r["true_label"] == 1 and r["correct"]]
    false_positives = [r for r in records if r["true_label"] == 0 and not r["correct"]] # Actual OPEN, Pred CLOSED
    false_negatives = [r for r in records if r["true_label"] == 1 and not r["correct"]] # Actual CLOSED, Pred OPEN

    num_cols = 4
    num_rows = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 14))

    # Row 0: Correct OPEN
    for col in range(num_cols):
        r = correct_open[col]
        img = load_and_verify_image(Path(r["filepath"]))
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(f"Actual: {r['true_class']}\nPredicted: {r['pred_class']}\nConfidence: {r['confidence']*100:.1f}%\n[CORRECT]",
                     color="#1b4332", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 1: Correct CLOSED
    for col in range(num_cols):
        r = correct_closed[col]
        img = load_and_verify_image(Path(r["filepath"]))
        ax = axes[1, col]
        ax.imshow(img)
        ax.set_title(f"Actual: {r['true_class']}\nPredicted: {r['pred_class']}\nConfidence: {r['confidence']*100:.1f}%\n[CORRECT]",
                     color="#1b4332", fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 2: False Positives (Actual OPEN -> Predicted CLOSED)
    fp_samples = false_positives[:num_cols]
    for col in range(num_cols):
        ax = axes[2, col]
        if col < len(fp_samples):
            r = fp_samples[col]
            img = load_and_verify_image(Path(r["filepath"]))
            ax.imshow(img)
            ax.set_title(f"Actual: {r['true_class']}\nPredicted: {r['pred_class']}\nConfidence: {r['confidence']*100:.1f}%\n[FALSE POSITIVE]",
                         color="#b00020", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No FP Sample", ha="center", va="center")
        ax.axis("off")

    # Row 3: False Negatives (Actual CLOSED -> Predicted OPEN)
    fn_samples = false_negatives[:num_cols]
    for col in range(num_cols):
        ax = axes[3, col]
        if col < len(fn_samples):
            r = fn_samples[col]
            img = load_and_verify_image(Path(r["filepath"]))
            ax.imshow(img)
            ax.set_title(f"Actual: {r['true_class']}\nPredicted: {r['pred_class']}\nConfidence: {r['confidence']*100:.1f}%\n[FALSE NEGATIVE]",
                         color="#b00020", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No FN Sample", ha="center", va="center")
        ax.axis("off")

    fig.text(0.5, 0.99, "Eye State CNN: Qualitative Predictions on Test Set",
             ha="center", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.88, "Correct OPEN", va="center", rotation=90, fontsize=11, fontweight="bold", color="#1b4332")
    fig.text(0.01, 0.65, "Correct CLOSED", va="center", rotation=90, fontsize=11, fontweight="bold", color="#1b4332")
    fig.text(0.01, 0.40, "False Positive (Open -> Closed)", va="center", rotation=90, fontsize=11, fontweight="bold", color="#b00020")
    fig.text(0.01, 0.16, "False Negative (Closed -> Open)", va="center", rotation=90, fontsize=11, fontweight="bold", color="#b00020")

    plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.98])
    output_path = PREDICTIONS_DIR / "eye_model_sample_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Prediction Visualizations: {output_path}")
    return output_path


# ==============================================================================
# 3. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Eye Classification CNN")
    parser.add_argument("--model-path", type=str, default=str(EYE_MODEL_DIR / "best_eye_model.keras"),
                        help="Path to trained .keras model")
    parser.add_argument("--samples", type=int, default=3200,
                        help="Number of held-out test samples to evaluate (default 3200)")
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("  STAGE 4.5: RIGOROUS EVALUATION ON HELD-OUT TEST SUBJECTS")
    print("#" * 65 + "\n")

    run_evaluation(Path(args.model_path), max_samples=args.samples)


if __name__ == "__main__":
    main()
