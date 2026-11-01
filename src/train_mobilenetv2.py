"""
Transfer Learning with MobileNetV2 for Eye State Classification (Stage 5).

Trains a pre-trained MobileNetV2 on the MRL Eye Dataset to classify:
- OPEN EYE (Class 0 / Alert)
- CLOSED EYE (Class 1 / Sleepy)

Pipeline:
1. Load MobileNetV2 with pre-trained ImageNet weights.
2. Initially freeze all base layers.
3. Attach custom classification head:
    GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1, sigmoid)
4. Train head (Feature Extraction).
5. Fine-tune top layers with reduced learning rate.
6. Evaluate on the exact same held-out test set as the baseline CNN.
7. Benchmark empirical inference latency (ms/frame) and memory footprint.
8. Generate side-by-side comparative performance report.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict

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
from keras import layers, models, callbacks, optimizers, metrics, utils
from keras.applications import MobileNetV2

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing import preprocess_single_image

# Directories
PROCESSED_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
CNN_MODEL_PATH = MODELS_DIR / "eye_model" / "best_eye_model.keras"
MOBILENETV2_DIR = getattr(config, "MOBILENETV2_DIR", MODELS_DIR / "mobilenetv2")
GRAPHS_DIR = config.GRAPHS_DIR
CONFUSION_MATRIX_DIR = config.CONFUSION_MATRIX_DIR
PREDICTIONS_DIR = config.PREDICTIONS_DIR

MOBILENETV2_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (96, 96)
INPUT_SHAPE = (96, 96, 3)
CLASS_NAMES = ["OPEN", "CLOSED"]


# ==============================================================================
# 1. MOBILENETV2 MODEL BUILDER
# ==============================================================================

def build_mobilenetv2_transfer_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                                     freeze_base: bool = True) -> Tuple[models.Model, models.Model]:
    """
    Builds MobileNetV2 with pre-trained ImageNet weights and custom classification head:
    GlobalAveragePooling -> Dropout -> Dense(1, sigmoid).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="mobilenetv2_eye_input")
    # Preprocess inputs for MobileNetV2 (scales [0, 1] to [-1, 1])
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="eye_state_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_EyeClassifier")
    return model, base_model


# ==============================================================================
# 2. DATA LOADER
# ==============================================================================

class MRLMobileNetDataset(utils.PyDataset):
    """
    Streams and resizes images to 96x96 on-the-fly for MobileNetV2.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 batch_size: int = 64,
                 augment: bool = False,
                 shuffle: bool = True,
                 max_samples: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        if max_samples and max_samples < len(df):
            sampled_dfs = []
            for _, grp in df.groupby("label"):
                n = min(len(grp), max_samples // 2)
                sampled_dfs.append(grp.sample(n=n, random_state=42))
            self.df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(len(self.df) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        sub = self.df.iloc[batch_idx]

        batch_x = []
        batch_y = []
        for _, row in sub.iterrows():
            fp = Path(row["filepath"])
            img = preprocess_single_image(fp, target_size=self.target_size, augment=self.augment)
            if img is not None:
                batch_x.append(img)
                batch_y.append(row["label"])

        if len(batch_x) == 0:
            h, w = self.target_size
            return np.zeros((0, h, w, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32).reshape(-1, 1)


# ==============================================================================
# 3. TRAINING AND OPTIONAL FINE-TUNING
# ==============================================================================

def train_mobilenetv2_pipeline(initial_epochs: int = 4,
                               finetune_epochs: int = 3,
                               batch_size: int = 64,
                               train_samples: int = 12000,
                               val_samples: int = 3000) -> Tuple[models.Model, Dict]:
    """
    Two-stage training:
    Phase 1: Feature Extraction (base frozen)
    Phase 2: Fine-Tuning (top layers unfrozen with low LR)
    """
    manifest_csv = PROCESSED_DIR / "mrl_splits.csv"
    df = pd.read_csv(manifest_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print("\nDataset split confirmed (Identical partitions to baseline CNN):")
    print(f"  • Train: {len(train_df):,} total (using {train_samples:,} per epoch)")
    print(f"  • Val:   {len(val_df):,} total (using {val_samples:,} for validation)")
    print(f"  • Test:  {len(test_df):,} total (held-out test set)\n")

    train_ds = MRLMobileNetDataset(train_df, target_size=TARGET_SIZE, batch_size=batch_size,
                                   augment=True, shuffle=True, max_samples=train_samples)
    val_ds = MRLMobileNetDataset(val_df, target_size=TARGET_SIZE, batch_size=batch_size,
                                 augment=False, shuffle=False, max_samples=val_samples)

    # 1. Build Model with Frozen Base
    print("Building MobileNetV2 with pre-trained ImageNet weights...")
    model, base_model = build_mobilenetv2_transfer_model(input_shape=INPUT_SHAPE, freeze_base=True)
    model.summary()

    # Phase 1 Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    best_checkpoint_path = MOBILENETV2_DIR / "best_mobilenetv2.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD (BASE FROZEN)")
    print("=" * 60)
    t0 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
        verbose=1,
    )
    t_p1 = time.time() - t0
    print(f"Phase 1 finished in {t_p1:.1f}s.")

    # Phase 2: Fine-Tuning Top Layers
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING SELECTED UPPER LAYERS")
    print("=" * 60)
    # Unfreeze upper layers of MobileNetV2 (from layer 120 onwards)
    base_model.trainable = True
    fine_tune_at = 120
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"Unfrozen layers from layer {fine_tune_at} to end.")
    print(f"Trainable parameters for fine-tuning: {trainable_count:,}")

    # Recompile with small learning rate to avoid destroying pre-trained representations
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    t0_ft = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
        verbose=1,
    )
    t_p2 = time.time() - t0_ft
    print(f"Phase 2 finished in {t_p2:.1f}s.")

    # Save final model
    final_model_path = MOBILENETV2_DIR / "mobilenetv2_eye.keras"
    model.save(str(final_model_path))
    print(f"\nSaved final MobileNetV2 model: {final_model_path}")
    print(f"Saved best MobileNetV2 weights: {best_checkpoint_path}")

    # Merge histories
    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = history_p1.history[key] + history_p2.history.get(key, [])

    return model, combined_history


# ==============================================================================
# 4. BENCHMARKING & RIGOROUS MODEL COMPARISON
# ==============================================================================

def measure_inference_speed(model: models.Model, input_shape: Tuple[int, int, int], num_runs: int = 100) -> float:
    """
    Empirically benchmarks single-image inference latency on CPU (averaged over num_runs).
    """
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    # Warmup
    for _ in range(15):
        _ = model(dummy_input, training=False)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input, training=False)
        latencies.append((time.perf_counter() - start) * 1000.0)  # ms

    return float(np.mean(latencies))


def evaluate_model_on_test_set(model: models.Model,
                               target_size: Tuple[int, int],
                               test_df: pd.DataFrame,
                               max_samples: int = 3200) -> Dict:
    """
    Evaluates a given model on the identical held-out test set.
    """
    # Sample balanced test subset
    dfs = []
    for _, grp in test_df.groupby("label"):
        dfs.append(grp.sample(n=min(len(grp), max_samples // 2), random_state=42))
    eval_df = pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    all_preds = []
    all_targets = []
    batch_size = 64

    for start in range(0, len(eval_df), batch_size):
        end = min(start + batch_size, len(eval_df))
        batch_slice = eval_df.iloc[start:end]
        batch_x = []
        for _, row in batch_slice.iterrows():
            img = preprocess_single_image(Path(row["filepath"]), target_size=target_size, augment=False)
            if img is not None:
                batch_x.append(img)
                all_targets.append(int(row["label"]))
        if len(batch_x) > 0:
            preds = model.predict_on_batch(np.array(batch_x, dtype=np.float32)).flatten()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    prec_macro = precision_score(all_targets, binary_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, binary_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_targets, binary_preds, average="macro", zero_division=0)

    prec_closed = precision_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    rec_closed = recall_score(all_targets, binary_preds, pos_label=1, zero_division=0)
    f1_closed = f1_score(all_targets, binary_preds, pos_label=1, zero_division=0)

    cm = confusion_matrix(all_targets, binary_preds)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_score_macro": float(f1_macro),
        "precision_closed": float(prec_closed),
        "recall_closed": float(rec_closed),
        "f1_closed": float(f1_closed),
        "confusion_matrix": cm.tolist(),
        "total_evaluated": len(all_targets),
    }


def perform_factual_comparison(mobilenet_model: models.Model, mobilenet_history: Dict):
    """
    Performs empirical benchmark comparing Baseline CNN and MobileNetV2.
    """
    print("\n" + "=" * 65)
    print("  FACTUAL COMPARISON: BASELINE CNN vs MOBILENETV2 TRANSFER LEARNING")
    print("=" * 65)

    manifest_csv = PROCESSED_DIR / "mrl_splits.csv"
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"]

    # 1. Load Baseline CNN
    print("Loading Baseline CNN from:", CNN_MODEL_PATH)
    cnn_model = keras.models.load_model(CNN_MODEL_PATH)

    # 2. Evaluate both models on IDENTICAL test samples
    print("\nEvaluating Baseline CNN on Test Set (64x64 input)...")
    cnn_results = evaluate_model_on_test_set(cnn_model, target_size=(64, 64), test_df=test_df, max_samples=3200)

    print("Evaluating MobileNetV2 on Test Set (96x96 input)...")
    mobilenet_results = evaluate_model_on_test_set(mobilenet_model, target_size=TARGET_SIZE, test_df=test_df, max_samples=3200)

    # 3. Measure Inference Speed (Latency per frame)
    print("\nMeasuring CPU single-image inference latency (100 runs each)...")
    cnn_latency_ms = measure_inference_speed(cnn_model, input_shape=(64, 64, 3), num_runs=100)
    mobilenet_latency_ms = measure_inference_speed(mobilenet_model, input_shape=INPUT_SHAPE, num_runs=100)

    cnn_fps = 1000.0 / cnn_latency_ms
    mobilenet_fps = 1000.0 / mobilenet_latency_ms

    # 4. Measure Model Size & Parameters
    cnn_params = cnn_model.count_params()
    mobilenet_params = mobilenet_model.count_params()

    cnn_file_mb = CNN_MODEL_PATH.stat().st_size / (1024 * 1024)
    mobilenet_path = MOBILENETV2_DIR / "best_mobilenetv2.keras"
    mobilenet_file_mb = mobilenet_path.stat().st_size / (1024 * 1024)

    # 5. Display Factual Comparison Table
    comparison_table = [
        {"Model": "Baseline Custom CNN",
         "Accuracy": f"{cnn_results['accuracy']*100:.2f}%",
         "Precision (Closed)": f"{cnn_results['precision_closed']*100:.2f}%",
         "Recall (Closed)": f"{cnn_results['recall_closed']*100:.2f}%",
         "F1-Score (Closed)": f"{cnn_results['f1_closed']*100:.2f}%",
         "Inference Latency": f"{cnn_latency_ms:.2f} ms",
         "Throughput (FPS)": f"{cnn_fps:.1f} FPS",
         "Total Parameters": f"{cnn_params:,}",
         "Model File Size": f"{cnn_file_mb:.2f} MB"},

        {"Model": "MobileNetV2 (Transfer Learning)",
         "Accuracy": f"{mobilenet_results['accuracy']*100:.2f}%",
         "Precision (Closed)": f"{mobilenet_results['precision_closed']*100:.2f}%",
         "Recall (Closed)": f"{mobilenet_results['recall_closed']*100:.2f}%",
         "F1-Score (Closed)": f"{mobilenet_results['f1_closed']*100:.2f}%",
         "Inference Latency": f"{mobilenet_latency_ms:.2f} ms",
         "Throughput (FPS)": f"{mobilenet_fps:.1f} FPS",
         "Total Parameters": f"{mobilenet_params:,}",
         "Model File Size": f"{mobilenet_file_mb:.2f} MB"},
    ]

    comp_df = pd.DataFrame(comparison_table)
    print("\n" + comp_df.to_string(index=False))
    print("\n" + "=" * 65)

    # 6. Save Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Metric Bars
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    cnn_vals = [cnn_results["accuracy"]*100, cnn_results["precision_closed"]*100, cnn_results["recall_closed"]*100, cnn_results["f1_closed"]*100]
    mb_vals = [mobilenet_results["accuracy"]*100, mobilenet_results["precision_closed"]*100, mobilenet_results["recall_closed"]*100, mobilenet_results["f1_closed"]*100]

    x = np.arange(len(metrics_names))
    width = 0.35
    axes[0].bar(x - width/2, cnn_vals, width, label="Baseline CNN", color="#2b5c8f", edgecolor="black")
    axes[0].bar(x + width/2, mb_vals, width, label="MobileNetV2", color="#e26d5c", edgecolor="black")
    axes[0].set_title("Test Set Classification Metrics (%)", fontweight="bold", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].set_ylim(60, 100)
    axes[0].legend(loc="lower right")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.1f%%", padding=2, fontsize=9)

    # Latency Comparison
    lat_bars = axes[1].bar(["Baseline CNN", "MobileNetV2"], [cnn_latency_ms, mobilenet_latency_ms],
                           color=["#2b5c8f", "#e26d5c"], width=0.45, edgecolor="black")
    axes[1].set_title("Single-Image CPU Inference Latency (ms)\n[Lower is Faster]", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Milliseconds (ms)")
    for b in lat_bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.2f} ms\n({1000/h:.0f} FPS)", ha="center", fontweight="bold")

    # Parameters & Memory Footprint
    param_bars = axes[2].bar(["Baseline CNN", "MobileNetV2"], [cnn_params / 1e3, mobilenet_params / 1e3],
                             color=["#2b5c8f", "#e26d5c"], width=0.45, edgecolor="black")
    axes[2].set_title("Model Complexity (Total Parameters in Thousands)\n[Lower is Lighter]", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Parameters (K)")
    for b in param_bars:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2, h + 40, f"{h:.0f} K\n({h/1000:.2f}M)", ha="center", fontweight="bold")

    plt.tight_layout()
    comp_plot_path = GRAPHS_DIR / "model_comparison_cnn_vs_mobilenetv2.png"
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Comparative Visualizations: {comp_plot_path}")

    # 7. Save MobileNet Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(mobilenet_results["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"MobileNetV2: Confusion Matrix (Test Set)\nAccuracy: {mobilenet_results['accuracy']*100:.2f}%",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Eye State", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")
    plt.tight_layout()
    mb_cm_path = CONFUSION_MATRIX_DIR / "mobilenetv2_confusion_matrix.png"
    plt.savefig(mb_cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MobileNetV2 Confusion Matrix: {mb_cm_path}")

    # 8. Save Comparative Metrics JSON
    export_data = {
        "dataset": "MRL Eye Dataset",
        "test_samples_evaluated": cnn_results["total_evaluated"],
        "disjoint_test_subjects": sorted(test_df["subject_id"].unique().tolist()),
        "baseline_custom_cnn": {
            "input_resolution": [64, 64, 3],
            "total_parameters": cnn_params,
            "model_file_mb": round(cnn_file_mb, 2),
            "inference_latency_ms": round(cnn_latency_ms, 2),
            "fps_cpu": round(cnn_fps, 1),
            "metrics": cnn_results,
        },
        "mobilenetv2_transfer_learning": {
            "input_resolution": list(INPUT_SHAPE),
            "total_parameters": mobilenet_params,
            "model_file_mb": round(mobilenet_file_mb, 2),
            "inference_latency_ms": round(mobilenet_latency_ms, 2),
            "fps_cpu": round(mobilenet_fps, 1),
            "metrics": mobilenet_results,
        },
    }

    metrics_export_path = PREDICTIONS_DIR / "model_comparison_metrics.json"
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    print(f"Saved Comparative Metrics JSON: {metrics_export_path}\n")

    return export_data


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and Compare MobileNetV2 Eye Classifier")
    parser.add_argument("--initial-epochs", type=int, default=3, help="Phase 1 frozen epochs")
    parser.add_argument("--finetune-epochs", type=int, default=2, help="Phase 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--train-samples", type=int, default=9600, help="Train samples per epoch")
    parser.add_argument("--val-samples", type=int, default=2400, help="Validation samples")
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("  STAGE 5: TRANSFER LEARNING (MOBILENETV2) & FACTUAL BENCHMARKING")
    print("#" * 65 + "\n")

    model, history = train_mobilenetv2_pipeline(
        initial_epochs=args.initial_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )

    # Plot MobileNetV2 learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["accuracy"]) + 1)
    axes[0].plot(ep, history["accuracy"], "o-", color="#e26d5c", label="Train Accuracy")
    axes[0].plot(ep, history["val_accuracy"], "s--", color="#2b5c8f", label="Val Accuracy")
    axes[0].set_title("MobileNetV2: Accuracy History", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(ep, history["loss"], "o-", color="#e26d5c", label="Train Loss")
    axes[1].plot(ep, history["val_loss"], "s--", color="#2b5c8f", label="Val Loss")
    axes[1].set_title("MobileNetV2: Loss History", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    hist_path = GRAPHS_DIR / "mobilenetv2_training_history.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MobileNetV2 Training History: {hist_path}")

    # Perform factual comparison with Baseline CNN
    perform_factual_comparison(model, history)


if __name__ == "__main__":
    main()
