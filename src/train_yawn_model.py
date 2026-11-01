"""
Train Yawn State Classification CNN.

Trains a Convolutional Neural Network on the preprocessed Yawn Dataset
to classify:
- NO_YAWN (Class 0 / Normal mouth state)
- YAWN (Class 1 / Yawning mouth state)

Key Features:
- Standardized CNN architecture:
    Input -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool
    -> Conv2D -> ReLU -> GlobalAveragePooling -> Dense -> Dropout -> Output
- Adam Optimizer & Binary Crossentropy Loss
- Metrics: Accuracy, Precision, Recall, F1-Score
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Evaluates on leak-free test set (contiguous temporal block partitioning)
- Generates training history curves and saves artifacts to models/yawn_model/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict

# Set Keras backend to torch (fully compatible with Python 3.14)
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import keras
from keras import layers, models, callbacks, optimizers, metrics, utils

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
YAWN_MODEL_DIR = getattr(config, "YAWN_MODEL_DIR", MODELS_DIR / "yawn_model")
GRAPHS_DIR = config.GRAPHS_DIR
CONFUSION_MATRIX_DIR = config.CONFUSION_MATRIX_DIR

YAWN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)


# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================

def build_yawn_cnn(input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> models.Model:
    """
    Constructs the Yawn State CNN architecture:
    Input -> Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling
    -> Conv2D -> ReLU -> GlobalAveragePooling -> Dense -> Dropout -> Output
    """
    inputs = layers.Input(shape=input_shape, name="mouth_image_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv2d_1")(inputs)
    x = layers.ReLU(name="relu_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling_1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2d_2")(x)
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling_2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_3")(x)
    x = layers.ReLU(name="relu_3")(x)

    # Dimensionality Reduction: GlobalAveragePooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense Feature Extraction
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)

    # Output: Single sigmoid unit (0 = NO_YAWN, 1 = YAWN)
    outputs = layers.Dense(1, activation="sigmoid", name="yawn_state_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="YawnStateCNN")
    return model


# ==============================================================================
# 2. DATASET LOADER (PYDATASET)
# ==============================================================================

class YawnDataset(utils.PyDataset):
    """
    High-performance PyDataset loader for Yawn Dataset images.
    Streams and preprocesses images on-the-fly without duplicate on-disk copies.
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
        return int(np.ceil(len(self.df) / self.batch_size))

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
# 3. TRAINING ROUTINE
# ==============================================================================

def train_yawn_model(epochs: int = 15,
                     batch_size: int = 64,
                     learning_rate: float = 1e-3) -> Tuple[models.Model, Dict]:
    """
    Trains the YawnStateCNN model and returns the trained model and history.
    """
    manifest_csv = PROCESSED_DIR / "yawn_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Yawn splits manifest not found at {manifest_csv}. Run src/preprocessing.py first!")

    print(f"Loading split manifest: {manifest_csv}")
    df = pd.read_csv(manifest_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Dataset partitions available:")
    print(f"  • Train: {len(train_df):,} images")
    print(f"  • Val:   {len(val_df):,} images")
    print(f"  • Test:  {len(test_df):,} images (held out for final evaluation)")

    # Data loaders
    train_ds = YawnDataset(train_df, target_size=TARGET_SIZE, batch_size=batch_size,
                           augment=True, shuffle=True)
    val_ds = YawnDataset(val_df, target_size=TARGET_SIZE, batch_size=batch_size,
                         augment=False, shuffle=False)

    # Build CNN
    model = build_yawn_cnn(input_shape=INPUT_SHAPE)
    model.summary()

    # Compile with Adam optimizer, binary crossentropy, and target metrics
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    # Callbacks
    best_model_path = YAWN_MODEL_DIR / "best_yawn_model.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
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

    cb_list = [cb_checkpoint, cb_early_stop, cb_reduce_lr]

    print("\nStarting Yawn CNN model training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1,
    )
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f} seconds ({duration/60:.2f} minutes).")

    # Save final model version
    final_model_path = YAWN_MODEL_DIR / "yawn_model.keras"
    model.save(str(final_model_path))
    print(f"Saved final trained model: {final_model_path}")
    print(f"Saved best checkpoint:    {best_model_path}")

    return model, history.history


# ==============================================================================
# 4. EVALUATION & VISUALIZATION
# ==============================================================================

def evaluate_on_test_set(model: models.Model) -> Dict:
    """
    Evaluates the trained model on unseen test set and computes metrics.
    """
    manifest_csv = PROCESSED_DIR / "yawn_splits.csv"
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"]

    print(f"\nEvaluating on Test Set ({len(test_df):,} samples)...")
    test_ds = YawnDataset(test_df, target_size=TARGET_SIZE, batch_size=64,
                          augment=False, shuffle=False)

    all_preds = []
    all_targets = []

    for i in range(len(test_ds)):
        x_b, y_b = test_ds[i]
        if len(x_b) == 0:
            continue
        preds = model.predict_on_batch(x_b)
        all_preds.extend(preds.flatten())
        all_targets.extend(y_b.flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(int)

    acc = float(np.mean(binary_preds == all_targets))
    prec = float(precision_score(all_targets, binary_preds, zero_division=0))
    rec = float(recall_score(all_targets, binary_preds, zero_division=0))
    f1 = float(f1_score(all_targets, binary_preds, zero_division=0))
    cm = confusion_matrix(all_targets, binary_preds)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS (YAWN MODEL)")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_targets, binary_preds, target_names=["NO_YAWN", "YAWN"], digits=4))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["NO_YAWN (0)", "YAWN (1)"],
                yticklabels=["NO_YAWN (0)", "YAWN (1)"])
    plt.title(f"Yawn CNN Test Confusion Matrix\nAccuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    cm_path = CONFUSION_MATRIX_DIR / "yawn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(all_targets),
    }


def plot_training_history(history: Dict, save_path: Path):
    """
    Plots training & validation curves for loss and accuracy.
    """
    epochs_range = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, history["accuracy"], label="Train Accuracy", color="#2b5c8f", lw=2)
    if "val_accuracy" in history:
        axes[0].plot(epochs_range, history["val_accuracy"], label="Val Accuracy", color="#e76f51", lw=2)
    axes[0].set_title("Yawn Model - Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(epochs_range, history["loss"], label="Train Loss", color="#2b5c8f", lw=2)
    if "val_loss" in history:
        axes[1].plot(epochs_range, history["val_loss"], label="Val Loss", color="#e76f51", lw=2)
    axes[1].set_title("Yawn Model - Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history curves: {save_path}")


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Yawn State Classification CNN")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    # Train model
    model, history = train_yawn_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Plot history
    history_plot_path = YAWN_MODEL_DIR / "yawn_training_history.png"
    plot_training_history(history, history_plot_path)

    # Evaluate on held-out test split
    metrics_res = evaluate_on_test_set(model)

    # Save metrics JSON
    metrics_path = YAWN_MODEL_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "test_metrics": metrics_res,
        }, f, indent=4)
    print(f"Saved training metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
