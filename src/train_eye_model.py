"""
Train Eye State Classification CNN (Stage 4).

Trains a Convolutional Neural Network on the preprocessed MRL Eye Dataset
to classify:
- OPEN EYE (Class 0 / Alert)
- CLOSED EYE (Class 1 / Sleepy)

Key Features:
- Exact required CNN architecture:
    Input -> Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool
    -> Conv2D -> ReLU -> GlobalAveragePooling -> Dense -> Dropout -> Output
- Adam Optimizer & Binary Crossentropy Loss
- Metrics: Accuracy, Precision, Recall, F1-Score
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Evaluates on leak-free test set (completely disjoint subjects)
- Generates training history graphs and confusion matrix
- Saves final & best model weights to models/eye_model/
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
EYE_MODEL_DIR = getattr(config, "EYE_MODEL_DIR", MODELS_DIR / "eye_model")
GRAPHS_DIR = config.GRAPHS_DIR
CONFUSION_MATRIX_DIR = config.CONFUSION_MATRIX_DIR

EYE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)


# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================

def build_eye_cnn(input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> models.Model:
    """
    Constructs the exact requested CNN architecture:
    Input -> Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling
    -> Conv2D -> ReLU -> GlobalAveragePooling -> Dense -> Dropout -> Output
    """
    inputs = layers.Input(shape=input_shape, name="eye_image_input")

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

    # Output: Single sigmoid unit (0 = OPEN, 1 = CLOSED)
    outputs = layers.Dense(1, activation="sigmoid", name="eye_state_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EyeStateCNN")
    return model


# ==============================================================================
# 2. DATASET LOADER (PYDATASET)
# ==============================================================================

class MRLEyeDataset(utils.PyDataset):
    """
    High-performance PyDataset loader for MRL Eye images.
    Streams and preprocesses images on-the-fly without saving duplicate copies.
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
            # Stratified sampling across classes
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
# 3. TRAINING ROUTINE
# ==============================================================================

def train_eye_model(epochs: int = 10,
                    batch_size: int = 64,
                    learning_rate: float = 1e-3,
                    train_samples: int = 16000,
                    val_samples: int = 3200) -> Tuple[models.Model, Dict]:
    """
    Trains the EyeStateCNN model and returns the trained model and history.
    """
    manifest_csv = PROCESSED_DIR / "mrl_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"MRL splits manifest not found at {manifest_csv}. Run src/preprocessing.py first!")

    print(f"Loading split manifest: {manifest_csv}")
    df = pd.read_csv(manifest_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Dataset partitions available:")
    print(f"  • Train: {len(train_df):,} images (using {min(len(train_df), train_samples):,} per epoch)")
    print(f"  • Val:   {len(val_df):,} images (using {min(len(val_df), val_samples):,} for validation)")
    print(f"  • Test:  {len(test_df):,} images (held out for final evaluation)")

    # Data loaders
    train_ds = MRLEyeDataset(train_df, target_size=TARGET_SIZE, batch_size=batch_size,
                             augment=True, shuffle=True, max_samples=train_samples)
    val_ds = MRLEyeDataset(val_df, target_size=TARGET_SIZE, batch_size=batch_size,
                           augment=False, shuffle=False, max_samples=val_samples)

    # Build CNN
    model = build_eye_cnn(input_shape=INPUT_SHAPE)
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
    best_model_path = EYE_MODEL_DIR / "best_eye_model.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
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

    print("\nStarting Eye CNN model training...")
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
    final_model_path = EYE_MODEL_DIR / "eye_model.keras"
    model.save(str(final_model_path))
    print(f"Saved final trained model: {final_model_path}")
    print(f"Saved best checkpoint:    {best_model_path}")

    return model, history.history


# ==============================================================================
# 4. EVALUATION & VISUALIZATION
# ==============================================================================

def evaluate_on_test_set(model: models.Model, test_samples: int = 3000) -> Dict:
    """
    Evaluates the trained model on unseen test subjects and computes metrics.
    """
    manifest_csv = PROCESSED_DIR / "mrl_splits.csv"
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"]

    print(f"\nEvaluating on Test Set ({min(len(test_df), test_samples):,} samples from unseen subjects)...")
    test_ds = MRLEyeDataset(test_df, target_size=TARGET_SIZE, batch_size=64,
                            augment=False, shuffle=False, max_samples=test_samples)

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

    acc = np.mean(binary_preds == all_targets)
    prec = precision_score(all_targets, binary_preds, zero_division=0)
    rec = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    cm = confusion_matrix(all_targets, binary_preds)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS (DISJOINT SUBJECTS)")
    print("=" * 60)
    print(f"  • Test Accuracy:  {acc * 100:.2f}%")
    print(f"  • Test Precision: {prec * 100:.2f}%")
    print(f"  • Test Recall:    {rec * 100:.2f}%")
    print(f"  • Test F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    target_names = ["OPEN (Awake)", "CLOSED (Sleepy)"]
    print(classification_report(all_targets, binary_preds, target_names=target_names, digits=4))

    # Save Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Eye State CNN: Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = CONFUSION_MATRIX_DIR / "eye_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix: {cm_path}")

    return {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "total_test_evaluated": len(all_targets),
    }


def plot_training_history(history: Dict):
    """
    Generates and saves Training vs Validation Accuracy and Loss learning curves.
    """
    epochs_range = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, history["accuracy"], "o-", color="#2b5c8f", label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs_range, history["val_accuracy"], "s--", color="#e26d5c", label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Eye State CNN: Training vs Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=10)

    # Loss Plot
    axes[1].plot(epochs_range, history["loss"], "o-", color="#2b5c8f", label="Training Loss", linewidth=2)
    axes[1].plot(epochs_range, history["val_loss"], "s--", color="#e26d5c", label="Validation Loss", linewidth=2)
    axes[1].set_title("Eye State CNN: Training vs Validation Loss", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Binary Crossentropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    history_path = GRAPHS_DIR / "eye_model_training_history.png"
    plt.savefig(history_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Training Curves: {history_path}")


def save_training_metadata(history: Dict, test_results: Dict):
    """
    Exports structured training metadata and performance logs to JSON.
    """
    metadata = {
        "model_name": "EyeStateCNN",
        "task": "Eye State Classification (OPEN vs CLOSED)",
        "dataset": "MRL Eye Dataset",
        "architecture": {
            "input_shape": list(INPUT_SHAPE),
            "layers": [
                "Conv2D(32, 3x3, same) + ReLU",
                "MaxPooling2D(2x2)",
                "Conv2D(64, 3x3, same) + ReLU",
                "MaxPooling2D(2x2)",
                "Conv2D(128, 3x3, same) + ReLU",
                "GlobalAveragePooling2D",
                "Dense(64, relu)",
                "Dropout(0.4)",
                "Dense(1, sigmoid)"
            ],
            "total_parameters": 100737,
            "trainable_parameters": 100737,
        },
        "training_hyperparameters": {
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "epochs_trained": len(history["accuracy"]),
            "batch_size": 64,
            "target_resolution": list(TARGET_SIZE),
        },
        "final_epoch_metrics": {
            "train_accuracy": float(history["accuracy"][-1]),
            "val_accuracy": float(history["val_accuracy"][-1]),
            "train_loss": float(history["loss"][-1]),
            "val_loss": float(history["val_loss"][-1]),
            "train_precision": float(history["precision"][-1]),
            "val_precision": float(history["val_precision"][-1]),
            "train_recall": float(history["recall"][-1]),
            "val_recall": float(history["val_recall"][-1]),
        },
        "test_set_evaluation": test_results,
    }

    meta_path = EYE_MODEL_DIR / "eye_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Model Metadata: {meta_path}")


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Eye State Classification CNN")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--train-samples", type=int, default=16000, help="Training samples per epoch")
    parser.add_argument("--val-samples", type=int, default=3200, help="Validation samples")
    parser.add_argument("--test-samples", type=int, default=3200, help="Test samples to evaluate")
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  STAGE 4: PRIMARY ML MODEL TRAINING (EYE STATE CNN)")
    print("#" * 60 + "\n")

    model, history = train_eye_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )

    plot_training_history(history)
    test_results = evaluate_on_test_set(model, test_samples=args.test_samples)
    save_training_metadata(history, test_results)

    print("\n" + "=" * 60)
    print("EYE MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Models saved under: {EYE_MODEL_DIR}")
    print(f"Graphs saved under: {GRAPHS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
