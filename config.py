"""
Configuration settings for Driver Drowsiness Detection Project.
All paths are defined relative to the project root directory using pathlib.
"""

from pathlib import Path

# Base project directory (resolved dynamically relative to this config file)
BASE_DIR = Path(__file__).resolve().parent

# Data directory paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Existing dataset paths (physically uploaded to dataset/ inside the project)
DATASET_DIR = BASE_DIR / "dataset"
DDD_DATASET_DIR = DATASET_DIR / "DDD Dataset"
MRL_DATASET_DIR = DATASET_DIR / "MRL Dataset"
YAWN_DATASET_DIR = DATASET_DIR / "Yawn Dataset"

# Curated sample dataset paths (tracked in git for immediate out-of-the-box execution)
SAMPLE_DATASET_DIR = DATA_DIR / "sample_dataset"
SAMPLE_DDD_DIR = SAMPLE_DATASET_DIR / "DDD Dataset"
SAMPLE_MRL_DIR = SAMPLE_DATASET_DIR / "MRL Dataset"
SAMPLE_YAWN_DIR = SAMPLE_DATASET_DIR / "Yawn Dataset"

# Automatic fallback to sample dataset if full dataset is not present
if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
    if SAMPLE_DATASET_DIR.exists():
        DATASET_DIR = SAMPLE_DATASET_DIR
        DDD_DATASET_DIR = SAMPLE_DDD_DIR
        MRL_DATASET_DIR = SAMPLE_MRL_DIR
        YAWN_DATASET_DIR = SAMPLE_YAWN_DIR

# Model artifacts and output paths
MODELS_DIR = BASE_DIR / "models"
EYE_MODEL_DIR = MODELS_DIR / "eye_model"
YAWN_MODEL_DIR = MODELS_DIR / "yawn_model"
MOBILENETV2_DIR = MODELS_DIR / "mobilenetv2"
OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
CONFUSION_MATRIX_DIR = OUTPUTS_DIR / "confusion_matrix"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Application, source, and notebook paths
APP_DIR = BASE_DIR / "app"
SRC_DIR = BASE_DIR / "src"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Deep Learning Model & Training Hyperparameters
IMAGE_SIZE = (224, 224)          # Standard input dimensions (Height, Width) for full-frame/face models
IMAGE_SHAPE = (224, 224, 3)       # Full tensor shape (H, W, C)
EYE_IMAGE_SIZE = (64, 64)        # Standard input dimensions for close-up eye region CNN
EYE_IMAGE_SHAPE = (64, 64, 3)
MOUTH_IMAGE_SIZE = (64, 64)      # Standard input dimensions for mouth region Yawn CNN
MOUTH_IMAGE_SHAPE = (64, 64, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_TYPE = "MobileNetV2"        # Model architecture selection (e.g., 'MobileNetV2', 'CustomCNN', 'ResNet50')

# Real-Time Inference & Alert Thresholds
CAMERA_INDEX = 0                  # Default webcam device index
DROWSINESS_THRESHOLD = 0.5        # Raw classification probability threshold for drowsiness
ALARM_COOLDOWN = 3.0              # Cooldown period in seconds between alerts to prevent audio spam
CLOSED_EYES_CONSECUTIVE_FRAMES = 15 # Frame buffer for persistent eye closure
YAWN_CONSECUTIVE_FRAMES = 20        # Frame buffer for continuous yawning detection
ALARM_SPEECH_MESSAGE = "WARNING! DRIVER DROWSINESS DETECTED. PLEASE TAKE A BREAK."
ENABLE_AUDIO_ALARM = True           # Master audio alert toggle

# Multilingual Alert Configuration (English, Hindi, Punjabi)
ALERT_AUDIO_DIR = BASE_DIR / "data" / "alerts"
ALERT_LANGUAGE_MODE = "all"        # Options: "all" (multilingual EN->HI->PA), "hi", "pa", "en"
ALERT_MESSAGES = {
    "en": {
        "name": "English",
        "audio_file": "alert_en.mp3",
        "speech": "Warning! Driver drowsiness detected. Please take a break.",
        "hud_critical": "CRITICAL: DROWSINESS DETECTED!",
        "hud_sub": "PLEASE PULL OVER AND TAKE A BREAK",
        "hud_warning": "WARNING: DRIVER FATIGUE DETECTED",
    },
    "hi": {
        "name": "Hindi (हिन्दी)",
        "audio_file": "alert_hi.mp3",
        "speech": "चेतावनी! चालक को नींद आ रही है। कृपया गाड़ी रोकें और आराम करें।",
        "hud_critical": "सावधान! चालक को नींद आ रही है!",
        "hud_sub": "कृपया गाड़ी रोकें और विश्राम करें",
        "hud_warning": "चेतावनी: चालक को थकान महसूस हो रही है",
    },
    "pa": {
        "name": "Punjabi (ਪੰਜਾਬੀ)",
        "audio_file": "alert_pa.mp3",
        "speech": "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।",
        "hud_critical": "ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ!",
        "hud_sub": "ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ",
        "hud_warning": "ਚੇਤਾਵਨੀ: ਡਰਾਈਵਰ ਨੂੰ ਥਕਾਵਟ ਮਹਿਸੂਸ ਹੋ ਰਹੀ ਹੈ",
    },
}

# Emergency Red Light Wake-Up Flash Settings
ENABLE_FACE_RED_LIGHT_FLASH = True  # High-intensity red light strobe flash to wake up sleeping driver
RED_LIGHT_FLASH_FREQ_HZ = 3.5       # Strobe flash frequency in Hz (3.5 Hz alerting strobe rate)
RED_LIGHT_INTENSITY = 1.0           # Maximum red strobe illumination intensity (0.0 to 1.0)
RED_LIGHT_FULLSCREEN_BURST = True   # Full-frame red flash bursts to physically illuminate driver in cabin

# Alert Cycle Completion Policy
# Once triggered, the alert warning MUST complete its entire cycle (audio warnings + wake-up strobe)
# regardless of whether the driver momentarily opens eyes or yawning stops, ensuring driver is fully awake.
ALERT_MUST_COMPLETE_CYCLE = True

# Temporal Drowsiness Decision System Parameters
# (Calibrated engineering parameters; not medical diagnostic standards)
ROLLING_WINDOW_SIZE = 90          # Rolling frame window for temporal analysis (~3 sec at 30 fps)
BLINK_MAX_FRAMES = 6              # Maximum consecutive closed frames for a normal blink (~0.2 sec)
WARNING_CLOSED_FRAMES = 12        # Persistent eye closure frame threshold for WARNING (~0.4 sec)
DROWSY_CLOSED_FRAMES = 24         # Persistent eye closure frame threshold for DROWSY (~0.8 sec)
YAWN_SUSTAINED_FRAMES = 15        # Consecutive frames to register a sustained yawn event (~0.5 sec)

# Biometric Baseline Thresholds (Engineering reference values)
EAR_CLOSED_THRESHOLD = 0.19       # EAR below which eye aperture is physically constrained
MAR_YAWN_THRESHOLD = 0.40         # MAR above which mouth is excessively opened in yawn posture

# Multi-Signal Drowsiness Score Fusion Weights (Sums to 1.0)
WEIGHT_EYE_CNN = 0.35             # Deep learning eye classification contribution
WEIGHT_EAR = 0.20                 # Continuous geometric Eye Aspect Ratio contribution
WEIGHT_YAWN = 0.20                # Yawn classification and MAR contribution
WEIGHT_TEMPORAL = 0.25            # Temporal duration and PERCLOS contribution

# Drowsiness State Decision Thresholds (Range: 0.0 to 1.0)
SCORE_WARNING_THRESHOLD = 0.40    # Score threshold transitioning state from ALERT to WARNING
SCORE_DROWSY_THRESHOLD = 0.65     # Score threshold transitioning state to DROWSY
