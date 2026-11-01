# Real-Time Driver Drowsiness Detection and Alert System Using Deep Learning

An academic Machine Learning and Computer Vision system designed to monitor driver alertness, classify drowsy states (eye closure, persistent blinking, and yawning), and trigger real-time auditory/visual safety alerts.

## Project Structure

```text
driver_drowsiness_detection/
│
├── dataset/                    # Uploaded Kaggle datasets (physically present)
│   ├── DDD Dataset/            # Driver Drowsiness Dataset (41,793 images)
│   ├── MRL Dataset/            # MRL Eye Dataset (84,898 images across train/val/test)
│   └── Yawn Dataset/           # Yawn Dataset (5,119 images)
│
├── data/
│   ├── raw/                    # Raw inputs and links
│   ├── processed/              # Normalized, resized, and prepared datasets
│   └── README.md               # Dataset descriptions and schemas
│
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│
├── src/                        # Core Python source modules
│   └── inspect_datasets.py     # Comprehensive dataset verification & inspection tool
│
├── models/                     # Saved model architectures and weights (.keras, .h5, .tflite)
│
├── outputs/                    # Output visual artifacts
│   ├── graphs/                 # Training curves and evaluation plots
│   ├── confusion_matrix/       # Confusion matrices
│   └── predictions/            # Visual prediction outputs and test qualitative samples
│
├── app/                        # Real-time computer vision application & UI (Streamlit / OpenCV)
│
├── config.py                   # Centralized project configuration & hyperparameters
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Dataset Overview

The system supports both an **instant curated sample dataset** (included directly in the Git repository) and the **full large-scale datasets** (downloadable from Kaggle):

| Dataset | Modality | Classes / Splits | Full Images | Kaggle Source Link |
| :--- | :--- | :--- | :--- | :--- |
| **Driver Drowsiness Dataset (DDD)** | Driver Face/Posture | Drowsy (22,348), Non Drowsy (19,445) | 41,793 | [Kaggle DDD Dataset](https://www.kaggle.com/datasets/ismailnasrummin/driver-drowsiness-dataset-ddd) |
| **MRL Eye Dataset** | Infrared Eye Region | Awake (42,952), Sleepy (41,946) split into train/val/test | 84,898 | [Kaggle MRL Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) |
| **Yawn Dataset** | Mouth / Facial Region | no yawn (2,591), yawn (2,528) | 5,119 | [Kaggle Yawn Dataset](https://www.kaggle.com/datasets/vipoooool/yawn-dataset) |

> 💡 **Ready Out-of-the-Box:** A curated sample dataset (~9.2 MB, 440 real images across all classes) is staged in `data/sample_dataset/` and tracked in Git so you can immediately inspect, test, and run the pipeline upon cloning without downloading 2.89 GB. Full download instructions are documented in [data/README.md](data/README.md).

## Quickstart

### 1. Inspect Datasets
To inspect and verify datasets (automatically checks `dataset/` or falls back to `data/sample_dataset/`):

```bash
python src/inspect_datasets.py
```

### 2. Multilingual Voice & HUD Alerts (English, Hindi, Punjabi)
The alert system provides real-time voice warnings and visual HUD banners in three languages:
- **English**: *"Warning! Driver drowsiness detected. Please take a break."*
- **Hindi (हिन्दी)**: *"चेतावनी! चालक को नींद आ रही है। कृपया गाड़ी रोकें और आराम करें।"*
- **Punjabi (ਪੰਜਾਬੀ)**: *"ਖ਼ਬਰਦਾਰ! ਡਰਾਈਵਰ ਨੂੰ ਨੀਂਦ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਗੱਡੀ ਰੋਕੋ ਅਤੇ ਆਰਾਮ ਕਰੋ।"*

#### Running the System with Multilingual Alerts:
```bash
# Run with All 3 languages (English -> Hindi -> Punjabi sequential alert)
python main.py --lang all

# Run specifically in Hindi
python main.py --lang hi

# Run specifically in Punjabi
python main.py --lang pa

# Run specifically in English
python main.py --lang en

# Live interactive webcam controls:
# Press 'l' during monitoring to cycle languages (ALL -> HI -> PA -> EN)
# Press 'm' to mute/unmute audio
# Press 's' to capture a screenshot
# Press 'q' or ESC to exit
```

#### Running the Streamlit Academic Dashboard:
```bash
streamlit run app/app.py
```
Select your preferred language in the sidebar under **"Alert Language"** and click **"Test Voice Alert"** to preview.

### 3. Emergency Red Light Face Strobe (Wake-Up System)
When a critical drowsiness event is detected, the system illuminates and pulses an emergency **Red Light Flash** directly onto the driver's face:
- **Alert Frequency**: 3.5 Hz pulsating strobe wave designed to stimulate optic photoreceptors and wake the driver up immediately.
- **Radial Face Illumination**: High-intensity red wash gradient dynamically centered on the facial region with soft Gaussian falloff.
- **Peripheral Strobe Border**: Synchronized red beacon frame bordering the viewport to catch peripheral vision even if eyelids are drooping.
- **UI & Dashboard Integration**: Fully synchronized in both the OpenCV master HUD and the Streamlit dashboard with emergency pulsing animations.


