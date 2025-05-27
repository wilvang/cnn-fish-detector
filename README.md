# Low-Cost Fish Detection in Live Video

## Overview

This project presents a low-cost fish detection system designed to run on a Raspberry Pi. It uses live video surveillance to detect the presence of fish in Norwegian rivers, automating fish population monitoring. The system is built using a custom-trained Convolutional Neural Network (CNN) and optimized for real-time performance on embedded hardware.

> **Collaboration**: Anadrom & Nordavind Utvikling AS <br> > **Platform**: Raspberry Pi <br>
> **Frameworks**: TensorFlow/Keras, OpenCV, TFLite (for deployment<br>

## Objectives

- Detect fish (`fish` / `no_fish`) in live video.
- Run efficiently on low-cost hardware (Raspberry Pi).
- Handle varying lighting and visibility conditions.
- Evaluate the impact of preprocessing on model performance

## Requirements

- Raspberry Pi
- Python 3.9
- OpenCV
- TensorFlow/Keras
- TensorFlow Lite

## How to Deploy and Run the Fish Detector

Follow these steps to set up the environment and run the real-time fish detection system using your video recorder.

---

### 1. Prerequisites

- A working **video recorder**
- Python **3.9**
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for managing dependencies)

---

### 2. Install Conda (if not already installed)

Download and install **Miniconda** for your platform:

- [Miniconda Installer – Windows/macOS/Linux](https://docs.conda.io/en/latest/miniconda.html)

---

### 3. Create and Activate Virtual Environment

Open a terminal and run:

```bash
# Create a new environment with Python 3.9
conda create -n fish-env python=3.9 -y

# Activate the environment
conda activate fish-env
```

### 4. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then, install the TensorFlow Lite runtime manually (required for some platforms like Raspberry Pi):

```bash
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.10.0-cp39-cp39-linux_x86_64.whl
```

### 5. Run the Detector

To launch the real-time fish detector using your webcam, run:

```bash
python3 deploy/deploy.py
```

You will see:

- A live video feed window titled "Fish Detector"
- Prediction labels overlaid on the video
- Detection results logged to detections.log

### 6. How to Exit

Press `q` in the video window or hit `Ctrl + C` in the terminal.

Output example:

```
2025-05-27 15:04:21, fish, score=0.9312
2025-05-27 15:04:23, no_fish, score=0.2164
```

## Dataset

- **Source**: Monitoring stations across Norwegian rivers.
- **Size**: 34,492 labeled images
  - 19,226 with fish
  - 15,266 without fish
- **Challenges**: Low illumination, noise, blur, and color similarity between fish and background.

## Methodology

### 1. Image Preprocessing & Augmentation

To improve generalization and simulate real-world conditions, the following techniques were applied:

- **Gaussian Blur**: Simulates underwater noise
- **Rotation**: Random angles (0–360°)
- **Mirroring**: Horizontal flips
- **Brightness Adjustment**: Simulates different lighting conditions
- **Contrast Adjustment**: Varies image contrast
- **Zooming**: Random scaling

### 2. Model Architecture

A custom Convolutional Neural Network (CNN) was developed using **TensorFlow/Keras** for binary classification.

**Input Shape**  
`(64, 64, 1)` — grayscale images (Y channel from YCbCr color space)

---

**Model Layers**

The CNN architecture follows a typical feature extraction and classification structure:

1. **Feature Extraction**

   - `Conv2D(16, kernel_size=3, activation='relu')`
   - `MaxPooling2D(pool_size=2)`
   - `Conv2D(32, kernel_size=3, activation='relu')`
   - `MaxPooling2D(pool_size=2)`
   - `Conv2D(64, kernel_size=3, activation='relu')`
   - `MaxPooling2D(pool_size=2)`

2. **Classification Head**
   - `Flatten`
   - `Dense(64, activation='relu')`
   - `Dropout(0.5)` — helps prevent overfitting
   - `Dense(1, activation='sigmoid')` — outputs probability

---

**Training Configuration**

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 32
- **Data Split**:
  - 70% Training
  - 10% Validation
  - 20% Testing

## Results

| Metric             | Without Preprocessing | With Preprocessing |
| :----------------- | --------------------: | -----------------: |
| Accuracy (%)       |                    93 |                 95 |
| Precision (Fish)   |                  0.99 |               0.98 |
| Recall (Fish)      |                  0.88 |               0.92 |
| F1-Score (Fish)    |                  0.93 |               0.95 |
| Precision (NoFish) |                  0.87 |               0.91 |
| Recall (NoFish)    |                  0.99 |               0.97 |
| F1-Score (NoFish)  |                  0.93 |               0.94 |

> Preprocessing improved recall and overall balance between precision and recall, making the model more robust in real-world conditions.

## Project structure

```
├── data/
│   ├── fish/
│   └── no_fish/
├── deploy/
│   └── deploy.py               # TFLite deployment script
├── logs/                       # Inference logs
├── model/
│   ├── conv_tflite.py          # TFLite conversion script
│   ├── fish_detector.keras     # Original trained model
│   ├── fish_detector.tflite    # Converted TFLite model
│   ├── fish_detector-v2.keras
│   └── fish_detector-v2.tflite
├── src/
│   ├── main.py                 # Entry point for training
│   ├── preprocess.py           # Preprocessing and augmentation
│   ├── segment.py              # Data segmentation
│   └── train.py                # CNN training script
├── requirements.txt
├── README.md
```
