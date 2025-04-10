# Glasses Detection ML Pipeline

A comprehensive machine learning pipeline for detecting glasses in images and video streams using YOLOv8. This project integrates real-time detection, model training, and MLflow experiment tracking.

## Features

- Real-time glasses detection from webcam or video sources
- Batch processing of image directories
- Training pipeline with MLflow experiment tracking
- Pre-trained YOLOv11 model fine-tuning for glasses detection
- Configurable parameters for training and inference
- Detailed logging and metrics collection

## Project Structure

```
glasses-detection/
├── configs/           # Configuration files
├── data/              # Dataset directory
├── logs/              # Log files
├── mlruns/           # MLflow runs directory
├── notebooks/         # Jupyter notebooks for exploration
├── output/            # Output images and results
├── src/               # Source code
│   ├── __init__.py
│   ├── logger.py      # Logging setup
│   ├── config         # Configuration management
        ├── config.py  # Configuration loader
├── main.py            # Main application 
├── predict.py         # Prediction script
├── train.py           # Training script
└── app.py             # Streamlit Application 
└── requirements.txt   # Dependencies

```

## Installation


1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MLflow tracking server (optional for tracking experiments):
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## Dataset Preparation

Prepare your dataset in YOLO format:

```
data/glasses/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Update the `data/data.yaml` file with your dataset details.

## Usage

### Training

Train the model using the following command:

```bash
python train.py 
```

Training progress and results will be tracked in MLflow.

### Live Inference
Designed specifically for running inference (prediction) tasks.
Run real-time inference using your webcam:

```bash
python predict.py --mode live
```
#press 'q' to exit

### Batch Inference
Processes a directory of images for batch inference.
Process a directory of images:

```bash
python predict.py --mode batch --input path/to/images --output path/to/output 
```

### Running from Main
It initializes the pipeline and runs it in a general-purpose way, likely for live inference or a default mode.

To run the main application:

```bash
python main.py 
```

## RUnning streamlit application
``` bash
streamlit run app.py 
```

## Configuration

The application is configured using YAML files:

- `configs/app_config.yaml`: Main application configuration
- `configs/mlflow.yaml`: MLflow tracking configuration
- `data/data.yaml`: Dataset configuration

Edit these files to customize the behavior of the application.

### Key Configuration Parameters

#### Model Settings
```yaml
model:
  path: "yolo11l.pt"  # Base model for training
  confidence_threshold: 0.25
```

#### Training Settings
```yaml
training:
  data: "data/glasses-dataset.yaml"
  epochs: 100
  batch_size: 8
  image_size: 640
  device: 0  # 0 for GPU, 'cpu' for CPU
```

#### Camera Settings
```yaml
camera:
  source: 0  # 0 for webcam, or video file path
  width: 640
  height: 480
```

## MLflow Tracking

This project uses MLflow for experiment tracking. You can view training metrics and results by running:

```bash
mlflow ui
```

Then open your browser at http://localhost:5000 (or the configured port).

## Acknowledgments

- YOLOv8 by Ultralytics
- MLflow by Databricks
- Project carried out by Ts. Amirah Heng, AI Engineer