# Semantic Segmentation Lab

## Setup

```bash
uv sync
uv pip install torch torchvision torchaudio albumentations opencv-python optuna scikit-learn
```

## Download Dataset

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```

## Run

```bash
marimo edit semantic_segmentation.py
```

## Implementation

DeepLabV3-ResNet50 on Pascal VOC 2012

**Components:**
- Data loading with augmentations (albumentations)
- Model: DeepLabV3 with pretrained ResNet-50
- Training with Optuna hyperparameter tuning
- Evaluation with mIoU metric
- Logging via logly

**Hyperparameter Search:**
- Learning rate: [1e-5, 1e-3]
- Batch size: [4, 8, 16]
- Optimizer: [Adam, SGD]
- Weight decay: [1e-5, 1e-3]
