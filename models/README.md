# Model Checkpoints

This directory contains trained model weights.

## Available Models

Place your trained model weights in this directory:

- `best.pt` - Best performing model from training
- `last.pt` - Latest checkpoint from training

## Model Information

The models in this project are based on YOLOv5 architecture, fine-tuned for metal surface defect detection.

### Training Configuration

Default training configuration used:
- Base model: YOLOv5s/m/l
- Input size: 640x640
- Dataset: Severstal Steel Defect Dataset
- Classes: 6 defect types

## Usage

Load a trained model for inference:

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model('path/to/image.jpg')
```

Or use the inference script:

```bash
python detect.py --weights models/best.pt --source image.jpg
```

## Model Performance

Performance metrics are generated using:

```bash
python scripts/evaluate_model.py --weights models/best.pt \
                                  --data yolov5/custom_dataset.yaml
```

## Download Pre-trained Weights

If you want to start training from scratch, download YOLOv5 pre-trained weights:

- [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- [yolov5m.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)
- [yolov5l.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)

The training script will automatically download weights if they don't exist locally.
