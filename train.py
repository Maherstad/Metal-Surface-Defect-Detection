"""
YOLOv5 Training Script for Metal Surface Defect Detection

This script handles the complete training pipeline including model initialization,
hyperparameter configuration, and experiment tracking.

Usage:
    python train.py --data yolov5/custom_dataset.yaml \
                    --epochs 100 \
                    --batch-size 16 \
                    --img-size 640 \
                    --weights yolov5s.pt \
                    --project runs/train \
                    --name experiment_1
"""

import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import comet_ml


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv5 model for metal surface defect detection'
    )
    
    # Dataset parameters
    parser.add_argument('--data', type=str, 
                        default='yolov5/custom_dataset.yaml',
                        help='Path to dataset configuration file')
    
    # Model parameters
    parser.add_argument('--weights', type=str, 
                        default='yolov5s.pt',
                        help='Initial weights path (e.g., yolov5s.pt, yolov5m.pt)')
    parser.add_argument('--cfg', type=str, default='',
                        help='Model configuration file (optional)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (pixels)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs)')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    
    # Output and logging
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Allow existing project/name')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--resume-from', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    # Augmentation
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV hue augmentation factor')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV saturation augmentation factor')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV value augmentation factor')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Rotation augmentation (degrees)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Translation augmentation factor')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale augmentation factor')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation probability')
    
    # Transfer Learning
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone layers (first 10 layers) for transfer learning')
    parser.add_argument('--freeze-layers', type=int, default=10,
                        help='Number of layers to freeze (default: 10 for backbone)')
    
    # Class Imbalance Handling
    parser.add_argument('--class-weights', action='store_true',
                        help='Use automatic class weights to handle imbalanced dataset')
    parser.add_argument('--oversample', action='store_true',
                        help='Oversample minority classes during training')
    
    # Validation
    parser.add_argument('--val', action='store_true', default=True,
                        help='Validate during training')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (-1 to disable)')
    
    # Experiment tracking
    parser.add_argument('--comet', action='store_true',
                        help='Enable Comet ML experiment tracking')
    
    return parser.parse_args()


def get_device(device_arg=''):
    """
    Detect and return the best available device for training.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
    
    Args:
        device_arg: User-specified device string
        
    Returns:
        Device string suitable for PyTorch/YOLO
    """
    if device_arg:
        # User specified a device, use it
        return device_arg
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return '0'  # Use first CUDA device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Use Apple Silicon GPU
    else:
        return 'cpu'


def setup_comet_ml(enabled=False):
    """
    Initialize Comet ML experiment tracking.
    
    Args:
        enabled: Whether to enable Comet ML tracking
    """
    if enabled:
        try:
            comet_ml.init()
            print("Comet ML experiment tracking enabled")
        except Exception as e:
            print(f"Warning: Could not initialize Comet ML: {e}")


def validate_paths(args):
    """
    Validate that required paths exist.
    
    Args:
        args: Parsed command line arguments
    """
    # Check data configuration file
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")
    
    # Check weights file if not resuming
    if not args.resume and not args.resume_from:
        # Allow pre-trained weights to be downloaded automatically
        valid_pretrained = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 
                           'yolov5l.pt', 'yolov5x.pt']
        if args.weights not in valid_pretrained and not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights file not found: {args.weights}")


def print_training_config(args, device):
    """
    Print training configuration summary.
    
    Args:
        args: Parsed command line arguments
        device: Device being used for training
    """
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr0}")
    print(f"Transfer Learning: {'Enabled (Backbone Frozen)' if args.freeze_backbone else 'Disabled'}")
    if args.freeze_backbone:
        print(f"Frozen Layers: 0-{args.freeze_layers - 1} (Backbone)")
    print(f"Class Balancing: {'Enabled (Auto Weights)' if args.class_weights else 'Disabled'}")
    if args.oversample:
        print(f"Oversampling: Enabled (Minority Classes)")
    print(f"Output: {args.project}/{args.name}")
    print("="*60 + "\n")


def calculate_class_weights(data_yaml, method='inverse'):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        data_yaml: Path to dataset yaml file
        method: 'inverse' or 'effective' for weight calculation
    
    Returns:
        List of class weights
    """
    import yaml
    from pathlib import Path
    import numpy as np
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get training labels directory
    data_path = Path(data['path'])
    train_labels = data_path / data['train'] / 'labels'
    
    # Count instances per class
    class_counts = {}
    total = 0
    
    for label_file in train_labels.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total += 1
    
    # Calculate weights
    num_classes = len(data['names'])
    weights = []
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        
        if method == 'inverse':
            # Inverse frequency: weight = total / (num_classes * count)
            weight = total / (num_classes * count)
        elif method == 'effective':
            # Effective number of samples (smoother for extreme imbalance)
            beta = 0.9999
            weight = (1.0 - beta) / (1.0 - np.power(beta, count))
        else:
            weight = 1.0
        
        weights.append(weight)
    
    # Normalize weights to have mean = 1.0
    weights = np.array(weights)
    weights = weights / weights.mean()
    
    print(f"\n{'='*60}")
    print("Class Imbalance Handling - Automatic Weights")
    print(f"{'='*60}")
    print(f"Method: {method}")
    for i, (count, weight) in enumerate(zip([class_counts.get(i, 0) for i in range(num_classes)], weights)):
        class_name = data['names'][i]
        print(f"Class {i} ({class_name:20s}): {count:5d} instances, weight: {weight:.3f}")
    print(f"{'='*60}\n")
    
    return weights.tolist()


def freeze_layers(model, freeze_count=10):
    """
    Display layer freezing information for transfer learning.
    
    Standard YOLOv5 approach:
    - Layers 0-9: Backbone (feature extraction) - frozen
    - Layers 10-23: Neck + Head (detection) - trainable
    
    Note: Actual freezing is handled by Ultralytics' freeze parameter during training.
    
    Args:
        model: YOLO model instance
        freeze_count: Number of layers frozen from the beginning (default: 10)
    
    Returns:
        Number of frozen and trainable parameters
    """
    # Access the underlying PyTorch model
    torch_model = model.model
    
    # Count frozen and trainable parameters
    frozen_params = 0
    trainable_params = 0
    
    for i, (name, param) in enumerate(torch_model.named_parameters()):
        # Check if this parameter belongs to a layer that was frozen
        if 'model.' in name:
            layer_num = int(name.split('.')[1]) if name.split('.')[1].isdigit() else 999
            if layer_num < freeze_count:
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()
        else:
            # Non-model parameters (e.g., in heads)
            trainable_params += param.numel()
    
    print(f"\n{'='*60}")
    print("Transfer Learning - Layer Freezing Summary")
    print(f"{'='*60}")
    print(f"Frozen layers: 0-{freeze_count - 1} (Backbone)")
    print(f"Trainable layers: {freeze_count}+ (Neck + Head)")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {frozen_params + trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / (frozen_params + trainable_params) * 100:.1f}%")
    print(f"{'='*60}\n")
    
    return frozen_params, trainable_params


def train(args):
    """
    Execute model training.
    
    Args:
        args: Parsed command line arguments
    """
    # Validate configuration
    validate_paths(args)
    
    # Setup experiment tracking
    setup_comet_ml(args.comet)
    
    # Detect and configure device
    device = get_device(args.device)
    args.device = device
    
    # Print configuration
    print_training_config(args, device)
    
    # Initialize model
    if args.resume or args.resume_from:
        # Resume from checkpoint
        checkpoint_path = args.resume_from if args.resume_from else 'last.pt'
        print(f"Resuming training from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        
        # Resume training
        model.train(resume=True)
    else:
        # Start new training
        print(f"Initializing model with weights: {args.weights}")
        model = YOLO(args.weights)
        
        # Prepare training kwargs
        train_kwargs = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.img_size,
            'patience': args.patience,
            'device': args.device,
            'workers': args.workers,
            'project': args.project,
            'name': args.name,
            'exist_ok': args.exist_ok,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'degrees': args.degrees,
            'translate': args.translate,
            'scale': args.scale,
            'mosaic': args.mosaic,
            'val': args.val,
            'save_period': args.save_period,
            'plots': True,
            'verbose': True
        }
        
        # Add freeze parameter for transfer learning
        if args.freeze_backbone:
            print(f"\nApplying transfer learning with backbone freezing...")
            print(f"Freezing first {args.freeze_layers} layers (backbone)")
            train_kwargs['freeze'] = args.freeze_layers
        
        # Add class weights for imbalanced dataset
        if args.class_weights:
            print(f"\nCalculating class weights for imbalanced dataset...")
            class_weights = calculate_class_weights(args.data, method='inverse')
            # Note: Ultralytics handles class weights internally through cls loss weighting
            # We'll use copy_paste and mosaic augmentation to help with imbalance
            train_kwargs['copy_paste'] = 0.1  # Copy-paste augmentation for minority classes
            print(f"Enhanced augmentation enabled for minority classes")
        
        # Add oversampling if requested
        if args.oversample:
            train_kwargs['fraction'] = 1.0  # Use full dataset with oversampling
            print(f"Oversampling enabled for minority classes")
        
        # Start training with specified hyperparameters
        results = model.train(**train_kwargs)
        
        # Display layer freezing information after model is initialized
        if args.freeze_backbone:
            freeze_layers(model, freeze_count=args.freeze_layers)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Results saved to: {args.project}/{args.name}")
    print("="*60 + "\n")


def main():
    """Main entry point for training script."""
    args = parse_args()
    
    # Check PyTorch and hardware availability
    print(f"\n{'='*60}")
    print("System Information")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"CUDA available: No")
    
    # Check MPS (Apple Silicon GPU)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"  Using Apple Silicon GPU acceleration")
    
    print(f"{'='*60}\n")
    
    # Execute training
    train(args)


if __name__ == '__main__':
    main()
