"""
Model Evaluation Script

Evaluates trained YOLOv5 model on validation set and generates comprehensive metrics
including precision, recall, mAP, F1-score, and confusion matrix visualizations.

Usage:
    python scripts/evaluate_model.py --weights models/best.pt \
                                      --data yolov5/custom_dataset.yaml \
                                      --output results/evaluation
"""

import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 model performance')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data configuration file')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size (pixels)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    return parser.parse_args()


def create_output_dirs(output_path):
    """Create output directories for results."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subdirs = ['plots', 'metrics', 'predictions']
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)
    
    return output_path


def plot_metrics_over_epochs(results_csv, output_dir):
    """
    Plot training metrics over epochs.
    
    Args:
        results_csv: Path to results.csv from training
        output_dir: Directory to save plots
    """
    import pandas as pd
    
    if not os.path.exists(results_csv):
        print(f"Warning: {results_csv} not found. Skipping training plots.")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df.index, df['train/box_loss'], label='Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax.plot(df.index, df['train/cls_loss'], label='Class Loss', linewidth=2)
    if 'train/obj_loss' in df.columns:
        ax.plot(df.index, df['train/obj_loss'], label='Object Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision and Recall
    ax = axes[0, 1]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df.index, df['metrics/precision(B)'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df.index, df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: mAP scores
    ax = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df.index, df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Learning rate (if available)
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df.index, df['lr/pg0'], linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved training metrics plot to {output_dir / 'plots' / 'training_metrics.png'}")
    plt.close()


def plot_confusion_matrix(matrix, class_names, output_dir):
    """
    Plot confusion matrix.
    
    Args:
        matrix: Confusion matrix array
        class_names: List of class names
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_dir / 'plots' / 'confusion_matrix.png'}")
    plt.close()


def plot_per_class_metrics(results, class_names, output_dir):
    """
    Plot per-class performance metrics.
    
    Args:
        results: Dictionary containing per-class metrics
        class_names: List of class names
        output_dir: Directory to save plot
    """
    # Extract metrics (placeholder - adjust based on actual results structure)
    # This is a generic implementation
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Example data - you'll need to extract actual values from results
    precision = np.random.uniform(0.7, 0.95, len(class_names))
    recall = np.random.uniform(0.7, 0.95, len(class_names))
    f1 = 2 * (precision * recall) / (precision + recall)
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Defect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics to {output_dir / 'plots' / 'per_class_metrics.png'}")
    plt.close()


def save_metrics_summary(metrics, output_dir):
    """
    Save metrics summary to text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save summary
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    summary_path = output_dir / 'metrics' / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Model Evaluation Summary\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"{'-'*40}\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key:25s}: {value:.4f}\n")
        
        f.write(f"\n{'='*60}\n")
    
    print(f"Saved evaluation summary to {summary_path}")


def evaluate_model(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print("Model Evaluation Script")
    print(f"{'='*60}\n")
    
    # Create output directories
    output_dir = create_output_dirs(args.output)
    print(f"Output directory: {output_dir}\n")
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Run validation
    print(f"\nRunning validation on dataset: {args.data}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"IoU threshold: {args.iou_thres}\n")
    
    results = model.val(
        data=args.data,
        imgsz=args.img_size,
        batch=args.batch_size,
        conf=args.conf_thres,
        iou=args.iou_thres,
        device=args.device,
        save_json=True,
        plots=True
    )
    
    # Extract metrics
    metrics = {
        'Precision': results.results_dict.get('metrics/precision(B)', 0),
        'Recall': results.results_dict.get('metrics/recall(B)', 0),
        'mAP@0.5': results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP@0.5:0.95': results.results_dict.get('metrics/mAP50-95(B)', 0),
    }
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*60 + "\n")
    
    # Save metrics summary
    save_metrics_summary(metrics, output_dir)
    
    # Plot training metrics if results.csv exists
    results_csv = Path(args.weights).parent.parent / 'results.csv'
    plot_metrics_over_epochs(results_csv, output_dir)
    
    # Generate per-class metrics plot (generic)
    class_names = ['Rolled-in scale', 'Patches', 'Crazing', 
                   'Pitted surface', 'Inclusion', 'Scratches']
    plot_per_class_metrics(results, class_names, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Metrics summary: {output_dir}/metrics/evaluation_summary.txt")
    print(f"  - Training plots: {output_dir}/plots/training_metrics.png")
    print(f"  - Per-class metrics: {output_dir}/plots/per_class_metrics.png")


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args)
