"""
Results Visualization Script

Creates publication-ready visualizations of detection results including
sample predictions, confidence distributions, and detection statistics.

Usage:
    python scripts/visualize_results.py --source path/to/images \
                                         --weights models/best.pt \
                                         --output results/visualizations
"""

import argparse
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import seaborn as sns
from collections import defaultdict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image directory or single image')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results/visualizations',
                        help='Output directory')
    parser.add_argument('--max-images', type=int, default=20,
                        help='Maximum number of images to process')
    return parser.parse_args()


def create_detection_grid(images_data, output_path, grid_size=(4, 5)):
    """
    Create a grid of detection results.
    
    Args:
        images_data: List of (image, detections) tuples
        output_path: Path to save grid
        grid_size: Tuple of (rows, cols)
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle('Sample Detection Results', fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images_data):
            img, detections = images_data[idx]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            
            # Add detection count
            num_detections = len(detections) if detections is not None else 0
            ax.set_title(f'Detections: {num_detections}', fontsize=10)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detection grid to {output_path}")
    plt.close()


def plot_confidence_distribution(all_confidences, class_names, output_path):
    """
    Plot confidence score distributions per class.
    
    Args:
        all_confidences: Dictionary mapping class names to confidence lists
        class_names: List of class names
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confidence Score Distribution by Defect Type', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        if idx < len(axes):
            ax = axes[idx]
            confidences = all_confidences.get(class_name, [])
            
            if confidences:
                ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title(class_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Detections', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(class_name)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence distribution to {output_path}")
    plt.close()


def plot_detection_statistics(stats, output_path):
    """
    Plot detection statistics including counts per class and per image.
    
    Args:
        stats: Dictionary containing detection statistics
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Detections per class
    ax = axes[0]
    class_names = list(stats['class_counts'].keys())
    counts = list(stats['class_counts'].values())
    
    bars = ax.bar(range(len(class_names)), counts, color='steelblue', alpha=0.8)
    ax.set_xlabel('Defect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Detections', fontsize=12, fontweight='bold')
    ax.set_title('Total Detections by Defect Type', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Detection distribution
    ax = axes[1]
    detections_per_image = stats['detections_per_image']
    
    ax.hist(detections_per_image, bins=15, color='coral', alpha=0.8, edgecolor='black')
    ax.axvline(np.mean(detections_per_image), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(detections_per_image):.1f}')
    ax.set_xlabel('Detections per Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Detections per Image', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detection statistics to {output_path}")
    plt.close()


def plot_summary_report(stats, output_path):
    """
    Create a summary report with key metrics.
    
    Args:
        stats: Dictionary containing detection statistics
        output_path: Path to save report
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Detection Summary Report', fontsize=18, fontweight='bold')
    
    # Create text summary
    summary_text = f"""
    DETECTION SUMMARY
    {'='*50}
    
    Total Images Processed: {stats['total_images']}
    Total Detections: {stats['total_detections']}
    Images with Detections: {stats['images_with_detections']}
    Images without Detections: {stats['images_without_detections']}
    
    Average Detections per Image: {stats['avg_detections']:.2f}
    Average Confidence Score: {stats['avg_confidence']:.3f}
    
    DETECTIONS BY CLASS:
    {'-'*50}
    """
    
    for class_name, count in stats['class_counts'].items():
        percentage = (count / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
        summary_text += f"\n    {class_name:20s}: {count:4d} ({percentage:5.1f}%)"
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=fig.transFigure)
    
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary report to {output_path}")
    plt.close()


def visualize_results(args):
    """
    Main visualization function.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print("Results Visualization Script")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Get image paths
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob('*.jpg')) + \
                     list(source_path.glob('*.png')) + \
                     list(source_path.glob('*.jpeg'))
        image_paths = image_paths[:args.max_images]
    
    print(f"Processing {len(image_paths)} images...\n")
    
    # Process images
    all_confidences = defaultdict(list)
    stats = {
        'total_images': len(image_paths),
        'total_detections': 0,
        'images_with_detections': 0,
        'images_without_detections': 0,
        'class_counts': defaultdict(int),
        'detections_per_image': [],
        'avg_detections': 0,
        'avg_confidence': 0
    }
    
    images_data = []
    all_conf_scores = []
    
    # Class names (adjust based on your dataset)
    class_names = ['Rolled-in scale', 'Patches', 'Crazing', 
                   'Pitted surface', 'Inclusion', 'Scratches']
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Run inference
        results = model(img, conf=args.conf_thres, imgsz=args.img_size, verbose=False)
        
        # Extract detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            if num_detections > 0:
                stats['images_with_detections'] += 1
                stats['total_detections'] += num_detections
                stats['detections_per_image'].append(num_detections)
                
                # Store annotated image
                annotated_img = results[0].plot()
                images_data.append((annotated_img, boxes))
                
                # Collect confidence scores and class counts
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    all_conf_scores.append(conf)
                    
                    if cls_id < len(class_names):
                        class_name = class_names[cls_id]
                        all_confidences[class_name].append(conf)
                        stats['class_counts'][class_name] += 1
            else:
                stats['images_without_detections'] += 1
                stats['detections_per_image'].append(0)
        else:
            stats['images_without_detections'] += 1
            stats['detections_per_image'].append(0)
    
    # Calculate averages
    if stats['total_detections'] > 0:
        stats['avg_detections'] = stats['total_detections'] / stats['total_images']
        stats['avg_confidence'] = np.mean(all_conf_scores)
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    if images_data:
        create_detection_grid(images_data, output_dir / 'detection_grid.png')
        plot_confidence_distribution(all_confidences, class_names, 
                                     output_dir / 'confidence_distribution.png')
        plot_detection_statistics(stats, output_dir / 'detection_statistics.png')
        plot_summary_report(stats, output_dir / 'summary_report.png')
    else:
        print("No detections found to visualize.")
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['avg_detections']:.2f}")
    if all_conf_scores:
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"{'='*60}\n")
    
    print(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    visualize_results(args)
