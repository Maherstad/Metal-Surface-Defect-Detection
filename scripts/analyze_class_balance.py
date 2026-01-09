"""
Analyze and visualize class imbalance in the dataset.

Usage:
    python scripts/analyze_class_balance.py
"""

import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def analyze_class_distribution(data_yaml='yolov5/custom_dataset.yaml'):
    """Analyze and visualize class distribution."""
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get directories - handle relative paths
    data_path_str = data['path']
    data_path = Path(data_path_str)
    
    # If path starts with ../, resolve it relative to the yaml file location
    if data_path_str.startswith('../'):
        yaml_dir = Path(data_yaml).parent
        data_path = (yaml_dir / data_path).resolve()
    elif not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    
    # Construct paths
    train_path = data['train'] if 'train' in data and data['train'] else 'train'
    val_path = data['val'] if 'val' in data and data['val'] else 'val'
    
    train_labels = data_path / train_path / 'labels'
    val_labels = data_path / val_path / 'labels'
    
    print(f"Analyzing: {train_labels}")
    print(f"Analyzing: {val_labels}")
    
    # Count instances
    def count_classes(label_dir):
        class_counts = {}
        if not label_dir.exists():
            print(f"Warning: {label_dir} not found")
            return class_counts
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return class_counts
    
    train_counts = count_classes(train_labels)
    val_counts = count_classes(val_labels)
    
    # Get class names
    names = data['names']
    num_classes = len(names)
    
    # Print statistics
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print("\nTRAINING SET:")
    print("-"*70)
    train_total = sum(train_counts.values())
    for i in range(num_classes):
        count = train_counts.get(i, 0)
        percent = (count / train_total) * 100 if train_total > 0 else 0
        bar = "█" * int(percent / 2)
        print(f"{i}: {names[i]:20s} | {count:5d} ({percent:5.1f}%) {bar}")
    
    print("\nVALIDATION SET:")
    print("-"*70)
    val_total = sum(val_counts.values())
    for i in range(num_classes):
        count = val_counts.get(i, 0)
        percent = (count / val_total) * 100 if val_total > 0 else 0
        bar = "█" * int(percent / 2)
        print(f"{i}: {names[i]:20s} | {count:5d} ({percent:5.1f}%) {bar}")
    
    # Calculate imbalance metrics
    if not train_counts:
        print("\n⚠️  No training data found!")
        return
    
    max_train = max(train_counts.values())
    min_train = min(train_counts.values())
    
    print("\n" + "="*70)
    print("IMBALANCE METRICS")
    print("="*70)
    print(f"Total training instances: {train_total}")
    print(f"Total validation instances: {val_total}")
    print(f"Imbalance ratio: {max_train/min_train:.1f}:1")
    print(f"Most common: {names[max(train_counts, key=train_counts.get)]} ({max_train} instances)")
    print(f"Least common: {names[min(train_counts, key=train_counts.get)]} ({min_train} instances)")
    
    # Calculate recommended class weights
    print("\n" + "="*70)
    print("RECOMMENDED CLASS WEIGHTS (Inverse Frequency)")
    print("="*70)
    weights = []
    for i in range(num_classes):
        count = train_counts.get(i, 1)
        weight = train_total / (num_classes * count)
        weights.append(weight)
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.mean()
    
    for i in range(num_classes):
        print(f"Class {i} ({names[i]:20s}): weight = {weights[i]:.3f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if max_train / min_train > 10:
        print("⚠️  SEVERE IMBALANCE DETECTED (>10:1)")
        print("\nRecommended training approach:")
        print("1. Use --class-weights flag for automatic balancing")
        print("2. Increase data augmentation for minority classes")
        print("3. Consider collecting more data for rare defects")
        print("\nExample command:")
        print("python train.py --data yolov5/custom_dataset.yaml \\")
        print("                --freeze-backbone \\")
        print("                --class-weights \\")
        print("                --epochs 100")
    elif max_train / min_train > 5:
        print("⚠️  MODERATE IMBALANCE DETECTED (5:1 - 10:1)")
        print("\nRecommended: Use --class-weights flag")
    else:
        print("✓ Dataset is relatively balanced")
    
    print("="*70 + "\n")
    
    # Create visualization
    create_distribution_plot(train_counts, val_counts, names)


def create_distribution_plot(train_counts, val_counts, names):
    """Create and save class distribution plot."""
    
    num_classes = len(names)
    x = np.arange(num_classes)
    width = 0.35
    
    train_values = [train_counts.get(i, 0) for i in range(num_classes)]
    val_values = [val_counts.get(i, 0) for i in range(num_classes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    ax1.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    ax1.bar(x + width/2, val_values, width, label='Val', alpha=0.8)
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Instances')
    ax1.set_title('Class Distribution - Train vs Val')
    ax1.set_xticks(x)
    ax1.set_xticklabels([names[i] for i in range(num_classes)], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart for training set
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    ax2.pie(train_values, labels=[names[i] for i in range(num_classes)], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Training Set Class Distribution')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path('results/class_distribution.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    

if __name__ == '__main__':
    analyze_class_distribution()
