"""
Real-time Defect Detection Inference Script

Production-ready inference script for metal surface defect detection with support
for various input sources (images, video, webcam, directory) and configurable output options.

Usage:
    # Single image
    python detect.py --source image.jpg --weights models/best.pt
    
    # Directory of images
    python detect.py --source path/to/images/ --weights models/best.pt
    
    # Video file
    python detect.py --source video.mp4 --weights models/best.pt
    
    # Webcam (real-time)
    python detect.py --source 0 --weights models/best.pt
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv5 inference for metal surface defect detection'
    )
    
    # Input configuration
    parser.add_argument('--source', type=str, required=True,
                        help='Input source (image, video, directory, or webcam index)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    
    # Inference parameters
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='Maximum number of detections per image')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='runs/detect',
                        help='Output directory for results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detection results (images/videos)')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results in YOLO format txt files')
    parser.add_argument('--save-conf', action='store_true',
                        help='Include confidence scores in txt files')
    parser.add_argument('--nosave', action='store_true',
                        help='Do not save images/videos')
    parser.add_argument('--view-img', action='store_true',
                        help='Display results in window')
    
    # Visualization parameters
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Bounding box line thickness')
    parser.add_argument('--hide-labels', action='store_true',
                        help='Hide class labels')
    parser.add_argument('--hide-conf', action='store_true',
                        help='Hide confidence scores')
    
    # Performance monitoring
    parser.add_argument('--show-fps', action='store_true',
                        help='Display FPS on output')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode (report inference time statistics)')
    
    return parser.parse_args()


def get_device(device_arg=''):
    """
    Detect and return the best available device for inference.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
    
    Args:
        device_arg: User-specified device string
        
    Returns:
        Device string suitable for PyTorch/YOLO
    """
    if device_arg:
        return device_arg
    
    if torch.cuda.is_available():
        return '0'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def create_output_dir(base_path, name):
    """
    Create output directory with unique name.
    
    Args:
        base_path: Base output path
        name: Experiment name
        
    Returns:
        Path object for output directory
    """
    output_dir = Path(base_path) / name
    counter = 1
    original_name = name
    
    while output_dir.exists():
        name = f"{original_name}{counter}"
        output_dir = Path(base_path) / name
        counter += 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_input_type(source):
    """
    Determine input type from source.
    
    Args:
        source: Input source string
        
    Returns:
        Input type: 'image', 'video', 'directory', or 'webcam'
    """
    if source.isdigit():
        return 'webcam'
    
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            return 'image'
        elif path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video'
    elif path.is_dir():
        return 'directory'
    
    raise ValueError(f"Invalid source: {source}")


def process_image(model, img_path, args, output_dir, save_results=True):
    """
    Process single image.
    
    Args:
        model: YOLO model
        img_path: Path to image
        args: Command line arguments
        output_dir: Output directory
        save_results: Whether to save results
        
    Returns:
        Inference time in milliseconds
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return 0
    
    # Run inference
    start_time = time.time()
    results = model(img, 
                   imgsz=args.img_size,
                   conf=args.conf_thres,
                   iou=args.iou_thres,
                   max_det=args.max_det,
                   verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    # Get detections
    result = results[0]
    num_detections = len(result.boxes) if result.boxes is not None else 0
    
    print(f"Processed: {img_path.name} | Detections: {num_detections} | "
          f"Time: {inference_time:.1f}ms")
    
    # Save results
    if save_results and not args.nosave:
        # Save annotated image
        annotated_img = result.plot(
            line_width=args.line_thickness,
            labels=not args.hide_labels,
            conf=not args.hide_conf
        )
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), annotated_img)
        
        # Save txt file if requested
        if args.save_txt and result.boxes is not None:
            txt_path = output_dir / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Convert to YOLO format (normalized xywh)
                    img_h, img_w = img.shape[:2]
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                    width = (xyxy[2] - xyxy[0]) / img_w
                    height = (xyxy[3] - xyxy[1]) / img_h
                    
                    if args.save_conf:
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} "
                               f"{width:.6f} {height:.6f} {conf:.4f}\n")
                    else:
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} "
                               f"{width:.6f} {height:.6f}\n")
    
    # Display if requested
    if args.view_img:
        annotated_img = result.plot()
        cv2.imshow('Detection', annotated_img)
        cv2.waitKey(1)
    
    return inference_time


def process_video(model, video_path, args, output_dir, save_results=True):
    """
    Process video file or webcam stream.
    
    Args:
        model: YOLO model
        video_path: Path to video or webcam index
        args: Command line arguments
        output_dir: Output directory
        save_results: Whether to save results
    """
    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        is_webcam = True
    else:
        cap = cv2.VideoCapture(str(video_path))
        is_webcam = False
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    if not is_webcam:
        print(f"  Total frames: {total_frames}")
    print()
    
    # Setup video writer
    video_writer = None
    if save_results and not args.nosave and not is_webcam:
        output_path = output_dir / f"{Path(video_path).stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    inference_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            start_time = time.time()
            results = model(frame,
                          imgsz=args.img_size,
                          conf=args.conf_thres,
                          iou=args.iou_thres,
                          max_det=args.max_det,
                          verbose=False)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            # Get annotated frame
            result = results[0]
            annotated_frame = result.plot(
                line_width=args.line_thickness,
                labels=not args.hide_labels,
                conf=not args.hide_conf
            )
            
            # Add FPS overlay if requested
            if args.show_fps:
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)
            
            # Save frame
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Display frame
            if args.view_img or is_webcam:
                cv2.imshow('Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Print progress
            if not is_webcam and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(inference_times[-30:])
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"Avg time: {avg_time:.1f}ms | FPS: {1000/avg_time:.1f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if inference_times:
            print(f"\n{'='*60}")
            print("Processing Statistics:")
            print(f"{'='*60}")
            print(f"Frames processed: {frame_count}")
            print(f"Average inference time: {np.mean(inference_times):.1f}ms")
            print(f"Average FPS: {1000/np.mean(inference_times):.1f}")
            print(f"Min/Max inference time: {np.min(inference_times):.1f}ms / "
                  f"{np.max(inference_times):.1f}ms")
            print(f"{'='*60}\n")


def run_inference(args):
    """
    Main inference function.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print("Metal Surface Defect Detection - Inference")
    print(f"{'='*60}\n")
    
    # Detect and configure device
    device = get_device(args.device)
    args.device = device
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    print(f"Model loaded successfully\n")
    
    # Create output directory
    save_results = args.save_results or args.save_txt
    output_dir = None
    if save_results:
        output_dir = create_output_dir(args.output, args.name)
        print(f"Output directory: {output_dir}\n")
    
    # Determine input type and process
    input_type = get_input_type(args.source)
    print(f"Input type: {input_type}")
    print(f"Source: {args.source}\n")
    
    if input_type == 'image':
        inference_time = process_image(model, Path(args.source), args, 
                                       output_dir, save_results)
        
    elif input_type == 'directory':
        source_dir = Path(args.source)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(source_dir.glob(ext))
        
        print(f"Found {len(image_files)} images\n")
        
        inference_times = []
        for img_path in image_files:
            inf_time = process_image(model, img_path, args, output_dir, save_results)
            inference_times.append(inf_time)
        
        # Print statistics
        if inference_times:
            print(f"\n{'='*60}")
            print("Processing Statistics:")
            print(f"{'='*60}")
            print(f"Images processed: {len(inference_times)}")
            print(f"Average inference time: {np.mean(inference_times):.1f}ms")
            print(f"Average FPS: {1000/np.mean(inference_times):.1f}")
            print(f"{'='*60}\n")
    
    elif input_type in ['video', 'webcam']:
        process_video(model, args.source, args, output_dir, save_results)
    
    if save_results and output_dir:
        print(f"Results saved to: {output_dir}")
    
    print("\nInference complete!")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print system info
    print(f"\n{'='*60}")
    print("System Information")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA available: No")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    
    print(f"{'='*60}\n")
    
    # Run inference
    run_inference(args)


if __name__ == '__main__':
    main()
