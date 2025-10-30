#!/usr/bin/env python3
"""Run inference with a trained YOLO model for traffic sign detection.

This script loads a trained model and runs prediction on images or video,
saving the annotated results to an output directory.
"""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run traffic sign detection")
    p.add_argument("--model", required=True,
                   help="Path to trained model weights (e.g., best.pt)")
    p.add_argument("--source", required=True,
                   help="Path to image/video file or directory")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference size (pixels)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--device", default="0",
                   help="cuda device='0' or 'cpu'")
    p.add_argument("--save-txt", action="store_true",
                   help="Save results as .txt files")
    p.add_argument("--output", default="runs/detect",
                   help="Path to output directory")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics not found. Install with: pip install ultralytics") from e

    # Load the model
    model = YOLO(args.model)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference:")
    print(f"- Model: {args.model}")
    print(f"- Source: {args.source}")
    print(f"- Output: {output_dir}")
    print(f"- Confidence threshold: {args.conf}")

    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=str(output_dir),
        name="results"
    )

    print(f"Inference complete. Results saved to: {output_dir / 'results'}")


if __name__ == "__main__":
    main()
