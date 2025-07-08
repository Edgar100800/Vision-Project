#!/usr/bin/env python3
"""
YOLO Inference Script for Person Detection
"""

from config.global_config import get_config
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class YOLODetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = None):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to trained YOLO model (uses config default if None)
            confidence_threshold: Minimum confidence threshold for detections (uses config default if None)
        """
        # Get configuration
        self.config = get_config('detection')

        # Use provided values or defaults from config
        self.model_path = model_path or self.config['model_path']
        self.confidence_threshold = confidence_threshold or self.config['confidence_threshold']

        # Store other config values
        self.iou_threshold = self.config['iou_threshold']
        self.input_size = self.config['input_size']
        self.target_classes = self.config['target_classes']
        self.max_detections = self.config['max_detections']
        self.device = self.config['device']
        self.verbose = self.config['verbose']

        self.model = None
        self.load_model()

    def load_model(self):
        """Load YOLO model"""
        if self.verbose:
            print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Configure model settings
        if self.device != 'auto':
            self.model.to(self.device)

    def detect_persons(self, image: np.ndarray,
                       return_crops: bool = False) -> Dict:
        """
        Detect persons in image

        Args:
            image: Input image as numpy array
            return_crops: Whether to return cropped person images

        Returns:
            Dictionary containing detection results
        """
        # Run inference
        results = self.model(image,
                             verbose=self.verbose,
                             imgsz=self.input_size,
                             conf=self.confidence_threshold,
                             iou=self.iou_threshold,
                             max_det=self.max_detections)

        detections = []
        crops = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Filter by confidence and target classes
                    if confidence >= self.confidence_threshold and class_id in self.target_classes:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': 'person'
                        }
                        detections.append(detection)

                        # Extract crop if requested
                        if return_crops:
                            crop = image[int(y1):int(y2), int(x1):int(x2)]
                            crops.append(crop)

        result_dict = {
            'detections': detections,
            'num_detections': len(detections)
        }

        if return_crops:
            result_dict['crops'] = crops

        return result_dict

    def detect_video(self, video_path: str,
                     output_path: Optional[str] = None,
                     save_crops: bool = False,
                     crops_dir: Optional[str] = None) -> List[Dict]:
        """
        Detect persons in video

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_crops: Whether to save person crops
            crops_dir: Directory to save crops

        Returns:
            List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Setup crops directory
        if save_crops and crops_dir:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)

        all_detections = []
        frame_idx = 0

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect persons in frame
            results = self.detect_persons(frame, return_crops=save_crops)

            # Add frame information
            frame_results = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'detections': results['detections'],
                'num_detections': results['num_detections']
            }

            all_detections.append(frame_results)

            # Save crops if requested
            if save_crops and crops_dir and 'crops' in results:
                for i, crop in enumerate(results['crops']):
                    crop_filename = f"frame_{frame_idx:06d}_person_{i:02d}.jpg"
                    crop_path = Path(crops_dir) / crop_filename
                    cv2.imwrite(str(crop_path), crop)

            # Draw bounding boxes on frame
            annotated_frame = self.draw_detections(
                frame, results['detections'])

            # Write frame to output video
            if writer:
                writer.write(annotated_frame)

            # Progress update
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

            frame_idx += 1

        # Cleanup
        cap.release()
        if writer:
            writer.release()

        print(
            f"Video processing completed. Total frames processed: {frame_idx}")

        return all_detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        annotated = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(
                annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Draw confidence score
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (bbox[0], bbox[1] - label_size[1] - 10),
                          (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(annotated, label, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated

    def save_detections(self, detections: List[Dict], output_path: str):
        """
        Save detection results to JSON file

        Args:
            detections: List of detection results
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)

        print(f"Detection results saved to: {output_path}")


def main():
    """Main inference function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLO Person Detection Inference')
    parser.add_argument('--model', required=True,
                        help='Path to trained YOLO model')
    parser.add_argument('--input', required=True,
                        help='Path to input video or image')
    parser.add_argument('--output', help='Path to output video or image')
    parser.add_argument('--confidence', type=float,
                        default=0.5, help='Confidence threshold')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save person crops')
    parser.add_argument('--crops-dir', default='person_crops',
                        help='Directory to save crops')
    parser.add_argument(
        '--save-json', help='Path to save detection results as JSON')

    args = parser.parse_args()

    # Initialize detector
    detector = YOLODetector(args.model, args.confidence)

    # Check if input is image or video
    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        image = cv2.imread(str(input_path))
        results = detector.detect_persons(image, return_crops=args.save_crops)

        print(f"Detected {results['num_detections']} persons in image")

        # Save crops if requested
        if args.save_crops:
            Path(args.crops_dir).mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(results['crops']):
                crop_filename = f"{input_path.stem}_person_{i:02d}.jpg"
                crop_path = Path(args.crops_dir) / crop_filename
                cv2.imwrite(str(crop_path), crop)

        # Save annotated image
        if args.output:
            annotated = detector.draw_detections(image, results['detections'])
            cv2.imwrite(args.output, annotated)
            print(f"Annotated image saved to: {args.output}")

    else:
        # Process video
        all_detections = detector.detect_video(
            args.input,
            args.output,
            args.save_crops,
            args.crops_dir
        )

        # Save JSON results if requested
        if args.save_json:
            detector.save_detections(all_detections, args.save_json)

        total_detections = sum(frame['num_detections']
                               for frame in all_detections)
        print(f"Total persons detected across all frames: {total_detections}")


if __name__ == "__main__":
    main()
