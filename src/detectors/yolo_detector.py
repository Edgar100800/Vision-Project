"""
YOLO-based person detector for the store monitoring system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DetectionConfig
from utils.logger import setup_logger


class YOLODetector:
    """
    Person detector using YOLO model.
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize the YOLO detector.

        Args:
            config: Detection configuration
        """
        self.config = config
        self.logger = setup_logger("YOLODetector")

        # Set device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.logger.info(f"Using device: {self.device}")

        # Load YOLO model
        try:
            self.model = YOLO(config.model_name)
            self.model.to(self.device)
            self.logger.info(f"Loaded YOLO model: {config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections as (x1, y1, x2, y2, confidence)
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.classes,
                verbose=False
            )

            detections = []

            if results and len(results) > 0:
                boxes = results[0].boxes

                if boxes is not None:
                    # Extract bounding boxes and confidences
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        confidence = float(confidence)

                        detections.append((x1, y1, x2, y2, confidence))

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def detect_and_draw(self, frame: np.ndarray, draw_boxes: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        Detect persons and optionally draw bounding boxes.

        Args:
            frame: Input frame
            draw_boxes: Whether to draw bounding boxes

        Returns:
            Tuple of (annotated_frame, detections)
        """
        detections = self.detect(frame)

        if draw_boxes:
            annotated_frame = frame.copy()

            for x1, y1, x2, y2, conf in detections:
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)

                # Draw confidence score
                label = f"Person: {conf:.2f}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

            return annotated_frame, detections

        return frame, detections

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "classes": self.config.classes
        }
