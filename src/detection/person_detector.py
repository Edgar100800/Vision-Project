"""
Person detection using YOLO11m-pose model.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

from ..utils.data_structures import Detection
from ..utils.config_loader import ConfigLoader


class PersonDetector:
    """Person detector using YOLO11m-pose model."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize person detector.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.detection_config = config.get_detection_config()
        
        # Load YOLO model
        model_path = config.get_model_path("yolo_pose")
        self.model = YOLO(model_path)
        
        # Detection parameters
        self.confidence_threshold = self.detection_config.get("confidence_threshold", 0.5)
        self.nms_threshold = self.detection_config.get("nms_threshold", 0.4)
        self.max_detections = self.detection_config.get("max_detections", 50)
        self.min_person_height = self.detection_config.get("min_person_height", 50)
        
        # Device setup
        self.device = self._setup_device()
        self.model.to(self.device)
        
        logging.info(f"PersonDetector initialized with device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup device for inference."""
        device_config = self.detection_config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            return device_config
    
    def detect(self, frame: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> List[Detection]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            List of person detections
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    # Get bounding box
                    box = boxes[i]
                    confidence = float(box.conf[0])
                    
                    # Filter by confidence
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    
                    # Filter by minimum height
                    height = y2 - y1
                    if height < self.min_person_height:
                        continue
                    
                    # Get pose keypoints if available
                    pose_keypoints = None
                    if keypoints is not None and i < len(keypoints):
                        kpts = keypoints[i]
                        if kpts is not None:
                            pose_keypoints = kpts.data[0].cpu().numpy()
                    
                    # Create detection
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        pose_keypoints=pose_keypoints,
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                    
                    detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit number of detections
        detections = detections[:self.max_detections]
        
        logging.debug(f"Detected {len(detections)} persons in frame {frame_id}")
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray], frame_ids: List[int] = None, timestamps: List[float] = None) -> List[List[Detection]]:
        """
        Detect persons in a batch of frames.
        
        Args:
            frames: List of input frames
            frame_ids: List of frame identifiers
            timestamps: List of frame timestamps
            
        Returns:
            List of detection lists for each frame
        """
        if frame_ids is None:
            frame_ids = list(range(len(frames)))
        if timestamps is None:
            timestamps = [0.0] * len(frames)
        
        all_detections = []
        
        for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps):
            detections = self.detect(frame, frame_id, timestamp)
            all_detections.append(detections)
        
        return all_detections
    
    def filter_detections(self, detections: List[Detection], min_area: float = 1000) -> List[Detection]:
        """
        Filter detections based on criteria.
        
        Args:
            detections: List of detections to filter
            min_area: Minimum bounding box area
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for detection in detections:
            area = detection.get_area()
            
            if area >= min_area:
                filtered.append(detection)
        
        return filtered
    
    def get_detection_statistics(self, detections: List[Detection]) -> dict:
        """
        Get statistics about detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_area': 0.0,
                'avg_aspect_ratio': 0.0
            }
        
        confidences = [d.confidence for d in detections]
        areas = [d.get_area() for d in detections]
        aspect_ratios = [d.get_aspect_ratio() for d in detections]
        
        return {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        } 