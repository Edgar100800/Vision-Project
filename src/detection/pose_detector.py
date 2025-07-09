"""
Pose detection utilities for person re-identification.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

from ..utils.data_structures import Detection


class PoseDetector:
    """Pose detection utilities."""
    
    def __init__(self):
        """Initialize pose detector."""
        logging.info("PoseDetector initialized")
    
    def extract_pose_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract features from pose keypoints.
        
        Args:
            keypoints: Pose keypoints array
            
        Returns:
            Pose feature vector
        """
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(64)  # Default pose feature dimension
        
        # Extract keypoint positions and confidences
        positions = keypoints[:, :2]  # x, y coordinates
        confidences = keypoints[:, 2]  # confidence scores
        
        # Normalize positions
        if np.max(positions) > 0:
            positions = positions / np.max(positions)
        
        # Create pose features
        pose_features = []
        
        # Keypoint positions
        pose_features.extend(positions.flatten())
        
        # Confidence scores
        pose_features.extend(confidences)
        
        # Keypoint distances (pairwise)
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            pose_features.extend(distances[:10])  # Limit to first 10 distances
        
        # Pad or truncate to fixed size
        target_size = 64
        if len(pose_features) < target_size:
            pose_features.extend([0] * (target_size - len(pose_features)))
        else:
            pose_features = pose_features[:target_size]
        
        return np.array(pose_features)
    
    def get_pose_confidence(self, keypoints: np.ndarray) -> float:
        """
        Get overall pose confidence.
        
        Args:
            keypoints: Pose keypoints array
            
        Returns:
            Average confidence score
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        confidences = keypoints[:, 2]
        return np.mean(confidences)
    
    def get_pose_center(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get center of pose keypoints.
        
        Args:
            keypoints: Pose keypoints array
            
        Returns:
            Center coordinates (x, y) or None
        """
        if keypoints is None or len(keypoints) == 0:
            return None
        
        positions = keypoints[:, :2]
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        
        return (center_x, center_y) 