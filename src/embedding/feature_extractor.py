"""
Feature extraction utilities for person re-identification.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ..utils.data_structures import Detection


class FeatureExtractor:
    """Extract various features from person detections."""
    
    def __init__(self):
        """Initialize feature extractor."""
        logging.info("FeatureExtractor initialized")
    
    def extract_color_features(self, person_crop: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            person_crop: Person image crop
            bins: Number of histogram bins
            
        Returns:
            Color feature vector
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate features
        color_features = np.concatenate([h_hist, s_hist, v_hist])
        
        return color_features
    
    def extract_texture_features(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns.
        
        Args:
            person_crop: Person image crop
            
        Returns:
            Texture feature vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply LBP
        radius = 3
        n_points = 8 * radius
        
        lbp = self._local_binary_pattern(gray, n_points, radius)
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        
        return hist
    
    def _local_binary_pattern(self, image: np.ndarray, n_points: int, radius: int) -> np.ndarray:
        """Compute Local Binary Pattern."""
        # Implementation of LBP
        lbp = np.zeros_like(image)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                code = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if image[x, y] >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def extract_shape_features(self, detection: Detection) -> np.ndarray:
        """
        Extract shape features from detection.
        
        Args:
            detection: Person detection
            
        Returns:
            Shape feature vector
        """
        bbox = detection.bbox
        x1, y1, x2, y2 = bbox
        
        # Basic shape features
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        perimeter = 2 * (width + height)
        
        # Position features (normalized)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Confidence
        confidence = detection.confidence
        
        # Create feature vector
        shape_features = np.array([
            width, height, area, aspect_ratio, perimeter,
            center_x, center_y, confidence
        ])
        
        return shape_features
    
    def extract_motion_features(self, track_trajectory: List[tuple], window_size: int = 10) -> np.ndarray:
        """
        Extract motion features from track trajectory.
        
        Args:
            track_trajectory: List of (x, y, timestamp) tuples
            window_size: Window size for motion analysis
            
        Returns:
            Motion feature vector
        """
        if len(track_trajectory) < 2:
            return np.zeros(8)
        
        # Get recent trajectory points
        recent_trajectory = track_trajectory[-window_size:] if len(track_trajectory) >= window_size else track_trajectory
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(recent_trajectory)):
            x1, y1, t1 = recent_trajectory[i-1]
            x2, y2, t2 = recent_trajectory[i]
            
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                velocities.append([vx, vy])
        
        if not velocities:
            return np.zeros(8)
        
        velocities = np.array(velocities)
        
        # Motion features
        avg_velocity_x = np.mean(velocities[:, 0])
        avg_velocity_y = np.mean(velocities[:, 1])
        velocity_magnitude = np.mean(np.linalg.norm(velocities, axis=1))
        velocity_std = np.std(np.linalg.norm(velocities, axis=1))
        
        # Direction features
        if len(velocities) > 1:
            directions = np.arctan2(velocities[:, 1], velocities[:, 0])
            direction_std = np.std(directions)
            direction_change = np.mean(np.abs(np.diff(directions)))
        else:
            direction_std = 0
            direction_change = 0
        
        # Trajectory length
        trajectory_length = len(recent_trajectory)
        
        # Create feature vector
        motion_features = np.array([
            avg_velocity_x, avg_velocity_y, velocity_magnitude, velocity_std,
            direction_std, direction_change, trajectory_length, len(velocities)
        ])
        
        return motion_features
    
    def extract_combined_features(self, person_crop: np.ndarray, detection: Detection, 
                                track_trajectory: Optional[List[tuple]] = None) -> Dict[str, np.ndarray]:
        """
        Extract all features combined.
        
        Args:
            person_crop: Person image crop
            detection: Person detection
            track_trajectory: Track trajectory (optional)
            
        Returns:
            Dictionary with all feature vectors
        """
        features = {}
        
        # Color features
        features['color'] = self.extract_color_features(person_crop)
        
        # Texture features
        features['texture'] = self.extract_texture_features(person_crop)
        
        # Shape features
        features['shape'] = self.extract_shape_features(detection)
        
        # Motion features (if trajectory available)
        if track_trajectory:
            features['motion'] = self.extract_motion_features(track_trajectory)
        else:
            features['motion'] = np.zeros(8)
        
        return features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature type."""
        return {
            'color': 96,      # 32 bins * 3 channels
            'texture': 26,    # LBP histogram
            'shape': 8,       # Basic shape features
            'motion': 8       # Motion features
        } 