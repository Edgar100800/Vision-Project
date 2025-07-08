#!/usr/bin/env python3
"""
OSNet ReID Module for Person Re-identification
Uses OSNet model from torchreid library to extract person features
"""

from config.global_config import get_config
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Try to import torchreid - handle different versions
try:
    import torchreid
    from torchreid import models
    # Access FeatureExtractor through dot notation instead of direct import
    FeatureExtractor = torchreid.utils.FeatureExtractor
    TORCHREID_AVAILABLE = True
except ImportError:
    print("Warning: torchreid not available. Using fallback implementation.")
    TORCHREID_AVAILABLE = False


class OSNetReID:
    """OSNet-based person re-identification module"""

    def __init__(self, model_name: str = None,
                 model_path: Optional[str] = None,
                 device: str = None,
                 image_size: Tuple[int, int] = None):
        """
        Initialize OSNet ReID model

        Args:
            model_name: Name of the OSNet model (uses config default if None)
            model_path: Path to pretrained model weights (uses config default if None)
            device: Device to use (uses config default if None)
            image_size: Input image size (uses config default if None)
        """
        # Get configuration
        self.config = get_config('reid')

        # Use provided values or defaults from config
        self.model_name = model_name or self.config['model_name']
        self.model_path = model_path or self.config['model_path']
        self.image_size = image_size or self.config['image_size']

        # Store other config values
        self.feature_dim = self.config['feature_dim']
        self.normalize_mean = self.config['normalize_mean']
        self.normalize_std = self.config['normalize_std']
        self.normalize_features = self.config['normalize_features']
        self.batch_size = self.config['batch_size']
        self.default_metric = self.config['default_metric']
        self.similarity_threshold = self.config['similarity_threshold']
        self.distance_threshold = self.config['distance_threshold']
        self.min_crop_size = self.config['min_crop_size']
        self.max_crop_size = self.config['max_crop_size']
        self.aspect_ratio_range = self.config['aspect_ratio_range']
        self.blur_threshold = self.config['blur_threshold']
        self.fallback_model = self.config['fallback_model']
        self.fallback_pretrained = self.config['fallback_pretrained']
        self.fallback_dropout = self.config['fallback_dropout']

        # Set device
        device_config = device or self.config['device']
        if device_config == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = None
        self.extractor = None
        self.transform = None

        self.setup_model()
        self.setup_transforms()

    def setup_model(self):
        """Setup OSNet model"""
        if TORCHREID_AVAILABLE:
            self.setup_torchreid_model()
        else:
            self.setup_fallback_model()

    def setup_torchreid_model(self):
        """Setup model using torchreid library"""
        try:
            # Create feature extractor
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.model_path,
                device=str(self.device)
            )

            print(f"Successfully loaded {self.model_name} using torchreid")

        except Exception as e:
            print(f"Error loading model with torchreid: {e}")
            print("Falling back to manual model setup...")
            self.setup_fallback_model()

    def setup_fallback_model(self):
        """Setup fallback model (simplified ResNet-based)"""
        print("Using fallback ResNet-based model for ReID")

        # Create a simple ResNet-based model
        import torchvision.models as models

        if self.fallback_model == 'resnet50':
            self.model = models.resnet50(pretrained=self.fallback_pretrained)
            feature_dim = 2048
        elif self.fallback_model == 'resnet101':
            self.model = models.resnet101(pretrained=self.fallback_pretrained)
            feature_dim = 2048
        else:
            self.model = models.resnet50(pretrained=self.fallback_pretrained)
            feature_dim = 2048

        # Replace the final classification layer with a feature layer
        self.model.fc = nn.Sequential(
            nn.Dropout(self.fallback_dropout),
            nn.Linear(feature_dim, self.feature_dim)
        )

        self.model.to(self.device)
        self.model.eval()

    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean,
                                 std=self.normalize_std)
        ])

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for ReID model

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def extract_features(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract features from person images

        Args:
            images: Single image or list of images (numpy arrays)

        Returns:
            Feature vectors (N x 2048)
        """
        if isinstance(images, np.ndarray):
            images = [images]

        features = []

        with torch.no_grad():
            for image in images:
                if image.size == 0:  # Skip empty images
                    features.append(np.zeros(2048))
                    continue

                # Preprocess image
                tensor = self.preprocess_image(image)

                # Extract features
                if self.extractor is not None:
                    # Use torchreid extractor
                    feature = self.extractor(tensor)
                    if isinstance(feature, torch.Tensor):
                        feature = feature.cpu().numpy()
                else:
                    # Use fallback model
                    feature = self.model(tensor)
                    feature = feature.cpu().numpy()

                # Normalize features
                feature = feature.flatten()
                if self.normalize_features:
                    feature = feature / (np.linalg.norm(feature) + 1e-8)

                features.append(feature)

        return np.array(features)

    def extract_features_from_detections(self, frame: np.ndarray,
                                         detections: List[Dict]) -> List[np.ndarray]:
        """
        Extract features from person detections in a frame

        Args:
            frame: Input frame (numpy array)
            detections: List of detection dictionaries with 'bbox' key

        Returns:
            List of feature vectors
        """
        person_crops = []

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Extract person crop
            crop = frame[y1:y2, x1:x2]

            # Skip if crop is too small
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                person_crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            else:
                person_crops.append(crop)

        # Extract features from all crops
        if person_crops:
            features = self.extract_features(person_crops)
            return [feat for feat in features]
        else:
            return []

    def compute_similarity(self, features1: np.ndarray,
                           features2: np.ndarray,
                           metric: str = 'cosine') -> float:
        """
        Compute similarity between two feature vectors

        Args:
            features1: First feature vector
            features2: Second feature vector
            metric: Similarity metric ('cosine', 'euclidean')

        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            return dot_product / (norm1 * norm2 + 1e-8)

        elif metric == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(features1 - features2)
            return 1.0 / (1.0 + distance)

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def compute_distance_matrix(self, features1: np.ndarray,
                                features2: np.ndarray,
                                metric: str = 'cosine') -> np.ndarray:
        """
        Compute distance matrix between two sets of features

        Args:
            features1: First set of features (N x D)
            features2: Second set of features (M x D)
            metric: Distance metric ('cosine', 'euclidean')

        Returns:
            Distance matrix (N x M)
        """
        if metric == 'cosine':
            # Cosine distance
            # Normalize features
            features1_norm = features1 / \
                (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
            features2_norm = features2 / \
                (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

            # Compute cosine similarity
            similarity = np.dot(features1_norm, features2_norm.T)

            # Convert to distance
            distance = 1.0 - similarity

        elif metric == 'euclidean':
            # Euclidean distance
            distance = np.sqrt(
                ((features1[:, np.newaxis] - features2) ** 2).sum(axis=2))

        else:
            raise ValueError(f"Unknown distance metric: {metric}")

        return distance

    def save_features(self, features: np.ndarray, filepath: str):
        """Save features to file"""
        np.save(filepath, features)
        print(f"Features saved to {filepath}")

    def load_features(self, filepath: str) -> np.ndarray:
        """Load features from file"""
        features = np.load(filepath)
        print(f"Features loaded from {filepath}")
        return features

    def extract_features_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from batch of images (alias for extract_features)

        Args:
            images: List of images (numpy arrays)

        Returns:
            List of feature vectors
        """
        features = self.extract_features(images)
        return [features[i] for i in range(len(features))]


def main():
    """Example usage"""
    # Initialize ReID model
    reid_model = OSNetReID(model_name='osnet_x1_0')

    # Create dummy person images
    dummy_images = [
        np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8),
        np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8),
        np.random.randint(0, 255, (140, 70, 3), dtype=np.uint8),
    ]

    # Extract features
    print("Extracting features from dummy images...")
    features = reid_model.extract_features(dummy_images)

    print(f"Extracted features shape: {features.shape}")
    print(f"Feature vector dimension: {features.shape[1]}")

    # Compute similarity between first two features
    similarity = reid_model.compute_similarity(features[0], features[1])
    print(f"Similarity between image 0 and 1: {similarity:.4f}")

    # Compute distance matrix
    distance_matrix = reid_model.compute_distance_matrix(
        features[:2], features[1:])
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance matrix:\n{distance_matrix}")

    # Test with frame and detections
    print("\nTesting with frame and detections...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = [
        {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
        {'bbox': [300, 150, 400, 350], 'confidence': 0.8},
    ]

    detection_features = reid_model.extract_features_from_detections(
        frame, detections)
    print(f"Extracted {len(detection_features)} features from detections")

    for i, feat in enumerate(detection_features):
        print(f"Detection {i} feature shape: {feat.shape}")


if __name__ == "__main__":
    main()
