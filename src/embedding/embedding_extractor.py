"""
Embedding extraction for person re-identification.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Optional, Tuple
import logging

from ..utils.data_structures import Detection, Person
from ..utils.config_loader import ConfigLoader


class EmbeddingExtractor:
    """Extract robust embeddings from person detections."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize embedding extractor.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.embedding_config = config.get_embedding_config()
        
        # Model parameters
        self.model_type = self.embedding_config.get("model_type", "resnet50")
        self.embedding_dim = self.embedding_config.get("embedding_dim", 512)
        self.input_size = self.embedding_config.get("input_size", [256, 128])
        self.normalize = self.embedding_config.get("normalize", True)
        self.use_pose_features = self.embedding_config.get("use_pose_features", True)
        
        # Initialize model
        self.model = self._build_model()
        self.transform = self._build_transform()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"EmbeddingExtractor initialized with {self.model_type}")
    
    def _build_model(self) -> nn.Module:
        """Build the embedding model."""
        if self.model_type == "resnet50":
            return self._build_resnet50()
        elif self.model_type == "transformer":
            return self._build_transformer()
        elif self.model_type == "efficientnet":
            return self._build_efficientnet()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _build_resnet50(self) -> nn.Module:
        """Build ResNet50-based embedding model."""
        import torchvision.models as models
        
        # Load pre-trained ResNet50 with new API
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the last classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Add embedding layer
        embedding_model = nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        return embedding_model
    
    def _build_transformer(self) -> nn.Module:
        """Build Transformer-based embedding model."""
        from transformers import ViTModel, ViTConfig
        
        config = ViTConfig(
            image_size=self.input_size[0],
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=False,
            image_grid_pin_layout=None,
            use_cache=True,
            num_labels=1000,
            id2label=None,
            label2id=None,
        )
        
        model = ViTModel(config)
        
        # Add embedding head
        embedding_model = nn.Sequential(
            model,
            nn.Linear(768, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        return embedding_model
    
    def _build_efficientnet(self) -> nn.Module:
        """Build EfficientNet-based embedding model."""
        import timm
        
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Add embedding head
        embedding_model = nn.Sequential(
            model,
            nn.Linear(model.num_features, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        return embedding_model
    
    def _build_transform(self) -> transforms.Compose:
        """Build image transformation pipeline."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_embedding(self, person_crop: np.ndarray, pose_keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract embedding from person crop.
        
        Args:
            person_crop: Person image crop (BGR format)
            pose_keypoints: Pose keypoints if available
            
        Returns:
            Embedding vector
        """
        # Convert BGR to RGB
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        input_tensor = self.transform(person_crop_rgb).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize if required
        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Combine with pose features if available
        if self.use_pose_features and pose_keypoints is not None:
            pose_features = self._extract_pose_features(pose_keypoints)
            embedding = np.concatenate([embedding, pose_features])
        
        return embedding
    
    def _extract_pose_features(self, pose_keypoints: np.ndarray) -> np.ndarray:
        """
        Extract features from pose keypoints.
        
        Args:
            pose_keypoints: Pose keypoints array
            
        Returns:
            Pose feature vector
        """
        if pose_keypoints is None or len(pose_keypoints) == 0:
            return np.zeros(64)  # Default pose feature dimension
        
        # Extract keypoint positions and confidences
        positions = pose_keypoints[:, :2]  # x, y coordinates
        confidences = pose_keypoints[:, 2]  # confidence scores
        
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
    
    def extract_embeddings_batch(self, person_crops: List[np.ndarray], pose_keypoints: List[Optional[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Extract embeddings from a batch of person crops.
        
        Args:
            person_crops: List of person image crops
            pose_keypoints: List of pose keypoints
            
        Returns:
            List of embedding vectors
        """
        if pose_keypoints is None:
            pose_keypoints = [None] * len(person_crops)
        
        embeddings = []
        
        for crop, pose in zip(person_crops, pose_keypoints):
            embedding = self.extract_embedding(crop, pose)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine") -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method ("cosine", "euclidean", "manhattan")
            
        Returns:
            Similarity score
        """
        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2 + 1e-8)
        
        elif method == "euclidean":
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        
        elif method == "manhattan":
            # Manhattan distance (convert to similarity)
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of extracted embeddings."""
        base_dim = self.embedding_dim
        if self.use_pose_features:
            base_dim += 64  # Pose feature dimension
        return base_dim 