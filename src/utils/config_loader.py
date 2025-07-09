"""
Configuration loader for the person re-identification system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration settings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get model path from configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model file
        """
        model_path = self.get(f"models.{model_name}")
        if not model_path:
            raise ValueError(f"Model path not found for: {model_name}")
        
        # Check if path is relative and make it absolute
        if not os.path.isabs(model_path):
            model_path = Path.cwd() / model_path
        
        return str(model_path)
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.get("detection", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self.get("embedding", {})
    
    def get_reid_config(self) -> Dict[str, Any]:
        """Get re-identification configuration."""
        return self.get("reid", {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration."""
        return self.get("tracking", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get("visualization", {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """Get video processing configuration."""
        return self.get("video", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get("performance", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
        """
        required_keys = [
            "models.yolo_pose",
            "detection.confidence_threshold",
            "embedding.embedding_dim",
            "reid.similarity_threshold"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Required configuration key missing: {key}")
        
        return True 