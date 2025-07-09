"""
Embedding extraction module for person re-identification.
"""

from .embedding_extractor import EmbeddingExtractor
from .feature_extractor import FeatureExtractor

__all__ = [
    'EmbeddingExtractor',
    'FeatureExtractor'
] 