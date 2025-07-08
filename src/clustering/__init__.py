#!/usr/bin/env python3
"""
Clustering Module
Módulo de clustering gaussiano para re-identificación de personas
"""

from .gaussian_cluster_manager import GaussianClusterManager, GaussianCluster
from .hnsw_gaussian_integration import HNSWGaussianIntegration

__all__ = [
    'GaussianClusterManager',
    'GaussianCluster',
    'HNSWGaussianIntegration'
]

__version__ = "1.0.0"
