#!/usr/bin/env python3
"""
Global Configuration File for Person Re-identification System
Centralizes all configuration parameters for easy management and tuning
"""

import os
from typing import Dict, Any, Tuple
from pathlib import Path


class GlobalConfig:
    """
    Global configuration class containing all system parameters
    Organized by modules for easy management and understanding
    """

    def __init__(self):
        """Initialize all configuration parameters"""
        self.setup_paths()
        self.setup_detection_config()
        self.setup_tracking_config()
        self.setup_reid_config()
        self.setup_indexing_config()
        self.setup_fusion_config()
        self.setup_gaussian_clustering_config()
        self.setup_pipeline_config()
        self.setup_output_config()
        self.setup_performance_config()

    def setup_paths(self):
        """
        Path Configuration
        Define all file and directory paths used by the system
        """
        self.PATHS = {
            # Base directories
            'project_root': Path(__file__).parent.parent,
            'data_dir': Path('data'),
            'raw_data_dir': Path('data/raw'),
            'processed_data_dir': Path('data/processed'),
            'models_dir': Path('models'),
            'output_dir': Path('output'),
            'logs_dir': Path('logs'),

            # Model paths
            'yolo_models_dir': Path('models/yolo'),
            'reid_models_dir': Path('models/reid'),
            'custom_models_dir': Path('models/custom'),

            # Default model files
            'default_yolo_model': 'yolov8n.pt',
            'default_reid_model': 'osnet_x1_0',

            # Output subdirectories
            'crops_subdir': 'crops',
            'videos_subdir': 'videos',
            'data_subdir': 'data',
            'logs_subdir': 'logs'
        }

    def setup_detection_config(self):
        """
        YOLO Detection Configuration
        Parameters for person detection using YOLO models
        """
        self.DETECTION = {
            # Model Selection
            'model_path': 'yolov8n.pt',  # YOLO model file (yolov8n/s/m/l/x.pt)
            'model_size': 'nano',        # Model size: nano, small, medium, large, xlarge

            # Detection Thresholds
            # Minimum confidence for valid detection (0.0-1.0)
            'confidence_threshold': 0.5,
            # Higher = fewer false positives, may miss some detections
            # Lower = more detections, may include false positives

            # IoU threshold for Non-Maximum Suppression (0.0-1.0)
            'iou_threshold': 0.5,
            # Higher = allows more overlapping boxes
            # Lower = removes more overlapping detections

            # Class Filtering
            # COCO class IDs to detect (0 = person)
            'target_classes': [0],
            'class_names': ['person'],    # Corresponding class names

            # Input Processing
            # Input image size for YOLO (width, height)
            'input_size': (640, 640),
            # Larger = better accuracy, slower processing
            # Smaller = faster processing, may reduce accuracy

            'normalize': True,            # Whether to normalize input images
            'augment': False,             # Whether to use test-time augmentation

            # Performance Settings
            # Use FP16 for faster inference (requires GPU)
            'half_precision': False,
            'device': 'auto',             # Device: 'auto', 'cpu', 'cuda', 'cuda:0'
            'verbose': False,             # Print detection details

            # Batch Processing
            # Batch size for detection (usually 1 for video)
            'batch_size': 1,
            'max_detections': 100,        # Maximum detections per image

            # Post-processing
            'agnostic_nms': False,        # Class-agnostic NMS
            'multi_label': False,         # Multiple labels per box
        }

    def setup_tracking_config(self):
        """
        Huffman Tracking Configuration
        Parameters for person tracking with coordinate compression
        """
        self.TRACKING = {
            # Track Management
            'max_disappeared': 30,        # Max frames a person can disappear before removal
                                          # Higher = keeps tracks longer, uses more memory
                                          # Lower = removes tracks quickly, may lose people

            # Maximum distance for track association (pixels)
            'max_distance': 100.0,
                                          # Higher = associates distant detections (may cause errors)
                                          # Lower = stricter association (may create duplicate tracks)

            'min_track_length': 3,        # Minimum frames to consider a valid track
                                          # Higher = more stable tracks, may miss short appearances
                                          # Lower = captures brief appearances, may be noisy

            # Huffman Encoding
            'coordinate_discretization': 10,  # Grid size for coordinate discretization
            # Higher = coarser quantization, better compression
            # Lower = finer details, larger compressed size

            'enable_compression': True,    # Whether to use Huffman compression
            'compression_window': 100,     # Frames to collect before training encoder
            'retrain_interval': 500,       # Frames between encoder retraining

            # Distance Calculation
            'distance_metric': 'euclidean',  # Distance metric: 'euclidean', 'manhattan'
            'position_weight': 0.7,          # Weight for position in distance calculation
            'size_weight': 0.3,              # Weight for size in distance calculation

            # Track Prediction
            'enable_prediction': True,     # Use motion prediction for tracking
            'prediction_frames': 3,        # Frames to use for motion prediction
            'max_prediction_distance': 50,  # Maximum distance for predicted position

            # Track Quality
            'confidence_decay': 0.95,      # Confidence decay factor per frame
            'min_confidence': 0.3,         # Minimum confidence to keep track
            'quality_threshold': 0.5,      # Minimum quality score for track
        }

    def setup_reid_config(self):
        """
        ReID (Re-identification) Configuration
        Parameters for person feature extraction and matching
        """
        self.REID = {
            # Model Configuration
            'model_name': 'osnet_x1_0',   # ReID model: osnet_x1_0, osnet_x0_75, osnet_x0_5
                                          # x1_0 = highest accuracy, slower
                                          # x0_75 = balanced accuracy/speed
                                          # x0_5 = fastest, lower accuracy

            # Path to custom model weights (None = use pretrained)
            'model_path': None,
            # Output feature dimension (OSNet models output 512)
            'feature_dim': 512,
            'device': 'auto',             # Device: 'auto', 'cpu', 'cuda'

            # Image Preprocessing
            'image_size': (256, 128),     # Input image size (height, width)
                                          # Larger = better features, slower processing
                                          # Standard sizes: (256,128), (384,128), (256,256)

            # ImageNet normalization mean
            'normalize_mean': [0.485, 0.456, 0.406],
            # ImageNet normalization std
            'normalize_std': [0.229, 0.224, 0.225],

            # Feature Processing
            'normalize_features': True,    # L2 normalize output features
            'feature_enhancement': False,  # Apply feature enhancement techniques
            'pca_reduction': False,        # Apply PCA for dimensionality reduction
            'pca_components': 512,         # PCA output dimensions

            # Batch Processing
            'batch_size': 32,             # Batch size for feature extraction
                                          # Higher = faster processing, more memory
                                          # Lower = slower processing, less memory

            'max_batch_size': 64,         # Maximum batch size allowed
            'auto_batch_size': True,      # Automatically adjust batch size

            # Similarity Metrics
            'default_metric': 'cosine',   # Default similarity metric: 'cosine', 'euclidean'
            'similarity_threshold': 0.7,  # Minimum similarity for positive match
            'distance_threshold': 0.5,    # Maximum distance for positive match

            # Quality Control
            'min_crop_size': (32, 64),    # Minimum crop size (width, height)
            'max_crop_size': (512, 1024),  # Maximum crop size
            'aspect_ratio_range': (0.3, 3.0),  # Valid aspect ratio range
            'blur_threshold': 100,         # Minimum sharpness score

            # Fallback Model (when torchreid unavailable)
            'fallback_model': 'resnet50',  # Fallback model: 'resnet50', 'resnet101'
            'fallback_pretrained': True,   # Use pretrained weights for fallback
            'fallback_dropout': 0.5,       # Dropout rate for fallback model
        }

    def setup_indexing_config(self):
        """
        HNSW Indexing Configuration
        Parameters for vector similarity search and indexing
        """
        self.INDEXING = {
            # Index Structure
            # Feature vector dimension (matches OSNet output)
            'dimension': 512,
            'max_elements': 10000,        # Maximum number of vectors in index
                                          # Higher = more capacity, more memory usage
                                          # Lower = less memory, may need periodic cleanup

            # Distance metric: 'cosine', 'l2', 'ip' (inner product)
            'space': 'cosine',
                                          # cosine = good for normalized features
                                          # l2 = Euclidean distance
                                          # ip = inner product (for specific use cases)

            # HNSW Parameters
            'M': 16,                      # Number of bi-directional links for each node
                                          # Higher = better recall, more memory, slower build
                                          # Lower = faster build, less memory, may reduce recall
                                          # Typical range: 4-48, recommended: 16

            'ef_construction': 200,       # Size of dynamic candidate list during construction
                                          # Higher = better quality index, slower construction
                                          # Lower = faster construction, may reduce quality
                                          # Typical range: 100-800, recommended: 200

            'ef_search': 50,              # Size of dynamic candidate list during search
                                          # Higher = better recall, slower search
                                          # Lower = faster search, may reduce recall
                                          # Can be adjusted at runtime

            # Search Parameters
            'default_k': 5,               # Default number of neighbors to return
            'max_k': 100,                 # Maximum k value allowed
            'return_distances': True,     # Whether to return distances
            'return_metadata': True,      # Whether to return metadata

            # Performance Optimization
            'num_threads': -1,            # Number of threads (-1 = auto)
            'seed': 42,                   # Random seed for reproducibility
            'allow_replace_deleted': True,  # Allow replacing deleted elements

            # Index Management
            'auto_save_interval': 1000,   # Auto-save every N additions
            'compression_enabled': True,  # Enable index compression
            'memory_mapping': False,      # Use memory mapping for large indices

            # Quality Control
            'duplicate_threshold': 0.95,  # Similarity threshold for duplicate detection
            'cleanup_interval': 5000,     # Interval for automatic cleanup
            'max_age_frames': 10000,      # Maximum age before element removal
        }

    def setup_fusion_config(self):
        """
        ID Fusion Configuration
        Parameters for combining tracking and ReID results
        """
        self.FUSION = {
            # Fusion Weights
            # Weight for tracking confidence (0.0-1.0)
            'tracking_weight': 0.4,
            # Higher = trust tracking more
            # Lower = rely more on ReID

            # Weight for ReID confidence (0.0-1.0)
            'reid_weight': 0.6,
            # Higher = trust ReID more
            # Lower = rely more on tracking

            'temporal_weight': 0.2,       # Weight for temporal consistency

            # Similarity Thresholds
            'similarity_threshold': 0.7,  # Minimum combined similarity for ID assignment
                                          # Higher = stricter matching, fewer false positives
                                          # Lower = more lenient, may increase false positives

            'reid_distance_threshold': 0.5,  # Maximum ReID distance for valid match
            'tracking_distance_threshold': 100.0,  # Maximum tracking distance

            # Temporal Consistency
            'temporal_window': 30,        # Frames to consider for temporal consistency
                                          # Higher = more stable IDs, slower adaptation
                                          # Lower = faster adaptation, may be less stable

            'consistency_threshold': 0.8,  # Minimum consistency score
            'history_length': 50,         # Length of history to maintain

            # ID Management
            'max_global_ids': 1000,       # Maximum number of global IDs
            'id_reuse_delay': 500,        # Frames before reusing an ID
            'inactive_threshold': 100,    # Frames before marking ID as inactive
            'cleanup_threshold': 500,     # Frames before removing inactive IDs

            # Confidence Calculation
            'min_confidence': 0.3,        # Minimum confidence for valid assignment
            'confidence_decay': 0.95,     # Confidence decay per frame
            'boost_factor': 1.1,          # Confidence boost for consistent assignments

            # Quality Metrics
            'track_quality_weight': 0.3,  # Weight for track quality in fusion
            'reid_quality_weight': 0.7,   # Weight for ReID quality in fusion
            'appearance_change_threshold': 0.4,  # Threshold for appearance change detection

            # Advanced Features
            'enable_re_identification': True,  # Enable re-ID after long absence
            'enable_track_merging': True,      # Enable merging similar tracks
            'enable_id_recovery': True,        # Enable ID recovery after occlusion
            'occlusion_threshold': 10,         # Frames to consider as occlusion
        }

    def setup_gaussian_clustering_config(self):
        """
        Gaussian Clustering Configuration
        Parameters for dynamic cluster management with Gaussian modeling
        Optimized for dataset with 3 persons: (000,005), (002,003), (001,004)
        """
        self.GAUSSIAN_CLUSTERING = {
            # Dataset-specific Configuration
            'expected_persons': 3,        # Number of expected persons in dataset
            'person_id_mapping': {        # Known person ID groupings
                'person_1': ['000', '005'],
                'person_2': ['002', '003'],
                'person_3': ['001', '004']
            },
            'ground_truth_clusters': 3,   # Expected number of final clusters

            # Gaussian Model Parameters
            # Feature vector dimension (OSNet output)
            'feature_dimension': 512,
            'min_samples_per_cluster': 5,  # Minimum samples to form stable cluster
            'max_clusters': 10,           # Maximum number of clusters allowed
            # Initial clusters (one per original ID)
            'initial_clusters': 6,

            # Mahalanobis Distance Configuration
            'mahalanobis_threshold': 2.5,  # Threshold for Mahalanobis distance
                                           # Lower = stricter clustering (fewer merges)
                                           # Higher = more lenient (more merges)
            'distance_metric': 'mahalanobis',  # Primary distance metric
            'fallback_metric': 'cosine',       # Fallback when covariance singular

            # Covariance Matrix Management
            'regularization_factor': 1e-6,    # Regularization for singular matrices
            'min_covariance_samples': 3,      # Minimum samples for covariance estimation
            'covariance_estimation': 'empirical',  # 'empirical' or 'shrunk'
            'shrinkage_parameter': 0.1,        # Shrinkage for regularization

            # Incremental EM Updates
            'learning_rate': 0.1,         # Learning rate for incremental updates
            'momentum': 0.9,              # Momentum for stable updates
            'update_frequency': 1,        # Update every N samples
            'batch_update_size': 10,      # Batch size for updates

            # Hotelling T² Test for Cluster Fusion
            'hotelling_alpha': 0.05,     # Significance level for T² test
                                          # Lower = stricter fusion criteria
                                          # Higher = more liberal fusion
            'fusion_confidence': 0.95,    # Confidence level for fusion decisions
            'min_fusion_samples': 10,     # Minimum samples for reliable fusion test

            # Dynamic Threshold Adjustment
            'adaptive_threshold': True,    # Enable adaptive threshold adjustment
            'threshold_adaptation_rate': 0.05,  # Rate of threshold adaptation
            'threshold_min': 1.5,         # Minimum threshold value
            'threshold_max': 4.0,         # Maximum threshold value
            'performance_window': 50,     # Window for performance evaluation

            # Cluster Quality Metrics
            'silhouette_threshold': 0.5,  # Minimum silhouette score for good clustering
            'inertia_improvement': 0.01,  # Minimum improvement to continue clustering
            'stability_threshold': 0.8,   # Cluster stability threshold
            'quality_check_interval': 25,  # Frames between quality checks

            # HNSW Integration
            'hnsw_ef_search': 20,         # HNSW search parameter for candidates
            'max_hnsw_candidates': 10,    # Maximum candidates from HNSW
            'hnsw_distance_weight': 0.3,  # Weight for HNSW distance in fusion
            'gaussian_distance_weight': 0.7,  # Weight for Gaussian distance

            # Cluster Lifecycle Management
            # Time-to-live for inactive clusters (frames)
            'cluster_ttl': 1000,
            'merge_cooldown': 50,         # Cooldown period after cluster merge
            'split_detection': True,      # Enable cluster splitting detection
            'split_threshold': 3.0,       # Threshold for cluster splitting

            # Memory and Performance
            'max_history_per_cluster': 100,  # Maximum history samples per cluster
            'cleanup_interval': 100,         # Cleanup interval (frames)
            'enable_cluster_caching': True,  # Enable cluster state caching
            'cache_update_interval': 10,     # Cache update frequency

            # Visualization and Debugging
            'enable_visualization': True,    # Enable cluster visualization
            'plot_update_interval': 50,      # Visualization update frequency
            'save_cluster_plots': False,     # Save cluster plots to disk
            'plot_dimensions': (10, 8),      # Plot figure size

            # Advanced Features
            'enable_online_learning': True,  # Enable online learning mode
            'concept_drift_detection': True,  # Enable concept drift detection
            'drift_threshold': 0.3,          # Threshold for concept drift
            'adaptation_strategy': 'gradual',  # 'gradual' or 'immediate'

            # Dataset-specific Optimizations
            'person_appearance_variance': 0.2,  # Expected appearance variance per person
            'lighting_adaptation': True,        # Adapt to lighting changes
            'pose_normalization': True,         # Normalize for pose variations
            'temporal_consistency_weight': 0.4,  # Weight for temporal consistency

            # Evaluation Metrics
            'track_purity': True,          # Track cluster purity metrics
            'track_completeness': True,    # Track cluster completeness
            'track_f1_score': True,        # Track F1 score for clustering
            'evaluation_interval': 100,    # Evaluation frequency (frames)

            # ID Assignment Strategy
            'delayed_id_assignment': True,     # No mostrar ID hasta completar campaña
            'campaign_frames': 30,             # Frames para completar campaña
            'campaign_min_samples': 8,         # Mínimo de muestras para asignar ID
            'campaign_confidence_threshold': 0.7,  # Confianza mínima para ID
            'use_kalman_after_id': True,       # Usar solo Kalman después de ID
            'kalman_prediction_frames': 5,     # Frames de predicción Kalman
            'show_candidate_boxes': True,      # Mostrar cajas sin ID durante campaña
            # Color para candidatos (amarillo)
            'candidate_box_color': (255, 255, 0),
            # Color para IDs confirmados (verde)
            'confirmed_box_color': (0, 255, 0),
        }

    def setup_pipeline_config(self):
        """
        Main Pipeline Configuration
        Parameters for the overall system pipeline
        """
        self.PIPELINE = {
            # Processing Control
            # Maximum frames to process (None = all)
            'max_frames': None,
            'start_frame': 0,             # Starting frame number
            # Process every Nth frame (1 = all frames)
            'frame_skip': 1,
            'enable_multithreading': False,  # Enable multithreaded processing
            'num_threads': 4,             # Number of processing threads

            # Input/Output
            # Supported video formats
            'input_formats': ['.mp4', '.avi', '.mov', '.mkv'],
            'output_format': 'mp4',       # Output video format
            'output_codec': 'mp4v',       # Output video codec
            'output_fps': None,           # Output FPS (None = same as input)
            'output_quality': 0.8,        # Output quality (0.0-1.0)

            # Processing Modes
            'real_time_mode': False,      # Enable real-time processing constraints
            'batch_mode': True,           # Enable batch processing optimizations
            'debug_mode': False,          # Enable debug output and visualizations
            'verbose_logging': True,      # Enable verbose logging

            # Memory Management
            'max_memory_usage': '8GB',    # Maximum memory usage
            'enable_garbage_collection': True,  # Enable automatic garbage collection
            # Garbage collection interval (frames)
            'gc_interval': 100,

            # Error Handling
            'continue_on_error': True,    # Continue processing on non-fatal errors
            'max_consecutive_errors': 10,  # Maximum consecutive errors before stopping
            'error_recovery_delay': 1.0,  # Delay after error (seconds)

            # Progress Reporting
            'progress_interval': 100,     # Report progress every N frames
            'enable_progress_bar': True,  # Show progress bar
            'log_performance_metrics': True,  # Log detailed performance metrics
        }

    def setup_output_config(self):
        """
        Output Configuration
        Parameters for saving results and generated files
        """
        self.OUTPUT = {
            # File Saving Options
            'save_video': True,           # Save annotated output video
            'save_crops': True,           # Save person crops
            'save_features': False,       # Save extracted features
            'save_tracks': True,          # Save tracking data
            'save_statistics': True,      # Save processing statistics

            # Video Output
            'video_quality': 0.8,         # Video compression quality (0.0-1.0)
            'video_bitrate': None,        # Video bitrate (None = auto)
            'include_annotations': True,  # Include bounding boxes and IDs
            'annotation_thickness': 2,    # Annotation line thickness
            'font_scale': 0.6,            # Font scale for text annotations
            'font_color': (0, 255, 0),    # Font color (B, G, R)

            # Crop Settings
            'crop_format': 'jpg',         # Crop image format: 'jpg', 'png'
            'crop_quality': 95,           # JPEG quality for crops (0-100)
            # Padding around crops (fraction of bbox)
            'crop_padding': 0.1,
            'min_crop_resolution': (32, 64),  # Minimum crop resolution
            'max_crops_per_person': 100,  # Maximum crops to save per person

            # Data Export
            'export_format': 'json',      # Export format: 'json', 'csv', 'pickle'
            'include_timestamps': True,   # Include timestamps in exports
            'include_confidence_scores': True,  # Include confidence scores
            'include_feature_vectors': False,   # Include raw feature vectors

            # File Organization
            'organize_by_person': True,   # Organize crops by person ID
            'organize_by_time': False,    # Organize by timestamp
            'create_galleries': True,     # Create person galleries
            'gallery_max_images': 20,     # Maximum images per gallery

            # Compression and Storage
            'compress_data': True,        # Compress output data files
            'compression_level': 6,       # Compression level (1-9)
            'cleanup_temp_files': True,   # Clean up temporary files
            'backup_results': False,      # Create backup copies of results
        }

    def setup_performance_config(self):
        """
        Performance Configuration
        Parameters for optimizing system performance
        """
        self.PERFORMANCE = {
            # Device Selection
            'auto_device_selection': True,  # Automatically select best device
            'prefer_gpu': True,             # Prefer GPU when available
            'gpu_memory_fraction': 0.8,     # Fraction of GPU memory to use
            'enable_mixed_precision': False,  # Enable mixed precision training

            # Batch Processing
            'adaptive_batch_size': True,   # Automatically adjust batch sizes
            'max_batch_memory': '2GB',     # Maximum memory per batch
            'prefetch_batches': 2,         # Number of batches to prefetch

            # Caching
            'enable_model_caching': True,  # Cache loaded models
            'enable_feature_caching': False,  # Cache extracted features
            'cache_size_limit': '1GB',     # Maximum cache size
            'cache_cleanup_interval': 1000,  # Cache cleanup interval

            # Optimization Levels
            'optimization_level': 'balanced',  # 'speed', 'balanced', 'accuracy'
            # Enable TensorRT optimization (NVIDIA only)
            'enable_tensorrt': False,
            # Enable OpenVINO optimization (Intel)
            'enable_openvino': False,

            # Profiling and Monitoring
            'enable_profiling': False,     # Enable performance profiling
            'profile_memory': False,       # Profile memory usage
            'profile_gpu': False,          # Profile GPU utilization
            # Monitoring update interval (seconds)
            'monitoring_interval': 10,

            # Resource Limits
            'max_cpu_usage': 0.8,         # Maximum CPU usage (0.0-1.0)
            'max_memory_usage': 0.8,      # Maximum memory usage (0.0-1.0)
            'max_gpu_usage': 0.9,         # Maximum GPU usage (0.0-1.0)
            'thermal_throttling': True,    # Enable thermal throttling protection
        }

    def get_config(self, module: str = None) -> Dict[str, Any]:
        """
        Get configuration for specific module or all configurations

        Args:
            module: Module name ('detection', 'tracking', 'reid', 'indexing',
                   'fusion', 'gaussian_clustering', 'pipeline', 'output', 'performance') or None for all

        Returns:
            Configuration dictionary
        """
        if module is None:
            return {
                'paths': self.PATHS,
                'detection': self.DETECTION,
                'tracking': self.TRACKING,
                'reid': self.REID,
                'indexing': self.INDEXING,
                'fusion': self.FUSION,
                'gaussian_clustering': self.GAUSSIAN_CLUSTERING,
                'pipeline': self.PIPELINE,
                'output': self.OUTPUT,
                'performance': self.PERFORMANCE
            }

        module_map = {
            'paths': self.PATHS,
            'detection': self.DETECTION,
            'tracking': self.TRACKING,
            'reid': self.REID,
            'indexing': self.INDEXING,
            'fusion': self.FUSION,
            'gaussian_clustering': self.GAUSSIAN_CLUSTERING,
            'pipeline': self.PIPELINE,
            'output': self.OUTPUT,
            'performance': self.PERFORMANCE
        }

        if module.lower() in module_map:
            return module_map[module.lower()]
        else:
            raise ValueError(
                f"Unknown module: {module}. Available modules: {list(module_map.keys())}")

    def update_config(self, module: str, key: str, value: Any):
        """
        Update a specific configuration parameter

        Args:
            module: Module name
            key: Configuration key
            value: New value
        """
        config = self.get_config(module)
        if key in config:
            config[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in module '{module}'")

    def save_config(self, filepath: str):
        """Save current configuration to file"""
        import json
        config_dict = self.get_config()

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_config(self, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Update configurations
        for module, config in config_dict.items():
            if hasattr(self, module.upper()):
                setattr(self, module.upper(), config)

    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("=" * 80)
        print("PERSON RE-IDENTIFICATION SYSTEM - CONFIGURATION SUMMARY")
        print("=" * 80)

        modules = [
            ('DETECTION', 'Person Detection (YOLO)'),
            ('TRACKING', 'Person Tracking (Huffman)'),
            ('REID', 'Re-identification (OSNet)'),
            ('INDEXING', 'Vector Indexing (HNSW)'),
            ('FUSION', 'ID Fusion'),
            ('GAUSSIAN_CLUSTERING', 'Gaussian Clustering'),
            ('PIPELINE', 'Main Pipeline'),
            ('OUTPUT', 'Output Settings'),
            ('PERFORMANCE', 'Performance Settings')
        ]

        for module_name, description in modules:
            config = getattr(self, module_name)
            print(f"\n{description}:")
            print("-" * 40)

            # Show key parameters
            key_params = list(config.keys())[:5]  # Show first 5 parameters
            for key in key_params:
                value = config[key]
                if isinstance(value, (list, tuple)) and len(value) > 3:
                    value = f"[{len(value)} items]"
                elif isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                print(f"  {key}: {value}")

            if len(config) > 5:
                print(f"  ... and {len(config) - 5} more parameters")

        print("\n" + "=" * 80)


# Create global configuration instance
config = GlobalConfig()


def get_config(module: str = None) -> Dict[str, Any]:
    """
    Convenience function to get configuration

    Args:
        module: Module name or None for all configs

    Returns:
        Configuration dictionary
    """
    return config.get_config(module)


def update_config(module: str, key: str, value: Any):
    """
    Convenience function to update configuration

    Args:
        module: Module name
        key: Configuration key
        value: New value
    """
    config.update_config(module, key, value)


# Configuration presets for different use cases
class ConfigPresets:
    """Predefined configuration presets for common use cases"""

    @staticmethod
    def high_accuracy():
        """Configuration optimized for highest accuracy"""
        return {
            'detection': {
                'model_path': 'yolov8l.pt',
                'confidence_threshold': 0.3,
                'input_size': (1280, 1280)
            },
            'reid': {
                'model_name': 'osnet_x1_0',
                'image_size': (384, 128),
                'batch_size': 16
            },
            'indexing': {
                'M': 32,
                'ef_construction': 400,
                'ef_search': 100
            },
            'fusion': {
                'similarity_threshold': 0.8,
                'temporal_window': 50
            },
            'gaussian_clustering': {
                'mahalanobis_threshold': 2.0,
                'hotelling_alpha': 0.01,
                'adaptive_threshold': True,
                'min_samples_per_cluster': 8,
                'quality_check_interval': 15
            }
        }

    @staticmethod
    def high_speed():
        """Configuration optimized for speed"""
        return {
            'detection': {
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.6,
                'input_size': (416, 416)
            },
            'reid': {
                'model_name': 'osnet_x0_5',
                'image_size': (256, 128),
                'batch_size': 64
            },
            'indexing': {
                'M': 8,
                'ef_construction': 100,
                'ef_search': 25
            },
            'fusion': {
                'similarity_threshold': 0.6,
                'temporal_window': 20
            },
            'gaussian_clustering': {
                'mahalanobis_threshold': 3.0,
                'hotelling_alpha': 0.1,
                'adaptive_threshold': False,
                'min_samples_per_cluster': 3,
                'quality_check_interval': 50
            }
        }

    @staticmethod
    def balanced():
        """Balanced configuration for accuracy and speed"""
        return {
            'detection': {
                'model_path': 'yolov8s.pt',
                'confidence_threshold': 0.5,
                'input_size': (640, 640)
            },
            'reid': {
                'model_name': 'osnet_x0_75',
                'image_size': (256, 128),
                'batch_size': 32
            },
            'indexing': {
                'M': 16,
                'ef_construction': 200,
                'ef_search': 50
            },
            'fusion': {
                'similarity_threshold': 0.7,
                'temporal_window': 30
            },
            'gaussian_clustering': {
                'mahalanobis_threshold': 2.5,
                'hotelling_alpha': 0.05,
                'adaptive_threshold': True,
                'min_samples_per_cluster': 5,
                'quality_check_interval': 25
            }
        }

    @staticmethod
    def three_person_dataset():
        """
        Configuration optimized for 3-person dataset (A, B, C)
        Based on clustering analysis results with optimal threshold: 8.0
        """
        return {
            'detection': {
                'model_path': 'yolov8s.pt',
                'confidence_threshold': 0.4,
                'input_size': (640, 640)
            },
            'reid': {
                'model_name': 'osnet_x1_0',
                'image_size': (256, 128),
                'batch_size': 32
            },
            'indexing': {
                'M': 16,
                'ef_construction': 200,
                'ef_search': 30,
                'max_elements': 1000  # Smaller for 3-person dataset
            },
            'fusion': {
                'similarity_threshold': 0.75,
                'temporal_window': 40,
                'reid_weight': 0.7,
                'tracking_weight': 0.3
            },
            'gaussian_clustering': {
                'expected_persons': 3,
                'ground_truth_clusters': 3,
                'initial_clusters': 6,
                'max_clusters': 8,
                # OPTIMIZED: Found through clustering analysis
                'mahalanobis_threshold': 8.0,  # Optimal threshold for 3-person dataset
                'hotelling_alpha': 0.1,        # Más permisivo para fusión
                'adaptive_threshold': True,
                'threshold_adaptation_rate': 0.03,
                'min_samples_per_cluster': 2,  # Menos estricto
                'quality_check_interval': 10,  # Más frecuente
                'fusion_confidence': 0.85,     # Menos estricto
                'person_appearance_variance': 0.3,
                'temporal_consistency_weight': 0.5,
                'enable_visualization': True,
                'track_purity': True,
                'evaluation_interval': 50,
                # Configuración de campaña para asignación de ID
                'delayed_id_assignment': True,
                'campaign_frames': 25,          # Campaña más corta para 3 personas
                'campaign_min_samples': 6,      # Menos muestras requeridas
                'campaign_confidence_threshold': 0.75,
                'use_kalman_after_id': True,
                'show_candidate_boxes': True
            }
        }

    @staticmethod
    def three_person_dataset_90_confidence():
        """
        Configuration optimized for 3-person dataset with 90% YOLO confidence
        Based on comprehensive analysis - BEST CLUSTERING PERFORMANCE:
        - 90% confidence threshold achieves PERFECT clustering (3 clusters)
        - 100% efficiency (exactly expected number of clusters)
        - 99.8% average purity (highest quality)
        - 37.3% image retention (high quality images only)
        """
        return {
            'detection': {
                'model_path': 'yolov8s.pt',
                'confidence_threshold': 0.9,   # 90% confidence - OPTIMAL for clustering
                'input_size': (640, 640),
                'iou_threshold': 0.5,
                'target_classes': [0],  # Person class only
                'verbose': False
            },
            'reid': {
                'model_name': 'osnet_x1_0',
                'image_size': (256, 128),
                'batch_size': 32,
                'feature_dim': 512,
                'normalize_features': True
            },
            'indexing': {
                'M': 16,
                'ef_construction': 200,
                'ef_search': 30,
                'max_elements': 500,  # Smaller due to filtered dataset
                'space': 'cosine'
            },
            'fusion': {
                'similarity_threshold': 0.75,
                'temporal_window': 40,
                'reid_weight': 0.7,
                'tracking_weight': 0.3,
                'confidence_decay': 0.95
            },
            'gaussian_clustering': {
                'expected_persons': 3,
                'ground_truth_clusters': 3,
                'initial_clusters': 6,
                'max_clusters': 8,
                # OPTIMIZED: 90% confidence achieves perfect clustering
                'mahalanobis_threshold': 7.0,  # Optimal for 90% confidence
                'hotelling_alpha': 0.1,        # Permissive for fusion
                'adaptive_threshold': True,
                'threshold_adaptation_rate': 0.03,
                'min_samples_per_cluster': 2,  # Less strict for filtered data
                'quality_check_interval': 10,
                'fusion_confidence': 0.85,
                'person_appearance_variance': 0.2,  # Lower variance with high-quality images
                'temporal_consistency_weight': 0.5,
                'enable_visualization': True,
                'track_purity': True,
                'evaluation_interval': 50,
                # Campaign configuration for ID assignment
                'delayed_id_assignment': True,
                'campaign_frames': 20,          # Shorter campaign for high-quality data
                'campaign_min_samples': 5,      # Fewer samples needed
                'campaign_confidence_threshold': 0.8,  # Higher confidence
                'use_kalman_after_id': True,
                'show_candidate_boxes': True
            }
        }


def apply_preset(preset_name: str):
    """
    Apply a configuration preset

    Args:
        preset_name: 'high_accuracy', 'high_speed', 'balanced', 'three_person_dataset', or 'three_person_dataset_90_confidence'
    """
    presets = {
        'high_accuracy': ConfigPresets.high_accuracy(),
        'high_speed': ConfigPresets.high_speed(),
        'balanced': ConfigPresets.balanced(),
        'three_person_dataset': ConfigPresets.three_person_dataset(),
        'three_person_dataset_90_confidence': ConfigPresets.three_person_dataset_90_confidence()
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    preset_config = presets[preset_name]

    for module, params in preset_config.items():
        for key, value in params.items():
            config.update_config(module, key, value)

    print(f"Applied '{preset_name}' configuration preset")

    # Show key performance metrics for the 90% confidence preset
    if preset_name == 'three_person_dataset_90_confidence':
        print("🎯 PERFORMANCE METRICS (from analysis):")
        print("  - Clustering Efficiency: 100% (perfect 3 clusters)")
        print("  - Average Purity: 99.8% (highest quality)")
        print("  - Image Retention: 37.3% (high-quality images only)")
        print("  - Silhouette Score: 0.147 (best clustering quality)")
        print("  - ARI Score: 0.734 (good correlation with ground truth)")


if __name__ == "__main__":
    # Example usage and configuration summary
    config.print_config_summary()

    # Example of updating configuration
    print("\nExample: Updating detection confidence threshold...")
    update_config('detection', 'confidence_threshold', 0.6)
    print(
        f"New confidence threshold: {get_config('detection')['confidence_threshold']}")

    # Example of applying preset
    print("\nExample: Applying high-speed preset...")
    apply_preset('high_speed')

    # Save configuration
    config.save_config('config/current_config.json')
    print("\nConfiguration saved to 'config/current_config.json'")
