"""
Configuration file for the store monitoring system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DetectionConfig:
    """Configuration for person detection."""
    model_name: str = "yolo11m.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"  # auto, cpu, cuda
    classes: List[int] = field(default_factory=lambda: [0])  # Person class in COCO dataset


@dataclass
class TrackingConfig:
    """Configuration for Deep SORT tracking."""
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.2
    nn_budget: int = 100


@dataclass
class ReIDConfig:
    """Configuration for FastReID."""
    model_path: str = "fast-reid/configs/Market1501/bagtricks_R50.yml"
    weights_path: str = "fast-reid/logs/market1501/bagtricks_R50/model_final.pth"
    device: str = "auto"
    feature_dim: int = 2048


@dataclass
class ZoneConfig:
    """Configuration for zones of interest."""
    # Entrance/Exit line (x1, y1, x2, y2)
    entrance_line: Tuple[int, int, int, int] = (100, 0, 100, 720)
    exit_line: Tuple[int, int, int, int] = (1180, 0, 1180, 720)

    # Queue area (x1, y1, x2, y2)
    queue_area: Tuple[int, int, int, int] = (200, 400, 800, 600)

    # Cashier area (x1, y1, x2, y2)
    cashier_area: Tuple[int, int, int, int] = (900, 400, 1100, 600)

    # Minimum time in seconds to consider someone in queue
    min_queue_time: float = 5.0

    # Minimum time in seconds to consider someone being served
    min_service_time: float = 2.0


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    input_path: str = "data/video.mp4"
    output_path: str = "data/processed/output.mp4"
    fps: int = 30
    resolution: Tuple[int, int] = (1280, 720)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics and logging."""
    log_file: str = "data/processed/events.csv"
    dashboard_port: int = 8501
    update_interval: int = 1  # seconds


@dataclass
class SystemConfig:
    """Main system configuration."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    # Global settings
    debug: bool = False
    verbose: bool = True


def get_config() -> SystemConfig:
    """Get the system configuration."""
    return SystemConfig()


def load_config_from_file(config_path: str) -> SystemConfig:
    """Load configuration from YAML file."""
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # TODO: Implement proper config loading from dict
    return SystemConfig()


def save_config_to_file(config: SystemConfig, config_path: str):
    """Save configuration to YAML file."""
    import yaml
    from dataclasses import asdict

    config_dict = asdict(config)

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
