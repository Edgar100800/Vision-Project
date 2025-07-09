"""
Person detection module using YOLO11m-pose.
"""

from .person_detector import PersonDetector
from .pose_detector import PoseDetector

__all__ = [
    'PersonDetector',
    'PoseDetector'
] 