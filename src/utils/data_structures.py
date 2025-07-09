"""
Data structures for the person re-identification system.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Detection:
    """Represents a person detection in a single frame."""
    
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    pose_keypoints: Optional[np.ndarray] = None  # Shape: (num_keypoints, 3) - [x, y, confidence]
    frame_id: int = 0
    timestamp: float = 0.0
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio of bounding box."""
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1
        return width / height if height > 0 else 0


@dataclass
class Person:
    """Represents a person with detection and embedding information."""
    
    detection: Detection
    embedding: Optional[np.ndarray] = None
    person_id: Optional[int] = None
    track_id: Optional[int] = None
    status: str = "NEW"  # NEW, TRACKING, RE_IDENTIFIED, LOST
    
    def get_features(self) -> Dict[str, Any]:
        """Get all features for this person."""
        features = {
            'bbox': self.detection.bbox,
            'confidence': self.detection.confidence,
            'center': self.detection.get_center(),
            'area': self.detection.get_area(),
            'aspect_ratio': self.detection.get_aspect_ratio(),
            'frame_id': self.detection.frame_id,
            'timestamp': self.detection.timestamp
        }
        
        if self.embedding is not None:
            features['embedding'] = self.embedding
        
        if self.pose_keypoints is not None:
            features['pose_keypoints'] = self.detection.pose_keypoints
        
        return features
    
    @property
    def pose_keypoints(self) -> Optional[np.ndarray]:
        """Get pose keypoints."""
        return self.detection.pose_keypoints


@dataclass
class Track:
    """Represents a track of a person across multiple frames."""
    
    track_id: int
    person_id: int
    detections: List[Detection] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # [(x, y, timestamp), ...]
    
    # Tracking state
    is_active: bool = True
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_frames: int = 0
    disappeared_frames: int = 0
    
    def add_detection(self, detection: Detection, embedding: Optional[np.ndarray] = None):
        """Add a new detection to the track."""
        self.detections.append(detection)
        if embedding is not None:
            self.embeddings.append(embedding)
        
        # Update trajectory
        center = detection.get_center()
        self.trajectory.append((center[0], center[1], detection.timestamp))
        
        # Update tracking state
        if self.first_seen == 0.0:
            self.first_seen = detection.timestamp
        self.last_seen = detection.timestamp
        self.total_frames += 1
        self.disappeared_frames = 0
    
    def get_latest_detection(self) -> Optional[Detection]:
        """Get the most recent detection."""
        return self.detections[-1] if self.detections else None
    
    def get_latest_embedding(self) -> Optional[np.ndarray]:
        """Get the most recent embedding."""
        return self.embeddings[-1] if self.embeddings else None
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding across all detections."""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)
    
    def get_trajectory_window(self, window_size: int = 30) -> List[Tuple[float, float, float]]:
        """Get recent trajectory points."""
        return self.trajectory[-window_size:] if self.trajectory else []
    
    def update_disappeared(self, frames: int = 1):
        """Update disappeared frame count."""
        self.disappeared_frames += frames
        if self.disappeared_frames > 0:
            self.is_active = False
    
    def get_duration(self) -> float:
        """Get track duration in seconds."""
        return self.last_seen - self.first_seen if self.first_seen > 0 else 0.0


@dataclass
class TrackingResult:
    """Result of tracking process for a single frame."""
    
    frame_id: int
    timestamp: float
    persons: List[Person] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    new_tracks: List[Track] = field(default_factory=list)
    lost_tracks: List[Track] = field(default_factory=list)
    
    def get_person_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        for person in self.persons:
            if person.person_id == person_id:
                return person
        return None
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None


@dataclass
class ReIDMatch:
    """Represents a re-identification match."""
    
    query_person: Person
    gallery_person: Person
    similarity_score: float
    distance: float
    is_match: bool
    
    def __str__(self) -> str:
        return f"Match(query_id={self.query_person.person_id}, gallery_id={self.gallery_person.person_id}, score={self.similarity_score:.3f})"


@dataclass
class VideoMetadata:
    """Metadata for video processing."""
    
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    filename: str
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution."""
        return (self.width, self.height)
    
    def get_frame_time(self) -> float:
        """Get time per frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.0 