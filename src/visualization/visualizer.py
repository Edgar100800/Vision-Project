"""
Visualizer for person re-identification tracking results.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

from ..utils.data_structures import Person, Track, TrackingResult
from ..utils.config_loader import ConfigLoader


class Visualizer:
    """Visualize tracking results on video frames."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.viz_config = config.get_visualization_config()
        
        # Visualization parameters
        self.show_bbox = self.viz_config.get("show_bbox", True)
        self.show_id = self.viz_config.get("show_id", True)
        self.show_pose = self.viz_config.get("show_pose", True)
        self.show_trajectory = self.viz_config.get("show_trajectory", True)
        self.trajectory_length = self.viz_config.get("trajectory_length", 30)
        self.colors = self.viz_config.get("colors", [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255]   # Cyan
        ])
        self.font_scale = self.viz_config.get("font_scale", 0.6)
        self.thickness = self.viz_config.get("thickness", 2)
        
        logging.info("Visualizer initialized")
    
    def draw_frame(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """
        Draw tracking results on a frame.
        
        Args:
            frame: Input frame (BGR format)
            tracking_result: Tracking result for this frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw active tracks
        for track in tracking_result.tracks:
            if track.is_active:
                self._draw_track(annotated_frame, track)
        
        # Draw new tracks with different style
        for track in tracking_result.new_tracks:
            self._draw_new_track(annotated_frame, track)
        
        # Draw lost tracks (faded)
        for track in tracking_result.lost_tracks:
            self._draw_lost_track(annotated_frame, track)
        
        # Draw frame information
        self._draw_frame_info(annotated_frame, tracking_result)
        
        return annotated_frame
    
    def _draw_track(self, frame: np.ndarray, track: Track):
        """Draw an active track."""
        if not track.detections:
            return
        
        latest_detection = track.get_latest_detection()
        if latest_detection is None:
            return
        
        # Get color for this track
        color = self._get_track_color(track.track_id)
        
        # Draw bounding box
        if self.show_bbox:
            self._draw_bounding_box(frame, latest_detection, color)
        
        # Draw person ID with status
        if self.show_id:
            # Find the person with this track_id to get status
            status = "TRACKING"  # Default status
            self._draw_person_id(frame, latest_detection, track.person_id, color, status=status)
        
        # Draw pose keypoints
        if self.show_pose and latest_detection.pose_keypoints is not None:
            self._draw_pose_keypoints(frame, latest_detection.pose_keypoints, color)
        
        # Draw trajectory
        if self.show_trajectory:
            self._draw_trajectory(frame, track, color)
    
    def _draw_new_track(self, frame: np.ndarray, track: Track):
        """Draw a newly created track."""
        if not track.detections:
            return
        
        latest_detection = track.get_latest_detection()
        if latest_detection is None:
            return
        
        # Use bright green for new tracks
        color = [0, 255, 0]
        
        # Draw bounding box with dashed line
        if self.show_bbox:
            self._draw_bounding_box(frame, latest_detection, color, dashed=True)
        
        # Draw person ID
        if self.show_id:
            self._draw_person_id(frame, latest_detection, track.person_id, color, prefix="NEW")
    
    def _draw_lost_track(self, frame: np.ndarray, track: Track):
        """Draw a lost track (faded)."""
        if not track.detections:
            return
        
        latest_detection = track.get_latest_detection()
        if latest_detection is None:
            return
        
        # Use gray for lost tracks
        color = [128, 128, 128]
        
        # Draw bounding box with low opacity
        if self.show_bbox:
            self._draw_bounding_box(frame, latest_detection, color, opacity=0.3)
        
        # Draw person ID
        if self.show_id:
            self._draw_person_id(frame, latest_detection, track.person_id, color, prefix="LOST")
    
    def _draw_bounding_box(self, frame: np.ndarray, detection, color: List[int], 
                          dashed: bool = False, opacity: float = 1.0):
        """Draw bounding box."""
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        if opacity < 1.0:
            # Create overlay for transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.thickness)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        else:
            if dashed:
                # Draw dashed rectangle
                self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: Tuple[int, int], 
                              pt2: Tuple[int, int], color: List[int], dash_length: int = 10):
        """Draw dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines
        for i in range(0, x2 - x1, dash_length * 2):
            # Top line
            cv2.line(frame, (x1 + i, y1), (min(x1 + i + dash_length, x2), y1), color, self.thickness)
            # Bottom line
            cv2.line(frame, (x1 + i, y2), (min(x1 + i + dash_length, x2), y2), color, self.thickness)
        
        for i in range(0, y2 - y1, dash_length * 2):
            # Left line
            cv2.line(frame, (x1, y1 + i), (x1, min(y1 + i + dash_length, y2)), color, self.thickness)
            # Right line
            cv2.line(frame, (x2, y1 + i), (x2, min(y1 + i + dash_length, y2)), color, self.thickness)
    
    def _draw_person_id(self, frame: np.ndarray, detection, person_id: int, 
                       color: List[int], prefix: str = "", status: str = ""):
        """Draw person ID label with status."""
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        # Create label text with status
        if status == "TRACKING_LOCKED":
            label = f"ID:{person_id} (LOCKED)"
        elif status:
            label = f"ID:{person_id} ({status})"
        elif prefix:
            label = f"{prefix} ID:{person_id}"
        else:
            label = f"ID:{person_id}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
        )
        
        # Choose background color based on status
        bg_color = color
        if status == "NEW":
            bg_color = [0, 255, 0]  # Green for new
        elif status == "TRACKING":
            bg_color = [255, 165, 0]  # Orange for tracking
        elif status == "TRACKING_LOCKED":
            bg_color = [0, 0, 255]  # Blue for locked
        elif status == "RE_IDENTIFIED":
            bg_color = [255, 0, 255]  # Magenta for re-identified
        
        # Draw background rectangle
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, [255, 255, 255], self.thickness)
    
    def _draw_pose_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, color: List[int]):
        """Draw pose keypoints."""
        if keypoints is None or len(keypoints) == 0:
            return
        
        for kpt in keypoints:
            x, y, conf = kpt
            
            if conf > 0.5:  # Only draw confident keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    
    def _draw_trajectory(self, frame: np.ndarray, track: Track, color: List[int]):
        """Draw track trajectory."""
        if not track.trajectory:
            return
        
        # Get recent trajectory points
        trajectory_points = track.get_trajectory_window(self.trajectory_length)
        
        if len(trajectory_points) < 2:
            return
        
        # Draw trajectory lines
        for i in range(1, len(trajectory_points)):
            pt1 = (int(trajectory_points[i-1][0]), int(trajectory_points[i-1][1]))
            pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
            
            # Fade color based on recency
            alpha = i / len(trajectory_points)
            faded_color = [int(c * alpha) for c in color]
            
            cv2.line(frame, pt1, pt2, faded_color, max(1, self.thickness // 2))
    
    def _draw_frame_info(self, frame: np.ndarray, tracking_result: TrackingResult):
        """Draw frame information."""
        height, width = frame.shape[:2]
        
        # Create info text
        info_lines = [
            f"Frame: {tracking_result.frame_id}",
            f"Time: {tracking_result.timestamp:.2f}s",
            f"Active Tracks: {len([t for t in tracking_result.tracks if t.is_active])}",
            f"New Tracks: {len(tracking_result.new_tracks)}",
            f"Lost Tracks: {len(tracking_result.lost_tracks)}"
        ]
        
        # Draw info box
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)
            cv2.putText(frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)
    
    def _get_track_color(self, track_id: int) -> List[int]:
        """Get color for a track ID."""
        return self.colors[track_id % len(self.colors)]
    
    def create_legend(self, width: int = 300, height: int = 200) -> np.ndarray:
        """Create a legend image."""
        legend = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Draw legend items
        items = [
            ("Active Track", [255, 0, 0]),
            ("New Track", [0, 255, 0]),
            ("Lost Track", [128, 128, 128])
        ]
        
        y_offset = 30
        for i, (label, color) in enumerate(items):
            # Draw color box
            cv2.rectangle(legend, (20, y_offset + i * 40), (50, y_offset + i * 40 + 20), color, -1)
            
            # Draw label
            cv2.putText(legend, label, (60, y_offset + i * 40 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
        
        return legend 