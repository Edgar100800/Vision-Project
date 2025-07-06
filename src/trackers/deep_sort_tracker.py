"""
Deep SORT tracker for person tracking in the store monitoring system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import TrackingConfig
from utils.logger import setup_logger


class DeepSORTTracker:
    """
    Person tracker using Deep SORT algorithm.
    """

    def __init__(self, config: TrackingConfig):
        """
        Initialize the Deep SORT tracker.

        Args:
            config: Tracking configuration
        """
        self.config = config
        self.logger = setup_logger("DeepSORTTracker")

        # Initialize Deep SORT
        try:
            self.tracker = DeepSort(
                max_age=config.max_age,
                n_init=config.n_init,
                max_iou_distance=config.max_iou_distance,
                max_cosine_distance=config.max_cosine_distance,
                nn_budget=config.nn_budget,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            self.logger.info("Deep SORT tracker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Deep SORT tracker: {e}")
            raise

        # Track history for analytics
        # track_id -> [(x, y, timestamp), ...]
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}

    def update(self, detections: List[Tuple[int, int, int, int, float]], frame: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections as (x1, y1, x2, y2, confidence)
            frame: Current frame

        Returns:
            List of tracks as (x1, y1, x2, y2, track_id)
        """
        try:
            # Convert detections to Deep SORT format
            if not detections:
                tracks = self.tracker.update_tracks([], frame=frame)
                return []

            # Convert to required format: [[[x1, y1, x2, y2], confidence], ...]
            det_list = []
            for x1, y1, x2, y2, conf in detections:
                # Deep SORT expects: [[bbox], confidence]
                det_list.append([[x1, y1, x2, y2], conf])

            # Update tracker
            tracks = self.tracker.update_tracks(det_list, frame=frame)

            # Convert to output format
            track_results = []
            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = int(ltrb[0]), int(
                    ltrb[1]), int(ltrb[2]), int(ltrb[3])

                # Store track history for analytics
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append(
                    (center_x, center_y, current_time))

                # Keep only recent history (last 30 seconds)
                self.track_history[track_id] = [
                    (x, y, t) for x, y, t in self.track_history[track_id]
                    if current_time - t < 30.0
                ]

                track_results.append((x1, y1, x2, y2, track_id))

            return track_results

        except Exception as e:
            self.logger.error(f"Tracking update failed: {e}")
            return []

    def draw_tracks(self, frame: np.ndarray, tracks: List[Tuple[int, int, int, int, int]]) -> np.ndarray:
        """
        Draw tracking results on frame.

        Args:
            frame: Input frame
            tracks: List of tracks as (x1, y1, x2, y2, track_id)

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for x1, y1, x2, y2, track_id in tracks:
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw track ID
            label = f"ID: {track_id}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (255, 0, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw trajectory
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                # Last 10 points
                points = [(int(x), int(y))
                          for x, y, _ in self.track_history[track_id][-10:]]
                for i in range(1, len(points)):
                    cv2.line(annotated_frame,
                             points[i-1], points[i], (0, 255, 255), 2)

        return annotated_frame

    def get_track_history(self, track_id: int) -> List[Tuple[float, float, float]]:
        """
        Get trajectory history for a specific track.

        Args:
            track_id: Track identifier

        Returns:
            List of (x, y, timestamp) tuples
        """
        return self.track_history.get(track_id, [])

    def get_active_tracks(self) -> List[int]:
        """
        Get list of currently active track IDs.

        Returns:
            List of active track IDs
        """
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        active_tracks = []

        for track_id, history in self.track_history.items():
            # Active within last second
            if history and current_time - history[-1][2] < 1.0:
                active_tracks.append(track_id)

        return active_tracks

    def reset(self):
        """Reset the tracker and clear history."""
        self.tracker = DeepSort(
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            max_iou_distance=self.config.max_iou_distance,
            max_cosine_distance=self.config.max_cosine_distance,
            nn_budget=self.config.nn_budget,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.track_history.clear()
        self.logger.info("Tracker reset successfully")
