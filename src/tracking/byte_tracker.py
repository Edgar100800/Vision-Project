import numpy as np
import lap
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection with bounding box and confidence"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int = 0


class STrack:
    """Single target track for ByteTrack"""

    def __init__(self, bbox: np.ndarray, score: float, track_id: int):
        self.track_id = track_id
        self.bbox = bbox.copy()
        self.score = score
        self.tracklet_len = 0
        self.state = 'tracked'  # 'tracked', 'lost', 'removed'
        self.frame_id = 0
        self.start_frame = 0
        self.is_activated = False
        self.time_since_update = 0

        # Kalman filter state
        self.mean = np.zeros(8)  # [x, y, a, h, vx, vy, va, vh]
        self.covariance = np.eye(8)

        # Initialize mean from bbox
        self._update_mean_from_bbox(bbox)

    def _update_mean_from_bbox(self, bbox: np.ndarray):
        """Update Kalman mean from bounding box"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        a = (x2 - x1) / (y2 - y1)  # aspect ratio
        h = y2 - y1  # height

        self.mean[:4] = [cx, cy, a, h]

    def _get_bbox_from_mean(self) -> np.ndarray:
        """Get bounding box from Kalman mean"""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    def predict(self):
        """Predict next state using simple motion model"""
        # Simple constant velocity model
        dt = 1.0
        F = np.eye(8)
        F[0, 4] = dt  # x += vx * dt
        F[1, 5] = dt  # y += vy * dt
        F[2, 6] = dt  # a += va * dt
        F[3, 7] = dt  # h += vh * dt

        self.mean = F @ self.mean

        # Update bbox from predicted mean
        self.bbox = self._get_bbox_from_mean()

    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection"""
        self.bbox = detection.bbox.copy()
        self.score = detection.score
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0

        # Update Kalman mean
        self._update_mean_from_bbox(detection.bbox)

        if not self.is_activated:
            self.is_activated = True
            self.start_frame = frame_id

        self.state = 'tracked'

    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'

    def mark_removed(self):
        """Mark track as removed"""
        self.state = 'removed'


def iou_distance(tracks: List[STrack], detections: List[Detection]) -> np.ndarray:
    """Compute IoU distance matrix between tracks and detections"""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    distance_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            iou = compute_iou(track.bbox, detection.bbox)
            distance_matrix[i, j] = 1 - iou  # Convert IoU to distance

    return distance_matrix


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    # Intersection area
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Union area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Linear assignment using LAP solver"""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    # Use LAP solver
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    matches = []
    unmatched_a = []
    unmatched_b = []

    for i in range(len(x)):
        if x[i] >= 0:
            matches.append((i, x[i]))
        else:
            unmatched_a.append(i)

    for j in range(len(y)):
        if y[j] < 0:
            unmatched_b.append(j)

    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack implementation for robust multi-object tracking"""

    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5,
                 track_buffer: int = 30, match_thresh: float = 0.8,
                 high_thresh: float = 0.6, low_thresh: float = 0.1):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

        self.frame_id = 0
        self.track_id_count = 0

        # Track lists
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

    def update(self, detections: List[Detection]) -> List[STrack]:
        """Update tracker with new detections"""
        self.frame_id += 1

        # Separate detections by confidence
        high_dets = [
            det for det in detections if det.score >= self.high_thresh]
        low_dets = [det for det in detections if self.low_thresh <=
                    det.score < self.high_thresh]

        # Predict all tracks
        for track in self.tracked_tracks:
            track.predict()

        # First association with high confidence detections
        matches, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_dets, self.match_thresh
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracked_tracks[track_idx].update(
                high_dets[det_idx], self.frame_id)

        # Handle unmatched tracks
        unmatched_tracked_tracks = [self.tracked_tracks[i]
                                    for i in unmatched_tracks]

        # Second association with low confidence detections
        matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
            unmatched_tracked_tracks, low_dets, 0.5
        )

        # Update matched tracks from second association
        for track_idx, det_idx in matches2:
            unmatched_tracked_tracks[track_idx].update(
                low_dets[det_idx], self.frame_id)

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks2:
            track = unmatched_tracked_tracks[track_idx]
            track.mark_lost()

        # Create new tracks from unmatched high confidence detections
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets]
        for det in unmatched_high_dets:
            new_track = STrack(det.bbox, det.score, self._next_id())
            new_track.update(det, self.frame_id)
            self.tracked_tracks.append(new_track)

        # Update track lists
        self._update_track_lists()

        # Return active tracks
        return [track for track in self.tracked_tracks if track.is_activated]

    def _associate(self, tracks: List[STrack], detections: List[Detection],
                   thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU distance matrix
        distance_matrix = iou_distance(tracks, detections)

        # Linear assignment
        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            distance_matrix, thresh
        )

        return matches, unmatched_tracks, unmatched_dets

    def _update_track_lists(self):
        """Update track lists by removing old tracks"""
        # Move lost tracks to removed if they've been lost too long
        for track in self.lost_tracks[:]:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)

        # Move tracks from tracked to lost
        for track in self.tracked_tracks[:]:
            if track.state == 'lost':
                self.tracked_tracks.remove(track)
                self.lost_tracks.append(track)

        # Clean up removed tracks (keep only recent ones)
        if len(self.removed_tracks) > 1000:
            self.removed_tracks = self.removed_tracks[-500:]

    def _next_id(self) -> int:
        """Get next track ID"""
        self.track_id_count += 1
        return self.track_id_count
