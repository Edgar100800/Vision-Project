import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


class Track:
    """
    This class represents a single tracked object, holding its Kalman Filter,
    bounding box, and state.
    """

    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.kf = self._create_kalman_filter()
        self.kf.x[:4] = np.array([bbox]).T
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def _create_kalman_filter(self):
        """
        Creates a new Kalman Filter for a track.
        The state is [x, y, s, r, dx, dy, ds, dr] where:
        x, y are center coordinates
        s is scale (area)
        r is aspect ratio
        and the rest are their velocities.
        """
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1,
                                                    0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])

        kf.R[2:, 2:] *= 10.
        kf.P[4:, 4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        return kf

    def predict(self):
        """Predict the next state of the track."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._get_bbox_from_state()

    def update(self, bbox):
        """Update the track with a new detected bounding box."""
        self.bbox = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Convert bbox to [center_x, center_y, area, aspect_ratio]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        area = w * h
        aspect_ratio = w / h

        measurement = np.array([center_x, center_y, area, aspect_ratio]).T
        self.kf.update(measurement)

    def _get_bbox_from_state(self):
        """Convert state vector back to a bounding box."""
        state = self.kf.x.flatten()
        center_x, center_y, area, aspect_ratio = state[:4]

        w = np.sqrt(area * aspect_ratio)
        h = area / w
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = x1 + w
        y2 = y1 + h
        return [x1, y1, x2, y2]


class KalmanTracker:
    """
    Main class for Kalman Filter-based tracking.
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        Update the tracker with a new set of detections for a frame.

        Args:
            detections (list): A list of bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            A list of active tracks, where each track is [x1, y1, x2, y2, track_id].
        """
        # Predict next location of each track
        predicted_bboxes = []
        for t in self.tracks:
            predicted_bboxes.append(t.predict())

        # Associate detections with tracks
        matched_indices, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections, predicted_bboxes)

        # Update matched tracks
        for track_idx, detection_idx in matched_indices:
            self.tracks[track_idx].update(detections[detection_idx])

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[detection_idx])
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks
        active_tracks = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.age < self.min_hits or t.hits >= self.min_hits):
                active_tracks.append(np.concatenate(
                    (t.bbox, [t.track_id])).tolist())

        return active_tracks

    def _associate_detections_to_tracks(self, detections, predicted_bboxes):
        """Associate detections to tracks using IoU and Hungarian algorithm."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        iou_matrix = np.zeros(
            (len(detections), len(predicted_bboxes)), dtype=np.float32)

        for d_idx, det in enumerate(detections):
            for t_idx, pred in enumerate(predicted_bboxes):
                iou_matrix[d_idx, t_idx] = iou(det, pred)

        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_indices = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((c, r))
                unmatched_detections.discard(r)
                unmatched_tracks.discard(c)

        return matched_indices, list(unmatched_detections), list(unmatched_tracks)
