#!/usr/bin/env python3
"""
Huffman Tracking Module for Person Tracking
Uses Huffman coding to encode coordinate changes for efficient tracking
"""

from config.global_config import get_config
import numpy as np
import heapq
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
import json
import pickle
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class HuffmanNode:
    """Node for Huffman tree"""

    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanEncoder:
    """Huffman encoder for coordinate changes"""

    def __init__(self):
        self.codes = {}
        self.root = None

    def build_tree(self, frequencies: Dict[str, int]):
        """Build Huffman tree from character frequencies"""
        heap = []

        # Create leaf nodes for each character
        for char, freq in frequencies.items():
            node = HuffmanNode(char, freq)
            heapq.heappush(heap, node)

        # Build tree bottom-up
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = HuffmanNode(
                freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)

        self.root = heap[0] if heap else None

    def generate_codes(self, node=None, code=""):
        """Generate Huffman codes for each character"""
        if node is None:
            node = self.root

        if node.char is not None:  # Leaf node
            self.codes[node.char] = code if code else "0"
            return

        if node.left:
            self.generate_codes(node.left, code + "0")
        if node.right:
            self.generate_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """Encode text using Huffman codes"""
        return ''.join(self.codes.get(char, '') for char in text)

    def decode(self, encoded: str) -> str:
        """Decode Huffman encoded string"""
        if not self.root or not encoded:
            return ""

        decoded = []
        current = self.root

        for bit in encoded:
            if bit == '0':
                current = current.left
            else:
                current = current.right

            if current.char is not None:  # Leaf node
                decoded.append(current.char)
                current = self.root

        return ''.join(decoded)


class PersonTracker:
    """Person tracker using Huffman coding"""

    def __init__(self, max_disappeared: int = None, max_distance: float = None):
        """
        Initialize tracker

        Args:
            max_disappeared: Maximum frames a person can disappear before being removed (uses config default if None)
            max_distance: Maximum distance for associating detections with tracks (uses config default if None)
        """
        # Get configuration
        self.config = get_config('tracking')

        # Use provided values or defaults from config
        self.max_disappeared = max_disappeared or self.config['max_disappeared']
        self.max_distance = max_distance or self.config['max_distance']

        # Store other config values
        self.min_track_length = self.config['min_track_length']
        self.coordinate_discretization = self.config['coordinate_discretization']
        self.enable_compression = self.config['enable_compression']
        self.compression_window = self.config['compression_window']
        self.retrain_interval = self.config['retrain_interval']
        self.distance_metric = self.config['distance_metric']
        self.position_weight = self.config['position_weight']
        self.size_weight = self.config['size_weight']
        self.enable_prediction = self.config['enable_prediction']
        self.prediction_frames = self.config['prediction_frames']
        self.max_prediction_distance = self.config['max_prediction_distance']
        self.confidence_decay = self.config['confidence_decay']
        self.min_confidence = self.config['min_confidence']
        self.quality_threshold = self.config['quality_threshold']

        # Tracking state
        self.next_id = 0
        self.tracks = {}  # track_id -> track_info
        self.disappeared = {}  # track_id -> frames_disappeared

        # Huffman encoding
        self.encoder = HuffmanEncoder()
        self.coordinate_changes = []
        self.is_trained = False
        self.frames_processed = 0

    def discretize_coordinates(self, x: float, y: float, w: float, h: float,
                               grid_size: int = None) -> Tuple[int, int, int, int]:
        """
        Discretize coordinates to reduce vocabulary size

        Args:
            x, y, w, h: Bounding box coordinates
            grid_size: Grid size for discretization (uses config default if None)

        Returns:
            Discretized coordinates
        """
        if grid_size is None:
            grid_size = self.coordinate_discretization

        return (
            int(x // grid_size) * grid_size,
            int(y // grid_size) * grid_size,
            int(w // grid_size) * grid_size,
            int(h // grid_size) * grid_size
        )

    def encode_coordinate_change(self, prev_bbox: List[int], curr_bbox: List[int]) -> str:
        """
        Encode coordinate change as string

        Args:
            prev_bbox: Previous bounding box [x1, y1, x2, y2]
            curr_bbox: Current bounding box [x1, y1, x2, y2]

        Returns:
            Encoded coordinate change string
        """
        # Convert to center coordinates and dimensions
        prev_cx = (prev_bbox[0] + prev_bbox[2]) // 2
        prev_cy = (prev_bbox[1] + prev_bbox[3]) // 2
        prev_w = prev_bbox[2] - prev_bbox[0]
        prev_h = prev_bbox[3] - prev_bbox[1]

        curr_cx = (curr_bbox[0] + curr_bbox[2]) // 2
        curr_cy = (curr_bbox[1] + curr_bbox[3]) // 2
        curr_w = curr_bbox[2] - curr_bbox[0]
        curr_h = curr_bbox[3] - curr_bbox[1]

        # Discretize coordinates
        prev_cx, prev_cy, prev_w, prev_h = self.discretize_coordinates(
            prev_cx, prev_cy, prev_w, prev_h)
        curr_cx, curr_cy, curr_w, curr_h = self.discretize_coordinates(
            curr_cx, curr_cy, curr_w, curr_h)

        # Calculate changes
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        dw = curr_w - prev_w
        dh = curr_h - prev_h

        # Encode as string
        return f"{dx},{dy},{dw},{dh}"

    def calculate_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate distance between two bounding boxes

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            Distance between bounding boxes
        """
        # Calculate centers
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2

        # Euclidean distance
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def train_huffman_encoder(self):
        """Train Huffman encoder on collected coordinate changes"""
        if not self.coordinate_changes:
            print("No coordinate changes collected. Cannot train Huffman encoder.")
            return

        # Count frequencies
        frequencies = Counter(self.coordinate_changes)

        # Build Huffman tree
        self.encoder.build_tree(frequencies)
        self.encoder.generate_codes()

        self.is_trained = True
        print(
            f"Huffman encoder trained on {len(self.coordinate_changes)} coordinate changes")
        print(f"Vocabulary size: {len(frequencies)}")

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries with 'bbox' key

        Returns:
            Dictionary mapping track_id to track information
        """
        # If no existing tracks, create new ones
        if not self.tracks:
            for detection in detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection.get('confidence', 1.0),
                    'history': [detection['bbox']],
                    'encoded_changes': []
                }
                self.next_id += 1
            return self.tracks

        # Calculate distances between detections and existing tracks
        track_ids = list(self.tracks.keys())
        distances = np.zeros((len(detections), len(track_ids)))

        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                distances[i, j] = self.calculate_distance(
                    detection['bbox'],
                    self.tracks[track_id]['bbox']
                )

        # Hungarian algorithm for assignment (simplified greedy approach)
        used_tracks = set()
        used_detections = set()

        # Sort by distance and assign greedily
        assignments = []
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if distances[i, j] < self.max_distance:
                    assignments.append((i, j, distances[i, j]))

        assignments.sort(key=lambda x: x[2])  # Sort by distance

        # Process assignments
        for det_idx, track_idx, distance in assignments:
            if det_idx not in used_detections and track_idx not in used_tracks:
                track_id = track_ids[track_idx]
                detection = detections[det_idx]

                # Encode coordinate change
                prev_bbox = self.tracks[track_id]['bbox']
                change_str = self.encode_coordinate_change(
                    prev_bbox, detection['bbox'])
                self.coordinate_changes.append(change_str)

                # Update track
                self.tracks[track_id]['bbox'] = detection['bbox']
                self.tracks[track_id]['confidence'] = detection.get(
                    'confidence', 1.0)
                self.tracks[track_id]['history'].append(detection['bbox'])
                self.tracks[track_id]['encoded_changes'].append(change_str)

                # Reset disappeared counter
                if track_id in self.disappeared:
                    del self.disappeared[track_id]

                used_tracks.add(track_idx)
                used_detections.add(det_idx)

        # Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection.get('confidence', 1.0),
                    'history': [detection['bbox']],
                    'encoded_changes': []
                }
                self.next_id += 1

        # Mark disappeared tracks
        for j, track_id in enumerate(track_ids):
            if j not in used_tracks:
                if track_id not in self.disappeared:
                    self.disappeared[track_id] = 0
                self.disappeared[track_id] += 1

        # Remove tracks that have disappeared for too long
        to_remove = []
        for track_id, frames_disappeared in self.disappeared.items():
            if frames_disappeared > self.max_disappeared:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]
            del self.disappeared[track_id]

        return self.tracks

    def get_encoded_trajectories(self) -> Dict[int, str]:
        """
        Get Huffman encoded trajectories for all tracks

        Returns:
            Dictionary mapping track_id to encoded trajectory
        """
        if not self.is_trained:
            self.train_huffman_encoder()

        encoded_trajectories = {}
        for track_id, track_info in self.tracks.items():
            if track_info['encoded_changes']:
                # Join all coordinate changes
                trajectory_str = '|'.join(track_info['encoded_changes'])
                # Encode using Huffman
                encoded_trajectories[track_id] = self.encoder.encode(
                    trajectory_str)

        return encoded_trajectories

    def save_encoder(self, filepath: str):
        """Save trained Huffman encoder"""
        if not self.is_trained:
            print("Encoder not trained yet.")
            return

        with open(filepath, 'wb') as f:
            pickle.dump({
                'codes': self.encoder.codes,
                'coordinate_changes': self.coordinate_changes
            }, f)

        print(f"Encoder saved to {filepath}")

    def load_encoder(self, filepath: str):
        """Load trained Huffman encoder"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.encoder.codes = data['codes']
        self.coordinate_changes = data['coordinate_changes']

        # Rebuild tree from codes (simplified)
        frequencies = Counter(self.coordinate_changes)
        self.encoder.build_tree(frequencies)

        self.is_trained = True
        print(f"Encoder loaded from {filepath}")


def main():
    """Example usage"""
    # Create tracker
    tracker = PersonTracker(max_disappeared=30, max_distance=100)

    # Simulate detections for multiple frames
    frame_detections = [
        [{'bbox': [100, 100, 200, 300], 'confidence': 0.9}],
        [{'bbox': [105, 105, 205, 305], 'confidence': 0.8}],
        [{'bbox': [110, 110, 210, 310], 'confidence': 0.85}],
        [{'bbox': [115, 115, 215, 315], 'confidence': 0.9}],
    ]

    # Process frames
    for frame_idx, detections in enumerate(frame_detections):
        print(f"\nFrame {frame_idx}:")
        tracks = tracker.update(detections)

        for track_id, track_info in tracks.items():
            bbox = track_info['bbox']
            print(
                f"  Track {track_id}: bbox={bbox}, confidence={track_info['confidence']:.2f}")

    # Get encoded trajectories
    encoded_trajectories = tracker.get_encoded_trajectories()
    print(f"\nEncoded trajectories:")
    for track_id, encoded in encoded_trajectories.items():
        print(f"  Track {track_id}: {encoded[:50]}...")  # Show first 50 chars

    # Save encoder
    tracker.save_encoder('huffman_encoder.pkl')


if __name__ == "__main__":
    main()
