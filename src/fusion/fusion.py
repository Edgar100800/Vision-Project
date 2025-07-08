#!/usr/bin/env python3
"""
Fusion Module for Combining Tracking and ReID Results
Integrates Huffman tracking with ReID features for robust person identification
"""

from config.global_config import get_config
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class IDFusion:
    """Fusion module for combining tracking and ReID results"""

    def __init__(self,
                 similarity_threshold: float = None,
                 tracking_weight: float = None,
                 reid_weight: float = None,
                 max_reid_distance: float = None,
                 temporal_window: int = None):
        """
        Initialize ID fusion module

        Args:
            similarity_threshold: Minimum similarity for ID association (uses config default if None)
            tracking_weight: Weight for tracking confidence (uses config default if None)
            reid_weight: Weight for ReID confidence (uses config default if None)
            max_reid_distance: Maximum ReID distance for valid match (uses config default if None)
            temporal_window: Frames to consider for temporal consistency (uses config default if None)
        """
        # Get configuration
        self.config = get_config('fusion')

        # Use provided values or defaults from config
        self.similarity_threshold = similarity_threshold or self.config['similarity_threshold']
        self.tracking_weight = tracking_weight or self.config['tracking_weight']
        self.reid_weight = reid_weight or self.config['reid_weight']
        self.max_reid_distance = max_reid_distance or self.config['reid_distance_threshold']
        self.temporal_window = temporal_window or self.config['temporal_window']

        # Store other config values
        self.temporal_weight = self.config['temporal_weight']
        self.tracking_distance_threshold = self.config['tracking_distance_threshold']
        self.consistency_threshold = self.config['consistency_threshold']
        self.history_length = self.config['history_length']
        self.max_global_ids = self.config['max_global_ids']
        self.id_reuse_delay = self.config['id_reuse_delay']
        self.inactive_threshold = self.config['inactive_threshold']
        self.cleanup_threshold = self.config['cleanup_threshold']
        self.min_confidence = self.config['min_confidence']
        self.confidence_decay = self.config['confidence_decay']
        self.boost_factor = self.config['boost_factor']
        self.track_quality_weight = self.config['track_quality_weight']
        self.reid_quality_weight = self.config['reid_quality_weight']
        self.appearance_change_threshold = self.config['appearance_change_threshold']
        self.enable_re_identification = self.config['enable_re_identification']
        self.enable_track_merging = self.config['enable_track_merging']
        self.enable_id_recovery = self.config['enable_id_recovery']
        self.occlusion_threshold = self.config['occlusion_threshold']

        # State management
        self.person_database = {}  # global_id -> person_info
        self.track_to_global = {}  # track_id -> global_id
        self.global_to_tracks = defaultdict(list)  # global_id -> [track_ids]
        self.next_global_id = 0

        # Temporal consistency tracking
        # [(frame_idx, track_id, global_id, confidence)]
        self.recent_associations = []

        # Statistics
        self.stats = {
            'total_fusions': 0,
            'reid_matches': 0,
            'tracking_matches': 0,
            'new_identities': 0,
            'false_associations': 0
        }

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update_person_database(self, global_id: int,
                               reid_features: np.ndarray,
                               track_info: Dict,
                               frame_idx: int):
        """
        Update person database with new information

        Args:
            global_id: Global person ID
            reid_features: ReID feature vector
            track_info: Tracking information
            frame_idx: Current frame index
        """
        if global_id not in self.person_database:
            self.person_database[global_id] = {
                'reid_features': [],
                'track_history': [],
                'first_seen': frame_idx,
                'last_seen': frame_idx,
                'confidence_scores': [],
                'bboxes': []
            }

        person_info = self.person_database[global_id]

        # Update ReID features (keep only recent ones)
        person_info['reid_features'].append(reid_features)
        if len(person_info['reid_features']) > 10:  # Keep last 10 features
            person_info['reid_features'] = person_info['reid_features'][-10:]

        # Update tracking history
        person_info['track_history'].append(track_info)
        if len(person_info['track_history']) > 50:  # Keep last 50 track updates
            person_info['track_history'] = person_info['track_history'][-50:]

        # Update temporal info
        person_info['last_seen'] = frame_idx
        person_info['confidence_scores'].append(
            track_info.get('confidence', 1.0))
        person_info['bboxes'].append(track_info.get('bbox', []))

    def compute_reid_similarity(self, features1: np.ndarray,
                                features2_list: List[np.ndarray]) -> float:
        """
        Compute ReID similarity between a feature and a list of features

        Args:
            features1: Query feature vector
            features2_list: List of stored feature vectors

        Returns:
            Maximum similarity score
        """
        if not features2_list:
            return 0.0

        similarities = []
        for features2 in features2_list:
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            similarity = dot_product / (norm1 * norm2 + 1e-8)
            similarities.append(similarity)

        return max(similarities)

    def compute_tracking_confidence(self, track_info: Dict) -> float:
        """
        Compute tracking confidence based on track stability

        Args:
            track_info: Tracking information

        Returns:
            Tracking confidence score
        """
        confidence = track_info.get('confidence', 1.0)

        # Consider track length (longer tracks are more reliable)
        track_length = len(track_info.get('history', []))
        length_bonus = min(track_length / 30.0, 1.0)  # Max bonus at 30 frames

        # Consider bounding box stability
        history = track_info.get('history', [])
        if len(history) > 5:
            # Calculate bbox variance as stability measure
            recent_bboxes = history[-5:]
            bbox_vars = []
            for i in range(4):  # x1, y1, x2, y2
                coords = [bbox[i] for bbox in recent_bboxes if len(bbox) > i]
                if coords:
                    bbox_vars.append(np.var(coords))

            if bbox_vars:
                stability = 1.0 / (1.0 + np.mean(bbox_vars) / 1000.0)
            else:
                stability = 1.0
        else:
            stability = 0.5  # Lower confidence for new tracks

        return confidence * length_bonus * stability

    def compute_temporal_consistency(self, track_id: int,
                                     candidate_global_id: int,
                                     frame_idx: int) -> float:
        """
        Compute temporal consistency score

        Args:
            track_id: Current track ID
            candidate_global_id: Candidate global ID
            frame_idx: Current frame index

        Returns:
            Temporal consistency score
        """
        # Look at recent associations
        recent_window = frame_idx - self.temporal_window
        relevant_associations = [
            assoc for assoc in self.recent_associations
            if assoc[0] > recent_window
        ]

        # Count consistent associations
        consistent_count = 0
        total_count = 0

        for frame, t_id, g_id, conf in relevant_associations:
            if t_id == track_id:
                total_count += 1
                if g_id == candidate_global_id:
                    consistent_count += 1

        if total_count == 0:
            return 0.5  # Neutral score for new tracks

        return consistent_count / total_count

    def fuse_identities(self,
                        tracks: Dict[int, Dict],
                        reid_features: Dict[int, np.ndarray],
                        hnsw_results: Dict[int, Dict],
                        frame_idx: int) -> Dict[int, int]:
        """
        Fuse tracking and ReID results to assign global IDs

        Args:
            tracks: Dictionary of track_id -> track_info
            reid_features: Dictionary of track_id -> reid_features
            hnsw_results: Dictionary of track_id -> hnsw_search_results
            frame_idx: Current frame index

        Returns:
            Dictionary mapping track_id to global_id
        """
        assignments = {}

        for track_id, track_info in tracks.items():
            if track_id not in reid_features:
                # No ReID features available, use tracking only
                if track_id in self.track_to_global:
                    assignments[track_id] = self.track_to_global[track_id]
                else:
                    # Create new global ID
                    global_id = self.next_global_id
                    self.next_global_id += 1
                    assignments[track_id] = global_id
                    self.track_to_global[track_id] = global_id
                    self.global_to_tracks[global_id].append(track_id)
                    self.stats['new_identities'] += 1
                continue

            current_features = reid_features[track_id]
            hnsw_result = hnsw_results.get(track_id, {})

            # Get candidate global IDs from HNSW search
            candidate_global_ids = []
            reid_distances = []

            if 'external_ids' in hnsw_result and 'distances' in hnsw_result:
                for ext_id, distance in zip(hnsw_result['external_ids'],
                                            hnsw_result['distances']):
                    if ext_id is not None and distance < self.max_reid_distance:
                        try:
                            # Extract global_id from "person_X"
                            global_id = int(ext_id.split('_')[-1])
                            candidate_global_ids.append(global_id)
                            reid_distances.append(distance)
                        except (ValueError, AttributeError):
                            continue

            # Compute fusion scores for candidates
            best_global_id = None
            best_score = 0.0

            # Check existing assignment first
            if track_id in self.track_to_global:
                existing_global_id = self.track_to_global[track_id]
                if existing_global_id in self.person_database:
                    # Compute score for existing assignment
                    reid_sim = self.compute_reid_similarity(
                        current_features,
                        self.person_database[existing_global_id]['reid_features']
                    )
                    tracking_conf = self.compute_tracking_confidence(
                        track_info)
                    temporal_cons = self.compute_temporal_consistency(
                        track_id, existing_global_id, frame_idx
                    )

                    existing_score = (
                        self.reid_weight * reid_sim +
                        self.tracking_weight * tracking_conf +
                        0.2 * temporal_cons  # Bonus for temporal consistency
                    )

                    if existing_score > self.similarity_threshold:
                        best_global_id = existing_global_id
                        best_score = existing_score

            # Check HNSW candidates
            for i, candidate_id in enumerate(candidate_global_ids):
                if candidate_id not in self.person_database:
                    continue

                # Reid similarity (convert distance to similarity)
                reid_sim = 1.0 - reid_distances[i]

                # Tracking confidence
                tracking_conf = self.compute_tracking_confidence(track_info)

                # Temporal consistency
                temporal_cons = self.compute_temporal_consistency(
                    track_id, candidate_id, frame_idx
                )

                # Combined score
                fusion_score = (
                    self.reid_weight * reid_sim +
                    self.tracking_weight * tracking_conf +
                    0.1 * temporal_cons
                )

                if fusion_score > best_score and fusion_score > self.similarity_threshold:
                    best_global_id = candidate_id
                    best_score = fusion_score

            # Assign global ID
            if best_global_id is not None:
                assignments[track_id] = best_global_id

                # Update mappings if changed
                if track_id not in self.track_to_global or self.track_to_global[track_id] != best_global_id:
                    # Remove from old global ID if exists
                    if track_id in self.track_to_global:
                        old_global_id = self.track_to_global[track_id]
                        if track_id in self.global_to_tracks[old_global_id]:
                            self.global_to_tracks[old_global_id].remove(
                                track_id)

                    self.track_to_global[track_id] = best_global_id
                    self.global_to_tracks[best_global_id].append(track_id)
                    self.stats['reid_matches'] += 1
                else:
                    self.stats['tracking_matches'] += 1

            else:
                # Create new global ID
                global_id = self.next_global_id
                self.next_global_id += 1
                assignments[track_id] = global_id
                self.track_to_global[track_id] = global_id
                self.global_to_tracks[global_id].append(track_id)
                self.stats['new_identities'] += 1

            # Update person database
            self.update_person_database(
                assignments[track_id],
                current_features,
                track_info,
                frame_idx
            )

            # Record association for temporal consistency
            self.recent_associations.append((
                frame_idx, track_id, assignments[track_id], best_score
            ))

            # Cleanup old associations
            cutoff_frame = frame_idx - self.temporal_window * 2
            self.recent_associations = [
                assoc for assoc in self.recent_associations
                if assoc[0] > cutoff_frame
            ]

        self.stats['total_fusions'] += 1
        return assignments

    def get_person_info(self, global_id: int) -> Optional[Dict]:
        """
        Get information about a person by global ID

        Args:
            global_id: Global person ID

        Returns:
            Person information dictionary or None
        """
        return self.person_database.get(global_id)

    def get_active_persons(self, frame_idx: int,
                           max_absence: int = 100) -> List[int]:
        """
        Get list of currently active person IDs

        Args:
            frame_idx: Current frame index
            max_absence: Maximum frames since last seen

        Returns:
            List of active global IDs
        """
        active_ids = []
        for global_id, person_info in self.person_database.items():
            if frame_idx - person_info['last_seen'] <= max_absence:
                active_ids.append(global_id)
        return active_ids

    def cleanup_inactive_persons(self, frame_idx: int,
                                 max_absence: int = 500):
        """
        Remove inactive persons from database

        Args:
            frame_idx: Current frame index
            max_absence: Maximum frames since last seen before removal
        """
        to_remove = []
        for global_id, person_info in self.person_database.items():
            if frame_idx - person_info['last_seen'] > max_absence:
                to_remove.append(global_id)

        for global_id in to_remove:
            # Remove from database
            del self.person_database[global_id]

            # Remove track mappings
            if global_id in self.global_to_tracks:
                for track_id in self.global_to_tracks[global_id]:
                    if track_id in self.track_to_global:
                        del self.track_to_global[track_id]
                del self.global_to_tracks[global_id]

        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} inactive persons")

    def get_statistics(self) -> Dict:
        """Get fusion statistics"""
        return {
            **self.stats,
            'active_persons': len(self.person_database),
            'total_tracks': len(self.track_to_global),
            'next_global_id': self.next_global_id
        }

    def save_state(self, filepath: str):
        """Save fusion state to file"""
        state = {
            'person_database': {
                k: {
                    **v,
                    'reid_features': [feat.tolist() for feat in v['reid_features']]
                }
                for k, v in self.person_database.items()
            },
            'track_to_global': self.track_to_global,
            'global_to_tracks': dict(self.global_to_tracks),
            'next_global_id': self.next_global_id,
            'stats': self.stats,
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'tracking_weight': self.tracking_weight,
                'reid_weight': self.reid_weight,
                'max_reid_distance': self.max_reid_distance,
                'temporal_window': self.temporal_window
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Fusion state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load fusion state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore person database
        self.person_database = {}
        for k, v in state['person_database'].items():
            global_id = int(k)
            person_info = {**v}
            person_info['reid_features'] = [
                np.array(feat) for feat in v['reid_features']
            ]
            self.person_database[global_id] = person_info

        # Restore mappings
        self.track_to_global = {int(k): int(v)
                                for k, v in state['track_to_global'].items()}
        self.global_to_tracks = defaultdict(list)
        for k, v in state['global_to_tracks'].items():
            self.global_to_tracks[int(k)] = v

        self.next_global_id = state['next_global_id']
        self.stats = state['stats']

        # Restore config
        config = state.get('config', {})
        self.similarity_threshold = config.get(
            'similarity_threshold', self.similarity_threshold)
        self.tracking_weight = config.get(
            'tracking_weight', self.tracking_weight)
        self.reid_weight = config.get('reid_weight', self.reid_weight)
        self.max_reid_distance = config.get(
            'max_reid_distance', self.max_reid_distance)
        self.temporal_window = config.get(
            'temporal_window', self.temporal_window)

        self.logger.info(f"Fusion state loaded from {filepath}")


def main():
    """Example usage"""
    # Create fusion module
    fusion = IDFusion(
        similarity_threshold=0.7,
        tracking_weight=0.4,
        reid_weight=0.6,
        max_reid_distance=0.5,
        temporal_window=30
    )

    # Simulate tracking and ReID results
    frame_idx = 0

    # Frame 1
    tracks = {
        0: {'bbox': [100, 100, 200, 300], 'confidence': 0.9, 'history': [[100, 100, 200, 300]]},
        1: {'bbox': [300, 150, 400, 350], 'confidence': 0.8, 'history': [[300, 150, 400, 350]]}
    }

    reid_features = {
        0: np.random.randn(2048),
        1: np.random.randn(2048)
    }

    hnsw_results = {
        0: {'external_ids': [None], 'distances': [1.0]},  # No match
        1: {'external_ids': [None], 'distances': [1.0]}   # No match
    }

    assignments = fusion.fuse_identities(
        tracks, reid_features, hnsw_results, frame_idx)
    print(f"Frame {frame_idx} assignments: {assignments}")

    # Frame 2 - same persons moved
    frame_idx = 1
    tracks = {
        0: {'bbox': [105, 105, 205, 305], 'confidence': 0.85, 'history': [[100, 100, 200, 300], [105, 105, 205, 305]]},
        1: {'bbox': [305, 155, 405, 355], 'confidence': 0.82, 'history': [[300, 150, 400, 350], [305, 155, 405, 355]]}
    }

    reid_features = {
        0: np.random.randn(2048),
        1: np.random.randn(2048)
    }

    hnsw_results = {
        0: {'external_ids': ['person_0'], 'distances': [0.3]},  # Good match
        1: {'external_ids': ['person_1'], 'distances': [0.4]}   # Good match
    }

    assignments = fusion.fuse_identities(
        tracks, reid_features, hnsw_results, frame_idx)
    print(f"Frame {frame_idx} assignments: {assignments}")

    # Show statistics
    stats = fusion.get_statistics()
    print(f"Fusion statistics: {stats}")

    # Show active persons
    active = fusion.get_active_persons(frame_idx)
    print(f"Active persons: {active}")


if __name__ == "__main__":
    main()
