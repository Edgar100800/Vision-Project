"""
Track manager for persistent person tracking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

from ..utils.data_structures import Person, Track, Detection, TrackingResult
from ..utils.config_loader import ConfigLoader
from ..reid.reid_matcher import ReIDMatcher


class TrackManager:
    """Manage persistent tracks across video frames."""
    
    def __init__(self, config: ConfigLoader, reid_matcher: ReIDMatcher):
        """
        Initialize track manager.
        
        Args:
            config: Configuration loader
            reid_matcher: Re-identification matcher
        """
        self.config = config
        self.tracking_config = config.get_tracking_config()
        self.reid_matcher = reid_matcher
        
        # Tracking parameters
        self.max_disappeared = self.tracking_config.get("max_disappeared", 30)
        self.min_track_length = self.tracking_config.get("min_track_length", 10)
        self.prediction_window = self.tracking_config.get("prediction_window", 5)
        self.use_kalman_filter = self.tracking_config.get("use_kalman_filter", True)
        
        # Track storage
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.next_person_id = 0
        
        # Track statistics
        self.total_tracks = 0
        self.active_tracks = 0
        self.lost_tracks = 0
        
        logging.info("TrackManager initialized")
    
    def update_tracks(self, frame_id: int, timestamp: float, persons: List[Person]) -> TrackingResult:
        """
        Update tracks with new detections.
        
        Args:
            frame_id: Current frame ID
            timestamp: Current timestamp
            persons: List of detected persons
            
        Returns:
            Tracking result
        """
        # Update disappeared frames for existing tracks
        self._update_disappeared_tracks()
        
        # Match new detections with existing tracks
        matches = self._match_detections_to_tracks(persons)
        
        # Update matched tracks
        self._update_matched_tracks(matches, frame_id, timestamp)
        
        # Create new tracks for unmatched detections
        new_tracks = self._create_new_tracks(persons, matches, frame_id, timestamp)
        
        # Remove lost tracks
        lost_tracks = self._remove_lost_tracks()
        
        # Create tracking result
        result = TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            persons=persons,
            tracks=list(self.tracks.values()),
            new_tracks=new_tracks,
            lost_tracks=lost_tracks
        )
        
        return result
    
    def _update_disappeared_tracks(self):
        """Update disappeared frame count for all tracks."""
        for track in self.tracks.values():
            if track.is_active:
                track.update_disappeared()
    
    def _match_detections_to_tracks(self, persons: List[Person]) -> List[Tuple[Person, Track]]:
        """
        Match detections to existing tracks using ReID.
        
        Args:
            persons: List of detected persons
            
        Returns:
            List of (person, track) matches
        """
        if not persons or not self.tracks:
            return []
        
        # Get active tracks
        active_tracks = [track for track in self.tracks.values() if track.is_active]
        
        if not active_tracks:
            return []
        
        # Perform ReID matching
        reid_matches = self.reid_matcher.match_persons(persons, active_tracks)
        
        # Convert to (person, track) pairs
        matches = []
        matched_person_ids = set()
        matched_track_ids = set()
        
        for match in reid_matches:
            query_person = match.query_person
            gallery_track = self.tracks[match.gallery_person.track_id]
            
            if query_person.person_id not in matched_person_ids and gallery_track.track_id not in matched_track_ids:
                matches.append((query_person, gallery_track))
                matched_person_ids.add(query_person.person_id)
                matched_track_ids.add(gallery_track.track_id)
        
        return matches
    
    def _update_matched_tracks(self, matches: List[Tuple[Person, Track]], frame_id: int, timestamp: float):
        """Update tracks with matched detections."""
        for person, track in matches:
            # Add detection to track
            track.add_detection(person.detection, person.embedding)
            
            # Update person ID if not set
            if person.person_id is None:
                person.person_id = track.person_id
                person.track_id = track.track_id
    
    def _create_new_tracks(self, persons: List[Person], matches: List[Tuple[Person, Track]], 
                          frame_id: int, timestamp: float) -> List[Track]:
        """
        Create new tracks for unmatched detections.
        
        Args:
            persons: List of detected persons
            matches: List of matched (person, track) pairs
            
        Returns:
            List of newly created tracks
        """
        matched_person_ids = {person.person_id for person, _ in matches}
        unmatched_persons = [p for p in persons if p.person_id not in matched_person_ids]
        
        new_tracks = []
        
        for person in unmatched_persons:
            # Create new track
            track = Track(
                track_id=self.next_track_id,
                person_id=self.next_person_id
            )
            
            # Add detection
            track.add_detection(person.detection, person.embedding)
            
            # Update person IDs
            person.person_id = self.next_person_id
            person.track_id = self.next_track_id
            
            # Store track
            self.tracks[self.next_track_id] = track
            
            new_tracks.append(track)
            
            # Update counters
            self.next_track_id += 1
            self.next_person_id += 1
            self.total_tracks += 1
            self.active_tracks += 1
        
        return new_tracks
    
    def _remove_lost_tracks(self) -> List[Track]:
        """
        Remove tracks that have been lost for too long.
        
        Returns:
            List of removed tracks
        """
        lost_tracks = []
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.disappeared_frames > self.max_disappeared:
                # Check if track is long enough to keep
                if track.total_frames >= self.min_track_length:
                    lost_tracks.append(track)
                
                tracks_to_remove.append(track_id)
        
        # Remove tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.active_tracks -= 1
            self.lost_tracks += 1
        
        return lost_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_person_tracks(self, person_id: int) -> List[Track]:
        """Get all tracks for a specific person."""
        return [track for track in self.tracks.values() if track.person_id == person_id]
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return [track for track in self.tracks.values() if track.is_active]
    
    def get_track_statistics(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = self.get_active_tracks()
        
        if not active_tracks:
            return {
                'total_tracks': self.total_tracks,
                'active_tracks': 0,
                'lost_tracks': self.lost_tracks,
                'avg_track_length': 0,
                'avg_track_duration': 0
            }
        
        track_lengths = [track.total_frames for track in active_tracks]
        track_durations = [track.get_duration() for track in active_tracks]
        
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': len(active_tracks),
            'lost_tracks': self.lost_tracks,
            'avg_track_length': np.mean(track_lengths),
            'avg_track_duration': np.mean(track_durations),
            'min_track_length': np.min(track_lengths),
            'max_track_length': np.max(track_lengths)
        }
    
    def predict_track_positions(self, prediction_steps: int = 5) -> Dict[int, Tuple[float, float]]:
        """
        Predict future positions of active tracks.
        
        Args:
            prediction_steps: Number of steps to predict
            
        Returns:
            Dictionary mapping track_id to predicted position
        """
        predictions = {}
        
        for track in self.get_active_tracks():
            if len(track.trajectory) >= 2:
                # Simple linear prediction
                recent_points = track.trajectory[-2:]
                dx = recent_points[-1][0] - recent_points[-2][0]
                dy = recent_points[-1][1] - recent_points[-2][1]
                
                last_x, last_y, _ = recent_points[-1]
                predicted_x = last_x + dx * prediction_steps
                predicted_y = last_y + dy * prediction_steps
                
                predictions[track.track_id] = (predicted_x, predicted_y)
        
        return predictions
    
    def merge_tracks(self, track_id1: int, track_id2: int) -> bool:
        """
        Merge two tracks (e.g., when they belong to the same person).
        
        Args:
            track_id1: First track ID
            track_id2: Second track ID
            
        Returns:
            True if merge successful
        """
        if track_id1 not in self.tracks or track_id2 not in self.tracks:
            return False
        
        track1 = self.tracks[track_id1]
        track2 = self.tracks[track_id2]
        
        # Merge detections and embeddings
        track1.detections.extend(track2.detections)
        track1.embeddings.extend(track2.embeddings)
        track1.trajectory.extend(track2.trajectory)
        
        # Update statistics
        track1.total_frames += track2.total_frames
        track1.last_seen = max(track1.last_seen, track2.last_seen)
        
        # Remove second track
        del self.tracks[track_id2]
        self.active_tracks -= 1
        
        return True
    
    def reset(self):
        """Reset track manager state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.next_person_id = 0
        self.total_tracks = 0
        self.active_tracks = 0
        self.lost_tracks = 0 