"""
Persistent person tracker with global history.
"""

import numpy as np
import pickle
import json
from typing import List, Dict, Optional, Tuple, Set
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from ..utils.data_structures import Person, Track, Detection, TrackingResult, ReIDMatch
from ..utils.config_loader import ConfigLoader
from ..reid.reid_matcher import ReIDMatcher


class PersonHistory:
    """Represents the history of a person across multiple sessions."""
    
    def __init__(self, person_id: int, first_seen: float):
        self.person_id = person_id
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.total_appearances = 0
        self.total_duration = 0.0
        self.embeddings_history = deque(maxlen=100)  # Keep last 100 embeddings
        self.appearance_times = []  # List of (start_time, end_time) tuples
        self.current_session_start = None
        
    def add_embedding(self, embedding: np.ndarray):
        """Add embedding to history."""
        self.embeddings_history.append(embedding)
    
    def start_session(self, timestamp: float):
        """Start a new appearance session."""
        self.current_session_start = timestamp
        self.total_appearances += 1
    
    def end_session(self, timestamp: float):
        """End current appearance session."""
        if self.current_session_start is not None:
            duration = timestamp - self.current_session_start
            self.total_duration += duration
            self.appearance_times.append((self.current_session_start, timestamp))
            self.current_session_start = None
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from history."""
        if not self.embeddings_history:
            return None
        return np.mean(list(self.embeddings_history), axis=0)
    
    def get_recent_embeddings(self, num_embeddings: int = 10) -> List[np.ndarray]:
        """Get recent embeddings."""
        return list(self.embeddings_history)[-num_embeddings:]
    
    def update_last_seen(self, timestamp: float):
        """Update last seen timestamp."""
        self.last_seen = timestamp


class PersistentTracker:
    """Persistent person tracker with global history."""
    
    def __init__(self, config: ConfigLoader, reid_matcher: ReIDMatcher):
        """
        Initialize persistent tracker.
        
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
        self.reid_threshold = self.tracking_config.get("reid_threshold", 0.8)  # Increased threshold
        self.history_threshold = self.tracking_config.get("history_threshold", 0.75)  # Increased threshold
        self.max_history_age = self.tracking_config.get("max_history_age", 3600)  # 1 hour
        self.spatial_threshold = self.tracking_config.get("spatial_threshold", 100)  # pixels
        self.min_track_frames = self.tracking_config.get("min_track_frames", 5)  # minimum frames before reid
        
        # Global person history
        self.person_history: Dict[int, PersonHistory] = {}
        self.next_person_id = 0
        
        # Current session tracking
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        
        # ID consistency tracking
        self.id_lock_threshold = self.tracking_config.get("id_lock_threshold", 0.85)  # High threshold to lock ID
        self.id_lock_frames = self.tracking_config.get("id_lock_frames", 3)  # Frames to lock ID
        self.locked_persons: Dict[int, int] = {}  # track_id -> person_id mapping
        self.person_confidence: Dict[int, float] = {}  # track_id -> confidence score
        self.person_lock_frames: Dict[int, int] = {}  # track_id -> frames since lock
        
        # Session statistics
        self.session_start_time = None
        self.total_detections = 0
        self.total_reid_matches = 0
        
        # History file path
        self.history_file = Path("data/persistent/person_history.pkl")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self._load_history()
        
        logging.info("PersistentTracker initialized")
    
    def start_session(self, timestamp: float):
        """Start a new tracking session."""
        self.session_start_time = timestamp
        self.active_tracks.clear()
        self.next_track_id = 0
        
        # End any ongoing sessions in history
        for person_hist in self.person_history.values():
            if person_hist.current_session_start is not None:
                person_hist.end_session(timestamp)
        
        logging.info(f"Started new tracking session at {timestamp}")
    
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
        if self.session_start_time is None:
            self.start_session(timestamp)
        
        # Update disappeared frames for existing tracks
        self._update_disappeared_tracks()
        
        # Match new detections with existing tracks and history
        matches = self._match_with_history(persons, timestamp)
        
        # Update matched tracks
        self._update_matched_tracks(matches, frame_id, timestamp)
        
        # Create new tracks for unmatched detections
        new_tracks = self._create_new_tracks(persons, matches, frame_id, timestamp)
        
        # Remove lost tracks
        lost_tracks = self._remove_lost_tracks(timestamp)
        
        # Clean up locked IDs
        self._cleanup_locked_ids()
        
        # Create tracking result
        result = TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            persons=persons,
            tracks=list(self.active_tracks.values()),
            new_tracks=new_tracks,
            lost_tracks=lost_tracks
        )
        
        return result
    
    def _match_with_history(self, persons: List[Person], timestamp: float) -> List[Tuple[Person, Track, Optional[int], str]]:
        """
        Match detections with active tracks and person history.
        
        Args:
            persons: List of detected persons
            timestamp: Current timestamp
            
        Returns:
            List of (person, track, person_id, match_status) matches
        """
        matches = []
        matched_person_ids = set()
        matched_track_ids = set()
        
        for person in persons:
            if person.embedding is None:
                continue
            
            best_match = None
            best_score = -1
            best_person_id = None
            match_type = None
            match_status = "NEW"
            
            # First, try to match with active tracks (highest priority)
            for track in self.active_tracks.values():
                if not track.is_active or track.track_id in matched_track_ids:
                    continue
                
                # Skip tracks that are too new (need minimum frames for reliable matching)
                if track.total_frames < self.min_track_frames:
                    continue
                
                track_embedding = track.get_average_embedding()
                if track_embedding is None:
                    continue
                
                # Compute similarity
                similarity = self.reid_matcher.embedding_extractor.compute_similarity(
                    person.embedding, track_embedding, "cosine"
                )
                
                # Validate spatial consistency
                spatial_valid = self._validate_spatial_consistency(track, person.embedding, person.person_id, person.confidence)
                
                if similarity > best_score and similarity >= self.reid_threshold and spatial_valid:
                    best_score = similarity
                    best_match = track
                    best_person_id = track.person_id
                    match_type = "ACTIVE_TRACK"
                    match_status = "TRACKING"
            
            # Create match if found with active track
            if best_match is not None:
                track_id = best_match.track_id
                
                # If track already has a locked ID, ALWAYS use it
                if track_id in self.locked_persons:
                    best_person_id = self.locked_persons[track_id]
                    match_status = "TRACKING_LOCKED"
                    logging.info(f"Using locked ID {best_person_id} for track {track_id}")
                else:
                    # Check if we should lock this ID - be more aggressive
                    should_lock = False
                    
                    # Lock immediately if high confidence
                    if best_score >= self.id_lock_threshold:
                        should_lock = True
                    # Lock if we've seen this track for enough frames with decent confidence
                    elif (track_id in self.person_lock_frames and 
                          self.person_lock_frames[track_id] >= self.id_lock_frames and
                          best_score >= 0.6):  # Lower threshold for locking
                        should_lock = True
                    
                    if should_lock and best_person_id is not None:
                        self.locked_persons[track_id] = best_person_id
                        self.person_confidence[track_id] = best_score
                        self.person_lock_frames[track_id] = 0
                        match_status = "TRACKING_LOCKED"
                        logging.info(f"Locked ID {best_person_id} for track {track_id} (score: {best_score:.3f})")
                    else:
                        # Update confidence and frame count
                        self.person_confidence[track_id] = best_score
                        if track_id in self.person_lock_frames:
                            self.person_lock_frames[track_id] += 1
                        else:
                            self.person_lock_frames[track_id] = 0
                
                matches.append((person, best_match, best_person_id, match_status))
                matched_person_ids.add(best_person_id)
                matched_track_ids.add(best_match.track_id)
                
                logging.info(f"Person matching: {match_type} | Score: {best_score:.3f} | "
                           f"Person ID: {best_person_id} | Status: {match_status}")
                continue  # Skip history matching if we found an active track
            
            # Only try history matching if no active track was found
            for person_id, person_hist in self.person_history.items():
                if person_id in matched_person_ids:
                    continue
                
                # Check if person was seen recently
                if timestamp - person_hist.last_seen > self.max_history_age:
                    continue
                
                hist_embedding = person_hist.get_average_embedding()
                if hist_embedding is None:
                    continue
                
                similarity = self.reid_matcher.embedding_extractor.compute_similarity(
                    person.embedding, hist_embedding, "cosine"
                )
                
                # For history matching, require higher confidence
                if similarity > best_score and similarity >= self.history_threshold:
                    best_score = similarity
                    best_match = None  # Will create new track
                    best_person_id = person_id
                    match_type = "HISTORY"
                    match_status = "RE_IDENTIFIED"
            
            # Create match if found with history
            if best_match is None and best_person_id is not None:
                matches.append((person, best_match, best_person_id, match_status))
                matched_person_ids.add(best_person_id)
                
                logging.info(f"Person matching: {match_type} | Score: {best_score:.3f} | "
                           f"Person ID: {best_person_id} | Status: {match_status}")
        
        return matches
    
    def _validate_spatial_consistency(self, track, person_embedding, person_id, confidence_score):
        """
        Validate spatial consistency using Kalman filter predictions.
        """
        if not hasattr(track, 'kalman_filter') or track.kalman_filter is None:
            return True
            
        # Get current position
        current_bbox = track.get_current_bbox()
        if current_bbox is None:
            return True
            
        current_center = np.array([
            (current_bbox[0] + current_bbox[2]) / 2,
            (current_bbox[1] + current_bbox[3]) / 2
        ])
        
        # Predict next position using Kalman filter
        predicted_center = track.kalman_filter.predict()
        predicted_center = predicted_center[:2]  # Take only x, y coordinates
        
        # Calculate distance between current and predicted position
        distance = np.linalg.norm(current_center - predicted_center)
        
        # If distance is too large, the prediction is unreliable
        if distance > 100:  # pixels
            return True  # Allow re-identification if prediction is unreliable
            
        # Check if the person ID change makes spatial sense
        if track.person_id != person_id:
            # Get historical positions for both person IDs
            person_0_positions = self._get_person_positions(track.person_id)
            person_1_positions = self._get_person_positions(person_id)
            
            if len(person_0_positions) > 0 and len(person_1_positions) > 0:
                # Calculate average positions
                avg_pos_0 = np.mean(person_0_positions, axis=0)
                avg_pos_1 = np.mean(person_1_positions, axis=0)
                
                # Calculate distances
                dist_to_0 = np.linalg.norm(current_center - avg_pos_0)
                dist_to_1 = np.linalg.norm(current_center - avg_pos_1)
                
                # If current position is much closer to the original person,
                # don't allow the swap
                if dist_to_0 < dist_to_1 * 0.7:  # 30% closer to original
                    return False
                    
        return True

    def _get_person_positions(self, person_id):
        """
        Get historical positions for a person ID.
        """
        positions = []
        for track in self.active_tracks.values(): # Changed from self.tracks to self.active_tracks.values()
            if track.person_id == person_id and track.is_active:
                bbox = track.get_current_bbox()
                if bbox is not None:
                    center = np.array([
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2
                    ])
                    positions.append(center)
        return positions
    
    def _update_matched_tracks(self, matches: List[Tuple[Person, Track, Optional[int], str]], frame_id: int, timestamp: float):
        """Update tracks with matched detections."""
        for person, track, person_id, match_status in matches:
            if track is not None:
                # Update existing track
                track.add_detection(person.detection, person.embedding)
                person.person_id = track.person_id
                person.track_id = track.track_id
                
                # Update person history
                if track.person_id in self.person_history:
                    if person.embedding is not None:
                        self.person_history[track.person_id].add_embedding(person.embedding)
                    self.person_history[track.person_id].update_last_seen(timestamp)
            else:
                # Create new track for re-identified person
                if person_id is not None:
                    new_track = Track(
                        track_id=self.next_track_id,
                        person_id=person_id
                    )
                    new_track.add_detection(person.detection, person.embedding)
                    
                    self.active_tracks[self.next_track_id] = new_track
                    person.person_id = person_id
                    person.track_id = self.next_track_id
                    
                    # Update person history
                    if person_id in self.person_history:
                        self.person_history[person_id].start_session(timestamp)
                        if person.embedding is not None:
                            self.person_history[person_id].add_embedding(person.embedding)
                        self.person_history[person_id].update_last_seen(timestamp)
                    
                    self.next_track_id += 1
                    self.total_reid_matches += 1
    
    def _create_new_tracks(self, persons: List[Person], matches: List[Tuple[Person, Track, Optional[int], str]], 
                          frame_id: int, timestamp: float) -> List[Track]:
        """Create new tracks for unmatched detections."""
        matched_person_ids = {person.person_id for person, _, _, _ in matches}
        unmatched_persons = [p for p in persons if p.person_id not in matched_person_ids]
        
        new_tracks = []
        
        for person in unmatched_persons:
            # Create new person ID
            person_id = self.next_person_id
            self.next_person_id += 1
            
            # Create new track
            track = Track(
                track_id=self.next_track_id,
                person_id=person_id
            )
            track.add_detection(person.detection, person.embedding)
            
            # Create person history
            person_hist = PersonHistory(person_id, timestamp)
            person_hist.start_session(timestamp)
            if person.embedding is not None:
                person_hist.add_embedding(person.embedding)
            self.person_history[person_id] = person_hist
            
            # Update person IDs
            person.person_id = person_id
            person.track_id = self.next_track_id
            
            # Store track
            self.active_tracks[self.next_track_id] = track
            new_tracks.append(track)
            
            self.next_track_id += 1
            self.total_detections += 1
        
        return new_tracks
    
    def _update_disappeared_tracks(self):
        """Update disappeared frame count for all tracks."""
        for track in self.active_tracks.values():
            if track.is_active:
                track.update_disappeared()
    
    def _remove_lost_tracks(self, timestamp: float) -> List[Track]:
        """Remove tracks that have been lost for too long."""
        lost_tracks = []
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            if track.disappeared_frames > self.max_disappeared:
                if track.total_frames >= self.min_track_length:
                    lost_tracks.append(track)
                    
                    # End session in history
                    if track.person_id in self.person_history:
                        self.person_history[track.person_id].end_session(timestamp)
                
                tracks_to_remove.append(track_id)
        
        # Remove tracks and clean up locked IDs
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            # Clean up locked ID if this track was locked
            if track_id in self.locked_persons:
                del self.locked_persons[track_id]
            if track_id in self.person_confidence:
                del self.person_confidence[track_id]
            if track_id in self.person_lock_frames:
                del self.person_lock_frames[track_id]
        
        return lost_tracks
    
    def _cleanup_locked_ids(self):
        """Clean up locked IDs for tracks that no longer exist."""
        existing_track_ids = set(self.active_tracks.keys())
        
        # Remove locked IDs for non-existent tracks
        locked_track_ids = list(self.locked_persons.keys())
        for track_id in locked_track_ids:
            if track_id not in existing_track_ids:
                del self.locked_persons[track_id]
                if track_id in self.person_confidence:
                    del self.person_confidence[track_id]
                if track_id in self.person_lock_frames:
                    del self.person_lock_frames[track_id]
                logging.info(f"Cleaned up locked ID for non-existent track {track_id}")
    
    def get_locked_ids_info(self) -> Dict:
        """Get information about locked IDs."""
        return {
            'locked_persons': self.locked_persons.copy(),
            'person_confidence': self.person_confidence.copy(),
            'person_lock_frames': self.person_lock_frames.copy()
        }
    
    def end_session(self, timestamp: float):
        """End current tracking session."""
        # End all ongoing sessions
        for person_hist in self.person_history.values():
            if person_hist.current_session_start is not None:
                person_hist.end_session(timestamp)
        
        # Save history
        self._save_history()
        
        logging.info(f"Ended tracking session at {timestamp}")
    
    def get_person_history(self, person_id: int) -> Optional[PersonHistory]:
        """Get history for a specific person."""
        return self.person_history.get(person_id)
    
    def get_all_person_history(self) -> Dict[int, PersonHistory]:
        """Get all person history."""
        return self.person_history.copy()
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return list(self.active_tracks.values())
    
    def get_track_statistics(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = len([t for t in self.active_tracks.values() if t.is_active])
        total_persons = len(self.person_history)
        
        return {
            'active_tracks': active_tracks,
            'total_persons': total_persons,
            'total_detections': self.total_detections,
            'total_reid_matches': self.total_reid_matches,
            'session_duration': self.session_start_time - datetime.now().timestamp() if self.session_start_time else 0
        }
    
    def _save_history(self):
        """Save person history to file."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.person_history, f)
            logging.info(f"Saved person history to {self.history_file}")
        except Exception as e:
            logging.error(f"Failed to save person history: {e}")
    
    def _load_history(self):
        """Load person history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'rb') as f:
                    self.person_history = pickle.load(f)
                
                # Update next_person_id
                if self.person_history:
                    self.next_person_id = max(self.person_history.keys()) + 1
                
                logging.info(f"Loaded person history: {len(self.person_history)} persons")
        except Exception as e:
            logging.error(f"Failed to load person history: {e}")
            self.person_history = {}
    
    def export_history_json(self, output_file: str):
        """Export person history to JSON format."""
        history_data = {}
        
        for person_id, person_hist in self.person_history.items():
            history_data[person_id] = {
                'person_id': person_id,
                'first_seen': person_hist.first_seen,
                'last_seen': person_hist.last_seen,
                'total_appearances': person_hist.total_appearances,
                'total_duration': person_hist.total_duration,
                'appearance_times': person_hist.appearance_times,
                'embeddings_count': len(person_hist.embeddings_history)
            }
        
        with open(output_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logging.info(f"Exported history to {output_file}")
    
    def reset(self):
        """Reset tracker state."""
        self.active_tracks.clear()
        self.next_track_id = 0
        self.session_start_time = None
        self.total_detections = 0
        self.total_reid_matches = 0 

    def _should_lock_person_id(self, track):
        """
        Determine if a person ID should be locked based on tracking consistency.
        More aggressive locking to prevent ID swaps.
        """
        if not track.is_active:
            return False
            
        # Get track statistics
        track_frames = track.get_track_length()
        consistent_frames = track.get_consistent_frames()
        avg_confidence = track.get_average_confidence()
        
        # Lock if track has been consistent for a good amount of time
        if track_frames >= 15 and consistent_frames >= 10 and avg_confidence > 0.85:
            return True
            
        # Lock if track has very high confidence for a moderate time
        if track_frames >= 10 and avg_confidence > 0.9:
            return True
            
        # Lock if track has been active for a long time (prevent late swaps)
        if track_frames >= 25:
            return True
            
        return False

    def _is_person_id_locked(self, person_id):
        """
        Check if a person ID is locked and should not be reassigned.
        """
        return person_id in self.locked_persons

    def _should_allow_reidentification(self, track, new_person_id, confidence_score):
        """
        Determine if re-identification should be allowed.
        More restrictive to prevent ID swaps.
        """
        # Never allow re-identification if the person ID is locked
        if self._is_person_id_locked(track.person_id):
            return False
            
        # Never allow re-identification if the new person ID is locked
        if self._is_person_id_locked(new_person_id):
            return False
            
        # Don't allow re-identification for well-established tracks
        if track.get_track_length() >= 20:
            return False
            
        # Require very high confidence for re-identification
        if confidence_score < 0.9:
            return False
            
        # Additional spatial validation
        if hasattr(track, 'last_position') and hasattr(track, 'current_position'):
            distance = np.linalg.norm(track.current_position - track.last_position)
            if distance > 200:  # If person moved too much, be suspicious
                return False
                
        return True 