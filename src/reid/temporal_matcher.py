"""
Temporal matching for person re-identification.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

from ..utils.data_structures import Person, Track, ReIDMatch
from ..utils.config_loader import ConfigLoader


class TemporalMatcher:
    """Handle temporal aspects of person re-identification."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize temporal matcher.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.reid_config = config.get_reid_config()
        
        # Temporal parameters
        self.temporal_window = self.reid_config.get("temporal_window", 30)
        self.use_temporal_features = self.reid_config.get("use_temporal_features", True)
        
        logging.info("TemporalMatcher initialized")
    
    def match_with_temporal_context(self, query_persons: List[Person], 
                                  gallery_tracks: List[Track]) -> List[ReIDMatch]:
        """
        Match persons considering temporal context.
        
        Args:
            query_persons: List of persons to match
            gallery_tracks: List of existing tracks
            
        Returns:
            List of matches with temporal context
        """
        if not query_persons or not gallery_tracks:
            return []
        
        matches = []
        
        for query_person in query_persons:
            if query_person.embedding is None:
                continue
            
            best_match = None
            best_score = -1
            
            for track in gallery_tracks:
                if not track.is_active:
                    continue
                
                # Calculate temporal similarity
                temporal_score = self._calculate_temporal_similarity(query_person, track)
                
                # Combine with embedding similarity
                if track.get_average_embedding() is not None:
                    embedding_similarity = self._calculate_embedding_similarity(
                        query_person.embedding, track.get_average_embedding()
                    )
                    
                    # Weighted combination
                    combined_score = 0.7 * embedding_similarity + 0.3 * temporal_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = track
            
            # Create match if found
            if best_match is not None and best_score >= 0.6:
                match = ReIDMatch(
                    query_person=query_person,
                    gallery_person=Person(
                        detection=best_match.get_latest_detection(),
                        embedding=best_match.get_latest_embedding(),
                        person_id=best_match.person_id,
                        track_id=best_match.track_id
                    ),
                    similarity_score=best_score,
                    distance=1.0 - best_score,
                    is_match=True
                )
                matches.append(match)
        
        return matches
    
    def _calculate_temporal_similarity(self, query_person: Person, track: Track) -> float:
        """
        Calculate temporal similarity between query person and track.
        
        Args:
            query_person: Query person
            track: Gallery track
            
        Returns:
            Temporal similarity score
        """
        if not track.trajectory:
            return 0.0
        
        # Get recent trajectory
        recent_trajectory = track.get_trajectory_window(self.temporal_window)
        
        if not recent_trajectory:
            return 0.0
        
        # Calculate temporal features
        query_features = self._extract_temporal_features(query_person)
        track_features = self._extract_temporal_features_from_trajectory(recent_trajectory)
        
        # Calculate similarity
        similarity = self._cosine_similarity(query_features, track_features)
        
        return similarity
    
    def _extract_temporal_features(self, person: Person) -> np.ndarray:
        """
        Extract temporal features from person.
        
        Args:
            person: Person object
            
        Returns:
            Temporal feature vector
        """
        detection = person.detection
        
        # Basic temporal features
        timestamp = detection.timestamp
        confidence = detection.confidence
        
        # Position features
        center = detection.get_center()
        
        # Create feature vector
        features = np.array([
            timestamp,
            confidence,
            center[0],  # x position
            center[1],  # y position
            detection.get_area(),
            detection.get_aspect_ratio()
        ])
        
        return features
    
    def _extract_temporal_features_from_trajectory(self, trajectory: List[tuple]) -> np.ndarray:
        """
        Extract temporal features from trajectory.
        
        Args:
            trajectory: List of (x, y, timestamp) tuples
            
        Returns:
            Temporal feature vector
        """
        if not trajectory:
            return np.zeros(6)
        
        # Calculate trajectory statistics
        timestamps = [t[2] for t in trajectory]
        x_positions = [t[0] for t in trajectory]
        y_positions = [t[1] for t in trajectory]
        
        # Temporal features
        avg_timestamp = np.mean(timestamps)
        timestamp_std = np.std(timestamps)
        avg_x = np.mean(x_positions)
        avg_y = np.mean(y_positions)
        x_std = np.std(x_positions)
        y_std = np.std(y_positions)
        
        # Create feature vector
        features = np.array([
            avg_timestamp,
            timestamp_std,
            avg_x,
            avg_y,
            x_std,
            y_std
        ])
        
        return features
    
    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        return self._cosine_similarity(embedding1, embedding2)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def predict_person_reappearance(self, lost_tracks: List[Track], 
                                  current_time: float) -> List[Tuple[Track, float]]:
        """
        Predict when lost persons might reappear.
        
        Args:
            lost_tracks: List of lost tracks
            current_time: Current timestamp
            
        Returns:
            List of (track, predicted_time) tuples
        """
        predictions = []
        
        for track in lost_tracks:
            if len(track.trajectory) < 2:
                continue
            
            # Simple prediction based on last known velocity
            recent_trajectory = track.trajectory[-5:]  # Last 5 points
            
            if len(recent_trajectory) >= 2:
                # Calculate average velocity
                velocities = []
                for i in range(1, len(recent_trajectory)):
                    x1, y1, t1 = recent_trajectory[i-1]
                    x2, y2, t2 = recent_trajectory[i]
                    
                    dt = t2 - t1
                    if dt > 0:
                        vx = (x2 - x1) / dt
                        vy = (y2 - y1) / dt
                        velocities.append([vx, vy])
                
                if velocities:
                    avg_velocity = np.mean(velocities, axis=0)
                    
                    # Predict reappearance time (simple heuristic)
                    # Assume person will reappear in 5-15 seconds
                    predicted_time = current_time + np.random.uniform(5, 15)
                    
                    predictions.append((track, predicted_time))
        
        return predictions
    
    def analyze_temporal_patterns(self, tracks: List[Track]) -> Dict:
        """
        Analyze temporal patterns in tracks.
        
        Args:
            tracks: List of tracks
            
        Returns:
            Dictionary with temporal analysis
        """
        if not tracks:
            return {}
        
        # Collect temporal statistics
        durations = []
        reappearance_intervals = []
        
        for track in tracks:
            duration = track.get_duration()
            if duration > 0:
                durations.append(duration)
        
        # Calculate statistics
        analysis = {
            'total_tracks': len(tracks),
            'avg_duration': np.mean(durations) if durations else 0,
            'min_duration': np.min(durations) if durations else 0,
            'max_duration': np.max(durations) if durations else 0,
            'duration_std': np.std(durations) if durations else 0
        }
        
        return analysis 