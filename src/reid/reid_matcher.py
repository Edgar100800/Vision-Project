"""
Person re-identification matcher.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.data_structures import Person, Track, ReIDMatch
from ..utils.config_loader import ConfigLoader
from ..embedding.embedding_extractor import EmbeddingExtractor


class ReIDMatcher:
    """Match persons using embedding similarity."""
    
    def __init__(self, config: ConfigLoader, embedding_extractor: EmbeddingExtractor):
        """
        Initialize ReID matcher.
        
        Args:
            config: Configuration loader
            embedding_extractor: Embedding extractor instance
        """
        self.config = config
        self.reid_config = config.get_reid_config()
        self.embedding_extractor = embedding_extractor
        
        # Matching parameters
        self.similarity_threshold = self.reid_config.get("similarity_threshold", 0.7)
        self.max_distance = self.reid_config.get("max_distance", 0.3)
        self.matching_algorithm = self.reid_config.get("matching_algorithm", "cosine")
        
        logging.info(f"ReIDMatcher initialized with threshold: {self.similarity_threshold}")
    
    def match_persons(self, query_persons: List[Person], gallery_tracks: List[Track]) -> List[ReIDMatch]:
        """
        Match query persons with gallery tracks.
        
        Args:
            query_persons: List of persons to match
            gallery_tracks: List of existing tracks
            
        Returns:
            List of matches
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
                
                # Get track embedding (use average of recent embeddings)
                track_embedding = track.get_average_embedding()
                if track_embedding is None:
                    continue
                
                # Compute similarity
                similarity = self.embedding_extractor.compute_similarity(
                    query_person.embedding, 
                    track_embedding, 
                    self.matching_algorithm
                )
                
                # Check if this is the best match so far
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = track
            
            # Create match if found
            if best_match is not None:
                distance = 1.0 - best_score  # Convert similarity to distance
                match = ReIDMatch(
                    query_person=query_person,
                    gallery_person=Person(
                        detection=best_match.get_latest_detection(),
                        embedding=best_match.get_latest_embedding(),
                        person_id=best_match.person_id,
                        track_id=best_match.track_id
                    ),
                    similarity_score=best_score,
                    distance=distance,
                    is_match=True
                )
                matches.append(match)
        
        return matches
    
    def match_with_distance_matrix(self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance matrix between query and gallery embeddings.
        
        Args:
            query_embeddings: Query embeddings (N x D)
            gallery_embeddings: Gallery embeddings (M x D)
            
        Returns:
            Distance matrix and similarity matrix
        """
        if self.matching_algorithm == "cosine":
            # Cosine similarity
            similarity_matrix = cosine_similarity(query_embeddings, gallery_embeddings)
            distance_matrix = 1.0 - similarity_matrix
        
        elif self.matching_algorithm == "euclidean":
            # Euclidean distance
            distance_matrix = cdist(query_embeddings, gallery_embeddings, metric='euclidean')
            similarity_matrix = 1.0 / (1.0 + distance_matrix)
        
        elif self.matching_algorithm == "manhattan":
            # Manhattan distance
            distance_matrix = cdist(query_embeddings, gallery_embeddings, metric='manhattan')
            similarity_matrix = 1.0 / (1.0 + distance_matrix)
        
        else:
            raise ValueError(f"Unsupported matching algorithm: {self.matching_algorithm}")
        
        return distance_matrix, similarity_matrix
    
    def find_matches_from_matrix(self, similarity_matrix: np.ndarray, query_persons: List[Person], gallery_tracks: List[Track]) -> List[ReIDMatch]:
        """
        Find matches from similarity matrix using Hungarian algorithm.
        
        Args:
            similarity_matrix: Similarity matrix (N x M)
            query_persons: List of query persons
            gallery_tracks: List of gallery tracks
            
        Returns:
            List of matches
        """
        from scipy.optimize import linear_sum_assignment
        
        # Apply threshold
        thresholded_matrix = similarity_matrix.copy()
        thresholded_matrix[similarity_matrix < self.similarity_threshold] = 0
        
        # Use Hungarian algorithm for optimal assignment
        query_indices, gallery_indices = linear_sum_assignment(-thresholded_matrix)
        
        matches = []
        for query_idx, gallery_idx in zip(query_indices, gallery_indices):
            similarity = similarity_matrix[query_idx, gallery_idx]
            
            if similarity >= self.similarity_threshold:
                query_person = query_persons[query_idx]
                gallery_track = gallery_tracks[gallery_idx]
                
                match = ReIDMatch(
                    query_person=query_person,
                    gallery_person=Person(
                        detection=gallery_track.get_latest_detection(),
                        embedding=gallery_track.get_latest_embedding(),
                        person_id=gallery_track.person_id,
                        track_id=gallery_track.track_id
                    ),
                    similarity_score=similarity,
                    distance=1.0 - similarity,
                    is_match=True
                )
                matches.append(match)
        
        return matches
    
    def filter_matches(self, matches: List[ReIDMatch], max_distance: Optional[float] = None) -> List[ReIDMatch]:
        """
        Filter matches based on distance threshold.
        
        Args:
            matches: List of matches to filter
            max_distance: Maximum allowed distance (overrides config)
            
        Returns:
            Filtered list of matches
        """
        if max_distance is None:
            max_distance = self.max_distance
        
        filtered_matches = []
        
        for match in matches:
            if match.distance <= max_distance:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def get_match_statistics(self, matches: List[ReIDMatch]) -> Dict:
        """
        Get statistics about matches.
        
        Args:
            matches: List of matches
            
        Returns:
            Dictionary with statistics
        """
        if not matches:
            return {
                'count': 0,
                'avg_similarity': 0.0,
                'avg_distance': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        similarities = [m.similarity_score for m in matches]
        distances = [m.distance for m in matches]
        
        return {
            'count': len(matches),
            'avg_similarity': np.mean(similarities),
            'avg_distance': np.mean(distances),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
    
    def compute_rank_accuracy(self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, 
                            query_labels: List[int], gallery_labels: List[int], 
                            ranks: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """
        Compute rank accuracy for re-identification evaluation.
        
        Args:
            query_embeddings: Query embeddings
            gallery_embeddings: Gallery embeddings
            query_labels: Query person labels
            gallery_labels: Gallery person labels
            ranks: List of ranks to compute
            
        Returns:
            Dictionary with rank accuracies
        """
        # Compute similarity matrix
        _, similarity_matrix = self.match_with_distance_matrix(query_embeddings, gallery_embeddings)
        
        rank_accuracies = {}
        
        for rank in ranks:
            correct_matches = 0
            
            for i, query_label in enumerate(query_labels):
                # Get top-k matches for this query
                top_k_indices = np.argsort(similarity_matrix[i])[::-1][:rank]
                
                # Check if correct match is in top-k
                for idx in top_k_indices:
                    if gallery_labels[idx] == query_label:
                        correct_matches += 1
                        break
            
            rank_accuracies[rank] = correct_matches / len(query_labels)
        
        return rank_accuracies 