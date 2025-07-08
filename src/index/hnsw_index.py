#!/usr/bin/env python3
"""
HNSW Index Module for Efficient Vector Search
Uses HNSW (Hierarchical Navigable Small World) for fast similarity search
"""

from config.global_config import get_config
import numpy as np
import hnswlib
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class HNSWIndex:
    """HNSW index for efficient similarity search"""

    def __init__(self, dim: int = None,
                 max_elements: int = None,
                 M: int = None,
                 ef_construction: int = None,
                 ef_search: int = None,
                 space: str = None,
                 random_seed: int = None):
        """
        Initialize HNSW index

        Args:
            dim: Dimension of feature vectors (uses config default if None)
            max_elements: Maximum number of elements that can be stored (uses config default if None)
            M: Number of bi-directional links for each node (uses config default if None)
            ef_construction: Size of dynamic candidate list during construction (uses config default if None)
            ef_search: Size of dynamic candidate list during search (uses config default if None)
            space: Distance metric (uses config default if None)
            random_seed: Random seed for reproducibility (uses config default if None)
        """
        # Get configuration
        self.config = get_config('indexing')

        # Use provided values or defaults from config
        self.dim = dim or self.config['dimension']
        self.max_elements = max_elements or self.config['max_elements']
        self.M = M or self.config['M']
        self.ef_construction = ef_construction or self.config['ef_construction']
        self.ef_search = ef_search or self.config['ef_search']
        self.space = space or self.config['space']
        self.random_seed = random_seed or self.config['seed']

        # Store other config values
        self.default_k = self.config['default_k']
        self.max_k = self.config['max_k']
        self.return_distances = self.config['return_distances']
        self.return_metadata = self.config['return_metadata']
        self.num_threads = self.config['num_threads']
        self.allow_replace_deleted = self.config['allow_replace_deleted']
        self.auto_save_interval = self.config['auto_save_interval']
        self.compression_enabled = self.config['compression_enabled']
        self.memory_mapping = self.config['memory_mapping']
        self.duplicate_threshold = self.config['duplicate_threshold']
        self.cleanup_interval = self.config['cleanup_interval']
        self.max_age_frames = self.config['max_age_frames']

        # Initialize index
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, M=self.M,
                              ef_construction=self.ef_construction, random_seed=self.random_seed)
        self.index.set_ef(self.ef_search)

        # Set number of threads if specified
        if self.num_threads > 0:
            self.index.set_num_threads(self.num_threads)

        # Metadata storage
        self.id_to_metadata = {}  # internal_id -> metadata
        self.external_to_internal = {}  # external_id -> internal_id
        self.internal_to_external = {}  # internal_id -> external_id
        self.next_internal_id = 0

        # Statistics
        self.elements_added = 0
        self.last_cleanup_frame = 0

        print(f"HNSW Index initialized:")
        print(f"  - Dimension: {self.dim}")
        print(f"  - Max elements: {self.max_elements}")
        print(f"  - M: {self.M}")
        print(f"  - ef_construction: {self.ef_construction}")
        print(f"  - ef_search: {self.ef_search}")
        print(f"  - Space: {self.space}")
        print(f"  - Threads: {self.num_threads}")

    def add_vector(self, vector: np.ndarray,
                   external_id: Any,
                   metadata: Optional[Dict] = None) -> int:
        """
        Add a vector to the index

        Args:
            vector: Feature vector to add
            external_id: External identifier for the vector
            metadata: Optional metadata to store with the vector

        Returns:
            Internal ID assigned to the vector
        """
        # Ensure vector is the right shape
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        elif len(vector.shape) == 2 and vector.shape[0] == 1:
            pass  # Already correct shape
        else:
            raise ValueError(
                f"Vector must be 1D or 2D with shape (1, dim), got {vector.shape}")

        # Check dimension
        if vector.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[1]} doesn't match index dimension {self.dim}")

        # Get internal ID
        internal_id = self.next_internal_id
        self.next_internal_id += 1

        # Add to index
        self.index.add_items(vector, [internal_id])

        # Store mappings and metadata
        self.external_to_internal[external_id] = internal_id
        self.internal_to_external[internal_id] = external_id
        self.id_to_metadata[internal_id] = metadata or {}

        return internal_id

    def add_vectors(self, vectors: np.ndarray,
                    external_ids: List[Any],
                    metadata_list: Optional[List[Dict]] = None) -> List[int]:
        """
        Add multiple vectors to the index

        Args:
            vectors: Feature vectors to add (N x dim)
            external_ids: List of external identifiers
            metadata_list: Optional list of metadata dictionaries

        Returns:
            List of internal IDs assigned to the vectors
        """
        if len(vectors.shape) != 2:
            raise ValueError(
                f"Vectors must be 2D array, got shape {vectors.shape}")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dim}")

        if len(external_ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of external_ids ({len(external_ids)}) doesn't match number of vectors ({vectors.shape[0]})")

        if metadata_list is None:
            metadata_list = [{}] * len(external_ids)
        elif len(metadata_list) != len(external_ids):
            raise ValueError(
                f"Number of metadata entries ({len(metadata_list)}) doesn't match number of external_ids ({len(external_ids)})")

        # Get internal IDs
        internal_ids = list(range(self.next_internal_id,
                            self.next_internal_id + len(external_ids)))
        self.next_internal_id += len(external_ids)

        # Add to index
        self.index.add_items(vectors, internal_ids)

        # Store mappings and metadata
        for i, (external_id, internal_id, metadata) in enumerate(zip(external_ids, internal_ids, metadata_list)):
            self.external_to_internal[external_id] = internal_id
            self.internal_to_external[internal_id] = external_id
            self.id_to_metadata[internal_id] = metadata

        return internal_ids

    def search(self, query_vector: np.ndarray, k: int = None) -> Dict:
        """
        Search for k nearest neighbors

        Args:
            query_vector: Query vector
            k: Number of neighbors to return (uses config default if None)

        Returns:
            Dictionary with 'internal_ids', 'distances', and 'external_ids'
        """
        if self.index.get_current_count() == 0:
            # Return empty results if index is empty
            return {
                'internal_ids': [],
                'distances': [],
                'external_ids': []
            }

        # Use default k if not provided
        if k is None:
            k = self.default_k

        # Ensure k is within valid range
        k = min(k, self.max_k, self.index.get_current_count())

        # Normalize query vector
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        if k == 0:
            return {
                'internal_ids': [],
                'distances': [],
                'external_ids': []
            }

        try:
            # Search for nearest neighbors
            internal_ids, distances = self.index.knn_query(query_vector, k=k)

            # Convert to lists if they're arrays
            if isinstance(internal_ids, np.ndarray):
                internal_ids = internal_ids.tolist()
            if isinstance(distances, np.ndarray):
                distances = distances.tolist()

            # Get external IDs
            external_ids = []
            for internal_id in internal_ids:
                external_id = self.internal_to_external.get(internal_id)
                external_ids.append(external_id)

            return {
                'internal_ids': internal_ids,
                'distances': distances,
                'external_ids': external_ids
            }

        except Exception as e:
            # Return empty results on error
            return {
                'internal_ids': [],
                'distances': [],
                'external_ids': []
            }

    def search_batch(self, query_vectors: np.ndarray,
                     k: int = 10,
                     return_metadata: bool = True,
                     return_distances: bool = True) -> List[Dict]:
        """
        Search for similar vectors for multiple queries

        Args:
            query_vectors: Query vectors (N x dim)
            k: Number of nearest neighbors to return for each query
            return_metadata: Whether to return metadata
            return_distances: Whether to return distances

        Returns:
            List of dictionaries containing search results for each query
        """
        if len(query_vectors.shape) != 2:
            raise ValueError(
                f"Query vectors must be 2D array, got shape {query_vectors.shape}")

        if query_vectors.shape[1] != self.dim:
            raise ValueError(
                f"Query vector dimension {query_vectors.shape[1]} doesn't match index dimension {self.dim}")

        # Search
        internal_ids, distances = self.index.knn_query(query_vectors, k=k)

        # Process results for each query
        results = []
        for i in range(len(query_vectors)):
            query_results = {
                'external_ids': [],
                'internal_ids': internal_ids[i].tolist(),
            }

            if return_distances:
                query_results['distances'] = distances[i].tolist()

            if return_metadata:
                query_results['metadata'] = []

            for internal_id in internal_ids[i]:
                external_id = self.internal_to_external.get(internal_id)
                query_results['external_ids'].append(external_id)

                if return_metadata:
                    metadata = self.id_to_metadata.get(internal_id, {})
                    query_results['metadata'].append(metadata)

            results.append(query_results)

        return results

    def update_metadata(self, external_id: Any, metadata: Dict):
        """
        Update metadata for an existing vector

        Args:
            external_id: External identifier
            metadata: New metadata dictionary
        """
        if external_id not in self.external_to_internal:
            raise ValueError(f"External ID {external_id} not found in index")

        internal_id = self.external_to_internal[external_id]
        self.id_to_metadata[internal_id] = metadata

    def get_metadata(self, external_id: Any) -> Dict:
        """
        Get metadata for a vector

        Args:
            external_id: External identifier

        Returns:
            Metadata dictionary
        """
        if external_id not in self.external_to_internal:
            raise ValueError(f"External ID {external_id} not found in index")

        internal_id = self.external_to_internal[external_id]
        return self.id_to_metadata.get(internal_id, {})

    def remove_vector(self, external_id: Any):
        """
        Remove a vector from the index

        Args:
            external_id: External identifier of vector to remove
        """
        if external_id not in self.external_to_internal:
            raise ValueError(f"External ID {external_id} not found in index")

        internal_id = self.external_to_internal[external_id]

        # Remove from index (HNSW doesn't support removal, so we mark as deleted)
        self.index.mark_deleted(internal_id)

        # Remove from mappings
        del self.external_to_internal[external_id]
        del self.internal_to_external[internal_id]
        del self.id_to_metadata[internal_id]

    def get_stats(self) -> Dict:
        """
        Get index statistics

        Returns:
            Dictionary containing index statistics
        """
        return {
            'num_elements': self.index.get_current_count(),
            'max_elements': self.max_elements,
            'dimension': self.dim,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'space': self.space,
            'memory_usage_bytes': self.index.get_memory_usage()
        }

    def set_ef(self, ef: int):
        """
        Set the ef parameter for search

        Args:
            ef: New ef value (higher = better recall, slower search)
        """
        self.ef_search = ef
        self.index.set_ef(ef)

    def save_index(self, filepath: str):
        """
        Save index to file

        Args:
            filepath: Path to save the index
        """
        # Save HNSW index
        index_path = f"{filepath}.hnsw"
        self.index.save_index(index_path)

        # Save metadata
        metadata_path = f"{filepath}.metadata"
        metadata = {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'space': self.space,
            'random_seed': self.random_seed,
            'next_internal_id': self.next_internal_id,
            'id_to_metadata': self.id_to_metadata,
            'external_to_internal': self.external_to_internal,
            'internal_to_external': self.internal_to_external
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """
        Load index from file

        Args:
            filepath: Path to load the index from
        """
        # Load metadata
        metadata_path = f"{filepath}.metadata"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Restore attributes
        self.dim = metadata['dim']
        self.max_elements = metadata['max_elements']
        self.M = metadata['M']
        self.ef_construction = metadata['ef_construction']
        self.ef_search = metadata['ef_search']
        self.space = metadata['space']
        self.random_seed = metadata['random_seed']
        self.next_internal_id = metadata['next_internal_id']
        self.id_to_metadata = metadata['id_to_metadata']
        self.external_to_internal = metadata['external_to_internal']
        self.internal_to_external = metadata['internal_to_external']

        # Load HNSW index
        index_path = f"{filepath}.hnsw"
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(index_path)
        self.index.set_ef(self.ef_search)

        print(f"Index loaded from {filepath}")

    def benchmark_search(self, query_vectors: np.ndarray,
                         k: int = 10,
                         num_runs: int = 10) -> Dict:
        """
        Benchmark search performance

        Args:
            query_vectors: Query vectors for benchmarking
            k: Number of nearest neighbors to return
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            self.search_batch(query_vectors, k=k,
                              return_metadata=False, return_distances=False)
            end_time = time.time()
            times.append(end_time - start_time)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'queries_per_second': len(query_vectors) / np.mean(times)
        }


def main():
    """Example usage"""
    # Create index
    index = HNSWIndex(dim=128, max_elements=1000, M=16,
                      ef_construction=200, ef_search=50)

    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.randn(100, 128).astype(np.float32)

    # Normalize vectors (for cosine similarity)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Add vectors to index
    external_ids = [f"person_{i}" for i in range(100)]
    metadata_list = [{'track_id': i, 'timestamp': i * 0.1} for i in range(100)]

    print("Adding vectors to index...")
    start_time = time.time()
    internal_ids = index.add_vectors(vectors, external_ids, metadata_list)
    end_time = time.time()
    print(f"Added {len(vectors)} vectors in {end_time - start_time:.4f} seconds")

    # Search for similar vectors
    query_vector = np.random.randn(1, 128).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    print("\nSearching for similar vectors...")
    start_time = time.time()
    results = index.search(query_vector, k=5)
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds")

    print(f"Found {len(results['external_ids'])} similar vectors:")
    for i, (ext_id, dist, metadata) in enumerate(zip(results['external_ids'],
                                                     results['distances'],
                                                     results['metadata'])):
        print(f"  {i+1}. {ext_id}: distance={dist:.4f}, metadata={metadata}")

    # Batch search
    query_vectors = np.random.randn(10, 128).astype(np.float32)
    query_vectors = query_vectors / \
        np.linalg.norm(query_vectors, axis=1, keepdims=True)

    print(f"\nBatch search with {len(query_vectors)} queries...")
    start_time = time.time()
    batch_results = index.search_batch(
        query_vectors, k=3, return_metadata=False)
    end_time = time.time()
    print(f"Batch search completed in {end_time - start_time:.4f} seconds")

    print(f"Results for first query:")
    for i, (ext_id, dist) in enumerate(zip(batch_results[0]['external_ids'],
                                           batch_results[0]['distances'])):
        print(f"  {i+1}. {ext_id}: distance={dist:.4f}")

    # Show index statistics
    stats = index.get_stats()
    print(f"\nIndex statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save and load index
    print("\nSaving index...")
    index.save_index("test_index")

    print("Loading index...")
    new_index = HNSWIndex(dim=128)
    new_index.load_index("test_index")

    # Verify loaded index works
    loaded_results = new_index.search(query_vector, k=5)
    print(
        f"Loaded index search results: {len(loaded_results['external_ids'])} vectors found")


if __name__ == "__main__":
    main()
