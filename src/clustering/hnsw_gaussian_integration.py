#!/usr/bin/env python3
"""
HNSW-Gaussian Integration
Integración entre índice HNSW y gestor de clusters gaussianos para re-identificación de personas.

Flujo:
1. Nuevo embedding → consulta HNSW → candidatos más cercanos
2. Calcular distancia de Mahalanobis vs clusters candidatos
3. Si cumple umbral → asignar al cluster → ID
4. Si no → crear nuevo cluster e insertar en HNSW
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time

from .gaussian_cluster_manager import GaussianClusterManager, GaussianCluster
from ..index.hnsw_index import HNSWIndex


class HNSWGaussianIntegration:
    """
    Integración entre HNSW y gestión de clusters gaussianos
    """

    def __init__(self,
                 embedding_dim: int = 512,
                 hnsw_config: Optional[Dict] = None,
                 gaussian_config: Optional[Dict] = None,
                 top_k_candidates: int = 5,
                 auto_fusion_interval: int = 100,
                 auto_save_interval: int = 1000):
        """
        Inicializar la integración HNSW-Gaussiano

        Args:
            embedding_dim: Dimensión de embeddings
            hnsw_config: Configuración para HNSW
            gaussian_config: Configuración para clusters gaussianos
            top_k_candidates: Número de candidatos a obtener de HNSW
            auto_fusion_interval: Intervalo para fusión automática de clusters
            auto_save_interval: Intervalo para guardado automático
        """
        self.embedding_dim = embedding_dim
        self.top_k_candidates = top_k_candidates
        self.auto_fusion_interval = auto_fusion_interval
        self.auto_save_interval = auto_save_interval

        # Configuraciones por defecto
        default_hnsw_config = {
            'M': 16,
            'ef_construction': 200,
            'ef_search': 100,
            'max_elements': 10000,
            'space': 'cosine'
        }

        default_gaussian_config = {
            'initial_sigma_scale': 1.0,
            'mahalanobis_threshold': 3.0,
            'min_samples_for_update': 2,
            'max_clusters': 1000,
            'fusion_threshold': 2.0,
            'split_threshold': 10.0
        }

        # Aplicar configuraciones
        self.hnsw_config = {**default_hnsw_config, **(hnsw_config or {})}
        self.gaussian_config = {
            **default_gaussian_config, **(gaussian_config or {})}

        # Inicializar componentes
        self.hnsw_index = HNSWIndex(
            dim=embedding_dim,
            max_elements=self.hnsw_config['max_elements'],
            M=self.hnsw_config['M'],
            ef_construction=self.hnsw_config['ef_construction'],
            space=self.hnsw_config['space']
        )

        self.cluster_manager = GaussianClusterManager(
            embedding_dim=embedding_dim,
            **self.gaussian_config
        )

        # Mapeo entre IDs externos de HNSW y cluster IDs
        self.hnsw_to_cluster_map: Dict[str, str] = {}
        self.cluster_to_hnsw_map: Dict[str, List[str]] = {}

        # Contadores para operaciones automáticas
        self.operation_count = 0
        self.last_fusion_check = 0
        self.last_save = 0

        # Configurar logging
        self.logger = logging.getLogger(__name__)

    def _generate_hnsw_id(self, cluster_id: str, sample_idx: int) -> str:
        """Generar ID único para HNSW basado en cluster ID y índice de muestra"""
        return f"{cluster_id}_sample_{sample_idx:04d}"

    def _add_to_hnsw(self, embedding: np.ndarray, cluster_id: str) -> str:
        """
        Agregar embedding al índice HNSW

        Returns:
            hnsw_id: ID asignado en HNSW
        """
        # Generar ID único para HNSW
        cluster = self.cluster_manager.clusters[cluster_id]
        hnsw_id = self._generate_hnsw_id(cluster_id, cluster.n_samples)

        # Agregar al índice HNSW
        self.hnsw_index.add_vector(embedding.reshape(1, -1), hnsw_id)

        # Actualizar mapeos
        self.hnsw_to_cluster_map[hnsw_id] = cluster_id
        if cluster_id not in self.cluster_to_hnsw_map:
            self.cluster_to_hnsw_map[cluster_id] = []
        self.cluster_to_hnsw_map[cluster_id].append(hnsw_id)

        self.logger.debug(
            f"Embedding agregado a HNSW: {hnsw_id} -> {cluster_id}")
        return hnsw_id

    def _get_candidate_clusters(self, embedding: np.ndarray) -> List[str]:
        """
        Obtener clusters candidatos usando HNSW

        Returns:
            Lista de cluster IDs candidatos
        """
        if self.hnsw_index.index.get_current_count() == 0:
            return []

        # Configurar ef_search
        self.hnsw_index.set_ef(self.hnsw_config['ef_search'])

        # Buscar vecinos más cercanos
        search_results = self.hnsw_index.search(
            embedding.reshape(1, -1),
            k=min(self.top_k_candidates, self.hnsw_index.index.get_current_count())
        )

        # Extraer cluster IDs únicos
        candidate_clusters = []
        seen_clusters = set()

        if search_results['external_ids']:
            for hnsw_id in search_results['external_ids']:
                if hnsw_id in self.hnsw_to_cluster_map:
                    cluster_id = self.hnsw_to_cluster_map[hnsw_id]
                    if cluster_id not in seen_clusters:
                        candidate_clusters.append(cluster_id)
                        seen_clusters.add(cluster_id)

        self.logger.debug(
            f"Clusters candidatos encontrados: {candidate_clusters}")
        return candidate_clusters

    def assign_identity(self, embedding: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Asignar identidad a un embedding usando el pipeline completo

        Args:
            embedding: Vector de características (embedding_dim,)

        Returns:
            Tuple[cluster_id, distance, metadata]: 
                - cluster_id: ID del cluster asignado
                - distance: Distancia de Mahalanobis
                - metadata: Información adicional sobre la asignación
        """
        if len(embedding.shape) != 1 or embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding debe ser 1D con dimensión {self.embedding_dim}")

        start_time = time.time()

        # Paso 1: Obtener candidatos de HNSW
        candidate_clusters = self._get_candidate_clusters(embedding)

        # Paso 2: Asignar usando clusters gaussianos
        cluster_id, mahalanobis_distance = self.cluster_manager.assign_to_cluster(
            embedding, candidate_clusters
        )

        # Paso 3: Si es nuevo cluster o cluster existente, agregar a HNSW
        cluster = self.cluster_manager.clusters[cluster_id]

        # Determinar si es nuevo cluster
        is_new_cluster = cluster.n_samples == 1

        if is_new_cluster or cluster_id not in self.cluster_to_hnsw_map:
            # Agregar primer embedding del cluster a HNSW
            hnsw_id = self._add_to_hnsw(embedding, cluster_id)
        else:
            # Cluster existente, agregar muestra adicional ocasionalmente
            # (para mantener representación actualizada en HNSW)
            if cluster.n_samples % 5 == 0:  # Cada 5 muestras
                hnsw_id = self._add_to_hnsw(embedding, cluster_id)
            else:
                hnsw_id = None

        # Incrementar contador de operaciones
        self.operation_count += 1

        # Operaciones automáticas periódicas
        self._check_auto_operations()

        # Preparar metadata
        processing_time = time.time() - start_time
        metadata = {
            'is_new_cluster': is_new_cluster,
            'candidate_clusters': candidate_clusters,
            'n_candidates_checked': len(candidate_clusters),
            'cluster_size': cluster.n_samples,
            'hnsw_id': hnsw_id,
            'processing_time_ms': processing_time * 1000,
            'total_clusters': len(self.cluster_manager.clusters),
            'hnsw_size': self.hnsw_index.index.get_current_count()
        }

        self.logger.info(f"Identidad asignada: {cluster_id} (distancia: {mahalanobis_distance:.4f}, "
                         f"nuevo: {is_new_cluster}, tiempo: {processing_time*1000:.2f}ms)")

        return cluster_id, mahalanobis_distance, metadata

    def _check_auto_operations(self):
        """Verificar y ejecutar operaciones automáticas periódicas"""

        # Fusión automática de clusters
        if (self.operation_count - self.last_fusion_check) >= self.auto_fusion_interval:
            self.logger.info("Ejecutando fusión automática de clusters...")
            fusions = self.cluster_manager.check_and_fuse_clusters()

            if fusions:
                # Actualizar mapeos después de fusiones
                self._update_mappings_after_fusions(fusions)
                self.logger.info(f"Fusiones realizadas: {len(fusions)}")

            self.last_fusion_check = self.operation_count

        # Guardado automático
        if (self.operation_count - self.last_save) >= self.auto_save_interval:
            self.logger.info("Ejecutando guardado automático...")
            # Nota: implementar guardado automático si se desea
            self.last_save = self.operation_count

    def _update_mappings_after_fusions(self, fusions: List[Tuple[str, str, str]]):
        """Actualizar mapeos después de fusiones de clusters"""
        for cluster1_id, cluster2_id, fused_id in fusions:
            # Combinar mapeos de HNSW
            hnsw_ids_1 = self.cluster_to_hnsw_map.get(cluster1_id, [])
            hnsw_ids_2 = self.cluster_to_hnsw_map.get(cluster2_id, [])

            # Actualizar mapeo cluster -> HNSW
            self.cluster_to_hnsw_map[fused_id] = hnsw_ids_1 + hnsw_ids_2

            # Actualizar mapeo HNSW -> cluster
            for hnsw_id in hnsw_ids_1 + hnsw_ids_2:
                self.hnsw_to_cluster_map[hnsw_id] = fused_id

            # Limpiar mapeos antiguos
            if cluster1_id in self.cluster_to_hnsw_map:
                del self.cluster_to_hnsw_map[cluster1_id]
            if cluster2_id in self.cluster_to_hnsw_map:
                del self.cluster_to_hnsw_map[cluster2_id]

    def get_cluster_by_id(self, cluster_id: str) -> Optional[GaussianCluster]:
        """Obtener cluster por ID"""
        return self.cluster_manager.clusters.get(cluster_id)

    def search_similar_identities(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Buscar identidades similares sin asignar

        Returns:
            Lista de (cluster_id, distance) ordenada por similitud
        """
        candidate_clusters = self._get_candidate_clusters(embedding)

        if not candidate_clusters:
            return []

        # Calcular distancias de Mahalanobis
        results = []
        for cluster_id in candidate_clusters:
            if cluster_id in self.cluster_manager.clusters:
                cluster = self.cluster_manager.clusters[cluster_id]
                distance = self.cluster_manager._mahalanobis_distance(
                    embedding, cluster)
                results.append((cluster_id, distance))

        # Ordenar por distancia
        results.sort(key=lambda x: x[1])

        return results[:k]

    def get_system_statistics(self) -> Dict:
        """Obtener estadísticas completas del sistema"""
        cluster_stats = self.cluster_manager.get_cluster_statistics()

        return {
            **cluster_stats,
            'hnsw_size': self.hnsw_index.index.get_current_count(),
            'hnsw_max_elements': self.hnsw_config['max_elements'],
            'operation_count': self.operation_count,
            'mappings': {
                'hnsw_to_cluster_entries': len(self.hnsw_to_cluster_map),
                'cluster_to_hnsw_entries': len(self.cluster_to_hnsw_map)
            },
            'config': {
                'hnsw': self.hnsw_config,
                'gaussian': self.gaussian_config,
                'top_k_candidates': self.top_k_candidates,
                'auto_fusion_interval': self.auto_fusion_interval
            }
        }

    def save_system(self, base_path: Union[str, Path]) -> None:
        """Guardar sistema completo"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar clusters gaussianos
        cluster_path = base_path / "gaussian_clusters.pkl"
        self.cluster_manager.save_clusters(cluster_path)

        # Guardar índice HNSW
        hnsw_path = base_path / "hnsw_index.bin"
        self.hnsw_index.save_index(str(hnsw_path))

        # Guardar mapeos
        import pickle
        mappings_path = base_path / "mappings.pkl"
        mappings_data = {
            'hnsw_to_cluster_map': self.hnsw_to_cluster_map,
            'cluster_to_hnsw_map': self.cluster_to_hnsw_map,
            'operation_count': self.operation_count,
            'last_fusion_check': self.last_fusion_check,
            'last_save': self.last_save,
            'config': {
                'hnsw': self.hnsw_config,
                'gaussian': self.gaussian_config,
                'top_k_candidates': self.top_k_candidates,
                'auto_fusion_interval': self.auto_fusion_interval,
                'auto_save_interval': self.auto_save_interval
            }
        }

        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings_data, f)

        self.logger.info(f"Sistema guardado en {base_path}")

    def load_system(self, base_path: Union[str, Path]) -> None:
        """Cargar sistema completo"""
        base_path = Path(base_path)

        # Cargar clusters gaussianos
        cluster_path = base_path / "gaussian_clusters.pkl"
        if cluster_path.exists():
            self.cluster_manager.load_clusters(cluster_path)

        # Cargar índice HNSW
        hnsw_path = base_path / "hnsw_index.bin"
        if hnsw_path.exists():
            self.hnsw_index.load_index(str(hnsw_path))

        # Cargar mapeos
        import pickle
        mappings_path = base_path / "mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, 'rb') as f:
                mappings_data = pickle.load(f)

            self.hnsw_to_cluster_map = mappings_data['hnsw_to_cluster_map']
            self.cluster_to_hnsw_map = mappings_data['cluster_to_hnsw_map']
            self.operation_count = mappings_data['operation_count']
            self.last_fusion_check = mappings_data['last_fusion_check']
            self.last_save = mappings_data['last_save']

        self.logger.info(f"Sistema cargado desde {base_path}")

    def reset_system(self) -> None:
        """Reiniciar sistema completo"""
        # Reiniciar HNSW
        self.hnsw_index = HNSWIndex(
            dim=self.embedding_dim,
            max_elements=self.hnsw_config['max_elements'],
            M=self.hnsw_config['M'],
            ef_construction=self.hnsw_config['ef_construction'],
            space=self.hnsw_config['space']
        )

        # Reiniciar cluster manager
        self.cluster_manager = GaussianClusterManager(
            embedding_dim=self.embedding_dim,
            **self.gaussian_config
        )

        # Limpiar mapeos
        self.hnsw_to_cluster_map.clear()
        self.cluster_to_hnsw_map.clear()

        # Reiniciar contadores
        self.operation_count = 0
        self.last_fusion_check = 0
        self.last_save = 0

        self.logger.info("Sistema reiniciado")
