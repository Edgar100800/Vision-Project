#!/usr/bin/env python3
"""
Gaussian Cluster Manager
Sistema de gestión dinámica de clusters con modelado gaussiano para re-identificación de personas.

Implementa:
- Clusters como distribuciones gaussianas multivariadas N(μ_k, Σ_k)
- Actualización incremental (EM online)
- Fusión y división de clusters
- Umbral dinámico
- Integración con HNSW
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.stats import chi2
from scipy.linalg import inv, det
import uuid
from pathlib import Path


@dataclass
class GaussianCluster:
    """Representa un cluster como distribución gaussiana multivariada"""
    cluster_id: str
    mu: np.ndarray  # Media (2048D)
    sigma: np.ndarray  # Matriz de covarianza (2048x2048)
    n_samples: int  # Número de muestras
    creation_time: float
    last_update_time: float

    def __post_init__(self):
        """Validar dimensiones después de inicialización"""
        if self.mu.shape[0] != self.sigma.shape[0] or self.sigma.shape[0] != self.sigma.shape[1]:
            raise ValueError(
                f"Dimensiones inconsistentes: mu={self.mu.shape}, sigma={self.sigma.shape}")


class GaussianClusterManager:
    """
    Gestor de clusters gaussianos para re-identificación de personas
    """

    def __init__(self,
                 embedding_dim: int = 512,
                 initial_sigma_scale: float = 1.0,
                 mahalanobis_threshold: float = 3.0,
                 min_samples_for_update: int = 2,
                 max_clusters: int = 1000,
                 fusion_threshold: float = 2.0,
                 split_threshold: float = 10.0):
        """
        Inicializar el gestor de clusters gaussianos

        Args:
            embedding_dim: Dimensión de los embeddings (512 para OSNet)
            initial_sigma_scale: Escala inicial para matriz de covarianza
            mahalanobis_threshold: Umbral de distancia de Mahalanobis
            min_samples_for_update: Mínimo de muestras para actualizar covarianza
            max_clusters: Máximo número de clusters
            fusion_threshold: Umbral para fusión de clusters
            split_threshold: Umbral para división de clusters
        """
        self.embedding_dim = embedding_dim
        self.initial_sigma_scale = initial_sigma_scale
        self.mahalanobis_threshold = mahalanobis_threshold
        self.min_samples_for_update = min_samples_for_update
        self.max_clusters = max_clusters
        self.fusion_threshold = fusion_threshold
        self.split_threshold = split_threshold

        # Almacenamiento de clusters
        self.clusters: Dict[str, GaussianCluster] = {}

        # Estadísticas globales
        self.total_assignments = 0
        self.total_new_clusters = 0
        self.total_fusions = 0
        self.total_splits = 0

        # Configurar logging
        self.logger = logging.getLogger(__name__)

        # Matriz de covarianza inicial (identidad escalada)
        self.initial_sigma = self.initial_sigma_scale * np.eye(embedding_dim)

    def _generate_cluster_id(self) -> str:
        """Generar ID único para cluster"""
        return f"cluster_{uuid.uuid4().hex[:8]}"

    def _mahalanobis_distance(self, x: np.ndarray, cluster: GaussianCluster) -> float:
        """
        Calcular distancia de Mahalanobis entre vector x y cluster

        d_k(x) = sqrt((x - μ_k)^T * Σ_k^(-1) * (x - μ_k))
        """
        try:
            diff = x - cluster.mu
            sigma_inv = inv(cluster.sigma)
            distance = np.sqrt(diff.T @ sigma_inv @ diff)
            return float(distance)
        except np.linalg.LinAlgError:
            # Si la matriz no es invertible, usar regularización
            self.logger.warning(
                f"Matriz singular en cluster {cluster.cluster_id}, aplicando regularización")
            regularized_sigma = cluster.sigma + \
                1e-6 * np.eye(self.embedding_dim)
            diff = x - cluster.mu
            sigma_inv = inv(regularized_sigma)
            distance = np.sqrt(diff.T @ sigma_inv @ diff)
            return float(distance)

    def _update_cluster_incremental(self, cluster: GaussianCluster, x: np.ndarray) -> None:
        """
        Actualización incremental de estadísticas del cluster (EM online)

        N_k ← N_k + 1
        μ_k ← μ_k + (x - μ_k) / N_k
        Σ_k ← ((N_k - 1) / N_k) * Σ_k + (1 / N_k) * (x - μ_k) * (x - μ_k)^T
        """
        import time

        # Actualizar número de muestras
        cluster.n_samples += 1
        n = cluster.n_samples

        # Calcular diferencia antes de actualizar media
        diff_old = x - cluster.mu

        # Actualizar media
        cluster.mu = cluster.mu + diff_old / n

        # Actualizar covarianza solo si tenemos suficientes muestras
        if n >= self.min_samples_for_update:
            diff_new = x - cluster.mu
            # Fórmula incremental para covarianza
            cluster.sigma = ((n - 1) / n) * cluster.sigma + \
                (1 / n) * np.outer(diff_old, diff_old)

            # Regularización para evitar matrices singulares
            cluster.sigma += 1e-8 * np.eye(self.embedding_dim)

        # Actualizar timestamp
        cluster.last_update_time = time.time()

        self.logger.debug(
            f"Cluster {cluster.cluster_id} actualizado: n_samples={n}")

    def _create_new_cluster(self, x: np.ndarray) -> GaussianCluster:
        """
        Crear nuevo cluster con μ = x, Σ = σ_0 * I, N = 1
        """
        import time

        cluster_id = self._generate_cluster_id()
        current_time = time.time()

        cluster = GaussianCluster(
            cluster_id=cluster_id,
            mu=x.copy(),
            sigma=self.initial_sigma.copy(),
            n_samples=1,
            creation_time=current_time,
            last_update_time=current_time
        )

        self.clusters[cluster_id] = cluster
        self.total_new_clusters += 1

        self.logger.info(f"Nuevo cluster creado: {cluster_id}")
        return cluster

    def assign_to_cluster(self, x: np.ndarray, candidate_clusters: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Asignar vector x al cluster más apropiado o crear uno nuevo

        Args:
            x: Vector de embedding (embedding_dim,)
            candidate_clusters: Lista de IDs de clusters candidatos (de HNSW)

        Returns:
            Tuple[cluster_id, distance]: ID del cluster asignado y distancia
        """
        if len(x.shape) != 1 or x.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Vector debe ser 1D con dimensión {self.embedding_dim}")

        # Si no hay clusters, crear el primero
        if not self.clusters:
            cluster = self._create_new_cluster(x)
            return cluster.cluster_id, 0.0

        # Determinar clusters a evaluar
        if candidate_clusters:
            # Filtrar candidatos válidos
            clusters_to_check = [
                cid for cid in candidate_clusters if cid in self.clusters]
        else:
            # Evaluar todos los clusters
            clusters_to_check = list(self.clusters.keys())

        if not clusters_to_check:
            # No hay clusters válidos, crear nuevo
            cluster = self._create_new_cluster(x)
            return cluster.cluster_id, 0.0

        # Calcular distancias de Mahalanobis
        best_cluster_id = None
        best_distance = float('inf')

        for cluster_id in clusters_to_check:
            cluster = self.clusters[cluster_id]
            distance = self._mahalanobis_distance(x, cluster)

            if distance < best_distance:
                best_distance = distance
                best_cluster_id = cluster_id

        # Decidir si asignar al mejor cluster o crear uno nuevo
        if best_distance < self.mahalanobis_threshold:
            # Asignar al cluster existente
            self._update_cluster_incremental(self.clusters[best_cluster_id], x)
            self.total_assignments += 1
            self.logger.debug(
                f"Vector asignado a cluster {best_cluster_id} con distancia {best_distance:.4f}")
            return best_cluster_id, best_distance
        else:
            # Crear nuevo cluster
            cluster = self._create_new_cluster(x)
            self.logger.info(
                f"Nuevo cluster {cluster.cluster_id} creado (distancia mínima: {best_distance:.4f})")
            return cluster.cluster_id, 0.0

    def _hotelling_t2_test(self, cluster1: GaussianCluster, cluster2: GaussianCluster, alpha: float = 0.05) -> bool:
        """
        Test de Hotelling T² para determinar si dos clusters deben fusionarse

        Returns:
            True si los clusters son estadísticamente similares (deben fusionarse)
        """
        if cluster1.n_samples < 2 or cluster2.n_samples < 2:
            return False

        try:
            n1, n2 = cluster1.n_samples, cluster2.n_samples
            mu1, mu2 = cluster1.mu, cluster2.mu
            sigma1, sigma2 = cluster1.sigma, cluster2.sigma

            # Covarianza pooled
            sigma_pooled = ((n1 - 1) * sigma1 + (n2 - 1)
                            * sigma2) / (n1 + n2 - 2)

            # Estadístico T²
            diff = mu1 - mu2
            factor = (n1 * n2) / (n1 + n2)
            t2 = factor * diff.T @ inv(sigma_pooled) @ diff

            # Convertir a F-statistic
            f_stat = ((n1 + n2 - self.embedding_dim - 1) /
                      ((n1 + n2 - 2) * self.embedding_dim)) * t2

            # Grados de libertad
            df1 = self.embedding_dim
            df2 = n1 + n2 - self.embedding_dim - 1

            # Valor crítico (aproximación usando chi2)
            critical_value = chi2.ppf(1 - alpha, df1)

            return f_stat < critical_value

        except (np.linalg.LinAlgError, ValueError) as e:
            self.logger.warning(f"Error en test Hotelling T²: {e}")
            return False

    def _fuse_clusters(self, cluster1_id: str, cluster2_id: str) -> str:
        """
        Fusionar dos clusters combinando sus estadísticas
        """
        cluster1 = self.clusters[cluster1_id]
        cluster2 = self.clusters[cluster2_id]

        n1, n2 = cluster1.n_samples, cluster2.n_samples
        n_total = n1 + n2

        # Media ponderada
        mu_fused = (n1 * cluster1.mu + n2 * cluster2.mu) / n_total

        # Covarianza ponderada
        sigma_fused = ((n1 - 1) * cluster1.sigma + (n2 - 1)
                       * cluster2.sigma) / (n_total - 1)

        # Crear cluster fusionado
        import time
        fused_cluster = GaussianCluster(
            cluster_id=self._generate_cluster_id(),
            mu=mu_fused,
            sigma=sigma_fused,
            n_samples=n_total,
            creation_time=min(cluster1.creation_time, cluster2.creation_time),
            last_update_time=time.time()
        )

        # Remover clusters originales y agregar fusionado
        del self.clusters[cluster1_id]
        del self.clusters[cluster2_id]
        self.clusters[fused_cluster.cluster_id] = fused_cluster

        self.total_fusions += 1
        self.logger.info(
            f"Clusters {cluster1_id} y {cluster2_id} fusionados en {fused_cluster.cluster_id}")

        return fused_cluster.cluster_id

    def check_and_fuse_clusters(self) -> List[Tuple[str, str, str]]:
        """
        Evaluar y fusionar clusters similares

        Returns:
            Lista de fusiones realizadas (cluster1_id, cluster2_id, fused_id)
        """
        if len(self.clusters) < 2:
            return []

        fusions = []
        cluster_ids = list(self.clusters.keys())

        # Evaluar pares de clusters
        i = 0
        while i < len(cluster_ids):
            j = i + 1
            while j < len(cluster_ids):
                cluster1_id = cluster_ids[i]
                cluster2_id = cluster_ids[j]

                # Verificar que ambos clusters aún existen
                if cluster1_id in self.clusters and cluster2_id in self.clusters:
                    cluster1 = self.clusters[cluster1_id]
                    cluster2 = self.clusters[cluster2_id]

                    # Test de fusión
                    if self._hotelling_t2_test(cluster1, cluster2):
                        fused_id = self._fuse_clusters(
                            cluster1_id, cluster2_id)
                        fusions.append((cluster1_id, cluster2_id, fused_id))

                        # Actualizar lista de IDs
                        cluster_ids = [cid for cid in cluster_ids if cid not in [
                            cluster1_id, cluster2_id]]
                        cluster_ids.append(fused_id)

                        # Reiniciar búsqueda
                        i = 0
                        break
                j += 1
            else:
                i += 1

        return fusions

    def get_cluster_statistics(self) -> Dict:
        """Obtener estadísticas del gestor de clusters"""
        if not self.clusters:
            return {
                'n_clusters': 0,
                'total_assignments': self.total_assignments,
                'total_new_clusters': self.total_new_clusters,
                'total_fusions': self.total_fusions,
                'total_splits': self.total_splits
            }

        cluster_sizes = [
            cluster.n_samples for cluster in self.clusters.values()]

        return {
            'n_clusters': len(self.clusters),
            'total_assignments': self.total_assignments,
            'total_new_clusters': self.total_new_clusters,
            'total_fusions': self.total_fusions,
            'total_splits': self.total_splits,
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'std_cluster_size': np.std(cluster_sizes)
        }

    def save_clusters(self, filepath: Union[str, Path]) -> None:
        """Guardar clusters en archivo"""
        import pickle

        data = {
            'clusters': self.clusters,
            'config': {
                'embedding_dim': self.embedding_dim,
                'initial_sigma_scale': self.initial_sigma_scale,
                'mahalanobis_threshold': self.mahalanobis_threshold,
                'min_samples_for_update': self.min_samples_for_update,
                'max_clusters': self.max_clusters,
                'fusion_threshold': self.fusion_threshold,
                'split_threshold': self.split_threshold
            },
            'statistics': {
                'total_assignments': self.total_assignments,
                'total_new_clusters': self.total_new_clusters,
                'total_fusions': self.total_fusions,
                'total_splits': self.total_splits
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Clusters guardados en {filepath}")

    def load_clusters(self, filepath: Union[str, Path]) -> None:
        """Cargar clusters desde archivo"""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.clusters = data['clusters']

        # Cargar estadísticas
        stats = data['statistics']
        self.total_assignments = stats['total_assignments']
        self.total_new_clusters = stats['total_new_clusters']
        self.total_fusions = stats['total_fusions']
        self.total_splits = stats['total_splits']

        self.logger.info(
            f"Clusters cargados desde {filepath}: {len(self.clusters)} clusters")
