#!/usr/bin/env python3
"""
Campaign Manager para asignación retardada de IDs
Maneja el proceso de "campaña" antes de asignar IDs definitivos y luego usa Kalman para seguimiento
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
import logging


@dataclass
class PersonCandidate:
    """Representa un candidato a persona durante la campaña"""
    candidate_id: str
    first_detection_frame: int
    last_detection_frame: int
    detection_count: int
    confidence_scores: List[float] = field(default_factory=list)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(
        default_factory=list)  # (x1, y1, x2, y2)
    embeddings: List[np.ndarray] = field(default_factory=list)
    cluster_assignments: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)

    @property
    def campaign_duration(self) -> int:
        """Duración de la campaña en frames"""
        return self.last_detection_frame - self.first_detection_frame + 1

    @property
    def average_confidence(self) -> float:
        """Confianza promedio durante la campaña"""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0

    @property
    def most_common_cluster(self) -> str:
        """Cluster más común durante la campaña"""
        if not self.cluster_assignments:
            return ""
        return max(set(self.cluster_assignments), key=self.cluster_assignments.count)

    @property
    def cluster_consistency(self) -> float:
        """Consistencia del cluster asignado"""
        if not self.cluster_assignments:
            return 0.0
        most_common = self.most_common_cluster
        return self.cluster_assignments.count(most_common) / len(self.cluster_assignments)


@dataclass
class ConfirmedPerson:
    """Representa una persona con ID confirmado"""
    person_id: int
    cluster_id: str
    confirmation_frame: int
    kalman_filter: KalmanFilter
    last_detection: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    last_detection_frame: int
    consecutive_misses: int = 0
    total_detections: int = 0

    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """Predecir próxima posición usando Kalman"""
        self.kalman_filter.predict()
        state = self.kalman_filter.x

        # Extraer posición del centro y tamaño
        cx, cy, w, h = state[0], state[1], state[4], state[5]

        # Convertir a bounding box
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)

        return (x1, y1, x2, y2)

    def update_kalman(self, bbox: Tuple[int, int, int, int]):
        """Actualizar filtro Kalman con nueva detección"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Actualizar Kalman
        measurement = np.array([cx, cy, w, h])
        self.kalman_filter.update(measurement)

        # Actualizar estado
        self.last_detection = bbox
        self.last_detection_frame = self.last_detection_frame + 1
        self.consecutive_misses = 0
        self.total_detections += 1


class CampaignManager:
    """
    Gestor de campañas para asignación retardada de IDs
    """

    def __init__(self,
                 campaign_frames: int = 30,
                 campaign_min_samples: int = 8,
                 campaign_confidence_threshold: float = 0.7,
                 max_consecutive_misses: int = 10,
                 kalman_prediction_frames: int = 5):
        """
        Inicializar el gestor de campañas

        Args:
            campaign_frames: Frames mínimos para completar campaña
            campaign_min_samples: Mínimo de muestras para asignar ID
            campaign_confidence_threshold: Confianza mínima para asignar ID
            max_consecutive_misses: Máximo frames perdidos antes de eliminar persona
            kalman_prediction_frames: Frames máximos de predicción Kalman
        """
        self.campaign_frames = campaign_frames
        self.campaign_min_samples = campaign_min_samples
        self.campaign_confidence_threshold = campaign_confidence_threshold
        self.max_consecutive_misses = max_consecutive_misses
        self.kalman_prediction_frames = kalman_prediction_frames

        # Almacenamiento
        self.candidates: Dict[str, PersonCandidate] = {}
        self.confirmed_persons: Dict[int, ConfirmedPerson] = {}

        # Contadores
        self.next_person_id = 0
        self.current_frame = 0

        # Estadísticas
        self.total_campaigns = 0
        self.successful_campaigns = 0
        self.failed_campaigns = 0

        # Logging
        self.logger = logging.getLogger(__name__)

    def _create_kalman_filter(self, bbox: Tuple[int, int, int, int]) -> KalmanFilter:
        """Crear filtro Kalman para seguimiento"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Crear filtro Kalman con 6 estados: [cx, cy, vx, vy, w, h]
        kf = KalmanFilter(dim_x=6, dim_z=4)

        # Estado inicial
        kf.x = np.array([cx, cy, 0, 0, w, h])

        # Matriz de transición de estado
        kf.F = np.array([
            [1, 0, 1, 0, 0, 0],  # cx = cx + vx
            [0, 1, 0, 1, 0, 0],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 1, 0],  # w = w
            [0, 0, 0, 0, 0, 1]   # h = h
        ])

        # Matriz de observación
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # observamos cx
            [0, 1, 0, 0, 0, 0],  # observamos cy
            [0, 0, 0, 0, 1, 0],  # observamos w
            [0, 0, 0, 0, 0, 1]   # observamos h
        ])

        # Covarianza del proceso
        kf.Q = np.eye(6) * 0.1

        # Covarianza de la medición
        kf.R = np.eye(4) * 1.0

        # Covarianza inicial
        kf.P *= 10.0

        return kf

    def update_frame(self, frame_number: int):
        """Actualizar número de frame actual"""
        self.current_frame = frame_number

        # Actualizar frames perdidos para personas confirmadas
        for person_id, person in self.confirmed_persons.items():
            if person.last_detection_frame < frame_number - 1:
                person.consecutive_misses += 1

    def add_detection(self,
                      candidate_id: str,
                      bbox: Tuple[int, int, int, int],
                      confidence: float,
                      embedding: np.ndarray,
                      cluster_id: str) -> Optional[int]:
        """
        Agregar nueva detección a la campaña

        Returns:
            person_id si se confirma el ID, None si aún está en campaña
        """
        # Si ya es una persona confirmada, actualizar Kalman
        for person_id, person in self.confirmed_persons.items():
            if person.cluster_id == cluster_id:
                person.update_kalman(bbox)
                return person_id

        # Si no existe el candidato, crearlo
        if candidate_id not in self.candidates:
            self.candidates[candidate_id] = PersonCandidate(
                candidate_id=candidate_id,
                first_detection_frame=self.current_frame,
                last_detection_frame=self.current_frame,
                detection_count=1
            )

        # Actualizar candidato
        candidate = self.candidates[candidate_id]
        candidate.last_detection_frame = self.current_frame
        candidate.detection_count += 1
        candidate.confidence_scores.append(confidence)
        candidate.bounding_boxes.append(bbox)
        candidate.embeddings.append(embedding)
        candidate.cluster_assignments.append(cluster_id)

        # Verificar si la campaña está completa
        if self._is_campaign_complete(candidate):
            person_id = self._confirm_person(candidate)
            if person_id is not None:
                # Eliminar candidato después de confirmación
                del self.candidates[candidate_id]
                return person_id

        return None

    def _is_campaign_complete(self, candidate: PersonCandidate) -> bool:
        """Verificar si la campaña está completa"""
        # Verificar duración mínima
        if candidate.campaign_duration < self.campaign_frames:
            return False

        # Verificar mínimo de muestras
        if candidate.detection_count < self.campaign_min_samples:
            return False

        # Verificar confianza promedio
        if candidate.average_confidence < self.campaign_confidence_threshold:
            return False

        return True

    def _confirm_person(self, candidate: PersonCandidate) -> Optional[int]:
        """Confirmar persona y crear ID definitivo"""
        try:
            # Crear ID de persona
            person_id = self.next_person_id
            self.next_person_id += 1

            # Crear filtro Kalman con la última detección
            last_bbox = candidate.bounding_boxes[-1]
            kalman_filter = self._create_kalman_filter(last_bbox)

            # Crear persona confirmada
            confirmed_person = ConfirmedPerson(
                person_id=person_id,
                cluster_id=candidate.most_common_cluster,
                confirmation_frame=self.current_frame,
                kalman_filter=kalman_filter,
                last_detection=last_bbox,
                last_detection_frame=self.current_frame,
                total_detections=candidate.detection_count
            )

            # Agregar a personas confirmadas
            self.confirmed_persons[person_id] = confirmed_person

            # Actualizar estadísticas
            self.total_campaigns += 1
            self.successful_campaigns += 1

            self.logger.info(
                f"Persona confirmada: ID {person_id}, Cluster {candidate.most_common_cluster}")
            self.logger.info(
                f"  Campaña: {candidate.campaign_duration} frames, {candidate.detection_count} detecciones")
            self.logger.info(
                f"  Confianza: {candidate.average_confidence:.3f}, Consistencia: {candidate.cluster_consistency:.3f}")

            return person_id

        except Exception as e:
            self.logger.error(f"Error confirmando persona: {e}")
            self.failed_campaigns += 1
            return None

    def get_candidate_boxes(self) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Obtener cajas de candidatos (sin ID asignado)"""
        boxes = []
        for candidate in self.candidates.values():
            if candidate.bounding_boxes:
                last_bbox = candidate.bounding_boxes[-1]
                boxes.append(
                    (last_bbox, f"Candidato ({candidate.detection_count})"))
        return boxes

    def get_confirmed_boxes(self) -> List[Tuple[Tuple[int, int, int, int], int]]:
        """Obtener cajas de personas confirmadas con predicción Kalman"""
        boxes = []
        for person_id, person in self.confirmed_persons.items():
            if person.consecutive_misses <= self.kalman_prediction_frames:
                if person.consecutive_misses == 0:
                    # Usar detección real
                    bbox = person.last_detection
                else:
                    # Usar predicción Kalman
                    bbox = person.predict_next_position()

                boxes.append((bbox, person_id))
        return boxes

    def cleanup_old_candidates(self, max_age_frames: int = 100):
        """Limpiar candidatos antiguos que no han progresado"""
        to_remove = []
        for candidate_id, candidate in self.candidates.items():
            age = self.current_frame - candidate.last_detection_frame
            if age > max_age_frames:
                to_remove.append(candidate_id)

        for candidate_id in to_remove:
            del self.candidates[candidate_id]
            self.failed_campaigns += 1
            self.logger.debug(f"Candidato eliminado por edad: {candidate_id}")

    def cleanup_lost_persons(self):
        """Limpiar personas confirmadas que se han perdido"""
        to_remove = []
        for person_id, person in self.confirmed_persons.items():
            if person.consecutive_misses > self.max_consecutive_misses:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.confirmed_persons[person_id]
            self.logger.info(f"Persona perdida: ID {person_id}")

    def get_statistics(self) -> Dict:
        """Obtener estadísticas del gestor"""
        return {
            'total_campaigns': self.total_campaigns,
            'successful_campaigns': self.successful_campaigns,
            'failed_campaigns': self.failed_campaigns,
            'success_rate': self.successful_campaigns / max(1, self.total_campaigns),
            'active_candidates': len(self.candidates),
            'confirmed_persons': len(self.confirmed_persons),
            'next_person_id': self.next_person_id
        }

    def get_person_info(self, person_id: int) -> Optional[Dict]:
        """Obtener información detallada de una persona"""
        if person_id not in self.confirmed_persons:
            return None

        person = self.confirmed_persons[person_id]
        return {
            'person_id': person_id,
            'cluster_id': person.cluster_id,
            'confirmation_frame': person.confirmation_frame,
            'total_detections': person.total_detections,
            'consecutive_misses': person.consecutive_misses,
            'last_detection': person.last_detection,
            'last_detection_frame': person.last_detection_frame
        }
