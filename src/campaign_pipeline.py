#!/usr/bin/env python3
"""
Campaign-based Person Re-identification Pipeline
Integra el sistema de campañas con el pipeline existente para asignación retardada de IDs
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .main_pipeline import PersonReIDPipeline
from .clustering.campaign_manager import CampaignManager
from .clustering.hnsw_gaussian_integration import HNSWGaussianIntegration
from config.global_config import get_config


class CampaignPersonReIDPipeline(PersonReIDPipeline):
    """
    Pipeline con sistema de campañas para asignación retardada de IDs
    """

    def __init__(self,
                 enable_campaigns: bool = True,
                 campaign_frames: int = 25,
                 campaign_min_samples: int = 6,
                 campaign_confidence_threshold: float = 0.75,
                 **kwargs):
        """
        Inicializar pipeline con sistema de campañas

        Args:
            enable_campaigns: Activar sistema de campañas
            campaign_frames: Frames mínimos para completar campaña
            campaign_min_samples: Mínimo de muestras para asignar ID
            campaign_confidence_threshold: Confianza mínima para asignar ID
            **kwargs: Argumentos para el pipeline base
        """
        # Inicializar pipeline base
        super().__init__(**kwargs)

        # Configuración de campañas
        self.enable_campaigns = enable_campaigns

        if self.enable_campaigns:
            # Obtener configuración de clustering Gaussiano
            gaussian_config = get_config('gaussian_clustering')

            # Inicializar gestor de campañas
            self.campaign_manager = CampaignManager(
                campaign_frames=campaign_frames,
                campaign_min_samples=campaign_min_samples,
                campaign_confidence_threshold=campaign_confidence_threshold,
                max_consecutive_misses=10,
                kalman_prediction_frames=5
            )

            # Inicializar sistema de clustering Gaussiano
            hnsw_config = {
                'M': 16,
                'ef_construction': 200,
                'ef_search': 50,
                'max_elements': 500,
                'space': 'cosine'
            }

            gaussian_config_params = {
                'mahalanobis_threshold': gaussian_config.get('mahalanobis_threshold', 4.0),
                'min_samples_for_update': gaussian_config.get('min_samples_per_cluster', 1),
                'max_clusters': gaussian_config.get('max_clusters', 10),
                'fusion_threshold': 1.5,
                'split_threshold': 8.0
            }

            self.gaussian_clustering = HNSWGaussianIntegration(
                embedding_dim=512,
                hnsw_config=hnsw_config,
                gaussian_config=gaussian_config_params,
                top_k_candidates=10,
                auto_fusion_interval=5
            )

            # Colores para visualización
            self.candidate_color = (0, 255, 255)  # Amarillo para candidatos
            self.confirmed_color = (0, 255, 0)    # Verde para confirmados

            self.logger.info("Sistema de campañas inicializado")
        else:
            self.campaign_manager = None
            self.gaussian_clustering = None
            self.logger.info("Sistema de campañas desactivado")

    def _process_frame_with_campaigns(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict]:
        """
        Procesar frame con sistema de campañas
        """
        # Detectar personas
        detection_results = self.detector.detect_persons(frame)
        detections = detection_results['detections']

        # Actualizar frame en campaign manager
        self.campaign_manager.update_frame(frame_number)

        # Procesar cada detección
        frame_results = {
            'detections': len(detections),
            'candidates': 0,
            'confirmed_persons': 0,
            'new_confirmations': 0
        }

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Extraer crop de la persona
            crop = self._extract_crop(frame, bbox)

            if crop is not None:
                # Extraer embedding
                embedding = self.reid.extract_features([crop])[0]

                # Asignar a cluster usando sistema Gaussiano
                cluster_id, distance, metadata = self.gaussian_clustering.assign_identity(
                    embedding)

                # Generar ID de candidato único
                candidate_id = f"det_{frame_number}_{i}"

                # Procesar con campaign manager
                person_id = self.campaign_manager.add_detection(
                    candidate_id=candidate_id,
                    bbox=bbox,
                    confidence=confidence,
                    embedding=embedding,
                    cluster_id=cluster_id
                )

                if person_id is not None:
                    frame_results['new_confirmations'] += 1
                    self.logger.info(
                        f"Nueva persona confirmada: ID {person_id}")

        # Obtener cajas para visualización
        candidate_boxes = self.campaign_manager.get_candidate_boxes()
        confirmed_boxes = self.campaign_manager.get_confirmed_boxes()

        frame_results['candidates'] = len(candidate_boxes)
        frame_results['confirmed_persons'] = len(confirmed_boxes)

        # Dibujar cajas en el frame
        annotated_frame = self._draw_campaign_boxes(
            frame, candidate_boxes, confirmed_boxes)

        # Limpiar candidatos antiguos
        if frame_number % 50 == 0:
            self.campaign_manager.cleanup_old_candidates()
            self.campaign_manager.cleanup_lost_persons()

        return annotated_frame, frame_results

    def _draw_campaign_boxes(self,
                             frame: np.ndarray,
                             candidate_boxes: List[Tuple[Tuple[int, int, int, int], str]],
                             confirmed_boxes: List[Tuple[Tuple[int, int, int, int], int]]) -> np.ndarray:
        """
        Dibujar cajas de candidatos y personas confirmadas
        """
        annotated_frame = frame.copy()

        # Dibujar candidatos (sin ID, color amarillo)
        for (x1, y1, x2, y2), label in candidate_boxes:
            cv2.rectangle(annotated_frame, (x1, y1),
                          (x2, y2), self.candidate_color, 2)

            # Etiqueta para candidato
            label_text = f"Candidato"
            label_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          self.candidate_color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Dibujar personas confirmadas (con ID, color verde)
        for (x1, y1, x2, y2), person_id in confirmed_boxes:
            cv2.rectangle(annotated_frame, (x1, y1),
                          (x2, y2), self.confirmed_color, 2)

            # Etiqueta con ID
            label_text = f"Persona {person_id}"
            label_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          self.confirmed_color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_frame

    def _extract_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extraer crop de persona del frame
        """
        try:
            x1, y1, x2, y2 = bbox

            # Validar coordenadas
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            # Extraer crop
            crop = frame[y1:y2, x1:x2]

            # Validar tamaño mínimo
            if crop.shape[0] < 32 or crop.shape[1] < 16:
                return None

            return crop

        except Exception as e:
            self.logger.error(f"Error extrayendo crop: {e}")
            return None

    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """
        Procesar video con sistema de campañas
        """
        self.logger.info(f"Procesando video: {video_path}")

        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(
            f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Configurar límite de frames
        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Configurar writer de video de salida
        output_video_path = None
        video_writer = None

        if self.save_video:
            output_video_path = self.output_dir / "videos" / \
                f"campaign_output_{int(time.time())}.mp4"
            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height))

        # Estadísticas de procesamiento
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_candidates': 0,
            'total_confirmed': 0,
            'total_confirmations': 0,
            'processing_times': [],
            'avg_fps': 0.0,
            'campaign_stats': {}
        }

        frame_number = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_number >= max_frames):
                    break

                frame_start_time = time.time()

                # Procesar frame
                if self.enable_campaigns:
                    annotated_frame, frame_results = self._process_frame_with_campaigns(
                        frame, frame_number)
                else:
                    # Usar procesamiento estándar
                    annotated_frame, frame_results = self._process_frame_standard(
                        frame, frame_number)

                frame_processing_time = time.time() - frame_start_time
                stats['processing_times'].append(frame_processing_time)

                # Actualizar estadísticas
                stats['total_frames'] += 1
                stats['total_detections'] += frame_results.get('detections', 0)
                stats['total_candidates'] += frame_results.get('candidates', 0)
                stats['total_confirmed'] += frame_results.get(
                    'confirmed_persons', 0)
                stats['total_confirmations'] += frame_results.get(
                    'new_confirmations', 0)

                # Guardar frame anotado
                if video_writer:
                    video_writer.write(annotated_frame)

                # Mostrar progreso
                if frame_number % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_number / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Frame {frame_number}/{total_frames} ({fps_current:.1f} FPS)")

                frame_number += 1

        finally:
            cap.release()
            if video_writer:
                video_writer.release()

        # Calcular estadísticas finales
        total_time = time.time() - start_time
        stats['avg_fps'] = stats['total_frames'] / \
            total_time if total_time > 0 else 0

        # Obtener estadísticas de campañas
        if self.enable_campaigns:
            stats['campaign_stats'] = self.campaign_manager.get_statistics()

        self.logger.info(
            f"Procesamiento completado: {stats['total_frames']} frames en {total_time:.2f}s")
        self.logger.info(
            f"Detecciones: {stats['total_detections']}, Confirmaciones: {stats['total_confirmations']}")

        if output_video_path:
            self.logger.info(f"Video guardado: {output_video_path}")

        return stats

    def _process_frame_standard(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict]:
        """
        Procesamiento estándar (sin campañas) para comparación
        """
        # Usar el procesamiento del pipeline base
        detection_results = self.detector.detect_persons(frame)
        detections = detection_results['detections']

        frame_results = {
            'detections': len(detections),
            'candidates': 0,
            'confirmed_persons': len(detections),
            'new_confirmations': 0
        }

        # Dibujar detecciones básicas
        annotated_frame = frame.copy()
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person {i}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame, frame_results

    def get_campaign_statistics(self) -> Optional[Dict]:
        """
        Obtener estadísticas detalladas de las campañas
        """
        if not self.enable_campaigns:
            return None

        return self.campaign_manager.get_statistics()

    def get_confirmed_persons(self) -> List[Dict]:
        """
        Obtener información de todas las personas confirmadas
        """
        if not self.enable_campaigns:
            return []

        persons = []
        for person_id in self.campaign_manager.confirmed_persons.keys():
            person_info = self.campaign_manager.get_person_info(person_id)
            if person_info:
                persons.append(person_info)

        return persons
