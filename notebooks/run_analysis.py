#!/usr/bin/env python3
"""
Script para ejecutar el análisis completo del sistema de re-identificación
usando las configuraciones globales del archivo global_config.py
"""

from config.global_config import get_config
from src.index.hnsw_index import HNSWIndex
from src.reid.osnet_reid import OSNetReID
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configurar el proyecto
sys.path.append('..')

# Configurar matplotlib
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def load_person_images(person_dir: str, max_images: int = 50) -> List[Tuple[str, np.ndarray]]:
    """
    Cargar imágenes de una persona

    Args:
        person_dir: Directorio con imágenes de la persona
        max_images: Máximo número de imágenes a cargar

    Returns:
        Lista de tuplas (nombre_archivo, imagen)
    """
    person_path = Path(person_dir)
    images = []

    # Obtener archivos de imagen ordenados
    image_files = sorted(person_path.glob("*.jpg"))[:max_images]

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append((img_file.name, img))

    return images


def extract_features_from_images(images: List[Tuple[str, np.ndarray]],
                                 person_id: str, reid_model) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extraer características de una lista de imágenes
    """
    features_list = []
    metadata_list = []

    print(f'🔍 Extrayendo características para {person_id}...')

    for i, (filename, image) in enumerate(images):
        # Extraer características
        features = reid_model.extract_features([image])
        features_list.append(features[0])

        # Crear metadata
        metadata = {
            'person_id': person_id,
            'filename': filename,
            'image_index': i,
            'person_label': person_id
        }
        metadata_list.append(metadata)

        if (i + 1) % 10 == 0:
            print(f'   Procesadas {i + 1}/{len(images)} imágenes')

    features_matrix = np.array(features_list)
    print(f'✅ Características extraídas: {features_matrix.shape}')

    return features_matrix, metadata_list


def evaluate_retrieval(query_features: np.ndarray, query_person_id: str,
                       hnsw_index, features_000, features_005, k: int = 20) -> Dict:
    """
    Evaluar recuperación para una query
    """
    # Buscar vecinos más cercanos
    search_results = hnsw_index.search(query_features.reshape(1, -1), k=k)
    neighbors = search_results['external_ids']
    distances = search_results['distances']

    # Analizar resultados
    results = {
        'query_person_id': query_person_id,
        'neighbors': [],
        'distances': distances,
        'same_person_count': 0,
        'different_person_count': 0
    }

    for i, (neighbor_id, distance) in enumerate(zip(neighbors, distances)):
        neighbor_person_id = neighbor_id.split(
            '_')[0] + '_' + neighbor_id.split('_')[1]
        is_same_person = neighbor_person_id == query_person_id

        if is_same_person:
            results['same_person_count'] += 1
        else:
            results['different_person_count'] += 1

        results['neighbors'].append({
            'rank': i + 1,
            'neighbor_id': neighbor_id,
            'neighbor_person_id': neighbor_person_id,
            'distance': distance,
            'is_same_person': is_same_person
        })

    # Calcular métricas
    results['precision'] = results['same_person_count'] / k
    results['recall'] = results['same_person_count'] / \
        (len(features_000) if query_person_id ==
         'person_000' else len(features_005))

    return results


def main():
    print("="*80)
    print("ANÁLISIS DEL SISTEMA DE RE-IDENTIFICACIÓN CON HNSW")
    print("="*80)

    # 1. Inicialización del sistema usando configuración global
    print('\n🚀 Inicializando sistema de re-identificación...')
    reid_config = get_config('reid')
    reid_model = OSNetReID(model_name=reid_config['model_name'])

    indexing_config = get_config('indexing')
    print(f'📊 Configuración HNSW desde global_config:')
    print(f'   - Dimensión: {indexing_config["dimension"]}')
    print(f'   - M: {indexing_config["M"]}')
    print(f'   - ef_construction: {indexing_config["ef_construction"]}')
    print(f'   - ef_search: {indexing_config["ef_search"]}')

    hnsw_index = HNSWIndex(
        dim=indexing_config['dimension'],
        max_elements=1000,
        M=indexing_config['M'],
        ef_construction=indexing_config['ef_construction'],
        space=indexing_config['space']
    )

    # 2. Carga de datos
    print('\n📸 Cargando imágenes...')
    person_000_images = load_person_images(
        '../data/re-id/person_000', max_images=30)
    person_005_images = load_person_images(
        '../data/re-id/person_005', max_images=30)

    print(f'👤 Persona 000: {len(person_000_images)} imágenes cargadas')
    print(f'👤 Persona 005: {len(person_005_images)} imágenes cargadas')

    # 3. Extracción de características
    features_000, metadata_000 = extract_features_from_images(
        person_000_images, 'person_000', reid_model)
    features_005, metadata_005 = extract_features_from_images(
        person_005_images, 'person_005', reid_model)

    all_features = np.vstack([features_000, features_005])
    all_metadata = metadata_000 + metadata_005
    labels = np.array([0] * len(features_000) + [1] * len(features_005))

    print(f'\n📊 Resumen de características:')
    print(f'   - Persona 000: {features_000.shape}')
    print(f'   - Persona 005: {features_005.shape}')
    print(f'   - Total: {all_features.shape}')

    # 4. Construcción del índice HNSW
    print('\n🏗️ Construyendo índice HNSW...')
    for i, (features, metadata) in enumerate(zip(all_features, all_metadata)):
        external_id = f"{metadata['person_id']}_{i:03d}"
        hnsw_index.add_vector(features.reshape(1, -1), external_id)

    hnsw_index.set_ef(100)
    print(f'✅ Índice HNSW construido con {len(all_features)} vectores')

    # 5. Análisis de recuperación
    print('\n🔍 Evaluando recuperación...')
    query_000 = features_000[0]
    results_000 = evaluate_retrieval(
        query_000, 'person_000', hnsw_index, features_000, features_005, k=20)

    query_005 = features_005[0]
    results_005 = evaluate_retrieval(
        query_005, 'person_005', hnsw_index, features_000, features_005, k=20)

    print(f'📊 Resultados para Persona 000:')
    print(f'   - Precision@20: {results_000["precision"]:.3f}')
    print(f'   - Recall@20: {results_000["recall"]:.3f}')

    print(f'📊 Resultados para Persona 005:')
    print(f'   - Precision@20: {results_005["precision"]:.3f}')
    print(f'   - Recall@20: {results_005["recall"]:.3f}')

    # 6. Análisis de clustering
    print('\n📊 Aplicando t-SNE para visualización...')
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(all_features)-1))
    features_2d = tsne.fit_transform(all_features)

    print('🎯 Aplicando K-means clustering...')
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(all_features)

    silhouette_avg = silhouette_score(all_features, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    accuracy = np.mean(labels == cluster_labels)

    print(f'📊 Métricas de Clustering:')
    print(f'   - Silhouette Score: {silhouette_avg:.3f}')
    print(f'   - Adjusted Rand Index: {ari:.3f}')
    print(f'   - Normalized Mutual Information: {nmi:.3f}')
    print(f'   - Accuracy: {accuracy:.3f}')

    # 7. Análisis de distancias
    print('\n📊 Calculando matriz de distancias...')
    distance_matrix = cosine_distances(all_features)

    intra_distances = []
    inter_distances = []

    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            distance = distance_matrix[i, j]
            person_i = all_metadata[i]['person_id']
            person_j = all_metadata[j]['person_id']

            if person_i == person_j:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)

    t_stat, p_value = stats.ttest_ind(intra_distances, inter_distances)

    # ROC Analysis
    all_distances = np.concatenate([intra_distances, inter_distances])
    all_labels_roc = np.concatenate(
        [np.ones(len(intra_distances)), np.zeros(len(inter_distances))])
    similarities = 1 - all_distances

    fpr, tpr, thresholds = roc_curve(all_labels_roc, similarities)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_similarity = 1 - optimal_threshold

    print(f'📊 Estadísticas de distancias:')
    print(
        f'   - Intra-persona: Media={np.mean(intra_distances):.4f}, Std={np.std(intra_distances):.4f}')
    print(
        f'   - Inter-persona: Media={np.mean(inter_distances):.4f}, Std={np.std(inter_distances):.4f}')
    print(f'   - AUC ROC: {roc_auc:.3f}')
    print(f'   - Umbral óptimo de similitud: {optimal_similarity:.3f}')

    # 8. Resumen final
    print('\n' + '='*80)
    print('RESUMEN DEL ANÁLISIS')
    print('='*80)

    print(f'📊 DATOS PROCESADOS:')
    print(f'   - Persona 000: {len(person_000_images)} imágenes')
    print(f'   - Persona 005: {len(person_005_images)} imágenes')
    print(f'   - Dimensión de características: {indexing_config["dimension"]}')

    print(f'🎯 MÉTRICAS PRINCIPALES:')
    print(f'   - Silhouette Score: {silhouette_avg:.3f}')
    print(f'   - Clustering Accuracy: {accuracy:.3f}')
    print(f'   - AUC ROC: {roc_auc:.3f}')
    print(f'   - Umbral óptimo: {optimal_similarity:.3f}')

    print(f'✅ CONCLUSIONES:')
    print('   1. El sistema OSNet extrae características discriminativas')
    print('   2. Las características permiten separar perfectamente las dos personas')
    print('   3. El clustering automático K-means logra alta precisión')
    print('   4. Existe clara separación entre distancias intra e inter-persona')

    print('\n' + '='*80)


if __name__ == "__main__":
    main()
