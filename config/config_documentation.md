# Documentación de Configuración Global
## Sistema de Re-identificación de Personas

Este documento explica detalladamente cada variable de configuración del sistema, su propósito teórico y cómo afecta el rendimiento del sistema.

---

## 1. CONFIGURACIÓN DE RUTAS (PATHS)

### Propósito Teórico
Las rutas definen la estructura de directorios del proyecto y ubicaciones de archivos críticos. Una organización clara facilita el mantenimiento y escalabilidad del sistema.

### Variables

#### Directorios Base
- **`project_root`**: Directorio raíz del proyecto
  - **Propósito**: Punto de referencia para todas las rutas relativas
  - **Impacto**: Facilita la portabilidad del código entre diferentes sistemas

- **`data_dir`**: Directorio principal de datos
  - **Propósito**: Centraliza todos los datos del proyecto
  - **Impacto**: Organización clara para datos de entrada y salida

- **`raw_data_dir`**: Datos sin procesar (videos originales)
  - **Propósito**: Almacena videos y datos originales sin modificar
  - **Impacto**: Preserva datos originales para reproducibilidad

- **`processed_data_dir`**: Datos procesados (resultados, crops, etc.)
  - **Propósito**: Almacena resultados del procesamiento
  - **Impacto**: Separación clara entre entrada y salida

- **`models_dir`**: Directorio de modelos
  - **Propósito**: Centraliza todos los modelos de ML
  - **Impacto**: Facilita gestión y versionado de modelos

- **`output_dir`**: Directorio de salida
  - **Propósito**: Almacena resultados finales del sistema
  - **Impacto**: Organización clara de productos finales

#### Subdirectorios Especializados
- **`yolo_models_dir`**: Modelos YOLO específicos
  - **Propósito**: Organiza diferentes versiones de YOLO
  - **Impacto**: Facilita cambio entre modelos de detección

- **`reid_models_dir`**: Modelos de re-identificación
  - **Propósito**: Almacena modelos OSNet y similares
  - **Impacto**: Gestión eficiente de modelos ReID

---

## 2. CONFIGURACIÓN DE DETECCIÓN (DETECTION)

### Propósito Teórico
La detección es el primer paso del pipeline. Identifica personas en cada frame del video. Los parámetros controlan el balance entre precisión, velocidad y consumo de recursos.

### Variables de Modelo

#### `model_path` (str)
- **Propósito**: Especifica qué modelo YOLO usar
- **Opciones**: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
- **Impacto**: 
  - **n (nano)**: Más rápido, menor precisión
  - **x (extra-large)**: Más lento, mayor precisión
- **Recomendación**: 'yolov8s.pt' para balance óptimo

#### `confidence_threshold` (float: 0.0-1.0)
- **Propósito**: Filtra detecciones con baja confianza
- **Teoría**: Evita falsos positivos manteniendo detecciones válidas
- **Impacto**:
  - **Alto (0.7-0.9)**: Menos falsos positivos, puede perder detecciones reales
  - **Bajo (0.3-0.5)**: Más detecciones, mayor riesgo de falsos positivos
- **Recomendación**: 0.5 para uso general

#### `iou_threshold` (float: 0.0-1.0)
- **Propósito**: Controla Non-Maximum Suppression (NMS)
- **Teoría**: Elimina detecciones duplicadas del mismo objeto
- **Impacto**:
  - **Alto (0.7-0.9)**: Permite más solapamiento, puede mantener duplicados
  - **Bajo (0.3-0.5)**: Elimina más solapamientos, puede eliminar detecciones válidas
- **Recomendación**: 0.5 para balance óptimo

#### `input_size` (tuple: width, height)
- **Propósito**: Tamaño de imagen para procesamiento YOLO
- **Teoría**: Resolución mayor = mejor detección de objetos pequeños
- **Impacto**:
  - **Grande (1280, 1280)**: Mejor precisión, mucho más lento
  - **Pequeño (416, 416)**: Más rápido, puede perder objetos pequeños
- **Recomendación**: (640, 640) para balance

### Variables de Rendimiento

#### `batch_size` (int)
- **Propósito**: Número de imágenes procesadas simultáneamente
- **Teoría**: Batch mayor = mejor utilización de GPU
- **Impacto**:
  - **Alto**: Más rápido en GPU, mayor uso de memoria
  - **Bajo**: Menos memoria, puede ser más lento
- **Recomendación**: 1 para video en tiempo real

#### `half_precision` (bool)
- **Propósito**: Usa FP16 en lugar de FP32
- **Teoría**: Reduce uso de memoria y aumenta velocidad en GPUs compatibles
- **Impacto**: 2x más rápido, ligeramente menos preciso
- **Recomendación**: True para GPUs modernas

---

## 3. CONFIGURACIÓN DE SEGUIMIENTO (TRACKING)

### Propósito Teórico
El seguimiento mantiene identidades temporales de personas entre frames. Usa compresión Huffman para optimizar almacenamiento de trayectorias.

### Variables de Gestión de Tracks

#### `max_disappeared` (int)
- **Propósito**: Frames máximos que una persona puede desaparecer
- **Teoría**: Balance entre persistencia y limpieza de memoria
- **Impacto**:
  - **Alto (50+)**: Mantiene tracks más tiempo, más memoria
  - **Bajo (10-20)**: Limpia tracks rápido, puede perder personas temporalmente ocultas
- **Recomendación**: 30 para videos típicos

#### `max_distance` (float)
- **Propósito**: Distancia máxima para asociar detecciones con tracks
- **Teoría**: Controla qué tan lejos puede moverse una persona entre frames
- **Impacto**:
  - **Alto**: Asocia movimientos rápidos, puede causar asociaciones incorrectas
  - **Bajo**: Asociaciones más precisas, puede crear tracks duplicados
- **Recomendación**: 100.0 píxeles para videos HD

#### `min_track_length` (int)
- **Propósito**: Frames mínimos para considerar un track válido
- **Teoría**: Filtra detecciones esporádicas y ruido
- **Impacto**:
  - **Alto**: Tracks más estables, puede perder apariciones breves
  - **Bajo**: Captura apariciones breves, puede incluir ruido
- **Recomendación**: 3 frames mínimo

### Variables de Compresión Huffman

#### `coordinate_discretization` (int)
- **Propósito**: Tamaño de grilla para discretizar coordenadas
- **Teoría**: Reduce precisión para mejorar compresión
- **Impacto**:
  - **Alto**: Mejor compresión, menos precisión espacial
  - **Bajo**: Más precisión, menor compresión
- **Recomendación**: 10 para balance óptimo

#### `compression_window` (int)
- **Propósito**: Frames para entrenar el codificador Huffman
- **Teoría**: Ventana mayor = mejor modelo de compresión
- **Impacto**:
  - **Alto**: Mejor compresión, mayor latencia inicial
  - **Bajo**: Compresión rápida, puede ser menos eficiente
- **Recomendación**: 100 frames

### Variables de Predicción

#### `enable_prediction` (bool)
- **Propósito**: Usa predicción de movimiento para tracking
- **Teoría**: Predice posición futura basada en movimiento pasado
- **Impacto**: Mejor asociación en movimientos rápidos
- **Recomendación**: True para mejorar robustez

#### `prediction_frames` (int)
- **Propósito**: Frames históricos para predicción
- **Teoría**: Más frames = mejor modelo de movimiento
- **Impacto**:
  - **Alto**: Predicción más precisa, mayor cómputo
  - **Bajo**: Predicción rápida, puede ser menos precisa
- **Recomendación**: 3 frames para balance

---

## 4. CONFIGURACIÓN DE RE-IDENTIFICACIÓN (REID)

### Propósito Teórico
La re-identificación extrae características visuales únicas de cada persona para reconocerlas en diferentes frames y cámaras.

### Variables de Modelo

#### `model_name` (str)
- **Propósito**: Selecciona arquitectura de red neuronal
- **Opciones**: 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5'
- **Teoría**: Balance entre precisión y velocidad
- **Impacto**:
  - **x1_0**: Máxima precisión, más lento
  - **x0_5**: Más rápido, menor precisión
- **Recomendación**: 'osnet_x0_75' para balance

#### `feature_dim` (int)
- **Propósito**: Dimensionalidad del vector de características
- **Teoría**: Más dimensiones = mayor capacidad representativa
- **Impacto**:
  - **Alto (2048+)**: Mejor discriminación, más memoria
  - **Bajo (512)**: Menos memoria, puede reducir precisión
- **Recomendación**: 2048 para alta precisión

#### `image_size` (tuple: height, width)
- **Propósito**: Tamaño de entrada para el modelo ReID
- **Teoría**: Resolución mayor = mejor captura de detalles
- **Impacto**:
  - **Grande (384, 128)**: Mejor precisión, más lento
  - **Pequeño (256, 128)**: Más rápido, puede perder detalles
- **Recomendación**: (256, 128) estándar

### Variables de Procesamiento

#### `batch_size` (int)
- **Propósito**: Imágenes procesadas simultáneamente
- **Teoría**: Batch mayor = mejor utilización de GPU
- **Impacto**:
  - **Alto (64+)**: Más rápido en GPU, mayor memoria
  - **Bajo (16)**: Menos memoria, puede ser más lento
- **Recomendación**: 32 para balance

#### `normalize_features` (bool)
- **Propósito**: Normaliza vectores de características
- **Teoría**: Mejora comparación de similitud coseno
- **Impacto**: Mejor rendimiento en búsqueda de similitud
- **Recomendación**: True siempre

### Variables de Calidad

#### `similarity_threshold` (float: 0.0-1.0)
- **Propósito**: Umbral mínimo para considerar match positivo
- **Teoría**: Balance entre precisión y recall
- **Impacto**:
  - **Alto (0.8+)**: Menos falsos positivos, puede perder matches reales
  - **Bajo (0.5)**: Más matches, mayor riesgo de falsos positivos
- **Recomendación**: 0.7 para balance

#### `min_crop_size` (tuple: width, height)
- **Propósito**: Tamaño mínimo de recorte de persona
- **Teoría**: Filtra detecciones muy pequeñas con poco detalle
- **Impacto**: Mejora calidad de características extraídas
- **Recomendación**: (32, 64) mínimo

---

## 5. CONFIGURACIÓN DE INDEXACIÓN (INDEXING)

### Propósito Teórico
HNSW (Hierarchical Navigable Small World) es una estructura de datos para búsqueda eficiente de vectores similares en alta dimensionalidad.

### Variables de Estructura

#### `dimension` (int)
- **Propósito**: Dimensionalidad de vectores de características
- **Teoría**: Debe coincidir con salida del modelo ReID
- **Impacto**: Determina capacidad de representación
- **Recomendación**: 2048 para OSNet

#### `max_elements` (int)
- **Propósito**: Número máximo de vectores en índice
- **Teoría**: Capacidad del sistema de personas únicas
- **Impacto**:
  - **Alto**: Más capacidad, mayor uso de memoria
  - **Bajo**: Menos memoria, puede necesitar limpieza periódica
- **Recomendación**: 10000 para sistemas grandes

#### `space` (str)
- **Propósito**: Métrica de distancia para comparación
- **Opciones**: 'cosine', 'l2', 'ip'
- **Teoría**:
  - **cosine**: Mejor para vectores normalizados
  - **l2**: Distancia euclidiana clásica
  - **ip**: Producto interno para casos específicos
- **Recomendación**: 'cosine' para características ReID

### Variables HNSW

#### `M` (int)
- **Propósito**: Enlaces bidireccionales por nodo
- **Teoría**: Controla conectividad del grafo
- **Impacto**:
  - **Alto (32+)**: Mejor recall, más memoria, construcción más lenta
  - **Bajo (8)**: Construcción rápida, puede reducir recall
- **Recomendación**: 16 para balance óptimo

#### `ef_construction` (int)
- **Propósito**: Tamaño de lista de candidatos durante construcción
- **Teoría**: Controla calidad del índice construido
- **Impacto**:
  - **Alto (400+)**: Mejor calidad, construcción más lenta
  - **Bajo (100)**: Construcción rápida, puede reducir calidad
- **Recomendación**: 200 para balance

#### `ef_search` (int)
- **Propósito**: Tamaño de lista de candidatos durante búsqueda
- **Teoría**: Controla balance precisión/velocidad en búsqueda
- **Impacto**:
  - **Alto (100+)**: Mejor recall, búsqueda más lenta
  - **Bajo (25)**: Búsqueda rápida, puede reducir recall
- **Recomendación**: 50 para balance, ajustable en tiempo real

### Variables de Optimización

#### `num_threads` (int)
- **Propósito**: Hilos para operaciones paralelas
- **Teoría**: Paralelización mejora velocidad en CPUs multi-core
- **Impacto**: Mejor utilización de CPU, cuidado con overhead
- **Recomendación**: -1 para detección automática

#### `duplicate_threshold` (float: 0.0-1.0)
- **Propósito**: Umbral para detectar vectores duplicados
- **Teoría**: Evita almacenar características muy similares
- **Impacto**: Reduce redundancia, mejora eficiencia
- **Recomendación**: 0.95 para filtrar casi-duplicados

---

## 6. CONFIGURACIÓN DE FUSIÓN (FUSION)

### Propósito Teórico
La fusión combina resultados de tracking y ReID para asignar IDs globales consistentes, manteniendo identidades a través del tiempo.

### Variables de Pesos

#### `tracking_weight` (float: 0.0-1.0)
- **Propósito**: Peso de confianza del tracking en fusión
- **Teoría**: Balance entre continuidad temporal y precisión visual
- **Impacto**:
  - **Alto**: Confía más en tracking, IDs más estables
  - **Bajo**: Confía más en ReID, mejor precisión visual
- **Recomendación**: 0.4 para balance

#### `reid_weight` (float: 0.0-1.0)
- **Propósito**: Peso de confianza del ReID en fusión
- **Teoría**: Complementa tracking_weight (suma = 1.0)
- **Impacto**:
  - **Alto**: Mejor discriminación visual, puede ser menos estable
  - **Bajo**: Más estable temporalmente, puede perder precisión
- **Recomendación**: 0.6 para priorizar precisión visual

#### `temporal_weight` (float: 0.0-1.0)
- **Propósito**: Peso de consistencia temporal
- **Teoría**: Favorece asignaciones consistentes en el tiempo
- **Impacto**: Estabiliza IDs, reduce flickering
- **Recomendación**: 0.2 para estabilidad adicional

### Variables de Umbral

#### `similarity_threshold` (float: 0.0-1.0)
- **Propósito**: Umbral mínimo para asignación de ID
- **Teoría**: Controla cuándo crear nuevo ID vs reutilizar existente
- **Impacto**:
  - **Alto (0.8+)**: Menos falsos positivos, más IDs únicos
  - **Bajo (0.6)**: Más reutilización, riesgo de fusión incorrecta
- **Recomendación**: 0.7 para balance

#### `reid_distance_threshold` (float)
- **Propósito**: Distancia máxima ReID para match válido
- **Teoría**: Complementa similarity_threshold
- **Impacto**: Filtra matches de baja calidad
- **Recomendación**: 0.5 para filtrado efectivo

### Variables Temporales

#### `temporal_window` (int)
- **Propósito**: Frames considerados para consistencia temporal
- **Teoría**: Ventana mayor = decisiones más informadas
- **Impacto**:
  - **Alto (50+)**: IDs más estables, adaptación más lenta
  - **Bajo (10)**: Adaptación rápida, puede ser menos estable
- **Recomendación**: 30 para balance

#### `history_length` (int)
- **Propósito**: Longitud de historial mantenido por persona
- **Teoría**: Historia más larga = mejor caracterización
- **Impacto**: Mejor re-identificación, mayor uso de memoria
- **Recomendación**: 50 para balance memoria/precisión

### Variables de Gestión de IDs

#### `max_global_ids` (int)
- **Propósito**: Número máximo de IDs globales
- **Teoría**: Limita crecimiento de sistema
- **Impacto**: Previene uso excesivo de memoria
- **Recomendación**: 1000 para sistemas grandes

#### `id_reuse_delay` (int)
- **Propósito**: Frames antes de reutilizar un ID
- **Teoría**: Evita reutilización prematura de IDs
- **Impacto**: Reduce confusión de identidades
- **Recomendación**: 500 frames para seguridad

#### `inactive_threshold` (int)
- **Propósito**: Frames para marcar ID como inactivo
- **Teoría**: Gestión eficiente de IDs no utilizados
- **Impacto**: Limpia sistema de IDs obsoletos
- **Recomendación**: 100 frames para limpieza regular

---

## 7. CONFIGURACIÓN DE PIPELINE (PIPELINE)

### Propósito Teórico
El pipeline orquesta todos los módulos del sistema, controlando flujo de datos, procesamiento y optimizaciones generales.

### Variables de Control

#### `max_frames` (int o None)
- **Propósito**: Límite máximo de frames a procesar
- **Teoría**: Control de recursos y tiempo de procesamiento
- **Impacto**:
  - **None**: Procesa video completo
  - **Número**: Procesa solo N frames
- **Recomendación**: None para procesamiento completo

#### `frame_skip` (int)
- **Propósito**: Procesa cada N-ésimo frame
- **Teoría**: Reduce carga computacional sacrificando resolución temporal
- **Impacto**:
  - **1**: Procesa todos los frames, máxima precisión
  - **2+**: Más rápido, puede perder eventos rápidos
- **Recomendación**: 1 para máxima precisión

#### `start_frame` (int)
- **Propósito**: Frame inicial para procesamiento
- **Teoría**: Permite procesamiento de segmentos específicos
- **Impacto**: Útil para debugging y análisis parcial
- **Recomendación**: 0 para procesamiento completo

### Variables de Rendimiento

#### `enable_multithreading` (bool)
- **Propósito**: Habilita procesamiento multi-hilo
- **Teoría**: Paralelización mejora velocidad en CPUs multi-core
- **Impacto**: Más rápido, mayor complejidad, posible overhead
- **Recomendación**: False para estabilidad, True para velocidad

#### `num_threads` (int)
- **Propósito**: Número de hilos de procesamiento
- **Teoría**: Óptimo = número de cores CPU
- **Impacto**: Más hilos no siempre = más rápido
- **Recomendación**: 4 para balance

#### `real_time_mode` (bool)
- **Propósito**: Optimizaciones para procesamiento en tiempo real
- **Teoría**: Prioriza latencia sobre throughput
- **Impacto**: Menor latencia, puede reducir precisión
- **Recomendación**: False para máxima precisión

### Variables de Memoria

#### `max_memory_usage` (str)
- **Propósito**: Límite máximo de uso de memoria
- **Teoría**: Previene agotamiento de memoria del sistema
- **Impacto**: Limita batch sizes y caches
- **Recomendación**: '8GB' para sistemas típicos

#### `enable_garbage_collection` (bool)
- **Propósito**: Habilita limpieza automática de memoria
- **Teoría**: Previene acumulación de objetos no utilizados
- **Impacto**: Mejor gestión de memoria, ligero overhead
- **Recomendación**: True para estabilidad

#### `gc_interval` (int)
- **Propósito**: Intervalo para garbage collection
- **Teoría**: Balance entre limpieza y rendimiento
- **Impacto**: Más frecuente = menos memoria, más overhead
- **Recomendación**: 100 frames para balance

### Variables de Manejo de Errores

#### `continue_on_error` (bool)
- **Propósito**: Continúa procesamiento tras errores no fatales
- **Teoría**: Robustez vs precisión
- **Impacto**: Mayor robustez, puede perder algunos frames
- **Recomendación**: True para robustez

#### `max_consecutive_errors` (int)
- **Propósito**: Errores consecutivos antes de parar
- **Teoría**: Evita loops infinitos de error
- **Impacto**: Balance entre robustez y detección de problemas
- **Recomendación**: 10 para balance

---

## 8. CONFIGURACIÓN DE SALIDA (OUTPUT)

### Propósito Teórico
Controla qué datos se guardan, en qué formato y cómo se organizan los resultados del procesamiento.

### Variables de Guardado

#### `save_video` (bool)
- **Propósito**: Guarda video anotado con detecciones
- **Teoría**: Visualización de resultados
- **Impacto**: Archivo grande, útil para análisis visual
- **Recomendación**: True para verificación visual

#### `save_crops` (bool)
- **Propósito**: Guarda recortes individuales de personas
- **Teoría**: Dataset para análisis posterior
- **Impacto**: Muchos archivos pequeños, útil para ReID
- **Recomendación**: True para análisis detallado

#### `save_features` (bool)
- **Propósito**: Guarda vectores de características
- **Teoría**: Reutilización sin re-procesamiento
- **Impacto**: Archivos grandes, útil para análisis offline
- **Recomendación**: False para ahorrar espacio

#### `save_statistics` (bool)
- **Propósito**: Guarda métricas de rendimiento
- **Teoría**: Análisis de performance del sistema
- **Impacto**: Archivos pequeños, muy útil para optimización
- **Recomendación**: True para monitoreo

### Variables de Video

#### `video_quality` (float: 0.0-1.0)
- **Propósito**: Calidad de compresión del video
- **Teoría**: Balance entre tamaño y calidad visual
- **Impacto**:
  - **Alto (0.9+)**: Mejor calidad, archivos grandes
  - **Bajo (0.5)**: Archivos pequeños, calidad reducida
- **Recomendación**: 0.8 para balance

#### `include_annotations` (bool)
- **Propósito**: Incluye bounding boxes y IDs en video
- **Teoría**: Visualización de resultados de detección
- **Impacto**: Mejor comprensión visual de resultados
- **Recomendación**: True para análisis visual

#### `annotation_thickness` (int)
- **Propósito**: Grosor de líneas de anotación
- **Teoría**: Visibilidad vs interferencia visual
- **Impacto**: Líneas gruesas = más visibles, pueden obstruir
- **Recomendación**: 2 para balance

### Variables de Crops

#### `crop_format` (str)
- **Propósito**: Formato de imagen para recortes
- **Opciones**: 'jpg', 'png'
- **Teoría**: Balance entre calidad y tamaño
- **Impacto**:
  - **jpg**: Archivos pequeños, ligera pérdida de calidad
  - **png**: Sin pérdida, archivos más grandes
- **Recomendación**: 'jpg' para eficiencia

#### `crop_quality` (int: 0-100)
- **Propósito**: Calidad JPEG para recortes
- **Teoría**: Balance entre calidad y tamaño de archivo
- **Impacto**: Mayor calidad = archivos más grandes
- **Recomendación**: 95 para alta calidad

#### `crop_padding` (float: 0.0-1.0)
- **Propósito**: Padding alrededor de recortes
- **Teoría**: Contexto adicional vs tamaño de imagen
- **Impacto**: Más padding = más contexto, imágenes más grandes
- **Recomendación**: 0.1 para contexto mínimo

### Variables de Organización

#### `organize_by_person` (bool)
- **Propósito**: Organiza crops por ID de persona
- **Teoría**: Facilita análisis por individuo
- **Impacto**: Estructura de carpetas más clara
- **Recomendación**: True para análisis por persona

#### `max_crops_per_person` (int)
- **Propósito**: Límite de crops por persona
- **Teoría**: Controla crecimiento de datos
- **Impacto**: Evita acumulación excesiva de imágenes
- **Recomendación**: 100 para balance

#### `create_galleries` (bool)
- **Propósito**: Crea galerías visuales por persona
- **Teoría**: Visualización rápida de resultados
- **Impacto**: Archivos HTML adicionales, útil para revisión
- **Recomendación**: True para análisis visual

---

## 9. CONFIGURACIÓN DE RENDIMIENTO (PERFORMANCE)

### Propósito Teórico
Optimiza el uso de recursos del sistema (CPU, GPU, memoria) para maximizar velocidad y eficiencia.

### Variables de Hardware

#### `auto_device_selection` (bool)
- **Propósito**: Selección automática de dispositivo óptimo
- **Teoría**: Usa GPU si disponible, fallback a CPU
- **Impacto**: Optimización automática de recursos
- **Recomendación**: True para simplicidad

#### `prefer_gpu` (bool)
- **Propósito**: Preferencia por GPU sobre CPU
- **Teoría**: GPU generalmente más rápida para ML
- **Impacto**: Mejor rendimiento si GPU disponible
- **Recomendación**: True para sistemas con GPU

#### `gpu_memory_fraction` (float: 0.0-1.0)
- **Propósito**: Fracción de memoria GPU a usar
- **Teoría**: Evita agotamiento de memoria GPU
- **Impacto**: Limita uso de memoria, puede reducir batch sizes
- **Recomendación**: 0.8 para dejar margen

### Variables de Batch

#### `adaptive_batch_size` (bool)
- **Propósito**: Ajusta batch size automáticamente
- **Teoría**: Optimiza uso de memoria disponible
- **Impacto**: Mejor utilización de recursos
- **Recomendación**: True para optimización automática

#### `max_batch_memory` (str)
- **Propósito**: Memoria máxima por batch
- **Teoría**: Previene out-of-memory errors
- **Impacto**: Limita batch sizes, garantiza estabilidad
- **Recomendación**: '2GB' para sistemas típicos

#### `prefetch_batches` (int)
- **Propósito**: Batches a preparar por adelantado
- **Teoría**: Paralelización de carga de datos
- **Impacto**: Mejor utilización de CPU/GPU
- **Recomendación**: 2 para balance

### Variables de Cache

#### `enable_model_caching` (bool)
- **Propósito**: Mantiene modelos en memoria
- **Teoría**: Evita recargas costosas
- **Impacto**: Más rápido, mayor uso de memoria
- **Recomendación**: True para mejor rendimiento

#### `cache_size_limit` (str)
- **Propósito**: Límite de tamaño de cache
- **Teoría**: Balance entre velocidad y memoria
- **Impacto**: Cache mayor = más rápido, más memoria
- **Recomendación**: '1GB' para balance

### Variables de Optimización

#### `optimization_level` (str)
- **Propósito**: Nivel de optimización general
- **Opciones**: 'speed', 'balanced', 'accuracy'
- **Teoría**: Presets para diferentes prioridades
- **Impacto**:
  - **speed**: Máxima velocidad, puede reducir precisión
  - **accuracy**: Máxima precisión, más lento
  - **balanced**: Balance óptimo
- **Recomendación**: 'balanced' para uso general

#### `enable_tensorrt` (bool)
- **Propósito**: Optimización TensorRT para NVIDIA
- **Teoría**: Optimización específica de hardware
- **Impacto**: Mucho más rápido en GPUs NVIDIA
- **Recomendación**: True para GPUs NVIDIA compatibles

### Variables de Monitoreo

#### `enable_profiling` (bool)
- **Propósito**: Habilita profiling de rendimiento
- **Teoría**: Identificación de cuellos de botella
- **Impacto**: Overhead mínimo, información valiosa
- **Recomendación**: False para producción, True para debugging

#### `monitoring_interval` (int)
- **Propósito**: Intervalo de actualización de métricas
- **Teoría**: Balance entre información y overhead
- **Impacto**: Más frecuente = más información, más overhead
- **Recomendación**: 10 segundos para balance

### Variables de Límites

#### `max_cpu_usage` (float: 0.0-1.0)
- **Propósito**: Límite de uso de CPU
- **Teoría**: Previene saturación del sistema
- **Impacto**: Mantiene sistema responsivo
- **Recomendación**: 0.8 para dejar margen

#### `max_memory_usage` (float: 0.0-1.0)
- **Propósito**: Límite de uso de memoria
- **Teoría**: Previene out-of-memory del sistema
- **Impacto**: Estabilidad del sistema
- **Recomendación**: 0.8 para seguridad

#### `thermal_throttling` (bool)
- **Propósito**: Protección contra sobrecalentamiento
- **Teoría**: Reduce rendimiento para proteger hardware
- **Impacto**: Previene daños por temperatura
- **Recomendación**: True para protección

---

## PRESETS DE CONFIGURACIÓN

### High Accuracy (Alta Precisión)
- **Propósito**: Máxima precisión sin importar velocidad
- **Uso**: Análisis offline, investigación forense
- **Características**: Modelos grandes, umbrales estrictos, procesamiento completo

### High Speed (Alta Velocidad)
- **Propósito**: Máxima velocidad con precisión aceptable
- **Uso**: Procesamiento en tiempo real, sistemas con recursos limitados
- **Características**: Modelos pequeños, umbrales relajados, optimizaciones agresivas

### Balanced (Balanceado)
- **Propósito**: Balance óptimo entre precisión y velocidad
- **Uso**: Aplicaciones generales, producción
- **Características**: Configuración intermedia, balance en todos los parámetros

---

## RECOMENDACIONES DE USO

### Para Sistemas de Producción
1. Usar preset 'balanced' como base
2. Ajustar `max_memory_usage` según hardware disponible
3. Habilitar `thermal_throttling` para protección
4. Configurar `save_statistics` para monitoreo

### Para Desarrollo y Testing
1. Usar preset 'high_accuracy' para validación
2. Habilitar `enable_profiling` para optimización
3. Configurar `max_frames` para pruebas rápidas
4. Usar `debug_mode` para información detallada

### Para Análisis Offline
1. Usar preset 'high_accuracy'
2. Habilitar `save_features` para reutilización
3. Configurar `create_galleries` para visualización
4. Usar `organize_by_person` para análisis individual

### Para Tiempo Real
1. Usar preset 'high_speed'
2. Habilitar `real_time_mode`
3. Configurar `frame_skip` si es necesario
4. Optimizar `batch_size` para hardware específico 