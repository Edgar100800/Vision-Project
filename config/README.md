# Sistema de Configuración Global

Este directorio contiene el sistema de configuración global para el proyecto de Re-identificación de Personas. Permite gestionar de manera centralizada todos los parámetros del sistema.

## Archivos Principales

### `global_config.py`
Archivo principal que contiene todas las configuraciones del sistema organizadas por módulos:

- **PATHS**: Rutas de directorios y archivos
- **DETECTION**: Configuración de detección YOLO
- **TRACKING**: Configuración de seguimiento Huffman
- **REID**: Configuración de re-identificación OSNet
- **INDEXING**: Configuración de índice HNSW
- **FUSION**: Configuración de fusión de IDs
- **PIPELINE**: Configuración del pipeline principal
- **OUTPUT**: Configuración de salida
- **PERFORMANCE**: Configuración de rendimiento

### `config_documentation.md`
Documentación detallada de cada variable de configuración, incluyendo:
- Propósito teórico de cada parámetro
- Impacto en el rendimiento del sistema
- Valores recomendados
- Rangos válidos

## Uso Básico

### 1. Importar la configuración

```python
from config.global_config import get_config, update_config, apply_preset
```

### 2. Obtener configuración

```python
# Obtener configuración de un módulo específico
detection_config = get_config('detection')
print(f"Umbral de confianza: {detection_config['confidence_threshold']}")

# Obtener toda la configuración
all_config = get_config()
```

### 3. Actualizar configuración

```python
# Actualizar un parámetro específico
update_config('detection', 'confidence_threshold', 0.7)

# Aplicar un preset predefinido
apply_preset('high_accuracy')  # o 'high_speed', 'balanced'
```

### 4. Usar en el pipeline

```python
# El pipeline usa automáticamente la configuración global
pipeline = PersonReIDPipeline()

# También se pueden sobrescribir parámetros específicos
pipeline = PersonReIDPipeline(
    yolo_model_path='yolov8s.pt',
    config_preset='high_speed'
)
```

## Presets de Configuración

### `high_accuracy`
Optimizado para máxima precisión:
- Modelo YOLO grande (yolov8l.pt)
- Resolución alta (1280x1280)
- Modelo ReID completo (osnet_x1_0)
- Parámetros HNSW optimizados para precisión

### `high_speed`
Optimizado para máxima velocidad:
- Modelo YOLO pequeño (yolov8n.pt)
- Resolución reducida (416x416)
- Modelo ReID ligero (osnet_x0_5)
- Parámetros optimizados para velocidad

### `balanced`
Balance entre precisión y velocidad:
- Modelo YOLO medio (yolov8s.pt)
- Resolución estándar (640x640)
- Modelo ReID balanceado (osnet_x0_75)
- Parámetros equilibrados

## Ejemplos de Uso

### Ejemplo 1: Configuración básica

```python
from config.global_config import get_config
from src.main_pipeline import PersonReIDPipeline

# Usar configuración por defecto
pipeline = PersonReIDPipeline()

# Procesar video
results = pipeline.process_video('video.mp4')
```

### Ejemplo 2: Configuración personalizada

```python
from config.global_config import update_config, apply_preset

# Aplicar preset de alta precisión
apply_preset('high_accuracy')

# Personalizar algunos parámetros
update_config('detection', 'confidence_threshold', 0.8)
update_config('fusion', 'similarity_threshold', 0.75)

# Usar configuración personalizada
pipeline = PersonReIDPipeline()
```

### Ejemplo 3: Guardar y cargar configuración

```python
from config.global_config import config

# Guardar configuración actual
config.save_config('my_config.json')

# Cargar configuración guardada
config.load_config('my_config.json')
```

## Línea de Comandos

### Mostrar resumen de configuración

```bash
python run_demo.py --config-summary
```

### Usar preset de configuración

```bash
python run_demo.py --preset high_accuracy --video video.mp4
```

### Sobrescribir parámetros específicos

```bash
python run_demo.py --yolo-model yolov8s.pt --device cuda --video video.mp4
```

## Estructura de Configuración

```
config/
├── global_config.py          # Configuración principal
├── config_documentation.md   # Documentación detallada
├── README.md                 # Este archivo
└── current_config.json       # Configuración guardada (generado)
```

## Variables de Configuración por Módulo

### DETECTION (Detección)
- `model_path`: Modelo YOLO a usar
- `confidence_threshold`: Umbral de confianza
- `iou_threshold`: Umbral IoU para NMS
- `input_size`: Tamaño de entrada
- `target_classes`: Clases objetivo

### TRACKING (Seguimiento)
- `max_disappeared`: Frames máximos sin detección
- `max_distance`: Distancia máxima para asociación
- `coordinate_discretization`: Discretización de coordenadas
- `enable_compression`: Habilitar compresión Huffman

### REID (Re-identificación)
- `model_name`: Modelo OSNet
- `image_size`: Tamaño de imagen
- `batch_size`: Tamaño de batch
- `similarity_threshold`: Umbral de similitud

### INDEXING (Indexación)
- `dimension`: Dimensión de vectores
- `max_elements`: Elementos máximos
- `M`: Parámetro M de HNSW
- `ef_construction`: Parámetro ef_construction
- `ef_search`: Parámetro ef_search

### FUSION (Fusión)
- `similarity_threshold`: Umbral de similitud
- `tracking_weight`: Peso del tracking
- `reid_weight`: Peso del ReID
- `temporal_window`: Ventana temporal

### PIPELINE (Pipeline)
- `max_frames`: Frames máximos a procesar
- `frame_skip`: Salto de frames
- `enable_multithreading`: Habilitar multihilo
- `real_time_mode`: Modo tiempo real

### OUTPUT (Salida)
- `save_video`: Guardar video anotado
- `save_crops`: Guardar recortes
- `video_quality`: Calidad de video
- `crop_format`: Formato de recortes

### PERFORMANCE (Rendimiento)
- `optimization_level`: Nivel de optimización
- `auto_device_selection`: Selección automática de dispositivo
- `gpu_memory_fraction`: Fracción de memoria GPU
- `enable_profiling`: Habilitar profiling

## Mejores Prácticas

1. **Usar presets como punto de partida**: Comienza con un preset (`balanced`, `high_accuracy`, `high_speed`) y ajusta según necesidades específicas.

2. **Validar configuraciones**: Siempre verifica que los valores estén en rangos válidos.

3. **Documentar cambios**: Guarda configuraciones personalizadas para reproducibilidad.

4. **Probar incrementalmente**: Cambia un parámetro a la vez para entender su impacto.

5. **Monitorear rendimiento**: Usa las métricas de rendimiento para evaluar el impacto de los cambios.

## Solución de Problemas

### Error: "Key not found in module"
```python
# Verifica que el módulo y la clave existan
available_modules = get_config().keys()
detection_keys = get_config('detection').keys()
```

### Error: "Unknown module"
```python
# Módulos válidos
valid_modules = ['detection', 'tracking', 'reid', 'indexing', 'fusion', 'pipeline', 'output', 'performance']
```

### Configuración no se aplica
```python
# Asegúrate de que la configuración se aplique antes de inicializar el pipeline
update_config('detection', 'confidence_threshold', 0.7)
pipeline = PersonReIDPipeline()  # Usa la configuración actualizada
```

## Contribuir

Para agregar nuevos parámetros de configuración:

1. Añade el parámetro en `global_config.py` en el módulo apropiado
2. Documenta el parámetro en `config_documentation.md`
3. Actualiza el código del módulo para usar el parámetro
4. Agrega ejemplos de uso si es necesario

## Recursos Adicionales

- [Documentación completa](config_documentation.md)
- [Ejemplos de uso](../examples/config_usage_example.py)
- [Script de demostración](../run_demo.py)
- [Pipeline principal](../src/main_pipeline.py) 