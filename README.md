# Sistema de Re-identificación de Personas para Análisis de Actividad Laboral

## Descripción del Proyecto

Este proyecto implementa un sistema completo de re-identificación de personas (person re-identification) diseñado para mapear la actividad laboral de empleados en entornos de trabajo. El sistema es capaz de:

- **Detectar personas** usando YOLO11m-pose
- **Extraer embeddings robustos** de cada persona detectada
- **Re-identificar personas** que reaparecen en el video (después de salir de cámara, cambiar de ángulo, etc.)
- **Tracking persistente** usando redes complejas (ResNet, Transformer, LSTM)
- **Generar video de salida** con el proceso de tracking visualizado

## Arquitectura del Sistema

```
vision_project_v2/
├── src/                          # Código fuente principal
│   ├── detection/                # Módulo de detección de personas
│   ├── embedding/                # Módulo de extracción de embeddings
│   ├── reid/                     # Módulo de re-identificación
│   ├── tracking/                 # Módulo de tracking persistente
│   ├── visualization/            # Módulo de visualización
│   └── utils/                    # Utilidades comunes
├── models/                       # Modelos pre-entrenados
├── data/                         # Datos de entrada y salida
│   ├── raw/                      # Videos originales
│   └── processed/                # Videos procesados
├── config/                       # Configuraciones
├── requirements.txt              # Dependencias
└── main.py                       # Script principal
```

## Flujo de Procesamiento

### 1. Detección de Personas (`detection/`)
- **Entrada**: Video frame
- **Proceso**: YOLO11m-pose para detectar personas y extraer keypoints
- **Salida**: Bounding boxes y poses de personas detectadas

### 2. Extracción de Embeddings (`embedding/`)
- **Entrada**: Personas detectadas (crops de imágenes)
- **Proceso**: Red neuronal para generar embeddings robustos
- **Salida**: Vectores de características (embeddings) de cada persona

### 3. Re-identificación (`reid/`)
- **Entrada**: Embeddings actuales vs. embeddings históricos
- **Proceso**: Comparación de similitud usando redes complejas
- **Salida**: Matches de personas re-identificadas

### 4. Tracking Persistente (`tracking/`)
- **Entrada**: Personas re-identificadas
- **Proceso**: Mantener IDs únicos a través del tiempo
- **Salida**: IDs consistentes para cada persona

### 5. Visualización (`visualization/`)
- **Entrada**: Resultados de tracking
- **Proceso**: Dibujar bounding boxes, IDs, y trayectorias
- **Salida**: Video procesado con tracking visualizado

## Instalación y Uso

### Requisitos
- Python 3.8+
- CUDA (opcional, para aceleración GPU)
- FFmpeg (para procesamiento de video)

### Instalación
```bash
pip install -r requirements.txt
```

### Uso
```bash
python main.py --input data/raw/video.mp4 --output data/processed/output.mp4
```

## Configuración

El archivo `config/config.yaml` permite ajustar:
- Umbrales de detección
- Parámetros de re-identificación
- Configuración de visualización
- Rutas de modelos

## Características Técnicas

### Modelos Utilizados
- **YOLO11m-pose**: Detección de personas y poses
- **ResNet/Transformer**: Extracción de embeddings
- **LSTM/RNN**: Tracking temporal
- **CNN**: Procesamiento de características

### Robustez del Sistema
- **Oclusión**: Manejo de personas parcialmente ocultas
- **Cambio de ángulo**: Re-identificación desde diferentes perspectivas
- **Salida/reentrada**: Tracking persistente cuando personas salen y vuelven
- **Iluminación**: Embeddings robustos a cambios de luz

## Resultados Esperados

El sistema generará:
1. **Video procesado** con tracking visualizado
2. **Logs de actividad** con timestamps de cada persona
3. **Métricas de rendimiento** (precisión, recall, F1-score)
4. **Análisis de trayectorias** de movimiento

## Estructura de Datos

### Persona Detectada
```python
{
    'id': int,                    # ID único de la persona
    'bbox': [x1, y1, x2, y2],     # Bounding box
    'pose': np.array,             # Keypoints de pose
    'embedding': np.array,        # Vector de características
    'confidence': float,          # Confianza de detección
    'timestamp': float            # Timestamp del frame
}
```

### Historial de Tracking
```python
{
    'person_id': int,
    'trajectory': [(x, y, t), ...],  # Trayectoria temporal
    'appearances': [frame_ids],       # Frames donde aparece
    'embeddings_history': [embeddings] # Historial de embeddings
}
```

## Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Implementa los cambios
4. Ejecuta las pruebas
5. Envía un pull request

## Licencia

Este proyecto está bajo la licencia MIT. 